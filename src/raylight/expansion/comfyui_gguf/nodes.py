# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import torch
import logging
import collections
import contextlib

import ray

import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths

from .ops import move_patch_to_device, GGMLTensor
from .dequant import is_quantized, is_torch_compatible

from raylight.distributed_worker.ray_worker import ensure_fresh_actors, evict_page_cache


def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")


# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])


class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False
    release_mmap = False
    unet_path = None
    gguf_metadata = {}

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        
        # SOURCE OF TRUTH: Always prefer the backup (mmap/CPU) weight if available.
        # This allows us to "nuke" the parameter on GPU (nuclear offload) and still recover.
        if key in self.backup:
            weight = self.backup[key].weight
        else:
            weight = comfy.utils.get_attr(self.model, key)

        if is_quantized(weight):
            # Backup the original GGMLTensor (zero-copy mmap) before we replace it with a GPU copy.
            # This ensures we can restore the mmap state during unpatch/offload.
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight, False
                )

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        
        # Restore backups (putting GGMLTensors back in place)
        # CRITICAL: We pass device_to=None to super() to prevent it from moving the restored weights.
        
        # NOTE: We no longer preserve backup across unpatching.
        # Recovery is now done via /dev/shm cache, not backup dict (which causes memory leaks).
        super().unpatch_model(device_to=None, unpatch_weights=unpatch_weights)

        # Explicitly clear any lingering modifications to ensure VRAM release
        if unpatch_weights:
             for key in self.backup:
                 # The backup has been restored to param.data. 
                 # We want to ensure the *previous* param.data (the GPU tensor) is definitely dropped.
                 pass # Python assignment handles this decrement. GC handles the rest.
        
        # Move patches themselves back to offload device if they were on GPU
        # This fixes "model_to_remove" equivalent for patch weights (LoRAs)
        # ComfyUI patch format: (strength, v, strength_model, offset, function)
        # where v can be a tensor, tuple, list, or WeightAdapterBase
        for key, patch_list in self.patches.items():
            for i, patch in enumerate(patch_list):
                if len(patch) >= 2:
                    v = patch[1]  # The actual patch value is at index 1
                    # Handle different patch value types
                    if isinstance(v, torch.Tensor) and v.device != self.offload_device:
                        # Move LoRA weights back to CPU/offload to free VRAM
                        new_patch = list(patch)
                        new_patch[1] = v.to(self.offload_device)
                        patch_list[i] = tuple(new_patch)
                    elif isinstance(v, (tuple, list)) and len(v) > 0:
                        # Handle tuple/list values like ("diff", (tensor,)) format
                        # Check if the first element is a patch_type string
                        if isinstance(v[0], str) and len(v) > 1:
                            inner_v = v[1]
                            if isinstance(inner_v, tuple) and len(inner_v) > 0:
                                if isinstance(inner_v[0], torch.Tensor) and inner_v[0].device != self.offload_device:
                                    new_inner = list(inner_v)
                                    new_inner[0] = inner_v[0].to(self.offload_device)
                                    new_v = (v[0], tuple(new_inner)) + tuple(v[2:]) if len(v) > 2 else (v[0], tuple(new_inner))
                                    new_patch = list(patch)
                                    new_patch[1] = new_v
                                    patch_list[i] = tuple(new_patch)
                        elif torch.is_tensor(v[0]) and v[0].device != self.offload_device:
                            # Direct tensor tuple like (tensor,)
                            new_v = tuple(t.to(self.offload_device) if torch.is_tensor(t) and t.device != self.offload_device else t for t in v)
                            new_patch = list(patch)
                            new_patch[1] = new_v
                            patch_list[i] = tuple(new_patch)

        if device_to is not None:
             self.current_device = device_to
             # Manually move non-quantized weights to offload device.
             # We skip quantized (GGML) weights to avoid breaking mmap.
             from raylight.expansion.comfyui_gguf.ops import GGMLTensor
             for module in self.model.modules():
                 for name, buf in module.named_buffers(recurse=False):
                     if buf is not None and not is_quantized(buf) and not isinstance(buf, GGMLTensor):
                         # Skip if already on target device (avoids mmap move issues)
                         if buf.device != device_to:
                             try:
                                 module._buffers[name] = buf.to(device_to)
                             except Exception:
                                 pass  # Skip if device move fails (e.g., mmap tensor)
                 
                 for name, param in module.named_parameters(recurse=False):
                     if param is not None and not is_quantized(param) and not isinstance(param.data, GGMLTensor):
                         # Skip if already on target device (avoids mmap move issues)
                         if param.device != device_to:
                             try:
                                 param.data = param.data.to(device_to)
                                 if param._grad is not None:
                                     param._grad.data = param._grad.data.to(device_to)
                             except Exception:
                                 pass  # Skip if device move fails
            
             # Ensure Page Cache is evicted if we are still tracking the path
             if hasattr(self, "unet_path"):
                 evict_page_cache(self.unet_path)

    def pin_weight_to_device(self, key):
        op_key = key.rsplit('.', 1)[0]
        if not self.mmap_released and op_key in self.named_modules_to_munmap:
            # TODO: possible to OOM, find better way to detach
            self.named_modules_to_munmap[op_key].to(self.load_device).to(self.offload_device)
            del self.named_modules_to_munmap[op_key]
        super().pin_weight_to_device(key)

    mmap_released = False
    named_modules_to_munmap = {}

    def load(self, *args, force_patch_weights=False, **kwargs):
        if self.release_mmap and not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())

        # SELF-HEAL: Load from /dev/shm cache if available (zero-copy via mmap)
        # Only run if we have 0-byte nuked weights that need restoration
        cache_path = getattr(self, 'cache_path', None)
        needs_heal = False
        if cache_path:
            # Quick check: do we have any 0-byte weights?
            for name, param in self.model.named_parameters():
                if param.numel() == 0:
                    needs_heal = True
                    break
        
        if needs_heal and os.path.exists(cache_path):
            import torch
            import gc
            import pickle
            
            # ZERO-COPY STREAMING RESTORE
            # Key insight: mmap from /dev/shm means pages are already in RAM (shared memory).
            # Supports both old format (torch.save state_dict) and new streaming format.
            
            # Detect format: new streaming format starts with a header dict with "streaming" key
            state_dict = None
            is_streaming_format = False
            
            try:
                with open(cache_path, 'rb') as f:
                    first_obj = pickle.load(f)
                    if isinstance(first_obj, dict) and first_obj.get("streaming", False):
                        is_streaming_format = True
                    else:
                        # Old format - reload with torch.load for mmap support
                        state_dict = torch.load(cache_path, mmap=True, weights_only=False)
            except Exception as e:
                # Fallback to old format
                state_dict = torch.load(cache_path, mmap=True, weights_only=False)
            
            restored = 0
            skipped = 0
            moved_to_gpu = 0
            target_device = self.load_device
            
            # Use dedicated CUDA stream for async DMA operations
            stream = torch.cuda.Stream(device=target_device) if target_device.type == 'cuda' else None
            ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
            
            def restore_tensor(name, tensor):
                """Helper to restore a single tensor."""
                nonlocal restored, skipped, moved_to_gpu
                try:
                    if name.startswith("buf_"):
                        actual_name = name[4:]
                        curr = comfy.utils.get_attr(self.model, actual_name)
                    else:
                        actual_name = name
                        curr = comfy.utils.get_attr(self.model, name)
                    
                    if curr.numel() == 0:  # Only restore nuked weights
                        restored_tensor = tensor.to(target_device, non_blocking=True)
                        if hasattr(curr, 'data'):
                            curr.data = restored_tensor
                        else:
                            comfy.utils.set_attr_param(self.model, actual_name, restored_tensor)
                        restored += 1
                    else:
                        skipped += 1
                        if curr.device.type != 'cuda' and target_device.type == 'cuda':
                            if hasattr(curr, 'data'):
                                curr.data = curr.data.to(target_device, non_blocking=True)
                            moved_to_gpu += 1
                except Exception as e:
                    print(f"[GGUFModelPatcher] DEBUG: Failed to restore '{name}': {e}")
            
            with ctx:
                if is_streaming_format:
                    # NEW STREAMING FORMAT: Read tensors one at a time
                    with open(cache_path, 'rb') as f:
                        pickle.load(f)  # Skip header
                        tensor_count = 0
                        while True:
                            try:
                                obj = pickle.load(f)
                                if obj is None:  # End marker
                                    break
                                
                                # Handle both old (2-tuple) and new (3-tuple) formats for backward compatibility
                                if len(obj) == 3:
                                    name, tensor, metadata = obj
                                else:
                                    name, tensor = obj
                                    metadata = None

                                # Reconstruct GGMLTensor if metadata exists
                                if metadata:
                                    # Create new GGMLTensor with restored metadata
                                    # patches=[] ensures we start with clean base weights (fixing double-patch issue)
                                    tensor = GGMLTensor(
                                        tensor, 
                                        tensor_type=metadata["tensor_type"],
                                        tensor_shape=metadata["tensor_shape"],
                                        patches=metadata.get("patches", [])
                                    )

                                # DEBUG: Inspect restored tensor
                                if tensor_count < 3:
                                    t_type = getattr(tensor, 'tensor_type', 'NO_ATTR')
                                    print(f"[GGUF Reload DEBUG] Restoring '{name}': class={tensor.__class__.__name__}, dtype={tensor.dtype}, shape={tensor.shape}, tensor_type={t_type}")
                                
                                restore_tensor(name, tensor)
                                tensor_count += 1
                                del tensor  # Immediately release
                            except EOFError:
                                break
                    cache_key_count = tensor_count
                else:
                    # OLD FORMAT: state_dict already loaded
                    for name, tensor in state_dict.items():
                        restore_tensor(name, tensor)
                    cache_key_count = len(state_dict)
            
            # Synchronize stream to ensure all async transfers complete
            if stream:
                stream.synchronize()
            
            print(f"[GGUFModelPatcher] ZERO-COPY FAST LOAD: Restored {restored} weights, skipped {skipped} (moved {moved_to_gpu} to GPU), cache has {cache_key_count} keys")
            
            # CRITICAL: Cleanup mmap references (only for old format that loaded state_dict)
            if state_dict is not None:
                for key in list(state_dict.keys()):
                    state_dict[key] = None
                state_dict.clear()
                del state_dict
            
            # Force multiple GC passes to release mmap pages
            for _ in range(5):
                gc.collect()
            
            # CRITICAL: Sync CUDA after restore to catch async errors early
            torch.cuda.synchronize()
            print(f"[GGUFModelPatcher] CUDA sync after restore - OK")
            
            # Force libc to release freed memory back to OS
            try:
                import ctypes
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except Exception:
                pass
            
            # Mark mmap as released to skip native mmap release (would create duplicate CPU copies)
            self.mmap_released = True
            self.named_modules_to_munmap = {}

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # Evict original GGUF mmap after first successful load to free page cache
        if not getattr(self, '_gguf_mmap_evicted', False):
            if hasattr(self, "unet_path"):
                evict_page_cache(self.unet_path)
            self._gguf_mmap_evicted = True
            print("[GGUFModelPatcher] Evicted original GGUF mmap references")
        
        # CRITICAL: Clear backup dict EVERY load, not just first load.
        # super().load() calls patch_weight_to_device which repopulates backup.
        # Without clearing, mmap tensor refs accumulate across hot reloads.
        import gc
        self.backup.clear()
        
        # NUCLEAR CLEANUP: Release ALL large CPU tensor storage after weights are on GPU.
        # This runs on EVERY load (cold or hot) because:
        # - Cold load: mmap tensors need cleanup after VRAM load
        # - Hot reload: /dev/shm cache is the only RAM copy we need
        # We iterate through the heap to find and nuke ALL large CPU tensors.
        nuked_cpu_tensors = 0
        nuked_bytes = 0
        MIN_SIZE_BYTES = 10 * 1024 * 1024  # 10MB threshold
        
        for obj in gc.get_objects():
            try:
                # Check if it's any tensor subclass (including GGMLTensor) on CPU
                if torch.is_tensor(obj) and obj.device.type == 'cpu':
                    size_bytes = obj.numel() * obj.element_size()
                    if size_bytes > MIN_SIZE_BYTES:
                        # Replace the storage with empty to release RAM
                        obj.set_(torch.empty(0, dtype=obj.dtype, device='cpu').storage())
                        nuked_cpu_tensors += 1
                        nuked_bytes += size_bytes
            except Exception:
                continue
        if nuked_cpu_tensors > 0:
            print(f"[GGUFModelPatcher] Nuked {nuked_cpu_tensors} large CPU tensors ({nuked_bytes / 1024**3:.2f} GB) - weights are on GPU")
        
        gc.collect()
        
        # Force libc to release freed memory back to OS
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

        # make sure nothing stays linked to mmap after first load
        if self.release_mmap and not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.named_modules_to_munmap.items():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logging.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
            self.named_modules_to_munmap = {}

    def clone(self, *args, **kwargs):
        if isinstance(self, GGUFModelPatcher):
            n = super().clone(*args, **kwargs)
            n.patch_on_device = getattr(self, "patch_on_device", False)
            n.mmap_released = getattr(self, "mmap_released", False)
            n.release_mmap = getattr(self, "release_mmap", False)
            n.unet_path = getattr(self, "unet_path", None)
            
            # CRITICAL FIX: Do NOT copy backup to clones.
            # Copying backup creates additional references to mmap'd tensors, preventing release.
            # Clones can reload from /dev/shm cache if they need weights.
            n.backup = {}
            n.cache_path = getattr(self, 'cache_path', None)  # Share cache path for reload
            print(f"[GGUFModelPatcher] Cloned with empty backup (cache_path={n.cache_path is not None}).")
            
            return n
        
        # Upgrade case: converting standard ModelPatcher to GGUFModelPatcher
        # We manually construct the new class using the base properties
        # Note: ModelPatcher init args: (model, load_device, offload_device, size=0, current_device=None, weight_inplace_update=False)
        n = GGUFModelPatcher(
            self.model, 
            self.load_device, 
            self.offload_device, 
            self.size, 
            weight_inplace_update=self.weight_inplace_update
        )
        
        # Copy internal state
        import copy
        n.patches = {}
        for k, v in self.patches.items():
            n.patches[k] = v[:] # Shallow copy of list
            
        # CRITICAL FIX: Do NOT copy backup - prevents mmap reference accumulation
        n.backup = {}
        n.cache_path = getattr(self, 'cache_path', None)  # Share cache path for reload
        print(f"[GGUFModelPatcher] Upgraded clone with empty backup (cache_path={n.cache_path is not None}).")
        n.object_patches = self.object_patches.copy()
        
        # GGUF defaults
        n.patch_on_device = False
        n.mmap_released = False
        n.release_mmap = False # Default to keeping mmap
        n.unet_path = getattr(self, "unet_path", None)
        return n


class RayGGUFLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet_gguf"),),
                "dequant_dtype": (
                    ["default", "target", "float32", "float16", "bfloat16"],
                    {"default": "default"},
                ),
                "patch_dtype": (
                    ["default", "target", "float32", "float16", "bfloat16"],
                    {"default": "default"},
                ),
                "ray_actors_init": (
                    "RAY_ACTORS_INIT",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
            },
            "optional": {"lora": ("RAY_LORA", {"default": None})},
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "load_ray_unet"

    CATEGORY = "Raylight"

    def load_ray_unet(
        self,
        ray_actors_init,
        unet_name,
        dequant_dtype,
        patch_dtype,
        lora=None,
    ):
        ray_actors, gpu_actors, parallel_dict = ensure_fresh_actors(ray_actors_init)

        unet_path = folder_paths.get_full_path_or_raise("unet", unet_name)

        loaded_futures = []
        patched_futures = []

        for actor in gpu_actors:
            loaded_futures.append(actor.set_lora_list.remote(lora))
        ray.get(loaded_futures)
        loaded_futures = []

        if parallel_dict["is_fsdp"] is True:
            worker0 = ray.get_actor("RayWorker:0")
            ray.get(
                worker0.load_gguf_unet.remote(
                    unet_path,
                    dequant_dtype=dequant_dtype,
                    patch_dtype=patch_dtype,
                )
            )
            meta_model = ray.get(worker0.get_meta_model.remote())

            for actor in gpu_actors:
                if actor != worker0:
                    loaded_futures.append(actor.set_meta_model.remote(meta_model))

            ray.get(loaded_futures)
            loaded_futures = []

            for actor in gpu_actors:
                loaded_futures.append(actor.set_state_dict.remote())

            ray.get(loaded_futures)
            loaded_futures = []
        else:
            # 1. Leader (Worker 0) Load
            worker0 = ray.get_actor("RayWorker:0")
            print("[Raylight] Starting Leader GGUF Load on Worker 0...")
            ray.get(
                worker0.load_gguf_unet.remote(
                    unet_path,
                    dequant_dtype=dequant_dtype,
                    patch_dtype=patch_dtype,
                )
            )

            # 2. Get Ref & Metadata
            base_ref = ray.get(worker0.get_base_ref.remote())
            gguf_metadata = ray.get(worker0.get_gguf_metadata.remote())

            # Prepare params for fallback
            reload_params = {
                "unet_path": unet_path,
                "dequant_dtype": dequant_dtype,
                "patch_dtype": patch_dtype
            }

            # 3. Follower Hydration (Zero-Copy with Fallback)
            print("[Raylight] Initializing Followers via Shared RAM (with mmap fallback)...")
            for actor in gpu_actors:
                if actor != worker0:
                    loaded_futures.append(
                        actor.init_gguf_from_ref.remote(
                            base_ref,
                            gguf_metadata,
                            reload_params
                        )
                    )
            ray.get(loaded_futures)
            loaded_futures = []

            # 4. Synchronized Eviction
            # Now that ALL workers have loaded (mapped) the file, we can safely evict the page cache.
            # We use a barrier inside the worker to ensure perfect sync.
            print("[Raylight] All workers loaded. Triggering synchronized Page Cache Eviction...")
            evict_futures = [actor.barrier_and_evict.remote(unet_path) for actor in gpu_actors]
            ray.get(evict_futures)


        for actor in gpu_actors:
            if parallel_dict["is_xdit"]:
                patched_futures.append(actor.patch_usp.remote())

        ray.get(patched_futures)

        return (ray_actors,)


NODE_CLASS_MAPPINGS = {
    "RayGGUFLoader": RayGGUFLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayGGUFLoader": "Load Diffusion GGUF Model (Ray)",
}
