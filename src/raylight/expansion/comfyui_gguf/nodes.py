# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import torch
import logging
import collections

import ray

import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths

from .ops import move_patch_to_device
from .dequant import is_quantized, is_torch_compatible

from raylight.distributed_worker.ray_worker import ensure_fresh_actors


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
        
        # FIX: Preserve backup across unpatching.
        # ComfyUI ModelPatcher clears self.backup after unpatching.
        # But for GGUF Soft-Offload, we need these backups (mmap tensors) to survive
        # so we can recover from the "nuclear" 0-byte failsafe offload.
        preserved_backup = self.backup.copy()
        
        super().unpatch_model(device_to=None, unpatch_weights=unpatch_weights)

        # Restore the backup so we can reload/patch again later
        self.backup = preserved_backup

        # Explicitly clear any lingering modifications to ensure VRAM release
        if unpatch_weights:
             for key in self.backup:
                 # The backup has been restored to param.data. 
                 # We want to ensure the *previous* param.data (the GPU tensor) is definitely dropped.
                 pass # Python assignment handles this decrement. GC handles the rest.
        
        # Move patches themselves back to offload device if they were on GPU
        # This fixes "model_to_remove" equivalent for patch weights (LoRAs)
        for key, patch_list in self.patches.items():
            for i, (patch_weight, patch_type) in enumerate(patch_list):
                 if isinstance(patch_weight, torch.Tensor) and patch_weight.device != self.offload_device:
                      # Move LoRA weights back to CPU/offload to free VRAM
                      patch_list[i] = (patch_weight.to(self.offload_device), patch_type)

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
            state_dict = torch.load(cache_path, mmap=True, weights_only=False)
            restored = 0
            skipped = 0
            moved_to_gpu = 0
            target_device = self.load_device
            for name, tensor in state_dict.items():
                try:
                    # Handle buffer vs parameter naming (buf_ prefix from cache)
                    if name.startswith("buf_"):
                        actual_name = name[4:]  # Remove "buf_" prefix
                        curr = comfy.utils.get_attr(self.model, actual_name)
                    else:
                        curr = comfy.utils.get_attr(self.model, name)
                    
                    if curr.numel() == 0:  # Only restore nuked weights
                        # Move tensor to model's load device (GPU) before assigning
                        # Cache stores CPU copies, but ComfyUI expects GPU tensors
                        restored_tensor = tensor.to(target_device)
                        
                        # Directly assign data
                        if hasattr(curr, 'data'):
                            curr.data = restored_tensor
                        else:
                            # Fallback for non-parameter attributes
                            if name.startswith("buf_"):
                                comfy.utils.set_attr(self.model, name[4:], restored_tensor)
                            else:
                                comfy.utils.set_attr_param(self.model, name, restored_tensor)
                        restored += 1
                    else:
                        skipped += 1
                        # These weights weren't nuked (on CPU), but need to be on GPU for inference
                        if curr.device.type != 'cuda' and target_device.type == 'cuda':
                            if hasattr(curr, 'data'):
                                curr.data = curr.data.to(target_device)
                            moved_to_gpu += 1
                        # Debug: log what's being skipped
                        if skipped <= 5:
                            print(f"[GGUFModelPatcher] DEBUG: Skipped '{name}' (numel={curr.numel()}, device={curr.device})")
                except Exception as e:
                    print(f"[GGUFModelPatcher] DEBUG: Failed to restore '{name}': {e}")
            print(f"[GGUFModelPatcher] HOT LOAD: Restored {restored} weights, skipped {skipped} (moved {moved_to_gpu} to GPU), cache has {len(state_dict)} keys")
            del state_dict
            
            # CRITICAL: Sync CUDA after restore to catch async errors early
            torch.cuda.synchronize()
            print(f"[GGUFModelPatcher] CUDA sync after restore - OK")

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # Evict original GGUF mmap after first successful load to free page cache
        if not getattr(self, '_gguf_mmap_evicted', False):
            import gc
            # Clear backup dict - we'll use /dev/shm cache instead
            self.backup.clear()
            gc.collect()
            self._gguf_mmap_evicted = True
            print("[GGUFModelPatcher] Evicted original GGUF mmap references")

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
            
            # CRITICAL: Share backup with clones.
            # Clones need access to the mmap tensor backups to perform repatching/restoration
            # especially if the main model has been "nuked" to 0-byte tensors.
            # Use copy() to avoid reference issues if the original dict is cleared/replaced.
            n.backup = self.backup.copy()
            print(f"[GGUFModelPatcher] Cloned backup (len={len(n.backup) if n.backup else 0}).")
            
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
            
        n.backup = self.backup.copy()
        print(f"[GGUFModelPatcher] Upgraded clone backup (len={len(n.backup) if n.backup else 0}).")
        n.object_patches = self.object_patches.copy()
        
        # GGUF defaults
        n.patch_on_device = False
        n.mmap_released = False
        n.release_mmap = False # Default to keeping mmap
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
