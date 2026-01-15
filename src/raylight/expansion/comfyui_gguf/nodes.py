# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
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
             for module in self.model.modules():
                 for name, buf in module.named_buffers(recurse=False):
                     if buf is not None and not is_quantized(buf):
                         module._buffers[name] = buf.to(device_to)
                 
                 for name, param in module.named_parameters(recurse=False):
                     if param is not None and not is_quantized(param):
                         param.data = param.data.to(device_to)
                         if param._grad is not None:
                             param._grad.data = param._grad.data.to(device_to)

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

        # SELF-HEAL: Recover from 0-byte nuclear offload.
        # If we previously nuked weights to save VRAM, restore them from backup now.
        if self.backup:
            restored_count = 0
            restored_bytes = 0
            objectref_count = 0
            
            for name, saved_dim in self.backup.items():
                try:
                    curr_weight = comfy.utils.get_attr(self.model, name)
                    # Check if nuked (0-elements)
                    if curr_weight.numel() == 0:
                        # Restore the mmap tensor or Shared ObjectRef from backup
                        backup_data = saved_dim.weight
                        
                        # Resolve Ray ObjectRef if present (for modified/shape-changed weights)
                        if isinstance(backup_data, ray.ObjectRef):
                            backup_data = ray.get(backup_data)
                            objectref_count += 1

                        # usage of set_attr_param ensures we replace the nuked (0-byte) tensor
                        comfy.utils.set_attr_param(self.model, name, backup_data)
                        restored_count += 1
                        # Estimate size
                        if hasattr(backup_data, "numel"):
                            restored_bytes += backup_data.numel() * backup_data.element_size()
                        
                        # Release ref immediately to help GC
                        del backup_data
                except Exception as e:
                    pass
            
            if restored_count > 0:
                print(f"[GGUFModelPatcher] HOT LOAD (Self-Heal): Restored {restored_count} weights ({restored_bytes / 1024**2:.2f} MB) from backup. ObjectRefs resolved: {objectref_count}")
            
            # Force cleanup to release heap before super().load() moves to GPU
            import gc
            gc.collect()

        # Phase 1: Pre-Load Backup (Quantized/Mmap Weights)
        # We backup GGUF tensors HERE to preserve the mmap reference (approx 0 RAM).
        # If we wait until post-load, they become dense GPU tensors (High RAM to backup).
        mmap_backup_count = 0
        mmap_backup_bytes = 0
        for name, param in self.model.named_parameters():
             if is_quantized(param):
                 self.backup[name] = collections.namedtuple('Dimension', ['weight', 'inplace_update', 'shape'])(
                    param.data, False, param.shape  # Store GGMLTensor dict/ref + shape
                 )
                 mmap_backup_count += 1
                 # Estimate size (though it's mmap)
                 mmap_backup_bytes += param.numel() * param.element_size()
        
        if mmap_backup_count > 0:
             print(f"[GGUFModelPatcher] Pre-Load: Preserved {mmap_backup_count} Quantized Mmap weights ({mmap_backup_bytes / 1024**2:.2f} MB).")

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # Phase 2: Post-Load Backup (Standard/Modified Weights)
        # Skip if already completed for this model instance (avoids redundant work on sequential samplers)
        if getattr(self, '_phase2_complete', False):
            return
        
        # We backup standard weights HERE to capture their FINAL shape/value.
        # We ALSO check if any Pre-Load (Mmap) weights were modified (shape change). If so, we upgrade them to RAM.
        # CRITICAL: We use a Shared Ray Registry to deduplicate these upgrades!
        # This allows upgrading Massive weights (15GB) without OOMing (15GB vs 60GB).
        from raylight.distributed_worker.backup_registry import BackupRegistry
        try:
            registry = BackupRegistry.options(name="GGUFBackupRegistry", get_if_exists=True, lifetime="detached").remote()
        except:
            # Fallback if creation fails (rare), though options should handle it.
            # Or assume it was created by RayGGUFLoader/Worker0.
            registry = BackupRegistry.options(name="GGUFBackupRegistry", create_if_missing=True, lifetime="detached").remote()

        ram_backup_count = 0
        ram_backup_bytes = 0
        mismatch_log_count = 0
        
        for name, param in self.model.named_parameters():
             should_backup = False
             
             if name not in self.backup:
                 should_backup = True
             else:
                 # Check for Shape Mismatch - Use stored shape metadata (no ray.get needed!)
                 backup_entry = self.backup[name]
                 if hasattr(backup_entry, 'shape') and backup_entry.shape is not None:
                     saved_shape = backup_entry.shape
                 else:
                     # Fallback for old-style backup without shape (shouldn't happen)
                     backup_weight = backup_entry.weight
                     if isinstance(backup_weight, ray.ObjectRef):
                         temp_tensor = ray.get(backup_weight)
                         saved_shape = temp_tensor.shape
                         del temp_tensor
                     else:
                         saved_shape = backup_weight.shape
                 
                 curr_shape = param.shape
                 if saved_shape != curr_shape:
                     if mismatch_log_count < 5:
                         print(f"[GGUFModelPatcher] Backup/Runtime shape mismatch for {name}: {saved_shape} vs {curr_shape}. Upgrading to Shared RAM.")
                         mismatch_log_count += 1
                     should_backup = True
             
             if should_backup:
                 # 1. Capture Tensor (CPU)
                 tensor = param.detach().cpu()
                 tensor_shape = tensor.shape  # Cache shape before potential modifications
                 
                 # 2. Store in Plasma (Shared)
                 ref = ray.put(tensor)
                 del tensor  # Immediately release local copy
                 
                 # 3. Deduplicate via Registry (First Writer Wins)
                 key = f"GGUF_{name}" # Assumes single model usage per cluster for now
                 # Wrap ref in list to prevent Ray from resolving it (deser) on the actor side
                 final_ref_container = ray.get(registry.put_if_missing.remote(key, [ref]))
                 final_ref = final_ref_container[0]
                 
                 self.backup[name] = collections.namedtuple('Dimension', ['weight', 'inplace_update', 'shape'])(
                    final_ref, False, tensor_shape  # Store shape for fast lookup!
                 )
                 
                 ram_backup_count += 1
                 ram_backup_bytes += tensor_shape.numel() * param.element_size()
        
        for name, buf in self.model.named_buffers():
             if name not in self.backup:
                 # Buffers are usually small, but let's share them too for consistency
                 tensor = buf.detach().cpu()
                 tensor_shape = tensor.shape
                 ref = ray.put(tensor)
                 del tensor
                 key = f"GGUF_BUFFER_{name}"
                 final_ref_container = ray.get(registry.put_if_missing.remote(key, [ref]))
                 final_ref = final_ref_container[0]
                 
                 self.backup[name] = collections.namedtuple('Dimension', ['weight', 'inplace_update', 'shape'])(
                    final_ref, False, tensor_shape
                 )
                 ram_backup_count += 1
                 ram_backup_bytes += tensor_shape.numel() * buf.element_size()

        if ram_backup_count > 0:
            print(f"[GGUFModelPatcher] Post-Load: Backed up {ram_backup_count} Standard/Modified weights to Shared RAM ({ram_backup_bytes / 1024**2:.2f} MB).")

        # Verification: Ensure Backup covers complete model
        total_params = sum(1 for _ in self.model.named_parameters()) + sum(1 for _ in self.model.named_buffers())
        if len(self.backup) != total_params:
            print(f"[GGUFModelPatcher] WARNING: Backup count ({len(self.backup)}) != Model params/buffers ({total_params}). Some weights might be unbacked!")
        else:
            # Verified
            pass
        
        # Mark Phase 2 as complete to skip on subsequent samplers
        self._phase2_complete = True

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
