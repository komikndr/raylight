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
    mmap_cache = {}

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        
        # GHOST REFRESH: Pull from mmap_cache if available (handles 'meta' restoration)
        # This allows the model to re-hydrate from a fresh mapping.
        if hasattr(self, "mmap_cache") and key in self.mmap_cache:
            weight = self.mmap_cache[key]
        else:
            weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            
            # CRITICAL: Backup the original (likely mmap) weight before modifying or moving
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
            print(f"[GGUFModelPatcher] Unpatching model... (Target: {device_to}, Weights: {unpatch_weights})")
            
            # 1. Standard unpatch for non-weight state.
            # We pass device_to=None to prevent super() from moving/converting our 'meta' weights.
            super().unpatch_model(device_to=None, unpatch_weights=unpatch_weights)

            # 2. GHOST OFFLOAD: Convert all parameters to 'meta' and clear mmap_cache.
            # This destroys the process mappings and drops RSS to near-zero.
            print(f"[GGUFModelPatcher] Ghost Offload: Moving weights to 'meta' and clearing mappings...")
            for p in self.model.parameters():
                if p.device.type != 'meta':
                    p.data = torch.empty_like(p.data, device='meta')
            
            self.mmap_cache = {}
            from raylight.distributed_worker.utils import cleanup_memory
            cleanup_memory()
            print(f"[GGUFModelPatcher] Ghost Offload Complete. RSS should be released.")

        # Move patches themselves back to offload device if they were on GPU
        for key, patch_list in self.patches.items():
            for i, patch in enumerate(patch_list):
                if len(patch) >= 2:
                    v = patch[1] 
                    if isinstance(v, torch.Tensor) and v.device != self.offload_device:
                        if v.numel() > 0:
                            new_patch = list(patch)
                            new_patch[1] = v.to(self.offload_device)
                            patch_list[i] = tuple(new_patch)
                    elif isinstance(v, (tuple, list)) and len(v) > 0:
                        if isinstance(v[0], str) and len(v) > 1:
                            inner_v = v[1]
                            if isinstance(inner_v, tuple) and len(inner_v) > 0:
                                if isinstance(inner_v[0], torch.Tensor) and inner_v[0].device != self.offload_device:
                                    if inner_v[0].numel() > 0:
                                        new_inner = list(inner_v)
                                        new_inner[0] = inner_v[0].to(self.offload_device)
                                        new_v = (v[0], tuple(new_inner)) + tuple(v[2:]) if len(v) > 2 else (v[0], tuple(new_inner))
                                        new_patch = list(patch)
                                        new_patch[1] = new_v
                                        patch_list[i] = tuple(new_patch)
                        elif torch.is_tensor(v[0]) and v[0].device != self.offload_device:
                            if v[0].numel() > 0:
                                new_v = tuple(t.to(self.offload_device) if torch.is_tensor(t) and t.device != self.offload_device else t for t in v)
                                new_patch = list(patch)
                                new_patch[1] = new_v
                                patch_list[i] = tuple(new_patch)

        # Manually move non-GGUF weights to target device
        # This ensures real GPU weights (like biases or non-quantized layers) are offloaded,
        # while GGUF/Mmap weights stay untouched.
        if device_to is not None and unpatch_weights:
             self.current_device = device_to
             from raylight.expansion.comfyui_gguf.ops import GGMLTensor
             
             print(f"[GGUFModelPatcher] Offloading non-GGUF weights to {device_to}...")
             moved_count = 0
             
             # Move all parameters directly
             for param in self.model.parameters():
                 # SIMPLE RULE: If it's on CUDA, it must go to CPU to free VRAM.
                 # GGUF mmap tensors are always on CPU, so they are naturally skipped.
                 # We don't need complex isinstance checks which might fail.
                 if param.device.type == 'cuda' and device_to.type == 'cpu':
                      param.data = param.data.to(device_to)
                      if param._grad is not None:
                           param._grad.data = param._grad.data.to(device_to)
                      moved_count += 1
                 elif param.device != device_to and device_to.type != 'cpu':
                      # If moving TO cuda (reloading), we might want strict checks vs GGUF
                      # But here we are mostly concerned with Offload.
                      pass 

             # Move all buffers directly
             for buf in self.model.buffers():
                 if buf.device.type == 'cuda' and device_to.type == 'cpu':
                     buf.data = buf.data.to(device_to)
                     moved_count += 1
            
             print(f"[GGUFModelPatcher] Offloaded {moved_count} tensors to {device_to}.")

    def load(self, *args, force_patch_weights=False, **kwargs):
        # GHOST RE-HYDRATION: Re-map the GGUF file if cache was cleared during offload.
        if not hasattr(self, "mmap_cache") or (not self.mmap_cache and hasattr(self, "unet_path")):
            print(f"[GGUFModelPatcher] Ghost Re-hydration: Mapping {self.unet_path}...")
            from .loader import gguf_sd_loader
            sd, _ = gguf_sd_loader(self.unet_path)
            self.mmap_cache = sd
            print(f"[GGUFModelPatcher] Re-hydration complete.")

        super().load(*args, force_patch_weights=True, **kwargs)
        
        # Optimization: Clear backup after load to drop extra references?
        # Standard ComfyUI keeps backup to restore later. We need it for unpatching.
        # But if we accumulate backups across reloads (in clones), it leaks.
        # We'll handle leaks in clone().

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_cache = getattr(self, "mmap_cache", {})
        
        # CRITICAL FIX: Empty backup in clone to prevent mmap reference leaks
        # Clones will repopulate backup when they act.
        n.backup = {}
        
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
    ):
        ray_actors, gpu_actors, parallel_dict = ensure_fresh_actors(ray_actors_init)

        unet_path = folder_paths.get_full_path_or_raise("unet", unet_name)

        loaded_futures = []
        patched_futures = []

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
                "patch_dtype": patch_dtype,
                "model_options": {}
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
