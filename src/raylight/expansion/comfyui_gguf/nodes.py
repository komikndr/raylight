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
from raylight.comfy_dist.lora import calculate_weight as ray_calculate_weight


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
    # Instance attributes:
    # mmap_cache: Dict[str, GGMLTensor]
    # unet_path: str

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

            out_weight = ray_calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            print(f"[GGUFModelPatcher] Zero-Copy Offload: Restoring mmap references (Target: {device_to})")
            
            # 1. Standard unpatch for non-weight state.
            super().unpatch_model(device_to=None, unpatch_weights=unpatch_weights)

            # 2. POINTER SWAP: Restore mmap references to parameters
            # This drops the GPU reference and replaces it with the CPU mmap reference
            # without triggering a copy to RAM.
            from .ops import GGMLTensor
            
            moved_to_mmap = 0
            mmap_cache = getattr(self, "mmap_cache", None)
            
            if mmap_cache:
                # Build a local lookup for parameters
                param_map = {name: param for name, param in self.model.named_parameters()}
                
                for name in mmap_cache:
                    mmap_weight = mmap_cache[name]
                    
                    # Extract raw data from GGMLTensor if needed
                    if isinstance(mmap_weight, GGMLTensor):
                        mmap_data = mmap_weight.data
                    else:
                        mmap_data = mmap_weight

                    # Advanced Fuzzy Matching
                    target_param = None
                    
                    # 1. Exact & Standard Prefix Matches
                    if name in param_map:
                        target_param = param_map[name]
                    elif f"diffusion_model.{name}" in param_map:
                        target_param = param_map[f"diffusion_model.{name}"]
                    elif f"model.diffusion_model.{name}" in param_map:
                        target_param = param_map[f"model.diffusion_model.{name}"]
                    elif name.startswith("model.diffusion_model.") and name[len("model.diffusion_model."):] in param_map:
                        target_param = param_map[name[len("model.diffusion_model."):]]
                    
                    # 2. Fuzzy Suffix Match (if still not found)
                    if target_param is None:
                        # Try matching just the suffix (e.g. '0.attn_q.weight')
                        # normalize name
                        norm_name = name.replace("model.diffusion_model.", "")
                        for p_name, p_val in param_map.items():
                             norm_p_name = p_name.replace("model.diffusion_model.", "")
                             if norm_p_name == norm_name:
                                 target_param = p_val
                                 break

                    if target_param is not None:
                        # CRITICAL: Full replacement, not just pointer swap!
                        # Using .data = ... only changes the data pointer but leaves the GPU tensor
                        # object alive. We need to replace the entire parameter with the mmap tensor.
                        # This allows GC to release the old GPU tensor.
                        
                        # Get the matched full name for this parameter
                        matched_name = None
                        for p_name, p_val in param_map.items():
                            if p_val is target_param:
                                matched_name = p_name
                                break
                        
                        if matched_name is not None:
                            # Replace with mmap tensor wrapped in Parameter
                            comfy.utils.set_attr_param(self.model, matched_name, mmap_weight)
                            moved_to_mmap += 1
                        else:
                            # Fallback to pointer swap if we can't find the name
                            target_param.data = mmap_data.data
                            if hasattr(target_param, "patches"):
                                target_param.patches = []
                            moved_to_mmap += 1
            
            print(f"[GGUFModelPatcher] Zero-Copy: Restored {moved_to_mmap} parameters to mmap.")
            
            # CRITICAL: Clear .patches on ALL parameters, not just those matched during swap.
            # GGMLTensor.to() copies .patches, so there may be GPU tensor refs on params
            # that weren't in mmap_cache or weren't matched.
            cleared_patches = 0
            for name, param in self.model.named_parameters():
                if hasattr(param, "patches") and param.patches:
                    param.patches = []
                    cleared_patches += 1
            if cleared_patches > 0:
                print(f"[GGUFModelPatcher] Cleared .patches on {cleared_patches} parameters.")
            
            # FALLBACK: If we failed to swap ANYTHING, we MUST still offload the VRAM!
            if moved_to_mmap == 0 and device_to is not None and device_to.type == "cpu":
                 print(f"[GGUFModelPatcher] WARNING: Zero-Copy failed (0 swapped). Forcing standard offload to {device_to}...")
                 self.model.to(device_to)
            
            if device_to is not None:
                self.current_device = device_to
                
            from raylight.distributed_worker.utils import cleanup_memory
            cleanup_memory()
            print(f"[GGUFModelPatcher] Offload Complete.")

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

        # Manually move non-GGUF weights (buffers etc) to target device
        if device_to is not None and unpatch_weights:
             print(f"[GGUFModelPatcher] Offloading remaining buffers to {device_to}...")
             moved_count = 0
             for buf in self.model.buffers():
                 if buf.device.type == 'cuda' and device_to.type == 'cpu':
                     buf.data = buf.data.to(device_to)
                     moved_count += 1
             print(f"[GGUFModelPatcher] Offloaded {moved_count} buffers to {device_to}.")

    def load(self, *args, force_patch_weights=False, **kwargs):
        # GHOST RE-HYDRATION: Re-map the GGUF file ONLY if cache is missing and we have a path.
        m_cache = getattr(self, "mmap_cache", None)
        u_path = getattr(self, "unet_path", None)
        
        # Robust check for empty or missing cache
        if (m_cache is None or (isinstance(m_cache, dict) and len(m_cache) == 0)) and u_path:
            print(f"[GGUFModelPatcher] Ghost Re-hydration: Mapping {u_path}...")
            from .loader import gguf_sd_loader
            sd, _ = gguf_sd_loader(u_path)
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
        
        # FIX: Create a shallow copy of patches dict to prevent shared mutation.
        # The super().clone() already copies patches, but we ensure isolation here.
        if hasattr(self, "patches") and self.patches:
            n.patches = {k: list(v) for k, v in self.patches.items()}
        
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
