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

from .ops import GGMLTensor, move_patch_to_device
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

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight, inplace_update
                )
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

        # Restore mmap-backed parameters to preserve OS page sharing across workers.
        # Without this, super().unpatch_model() calls model.to(cpu) which copies
        # GPU tensors to private CPU memory (~16GB per worker), breaking mmap sharing.
        mmap_backup = getattr(self, '_mmap_param_backup', None)
        if mmap_backup:
            for name, param in self.model.named_parameters():
                if name in mmap_backup:
                    param.data = mmap_backup[name]
            # Move only non-mmap-backed params to offload device
            if device_to is not None:
                for name, param in self.model.named_parameters():
                    if name not in mmap_backup:
                        param.data = param.data.to(device_to)
            # Clear backup so super() skips weight restoration (already done above).
            # Also clear self.backup which may contain GPU tensor references from
            # patch_weight_to_device — keeping them would prevent GC of GPU memory.
            self.backup.clear()
            # Delegate all other cleanup to super() (hooks, pins, lowvram state,
            # comfy_patched_weights flags, memory tracking, object patches).
            # Pass device_to=None so super() skips model.to(device_to) since we
            # already moved params above.
            super().unpatch_model(device_to=None, unpatch_weights=unpatch_weights)

            return

        # Fallback: no mmap backup available
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

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
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())
            # Save mmap-backed quantized params before super().load() moves them to GPU.
            # super().load() calls m.to(device_to) on all modules (model_patcher.py:865),
            # which replaces mmap references with GPU copies. We save them here so
            # unpatch_model() can restore them instead of calling model.to(cpu) which
            # would create private ~16GB copies per worker.
            if not hasattr(self, '_mmap_param_backup') or self._mmap_param_backup is None:
                self._mmap_param_backup = {}
                for name, param in self.model.named_parameters():
                    # Save ALL mmap-backed params (both quantized and F16/F32),
                    # not just quantized ones. F16/F32 GGMLTensor params are also
                    # backed by mmap and lose their sharing when m.to(device) replaces
                    # them with GPU copies.
                    if isinstance(param.data, GGMLTensor):
                        self._mmap_param_backup[name] = param.data

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
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
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_released = getattr(self, "mmap_released", False)
        mmap_backup = getattr(self, '_mmap_param_backup', None)
        if mmap_backup is not None:
            n._mmap_param_backup = dict(mmap_backup)
        if src_cls != GGUFModelPatcher:
            n.size = 0  # force recalc
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
            for actor in gpu_actors:
                loaded_futures.append(
                    actor.load_gguf_unet.remote(
                        unet_path,
                        dequant_dtype=dequant_dtype,
                        patch_dtype=patch_dtype,
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
