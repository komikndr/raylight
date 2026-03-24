"""Site-packages comfy_kitchen gateway + distributed monkey patches."""

from functools import wraps
from typing import Iterable

import comfy_kitchen as ck

from comfy_kitchen import (
    apply_rope,
    apply_rope1,
    dequantize_nvfp4,
    dequantize_per_tensor_fp8,
    disable_backend,
    enable_backend,
    list_backends,
    quantize_nvfp4,
    quantize_per_tensor_fp8,
    scaled_mm_nvfp4,
    set_backend_priority,
    use_backend,
)
from comfy_kitchen.tensor import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreGGUFLayout,
    TensorCoreNVFP4Layout,
    get_layout_class,
    register_layout_class,
    register_layout_op,
)

from .kitchen_patches.fp8 import install_fp8_patches, restore_fp8_patches
from .kitchen_patches.gguf import install_gguf_patches, restore_gguf_patches
from .kitchen_patches.nvfp4 import install_nvfp4_patches, restore_nvfp4_patches


_SITEPKG_LAYOUT_PATCHERS = {
    "fp8": (install_fp8_patches, restore_fp8_patches),
    "gguf": (install_gguf_patches, restore_gguf_patches),
    "nvfp4": (install_nvfp4_patches, restore_nvfp4_patches),
}


def _normalize_layouts(layouts):
    if layouts is None:
        return tuple(_SITEPKG_LAYOUT_PATCHERS.keys())
    if isinstance(layouts, str):
        return (layouts,)
    if isinstance(layouts, Iterable):
        return tuple(layouts)
    return ("fp8", "gguf", "nvfp4")


def install_sitepkg_ck_patches(layouts=("fp8", "gguf", "nvfp4")):
    for layout in _normalize_layouts(layouts):
        patcher = _SITEPKG_LAYOUT_PATCHERS.get(layout)
        if patcher is not None:
            patcher[0]()


def restore_sitepkg_ck_patches(layouts=("fp8", "gguf", "nvfp4")):
    for layout in _normalize_layouts(layouts):
        patcher = _SITEPKG_LAYOUT_PATCHERS.get(layout)
        if patcher is not None:
            patcher[1]()


def register_sitepkg_ck_patcher(layout, install_fn, restore_fn):
    _SITEPKG_LAYOUT_PATCHERS[layout] = (install_fn, restore_fn)


def patch_enable_comfy_kitchen_fsdp(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        patched = False
        layouts = self.parallel_dict.get("comfy_kitchen_layouts", ("fp8", "gguf", "nvfp4"))
        if self.parallel_dict.get("is_fsdp", False):
            install_sitepkg_ck_patches(layouts=layouts)
            patched = True
        try:
            return fn(self, *args, **kwargs)
        finally:
            if patched:
                restore_sitepkg_ck_patches(layouts=layouts)

    return wrapper


__all__ = [
    "ck",
    "quantize_per_tensor_fp8",
    "dequantize_per_tensor_fp8",
    "quantize_nvfp4",
    "dequantize_nvfp4",
    "scaled_mm_nvfp4",
    "apply_rope",
    "apply_rope1",
    "set_backend_priority",
    "disable_backend",
    "enable_backend",
    "list_backends",
    "use_backend",
    "BaseLayoutParams",
    "QuantizedLayout",
    "QuantizedTensor",
    "TensorCoreFP8Layout",
    "TensorCoreGGUFLayout",
    "TensorCoreNVFP4Layout",
    "register_layout_op",
    "register_layout_class",
    "get_layout_class",
    "install_sitepkg_ck_patches",
    "restore_sitepkg_ck_patches",
    "register_sitepkg_ck_patcher",
    "patch_enable_comfy_kitchen_fsdp",
]
