"""Site-packages comfy_kitchen gateway + distributed monkey patches."""

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
    TensorCoreNVFP4Layout,
    get_layout_class,
    register_layout_class,
    register_layout_op,
)

from .kitchen_patches.fp8 import install_fp8_patches


def install_sitepkg_ck_patches(layouts=("fp8",)):
    if "fp8" in layouts:
        install_fp8_patches()


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
    "TensorCoreNVFP4Layout",
    "register_layout_op",
    "register_layout_class",
    "get_layout_class",
    "install_sitepkg_ck_patches",
]
