import comfy_kitchen as ck
from functools import wraps
from comfy_kitchen.tensor import TensorCoreFP8Layout, TensorCoreNVFP4Layout

_PATCH_INSTALL_LOGGED = False
_FP8_FALLBACK_LOGGED = False
_NVFP4_FALLBACK_LOGGED = False


def _tensor_desc(tensor):
    shape = getattr(tensor, "shape", None)
    device = getattr(tensor, "device", None)
    dtype = getattr(tensor, "dtype", None)
    return f"shape={tuple(shape) if shape is not None else None} device={device} dtype={dtype}"


def dequantize_ray_temp_fix_fp8(qdata, params, dtype):
    global _FP8_FALLBACK_LOGGED
    if not _FP8_FALLBACK_LOGGED:
        print(f"[Raylight][comfy_kitchen][fp8] fallback dequantize dtype={dtype} qdata={_tensor_desc(qdata)}")
        _FP8_FALLBACK_LOGGED = True
    return ck.dequantize_per_tensor_fp8(qdata, params.scale, dtype)


def dequantize_ray_temp_fix_nvfp4(qdata, params, dtype):
    global _NVFP4_FALLBACK_LOGGED
    if not _NVFP4_FALLBACK_LOGGED:
        print(f"[Raylight][comfy_kitchen][nvfp4] fallback dequantize dtype={dtype} qdata={_tensor_desc(qdata)}")
        _NVFP4_FALLBACK_LOGGED = True
    return ck.dequantize_nvfp4(qdata, params.scale, params.block_scale, dtype)


def patch_temp_fix_ck_ops(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _PATCH_INSTALL_LOGGED
        self = args[0]
        parallel_dict = getattr(self, "parallel_dict", {}) or {}
        overwrite_cast_dtype = getattr(self, "overwrite_cast_dtype", None)
        layouts = parallel_dict.get("comfy_kitchen_layouts", ("fp8", "nvfp4"))
        install_ck_patches = bool(parallel_dict.get("is_quant", False))
        ck_patched = False
        restore_sitepkg_ck_patches = None
        original_fp8 = None
        original_nvfp4 = None

        try:
            if install_ck_patches:
                from raylight.comfy_dist.kitchen_distributed import install_sitepkg_ck_patches, restore_sitepkg_ck_patches

                install_sitepkg_ck_patches(layouts=layouts)
                ck_patched = True
                if not _PATCH_INSTALL_LOGGED:
                    print(
                        "[Raylight][comfy_kitchen] installing quant patches "
                        f"layouts={layouts} is_fsdp={parallel_dict.get('is_fsdp', False)} is_quant={parallel_dict.get('is_quant', False)}"
                    )
                    _PATCH_INSTALL_LOGGED = True

            if overwrite_cast_dtype is not None:
                original_fp8 = TensorCoreFP8Layout.dequantize
                original_nvfp4 = TensorCoreNVFP4Layout.dequantize

                TensorCoreFP8Layout.dequantize = classmethod(
                    lambda cls, qdata, params:
                        dequantize_ray_temp_fix_fp8(
                            qdata,
                            params,
                            overwrite_cast_dtype
                        )
                )

                TensorCoreNVFP4Layout.dequantize = classmethod(
                    lambda cls, qdata, params:
                        dequantize_ray_temp_fix_nvfp4(
                            qdata,
                            params,
                            overwrite_cast_dtype
                        )
                )

            return func(*args, **kwargs)

        finally:
            if original_fp8 is not None:
                TensorCoreFP8Layout.dequantize = original_fp8
            if original_nvfp4 is not None:
                TensorCoreNVFP4Layout.dequantize = original_nvfp4
            if ck_patched and restore_sitepkg_ck_patches is not None:
                restore_sitepkg_ck_patches(layouts=layouts)

    return wrapper
