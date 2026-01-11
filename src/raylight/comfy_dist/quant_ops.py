import comfy_kitchen as ck
from functools import wraps
from comfy_kitchen.tensor import TensorCoreFP8Layout, TensorCoreNVFP4Layout


def dequantize_ray_temp_fix_fp8(qdata, params, dtype):
    return ck.dequantize_per_tensor_fp8(qdata, params.scale, dtype)


def dequantize_ray_temp_fix_nvfp4(qdata, params, dtype):
    return ck.dequantize_nvfp4(qdata, params.scale, params.block_scale, dtype)


def patch_temp_fix_ck_ops(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]

        # If no override → do nothing
        if getattr(self, "overwrite_cast_dtype", None) is None:
            return func(*args, **kwargs)

        original_fp8 = TensorCoreFP8Layout.dequantize
        original_nvfp4 = TensorCoreNVFP4Layout.dequantize

        try:
            TensorCoreFP8Layout.dequantize = classmethod(
                lambda cls, qdata, params:
                    dequantize_ray_temp_fix_fp8(
                        qdata,
                        params,
                        self.overwrite_cast_dtype
                    )
            )

            TensorCoreNVFP4Layout.dequantize = classmethod(
                lambda cls, qdata, params:
                    dequantize_ray_temp_fix_nvfp4(
                        qdata,
                        params,
                        self.overwrite_cast_dtype
                    )
            )

            return func(*args, **kwargs)

        finally:
            TensorCoreFP8Layout.dequantize = original_fp8
            TensorCoreNVFP4Layout.dequantize = original_nvfp4

    return wrapper
