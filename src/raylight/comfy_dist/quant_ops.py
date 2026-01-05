import torch
from functools import wraps
from comfy.quant_ops import TensorCoreFP8Layout


def dequantize_ray_temp_fix(qdata, scale, *, dtype, **kwargs):
    plain_tensor = torch.ops.aten._to_copy.default(qdata, dtype=dtype)
    plain_tensor.mul_(scale)
    return plain_tensor


def patch_temp_fix_fp8mixed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # If no override do NOTHING
        if getattr(self, "overwrite_cast_dtype", None) is None:
            return func(*args, **kwargs)

        original_dequantize = TensorCoreFP8Layout.dequantize

        try:
            TensorCoreFP8Layout.dequantize = staticmethod(
                lambda *a, **k: dequantize_ray_temp_fix(
                    *a, **k, dtype=self.overwrite_cast_dtype
                )
            )
            return func(*args, **kwargs)

        finally:
            TensorCoreFP8Layout.dequantize = original_dequantize

    return wrapper
