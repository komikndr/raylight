import torch
from functools import wraps
from comfy.quant_ops import TensorCoreFP8Layout


def dequantize_ray_temp_fix(qdata, scale, orig_dtype, **kwargs):
    plain_tensor = torch.ops.aten._to_copy.default(qdata, dtype=torch.bfloat16)
    plain_tensor.mul_(scale)
    return plain_tensor


def patch_temp_fix_fp8mixed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        original_dequantize = TensorCoreFP8Layout.dequantize

        try:
            TensorCoreFP8Layout.dequantize = staticmethod(
                dequantize_ray_temp_fix
            )
            return func(*args, **kwargs)

        finally:
            TensorCoreFP8Layout.dequantize = original_dequantize
    return wrapper
