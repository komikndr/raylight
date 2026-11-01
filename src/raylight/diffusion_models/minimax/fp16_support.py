from contextlib import contextmanager
from functools import wraps

import torch

from comfy import model_base, supported_models
import comfy.ops


# Adapted from ksanislo/raylight's MiniMax H3 fp16 implementation.
class MiniMaxH3FP16(model_base.MiniMaxH3):
    def extra_conds(self, **kwargs):
        cross_attn = kwargs.get("cross_attn")
        dtype = self.get_dtype_inference()
        if cross_attn is not None and dtype == torch.float16:
            # Project/refine before narrowing: unrefined text states exceed fp16's range.
            text = cross_attn.to(device=kwargs["device"], dtype=torch.float32)
            text = self.diffusion_model.preprocess_text_embeds(text)
            kwargs = dict(kwargs, cross_attn=text.to(dtype))
        return super().extra_conds(**kwargs)


class MiniMaxH3FP16Config(supported_models.MiniMaxH3):
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        compute_dtype = self.manual_cast_dtype or self.unet_config.get("dtype")
        if compute_dtype == torch.float16 and self.quant_config is None and self.custom_operations is None:
            # The fp32 text/residual paths also need input-aware casts for plain fp16 weights.
            self.custom_operations = comfy.ops.manual_cast
        return MiniMaxH3FP16(self, device=device)


@contextmanager
def minimax_fp16_config(options):
    enabled = options.fp32_residual and options.mlp_fp16
    if not enabled:
        yield
        return
    # Scope detection to this worker's load; do not alter the upstream dtype class.
    original = supported_models.models
    supported_models.models = [MiniMaxH3FP16Config if model is supported_models.MiniMaxH3 else model
                               for model in original]
    try:
        yield
    finally:
        supported_models.models = original


def minimax_precision_loading(fn):
    @wraps(fn)
    def load(self, *args, **kwargs):
        with minimax_fp16_config(self.config.model_options.minimax):
            return fn(self, *args, **kwargs)
    return load
