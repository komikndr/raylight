import folder_paths
import ray

from .ray_patch_decorator import ray_patch
from comfy.patcher_extension import WrappersMP
from raylight.distributed_worker.raylightworker_config_schema import MiniMaxOptions, ModelOptions
from raylight.distributed_modules.inner_attention import (
    H3_SLA_PREPARED_KEY, INNER_ATTENTION_KEY, MiniMaxH3SLA, clear_inner_attention, create_inner_attention, set_inner_attention,
)

try:
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
except ImportError as import_error:
    MiniMaxH3SigmaShift = None
    _MINIMAX_H3_IMPORT_ERROR = import_error


class RayWorkerMinimaxOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "attention_fp16": ("BOOLEAN", {"default": False, "tooltip": "Use fp16 attention projections with fp32 output rescaling."}),
            "mlp_fp16": ("BOOLEAN", {"default": False, "tooltip": "Use fp16 MLP projections with overflow-safe rescaling."}),
            "fp32_residual": ("BOOLEAN", {"default": False, "tooltip": "Keep residual accumulation in fp32. Together with fp16 MLP, allows H3 fp16 model loading."}),
            "mlp_chunk_tokens": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 512,
                                       "tooltip": "Token chunk size for the H3 MLP. 0 disables chunking. Chunking can change quantized seed results."}),
        }}

    RETURN_TYPES = ("RAY_MODEL_OPTIONS",)
    RETURN_NAMES = ("worker_options",)
    FUNCTION = "build"
    CATEGORY = "Raylight"

    def build(self, attention_fp16, mlp_fp16, fp32_residual, mlp_chunk_tokens):
        return (ModelOptions(minimax=MiniMaxOptions(
            attention_fp16=attention_fp16, mlp_fp16=mlp_fp16,
            fp32_residual=fp32_residual, mlp_chunk_tokens=mlp_chunk_tokens,
        )),)


class RayMiniMaxH3SigmaShift:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    @ray_patch
    def patch(self, model, shift_video, shift_audio):
        if MiniMaxH3SigmaShift is None:
            raise RuntimeError(
                "MiniMax H3 Sigma Shift is unavailable. Install or update ComfyUI to a version that provides "
                "comfy_extras.nodes_minimax_h3."
            ) from _MINIMAX_H3_IMPORT_ERROR

        return MiniMaxH3SigmaShift.execute(model, shift_video, shift_audio)[0]


H3_SLA_WRAPPER_KEY = "raylight:minimax_h3_sla"


def _h3_sla_diffusion_wrapper(executor, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options", {})
    positional = list(args)
    if "transformer_options" not in kwargs and len(positional) > 3:
        transformer_options = positional[3]
    processor = transformer_options.get(INNER_ATTENTION_KEY)
    if processor is not None:
        transformer_options = dict(transformer_options)
        payload = kwargs.get("minimax_payload")
        if payload is None and len(positional) > 4:
            payload = positional[4]
        transformer_options[H3_SLA_PREPARED_KEY] = processor.prepare(
            transformer_options, payload
        )
        if "transformer_options" in kwargs:
            kwargs["transformer_options"] = transformer_options
        elif len(positional) > 3:
            positional[3] = transformer_options
    return executor(*positional, **kwargs)


class RayMiniMaxH3SLA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "enabled": ("BOOLEAN", {"default": True}),
                "sparsity_ratio": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 0.95, "step": 0.01}),
                "block_size": (["32", "64", "128"], {"default": "64"}),
                "min_seq_len": ("INT", {"default": 8192, "min": 1, "max": 1000000}),
                "dense_last_steps": ("INT", {"default": 1, "min": 0, "max": 10000}),
                "protect_audio": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    @ray_patch
    def patch(self, model, enabled, sparsity_ratio, block_size, min_seq_len, dense_last_steps, protect_audio):
        if not enabled:
            model = clear_inner_attention(model, MiniMaxH3SLA)
            model.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, H3_SLA_WRAPPER_KEY)
            return model
        processor = create_inner_attention(
            "raylight:minimax_h3_sla",
            enabled=enabled,
            sparsity_ratio=sparsity_ratio,
            block_size=int(block_size),
            min_seq_len=min_seq_len,
            dense_last_steps=dense_last_steps,
            protect_audio=protect_audio,
        )
        model = set_inner_attention(model, processor)
        model.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, H3_SLA_WRAPPER_KEY)
        model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, H3_SLA_WRAPPER_KEY, _h3_sla_diffusion_wrapper)
        return model


class RayMiniMaxH3FunControlNetApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "ray_vae": ("RAY_VAE",),
                "model_patch_name": (folder_paths.get_filename_list("model_patches"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "control_video": ("IMAGE",),
                "mask": ("MASK", {"tooltip": "1 marks the regions to regenerate."}),
                "source_video": ("IMAGE", {"tooltip": "Video behind the mask; only read when a mask is given."}),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "apply"
    CATEGORY = "Raylight/extra"

    def apply(self, ray_actors, ray_vae, model_patch_name, strength, start_percent, end_percent,
              control_video=None, mask=None, source_video=None):
        if strength == 0 or (control_video is None and mask is None):
            return (ray_actors,)
        folder_paths.get_full_path_or_raise("model_patches", model_patch_name)
        control = control_video[..., :3].movedim(-1, 1) if control_video is not None else None
        source = source_video[..., :3].movedim(-1, 1) if mask is not None and source_video is not None else None
        ray.get([
            actor.ray_minimax_h3_fun_control_apply.remote(
                model_patch_name, ray_vae, strength, start_percent, end_percent, control, mask, source)
            for actor in ray_actors["workers"]
        ])
        return (ray_actors,)


NODE_CLASS_MAPPINGS = {
    "RayWorkerMinimaxOptions": RayWorkerMinimaxOptions,
    "RayMiniMaxH3SigmaShift": RayMiniMaxH3SigmaShift,
    "RayMiniMaxH3SLA": RayMiniMaxH3SLA,
    "RayMiniMaxH3FunControlNetApply": RayMiniMaxH3FunControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayWorkerMinimaxOptions": "RayWorker Minimax Option",
    "RayMiniMaxH3SigmaShift": "MiniMax H3 Sigma Shift (Ray)",
    "RayMiniMaxH3SLA": "MiniMax H3 SLA Attention (Ray)",
    "RayMiniMaxH3FunControlNetApply": "Apply MiniMax H3 Fun ControlNet (Ray)",
}
