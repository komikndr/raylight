import ray

import comfy
import comfy.model_patcher
from comfy.patcher_extension import WrappersMP

from .ray_patch_decorator import ray_patch
from ..diffusion_models.minimax.block_cache import ACTORS_CONFIG_KEY, CONFIG_KEY, RUNTIME_KEY, MiniMaxH3BlockCacheConfig
from ..diffusion_models.minimax.sla import (
    ACTORS_CONFIG_KEY as SLA_ACTORS_CONFIG_KEY,
    CONFIG_KEY as SLA_CONFIG_KEY,
    RUNTIME_KEY as SLA_RUNTIME_KEY,
    MiniMaxH3SLAConfig,
    set_active_runtime as h3_sla_set_active_runtime,
)

try:
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
except ImportError as import_error:
    MiniMaxH3SigmaShift = None
    _MINIMAX_H3_IMPORT_ERROR = import_error


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


def minimax_h3_block_cache_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original_model_options = guider.model_options
    transformer_options = original_model_options.get("transformer_options", {})
    config = transformer_options.get(CONFIG_KEY, {"enabled": False})
    if config.get("enabled", False) is True and ("easycache" in transformer_options or "teacache" in transformer_options):
        raise ValueError("MiniMax H3 Block Cache cannot be used with EasyCache or TeaCache")

    guider.model_options = comfy.model_patcher.create_model_options_clone(original_model_options)
    runtime = None
    try:
        transformer_options = guider.model_options["transformer_options"]
        config = dict(transformer_options.get(CONFIG_KEY, {"enabled": False}))
        sigmas = args[3] if len(args) > 3 else kwargs.get("sigmas", ())
        runtime = MiniMaxH3BlockCacheConfig(
            config.get("sigma_threshold", 0.0),
            config.get("start_percent", 0.0),
            config.get("end_percent", 1.0),
            config.get("max_cached_steps", 0),
            config.get("cache_depth", 0.0),
            config.get("debug", False),
        ).create_runtime(sigmas)
        runtime.log_start()
        transformer_options[CONFIG_KEY] = config
        transformer_options[RUNTIME_KEY] = runtime
        return executor(*args, **kwargs)
    finally:
        transformer_options = guider.model_options.get("transformer_options", {})
        transformer_options.pop(RUNTIME_KEY, None)
        if runtime is not None:
            runtime.log_summary()
            runtime.clear()
        guider.model_options = original_model_options


class RayMiniMaxH3BlockCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "enabled": ("BOOLEAN", {"default": True}),
                "sigma_threshold": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_cached_steps": ("INT", {"default": 2, "min": 0, "max": 10}),
                "cache_depth": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 0.95, "step": 0.05}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    def patch(self, ray_actors, enabled, sigma_threshold, start_percent, end_percent, max_cached_steps, cache_depth, debug=False):
        if start_percent > end_percent:
            raise ValueError("MiniMax H3 Block Cache start_percent must not exceed end_percent")

        config = MiniMaxH3BlockCacheConfig(
            sigma_threshold if enabled else 0.0,
            start_percent,
            end_percent,
            max_cached_steps if enabled else 0,
            cache_depth if enabled else 0.0,
            debug,
        ).to_dict()

        def apply_block_cache(model):
            if type(model.model).__name__ != "MiniMaxH3":
                raise ValueError("MiniMax H3 Block Cache only supports MiniMax H3 with Raylight USP")
            model = model.clone()
            transformer_options = model.model_options.setdefault("transformer_options", {})
            if config["enabled"] and ("easycache" in transformer_options or "teacache" in transformer_options):
                raise ValueError("MiniMax H3 Block Cache cannot be used with EasyCache or TeaCache")
            transformer_options[CONFIG_KEY] = dict(config)
            transformer_options.pop(RUNTIME_KEY, None)
            model.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, RUNTIME_KEY)
            model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, RUNTIME_KEY, minimax_h3_block_cache_sample_wrapper)
            return model

        if debug:
            print(f"[H3 Block Cache][host] node enabled={config['enabled']} workers={len(ray_actors['workers'])}", flush=True)
        ray.get([actor.model_function_runner.remote(apply_block_cache) for actor in ray_actors["workers"]])
        configured_actors = ray_actors.copy()
        configured_actors[ACTORS_CONFIG_KEY] = config
        return (configured_actors,)


def minimax_h3_sla_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original_model_options = guider.model_options

    guider.model_options = comfy.model_patcher.create_model_options_clone(original_model_options)
    runtime = None
    try:
        transformer_options = guider.model_options["transformer_options"]
        config = dict(transformer_options.get(SLA_CONFIG_KEY, {"enabled": False}))
        sigmas = args[3] if len(args) > 3 else kwargs.get("sigmas", ())
        runtime = MiniMaxH3SLAConfig.from_dict(config).create_runtime(sigmas)
        runtime.log_start()
        transformer_options[SLA_RUNTIME_KEY] = runtime
        return executor(*args, **kwargs)
    finally:
        transformer_options = guider.model_options.get("transformer_options", {})
        transformer_options.pop(SLA_RUNTIME_KEY, None)
        if runtime is not None:
            runtime.log_summary()
            runtime.clear()
        h3_sla_set_active_runtime(None)
        guider.model_options = original_model_options


class RayMiniMaxH3SLAAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "enabled": ("BOOLEAN", {"default": False}),
                "sparsity_ratio": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 0.95, "step": 0.05}),
                "block_size": (["64", "128"], {"default": "64"}),
                "min_seq_len": ("INT", {"default": 8192, "min": 0, "max": 1000000, "step": 1024}),
                "dense_last_steps": ("INT", {"default": 0, "min": 0, "max": 8}),
                "protect_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    def patch(self, ray_actors, enabled, sparsity_ratio, block_size, min_seq_len, dense_last_steps, protect_audio, debug=False):
        config = MiniMaxH3SLAConfig(
            enabled=enabled,
            sparsity_ratio=sparsity_ratio,
            block_size=int(block_size),
            min_seq_len=min_seq_len,
            dense_last_steps=dense_last_steps,
            protect_audio=protect_audio,
            debug=debug,
        ).to_dict()

        def apply_sla(model):
            if type(model.model).__name__ != "MiniMaxH3":
                raise ValueError("MiniMax H3 SLA Attention only supports MiniMax H3 with Raylight USP")
            model = model.clone()
            transformer_options = model.model_options.setdefault("transformer_options", {})
            transformer_options[SLA_CONFIG_KEY] = dict(config)
            transformer_options.pop(SLA_RUNTIME_KEY, None)
            model.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, SLA_RUNTIME_KEY)
            model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, SLA_RUNTIME_KEY, minimax_h3_sla_sample_wrapper)
            return model

        if debug:
            print(f"[H3 SLA][host] node enabled={config['enabled']} sparsity={config['sparsity_ratio']} workers={len(ray_actors['workers'])}", flush=True)
        ray.get([actor.model_function_runner.remote(apply_sla) for actor in ray_actors["workers"]])
        configured_actors = ray_actors.copy()
        configured_actors[SLA_ACTORS_CONFIG_KEY] = config
        return (configured_actors,)


NODE_CLASS_MAPPINGS = {
    "RayMiniMaxH3SigmaShift": RayMiniMaxH3SigmaShift,
    "RayMiniMaxH3BlockCache": RayMiniMaxH3BlockCache,
    "RayMiniMaxH3SLAAttention": RayMiniMaxH3SLAAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayMiniMaxH3SigmaShift": "MiniMax H3 Sigma Shift (Ray)",
    "RayMiniMaxH3BlockCache": "MiniMax H3 Block Cache (Ray USP)",
    "RayMiniMaxH3SLAAttention": "MiniMax H3 SLA Attention (Ray USP)",
}
