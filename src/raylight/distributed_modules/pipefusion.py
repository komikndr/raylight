from __future__ import annotations

from comfy import model_base

from raylight.distributed_worker.pipefusion_state import (
    PIPEFUSION_CONTEXT_KEY,
    PIPEFUSION_RUNTIME_ATTACHMENT,
    PIPEFUSION_SESSION_KEY,
    PipeFusionSession,
)


class PipeFusionInjectRegistry:
    _REGISTRY = {}

    @classmethod
    def register(cls, model_class):
        def decorator(inject_func):
            cls._REGISTRY[model_class] = inject_func
            return inject_func

        return decorator

    @classmethod
    def inject(cls, model_patcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
        base_model = model_patcher.model
        for registered_cls, inject_func in cls._REGISTRY.items():
            if isinstance(base_model, registered_cls):
                print(f"[PipeFusion] Initializing PipeFusion for {registered_cls.__name__}")
                return inject_func(
                    model_patcher,
                    base_model,
                    device_to,
                    lowvram_model_memory,
                    force_patch_weights,
                    full_load,
                )
        raise ValueError(f"Model: {type(base_model).__name__} is not yet supported for PipeFusion")


def pipefusion_outer_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    runtime = guider.model_patcher.get_attachment(PIPEFUSION_RUNTIME_ATTACHMENT)
    if runtime is None or not runtime.config.enabled:
        return executor(*args, **kwargs)

    sigmas = args[3] if len(args) > 3 else kwargs.get("sigmas", ())
    transformer_options = guider.model_options.setdefault("transformer_options", {})
    transformer_options[PIPEFUSION_SESSION_KEY] = PipeFusionSession(runtime).prepare(sigmas)
    diffusion_model = getattr(guider.model_patcher.model, "diffusion_model", None)
    if diffusion_model is not None and hasattr(diffusion_model, "reset_activation_cache"):
        diffusion_model.reset_activation_cache()

    try:
        return executor(*args, **kwargs)
    finally:
        transformer_options.pop(PIPEFUSION_SESSION_KEY, None)


def pipefusion_predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
    if model_options is None:
        model_options = {}

    transformer_options = model_options.setdefault("transformer_options", {})
    session = transformer_options.get(PIPEFUSION_SESSION_KEY)
    if session is not None:
        session.begin_step()
    return executor(x, timestep, model_options, seed)


def _resolve_transformer_options(args, kwargs):
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is not None:
        return args, kwargs, transformer_options

    if len(args) >= 6:
        transformer_options = args[5]
        if transformer_options is None:
            mutable_args = list(args)
            transformer_options = {}
            mutable_args[5] = transformer_options
            args = tuple(mutable_args)
        return args, kwargs, transformer_options

    transformer_options = {}
    kwargs["transformer_options"] = transformer_options
    return args, kwargs, transformer_options


def pipefusion_diffusion_model_wrapper(executor, *args, **kwargs):
    args, kwargs, transformer_options = _resolve_transformer_options(args, kwargs)

    session = transformer_options.get(PIPEFUSION_SESSION_KEY)
    if session is not None:
        transformer_options[PIPEFUSION_CONTEXT_KEY] = session.begin_forward()
    return executor(*args, **kwargs)


if hasattr(model_base, "WAN21"):
    from raylight.diffusion_models.wan.pipefusion import inject_wan21_pipefusion

    PipeFusionInjectRegistry.register(model_base.WAN21)(inject_wan21_pipefusion)
