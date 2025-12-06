from comfy import model_base


class CFGParallelInjectRegistry:
    """Registry for registering and applying CFG context parallelism injections."""

    _REGISTRY = {}

    @classmethod
    def register(cls, model_class):
        """Register a model class and its CFG injection handler.

        Most specific (child) models must be registered before base classes.
        Example:
            @CFGParallelInjectRegistry.register(model_base.Chroma)
            def _inject_chroma(model): ...
        """
        def decorator(inject_func):
            cls._REGISTRY[model_class] = inject_func
            return inject_func
        return decorator

    @classmethod
    def inject(cls, model_patcher):
        base_model = model_patcher.model
        for registered_cls, inject_func in cls._REGISTRY.items():
            if isinstance(base_model, registered_cls):
                print(f"[CFG] Initializing CFG Parallel for {registered_cls.__name__}")
                return inject_func()
        raise ValueError(f"Model: {type(base_model).__name__} is not yet supported for CFG Parallelism")


@CFGParallelInjectRegistry.register(model_base.WAN21)
def _inject_wan21():
    from ..diffusion_models.wan.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper


@CFGParallelInjectRegistry.register(model_base.QwenImage)
def _inject_qwen():
    from ..diffusion_models.qwen.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper


@CFGParallelInjectRegistry.register(model_base.ChromaRadiance)
@CFGParallelInjectRegistry.register(model_base.Chroma)
def _inject_chroma():
    from ..diffusion_models.chroma.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper


@CFGParallelInjectRegistry.register(model_base.Flux)
def _inject_flux():
    from ..diffusion_models.flux.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper


@CFGParallelInjectRegistry.register(model_base.Hunyuan3Dv2)
def _inject_hunyuan_3dv2():
    from ..diffusion_models.hunyuan3dv2_1.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper


@CFGParallelInjectRegistry.register(model_base.HunyuanVideo)
def _inject_hunyuan():
    from ..diffusion_models.hunyuan_video.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper


@CFGParallelInjectRegistry.register(model_base.AuraFlow)
def _inject_aura_flow():
    from ..diffusion_models.aura.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper


@CFGParallelInjectRegistry.register(model_base.CosmosVideo)
def _inject_cosmos():
    from ..diffusion_models.cosmos.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper


@CFGParallelInjectRegistry.register(model_base.HiDream)
def _inject_hidream():
    from ..diffusion_models.hidream.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper


@CFGParallelInjectRegistry.register(model_base.LTXV)
def _inject_ltxv():
    from ..diffusion_models.lightricks.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper


@CFGParallelInjectRegistry.register(model_base.Lumina2)
def _inject_lumina():
    from ..diffusion_models.lumina.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper


# This will cause error for all non specified models above, since all of that will be funneled into this
@CFGParallelInjectRegistry.register(model_base.BaseModel)
def _inject_unet():
    from ..diffusion_models.modules.diffusionmodules.xdit_cfg_parallel import cfg_parallel_forward_wrapper
    return cfg_parallel_forward_wrapper
