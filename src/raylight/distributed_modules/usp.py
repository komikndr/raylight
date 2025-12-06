import types
from comfy import model_base


class USPInjectRegistry:
    """Registry for registering and applying USP context parallelism injections."""

    _REGISTRY = {}

    @classmethod
    def register(cls, model_class):
        """Register a model class and its USP injection handler.

        Most specific (child) models must be registered before base classes.
        Example:
            @USPInjectRegistry.register(model_base.Chroma)
            def _inject_chroma(model): ...
        """
        def decorator(inject_func):
            cls._REGISTRY[model_class] = inject_func
            return inject_func
        return decorator

    @classmethod
    def inject(cls, model_patcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
        base_model = model_patcher.model
        for registered_cls, inject_func in cls._REGISTRY.items():
            if isinstance(base_model, registered_cls):
                print(f"[USP] Initializing USP for {registered_cls.__name__}")
                return inject_func(model_patcher, base_model, device_to, lowvram_model_memory, force_patch_weights, full_load)
        raise ValueError(f"Model: {type(base_model).__name__} is not yet supported for USP Parallelism")


@USPInjectRegistry.register(model_base.WAN21_Vace)
def _inject_wan21_vace(model_patcher, base_model, *args):
    pass


@USPInjectRegistry.register(model_base.WAN21_Camera)
def _inject_wan21_camera(model_patcher, base_model, *args):
    from ..diffusion_models.wan.xdit_context_parallel import (
        usp_camera_dit_forward,
        usp_self_attn_forward,
        usp_t2v_cross_attn_forward,
    )
    model = base_model.diffusion_model
    for block in model.blocks:
        block.self_attn.forward = types.MethodType(usp_self_attn_forward, block.self_attn)
        block.cross_attn.forward = types.MethodType(usp_t2v_cross_attn_forward, block.cross_attn)
    model.forward_orig = types.MethodType(usp_camera_dit_forward, model)


@USPInjectRegistry.register(model_base.WAN21_HuMo)
def _inject_wan21_humo(model_patcher, base_model, *args):
    from ..diffusion_models.wan.xdit_context_parallel import (
        usp_humo_dit_forward,
        usp_self_attn_forward,
        usp_t2v_cross_attn_forward,
        usp_t2v_cross_attn_gather_forward
    )
    model = base_model.diffusion_model
    model.wan_attn_block_class.audio_cross_attn.forward = type.MethodType(usp_t2v_cross_attn_gather_forward, model.wan_attn_block_class.audio_cross_attn)
    for block in model.blocks:
        block.self_attn.forward = types.MethodType(usp_self_attn_forward, block.self_attn)
        block.cross_attn.forward = types.MethodType(usp_t2v_cross_attn_forward, block.cross_attn)
    model.forward_orig = types.MethodType(usp_humo_dit_forward, model)


@USPInjectRegistry.register(model_base.WAN22_Animate)
def _inject_wan22_animate(model_patcher, base_model, *args):
    from ..diffusion_models.wan.xdit_context_parallel_animate import (
        usp_animate_dit_forward,
        usp_face_block_forward,
    )
    from ..diffusion_models.wan.xdit_context_parallel import (
        usp_self_attn_forward,
        usp_t2v_cross_attn_forward,
    )
    model = base_model.diffusion_model
    for block in model.blocks:
        block.self_attn.forward = types.MethodType(usp_self_attn_forward, block.self_attn)
        block.cross_attn.forward = types.MethodType(usp_t2v_cross_attn_forward, block.cross_attn)
    for face_block in model.face_adapter.fuser_blocks:
        face_block.forward = types.MethodType(usp_face_block_forward, face_block)
    model.forward_orig = types.MethodType(usp_animate_dit_forward, model)


@USPInjectRegistry.register(model_base.WAN22_S2V)
def _inject_wan22_s2v(model_patcher, base_model, *args):
    from ..diffusion_models.wan.xdit_context_parallel import (
        usp_s2v_dit_forward,
        usp_self_attn_forward,
        usp_t2v_cross_attn_forward,
    )
    model = base_model.diffusion_model
    for block in model.blocks:
        block.self_attn.forward = types.MethodType(usp_self_attn_forward, block.self_attn)
        block.cross_attn.forward = types.MethodType(usp_t2v_cross_attn_forward, block.cross_attn)
    model.forward_orig = types.MethodType(usp_s2v_dit_forward, model)


@USPInjectRegistry.register(model_base.WAN21)
def _inject_wan21(model_patcher, base_model, *args):
    from ..diffusion_models.wan.xdit_context_parallel import (
        usp_self_attn_forward,
        usp_dit_forward,
        usp_i2v_cross_attn_forward,
        usp_t2v_cross_attn_forward
    )
    from comfy.ldm.wan.model import WanT2VCrossAttention, WanI2VCrossAttention

    model = base_model.diffusion_model
    for block in model.blocks:
        block.self_attn.forward = types.MethodType(usp_self_attn_forward, block.self_attn)
        if isinstance(block.cross_attn, WanT2VCrossAttention):
            block.cross_attn.forward = types.MethodType(usp_t2v_cross_attn_forward, block.cross_attn)
        elif isinstance(block.cross_attn, WanI2VCrossAttention):
            block.cross_attn.forward = types.MethodType(usp_i2v_cross_attn_forward, block.cross_attn)
    model.forward_orig = types.MethodType(usp_dit_forward, model)


# Chroma Radiance should be using this since the forward_orig MRO in Chroma Radiance is from Chroma itself
@USPInjectRegistry.register(model_base.ChromaRadiance)
@USPInjectRegistry.register(model_base.Chroma)
def _inject_chroma(model_patcher, base_model, *args):
    from ..diffusion_models.chroma.xdit_context_parallel import (
        usp_dit_forward,
        usp_single_stream_forward,
        usp_double_stream_forward
    )
    model = base_model.diffusion_model
    for block in model.double_blocks:
        block.forward = types.MethodType(usp_double_stream_forward, block)
    for block in model.single_blocks:
        block.forward = types.MethodType(usp_single_stream_forward, block)
    model.forward_orig = types.MethodType(usp_dit_forward, model)


@USPInjectRegistry.register(model_base.Flux)
def _inject_flux(model_patcher, base_model, *args):
    from ..diffusion_models.flux.xdit_context_parallel import (
        usp_dit_forward,
        usp_single_stream_forward,
        usp_double_stream_forward
    )
    model = base_model.diffusion_model
    for block in model.double_blocks:
        block.forward = types.MethodType(usp_double_stream_forward, block)
    for block in model.single_blocks:
        block.forward = types.MethodType(usp_single_stream_forward, block)
    model.forward_orig = types.MethodType(usp_dit_forward, model)


@USPInjectRegistry.register(model_base.Hunyuan3Dv2)
def _inject_hunyuan_3dv2(model_patcher, base_model, *args):
    from ..diffusion_models.hunyuan3d.xdit_context_parallel import (
        usp_dit_forward,
    )
    from ..diffusion_models.flux.xdit_context_parallel import (
        usp_single_stream_forward,
        usp_double_stream_forward
    )
    model = base_model.diffusion_model
    for block in model.double_blocks:
        block.forward = types.MethodType(usp_double_stream_forward, block)
    for block in model.single_blocks:
        block.forward = types.MethodType(usp_single_stream_forward, block)
    model.forward_orig = types.MethodType(usp_dit_forward, model)


@USPInjectRegistry.register(model_base.HunyuanVideo)
def _inject_hunyuan(model_patcher, base_model, *args):
    from ..diffusion_models.hunyuan_video.xdit_context_parallel import (
        usp_dit_forward,
        usp_token_refiner_forward
    )
    from ..diffusion_models.flux.xdit_context_parallel import (
        usp_single_stream_forward,
        usp_double_stream_forward
    )
    model = base_model.diffusion_model
    for block in model.double_blocks:
        block.forward = types.MethodType(usp_double_stream_forward, block)
    for block in model.single_blocks:
        block.forward = types.MethodType(usp_single_stream_forward, block)
    for block_token_refiner in model.txt_in.individual_token_refiner.blocks:
        block_token_refiner.forward = types.MethodType(usp_token_refiner_forward, block_token_refiner)
    model.forward_orig = types.MethodType(usp_dit_forward, model)


@USPInjectRegistry.register(model_base.QwenImage)
def _inject_qwen(model_patcher, base_model, *args):
    from ..diffusion_models.qwen_image.xdit_context_parallel import (
        usp_dit_forward,
        usp_attn_forward,
    )
    model = base_model.diffusion_model
    for block in model.transformer_blocks:
        block.attn.forward = types.MethodType(usp_attn_forward, block.attn)
    model._forward = types.MethodType(usp_dit_forward, model)


@USPInjectRegistry.register(model_base.CosmosPredict2)
def _inject_cosmos_predict2(model_patcher, base_model, *args):
    from ..diffusion_models.cosmos.xdit_context_parallel import (
        usp_xfuser_attention_op,
        usp_mini_train_dit_forward
    )
    model = base_model.diffusion_model
    for block in model.blocks:
        block.cross_attn.attn_op = usp_xfuser_attention_op
        block.self_attn.attn_op = usp_xfuser_attention_op
    model._forward = types.MethodType(usp_mini_train_dit_forward, model)


@USPInjectRegistry.register(model_base.CosmosVideo)
def _inject_cosmos_video(model_patcher, base_model, *args):
    pass
    from ..diffusion_models.cosmos.xdit_context_parallel import (
        usp_general_dit_forward,
        usp_general_attention_forward
    )
    model = base_model.diffusion_model
    model._forward = types.MethodType(usp_general_dit_forward, model)


@USPInjectRegistry.register(model_base.Lumina2)  # Lumina and Z Image
def _inject_lumina(model_patcher, base_model, *args):
    from ..diffusion_models.lumina.xdit_context_parallel import (
        usp_dit_forward,
        usp_joint_attention_forward
    )
    model = base_model.diffusion_model
    for block in model.layers:
        block.attention.forward = types.MethodType(usp_joint_attention_forward, block.attention)
    model._forward = types.MethodType(usp_dit_forward, model)
