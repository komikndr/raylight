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


def _patch_wan_attention_blocks(
    model, usp_self_attn_forward, usp_t2v_cross_attn_forward, usp_i2v_cross_attn_forward=None, include_vace_blocks=False
):
    from comfy.ldm.wan.model import WanI2VCrossAttention, WanT2VCrossAttention

    block_groups = [getattr(model, "blocks", [])]
    if include_vace_blocks and hasattr(model, "vace_blocks"):
        block_groups.append(model.vace_blocks)

    for blocks in block_groups:
        for block in blocks:
            if hasattr(block, "self_attn"):
                block.self_attn.forward = types.MethodType(usp_self_attn_forward, block.self_attn)
            if not hasattr(block, "cross_attn"):
                continue
            if isinstance(block.cross_attn, WanT2VCrossAttention):
                block.cross_attn.forward = types.MethodType(usp_t2v_cross_attn_forward, block.cross_attn)
            elif usp_i2v_cross_attn_forward is not None and isinstance(block.cross_attn, WanI2VCrossAttention):
                block.cross_attn.forward = types.MethodType(usp_i2v_cross_attn_forward, block.cross_attn)


def _patch_pixeldit_blocks(model, usp_joint_attention_forward, usp_rotary_attention_forward, usp_pit_block_forward):
    for block in model.patch_blocks:
        block.attn.forward = types.MethodType(usp_joint_attention_forward, block.attn)
    for block in model.pixel_blocks:
        block.attn.forward = types.MethodType(usp_rotary_attention_forward, block.attn)
        block.forward = types.MethodType(usp_pit_block_forward, block)


if hasattr(model_base, "WAN21_Vace"):

    @USPInjectRegistry.register(model_base.WAN21_Vace)
    def _inject_wan21_vace(model_patcher, base_model, *args):
        from ..diffusion_models.wan.xdit_context_parallel import (
            usp_dit_forward,
            usp_self_attn_forward,
            usp_t2v_cross_attn_forward,
            usp_vace_dit_forward,
        )

        model = base_model.diffusion_model
        _patch_wan_attention_blocks(
            model,
            usp_self_attn_forward,
            usp_t2v_cross_attn_forward,
            include_vace_blocks=True,
        )
        model.forward_orig = types.MethodType(usp_vace_dit_forward, model)


if hasattr(model_base, "WAN21_Camera"):

    @USPInjectRegistry.register(model_base.WAN21_Camera)
    def _inject_wan21_camera(model_patcher, base_model, *args):
        from ..diffusion_models.wan.xdit_context_parallel import (
            usp_camera_dit_forward,
            usp_self_attn_forward,
            usp_i2v_cross_attn_forward,
            usp_t2v_cross_attn_forward,
        )

        model = base_model.diffusion_model
        _patch_wan_attention_blocks(
            model,
            usp_self_attn_forward,
            usp_t2v_cross_attn_forward,
            usp_i2v_cross_attn_forward,
        )
        model.forward_orig = types.MethodType(usp_camera_dit_forward, model)


if hasattr(model_base, "WAN21_HuMo"):

    @USPInjectRegistry.register(model_base.WAN21_HuMo)
    def _inject_wan21_humo(model_patcher, base_model, *args):
        from ..diffusion_models.wan.xdit_context_parallel import (
            usp_humo_dit_forward,
            usp_self_attn_forward,
            usp_t2v_cross_attn_forward,
            usp_t2v_cross_attn_gather_forward,
        )

        model = base_model.diffusion_model
        _patch_wan_attention_blocks(model, usp_self_attn_forward, usp_t2v_cross_attn_forward)
        for block in model.blocks:
            audio_cross_attn = getattr(getattr(block, "audio_cross_attn_wrapper", None), "audio_cross_attn", None)
            if audio_cross_attn is not None:
                audio_cross_attn.forward = types.MethodType(usp_t2v_cross_attn_gather_forward, audio_cross_attn)
        model.forward_orig = types.MethodType(usp_humo_dit_forward, model)


if hasattr(model_base, "WAN22_Animate"):

    @USPInjectRegistry.register(model_base.WAN22_Animate)
    def _inject_wan22_animate(model_patcher, base_model, *args):
        from ..diffusion_models.wan.xdit_context_parallel_animate import (
            usp_animate_dit_forward,
            usp_face_block_forward,
        )
        from ..diffusion_models.wan.xdit_context_parallel import (
            usp_self_attn_forward,
            usp_i2v_cross_attn_forward,
            usp_t2v_cross_attn_forward,
        )

        model = base_model.diffusion_model
        _patch_wan_attention_blocks(
            model,
            usp_self_attn_forward,
            usp_t2v_cross_attn_forward,
            usp_i2v_cross_attn_forward,
        )
        for face_block in model.face_adapter.fuser_blocks:
            face_block.forward = types.MethodType(usp_face_block_forward, face_block)
        model.forward_orig = types.MethodType(usp_animate_dit_forward, model)


if hasattr(model_base, "WAN22_S2V"):

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


if hasattr(model_base, "WAN21_FlowRVS"):

    @USPInjectRegistry.register(model_base.WAN21_FlowRVS)
    def _inject_wan21_flowrvs(model_patcher, base_model, *args):
        from ..diffusion_models.wan.xdit_context_parallel import (
            usp_self_attn_forward,
            usp_dit_forward,
            usp_i2v_cross_attn_forward,
            usp_t2v_cross_attn_forward,
        )

        model = base_model.diffusion_model
        _patch_wan_attention_blocks(model, usp_self_attn_forward, usp_t2v_cross_attn_forward, usp_i2v_cross_attn_forward)
        model.forward_orig = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "WAN21_CausalAR"):

    @USPInjectRegistry.register(model_base.WAN21_CausalAR)
    def _inject_wan21_causal_ar(model_patcher, base_model, *args):
        from comfy.ldm.wan.model import WanI2VCrossAttention, WanT2VCrossAttention
        from ..diffusion_models.wan.xdit_context_parallel import (
            usp_causal_ar_block_forward,
            usp_causal_ar_forward,
            usp_causal_ar_forward_block,
            usp_causal_ar_self_attn_forward,
            usp_dit_forward,
            usp_i2v_cross_attn_forward,
            usp_t2v_cross_attn_forward,
        )

        model = base_model.diffusion_model
        for block in model.blocks:
            block.forward = types.MethodType(usp_causal_ar_block_forward, block)
            block.self_attn.forward = types.MethodType(usp_causal_ar_self_attn_forward, block.self_attn)
            if not hasattr(block, "cross_attn"):
                continue
            if isinstance(block.cross_attn, WanT2VCrossAttention):
                block.cross_attn.forward = types.MethodType(usp_t2v_cross_attn_forward, block.cross_attn)
            elif isinstance(block.cross_attn, WanI2VCrossAttention):
                block.cross_attn.forward = types.MethodType(usp_i2v_cross_attn_forward, block.cross_attn)
        model.forward_orig = types.MethodType(usp_dit_forward, model)
        model.forward = types.MethodType(usp_causal_ar_forward, model)
        model.forward_block = types.MethodType(usp_causal_ar_forward_block, model)


if hasattr(model_base, "WAN22_WanDancer"):

    @USPInjectRegistry.register(model_base.WAN22_WanDancer)
    def _inject_wan22_wandancer(model_patcher, base_model, *args):
        from ..diffusion_models.wan.xdit_context_parallel import (
            usp_wandancer_dit_forward,
            usp_music_self_attn_forward,
            usp_self_attn_forward,
            usp_i2v_cross_attn_forward,
            usp_t2v_cross_attn_forward,
        )

        model = base_model.diffusion_model
        _patch_wan_attention_blocks(
            model,
            usp_self_attn_forward,
            usp_t2v_cross_attn_forward,
            usp_i2v_cross_attn_forward,
        )
        for layer in model.music_encoder:
            layer.self_attn.forward = types.MethodType(usp_music_self_attn_forward, layer.self_attn)
        model.forward_orig = types.MethodType(usp_wandancer_dit_forward, model)


if hasattr(model_base, "WAN21_SCAIL"):

    @USPInjectRegistry.register(model_base.WAN21_SCAIL)
    def _inject_wan21_scail(model_patcher, base_model, *args):
        from ..diffusion_models.wan.xdit_context_parallel import (
            usp_scail_dit_forward,
            usp_self_attn_forward,
            usp_i2v_cross_attn_forward,
            usp_t2v_cross_attn_forward,
        )

        model = base_model.diffusion_model
        _patch_wan_attention_blocks(
            model,
            usp_self_attn_forward,
            usp_t2v_cross_attn_forward,
            usp_i2v_cross_attn_forward,
        )
        model.forward_orig = types.MethodType(usp_scail_dit_forward, model)


if hasattr(model_base, "WAN21"):

    if hasattr(model_base, "WAN_Animate2"):

        @USPInjectRegistry.register(model_base.WAN_Animate2)
        def _inject_wan_animate2(model_patcher, base_model, *args):
            from comfy.ldm.wan.model import WanI2VCrossAttention
            from ..diffusion_models.wan.xdit_context_parallel import (
                usp_i2v_cross_attn_forward,
            )
            from ..diffusion_models.wan.xdit_context_parallel_animate2 import (
                usp_animate2_dit_forward,
                usp_animate2_kv_from_input,
                usp_animate2_self_attn_forward_gen,
                usp_animate2_self_attn_forward_pose,
            )

            model = base_model.diffusion_model
            for block in model.blocks:
                block.self_attn.forward_pose = types.MethodType(usp_animate2_self_attn_forward_pose, block.self_attn)
                block.self_attn.kv = types.MethodType(usp_animate2_kv_from_input, block.self_attn)
                block.self_attn.forward_gen = types.MethodType(usp_animate2_self_attn_forward_gen, block.self_attn)
                if isinstance(block.cross_attn, WanI2VCrossAttention):
                    block.cross_attn.forward = types.MethodType(usp_i2v_cross_attn_forward, block.cross_attn)
            model.forward_orig = types.MethodType(usp_animate2_dit_forward, model)

    @USPInjectRegistry.register(model_base.WAN21)
    def _inject_wan21(model_patcher, base_model, *args):
        from ..diffusion_models.wan.xdit_context_parallel import (
            usp_self_attn_forward,
            usp_dit_forward,
            usp_i2v_cross_attn_forward,
            usp_t2v_cross_attn_forward,
        )

        model = base_model.diffusion_model
        _patch_wan_attention_blocks(model, usp_self_attn_forward, usp_t2v_cross_attn_forward, usp_i2v_cross_attn_forward)
        model.forward_orig = types.MethodType(usp_dit_forward, model)

if hasattr(model_base, "WanMultiTalk"):

    @USPInjectRegistry.register(model_base.WanMultiTalk)
    def _inject_wan_multitalk(model_patcher, base_model, *args):
        from ..diffusion_models.wan.xdit_context_parallel import (
            usp_multitalk_dit_forward,
            usp_self_attn_forward,
            usp_t2v_cross_attn_forward,
        )

        model = base_model.diffusion_model
        _patch_wan_attention_blocks(
            model,
            usp_self_attn_forward,
            usp_t2v_cross_attn_forward,
        )
        model.forward_orig = types.MethodType(usp_multitalk_dit_forward, model)


# Chroma Radiance should be using this since the forward_orig MRO in Chroma Radiance is from Chroma itself
if hasattr(model_base, "ChromaRadiance"):

    @USPInjectRegistry.register(model_base.ChromaRadiance)
    @USPInjectRegistry.register(model_base.Chroma)
    def _inject_chroma(model_patcher, base_model, *args):
        from ..diffusion_models.chroma.xdit_context_parallel import usp_dit_forward, usp_single_stream_forward, usp_double_stream_forward

        model = base_model.diffusion_model
        for block in model.double_blocks:
            block.forward = types.MethodType(usp_double_stream_forward, block)
        for block in model.single_blocks:
            block.forward = types.MethodType(usp_single_stream_forward, block)
        model.forward_orig = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "Flux"):

    @USPInjectRegistry.register(model_base.Flux)
    def _inject_flux(model_patcher, base_model, *args):
        from ..diffusion_models.flux.xdit_context_parallel import usp_dit_forward, usp_single_stream_forward, usp_double_stream_forward

        model = base_model.diffusion_model
        for block in model.double_blocks:
            block.forward = types.MethodType(usp_double_stream_forward, block)
        for block in model.single_blocks:
            block.forward = types.MethodType(usp_single_stream_forward, block)
        model.forward_orig = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "Hunyuan3Dv2"):

    @USPInjectRegistry.register(model_base.Hunyuan3Dv2)
    def _inject_hunyuan_3dv2(model_patcher, base_model, *args):
        from ..diffusion_models.hunyuan3d.xdit_context_parallel import (
            usp_dit_forward,
        )
        from ..diffusion_models.flux.xdit_context_parallel import usp_single_stream_forward, usp_double_stream_forward

        model = base_model.diffusion_model
        for block in model.double_blocks:
            block.forward = types.MethodType(usp_double_stream_forward, block)
        for block in model.single_blocks:
            block.forward = types.MethodType(usp_single_stream_forward, block)
        model._forward = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "HunyuanVideo"):

    @USPInjectRegistry.register(model_base.HunyuanVideo)
    def _inject_hunyuan(model_patcher, base_model, *args):
        from ..diffusion_models.hunyuan_video.xdit_context_parallel import usp_dit_forward, usp_token_refiner_forward
        from ..diffusion_models.flux.xdit_context_parallel import usp_single_stream_forward, usp_double_stream_forward

        model = base_model.diffusion_model
        for block in model.double_blocks:
            block.forward = types.MethodType(usp_double_stream_forward, block)
        for block in model.single_blocks:
            block.forward = types.MethodType(usp_single_stream_forward, block)
        for block_token_refiner in model.txt_in.individual_token_refiner.blocks:
            block_token_refiner.forward = types.MethodType(usp_token_refiner_forward, block_token_refiner)
        model.forward_orig = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "QwenImage"):

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


if hasattr(model_base, "MiniMaxH3"):

    @USPInjectRegistry.register(model_base.MiniMaxH3)
    def _inject_minimax_h3(model_patcher, base_model, *args):
        from ..comfy_dist.sd import FSDP_LORA_SIDECAR_ATTACHMENT
        from ..diffusion_models.minimax.xdit_context_parallel import usp_attn_forward, usp_dit_forward, usp_mlp_forward

        model = base_model.diffusion_model
        sidecar_groups = model_patcher.get_attachment(FSDP_LORA_SIDECAR_ATTACHMENT) or {}
        for i, block in enumerate(model.blocks):
            block.attn.forward = types.MethodType(usp_attn_forward, block.attn)
            if f"diffusion_model.blocks.{i}.mlp.fc2" in sidecar_groups:
                block.mlp.forward = types.MethodType(usp_mlp_forward, block.mlp)
        for i, block in enumerate(model.token_refiner.blocks):
            if f"diffusion_model.token_refiner.blocks.{i}.mlp.fc2" in sidecar_groups:
                block.mlp.forward = types.MethodType(usp_mlp_forward, block.mlp)
        model._forward = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "JoyImage"):

    @USPInjectRegistry.register(model_base.JoyImage)
    def _inject_joyimage(model_patcher, base_model, *args):
        from ..diffusion_models.joyimage.xdit_context_parallel import usp_attention_forward, usp_dit_forward

        model = base_model.diffusion_model
        for block in model.double_blocks:
            block.attn.forward = types.MethodType(usp_attention_forward, block.attn)
        model._forward = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "Lens"):

    @USPInjectRegistry.register(model_base.Lens)
    def _inject_lens(model_patcher, base_model, *args):
        from ..diffusion_models.lens.xdit_context_parallel import usp_lens_forward, usp_lens_attention_forward

        model = base_model.diffusion_model
        for block in model.transformer_blocks:
            block.attn.forward = types.MethodType(usp_lens_attention_forward, block.attn)
        model._forward = types.MethodType(usp_lens_forward, model)


if hasattr(model_base, "PiD"):

    @USPInjectRegistry.register(model_base.PiD)
    def _inject_pid(model_patcher, base_model, *args):
        from ..diffusion_models.pixeldit.xdit_context_parallel import (
            usp_joint_attention_forward,
            usp_pid_forward,
            usp_pit_block_forward,
            usp_rotary_attention_forward,
        )

        model = base_model.diffusion_model
        _patch_pixeldit_blocks(model, usp_joint_attention_forward, usp_rotary_attention_forward, usp_pit_block_forward)
        model._forward = types.MethodType(usp_pid_forward, model)


if hasattr(model_base, "PixelDiTT2I"):

    @USPInjectRegistry.register(model_base.PixelDiTT2I)
    def _inject_pixeldit(model_patcher, base_model, *args):
        from ..diffusion_models.pixeldit.xdit_context_parallel import (
            usp_joint_attention_forward,
            usp_pit_block_forward,
            usp_pixdit_forward,
            usp_rotary_attention_forward,
        )

        model = base_model.diffusion_model
        _patch_pixeldit_blocks(model, usp_joint_attention_forward, usp_rotary_attention_forward, usp_pit_block_forward)
        model._forward = types.MethodType(usp_pixdit_forward, model)


if hasattr(model_base, "CosmosPredict2"):

    @USPInjectRegistry.register(model_base.CosmosPredict2)
    def _inject_cosmos_predict2(model_patcher, base_model, *args):
        from ..diffusion_models.cosmos.xdit_context_parallel import usp_xfuser_attention_op, usp_mini_train_dit_forward

        model = base_model.diffusion_model
        for block in model.blocks:
            block.cross_attn.attn_op = usp_xfuser_attention_op
            block.self_attn.attn_op = usp_xfuser_attention_op
        model._forward = types.MethodType(usp_mini_train_dit_forward, model)


if hasattr(model_base, "Anima"):

    @USPInjectRegistry.register(model_base.Anima)
    def _inject_anima(model_patcher, base_model, *args):
        from ..diffusion_models.cosmos.xdit_context_parallel import usp_xfuser_attention_op, usp_mini_train_dit_forward

        model = base_model.diffusion_model
        for block in model.blocks:
            block.cross_attn.attn_op = usp_xfuser_attention_op
            block.self_attn.attn_op = usp_xfuser_attention_op
        model._forward = types.MethodType(usp_mini_train_dit_forward, model)


if hasattr(model_base, "CosmosVideo"):

    @USPInjectRegistry.register(model_base.CosmosVideo)
    def _inject_cosmos_video(model_patcher, base_model, *args):
        from ..diffusion_models.cosmos.xdit_context_parallel import usp_general_dit_forward, usp_general_attention_forward

        model = base_model.diffusion_model
        for transformer_block in model.blocks.values():
            for block in transformer_block.blocks:
                attn = getattr(getattr(block, "block", None), "attn", None)
                if attn is not None:
                    attn.forward = types.MethodType(usp_general_attention_forward, attn)
        model._forward = types.MethodType(usp_general_dit_forward, model)


if hasattr(model_base, "Lumina2"):

    @USPInjectRegistry.register(model_base.Lumina2)  # Lumina and Z Image
    def _inject_lumina(model_patcher, base_model, *args):
        from ..diffusion_models.lumina.xdit_context_parallel import usp_dit_forward, usp_joint_attention_forward

        model = base_model.diffusion_model
        for block in model.layers:
            block.attention.forward = types.MethodType(usp_joint_attention_forward, block.attention)
        model._forward = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "Boogu"):

    @USPInjectRegistry.register(model_base.Boogu)
    def _inject_boogu(model_patcher, base_model, *args):
        from ..diffusion_models.boogu.xdit_context_parallel import (
            usp_dit_forward,
            usp_double_stream_forward,
            usp_img_self_attention_forward,
            usp_joint_attention_forward,
        )

        model = base_model.diffusion_model
        for block_group_name in ("context_refiner", "noise_refiner", "ref_image_refiner", "single_stream_layers"):
            for block in getattr(model, block_group_name, []):
                if hasattr(block, "attn"):
                    block.attn.forward = types.MethodType(usp_img_self_attention_forward, block.attn)
        for block in model.double_stream_layers:
            block.forward = types.MethodType(usp_double_stream_forward, block)
            block.img_instruct_attn.forward = types.MethodType(usp_joint_attention_forward, block.img_instruct_attn)
            block.img_self_attn.forward = types.MethodType(usp_img_self_attention_forward, block.img_self_attn)
        model.forward = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "Omnigen2"):

    @USPInjectRegistry.register(model_base.Omnigen2)
    def _inject_omnigen2(model_patcher, base_model, *args):
        from ..diffusion_models.omnigen.xdit_context_parallel import usp_attention_forward, usp_dit_forward

        model = base_model.diffusion_model
        if not hasattr(model, "layers"):
            raise ValueError(f"Model: {type(base_model).__name__} needs a model-specific USP handler")
        for block_group_name in ("context_refiner", "noise_refiner", "ref_image_refiner", "layers"):
            for block in getattr(model, block_group_name, []):
                if hasattr(block, "attn"):
                    block.attn.forward = types.MethodType(usp_attention_forward, block.attn)
        model.forward = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "HiDreamO1"):

    @USPInjectRegistry.register(model_base.HiDreamO1)
    def _inject_hidream_o1(model_patcher, base_model, *args):
        from ..diffusion_models.hidream_o1.xdit_context_parallel import usp_dit_forward

        model = base_model.diffusion_model
        model._forward = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "Krea2"):

    @USPInjectRegistry.register(model_base.Krea2)
    def _inject_krea2(model_patcher, base_model, *args):
        from ..diffusion_models.krea2.xdit_context_parallel import usp_attention_forward, usp_dit_forward

        model = base_model.diffusion_model
        for block in model.txtfusion.refiner_blocks:
            block.attn.forward = types.MethodType(usp_attention_forward, block.attn)
        for block in model.blocks:
            block.attn.forward = types.MethodType(usp_attention_forward, block.attn)
        model._forward = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "Kandinsky5"):

    @USPInjectRegistry.register(model_base.Kandinsky5)
    def _inject_kandinsky5(model_patcher, base_model, *args):
        from ..diffusion_models.kandinsky5.xdit_context_parallel import (
            usp_dit_forward,
            usp_self_attn_foward,
            usp_self_attn_forward_chunked,
            usp_cross_attn_forward,
        )

        model = base_model.diffusion_model
        for text_block in model.text_transformer_blocks:
            text_block.self_attention._forward = types.MethodType(usp_self_attn_foward, text_block.self_attention)

            text_block.self_attention._forward_chunked = types.MethodType(usp_self_attn_forward_chunked, text_block.self_attention)

        for visual_block in model.visual_transformer_blocks:
            visual_block.self_attention._forward = types.MethodType(usp_self_attn_foward, visual_block.self_attention)

            visual_block.self_attention._forward_chunked = types.MethodType(usp_self_attn_forward_chunked, visual_block.self_attention)

            visual_block.cross_attention.forward = types.MethodType(usp_cross_attn_forward, visual_block.cross_attention)

        model.forward_orig = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "LTXAV"):

    @USPInjectRegistry.register(model_base.LTXAV)
    def _inject_ltxvav(model_patcher, base_model, *args):
        from ..diffusion_models.lightricks.xdit_context_parallel import usp_dit_forward, usp_cross_attn_forward

        model = base_model.diffusion_model
        for block in model.transformer_blocks:
            block.attn1.forward = types.MethodType(usp_cross_attn_forward, block.attn1)
            block.audio_attn1.forward = types.MethodType(usp_cross_attn_forward, block.audio_attn1)
            block.attn2.forward = types.MethodType(usp_cross_attn_forward, block.attn2)
            block.audio_attn2.forward = types.MethodType(usp_cross_attn_forward, block.audio_attn2)
            block.audio_to_video_attn.forward = types.MethodType(usp_cross_attn_forward, block.audio_to_video_attn)
            block.video_to_audio_attn.forward = types.MethodType(usp_cross_attn_forward, block.video_to_audio_attn)

        model._forward = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "LTXV"):

    @USPInjectRegistry.register(model_base.LTXV)
    def _inject_ltxv(model_patcher, base_model, *args):
        from ..diffusion_models.lightricks.xdit_context_parallel import usp_dit_forward, usp_cross_attn_forward

        model = base_model.diffusion_model
        for block in model.transformer_blocks:
            block.attn1.forward = types.MethodType(usp_cross_attn_forward, block.attn1)
            block.attn2.forward = types.MethodType(usp_cross_attn_forward, block.attn2)

        model._forward = types.MethodType(usp_dit_forward, model)


if hasattr(model_base, "ErnieImage"):

    @USPInjectRegistry.register(model_base.ErnieImage)
    def _inject_ernie_image(model_patcher, base_model, *args):
        from ..diffusion_models.ernie.xdit_context_parallel import usp_attention_forward, usp_dit_forward, usp_forward

        model = base_model.diffusion_model
        for block in model.layers:
            block.self_attention.forward = types.MethodType(usp_attention_forward, block.self_attention)
        model._forward = types.MethodType(usp_dit_forward, model)
        model.forward = types.MethodType(usp_forward, model)


if hasattr(model_base, "LingBotVideo"):

    @USPInjectRegistry.register(model_base.LingBotVideo)
    def _inject_lingbot_video(model_patcher, base_model, *args):
        from ..diffusion_models.lingbot_video.xdit_context_parallel import usp_attention_forward, usp_dit_forward

        model = base_model.diffusion_model
        for block in model.blocks:
            block.attn.forward = types.MethodType(usp_attention_forward, block.attn)
        model.forward = types.MethodType(usp_dit_forward, model)
