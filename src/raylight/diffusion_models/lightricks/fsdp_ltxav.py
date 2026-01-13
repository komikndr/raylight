import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from raylight.distributed_modules.utils import detect_dtype_mismatch
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions


SAFE_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


def has_quantized_or_unsafe_params(module):
    for p in module.parameters(recurse=True):
        dt = p.dtype

        if not dt.is_floating_point:
            return True

        if dt not in SAFE_FLOAT_DTYPES:
            return True

    return False


def all_bf16(module):
    for p in module.parameters(recurse=True):
        if p.dtype != torch.bfloat16:
            return False
    return True


def auto_wrap_non_transformer(diffusion_model):
    """
    Fully shard everything EXCEPT transformer_blocks
    """
    for name, child in diffusion_model.named_children():
        if name == "transformer_blocks":
            continue

        if any(p.requires_grad for p in child.parameters()):
            diffusion_model._modules[name] = fully_shard(
                module=child,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True,
            )


def shard_transformer_blocks(diffusion_model):
    """
    For each block:
    - If full BF16 -> fully_shard
    - If quant/mixed -> fully_shard with ignored_params
    """
    blocks = diffusion_model.transformer_blocks

    ref_dtype = blocks[1].video_to_audio_attn.to_q.weight.dtype

    for i, block in enumerate(blocks):
        is_bf16 = all_bf16(block)
        is_quant = has_quantized_or_unsafe_params(block)

        if is_bf16 and not is_quant:
            blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True,
            )

        else:
            ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
            blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True,
                ignored_params=ignored_block_params,
            )


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model
    auto_wrap_non_transformer(diffusion_model)
    shard_transformer_blocks(diffusion_model)

    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if not name.startswith("transformer_blocks."):
            ignored_params.add(param)

    fully_shard(
        diffusion_model,
        ignored_params=ignored_params,
        reshard_after_forward=True,
    )

    model.diffusion_model = diffusion_model
    set_model_state_dict(
        model=model,
        model_state_dict=model_state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=enable_cpu_offload,
        ),
    )
    return model
