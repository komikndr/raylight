# No longer needed, only for backup for now, use newer general fsdp_utils inside comfy_dist.

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from raylight.distributed_modules.utils import detect_dtype_mismatch, ensure_no_scalar
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model
    # Shard only the blocks, since other modules have different dtype
    # Collect params we want to ignore (everything except blocks)
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if not name.startswith("blocks."):
            ignored_params.add(param)

    ref_dtype = diffusion_model.blocks[0].self_attn.v.weight.dtype
    for i, block in enumerate(diffusion_model.blocks):
        # This is for scaled model
        ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
        diffusion_model.blocks[i] = fully_shard(
            module=block, mp_policy=MixedPrecisionPolicy(), reshard_after_forward=True, ignored_params=ignored_block_params
        )

    fully_shard(diffusion_model, ignored_params=ignored_params, reshard_after_forward=True)
    model.diffusion_model = diffusion_model

    # CPU OFFLOAD ONLY FOR LOW END OF THE LOWEND
    set_model_state_dict(
        model=model,
        model_state_dict=model_state_dict,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=enable_cpu_offload),
    )

    return model
