from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from raylight.distributed_modules.utils import detect_dtype_mismatch


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model

    # Collect params we want to ignore (everything except single_blocks + double_blocks)
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if not name.startswith("transformer_blocks."):
            ignored_params.add(param)

    # Check dtype missmatch from scaled model, early or end blocks usually in bf16,
    # But the middle majority part usually in low-prec, hence 24th block
    ref_dtype = diffusion_model.transformer_blocks[24].attn1.to_k.weight.dtype

    for i, block in enumerate(diffusion_model.transformer_blocks):
        ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
        diffusion_model.transformer_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            ignored_params=ignored_block_params,
        )

    fully_shard(diffusion_model,
                ignored_params=ignored_params,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True)

    model.diffusion_model = diffusion_model
    set_model_state_dict(
        model=model,
        model_state_dict=model_state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=enable_cpu_offload
        ),
    )
    return model
