import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    x, timestep, context, attention_mask, frame_rate, transformer_options, keyframe_idxs = args

    if x[0].shape[0] == cfg_world_size:
        x[0] = torch.chunk(x[0], cfg_world_size, dim=0)[cfg_rank]
    else:
        raise ValueError("CFG = 1.0, disables guidance. Increase CFG > 1.0 or switch to another parallelism mode")
    timestep = torch.chunk(timestep, cfg_world_size, dim=0)[cfg_rank]
    context = torch.chunk(context, cfg_world_size, dim=0)[cfg_rank]

    if attention_mask is not None:
        attention_mask = torch.chunk(attention_mask, cfg_world_size, dim=0)[cfg_rank]

    if keyframe_idxs is not None:
        keyframe_idxs = torch.chunk(keyframe_idxs, cfg_world_size, dim=0)[cfg_rank]

    result = executor(x,
                      timestep,
                      context,
                      attention_mask,
                      frame_rate,
                      transformer_options,
                      keyframe_idxs,
                      **kwargs)

    result = get_cfg_group().all_gather(result, dim=0)
    return result
