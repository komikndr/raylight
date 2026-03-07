import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def _chunk_tensor(value, cfg_world_size, cfg_rank):
    return torch.chunk(value, cfg_world_size, dim=0)[cfg_rank]


def _chunk_tensor_kwargs(kwargs, cfg_world_size, cfg_rank):
    chunked = dict(kwargs)
    for key, value in list(chunked.items()):
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == cfg_world_size:
            chunked[key] = _chunk_tensor(value, cfg_world_size, cfg_rank)
    return chunked


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    x, timestep, context, clip_fea, time_dim_concat, transformer_options = args

    if x.shape[0] == cfg_world_size:
        x = _chunk_tensor(x, cfg_world_size, cfg_rank)
    else:
        raise ValueError("CFG = 1.0, disables guidance. Increase CFG > 1.0 or switch to another parallelism mode")
    timestep = _chunk_tensor(timestep, cfg_world_size, cfg_rank)
    context = _chunk_tensor(context, cfg_world_size, cfg_rank)

    if clip_fea is not None:
        clip_fea = _chunk_tensor(clip_fea, cfg_world_size, cfg_rank)

    if time_dim_concat is not None:
        time_dim_concat = _chunk_tensor(time_dim_concat, cfg_world_size, cfg_rank)

    kwargs = _chunk_tensor_kwargs(kwargs, cfg_world_size, cfg_rank)

    result = executor(x, timestep, context, clip_fea, time_dim_concat, transformer_options, **kwargs)
    result = get_cfg_group().all_gather(result, dim=0)
    return result
