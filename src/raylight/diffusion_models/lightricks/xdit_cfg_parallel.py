import torch
from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


def _chunk_tensor(value, cfg_world_size, cfg_rank):
    if value is None:
        return None
    return torch.chunk(value, cfg_world_size, dim=0)[cfg_rank]


def _chunk_x_branch(x, cfg_world_size, cfg_rank):
    if isinstance(x, list):
        x = list(x)
        for i, branch in enumerate(x):
            if branch is not None:
                x[i] = _chunk_tensor(branch, cfg_world_size, cfg_rank)
        return x
    if isinstance(x, tuple):
        return tuple(_chunk_tensor(branch, cfg_world_size, cfg_rank) if branch is not None else None for branch in x)
    return _chunk_tensor(x, cfg_world_size, cfg_rank)


def _chunk_timestep(timestep, cfg_world_size, cfg_rank):
    if isinstance(timestep, list):
        return [_chunk_tensor(item, cfg_world_size, cfg_rank) if item is not None else None for item in timestep]
    if isinstance(timestep, tuple):
        return tuple(_chunk_tensor(item, cfg_world_size, cfg_rank) if item is not None else None for item in timestep)
    return _chunk_tensor(timestep, cfg_world_size, cfg_rank)


def _validate_cfg_batch(x, cfg_world_size):
    if isinstance(x, (list, tuple)):
        batch_source = next((item for item in x if item is not None), None)
    else:
        batch_source = x

    if batch_source is None or batch_source.shape[0] != cfg_world_size:
        raise ValueError("CFG = 1.0, disables guidance. Increase CFG > 1.0 or switch to another parallelism mode")


def _finalize_result(result):
    return get_cfg_group().all_gather(result, dim=0)


def cfg_parallel_forward_wrapper_ltx(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    x, timestep, context, attention_mask, frame_rate, transformer_options, keyframe_idxs = args
    _validate_cfg_batch(x, cfg_world_size)

    x = _chunk_x_branch(x, cfg_world_size, cfg_rank)
    timestep = _chunk_timestep(timestep, cfg_world_size, cfg_rank)
    context = _chunk_tensor(context, cfg_world_size, cfg_rank)

    if attention_mask is not None:
        attention_mask = _chunk_tensor(attention_mask, cfg_world_size, cfg_rank)

    if keyframe_idxs is not None:
        keyframe_idxs = _chunk_tensor(keyframe_idxs, cfg_world_size, cfg_rank)

    if kwargs.get("denoise_mask") is not None:
        kwargs = dict(kwargs)
        kwargs["denoise_mask"] = _chunk_tensor(kwargs["denoise_mask"], cfg_world_size, cfg_rank)

    result = executor(
        x,
        timestep,
        context,
        attention_mask,
        frame_rate,
        transformer_options,
        keyframe_idxs,
        **kwargs,
    )
    return _finalize_result(result)


def cfg_parallel_forward_wrapper_ltxav(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    x, timestep, context, attention_mask, frame_rate, transformer_options, keyframe_idxs = args
    _validate_cfg_batch(x, cfg_world_size)

    x = _chunk_x_branch(x, cfg_world_size, cfg_rank)
    timestep = _chunk_timestep(timestep, cfg_world_size, cfg_rank)
    context = _chunk_tensor(context, cfg_world_size, cfg_rank)

    if attention_mask is not None:
        attention_mask = _chunk_tensor(attention_mask, cfg_world_size, cfg_rank)

    if keyframe_idxs is not None:
        keyframe_idxs = _chunk_tensor(keyframe_idxs, cfg_world_size, cfg_rank)

    chunkable_kwargs = ("denoise_mask", "audio_denoise_mask", "a_timestep")
    if any(kwargs.get(key) is not None for key in chunkable_kwargs):
        kwargs = dict(kwargs)
        for key in chunkable_kwargs:
            if kwargs.get(key) is not None:
                kwargs[key] = _chunk_tensor(kwargs[key], cfg_world_size, cfg_rank)

    result = executor(
        x,
        timestep,
        context,
        attention_mask,
        frame_rate,
        transformer_options,
        keyframe_idxs,
        **kwargs,
    )
    return _finalize_result(result)


cfg_parallel_forward_wrapper = cfg_parallel_forward_wrapper_ltx
