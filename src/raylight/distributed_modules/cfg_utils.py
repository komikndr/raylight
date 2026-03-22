from __future__ import annotations

import inspect

import torch
from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


_DEFAULT_SKIP_EXTRA_KWARGS = frozenset(
    {
        "transformer_options",
        "control",
        "disable_time_r",
        "data_type",
    }
)


def _find_batch_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _find_batch_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _has_cfg_batch(value, cfg_world_size):
    tensor = _find_batch_tensor(value)
    return tensor is not None and tensor.ndim > 0 and tensor.shape[0] == cfg_world_size


def _chunk_tensor(value, cfg_world_size, cfg_rank):
    if value.ndim == 0:
        return value
    if value.shape[0] == cfg_world_size:
        return torch.chunk(value, cfg_world_size, dim=0)[cfg_rank]
    if value.shape[0] == 1:
        return value
    return value


def _chunk_value(value, cfg_world_size, cfg_rank):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _chunk_tensor(value, cfg_world_size, cfg_rank)
    if isinstance(value, list):
        return [_chunk_value(item, cfg_world_size, cfg_rank) for item in value]
    if isinstance(value, tuple):
        return tuple(_chunk_value(item, cfg_world_size, cfg_rank) for item in value)
    return value


def _resolve_argument_name(bound_arguments, name):
    if isinstance(name, str):
        return name if name in bound_arguments else None
    for candidate in name:
        if candidate in bound_arguments:
            return candidate
    return None


def _validate_cfg_batch(value, cfg_world_size):
    tensor = _find_batch_tensor(value)
    if tensor is None or tensor.ndim == 0 or tensor.shape[0] != cfg_world_size:
        raise ValueError("CFG = 1.0, disables guidance. Increase CFG > 1.0 or switch to another parallelism mode")


def _chunk_extra_kwargs(extra_kwargs, cfg_world_size, cfg_rank, skip_names):
    chunked = dict(extra_kwargs)
    for key, value in extra_kwargs.items():
        if key in skip_names:
            continue
        if _has_cfg_batch(value, cfg_world_size):
            chunked[key] = _chunk_value(value, cfg_world_size, cfg_rank)
    return chunked


def cfg_parallel_forward(
    executor,
    *args,
    chunk_names=(),
    validate_name="x",
    auto_chunk_extra_kwargs=False,
    skip_extra_kwargs=(),
    **kwargs,
):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    signature = inspect.signature(executor.original)
    bound = signature.bind_partial(*args, **kwargs)

    validate_arg_name = _resolve_argument_name(bound.arguments, validate_name)
    if validate_arg_name is None:
        raise RuntimeError(f"CFG wrapper could not resolve validation argument {validate_name!r}")
    _validate_cfg_batch(bound.arguments[validate_arg_name], cfg_world_size)

    for name in chunk_names:
        arg_name = _resolve_argument_name(bound.arguments, name)
        if arg_name is None:
            continue
        bound.arguments[arg_name] = _chunk_value(bound.arguments[arg_name], cfg_world_size, cfg_rank)

    if auto_chunk_extra_kwargs:
        var_keyword_name = next(
            (param.name for param in signature.parameters.values() if param.kind == inspect.Parameter.VAR_KEYWORD),
            None,
        )
        if var_keyword_name is not None and var_keyword_name in bound.arguments:
            skip_names = _DEFAULT_SKIP_EXTRA_KWARGS.union(skip_extra_kwargs)
            bound.arguments[var_keyword_name] = _chunk_extra_kwargs(
                bound.arguments[var_keyword_name],
                cfg_world_size,
                cfg_rank,
                skip_names,
            )

    result = executor(*bound.args, **bound.kwargs)
    return get_cfg_group().all_gather(result, dim=0)
