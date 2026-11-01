from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import timedelta
from functools import wraps

import torch.distributed as dist
from .raylightworker_config_schema import RayWorkerConfig, XFuserParallelConfig

import raylight.distributed_modules.attention as xfuser_attn

from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)


@dataclass(frozen=True)
class XFuserParallelContext:
    config: XFuserParallelConfig
    global_rank: int
    global_world_size: int
    data_parallel_rank: int
    data_parallel_world_size: int
    cfg_rank: int
    cfg_world_size: int
    sequence_rank: int
    sequence_world_size: int

    def sp_group(self):
        return get_sp_group()

    def world_group(self):
        return get_world_group()


@contextmanager
def collective_group_timeout(timeout):
    # xFuser/yunchang create their own NCCL and Gloo groups without a timeout parameter.
    original = dist.new_group

    @wraps(original)
    def new_group(*args, **kwargs):
        if len(args) < 2:
            kwargs["timeout"] = timeout
        else:
            args = (args[0], timeout, *args[2:])
        return original(*args, **kwargs)

    dist.new_group = new_group
    try:
        yield
    finally:
        dist.new_group = original


def initialize_xfuser_parallel(local_rank: int, world_size: int, config: RayWorkerConfig) -> XFuserParallelContext:
    base_config = config.parallel
    if world_size % base_config.model_parallel_size != 0:
        raise ValueError(
            "Ray worker count must be divisible by "
            "ulysses_degree * ring_degree * cfg_degree: "
            f"{world_size} is not divisible by {base_config.ulysses_degree} * {base_config.ring_degree} * {base_config.cfg_degree}"
        )
    derived_data_parallel_degree = world_size // base_config.model_parallel_size
    parallel = XFuserParallelConfig(
        enabled=base_config.enabled,
        ulysses_degree=base_config.ulysses_degree,
        ring_degree=base_config.ring_degree,
        cfg_degree=base_config.cfg_degree,
        data_parallel_degree=derived_data_parallel_degree,
    )

    xfuser_attn.set_attn_type(config.attention_backend)
    xfuser_attn.set_sync_ulysses(config.sync_ulysses)

    with collective_group_timeout(timedelta(minutes=config.collective_timeout_minutes)):
        init_distributed_environment(rank=local_rank, world_size=world_size)
        initialize_model_parallel(
            data_parallel_degree=parallel.data_parallel_degree,
            sequence_parallel_degree=parallel.sequence_parallel_degree,
            classifier_free_guidance_degree=parallel.cfg_degree,
            ring_degree=parallel.ring_degree,
            ulysses_degree=parallel.ulysses_degree,
            pipeline_parallel_degree=1,
        )

    return XFuserParallelContext(
        config=parallel,
        global_rank=local_rank,
        global_world_size=world_size,
        data_parallel_rank=get_data_parallel_rank(),
        data_parallel_world_size=get_data_parallel_world_size(),
        cfg_rank=get_classifier_free_guidance_rank(),
        cfg_world_size=get_classifier_free_guidance_world_size(),
        sequence_rank=get_sequence_parallel_rank(),
        sequence_world_size=get_sequence_parallel_world_size(),
    )
