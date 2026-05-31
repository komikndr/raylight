from __future__ import annotations

from dataclasses import dataclass

import raylight.distributed_modules.attention as xfuser_attn

from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    get_pp_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from xfuser.core.distributed.utils import RankGenerator


def _normalized_degree(value: int | None) -> int:
    if value is None:
        return 1
    value = int(value)
    return 1 if value <= 0 else value


@dataclass(frozen=True)
class XFuserParallelConfig:
    ulysses_degree: int = 1
    ring_degree: int = 1
    cfg_degree: int = 1
    pp_degree: int = 1
    data_parallel_degree: int = 1

    @property
    def sequence_parallel_degree(self) -> int:
        return self.ulysses_degree * self.ring_degree

    @property
    def dit_parallel_size(self) -> int:
        return self.sequence_parallel_degree * self.cfg_degree * self.pp_degree * self.data_parallel_degree

    @property
    def model_parallel_size(self) -> int:
        return self.sequence_parallel_degree * self.cfg_degree * self.pp_degree

    @classmethod
    def from_parallel_dict(cls, parallel_dict: dict) -> "XFuserParallelConfig":
        return cls(
            ulysses_degree=_normalized_degree(parallel_dict.get("ulysses_degree")),
            ring_degree=_normalized_degree(parallel_dict.get("ring_degree")),
            cfg_degree=_normalized_degree(parallel_dict.get("cfg_degree")),
            pp_degree=_normalized_degree(parallel_dict.get("pp_degree")),
        )


@dataclass(frozen=True)
class XFuserParallelContext:
    config: XFuserParallelConfig
    rank_generator: RankGenerator
    global_rank: int
    global_world_size: int
    data_parallel_rank: int
    data_parallel_world_size: int
    cfg_rank: int
    cfg_world_size: int
    pipeline_rank: int
    pipeline_world_size: int
    sequence_rank: int
    sequence_world_size: int

    def get_rank_generator(self) -> RankGenerator:
        return self.rank_generator

    def pp_group(self):
        return get_pp_group()

    def sp_group(self):
        return get_sp_group()

    def world_group(self):
        return get_world_group()


def requires_xfuser_parallel(parallel_dict: dict) -> bool:
    return bool(parallel_dict.get("is_xdit") or parallel_dict.get("pipefusion_enabled"))


def initialize_xfuser_parallel(local_rank: int, world_size: int, parallel_dict: dict) -> XFuserParallelContext:
    base_config = XFuserParallelConfig.from_parallel_dict(parallel_dict)
    if world_size % base_config.model_parallel_size != 0:
        raise ValueError(
            "Ray worker count must be divisible by "
            "pp_degree * ulysses_degree * ring_degree * cfg_degree: "
            f"{world_size} is not divisible by {base_config.pp_degree} * {base_config.ulysses_degree} * {base_config.ring_degree} * {base_config.cfg_degree}"
        )
    derived_data_parallel_degree = world_size // base_config.model_parallel_size
    requested_data_parallel_degree = int(parallel_dict.get("dp_degree", 0) or 0)
    if requested_data_parallel_degree not in (0, derived_data_parallel_degree):
        raise ValueError(
            "Ray worker count must equal "
            "dp_degree * pp_degree * ulysses_degree * ring_degree * cfg_degree: "
            f"{world_size} != {requested_data_parallel_degree} * {base_config.pp_degree} * {base_config.ulysses_degree} * {base_config.ring_degree} * {base_config.cfg_degree}"
        )
    config = XFuserParallelConfig(
        ulysses_degree=base_config.ulysses_degree,
        ring_degree=base_config.ring_degree,
        cfg_degree=base_config.cfg_degree,
        pp_degree=base_config.pp_degree,
        data_parallel_degree=derived_data_parallel_degree,
    )

    if parallel_dict.get("is_xdit"):
        xfuser_attn.set_attn_type(parallel_dict["attention"])
        xfuser_attn.set_sync_ulysses(parallel_dict["sync_ulysses"])

    init_distributed_environment(rank=local_rank, world_size=world_size)
    initialize_model_parallel(
        data_parallel_degree=config.data_parallel_degree,
        sequence_parallel_degree=config.sequence_parallel_degree,
        classifier_free_guidance_degree=config.cfg_degree,
        ring_degree=config.ring_degree,
        ulysses_degree=config.ulysses_degree,
        pipeline_parallel_degree=config.pp_degree,
    )

    rank_generator = RankGenerator(
        1,
        config.sequence_parallel_degree,
        config.pp_degree,
        config.cfg_degree,
        config.data_parallel_degree,
        "tp-sp-pp-cfg-dp",
    )
    return XFuserParallelContext(
        config=config,
        rank_generator=rank_generator,
        global_rank=local_rank,
        global_world_size=world_size,
        data_parallel_rank=get_data_parallel_rank(),
        data_parallel_world_size=get_data_parallel_world_size(),
        cfg_rank=get_classifier_free_guidance_rank(),
        cfg_world_size=get_classifier_free_guidance_world_size(),
        pipeline_rank=get_pipeline_parallel_rank(),
        pipeline_world_size=get_pipeline_parallel_world_size(),
        sequence_rank=get_sequence_parallel_rank(),
        sequence_world_size=get_sequence_parallel_world_size(),
    )
