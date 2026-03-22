from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipeFusionConfig:
    enabled: bool = False
    pp_degree: int = 1
    num_pipeline_patch: int = 1
    warmup_steps: int = 0
    stage_splits: tuple[int, ...] | None = None
    debug: bool = False

    @classmethod
    def from_parallel_dict(cls, parallel_dict: dict) -> "PipeFusionConfig":
        stage_splits = parallel_dict.get("pipefusion_stage_splits")
        if stage_splits:
            stage_splits = tuple(int(x) for x in stage_splits)
        else:
            stage_splits = None

        return cls(
            enabled=bool(parallel_dict.get("pipefusion_enabled", False)),
            pp_degree=int(parallel_dict.get("pp_degree", 1)),
            num_pipeline_patch=int(parallel_dict.get("num_pipeline_patch", 1)),
            warmup_steps=int(parallel_dict.get("warmup_steps", 0)),
            stage_splits=stage_splits,
            debug=bool(parallel_dict.get("pipefusion_debug", False)),
        )


@dataclass(frozen=True)
class StagePlan:
    rank: int
    world_size: int
    group_ranks: tuple[int, ...]
    total_blocks: int
    stage_start: int
    stage_end: int
    is_first: bool
    is_last: bool
    num_pipeline_patch: int

    @property
    def local_block_count(self) -> int:
        return self.stage_end - self.stage_start


def _validate_stage_splits(total_blocks: int, world_size: int, config: PipeFusionConfig) -> tuple[int, ...]:
    if config.stage_splits is None:
        raise ValueError("stage_splits is not set")
    if len(config.stage_splits) != world_size:
        raise ValueError(f"PipeFusion stage_splits length must match pp_degree: {len(config.stage_splits)} != {world_size}")
    if any(split <= 0 for split in config.stage_splits):
        raise ValueError("PipeFusion stage_splits must all be positive")
    if sum(config.stage_splits) != total_blocks:
        raise ValueError(f"PipeFusion stage_splits must sum to the number of Wan blocks: {sum(config.stage_splits)} != {total_blocks}")
    return config.stage_splits


def build_stage_plan(
    total_blocks: int,
    rank: int,
    world_size: int,
    config: PipeFusionConfig,
    group_ranks: tuple[int, ...] | None = None,
) -> StagePlan:
    if not config.enabled:
        raise ValueError("PipeFusion is disabled")
    if total_blocks <= 0:
        raise ValueError("PipeFusion requires a model with at least one block")
    if world_size <= 0:
        raise ValueError("PipeFusion pp_degree must be at least 1")
    if config.pp_degree != world_size:
        raise ValueError(f"PipeFusion pp_degree does not match the initialized xFuser PP world size: {config.pp_degree} != {world_size}")
    if world_size > total_blocks:
        raise ValueError(f"PipeFusion pp_degree cannot exceed the number of Wan blocks: {world_size} > {total_blocks}")
    if config.num_pipeline_patch <= 0:
        raise ValueError("PipeFusion num_pipeline_patch must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"PipeFusion rank out of range: {rank}")
    if group_ranks is None:
        group_ranks = tuple(range(world_size))
    elif len(group_ranks) != world_size:
        raise ValueError(f"PipeFusion PP group rank count must match pp_degree: {len(group_ranks)} != {world_size}")

    if config.stage_splits is not None:
        stage_splits = _validate_stage_splits(total_blocks, world_size, config)
        stage_start = sum(stage_splits[:rank])
        stage_end = sum(stage_splits[: rank + 1])
    else:
        base_blocks = total_blocks // world_size
        remainder = total_blocks % world_size
        stage_start = rank * base_blocks + min(rank, remainder)
        stage_end = stage_start + base_blocks + (1 if rank < remainder else 0)

    if stage_start >= stage_end:
        raise ValueError("PipeFusion produced an empty stage; adjust pp_degree or stage_splits")

    return StagePlan(
        rank=rank,
        world_size=world_size,
        group_ranks=tuple(group_ranks),
        total_blocks=total_blocks,
        stage_start=stage_start,
        stage_end=stage_end,
        is_first=rank == 0,
        is_last=rank == (world_size - 1),
        num_pipeline_patch=config.num_pipeline_patch,
    )
