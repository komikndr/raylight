from __future__ import annotations

from dataclasses import dataclass

from raylight.distributed_worker.pipefusion_schema import (
    PipeFusionConfig,
    StagePlan,
)
from raylight.distributed_worker.parallel_group_manager import XFuserParallelContext


PIPEFUSION_RUNTIME_ATTACHMENT = "pipefusion_runtime"
PIPEFUSION_SESSION_KEY = "pipefusion_session"
PIPEFUSION_CONTEXT_KEY = "pipefusion_context"
PIPEFUSION_WRAPPER_KEY = "pipefusion"


@dataclass(frozen=True)
class PipeFusionRuntime:
    config: PipeFusionConfig
    stage: StagePlan
    model_name: str
    parallel: XFuserParallelContext

    @property
    def debug(self) -> bool:
        return self.config.debug


@dataclass(frozen=True)
class PipeFusionForwardContext:
    runtime: PipeFusionRuntime
    step_index: int
    total_steps: int
    forward_id: int
    step_forward_index: int
    mode: str

    @property
    def stage(self) -> StagePlan:
        return self.runtime.stage

    @property
    def debug(self) -> bool:
        return self.runtime.debug

    @property
    def num_pipeline_patch(self) -> int:
        return self.runtime.config.num_pipeline_patch

    def is_warmup(self) -> bool:
        return self.mode == "warmup"

    def trace(self, message: str) -> None:
        if not self.debug:
            return
        print(
            "[PipeFusion] "
            f"global_rank={self.runtime.parallel.global_rank} "
            f"pp_rank={self.stage.rank} "
            f"step={self.step_index} "
            f"forward={self.forward_id} "
            f"mode={self.mode} "
            f"{message}"
        )


class PipeFusionSession:
    def __init__(self, runtime: PipeFusionRuntime):
        self.runtime = runtime
        self.total_steps = 1
        self.step_index = -1
        self.forward_id = 0
        self.step_forward_index = 0

    def prepare(self, sigmas) -> "PipeFusionSession":
        self.total_steps = max(len(sigmas) - 1, 1) if hasattr(sigmas, "__len__") else 1
        self.step_index = -1
        self.forward_id = 0
        self.step_forward_index = 0
        return self

    def begin_step(self) -> None:
        self.step_index += 1
        self.step_forward_index = 0

    def begin_forward(self) -> PipeFusionForwardContext:
        mode = "warmup" if self.step_index < self.runtime.config.warmup_steps else "pipeline"
        context = PipeFusionForwardContext(
            runtime=self.runtime,
            step_index=self.step_index,
            total_steps=self.total_steps,
            forward_id=self.forward_id,
            step_forward_index=self.step_forward_index,
            mode=mode,
        )
        self.forward_id += 1
        self.step_forward_index += 1
        return context
