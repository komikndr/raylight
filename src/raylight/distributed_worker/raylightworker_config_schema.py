from dataclasses import dataclass, field
# Better than dict that for sure


@dataclass(frozen=True)
class MiniMaxOptions:
    attention_fp16: bool = False
    mlp_fp16: bool = False
    fp32_residual: bool = False
    mlp_chunk_tokens: int = 0

    @property
    def enabled(self):
        return self.attention_fp16 or self.mlp_fp16 or self.fp32_residual or self.mlp_chunk_tokens > 0


@dataclass(frozen=True)
class ModelOptions:
    minimax: MiniMaxOptions = field(default_factory=MiniMaxOptions)


@dataclass(frozen=True)
class XFuserParallelConfig:
    enabled: bool = False
    ulysses_degree: int = 1
    ring_degree: int = 1
    cfg_degree: int = 1
    data_parallel_degree: int = 1

    @property
    def sequence_parallel_degree(self):
        return self.ulysses_degree * self.ring_degree

    @property
    def model_parallel_size(self):
        return self.sequence_parallel_degree * self.cfg_degree

    @property
    def dit_parallel_size(self):
        return self.model_parallel_size * self.data_parallel_degree


@dataclass(frozen=True)
class LoadingOptions:
    parallel_quant_load: bool = False


@dataclass(frozen=True)
class RayWorkerConfig:
    parallel: XFuserParallelConfig = field(default_factory=XFuserParallelConfig)
    model_options: ModelOptions = field(default_factory=ModelOptions)
    attention_backend: str = "TORCH_FLASH"
    sync_ulysses: bool = False
    fsdp_enabled: bool = False
    fsdp_cpu_offload: bool = False
    use_mmap: bool = True
    pinned_memory: bool = False
    vram_headroom: float | None = None
    reserve_vram: float | None = None
    collective_timeout_minutes: float = 1.0
    clear_vram_after_sampling: bool = False
    selected_gpus: tuple[int, ...] | None = None

    def as_parallel_dict(self, world_size, group_id=0, is_quant=False):
        grouped = self.fsdp_enabled and self.parallel.data_parallel_degree > 1
        return {
            "is_xdit": self.parallel.enabled,
            "is_fsdp": self.fsdp_enabled,
            "is_quant": is_quant,
            "sync_ulysses": self.sync_ulysses,
            "global_world_size": world_size,
            "shard_size": world_size // self.parallel.data_parallel_degree if self.fsdp_enabled else world_size,
            "group_id": group_id,
            "use_group_process_group": grouped,
            "use_mmap": self.use_mmap,
            "dp_degree": self.parallel.data_parallel_degree,
            "ulysses_degree": self.parallel.ulysses_degree,
            "ring_degree": self.parallel.ring_degree,
            "cfg_degree": self.parallel.cfg_degree,
            "attention": self.attention_backend,
            "fsdp_cpu_offload": self.fsdp_cpu_offload,
            "clear_vram_after_sampling": self.clear_vram_after_sampling,
            "selected_gpus": self.selected_gpus,
        }


@dataclass(frozen=True)
class RaylightConfig:
    loading: LoadingOptions = field(default_factory=LoadingOptions)
    worker: RayWorkerConfig = field(default_factory=RayWorkerConfig)
