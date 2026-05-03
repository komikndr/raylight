import raylight
import os
import gc
import shutil
import tempfile
from typing import Any
from pathlib import Path
from copy import deepcopy

import ray
import torch
import comfy
import folder_paths
from yunchang.kernels import AttnType

# Must manually insert comfy package or ray cannot import raylight to cluster
from comfy import sd, sample, utils  # type: ignore

from .distributed_worker.ray_worker import (
    make_ray_actor_fn,
    ensure_fresh_actors,
    ray_nccl_tester,
)


def _raylight_ray_tmpdir() -> Path:
    return Path(os.environ.get("RAYLIGHT_RAY_TMPDIR", Path(tempfile.gettempdir()) / "raylight-ray")).resolve()


def _configure_raylight_ray_tmpdir(runtime_env: dict[str, Any]):
    ray_tmpdir = _raylight_ray_tmpdir()
    ray_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RAY_TMPDIR", str(ray_tmpdir))
    runtime_env.setdefault("env_vars", {}).setdefault("RAY_TMPDIR", str(ray_tmpdir))


def _cleanup_ray_temp():
    """Remove stale Ray files only from Raylight's owned temp directory."""
    ray_tmpdir = _raylight_ray_tmpdir()
    ray_dir = ray_tmpdir / "ray"
    if not ray_dir.is_dir():
        return
    try:
        shutil.rmtree(ray_dir, ignore_errors=True)
    except Exception:
        pass


# Workaround https://github.com/comfyanonymous/ComfyUI/pull/11134
# since in FSDPModelPatcher mode, ray cannot pickle None type cause by getattr
def _monkey():
    from raylight.comfy_dist import patch_base_getattr

    patch_base_getattr()


def _resolve_module_dir(module):
    module_file = getattr(module, "__file__", None)
    if module_file:
        path = Path(module_file).resolve()
        if path.is_file():
            return path.parent

    module_paths = getattr(module, "__path__", None)
    if module_paths:
        for path in module_paths:
            if path:
                resolved = Path(path).resolve()
                if resolved.exists():
                    return resolved

    raise RuntimeError(f"Unable to determine module path for {getattr(module, '__name__', module)}")


def _resolve_repo_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "main.py").exists() and (parent / "execution.py").exists():
            return parent
    raise RuntimeError("Unable to locate ComfyUI repository root")


def _ensure_runtime_workdir(module_dir: Path) -> Path:
    runtime_dir = module_dir.parent / "_ray_runtime_env"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _build_local_runtime_env(module_dir: Path, repo_root: Path, runtime_workdir: Path):
    python_path_entries = [str(repo_root)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_path_entries.extend(part for part in existing.split(os.pathsep) if part)
    python_path = os.pathsep.join(dict.fromkeys(python_path_entries))

    env_vars = {
        "PYTHONPATH": python_path,
        "COMFYUI_BASE_DIRECTORY": str(repo_root),
    }

    return {
        "py_modules": [str(module_dir)],
        "working_dir": str(runtime_workdir),
        "env_vars": env_vars,
    }


def _parse_gpu_select(gpu_select: str | None) -> tuple[int, ...] | None:
    if gpu_select is None:
        return None
    gpu_select = gpu_select.strip()
    if not gpu_select:
        return None

    selected: list[int] = []
    seen: set[int] = set()
    for raw in gpu_select.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            gpu_idx = int(token)
        except ValueError as exc:
            raise ValueError(f"GPU_SELECT contains non-integer entry: {token!r}") from exc
        if gpu_idx < 0:
            raise ValueError(f"GPU_SELECT only supports zero-based GPU indices, got {gpu_idx}")
        if gpu_idx in seen:
            raise ValueError(f"GPU_SELECT contains duplicate GPU index: {gpu_idx}")
        seen.add(gpu_idx)
        selected.append(gpu_idx)

    if not selected:
        return None
    return tuple(selected)


def _build_remote_runtime_env(module_dir: Path, repo_root: Path):
    excludes = [
        ".git",
        ".git/**",
        "__pycache__",
        "**/__pycache__",
        "*.pyc",
    ]

    return {
        "py_modules": [str(module_dir)],
        "working_dir": str(repo_root),
        "env_vars": {
            "COMFYUI_BASE_DIRECTORY": ".",
        },
        "excludes": excludes,
    }


_RAYLIGHT_MODULE_PATH = _resolve_module_dir(raylight)
_COMFY_ROOT_PATH = _resolve_repo_root()
_RAYLIGHT_RUNTIME_WORKDIR = _ensure_runtime_workdir(_RAYLIGHT_MODULE_PATH)
_RAY_RUNTIME_ENV_LOCAL = _build_local_runtime_env(_RAYLIGHT_MODULE_PATH, _COMFY_ROOT_PATH, _RAYLIGHT_RUNTIME_WORKDIR)
_RAY_RUNTIME_ENV_REMOTE = _build_remote_runtime_env(_RAYLIGHT_MODULE_PATH, _COMFY_ROOT_PATH)
_LOCAL_CLUSTER_ADDRESSES = {None, "", "local", "LOCAL"}


def _quant_metadata_checker(unet_path: str) -> bool:
    try:
        from safetensors import safe_open
    except Exception:
        return False

    try:
        with safe_open(unet_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            if "_quantization_metadata" in metadata:
                return True
            return any(key.endswith(".comfy_quant") or key.endswith(".scaled_fp8") or key == "scaled_fp8" for key in f.keys())
    except Exception:
        return False


def _parse_pipefusion_stage_splits(stage_splits: str | None) -> tuple[int, ...] | None:
    if stage_splits is None:
        return None
    stage_splits = stage_splits.strip()
    if not stage_splits:
        return None
    splits = tuple(int(part.strip()) for part in stage_splits.split(",") if part.strip())
    if not splits:
        return None
    return splits


def _effective_parallel_degree(value: int | None) -> int:
    if value is None:
        return 1
    value = int(value)
    return 1 if value <= 0 else value


def _normalize_grouped_inputs(values: list, expected_length: int, label: str):
    if len(values) == expected_length:
        return values
    if len(values) == 1:
        return values * expected_length
    if len(values) > expected_length:
        return values[:expected_length]
    raise ValueError(f"{label} must provide 1 item or at least {expected_length} items, got {len(values)}")


def _collect_grouped_results(results: list, expected_length: int, label: str):
    grouped_results = {}
    for result in results:
        if result is None:
            continue
        dp_rank = int(result["dp_rank"])
        if dp_rank in grouped_results:
            raise RuntimeError(f"{label} produced multiple outputs for dp_rank {dp_rank}")
        grouped_results[dp_rank] = result["result"]

    missing = [dp_rank for dp_rank in range(expected_length) if dp_rank not in grouped_results]
    if missing:
        raise RuntimeError(f"{label} missing outputs for dp_rank {missing}")

    return [grouped_results[dp_rank] for dp_rank in range(expected_length)]


def _validate_unified_parallel_setup(parallel_dict: dict[str, Any], group_infos: list[dict[str, Any]], label: str):
    ulysses_degree = _effective_parallel_degree(parallel_dict.get("ulysses_degree"))
    ring_degree = _effective_parallel_degree(parallel_dict.get("ring_degree"))
    cfg_degree = _effective_parallel_degree(parallel_dict.get("cfg_degree"))
    pp_degree = _effective_parallel_degree(parallel_dict.get("pp_degree"))
    dp_degree = max(int(info.get("dp_degree", 1)) for info in group_infos) if group_infos else 1
    effective_parallel_size = ulysses_degree * ring_degree * cfg_degree * pp_degree * dp_degree

    if effective_parallel_size <= 1:
        raise ValueError(
            f"{label} requires an active unified parallel topology with effective degree > 1. "
            f"Got ulysses={ulysses_degree}, ring={ring_degree}, cfg={cfg_degree}, pp={pp_degree}, dp={dp_degree}. "
            "Use DPKSampler for pure DP or single-worker execution."
        )

    if not parallel_dict.get("is_xdit") and not parallel_dict.get("pipefusion_enabled"):
        raise ValueError(
            f"{label} requires xFuser unified topology to be active. "
            f"Got ulysses={ulysses_degree}, ring={ring_degree}, cfg={cfg_degree}, pp={pp_degree}, dp={dp_degree}, "
            f"is_xdit={parallel_dict.get('is_xdit')}, pipefusion_enabled={parallel_dict.get('pipefusion_enabled')}. "
            "DP-only execution should use DPKSampler."
        )


def _reset_pipefusion_runtime_config(parallel_dict: dict[str, Any]):
    parallel_dict["pipefusion_enabled"] = False
    parallel_dict["num_pipeline_patch"] = 1
    parallel_dict["warmup_steps"] = 0
    parallel_dict["pipefusion_stage_splits"] = None
    parallel_dict["pipefusion_debug"] = False


def _apply_pipefusion_runtime_config(
    parallel_dict: dict[str, Any],
    *,
    world_size: int,
    num_pipeline_patch: int,
    warmup_steps: int,
    stage_splits: str | None,
    debug: bool,
):
    _reset_pipefusion_runtime_config(parallel_dict)

    pp_degree = _effective_parallel_degree(parallel_dict.get("pp_degree"))

    if parallel_dict.get("is_fsdp"):
        raise ValueError("PipeFusion v1 is exclusive with FSDP")
    if world_size < 1:
        raise ValueError("PipeFusion requires at least 1 GPU")
    if pp_degree < 1:
        raise ValueError("PipeFusion pp_degree must be at least 1")
    if num_pipeline_patch <= 0:
        raise ValueError("PipeFusion num_pipeline_patch must be positive")
    if warmup_steps < 0:
        raise ValueError("PipeFusion warmup_steps cannot be negative")

    effective_ulysses = _effective_parallel_degree(parallel_dict.get("ulysses_degree"))
    effective_ring = _effective_parallel_degree(parallel_dict.get("ring_degree"))
    effective_cfg = _effective_parallel_degree(parallel_dict.get("cfg_degree"))
    model_parallel_size = pp_degree * effective_ulysses * effective_ring * effective_cfg
    if world_size % model_parallel_size != 0:
        raise ValueError(
            "PipeFusion requires the Ray worker count to be divisible by "
            "pp_degree * ulysses_degree * ring_degree * cfg_degree: "
            f"{world_size} is not divisible by {pp_degree} * {effective_ulysses} * {effective_ring} * {effective_cfg}"
        )

    parallel_dict["pipefusion_enabled"] = True
    parallel_dict["num_pipeline_patch"] = num_pipeline_patch
    parallel_dict["warmup_steps"] = warmup_steps
    parallel_dict["pipefusion_stage_splits"] = _parse_pipefusion_stage_splits(stage_splits)
    parallel_dict["pipefusion_debug"] = debug


class RayInitializer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_cluster_address": (
                    "STRING",
                    {
                        "default": "local",
                        "tooltip": "Ray cluster address. Use `local` for one machine, or a Ray head address for a remote cluster.",
                    },
                ),
                "ray_cluster_namespace": (
                    "STRING",
                    {"default": "default", "tooltip": "Ray namespace used to isolate this session from other Ray jobs."},
                ),
                "GPU": ("INT", {"default": 2, "tooltip": "How many GPUs / Ray workers to launch."}),
                "ulysses_degree": (
                    "INT",
                    {"default": 2, "tooltip": "Sequence parallel degree for Ulysses. Set above 1 to split sequence work across GPUs."},
                ),
                "ring_degree": (
                    "INT",
                    {
                        "default": 1,
                        "tooltip": "Ring attention degree. Usually leave at 1 unless you are intentionally testing ring parallelism.",
                    },
                ),
                "cfg_degree": (
                    "INT",
                    {"default": 1, "tooltip": "CFG parallel degree. `2` splits conditional and unconditional passes across GPUs."},
                ),
                "dp_degree": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "tooltip": "Data-parallel degree. Just use 1 or leave 0 when using Unified Parallel Sampler to auto use the remaining GPUs after ulysses/ring/cfg.",
                    },
                ),
                "sync_ulysses": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force a more synchronized Ulysses path. Can help with some VRAM spikes, but may be slower.",
                    },
                ),
                "FSDP": ("BOOLEAN", {"default": False, "tooltip": "Enable FSDP weight sharding across GPUs."}),
                "FSDP_CPU_OFFLOAD": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "When FSDP is on, offload inactive model shards to CPU RAM."},
                ),
                "XFuser_attention": (
                    [member.name for member in AttnType],
                    {"default": "TORCH_FLASH", "tooltip": "Attention backend used by xFuser-enabled execution."},
                ),
                "skip_comm_test": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Skip the startup NCCL communication test. Faster startup, but distributed issues are caught later.",
                    },
                ),
                "use_mmap": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use mmap-backed safetensor loading. This can reduce RAM spikes during model load, especially for large checkpoints.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS_INIT",)
    RETURN_NAMES = ("ray_actors_init",)

    FUNCTION = "spawn_actor"
    CATEGORY = "Raylight"

    def spawn_actor(
        self,
        ray_cluster_address: str,
        ray_cluster_namespace: str,
        GPU: int,
        ulysses_degree: int,
        ring_degree: int,
        cfg_degree: int,
        dp_degree: int,
        sync_ulysses: bool,
        FSDP: bool,
        FSDP_CPU_OFFLOAD: bool,
        XFuser_attention: str,
        skip_comm_test: bool = True,
        use_mmap: bool = True,
        GPU_SELECT: str = "",
        ray_object_store_gb: float = 2.0,
        ray_dashboard_address: str = "None",
        torch_dist_address: str = "None",
    ):
        # THIS IS PYTORCH DIST ADDRESS
        # (TODO) Change so it can be use in cluster of nodes. but it is long waaaaay down in the priority list
        # os.environ['TORCH_CUDA_ARCH_LIST'] = ""
        if torch_dist_address != "None":
            torch_host, torch_port = torch_dist_address.rsplit(":", 1)
            os.environ.setdefault("MASTER_ADDR", torch_host)
            os.environ.setdefault("MASTER_PORT", torch_port)
        else:
            torch_host, torch_port = "127.0.0.1", "29500"
            os.environ.setdefault("MASTER_ADDR", torch_host)
            os.environ.setdefault("MASTER_PORT", torch_port)

        # HF Tokenizer warning when forking
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.parallel_dict: dict[str, Any] = dict()
        _monkey()

        world_size = GPU
        selected_gpus = _parse_gpu_select(GPU_SELECT)
        if selected_gpus is None:
            max_world_size = torch.cuda.device_count()
        else:
            visible_gpu_count = torch.cuda.device_count()
            invalid = [gpu_idx for gpu_idx in selected_gpus if gpu_idx >= visible_gpu_count]
            if invalid:
                raise ValueError(f"GPU_SELECT contains GPU index outside visible range 0-{visible_gpu_count - 1}: {invalid}")
            max_world_size = len(selected_gpus)
        if world_size > max_world_size:
            raise ValueError(f"Too many gpus: requested {world_size} but only {max_world_size} selected/visible")
        if world_size == 0:
            raise ValueError("Num of cuda/cudalike device is 0")
        if world_size < ulysses_degree * ring_degree * cfg_degree:
            raise ValueError(f"ERROR, num_gpus: {world_size}, is lower than {ulysses_degree=} x {ring_degree=} x {cfg_degree=}")
        if cfg_degree > 2:
            raise ValueError("CFG batch only can be divided into 2 degree of parallelism, since its dimension is only 2")
        if dp_degree < 0:
            raise ValueError("dp_degree cannot be negative")
        if FSDP:
            if dp_degree < 1:
                raise ValueError("dp_degree must be >= 1 when FSDP is enabled")
            if GPU % dp_degree != 0:
                raise ValueError(f"GPU count ({GPU}) must be evenly divisible by dp_degree ({dp_degree})")
        shard_size = GPU // dp_degree if dp_degree >= 1 else GPU

        self.parallel_dict["is_xdit"] = False
        self.parallel_dict["is_fsdp"] = False
        self.parallel_dict["sync_ulysses"] = False
        self.parallel_dict["global_world_size"] = world_size
        self.parallel_dict["shard_size"] = shard_size
        self.parallel_dict["use_mmap"] = use_mmap
        self.parallel_dict["pp_degree"] = 1
        self.parallel_dict["dp_degree"] = dp_degree if dp_degree >= 1 else 1
        _reset_pipefusion_runtime_config(self.parallel_dict)

        if ulysses_degree > 0 or ring_degree > 0 or cfg_degree > 0:
            model_parallel_size = ulysses_degree * ring_degree * cfg_degree
            if model_parallel_size == 0:
                raise ValueError(f"""ERROR, parallel product of {ulysses_degree=} x {ring_degree=} x {cfg_degree=} is 0.
                 Please make sure to set any parallel degree to be greater than 0,
                 or switch into DPKSampler and set 0 to all parallel degree""")
            if world_size % model_parallel_size != 0:
                raise ValueError(
                    "GPU count must be divisible by ulysses_degree x ring_degree x cfg_degree: "
                    f"{world_size} is not divisible by {ulysses_degree} x {ring_degree} x {cfg_degree}"
                )
            auto_dp_degree = world_size // model_parallel_size
            effective_dp_degree = auto_dp_degree if dp_degree == 0 else dp_degree
            if world_size != model_parallel_size * effective_dp_degree:
                raise ValueError(
                    "GPU count must equal dp_degree x ulysses_degree x ring_degree x cfg_degree: "
                    f"{world_size} != {effective_dp_degree} x {ulysses_degree} x {ring_degree} x {cfg_degree}"
                )
            self.parallel_dict["attention"] = XFuser_attention
            self.parallel_dict["is_xdit"] = True
            self.parallel_dict["ulysses_degree"] = ulysses_degree
            self.parallel_dict["ring_degree"] = ring_degree
            self.parallel_dict["cfg_degree"] = cfg_degree
            self.parallel_dict["sync_ulysses"] = sync_ulysses
            self.parallel_dict["dp_degree"] = effective_dp_degree

        if FSDP:
            self.parallel_dict["fsdp_cpu_offload"] = FSDP_CPU_OFFLOAD
            self.parallel_dict["is_fsdp"] = True
            final_dp = self.parallel_dict["dp_degree"]
            self.parallel_dict["shard_size"] = world_size // final_dp

        if ray_dashboard_address != "None":
            dashboard_host, dashboard_port = ray_dashboard_address.rsplit(":", 1)
            dashboard_port = int(dashboard_port)
            enable_dashboard = True
        else:
            dashboard_host, dashboard_port = "127.0.0.1", None
            enable_dashboard = False

        ray_object_store_gb = int(ray_object_store_gb * 1024**3)
        runtime_env_base = deepcopy(_RAY_RUNTIME_ENV_LOCAL)
        if ray_cluster_address not in _LOCAL_CLUSTER_ADDRESSES:
            runtime_env_base = deepcopy(_RAY_RUNTIME_ENV_REMOTE)

        if selected_gpus is not None:
            # Adapted from avtc's Ray GPU visibility restriction idea.
            runtime_env_base.setdefault("env_vars", {})["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_idx) for gpu_idx in selected_gpus)

        if ray_cluster_address in _LOCAL_CLUSTER_ADDRESSES:
            _configure_raylight_ray_tmpdir(runtime_env_base)

        try:
            # Shut down so if comfy user try another workflow it will not cause error
            ray.shutdown()
            _cleanup_ray_temp()
            RayControlNetLoader._current_controlnet_path = None
            original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            restricted_cuda_visible_devices = runtime_env_base.get("env_vars", {}).get("CUDA_VISIBLE_DEVICES")
            if restricted_cuda_visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = restricted_cuda_visible_devices
            try:
                ray.init(
                    ray_cluster_address,
                    namespace=ray_cluster_namespace,
                    runtime_env=deepcopy(runtime_env_base),
                    object_store_memory=ray_object_store_gb,
                    include_dashboard=enable_dashboard,
                    dashboard_host=dashboard_host,
                    dashboard_port=dashboard_port,
                )
            finally:
                if restricted_cuda_visible_devices is not None:
                    if original_cuda_visible_devices is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
                    else:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        except Exception as e:
            ray.shutdown()
            _cleanup_ray_temp()
            if restricted_cuda_visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = restricted_cuda_visible_devices
            try:
                ray.init(runtime_env=deepcopy(runtime_env_base))
            finally:
                if restricted_cuda_visible_devices is not None:
                    if original_cuda_visible_devices is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
                    else:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            raise RuntimeError(f"Ray connection failed: {e}")

        if not skip_comm_test:
            print("Running NCCL communication test...")
            ray_nccl_tester(world_size)
        else:
            print("Skipping NCCL test (skip_comm_test=True)")
        ray_actor_fn = make_ray_actor_fn(world_size, self.parallel_dict)
        ray_actors = ray_actor_fn()
        return ([ray_actors, ray_actor_fn],)


class RayInitializerAdvanced(RayInitializer):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_cluster_address": (
                    "STRING",
                    {
                        "default": "local",
                        "tooltip": "Ray cluster address. Use `local` for one machine, or a Ray head address for a remote cluster.",
                    },
                ),
                "ray_cluster_namespace": (
                    "STRING",
                    {"default": "default", "tooltip": "Ray namespace used to isolate this session from other Ray jobs."},
                ),
                "GPU": ("INT", {"default": 2, "tooltip": "How many GPUs / Ray workers to launch."}),
                "GPU_SELECT": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "GPU indices for Ray workers. Use comma separated values like 0,1,2 to limit Ray to those GPUs, or leave empty to use all visible GPUs. Useful if you want to reserve GPU 0 for CLIP or VAE.",
                    },
                ),
                "ulysses_degree": (
                    "INT",
                    {"default": 2, "tooltip": "Sequence parallel degree for Ulysses. Set above 1 to split sequence work across GPUs."},
                ),
                "ring_degree": (
                    "INT",
                    {
                        "default": 1,
                        "tooltip": "Ring attention degree. Usually leave at 1 unless you are intentionally testing ring parallelism.",
                    },
                ),
                "cfg_degree": (
                    "INT",
                    {"default": 1, "tooltip": "CFG parallel degree. `2` splits conditional and unconditional passes across GPUs."},
                ),
                "dp_degree": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "tooltip": "Data-parallel degree. Default 1 keeps the legacy layout. Leave 0 when using Unified Parallel Sampler to auto use the remaining GPUs after ulysses/ring/cfg.",
                    },
                ),
                "sync_ulysses": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force a more synchronized Ulysses path. Can help with some VRAM spikes, but may be slower.",
                    },
                ),
                "FSDP": ("BOOLEAN", {"default": False, "tooltip": "Enable FSDP weight sharding across GPUs."}),
                "FSDP_CPU_OFFLOAD": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "When FSDP is on, offload inactive model shards to CPU RAM."},
                ),
                "XFuser_attention": (
                    [member.name for member in AttnType],
                    {"default": "TORCH_FLASH", "tooltip": "Attention backend used by xFuser-enabled execution."},
                ),
                "skip_comm_test": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Skip the startup NCCL communication test. Faster startup, but distributed issues are caught later.",
                    },
                ),
                "use_mmap": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use mmap-backed safetensor loading. This can reduce RAM spikes during model load, especially for large checkpoints.",
                    },
                ),
            },
            "optional": {
                "ray_object_store_gb": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "tooltip": "Ray object-store size in GB. Usually the default is enough unless you move large tensors through Ray.",
                    },
                ),
                "ray_dashboard_address": (
                    "STRING",
                    {"default": "None", "tooltip": "Optional Ray dashboard bind address like `127.0.0.1:8265` for monitoring."},
                ),
                "torch_dist_address": (
                    "STRING",
                    {
                        "default": "127.0.0.1:29500",
                        "tooltip": "Torch distributed master address used by worker-side NCCL init. Restart ComfyUI if you change it.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS_INIT",)
    RETURN_NAMES = ("ray_actors_init",)

    FUNCTION = "spawn_actor"
    CATEGORY = "Raylight"


class RayPipeFusionConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors_init": ("RAY_ACTORS_INIT", {"tooltip": "Ray actor initialization payload to decorate with PipeFusion config."}),
                "num_pipeline_patch": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "tooltip": "How many smaller sequence chunks to split the sequence into after patch embedding. Higher values can reduce pipeline bubbles, but too high adds overhead.",
                    },
                ),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "tooltip": "How many denoising steps to run in simple synchronous mode before switching to steady-state PipeFusion scheduling. Since t0 usually required fresh KV (just like teacache warm up step)",
                    },
                ),
                "pipefusion_stage_splits": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional manual block split per stage, for example for wan 14b `20,20`. Leave empty to split transformer blocks evenly across PP stages.",
                    },
                ),
                "pipefusion_debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Print extra PipeFusion logs such as stage ownership, chunk flow, and warmup/pipeline mode decisions.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS_INIT",)
    RETURN_NAMES = ("ray_actors_init",)

    FUNCTION = "configure_pipefusion"
    CATEGORY = "Raylight"

    def configure_pipefusion(
        self,
        ray_actors_init,
        num_pipeline_patch: int,
        warmup_steps: int,
        pipefusion_stage_splits: str = "",
        pipefusion_debug: bool = False,
    ):
        ray_actors, gpu_actors, parallel_dict = ensure_fresh_actors(ray_actors_init)

        updated_parallel_dict = dict(parallel_dict)
        _apply_pipefusion_runtime_config(
            updated_parallel_dict,
            world_size=int(updated_parallel_dict.get("global_world_size", len(gpu_actors))),
            num_pipeline_patch=num_pipeline_patch,
            warmup_steps=warmup_steps,
            stage_splits=pipefusion_stage_splits,
            debug=pipefusion_debug,
        )

        updated_actor_fn = make_ray_actor_fn(int(updated_parallel_dict.get("global_world_size", len(gpu_actors))), updated_parallel_dict)
        ray.get([actor.set_parallel_dict.remote(updated_parallel_dict) for actor in gpu_actors])
        return ([ray_actors, updated_actor_fn],)


class RayUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("checkpoints"),),
                "weight_dtype": (
                    [
                        "default",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e5m2",
                        "bf16",
                        "fp16",
                    ],
                ),
                "ray_actors_init": (
                    "RAY_ACTORS_INIT",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
            },
            "optional": {"lora": ("RAY_LORA", {"default": None})},
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "load_ray_unet"

    CATEGORY = "Raylight"

    def load_ray_unet(self, ray_actors_init, unet_name, weight_dtype, lora=None):
        ray_actors, gpu_actors, parallel_dict = ensure_fresh_actors(ray_actors_init)

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        try:
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        except:
            unet_path = folder_paths.get_full_path_or_raise("checkpoints", unet_name)

        loaded_futures = []
        patched_futures = []

        for actor in gpu_actors:
            loaded_futures.append(actor.set_lora_list.remote(lora))
        ray.get(loaded_futures)
        loaded_futures = []

        parallel_dict["is_quant"] = _quant_metadata_checker(unet_path)
        for actor in gpu_actors:
            loaded_futures.append(actor.set_parallel_dict.remote(parallel_dict))
        ray.get(loaded_futures)
        loaded_futures = []

        if parallel_dict["is_fsdp"] is True:
            num_replicas = parallel_dict.get("dp_degree", 1)
            shard_size = parallel_dict.get("shard_size", len(gpu_actors))

            # Reuse detection for FSDP non-quant: if the first worker already
            # has this exact model + lora combo, all workers do (loaded together).
            # Skip the expensive reload — handles MCP re-creating the same
            # workflow and ComfyUI re-executing RayUNETLoader with identical inputs.
            if parallel_dict["is_quant"] is False:
                already_loaded = ray.get(gpu_actors[0].check_model_loaded.remote(unet_path, model_options))
                if already_loaded:
                    return (ray_actors,)

            if num_replicas <= 1:
                # Single replica — all GPUs share one FSDP-sharded model
                if parallel_dict["is_quant"] is False:
                    worker0 = ray.get_actor("RayWorker:0")
                    ray.get(worker0.load_unet.remote(unet_path, model_options=model_options))
                    meta_model = ray.get(worker0.get_meta_model.remote())

                    for actor in gpu_actors:
                        if actor != worker0:
                            loaded_futures.append(actor.set_meta_model.remote(meta_model))

                    ray.get(loaded_futures)
                    loaded_futures = []

                    for actor in gpu_actors:
                        loaded_futures.append(actor.set_state_dict.remote())

                else:
                    for actor in gpu_actors:
                        loaded_futures.append(actor.load_unet.remote(unet_path, model_options=model_options))

                    ray.get(loaded_futures)
                    loaded_futures = []

                    for actor in gpu_actors:
                        loaded_futures.append(actor.set_state_dict.remote())

            else:
                # Multiple replicas — load model per group
                for group_id in range(num_replicas):
                    group_actors = gpu_actors[group_id * shard_size : (group_id + 1) * shard_size]

                    if parallel_dict["is_quant"] is False:
                        rank0_name = f"RayWorker:{group_id}_0"
                        worker0 = ray.get_actor(rank0_name)
                        ray.get(worker0.load_unet.remote(unet_path, model_options=model_options))
                        meta_model = ray.get(worker0.get_meta_model.remote())

                        for actor in group_actors:
                            if actor != worker0:
                                loaded_futures.append(actor.set_meta_model.remote(meta_model))

                        ray.get(loaded_futures)
                        loaded_futures = []

                        for actor in group_actors:
                            loaded_futures.append(actor.set_state_dict.remote())

                    else:
                        for actor in group_actors:
                            loaded_futures.append(actor.load_unet.remote(unet_path, model_options=model_options))

                        ray.get(loaded_futures)
                        loaded_futures = []

                        for actor in group_actors:
                            loaded_futures.append(actor.set_state_dict.remote())

            ray.get(loaded_futures)
            loaded_futures = []
        else:
            for actor in gpu_actors:
                loaded_futures.append(actor.load_unet.remote(unet_path, model_options=model_options))
            ray.get(loaded_futures)
            loaded_futures = []

        for actor in gpu_actors:
            if parallel_dict["is_xdit"] and not parallel_dict.get("pipefusion_enabled"):
                if (parallel_dict["ulysses_degree"]) > 1 or (parallel_dict["ring_degree"] > 1):
                    patched_futures.append(actor.patch_usp.remote())
                if parallel_dict["cfg_degree"] > 1:
                    patched_futures.append(actor.patch_cfg.remote())
            if parallel_dict.get("pipefusion_enabled"):
                patched_futures.append(actor.patch_pipefusion.remote())

        ray.get(patched_futures)

        return (ray_actors,)


class RayLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "The name of the LoRA."},
                ),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
            },
            "optional": {"prev_ray_lora": ("RAY_LORA", {"default": None})},
        }

    RETURN_TYPES = ("RAY_LORA",)
    RETURN_NAMES = ("ray_lora",)
    FUNCTION = "load_lora"
    CATEGORY = "Raylight"

    def load_lora(self, lora_name, strength_model, prev_ray_lora=None):
        loras_list = []

        if strength_model == 0.0:
            if prev_ray_lora is not None:
                loras_list.extend(prev_ray_lora)
            return (loras_list,)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = {
            "path": lora_path,
            "strength_model": strength_model,
        }

        if prev_ray_lora is not None:
            loras_list.extend(prev_ray_lora)

        loras_list.append(lora)
        return (loras_list,)


class XFuserKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "ray_actors": (
                    "RAY_ACTORS",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "RAY_ACTORS")
    RETURN_NAMES = ("latent", "ray_actors")
    FUNCTION = "ray_sample"

    CATEGORY = "Raylight"

    def ray_sample(
        self,
        ray_actors,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):
        # Clean VRAM for preparation to load model
        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        gpu_actors = ray_actors["workers"]
        futures = [
            actor.common_ksampler.remote(
                noise_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=force_full_denoise,
            )
            for actor in gpu_actors
        ]

        results = ray.get(futures)
        return (results[0][0], ray_actors)


class UnifiedParallelSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "add_noise": (["enable", "disable"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "ray_actors": (
                    "RAY_ACTORS",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
                "noise_list": (
                    "NOISE",
                    {
                        "tooltip": "List of noise seeds for each xFuser data-parallel group. Use one item to share the same seed across all groups."
                    },
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "RAY_ACTORS")
    RETURN_NAMES = ("latent", "ray_actors")
    OUTPUT_IS_LIST = (True, False)
    INPUT_IS_LIST = True
    FUNCTION = "ray_sample"

    CATEGORY = "Raylight"

    def ray_sample(
        self,
        ray_actors,
        add_noise,
        noise_list,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):
        ray_actors = ray_actors[0]
        add_noise = add_noise[0]
        steps = steps[0]
        cfg = cfg[0]
        sampler_name = sampler_name[0]
        scheduler = scheduler[0]
        start_at_step = start_at_step[0]
        end_at_step = end_at_step[0]
        return_with_leftover_noise = return_with_leftover_noise[0]
        denoise = denoise[0]

        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        gpu_actors = ray_actors["workers"]
        parallel_dict = ray.get(gpu_actors[0].get_parallel_dict.remote())
        group_infos = ray.get([actor.get_exec_group_info.remote() for actor in gpu_actors])
        _validate_unified_parallel_setup(parallel_dict, group_infos, "Unified Parallel Sampler")
        dp_degree = int(group_infos[0]["dp_degree"])
        noise_list = _normalize_grouped_inputs(noise_list, dp_degree, "noise_list")
        positive = _normalize_grouped_inputs(positive, dp_degree, "positive")
        negative = _normalize_grouped_inputs(negative, dp_degree, "negative")
        latent_image = _normalize_grouped_inputs(latent_image, dp_degree, "latent_image")
        futures = [
            actor.common_ksampler.remote(
                noise_list[group_info["dp_rank"]],
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive[group_info["dp_rank"]],
                negative[group_info["dp_rank"]],
                latent_image[group_info["dp_rank"]],
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=force_full_denoise,
                grouped_output=True,
            )
            for actor, group_info in zip(gpu_actors, group_infos)
        ]

        results = ray.get(futures)
        results = _collect_grouped_results(results, dp_degree, "Unified Parallel Sampler")
        return (results, ray_actors)


class DPKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "add_noise": (["enable", "disable"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "ray_actors": (
                    "RAY_ACTORS",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
                "noise_list": ("NOISE", {"tooltip": "List of noise seeds for each GPU in data parallel mode"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "RAY_ACTORS")
    RETURN_NAMES = ("latent", "ray_actors")
    OUTPUT_IS_LIST = (True, False)
    INPUT_IS_LIST = True
    FUNCTION = "ray_sample"

    CATEGORY = "Raylight"

    def ray_sample(
        self,
        ray_actors,
        add_noise,
        noise_list,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):
        ray_actors = ray_actors[0]
        add_noise = add_noise[0]
        steps = steps[0]
        cfg = cfg[0]
        sampler_name = sampler_name[0]
        scheduler = scheduler[0]
        start_at_step = start_at_step[0]
        end_at_step = end_at_step[0]
        return_with_leftover_noise = return_with_leftover_noise[0]
        denoise = denoise[0]

        gpu_actors = ray_actors["workers"]
        parallel_dict = ray.get(gpu_actors[0].get_parallel_dict.remote())
        if parallel_dict["is_xdit"] is True:
            raise ValueError(
                """
            Data Parallel KSampler only supports FSDP or standard Data Parallel (DP).
            Please set both 'ulysses_degree' and 'ring_degree' to 0,
            or use the XFuser KSampler instead. More info on Raylight mode https://github.com/komikndr/raylight
            """
            )

        num_gpus = len(gpu_actors)
        # Replicate last latent to fill remaining slots, or truncate if too many
        if len(latent_image) < num_gpus:
            latent_image = latent_image + [latent_image[-1]] * (num_gpus - len(latent_image))
        elif len(latent_image) > num_gpus:
            latent_image = latent_image[:num_gpus]
        if len(positive) == 1:
            positive = positive * num_gpus
        if len(negative) == 1:
            negative = negative * num_gpus
        if len(noise_list) > num_gpus:
            noise_list = noise_list[:num_gpus]
        else:
            noise_list = [noise_list[0]] * num_gpus

        # Clean VRAM for preparation to load model
        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        # Each GPU gets its own noise/conditioning — decoupled from FSDP sharding
        futures = [
            actor.common_ksampler.remote(
                noise_list[i],
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive[i],
                negative[i],
                latent_image[i],
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=force_full_denoise,
            )
            for i, actor in enumerate(gpu_actors)
        ]

        results = ray.get(futures)
        results = [result[0] for result in results]
        return (results, ray_actors)


class RayKill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS", {"tooltip": "Ray actors to shut down cleanly."}),
                "kill_mode": (
                    ["Kill Workers Only", "Kill Entire Cluster"],
                    {
                        "tooltip": "This terminal node function to cleanly shutdown worker or entire cluster, this is usefull if you want to switch to regular WF without restarting ComfyUI, or incase of error in Raylight",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "kill_ray"
    OUTPUT_NODE = True
    CATEGORY = "Raylight"

    def kill_ray(self, ray_actors, kill_mode):
        gpu_actors = ray_actors["workers"]
        futures = [actor.kill.remote() for actor in gpu_actors]
        try:
            ray.get(futures)
        except ray.exceptions.RayActorError:
            pass

        if kill_mode == "Kill Entire Cluster":
            ray.shutdown()
            _cleanup_ray_temp()
            RayControlNetLoader._current_controlnet_path = None

        return ()


class RayControlNetLoader:
    """Load a ControlNet model into all Ray workers.

    Works like the standard ControlNetLoader but loads the model on each
    worker's GPU from disk, avoiding Ray serialization of multi-GB weights.
    Returns a lightweight reference that RayControlNetApply uses.

    Only one ControlNet model per workflow is supported. To apply the same
    model with different images/strengths, use multiple RayControlNetApply
    nodes connected to a single RayControlNetLoader.
    """

    _current_controlnet_path = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
            }
        }

    RETURN_TYPES = ("RAY_CONTROL_NET",)
    RETURN_NAMES = ("ray_control_net",)
    FUNCTION = "load_controlnet"
    CATEGORY = "Raylight"

    def load_controlnet(self, ray_actors, control_net_name):
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)

        if (RayControlNetLoader._current_controlnet_path is not None
                and RayControlNetLoader._current_controlnet_path != controlnet_path):
            raise RuntimeError(
                f"Only one ControlNet model per workflow is supported. "
                f"Already loaded '{RayControlNetLoader._current_controlnet_path}', "
                f"attempted '{controlnet_path}'. Use multiple RayControlNetApply nodes "
                f"to apply the same model with different images."
            )
        RayControlNetLoader._current_controlnet_path = controlnet_path

        # Validate the ControlNet loads in the main process first
        import comfy.controlnet as comfy_controlnet

        controlnet = comfy_controlnet.load_controlnet(controlnet_path)
        if controlnet is None:
            raise RuntimeError(f"Invalid ControlNet file: {control_net_name}")
        del controlnet
        gc.collect()

        # Send the path to all workers — each loads from disk independently
        gpu_actors = ray_actors["workers"]
        futures = [actor.load_controlnet.remote(controlnet_path) for actor in gpu_actors]
        results = ray.get(futures)
        if not all(results):
            raise RuntimeError(f"Failed to load ControlNet on one or more workers: {control_net_name}")

        return (controlnet_path,)


class RayControlNetApply:
    """Apply a Ray-loaded ControlNet to conditioning.

    Works like ApplyControlNet but uses a lightweight reference instead of
    embedding the full model in the conditioning data.  The workers restore
    the real ControlNet from their local cache during sampling.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "ray_control_net": ("RAY_CONTROL_NET",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "ray_vae": ("RAY_VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"
    CATEGORY = "Raylight"

    def apply_controlnet(self, positive, negative, ray_control_net, image, strength,
                         start_percent, end_percent, ray_vae=None, extra_concat=None):
        from .distributed_worker.ray_worker_controlnet import _RayControlNetRef

        if strength == 0:
            return (positive, negative)

        if extra_concat is None:
            extra_concat = []

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get("control", None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = _RayControlNetRef(
                        strength=strength,
                        timestep_percent_range=(start_percent, end_percent),
                        cond_hint_original=control_hint,
                        extra_concat_orig=extra_concat,
                        needs_vae=(ray_vae is not None),
                    )
                    if prev_cnet is not None:
                        c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d["control"] = c_net
                d["control_apply_to_uncond"] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])


class RayVAELoader:
    """Load a VAE model into all Ray workers.

    Loads the VAE on each worker's GPU from disk.  Used by RayControlNetApply
    when the ControlNet requires a VAE (e.g. for encoding the control image).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "vae_name": (folder_paths.get_filename_list("vae"),),
            }
        }

    RETURN_TYPES = ("RAY_VAE",)
    RETURN_NAMES = ("ray_vae",)
    FUNCTION = "load_vae"
    CATEGORY = "Raylight"

    def load_vae(self, ray_actors, vae_name):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)

        gpu_actors = ray_actors["workers"]
        ray.get([actor.ray_vae_loader.remote(vae_path) for actor in gpu_actors])

        return (vae_path,)


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


class DPNoiseList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **{
                    f"noise_seed_{i}": (
                        "INT",
                        {
                            "default": 0,
                            "min": 0,
                            "max": 0xFFFFFFFFFFFFFFFF,
                            "control_after_generate": True,
                            "tooltip": f"Noise seed for GPU {i} in data parallel mode. Different seeds vary outputs across GPUs.",
                        },
                    )
                    for i in range(8)
                }
            }
        }

    RETURN_TYPES = ("NOISE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "get_noise"
    CATEGORY = "Raylight"

    def get_noise(self, **kwargs):
        noise_list = []
        for key, seed in kwargs.items():
            if key.startswith("noise_seed_"):
                noise_list.append(seed)
        return (noise_list,)


class DPConditioningList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_0": ("CONDITIONING",),
                "negative_0": ("CONDITIONING",),
            },
            "optional": {
                **{k: ("CONDITIONING",) for i in range(1, 8) for k in (f"positive_{i}", f"negative_{i}")},
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "assemble"
    CATEGORY = "Raylight"

    def assemble(self, positive_0, negative_0, **kwargs):
        positives = [positive_0]
        negatives = [negative_0]
        for i in range(1, 8):
            positives.append(kwargs.get(f"positive_{i}", positive_0))
            negatives.append(kwargs.get(f"negative_{i}", negative_0))
        return (positives, negatives)


class DPLatentList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_0": ("LATENT", {"tooltip": "Latent for GPU 0"}),
            },
            "optional": {
                **{f"latent_{i}": ("LATENT", {"tooltip": f"Latent for GPU {i}"}) for i in range(1, 8)},
            },
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "assemble"
    CATEGORY = "Raylight"

    def assemble(self, latent_0, **kwargs):
        latents = [latent_0]
        for i in range(1, 8):
            latents.append(kwargs.get(f"latent_{i}", latent_0))
        return (latents,)


class RayVAEDecodeDistributed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS", {"tooltip": "Ray Actor to submit the model into"}),
                "samples": ("LATENT", {"tooltip": "Latent samples to decode."}),
                "vae_name": (folder_paths.get_filename_list("vae"), {"tooltip": "Name of the VAE model to use for decoding."}),
                "tile_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Tile size for spatial decoding. Larger tiles use more memory.",
                    },
                ),
                "overlap": (
                    "INT",
                    {"default": 64, "min": 0, "max": 4096, "step": 32, "tooltip": "Pixel overlap between tiles to prevent artifacts."},
                ),
                "temporal_size": (
                    "INT",
                    {
                        "default": 64,
                        "min": 8,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Only used for video VAEs: Amount of frames to decode at a time.",
                    },
                ),
                "temporal_overlap": (
                    "INT",
                    {
                        "default": 8,
                        "min": 4,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Only used for video VAEs: Amount of frames to overlap.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ray_decode"

    CATEGORY = "Raylight"

    def ray_decode(self, ray_actors, vae_name, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
        gpu_actors = ray_actors["workers"]
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)

        for actor in gpu_actors:
            ray.get(actor.ray_vae_loader.remote(vae_path))

        futures = [
            actor.ray_vae_decode.remote(samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8)
            for i, actor in enumerate(gpu_actors)
        ]

        image = ray.get(futures)
        return (image[0],)


NODE_CLASS_MAPPINGS = {
    "XFuserKSamplerAdvanced": XFuserKSamplerAdvanced,
    "UnifiedParallelSampler": UnifiedParallelSampler,
    "DPKSamplerAdvanced": DPKSamplerAdvanced,
    "RayKill": RayKill,
    "RayUNETLoader": RayUNETLoader,
    "RayLoraLoader": RayLoraLoader,
    "RayControlNetLoader": RayControlNetLoader,
    "RayControlNetApply": RayControlNetApply,
    "RayVAELoader": RayVAELoader,
    "RayInitializer": RayInitializer,
    "RayInitializerAdvanced": RayInitializerAdvanced,
    "DPNoiseList": DPNoiseList,
    "DPConditioningList": DPConditioningList,
    "DPLatentList": DPLatentList,
    "RayVAEDecodeDistributed": RayVAEDecodeDistributed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XFuserKSamplerAdvanced": "XFuser KSampler (Advanced)",
    "UnifiedParallelSampler": "Unified Parallel Sampler (Advance)",
    "DPKSamplerAdvanced": "Data Parallel KSampler (Advanced)",
    "RayKill": "Kill Ray",
    "RayUNETLoader": "Load Diffusion Model (Ray)",
    "RayLoraLoader": "Load Lora Model (Ray)",
    "RayControlNetLoader": "Load ControlNet (Ray)",
    "RayControlNetApply": "Apply ControlNet (Ray)",
    "RayVAELoader": "Load VAE (Ray)",
    "RayInitializer": "Ray Init Actor",
    "RayInitializerAdvanced": "Ray Init Actor (Advanced)",
    "DPNoiseList": "Data Parallel Noise List",
    "DPConditioningList": "Data Parallel Conditioning List",
    "DPLatentList": "Data Parallel Latent List",
    "RayVAEDecodeDistributed": "Distributed VAE (Ray)",
}
