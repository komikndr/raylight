from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from types import ModuleType

import torch


def _find_xfuser_root() -> Path | None:
    spec = importlib.util.find_spec("xfuser")
    if spec is None:
        return None
    locations = spec.submodule_search_locations
    if not locations:
        origin = spec.origin
        if origin is None:
            return None
        return Path(origin).resolve().parent
    return Path(next(iter(locations))).resolve()


def _ensure_package(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is not None:
        return module

    module = types.ModuleType(name)
    module.__file__ = str(path / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(path)]
    spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    if spec is not None:
        spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    sys.modules[name] = module
    return module


def _load_module(name: str, file_path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_logger_stub() -> None:
    if "xfuser.logger" in sys.modules:
        return

    module = types.ModuleType("xfuser.logger")
    module.__package__ = "xfuser"

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    root_logger = logging.getLogger("xfuser")
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    root_logger.propagate = False

    def init_logger(name: str):
        logger = logging.getLogger(name)
        logger.setLevel(root_logger.level)
        for handler in root_logger.handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
        logger.propagate = False
        return logger

    module.init_logger = init_logger
    sys.modules[module.__name__] = module


def _install_envs_stub() -> None:
    if "xfuser.envs" in sys.modules:
        return

    module = types.ModuleType("xfuser.envs")
    module.__package__ = "xfuser"

    def _is_hip() -> bool:
        return torch.version.hip is not None

    def _is_cuda() -> bool:
        return torch.version.cuda is not None

    def _is_musa() -> bool:
        musa = getattr(torch, "musa", None)
        return bool(musa and musa.is_available())

    def _is_mps() -> bool:
        return torch.backends.mps.is_available()

    def _is_npu() -> bool:
        try:
            return bool(hasattr(torch, "npu") and torch.npu.is_available())
        except ModuleNotFoundError:
            return False

    def get_device(local_rank: int) -> torch.device:
        if _is_cuda() or _is_hip():
            return torch.device("cuda", local_rank)
        if _is_musa():
            return torch.device("musa", local_rank)
        if _is_mps():
            return torch.device("mps")
        if _is_npu():
            return torch.device("npu", local_rank)
        return torch.device("cpu")

    def get_torch_distributed_backend() -> str:
        if _is_cuda() or _is_hip():
            return "nccl"
        if _is_musa():
            return "mccl"
        if _is_npu():
            return "hccl"
        return "gloo"

    class _PackagesChecker:
        def get_packages_info(self):
            return {
                "has_aiter": False,
                "has_flash_attn": False,
                "has_long_ctx_attn": True,
                "diffusers_version": None,
            }

    module.MASTER_ADDR = os.getenv("MASTER_ADDR", "")
    module.MASTER_PORT = int(os.getenv("MASTER_PORT", "0")) if "MASTER_PORT" in os.environ else None
    module.LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
    module.PACKAGES_CHECKER = _PackagesChecker()
    module._is_hip = _is_hip
    module._is_cuda = _is_cuda
    module._is_musa = _is_musa
    module._is_mps = _is_mps
    module._is_npu = _is_npu
    module.get_device = get_device
    module.get_torch_distributed_backend = get_torch_distributed_backend
    sys.modules[module.__name__] = module


def _install_runtime_state_stub() -> None:
    name = "xfuser.core.distributed.runtime_state"
    if name in sys.modules:
        return

    module = types.ModuleType(name)
    module.__package__ = "xfuser.core.distributed"

    def runtime_state_is_initialized() -> bool:
        return False

    def get_runtime_state():
        raise RuntimeError("xFuser runtime_state is not available in Raylight bootstrapless mode")

    def initialize_runtime_state(*args, **kwargs):
        raise RuntimeError("xFuser runtime_state is not available in Raylight bootstrapless mode")

    module.runtime_state_is_initialized = runtime_state_is_initialized
    module.get_runtime_state = get_runtime_state
    module.initialize_runtime_state = initialize_runtime_state
    sys.modules[name] = module


def _install_cache_manager_stub() -> None:
    package = "xfuser.core.cache_manager"
    _ensure_package(package, Path("."))
    name = f"{package}.cache_manager"
    if name in sys.modules:
        return

    module = types.ModuleType(name)
    module.__package__ = package

    class _CacheManager:
        def register_cache_entry(self, *args, **kwargs):
            return None

        def update_and_get_kv_cache(self, *args, **kwargs):
            raise RuntimeError("xFuser KV cache is not available in Raylight bootstrapless mode")

    _cache_manager = _CacheManager()

    def get_cache_manager():
        return _cache_manager

    module.CacheManager = _CacheManager
    module.get_cache_manager = get_cache_manager
    sys.modules[name] = module


def _install_distributed_modules(xfuser_root: Path) -> None:
    distributed_root = xfuser_root / "core" / "distributed"
    _ensure_package("xfuser.core.distributed", distributed_root)
    _install_runtime_state_stub()

    utils = _load_module("xfuser.core.distributed.utils", distributed_root / "utils.py")
    group = _load_module("xfuser.core.distributed.group_coordinator", distributed_root / "group_coordinator.py")
    parallel = _load_module("xfuser.core.distributed.parallel_state", distributed_root / "parallel_state.py")

    distributed_pkg = sys.modules["xfuser.core.distributed"]
    exported_names = [
        "get_world_group",
        "get_dp_group",
        "get_cfg_group",
        "get_sp_group",
        "get_pp_group",
        "get_pipeline_parallel_world_size",
        "get_pipeline_parallel_rank",
        "is_pipeline_first_stage",
        "is_pipeline_last_stage",
        "get_data_parallel_world_size",
        "get_data_parallel_rank",
        "is_dp_last_group",
        "get_classifier_free_guidance_world_size",
        "get_classifier_free_guidance_rank",
        "get_sequence_parallel_world_size",
        "get_sequence_parallel_rank",
        "get_ulysses_parallel_world_size",
        "get_ulysses_parallel_rank",
        "get_ring_parallel_world_size",
        "get_ring_parallel_rank",
        "init_distributed_environment",
        "initialize_model_parallel",
        "model_parallel_is_initialized",
        "get_tensor_model_parallel_world_size",
        "get_vae_parallel_group",
        "get_vae_parallel_rank",
        "get_vae_parallel_world_size",
        "get_dit_world_size",
        "init_vae_group",
        "init_dit_group",
        "get_dit_group",
    ]
    for name in exported_names:
        setattr(distributed_pkg, name, getattr(parallel, name))
    distributed_pkg.RankGenerator = utils.RankGenerator
    distributed_pkg.__all__ = exported_names + ["RankGenerator"]
    distributed_pkg.utils = utils
    distributed_pkg.group_coordinator = group
    distributed_pkg.parallel_state = parallel
    distributed_pkg.runtime_state = sys.modules["xfuser.core.distributed.runtime_state"]


def _install_long_ctx_attention(xfuser_root: Path) -> None:
    long_ctx_root = xfuser_root / "core" / "long_ctx_attention"
    hybrid_root = long_ctx_root / "hybrid"
    ring_root = long_ctx_root / "ring"
    _ensure_package("xfuser.core.long_ctx_attention", long_ctx_root)
    _ensure_package("xfuser.core.long_ctx_attention.hybrid", hybrid_root)
    _ensure_package("xfuser.core.long_ctx_attention.ring", ring_root)
    _install_cache_manager_stub()

    attn_layer = _load_module(
        "xfuser.core.long_ctx_attention.hybrid.attn_layer",
        hybrid_root / "attn_layer.py",
    )

    try:
        from yunchang.kernels import AttnType as _RealAttnType
    except ImportError:
        _RealAttnType = attn_layer.AttnType

    hybrid_pkg = sys.modules["xfuser.core.long_ctx_attention.hybrid"]
    hybrid_pkg.xFuserLongContextAttention = attn_layer.xFuserLongContextAttention
    hybrid_pkg.xFuserSanaLinearLongContextAttention = attn_layer.xFuserSanaLinearLongContextAttention
    hybrid_pkg.AttnType = _RealAttnType
    hybrid_pkg.__all__ = [
        "xFuserLongContextAttention",
        "xFuserSanaLinearLongContextAttention",
        "AttnType",
    ]

    long_ctx_pkg = sys.modules["xfuser.core.long_ctx_attention"]
    long_ctx_pkg.xFuserLongContextAttention = attn_layer.xFuserLongContextAttention
    long_ctx_pkg.xFuserSanaLinearLongContextAttention = attn_layer.xFuserSanaLinearLongContextAttention
    long_ctx_pkg.AttnType = _RealAttnType
    long_ctx_pkg.__all__ = hybrid_pkg.__all__

    _torch_cuda = None
    _orig_cuda_available = None
    try:
        import torch.cuda as _torch_cuda
        _orig_cuda_available = _torch_cuda.is_available
        _torch_cuda.is_available = lambda: True

        import yunchang.kernels as _yunchang_kernels
        if getattr(_yunchang_kernels, "AttnType", None) is None:
            _yunchang_kernels.AttnType = _RealAttnType
    except (ImportError, AttributeError):
        pass

    try:
        ring_mod = _load_module(
            "xfuser.core.long_ctx_attention.ring.ring_flash_attn",
            ring_root / "ring_flash_attn.py",
        )
    finally:
        if _torch_cuda is not None and _orig_cuda_available is not None:
            _torch_cuda.is_available = _orig_cuda_available
    ring_pkg = sys.modules["xfuser.core.long_ctx_attention.ring"]
    ring_pkg.xdit_ring_flash_attn_func = ring_mod.xdit_ring_flash_attn_func
    ring_pkg.xdit_sana_ring_flash_attn_func = ring_mod.xdit_sana_ring_flash_attn_func
    ring_pkg.__all__ = ["xdit_ring_flash_attn_func", "xdit_sana_ring_flash_attn_func"]


def install_minimal_xfuser() -> None:
    if os.getenv("RAYLIGHT_DISABLE_XFUSER_SHIM") == "1":
        return
    if "xfuser.core.distributed" in sys.modules and "xfuser.core.long_ctx_attention" in sys.modules:
        return

    xfuser_root = _find_xfuser_root()
    if xfuser_root is None:
        return

    _ensure_package("xfuser", xfuser_root)
    _ensure_package("xfuser.core", xfuser_root / "core")
    _install_logger_stub()
    _install_envs_stub()
    _install_distributed_modules(xfuser_root)
    _install_long_ctx_attention(xfuser_root)
