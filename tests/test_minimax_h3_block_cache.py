import ast
import importlib.util
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
BLOCK_CACHE_PATH = ROOT / "src/raylight/diffusion_models/minimax/block_cache.py"
NODES_PATH = ROOT / "src/raylight/comfy_extra_dist/nodes_minimax_h3.py"
FORWARD_PATH = ROOT / "src/raylight/diffusion_models/minimax/xdit_context_parallel.py"


def _load_block_cache():
    spec = importlib.util.spec_from_file_location("minimax_h3_block_cache_test", BLOCK_CACHE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


block_cache = _load_block_cache()


def _runtime(**overrides):
    values = {
        "sigma_threshold": 0.12,
        "start_percent": 0.0,
        "end_percent": 1.0,
        "max_cached_steps": 2,
        "cache_depth": 0.75,
    }
    values.update(overrides)
    config = block_cache.MiniMaxH3BlockCacheConfig(**values)
    return config.create_runtime([1.0, 0.9, 0.8, 0.7, 0.0])


def _signature(shape=(8, 16), dtype="torch.float16", device="cuda:0", layout=(77, 8, 32, 32, 16)):
    return shape, dtype, device, layout, ((0, 77, "text"), (77, 85, "video"))


def _store_full(runtime, key, sigma, signature=None, rank=0):
    signature = signature or _signature()
    plan = runtime.plan(key, sigma, signature, 50, rank)
    assert plan.mode == "FULL"
    runtime.store_residual(key, signature, torch.ones(8, 16))
    return plan


def test_disabled_config_keeps_baseline_path():
    runtime = _runtime(cache_depth=0.0)
    assert not runtime.enabled

    source = FORWARD_PATH.read_text()
    assert 'block_cache_config.get("enabled", False) is not True' in source
    assert "block_plan = None\n        blocks = list(self.blocks)" in source


def test_full_cache_cache_full_sequence_and_counts():
    runtime = _runtime()
    key = ("cond",)
    _store_full(runtime, key, 1.0)

    assert runtime.plan(key, 0.9, _signature(), 50, 0).mode == "CACHE"
    assert runtime.plan(key, 0.8, _signature(), 50, 0).mode == "CACHE"
    assert runtime.plan(key, 0.7, _signature(), 50, 0).mode == "FULL"
    assert (runtime.full_steps, runtime.cache_steps) == (2, 2)
    assert runtime.executed_blocks == 126
    assert runtime.avoided_blocks == 74


def test_start_end_window():
    runtime = _runtime(start_percent=0.5, end_percent=0.75)
    key = ("cond",)
    _store_full(runtime, key, 1.0)
    plan = runtime.plan(key, 0.9, _signature(), 50, 0)
    assert plan.mode == "FULL"
    runtime.store_residual(key, _signature(), torch.ones(8, 16))
    assert runtime.plan(key, 0.8, _signature(), 50, 0).mode == "CACHE"


def test_sigma_threshold_forces_full():
    runtime = _runtime(sigma_threshold=0.05)
    key = ("cond",)
    _store_full(runtime, key, 1.0)
    assert runtime.plan(key, 0.9, _signature(), 50, 0).mode == "FULL"


def test_max_cached_steps_forces_refresh():
    runtime = _runtime(max_cached_steps=1)
    key = ("cond",)
    _store_full(runtime, key, 1.0)
    assert runtime.plan(key, 0.9, _signature(), 50, 0).mode == "CACHE"
    assert runtime.plan(key, 0.8, _signature(), 50, 0).mode == "FULL"


def test_conditioning_uuids_have_separate_residuals():
    runtime = _runtime()
    _store_full(runtime, ("positive",), 1.0)
    assert runtime.plan(("negative",), 0.9, _signature(), 50, 0).mode == "FULL"
    assert runtime.plan(("positive",), 0.9, _signature(), 50, 0).mode == "CACHE"


@pytest.mark.parametrize(
    "changed_signature",
    [
        _signature(shape=(4, 16)),
        _signature(layout=(77, 16, 32, 32, 16)),
        _signature(dtype="torch.bfloat16"),
        _signature(device="cuda:1"),
    ],
)
def test_cache_invalidates_for_shard_metadata(changed_signature):
    runtime = _runtime()
    key = ("cond",)
    _store_full(runtime, key, 1.0)
    assert runtime.plan(key, 0.9, changed_signature, 50, 0).mode == "FULL"


def test_rank_local_runtimes_make_the_same_decisions():
    runtimes = [_runtime(), _runtime()]
    modes = []
    for rank, runtime in enumerate(runtimes):
        key = ("cond",)
        first = _store_full(runtime, key, 1.0, rank=rank)
        second = runtime.plan(key, 0.9, _signature(), 50, rank)
        modes.append((first.mode, second.mode, second.prefix, runtime.full_steps, runtime.cache_steps))
    assert modes[0] == modes[1] == ("FULL", "CACHE", 13, 1, 1)


def test_cache_path_prefetches_only_selected_prefix():
    tree = ast.parse(FORWARD_PATH.read_text())
    forward = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "usp_dit_forward")
    calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)]
    prefetch = next(
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "make_prefetch_queue"
    )
    assert isinstance(prefetch.args[0], ast.Name)
    assert prefetch.args[0].id == "blocks"
    assert "list(self.blocks)[:block_plan.prefix]" in FORWARD_PATH.read_text()


def _load_sample_wrapper(runtime=None):
    tree = ast.parse(NODES_PATH.read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "minimax_h3_block_cache_sample_wrapper")
    module = ast.Module(body=[wrapper], type_ignores=[])
    namespace = {
        "CONFIG_KEY": block_cache.CONFIG_KEY,
        "RUNTIME_KEY": block_cache.RUNTIME_KEY,
        "MiniMaxH3BlockCacheConfig": (
            (lambda *args, **kwargs: types.SimpleNamespace(create_runtime=lambda sigmas: runtime))
            if runtime is not None else block_cache.MiniMaxH3BlockCacheConfig
        ),
        "comfy": types.SimpleNamespace(
            model_patcher=types.SimpleNamespace(
                create_model_options_clone=lambda options: {
                    **options,
                    "transformer_options": options["transformer_options"].copy(),
                }
            )
        ),
    }
    exec(compile(module, str(NODES_PATH), "exec"), namespace)
    return namespace["minimax_h3_block_cache_sample_wrapper"]


def test_sampling_exception_always_clears_runtime():
    runtime = _runtime()

    guider = types.SimpleNamespace(
        model_options={"transformer_options": {block_cache.CONFIG_KEY: runtime.config.to_dict()}}
    )

    class Executor:
        class_obj = guider

        @staticmethod
        def __call__(*args, **kwargs):
            raise RuntimeError("sampling failed")

    runtime.streams[("cond",)] = object()
    original_options = guider.model_options
    with pytest.raises(RuntimeError, match="sampling failed"):
        _load_sample_wrapper(runtime)(Executor(), None, None, None, [1.0, 0.0])
    assert runtime.streams == {}
    assert runtime.full_steps == runtime.cache_steps == 0
    assert runtime.executed_blocks == runtime.avoided_blocks == 0
    assert guider.model_options is original_options


@pytest.mark.parametrize("cache_name", ["easycache", "teacache"])
def test_other_caches_are_rejected(cache_name):
    guider = types.SimpleNamespace(
        model_options={"transformer_options": {block_cache.CONFIG_KEY: _runtime().config.to_dict(), cache_name: object()}}
    )
    executor = types.SimpleNamespace(class_obj=guider)
    with pytest.raises(ValueError, match="cannot be used"):
        _load_sample_wrapper()(executor)
