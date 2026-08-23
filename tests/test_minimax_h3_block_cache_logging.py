import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
BLOCK_CACHE_PATH = ROOT / "src/raylight/diffusion_models/minimax/block_cache.py"


def _load_block_cache():
    spec = importlib.util.spec_from_file_location("minimax_h3_block_cache_logging_test", BLOCK_CACHE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime(debug):
    block_cache = _load_block_cache()
    config = block_cache.MiniMaxH3BlockCacheConfig(0.12, 0.0, 1.0, 2, 0.75, debug)
    return config.create_runtime([1.0, 0.9, 0.8, 0.7, 0.0])


def _signature():
    return (8, 16), "torch.float16", "cuda:0", (77, 8, 32, 32, 16), ((0, 77, "text"),)


def test_debug_prints_worker_config_and_each_step(capsys):
    runtime = _runtime(debug=True)
    key = ("cond",)

    first = runtime.plan(key, 1.0, _signature(), 50, 0)
    runtime.store_residual(key, _signature(), torch.ones(8, 16))
    second = runtime.plan(key, 0.9, _signature(), 50, 0)
    third = runtime.plan(key, 0.8, _signature(), 50, 0)
    fourth = runtime.plan(key, 0.7, _signature(), 50, 0)
    runtime.log_summary()

    assert [first.mode, second.mode, third.mode, fourth.mode] == ["FULL", "CACHE", "CACHE", "FULL"]
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[H3 Block Cache][rank0] enabled=True threshold=0.12 start=0.00 end=1.00 mcs=2 depth=0.75"
    assert "step=0" in lines[1] and "mode=FULL" in lines[1] and "prefix_blocks=50/50" in lines[1]
    assert "cached_steps_count=0" in lines[1] and "residual_valid=False" in lines[1]
    assert "step=1" in lines[2] and "mode=CACHE" in lines[2] and "prefix_blocks=13/50" in lines[2]
    assert "cached_steps_count=1" in lines[2] and "residual_valid=True" in lines[2]
    assert "step=2" in lines[3] and "mode=CACHE" in lines[3] and "cached_steps_count=2" in lines[3]
    assert "step=3" in lines[4] and "mode=FULL" in lines[4] and "prefix_blocks=50/50" in lines[4]
    assert "cached_steps_count=0" in lines[4] and "residual_valid=True" in lines[4]
    assert lines[5] == "[H3 Block Cache] FULL=2 CACHE=2 effective_blocks=126 skipped_blocks=74"


def test_debug_false_hides_worker_and_step_messages_but_keeps_summary(capsys):
    runtime = _runtime(debug=False)
    key = ("cond",)

    runtime.plan(key, 1.0, _signature(), 50, 0)
    runtime.log_summary()

    assert capsys.readouterr().out.splitlines() == [
        "[H3 Block Cache] FULL=1 CACHE=0 effective_blocks=50 skipped_blocks=0"
    ]


def test_worker_start_log_is_emitted_once(capsys):
    runtime = _runtime(debug=True)

    runtime.log_start(rank=1)
    runtime.log_start(rank=1)

    assert capsys.readouterr().out.splitlines() == [
        "[H3 Block Cache][rank1] enabled=True threshold=0.12 start=0.00 end=1.00 mcs=2 depth=0.75"
    ]
