import importlib.util
from pathlib import Path


BLOCK_CACHE_PATH = Path(__file__).parents[1] / "src/raylight/diffusion_models/minimax/block_cache.py"


def test_debug_logs_explicit_disabled_config(capsys):
    spec = importlib.util.spec_from_file_location("minimax_h3_block_cache_disabled_logging_test", BLOCK_CACHE_PATH)
    block_cache = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(block_cache)
    runtime = block_cache.MiniMaxH3BlockCacheConfig(0.0, 0.0, 1.0, 0, 0.0, True).create_runtime([1.0, 0.0])

    runtime.log_start(rank=0)

    assert capsys.readouterr().out.splitlines() == [
        "[H3 Block Cache][rank0] enabled=False threshold=0.00 start=0.00 end=1.00 mcs=0 depth=0.00"
    ]
