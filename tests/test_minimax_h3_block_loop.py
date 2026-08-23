import ast
import importlib.util
import types
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
BLOCK_CACHE_PATH = ROOT / "src/raylight/diffusion_models/minimax/block_cache.py"
FORWARD_PATH = ROOT / "src/raylight/diffusion_models/minimax/xdit_context_parallel.py"


def _load_block_cache():
    spec = importlib.util.spec_from_file_location("minimax_h3_block_loop_cache_test", BLOCK_CACHE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_block_loop(comfy, runtime_key):
    tree = ast.parse(FORWARD_PATH.read_text())
    forward = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "usp_dit_forward")
    start = next(
        index for index, node in enumerate(forward.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "patches_replace"
    )
    end = next(
        index for index, node in enumerate(forward.body[start:], start)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "h"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "all_gather"
    )
    wrapper = ast.parse(
        "def run(self, h, transformer_options, device, t_emb, mod_segments, rope_freqs, layout, t_v):\n"
        "    pass\n"
    ).body[0]
    wrapper.body = forward.body[start:end] + [ast.Return(value=ast.Name(id="h", ctx=ast.Load()))]
    namespace = {
        "CONFIG_KEY": "minimax_h3_block_cache_config",
        "RUNTIME_KEY": runtime_key,
        "comfy": comfy,
        "get_sequence_parallel_rank": lambda: 0,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(FORWARD_PATH), "exec"), namespace)
    return namespace["run"]


class _Block:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self, h, t_emb, mod_segments, rope_freqs, transformer_options):
        self.calls += 1
        return h + self.value


class _Prefetch:
    def __init__(self):
        self.created = []
        self.popped = []

    def make_prefetch_queue(self, blocks, device, transformer_options):
        self.created.append(list(blocks))
        return object()

    def prefetch_queue_pop(self, queue, device, block):
        self.popped.append(block)


def _inputs(config):
    cache = config.create_runtime([1.0, 0.9, 0.0])
    prefetch = _Prefetch()
    comfy = types.SimpleNamespace(model_prefetch=prefetch)
    blocks = [_Block(index + 1) for index in range(4)]
    model = types.SimpleNamespace(blocks=blocks)
    layout = types.SimpleNamespace(signature=(1, 1, 1, 1, 1), segments=((0, 1, "video"),))
    options = {"uuids": ["cond"], "minimax_h3_block_cache_config": config.to_dict(), "minimax_h3_block_cache": cache}
    return cache, prefetch, comfy, model, layout, options


def test_disabled_block_cache_is_numerically_identical_to_baseline_loop():
    block_cache = _load_block_cache()
    config = block_cache.MiniMaxH3BlockCacheConfig(0.12, 0.0, 1.0, 2, 0.0)
    cache, prefetch, comfy, model, layout, options = _inputs(config)
    run = _load_block_loop(comfy, block_cache.RUNTIME_KEY)

    output = run(model, torch.tensor(0.0), options, "cuda", None, None, None, layout, 0.0)

    assert torch.equal(output, torch.tensor(10.0))
    assert [block.calls for block in model.blocks] == [1, 1, 1, 1]
    assert prefetch.created == [model.blocks]
    assert cache.streams == {}


def test_cache_step_runs_and_prefetches_only_the_prefix():
    block_cache = _load_block_cache()
    config = block_cache.MiniMaxH3BlockCacheConfig(0.12, 0.0, 1.0, 2, 0.5)
    cache, prefetch, comfy, model, layout, options = _inputs(config)
    run = _load_block_loop(comfy, block_cache.RUNTIME_KEY)

    full = run(model, torch.tensor(0.0), options, "cuda", None, None, None, layout, 0.0)
    cached = run(model, torch.tensor(0.0), options, "cuda", None, None, None, layout, 0.1)

    assert torch.equal(full, torch.tensor(10.0))
    assert torch.equal(cached, torch.tensor(10.0))
    assert [block.calls for block in model.blocks] == [2, 2, 1, 1]
    assert prefetch.created == [model.blocks, model.blocks[:2]]
    assert cache.full_steps == 1
    assert cache.cache_steps == 1
