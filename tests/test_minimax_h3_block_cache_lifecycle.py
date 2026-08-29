import ast
import importlib.util
import types
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
BLOCK_CACHE_PATH = ROOT / "src/raylight/diffusion_models/minimax/block_cache.py"
FORWARD_PATH = ROOT / "src/raylight/diffusion_models/minimax/xdit_context_parallel.py"
WORKER_PATH = ROOT / "src/raylight/distributed_worker/ray_worker.py"


def _load_block_cache():
    spec = importlib.util.spec_from_file_location("minimax_h3_block_cache_lifecycle_test", BLOCK_CACHE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker_configure_method():
    tree = ast.parse(WORKER_PATH.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "configure_minimax_h3_block_cache")
    method.decorator_list = []
    namespace = {
        "H3_BLOCK_CACHE_CONFIG_KEY": "minimax_h3_block_cache_config",
        "H3_BLOCK_CACHE_RUNTIME_KEY": "minimax_h3_block_cache",
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(WORKER_PATH), "exec"), namespace)
    return namespace["configure_minimax_h3_block_cache"]


def _block_loop(comfy, config_key, runtime_key):
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
        "CONFIG_KEY": config_key,
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
    @staticmethod
    def make_prefetch_queue(blocks, device, transformer_options):
        return object()

    @staticmethod
    def prefetch_queue_pop(queue, device, block):
        return


def test_persistent_worker_on_off_on_has_no_stale_cache(monkeypatch):
    from raylight.distributed_worker import sampling_config

    block_cache = _load_block_cache()
    configure = _worker_configure_method()
    worker = types.SimpleNamespace(model=types.SimpleNamespace(model_options={"transformer_options": {}}))

    class RemoteConfigure:
        @staticmethod
        def remote(config):
            configure(worker, config)
            return True

    class RemoteSLAConfigure:
        # the dispatch also pushes the SLA config; this test only exercises the block cache
        @staticmethod
        def remote(config):
            return True

    actor = types.SimpleNamespace(
        configure_minimax_h3_block_cache=RemoteConfigure(),
        configure_minimax_h3_sla=RemoteSLAConfigure(),
    )
    monkeypatch.setattr(sampling_config.ray, "get", lambda values: values)
    base_actors = {"workers": [actor]}
    enabled_config = block_cache.MiniMaxH3BlockCacheConfig(0.12, 0.0, 1.0, 2, 0.5).to_dict()
    enabled_actors = {**base_actors, block_cache.ACTORS_CONFIG_KEY: enabled_config}

    def sample(ray_actors):
        sampling_config.configure_sampling_features(ray_actors)
        config = worker.model.model_options["transformer_options"][block_cache.CONFIG_KEY]
        runtime_config = block_cache.MiniMaxH3BlockCacheConfig(
            config["sigma_threshold"],
            config["start_percent"],
            config["end_percent"],
            config["max_cached_steps"],
            config["cache_depth"],
        )
        runtime = runtime_config.create_runtime([1.0, 0.9, 0.0])
        blocks = [_Block(index + 1) for index in range(4)]
        model = types.SimpleNamespace(blocks=blocks)
        layout = types.SimpleNamespace(signature=(1, 1, 1, 1, 1), segments=((0, 1, "video"),))
        options = {
            "uuids": ["cond"],
            block_cache.CONFIG_KEY: dict(config),
            block_cache.RUNTIME_KEY: runtime,
        }
        run = _block_loop(types.SimpleNamespace(model_prefetch=_Prefetch()), block_cache.CONFIG_KEY, block_cache.RUNTIME_KEY)
        run(model, torch.tensor(0.0), options, "cuda", None, None, None, layout, 0.0)
        run(model, torch.tensor(0.0), options, "cuda", None, None, None, layout, 0.1)
        calls = [block.calls for block in blocks]
        runtime.clear()
        options.pop(block_cache.RUNTIME_KEY)
        assert runtime.streams == {}
        assert runtime.full_steps == runtime.cache_steps == 0
        return config["enabled"], calls

    assert sample(enabled_actors) == (True, [2, 2, 1, 1])
    assert sample(base_actors) == (False, [2, 2, 2, 2])
    assert sample(enabled_actors) == (True, [2, 2, 1, 1])
