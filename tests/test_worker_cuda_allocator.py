import ast
from pathlib import Path


NODES_PATH = Path(__file__).parents[1] / "src/raylight/nodes.py"


def _function_source(name: str) -> str:
    tree = ast.parse(NODES_PATH.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.unparse(function)


def test_local_ray_workers_do_not_override_cuda_allocator_configuration():
    source = NODES_PATH.read_text()
    runtime_env_source = _function_source("_build_local_runtime_env")

    assert "_sanitized_worker_alloc_conf" not in source
    assert "RAYLIGHT_KEEP_CUDA_MALLOC_ASYNC" not in source
    assert "PYTORCH_CUDA_ALLOC_CONF" not in runtime_env_source
