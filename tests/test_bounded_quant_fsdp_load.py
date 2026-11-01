import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]
RAYLIGHT = ROOT / "custom_nodes/raylight/src/raylight"
NODES = RAYLIGHT / "nodes.py"
WORKER = RAYLIGHT / "distributed_worker/ray_worker.py"
FSDP_UTILS = RAYLIGHT / "comfy_dist/fsdp_utils.py"
MODEL_PATCHER = RAYLIGHT / "comfy_dist/model_patcher.py"
KITCHEN = RAYLIGHT / "comfy_dist/kitchen_distributed.py"


def _function(path, function_name):
    return next(node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.FunctionDef) and node.name == function_name)


def test_quant_fsdp_workers_load_sequentially_and_materialize_locally():
    loader_source = ast.unparse(_function(NODES, "load_ray_unet"))
    worker_source = ast.unparse(_function(WORKER, "load_unet"))

    assert loader_source.count("if ray_actors['config'].loading.parallel_quant_load:") == 2
    assert loader_source.count("ray.get(future)") == 2
    assert worker_source.index("self.set_state_dict()") < worker_source.index("self._patch_fsdp_for_sampling()")


def test_quant_fsdp_uses_private_spec_and_releases_consumed_state():
    loader_source = ast.unparse(_function(FSDP_UTILS, "load_from_full_model_state_dict"))
    release_source = ast.unparse(_function(FSDP_UTILS, "_release_quant_keys"))
    patcher_source = ast.unparse(_function(MODEL_PATCHER, "patch_fsdp"))

    assert "DTensor(quant_tensor, sharded_meta_param._spec, requires_grad=sharded_meta_param.requires_grad)" in loader_source
    assert "_release_quant_keys(full_sd, param_name, keep=own_params)" in loader_source
    assert "release_sd=True" in patcher_source
    assert "input_scale" not in release_source
    assert "scale_input" not in release_source


def test_kitchen_patches_are_restored_after_quant_fsdp_initialization():
    context_source = ast.unparse(_function(KITCHEN, "temporary_sitepkg_ck_patches"))
    patcher_source = ast.unparse(_function(MODEL_PATCHER, "patch_fsdp"))

    assert "try:\n        yield\n    finally:\n        restore_sitepkg_ck_patches(layouts=layouts)" in context_source
    assert "temporary_sitepkg_ck_patches() if use_quant_loader else nullcontext()" in patcher_source
    assert "with patch_context:" in patcher_source
    assert patcher_source.index("load_from_full_model_state_dict") < patcher_source.index("_pre_init_fsdp")
