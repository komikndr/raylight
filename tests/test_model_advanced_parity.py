import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]
MODEL_ADVANCED = ROOT / "custom_nodes/raylight/src/raylight/comfy_extra_dist/nodes_model_advanced.py"


def _method(path, class_name, function_name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == function_name)
    raise AssertionError(f"Missing {class_name}.{function_name}")


def test_ray_rescale_cfg_handles_flow_models_in_x0_space():
    source = ast.unparse(_method(MODEL_ADVANCED, "RayRescaleCFG", "patch"))

    assert "model.get_model_object('model_sampling')" in source or 'model.get_model_object("model_sampling")' in source
    assert "isinstance(model_sampling, comfy.model_sampling.CONST)" in source
    assert "cond_denoised" in source
    assert "uncond_denoised" in source
    assert "tuple(range(1, x_0_cond.ndim))" in source
    assert "clamp(min=1e-08)" in source
    assert "return x_orig - x_0_final" in source


def test_ray_rescale_cfg_preserves_non_flow_path():
    source = ast.unparse(_method(MODEL_ADVANCED, "RayRescaleCFG", "patch"))

    assert "args['cond']" in source
    assert "args['uncond']" in source
    assert "args['sigma']" in source
    assert "x_orig / (sigma * sigma + 1.0)" in source
    assert "dim=(1, 2, 3)" in source
