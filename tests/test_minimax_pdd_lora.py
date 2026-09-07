import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]
RAYLIGHT = ROOT / "custom_nodes/raylight/src/raylight"
LORA = RAYLIGHT / "comfy_dist/lora.py"
MODEL_PATCHER = RAYLIGHT / "comfy_dist/model_patcher.py"
SD = RAYLIGHT / "comfy_dist/sd.py"
ADAPTER_BASE = RAYLIGHT / "comfy_dist/weight_adapter/base.py"
LORA_ADAPTER = RAYLIGHT / "comfy_dist/weight_adapter/lora.py"


def _function(path, function_name, class_name=None):
    nodes = ast.parse(path.read_text()).body
    if class_name is not None:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == function_name)


def test_custom_lora_reports_reshape_target():
    base_source = ast.unparse(_function(ADAPTER_BASE, "calculate_shape", "WeightAdapterBase"))
    adapter_source = ast.unparse(_function(LORA_ADAPTER, "calculate_shape", "LoRAAdapter"))
    calculate_source = ast.unparse(_function(LORA, "calculate_shape"))

    assert "return None" in base_source
    assert "self.weights[5]" in adapter_source
    assert "v.calculate_shape(key)" in calculate_source


def test_fsdp_expands_shape_changing_parameters_before_sharding():
    expansion_source = ast.unparse(_function(MODEL_PATCHER, "_expand_shape_changing_patches"))
    patch_source = ast.unparse(_function(MODEL_PATCHER, "patch_fsdp"))
    load_source = ast.unparse(_function(MODEL_PATCHER, "load", "FSDPModelPatcher"))

    assert "fsdp_state_dict[key] = comfy_dist.lora.pad_tensor_to_shape" in expansion_source
    assert "fsdp_state_dict[bias_key] = comfy_dist.lora.pad_tensor_to_shape" in expansion_source
    assert patch_source.index("_expand_shape_changing_patches(self)") < patch_source.index("fully_shard_bottom_up")
    assert load_source.count("comfy_dist.lora.calculate_shape") == 2


def test_quantized_reshape_lora_uses_merged_replacement_forward():
    loader_source = ast.unparse(_function(SD, "load_lora_for_models_quantized"))
    adapter_source = ast.unparse(_function(SD, "_patched_bias", "MergedWeightBypassAdapter"))
    merge_source = ast.unparse(_function(SD, "_merge_replacement_entries"))

    assert "reshape_modules" in loader_source
    assert "group['weight_patches'].append((strength_model, patch_data, 1.0, offset, function))" in loader_source
    assert "group[f'{param_name}_patches'].append((strength_model, patch_data, 1.0, offset, function))" in loader_source
    assert "isinstance(entries[0]['adapter'], MergedWeightBypassAdapter)" in loader_source
    assert "comfy_dist.lora.pad_tensor_to_shape" in adapter_source
    assert "weight_patches.extend(adapter.weight_patches)" in merge_source
    assert "MergedWeightBypassAdapter(module_key, weight_patches, bias_patches)" in merge_source
