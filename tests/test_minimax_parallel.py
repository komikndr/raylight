import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]
CORE = ROOT / "comfy/ldm/minimax/model.py"
RAYLIGHT = ROOT / "custom_nodes/raylight/src/raylight/diffusion_models/minimax/xdit_context_parallel.py"
USP = ROOT / "custom_nodes/raylight/src/raylight/distributed_modules/usp.py"


def _module_symbols(path):
    symbols = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            symbols.update(target.id for target in targets if isinstance(target, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            symbols.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    return symbols


def _function(path, function_name, class_name=None):
    nodes = ast.parse(path.read_text()).body
    if class_name is not None:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == function_name)


def _nested_function(path, function_name):
    return next(node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.FunctionDef) and node.name == function_name)


def _call_name(call):
    parts = []
    node = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _signature(function):
    args = function.args
    parameters = []

    def add(kind, argument, default=None):
        parameters.append((kind, argument.arg, ast.dump(default, include_attributes=False) if default is not None else None))

    for argument in args.posonlyargs:
        add("positional-only", argument)
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
    for argument, default in zip(args.args, defaults):
        add("positional", argument, default)
    if args.vararg is not None:
        add("var-positional", args.vararg)
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        add("keyword-only", argument, default)
    if args.kwarg is not None:
        add("var-keyword", args.kwarg)
    return parameters


def test_minimax_core_imports_exist():
    imports = next(
        node.names
        for node in ast.parse(RAYLIGHT.read_text()).body
        if isinstance(node, ast.ImportFrom) and node.module == "comfy.ldm.minimax.model"
    )
    missing = {alias.name for alias in imports} - _module_symbols(CORE)
    assert not missing


def test_minimax_usp_forward_signature_matches_core():
    assert _signature(_function(RAYLIGHT, "usp_dit_forward")) == _signature(
        _function(CORE, "_forward", "MiniMaxH3Model")
    )


def test_minimax_usp_layout_and_mask_parity():
    function = _function(RAYLIGHT, "usp_dit_forward")
    packed_layout = next(node for node in ast.walk(function) if isinstance(node, ast.Call) and _call_name(node) == "PackedLayout")
    source = ast.unparse(function)

    assert all(keyword.arg != "frame_count" for keyword in packed_layout.keywords)
    for name in ("cond_audio", "mask_row_values", "video_rows_t", "audio_rows_t", "rows_to_mod_index"):
        assert name in source


def test_minimax_usp_slices_per_row_modulation_with_sequence():
    source = ast.unparse(_function(RAYLIGHT, "_split_packed_sequence"))

    assert "isinstance(row, torch.Tensor)" in source
    assert "segment_start - original_start" in source
    assert "segment_end - original_start" in source


def test_minimax_usp_attention_matches_core_kernels():
    function = _function(RAYLIGHT, "usp_attn_forward")
    calls = {_call_name(node) for node in ast.walk(function) if isinstance(node, ast.Call)}
    attention_call = next(node for node in ast.walk(function) if isinstance(node, ast.Call) and _call_name(node) == "xfuser_optimized_attention")

    assert "comfy.quant_ops.ck.rms_rope_split_half" in calls
    assert "comfy.quant_ops.ck.rms_rope_split_half_" in calls
    assert "v.clone" in calls
    assert any(keyword.arg == "transformer_options" for keyword in attention_call.keywords)


def test_minimax_lora_mlp_uses_module_forward():
    function = _function(RAYLIGHT, "usp_mlp_forward")
    calls = {_call_name(node) for node in ast.walk(function) if isinstance(node, ast.Call)}

    assert "self.fc1" in calls
    assert "torch.nn.functional.silu" in calls
    assert "self.fc2" in calls


def test_minimax_usp_routes_sidecar_fc2_lora():
    function = _nested_function(USP, "_inject_minimax_h3")
    source = ast.unparse(function)

    assert "FSDP_LORA_SIDECAR_ATTACHMENT" in source
    assert "diffusion_model.blocks." in source
    assert "diffusion_model.token_refiner.blocks." in source
    assert source.count("usp_mlp_forward") >= 2
