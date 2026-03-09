from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule

ROOT = Path(__file__).resolve().parents[3]
RAYLIGHT_SRC = ROOT / "custom_nodes" / "raylight" / "src"
KITCHEN_REPO = Path("/home/kxn/comfy-kitchen-distributed")
for candidate in (str(RAYLIGHT_SRC), str(KITCHEN_REPO), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from comfy_kitchen.tensor.base import QuantizedTensor

PAGESIZE = os.sysconf("SC_PAGE_SIZE")

LAYOUTS = {
    "fp8": {
        "layout_name": "TensorCoreFP8Layout",
        "format": "float8_e4m3fn",
        "patch_path": RAYLIGHT_SRC / "raylight" / "comfy_dist" / "kitchen_patches" / "fp8.py.unused",
        "install_name": "install_fp8_patches",
        "restore_name": "restore_fp8_patches",
    },
    "nvfp4": {
        "layout_name": "TensorCoreNVFP4Layout",
        "format": "nvfp4",
        "patch_path": RAYLIGHT_SRC / "raylight" / "comfy_dist" / "kitchen_patches" / "nvfp4.py",
        "install_name": "install_nvfp4_patches",
        "restore_name": "restore_nvfp4_patches",
    },
}


def _load_module(name: str, file_path: Path):
    loader = SourceFileLoader(name, str(file_path))
    spec = spec_from_file_location(name, file_path, loader=loader)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {file_path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_fsdp_utils = _load_module("raylight_fsdp_utils_compare", RAYLIGHT_SRC / "raylight" / "comfy_dist" / "fsdp_utils.py")
fully_shard_bottom_up = _fsdp_utils.fully_shard_bottom_up
load_from_full_model_state_dict = _fsdp_utils.load_from_full_model_state_dict


def rss_mb() -> float:
    with open("/proc/self/statm", "r", encoding="utf-8") as f:
        rss_pages = int(f.read().split()[1])
    return rss_pages * PAGESIZE / (1024 * 1024)


class InputScaledQuantLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, layout_name: str, bias: bool = True):
        super().__init__()
        self.layout_name = layout_name
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), dtype=torch.bfloat16))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((out_features,), dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)
        self.input_scale = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, QuantizedTensor):
            x = QuantizedTensor.from_float(x.to(dtype=torch.bfloat16), self.layout_name, scale=self.input_scale)
        return torch.nn.functional.linear(x, self.weight, self.bias)


class QuantMLP(torch.nn.Module):
    def __init__(self, dim: int, num_layers: int, layout_name: str, bias: bool, activation: bool):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [InputScaledQuantLinear(dim, dim, layout_name=layout_name, bias=bias) for _ in range(num_layers)]
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if self.activation and idx + 1 < len(self.layers):
                x = torch.nn.functional.gelu(x)
        return x


@dataclass
class BuildResult:
    model: torch.nn.Module
    full_sd: dict[str, Any]
    expected_input_scales: dict[str, torch.Tensor]
    num_layers: int
    approx_weight_bytes: int


@dataclass
class PatchFns:
    install: Any
    restore: Any


@dataclass
class QuantPayload:
    values: dict[str, Any]
    approx_bytes: int



def _clone_quantized_tensor(qt: QuantizedTensor) -> QuantizedTensor:
    params = qt._params
    cloned_params = replace(
        params,
        scale=params.scale.clone() if isinstance(params.scale, torch.Tensor) else params.scale,
    )
    if hasattr(params, "block_scale"):
        cloned_params = replace(cloned_params, block_scale=params.block_scale.clone())
    return QuantizedTensor(qt._qdata.clone(), qt._layout_cls, cloned_params)



def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())



def _build_payload(layout_key: str, prefix: str, weight: torch.Tensor) -> QuantPayload:
    cfg = LAYOUTS[layout_key]
    qt = QuantizedTensor.from_float(weight, cfg["layout_name"])

    if layout_key == "fp8":
        values: dict[str, Any] = {f"{prefix}.weight": _clone_quantized_tensor(qt)}
        approx_bytes = _tensor_bytes(qt._qdata) + _tensor_bytes(qt._params.scale)
        return QuantPayload(values=values, approx_bytes=approx_bytes)

    values = {key: value.detach().clone() for key, value in qt.state_dict(prefix=f"{prefix}.weight").items()}
    values[f"{prefix}.comfy_quant"] = json.dumps({"format": cfg["format"]})
    approx_bytes = 0
    for value in values.values():
        if isinstance(value, torch.Tensor):
            approx_bytes += _tensor_bytes(value)
        elif isinstance(value, str):
            approx_bytes += len(value.encode("utf-8"))
    return QuantPayload(values=values, approx_bytes=approx_bytes)



def _build_case(layout_key: str, mode: str, target_gib: float, dim: int) -> BuildResult:
    cfg = LAYOUTS[layout_key]
    if mode == "tiny":
        num_layers = 3
        bias = True
        activation = True
    else:
        bias = False
        activation = False
        base_weight = torch.randn((dim, dim), dtype=torch.bfloat16)
        base_payload = _build_payload(layout_key, "layer_template", base_weight)
        per_layer_bytes = base_payload.approx_bytes + _tensor_bytes(torch.ones((1,), dtype=torch.float32))
        target_bytes = int(target_gib * (1024 ** 3))
        num_layers = max(1, int(target_bytes // per_layer_bytes))

    model = QuantMLP(dim=dim, num_layers=num_layers, layout_name=cfg["layout_name"], bias=bias, activation=activation).to("meta")
    full_sd: dict[str, Any] = {}
    expected_input_scales: dict[str, torch.Tensor] = {}
    approx_weight_bytes = 0

    if mode == "tiny":
        for idx in range(num_layers):
            prefix = f"layers.{idx}"
            weight = torch.ones((dim, dim), dtype=torch.bfloat16) * (idx + 1)
            payload = _build_payload(layout_key, prefix, weight)
            full_sd.update(payload.values)
            approx_weight_bytes += payload.approx_bytes
            full_sd[f"{prefix}.input_scale"] = torch.tensor([1.0 + idx], dtype=torch.float32)
            expected_input_scales[f"{prefix}.input_scale"] = full_sd[f"{prefix}.input_scale"].clone()
            approx_weight_bytes += _tensor_bytes(full_sd[f"{prefix}.input_scale"])
            if bias:
                full_sd[f"{prefix}.bias"] = torch.zeros((dim,), dtype=torch.bfloat16)
                approx_weight_bytes += _tensor_bytes(full_sd[f"{prefix}.bias"])
    else:
        base_weight = torch.randn((dim, dim), dtype=torch.bfloat16)
        base_payload = _build_payload(layout_key, "layer_template", base_weight)
        for idx in range(num_layers):
            prefix = f"layers.{idx}"
            for key, value in base_payload.values.items():
                target_key = key.replace("layer_template", prefix, 1)
                if isinstance(value, QuantizedTensor):
                    full_sd[target_key] = _clone_quantized_tensor(value)
                elif isinstance(value, torch.Tensor):
                    full_sd[target_key] = value.clone()
                else:
                    full_sd[target_key] = value
            approx_weight_bytes += base_payload.approx_bytes
            full_sd[f"{prefix}.input_scale"] = torch.tensor([1.0], dtype=torch.float32)
            expected_input_scales[f"{prefix}.input_scale"] = full_sd[f"{prefix}.input_scale"].clone()
            approx_weight_bytes += _tensor_bytes(full_sd[f"{prefix}.input_scale"])

    return BuildResult(
        model=model,
        full_sd=full_sd,
        expected_input_scales=expected_input_scales,
        num_layers=num_layers,
        approx_weight_bytes=approx_weight_bytes,
    )



def _meta_params(model: torch.nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if getattr(param, "is_meta", False)]



def _input_scale_report(model: torch.nn.Module, expected: dict[str, torch.Tensor]) -> dict[str, Any]:
    meta = []
    missing = []
    mismatches = []
    found = 0
    for name, param in model.named_parameters():
        if not name.endswith("input_scale"):
            continue
        found += 1
        if getattr(param, "is_meta", False):
            meta.append(name)
            continue
        expected_value = expected.get(name)
        if expected_value is None:
            missing.append(name)
            continue
        if not torch.allclose(param.detach().cpu(), expected_value.detach().cpu()):
            mismatches.append(
                {
                    "name": name,
                    "actual": param.detach().cpu().tolist(),
                    "expected": expected_value.detach().cpu().tolist(),
                }
            )
    return {
        "expected": len(expected),
        "found": found,
        "meta": meta,
        "missing": missing,
        "mismatches": mismatches[:8],
    }



def _weight_summary(model: torch.nn.Module) -> dict[str, Any]:
    weight = model.layers[0].weight
    summary = {
        "type": type(weight).__name__,
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
    }
    local = getattr(weight, "_local_tensor", None)
    if local is not None:
        summary["local_type"] = type(local).__name__
        summary["local_shape"] = list(local.shape)
        summary["local_dtype"] = str(local.dtype)
    qt = weight if isinstance(weight, QuantizedTensor) else local if isinstance(local, QuantizedTensor) else None
    if qt is not None:
        summary["qdata_shape"] = list(qt._qdata.shape)
        summary["qdata_dtype"] = str(qt._qdata.dtype)
        summary["orig_shape"] = list(qt._params.orig_shape)
        if isinstance(qt._params.scale, torch.Tensor):
            summary["scale_shape"] = list(qt._params.scale.shape)
            summary["scale_dtype"] = str(qt._params.scale.dtype)
        if hasattr(qt._params, "block_scale"):
            summary["block_scale_shape"] = list(qt._params.block_scale.shape)
            summary["block_scale_dtype"] = str(qt._params.block_scale.dtype)
    return summary



def _load_patch_fns(layout_key: str) -> PatchFns:
    cfg = LAYOUTS[layout_key]
    module = _load_module(f"raylight_patch_{layout_key}_compare", cfg["patch_path"])
    return PatchFns(install=getattr(module, cfg["install_name"]), restore=getattr(module, cfg["restore_name"]))



def run_case(layout_key: str, mode: str, target_gib: float, dim: int) -> dict[str, Any]:
    built = _build_case(layout_key=layout_key, mode=mode, target_gib=target_gib, dim=dim)
    report: dict[str, Any] = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "layout": layout_key,
        "mode": mode,
        "dim": dim,
        "num_layers": built.num_layers,
        "approx_weight_gib": round(built.approx_weight_bytes / (1024 ** 3), 3),
        "rss_before_shard_mb": round(rss_mb(), 2),
    }

    fully_shard_bottom_up(built.model, fsdp_kwargs={"reshard_after_forward": True}, native_ignore_scale=False)
    report["rss_after_shard_mb"] = round(rss_mb(), 2)
    report["is_fsdp_module_after_shard"] = isinstance(built.model, FSDPModule)

    load_from_full_model_state_dict(
        built.model,
        built.full_sd,
        device=torch.device("cpu"),
        strict=False,
        cpu_offload=False,
        release_sd=True,
    )
    report["rss_after_load_mb"] = round(rss_mb(), 2)
    report["meta_params_after_load"] = _meta_params(built.model)
    report["input_scale_report"] = _input_scale_report(built.model, built.expected_input_scales)
    report["weight_summary"] = _weight_summary(built.model)

    if mode == "tiny":
        x = torch.ones((4, dim), dtype=torch.bfloat16)
        with torch.no_grad():
            y = built.model(x)
        report["rss_after_forward_mb"] = round(rss_mb(), 2)
        report["output_shape"] = list(y.shape)
        report["output_sum"] = float(y.float().sum().item())
    else:
        report["rss_after_forward_mb"] = None
        report["output_shape"] = None
        report["output_sum"] = None

    assert not report["meta_params_after_load"], report["meta_params_after_load"]
    assert not report["input_scale_report"]["meta"], report["input_scale_report"]
    assert not report["input_scale_report"]["missing"], report["input_scale_report"]
    assert not report["input_scale_report"]["mismatches"], report["input_scale_report"]
    return report



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=tuple(LAYOUTS.keys()), required=True)
    parser.add_argument("--mode", choices=("tiny", "memory"), default="tiny")
    parser.add_argument("--target-gib", type=float, default=2.0)
    parser.add_argument("--dim", type=int, default=1024)
    args = parser.parse_args()

    patch_fns = _load_patch_fns(args.layout)
    dist.init_process_group(backend="gloo")
    patch_fns.install()
    try:
        result = run_case(layout_key=args.layout, mode=args.mode, target_gib=args.target_gib, dim=args.dim)
        print(json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        patch_fns.restore()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
