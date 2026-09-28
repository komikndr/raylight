import importlib
import importlib.util
from pathlib import Path

import torch

import comfy.ops
from comfy_kitchen.tensor import QuantizedTensor


RAYLIGHT = Path(__file__).parents[1] / "src/raylight/comfy_dist"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FP8 = _load("raylight_fp8_patch_test", RAYLIGHT / "kitchen_patches/fp8.py")


def test_quant_fsdp_keeps_unsharded_mixed_precision_weight():
    fsdp_utils = _load("raylight_fsdp_utils_test", RAYLIGHT / "fsdp_utils.py")
    model = torch.nn.Module()
    model.projector = comfy.ops.mixed_precision_ops().Linear(12, 1, bias=False, device="meta")
    model.projector.register_parameter("weight", torch.nn.Parameter(torch.empty((1, 12), device="meta", dtype=torch.bfloat16)))
    weight = torch.arange(12, dtype=torch.bfloat16).reshape(1, 12)
    full_sd = {"projector.weight": weight}

    fsdp_utils.load_from_full_model_state_dict(model, full_sd, torch.device("cpu"))

    assert model.projector.weight is not None
    assert model.projector.weight.device.type == "cpu"
    torch.testing.assert_close(model.projector.weight, weight)
    torch.testing.assert_close(model.projector(torch.ones(1, 12, dtype=torch.bfloat16)), weight.sum().reshape(1, 1))
    assert full_sd["projector.weight"] is None


def test_fp8_unaligned_mm_falls_back_without_scaled_mm(monkeypatch):
    a = torch.ones((2, 12), dtype=torch.bfloat16)
    b = torch.ones((12, 1), dtype=torch.bfloat16)
    layout = "TensorCoreFP8Layout"
    qa = QuantizedTensor.from_float(a, layout)
    qb = QuantizedTensor.from_float(b, layout)

    def unexpected_scaled_mm(*args, **kwargs):
        raise AssertionError("unaligned FP8 matmul reached scaled_mm")

    monkeypatch.setattr(importlib.import_module("comfy_kitchen.scaled_mm_v2"), "scaled_mm_v2", unexpected_scaled_mm)
    FP8.install_fp8_patches()
    try:
        result = torch.mm(qa, qb)
    finally:
        FP8.restore_fp8_patches()

    torch.testing.assert_close(result, torch.mm(qa.dequantize(), qb.dequantize()))


def test_fp8_aligned_mm_uses_scaled_mm(monkeypatch):
    qa = QuantizedTensor.from_float(torch.ones((16, 16), dtype=torch.bfloat16), "TensorCoreFP8Layout")
    qb = QuantizedTensor.from_float(torch.ones((16, 16), dtype=torch.bfloat16), "TensorCoreFP8Layout")
    calls = []

    def scaled_mm(a, b, **kwargs):
        calls.append((a.shape, b.shape))
        return torch.zeros((16, 16), dtype=torch.bfloat16)

    monkeypatch.setattr(importlib.import_module("comfy_kitchen.scaled_mm_v2"), "scaled_mm_v2", scaled_mm)
    FP8.install_fp8_patches()
    try:
        result = torch.mm(qa, qb)
    finally:
        FP8.restore_fp8_patches()

    assert calls == [(torch.Size((16, 16)), torch.Size((16, 16)))], (calls, result)
    torch.testing.assert_close(result, torch.zeros((16, 16), dtype=torch.bfloat16))
