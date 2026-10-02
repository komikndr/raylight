import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from comfy_kitchen.tensor import QuantizedTensor


ROOT = Path(__file__).resolve().parents[1] / "src/raylight/comfy_dist"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from comfy.ops import mixed_precision_ops


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loader = load_module("mxfp8_loader", ROOT / "fsdp_utils.py")
patch = load_module("mxfp8_patch", ROOT / "kitchen_patches/mxfp8.py")
spec = importlib.util.spec_from_file_location("raylight_kitchen_test", ROOT / "kitchen_distributed.py", submodule_search_locations=[str(ROOT)])
kitchen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kitchen
spec.loader.exec_module(kitchen)


class MXFP8FSDPTest(unittest.TestCase):
    def test_default_patch_lifecycle(self):
        original_pre = getattr(QuantizedTensor, "fsdp_pre_all_gather", None)
        original_post = getattr(QuantizedTensor, "fsdp_post_all_gather", None)
        for _ in range(2):
            with kitchen.temporary_sitepkg_ck_patches():
                self.assertIn("mxfp8", kitchen._SITEPKG_LAYOUT_PATCHERS)
                qt = QuantizedTensor.from_float(torch.ones(32, 32), "TensorCoreFP8Layout")
                inputs, metadata = qt.fsdp_pre_all_gather(None, qt.shape, qt.stride(), None, None)
                torch.testing.assert_close(inputs[0].float(), qt._qdata.float())
                gathered, _ = qt.fsdp_post_all_gather(inputs, metadata, torch.float32)
                torch.testing.assert_close(gathered.dequantize(), qt.dequantize())
            self.assertIs(getattr(QuantizedTensor, "fsdp_pre_all_gather", None), original_pre)
            self.assertIs(getattr(QuantizedTensor, "fsdp_post_all_gather", None), original_post)

    def test_distributed_forward(self):
        if not dist.is_initialized():
            self.skipTest("Run with torchrun to exercise distributed FSDP")
        mesh = init_device_mesh("cpu", (dist.get_world_size(),))
        patch.install_mxfp8_patches()
        try:
            for rows, cols in ((64, 64), (70, 35), (65, 35), (1, 35)):
                for direct in (False, True):
                    torch.manual_seed(42)
                    weight = torch.randn(rows, cols, dtype=torch.bfloat16)
                    qt = QuantizedTensor.from_float(weight, "TensorCoreMXFP8Layout")
                    model = torch.nn.Linear(cols, rows, bias=False, device="meta", dtype=torch.bfloat16)
                    fully_shard(model, mesh=mesh, reshard_after_forward=True)
                    payload = {"weight": qt} if direct else {
                        "weight": qt._qdata, "weight_scale": qt._params.scale.view(torch.uint8),
                        "comfy_quant": {"format": "mxfp8"},
                    }
                    loader.load_from_full_model_state_dict(model, payload, torch.device("cpu"))
                    x = torch.randn(3, cols, dtype=torch.bfloat16)
                    expected = torch.nn.functional.linear(x, qt.dequantize())
                    for _ in range(2):
                        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
                    self.assertIsInstance(model.weight.to_local(), QuantizedTensor)

            weight = torch.randn(70, 35, dtype=torch.bfloat16)
            qt = QuantizedTensor.from_float(weight, "TensorCoreMXFP8Layout")
            model = mixed_precision_ops().Linear(35, 70, bias=False, device="cpu")
            model.load_state_dict({
                "weight": qt._qdata,
                "weight_scale": qt._params.scale.view(torch.uint8),
                "comfy_quant": torch.tensor(list(json.dumps({"format": "mxfp8"}).encode()), dtype=torch.uint8),
            })
            payload = model.state_dict()
            model.to("meta")
            fully_shard(model, mesh=mesh, reshard_after_forward=True)
            model.factory_kwargs["device"] = torch.device("cpu")
            loader.load_from_full_model_state_dict(model, payload, torch.device("cpu"))
            x = torch.randn(3, 35, dtype=torch.bfloat16)
            qx = QuantizedTensor.from_float(x, "TensorCoreMXFP8Layout")
            expected = torch.nn.functional.linear(qx, qt)
            for _ in range(2):
                torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        finally:
            patch.restore_mxfp8_patches()


if __name__ == "__main__":
    dist.init_process_group("gloo")
    try:
        unittest.main()
    finally:
        dist.destroy_process_group()
