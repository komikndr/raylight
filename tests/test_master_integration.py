import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys

import torch


ROOT = Path(__file__).resolve().parents[1] / "src/raylight"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from comfy import ops, model_detection
from comfy.ldm.lightricks.av_model import LTXAVModel
from comfy.ldm.modules.attention import ComfyAttention


def function(path, name, namespace):
    node = next(node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class MasterIntegrationTest(unittest.TestCase):
    def test_minimax_checkpoint_attention_is_checked_before_injection(self):
        attention = torch.nn.Module()
        attention.comfy_attention = ComfyAttention()
        attention.comfy_attention.function = lambda: None
        model = SimpleNamespace(blocks=[SimpleNamespace(attn=attention)])
        node = next(n for n in ast.walk(ast.parse((ROOT / "distributed_modules/usp.py").read_text())) if isinstance(n, ast.FunctionDef) and n.name == "_inject_minimax_h3")
        node.decorator_list = []
        node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
        namespace = {"ComfyAttention": ComfyAttention, "FSDP_LORA_SIDECAR_ATTACHMENT": "sidecar"}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "minimax_injection", "exec"), namespace)
        with self.assertRaisesRegex(ValueError, "Checkpoint-selected"):
            namespace["_inject_minimax_h3"](SimpleNamespace(get_attachment=lambda key: None), SimpleNamespace(diffusion_model=model))

    def test_gguf_embedded_config_and_return_contract(self):
        namespace = {
            "gguf": SimpleNamespace(GGUFReader=lambda path: SimpleNamespace(tensors=[])),
            "get_field": lambda reader, name, kind: {"general.architecture": "ltxv", "config": '{"transformer":{"rope_theta":42}}'}.get(name),
            "TXT_ARCH_LIST": set(), "IMG_ARCH_LIST": {"ltxv"},
            "logging": SimpleNamespace(info=lambda *args: None),
        }
        path = ROOT / "expansion/comfyui_gguf/loader.py"
        function(path, "_load_gguf_state_dict", namespace)
        load = function(path, "gguf_sd_loader", namespace)
        load_metadata = function(path, "gguf_sd_loader_with_metadata", namespace)
        self.assertEqual(load("test.gguf"), {})
        self.assertEqual(load("test.gguf", return_arch=True), ({}, "ltxv"))
        sd, metadata = load_metadata("test.gguf")
        self.assertEqual(metadata["config"], '{"transformer":{"rope_theta":42}}')

    def test_minimax_attention_without_checkpoint_selector(self):
        torch.manual_seed(42)
        attention = torch.nn.Module()
        attention.heads, attention.head_dim = 2, 4
        attention.qkv_proj = torch.nn.Linear(8, 24, bias=False)
        attention.q_norm = torch.nn.RMSNorm(4)
        attention.k_norm = torch.nn.RMSNorm(4)
        attention.out_proj = torch.nn.Linear(8, 8, bias=False)

        def distributed_attention(q, k, v, heads, **kwargs):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)

        forward = function(ROOT / "diffusion_models/minimax/xdit_context_parallel.py", "usp_attn_forward", {
            "torch": torch, "xfuser_optimized_attention": distributed_attention,
        })
        x = torch.randn(7, 8)
        q, k, v = attention.qkv_proj(x).chunk(3, dim=-1)
        q = attention.q_norm(q.reshape(7, 2, 4)).transpose(0, 1).unsqueeze(0)
        k = attention.k_norm(k.reshape(7, 2, 4)).transpose(0, 1).unsqueeze(0)
        v = v.reshape(7, 2, 4).transpose(0, 1).unsqueeze(0)
        expected = attention.out_proj(distributed_attention(q, k, v, 2).squeeze(0))
        torch.testing.assert_close(forward(attention, x), expected)

    def test_ltx_23_gguf_configuration_matches_master_shapes(self):
        state_dict = {
            "adaln_single.emb.timestep_embedder.linear_1.bias": torch.empty(1, device="meta"),
            "audio_adaln_single.linear.weight": torch.empty(1, 1, device="meta"),
            "transformer_blocks.0.attn2.to_k.weight": torch.empty(4096, 4096, device="meta"),
            "transformer_blocks.0.scale_shift_table": torch.empty(9, 4096, device="meta"),
            "video_embeddings_connector.learnable_registers": torch.empty(128, 4096, device="meta"),
            "audio_embeddings_connector.learnable_registers": torch.empty(128, 2048, device="meta"),
            "caption_projection.linear_1.weight": torch.empty(4096, 3840, device="meta"),
        }
        infer = function(ROOT / "expansion/comfyui_gguf/loader.py", "_ltx_gguf_config", {})
        legacy = dict(state_dict)
        legacy["transformer_blocks.0.scale_shift_table"] = torch.empty(6, 4096, device="meta")
        self.assertEqual(infer(legacy), {})
        metadata = {"config": json.dumps({"transformer": infer(state_dict)})}
        config = model_detection.detect_unet_config(state_dict, "", metadata=metadata)
        config.pop("image_model")
        model = LTXAVModel(**config, device="meta", dtype=torch.bfloat16, operations=ops.disable_weight_init)
        loaded_shapes = model.state_dict()
        for key in ("transformer_blocks.0.scale_shift_table", "video_embeddings_connector.learnable_registers", "audio_embeddings_connector.learnable_registers"):
            self.assertEqual(loaded_shapes[key].shape, state_dict[key].shape)

    def test_gguf_loader_passes_metadata_and_model_options(self):
        calls = []
        model = SimpleNamespace()
        def load(sd, **kwargs):
            calls.append((sd, kwargs))
            return model
        payload, metadata = {"weight": "payload"}, {"config": "config"}
        import_node = ast.parse((ROOT / "comfy_dist/sd.py").read_text())
        node = next(n for n in import_node.body if isinstance(n, ast.FunctionDef) and n.name == "gguf_load_diffusion_model")
        node.body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        namespace = {
            "GGMLOps": lambda: SimpleNamespace(Linear=SimpleNamespace()),
            "gguf_sd_loader_with_metadata": lambda *args, **kwargs: (payload, metadata),
            "GGUFModelPatcher": SimpleNamespace(clone=lambda m: m),
            "comfy": SimpleNamespace(sd=SimpleNamespace(load_diffusion_model_state_dict=load)),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), "gguf_loader", "exec"), namespace)
        self.assertIs(namespace["gguf_load_diffusion_model"]("model.gguf", {"dtype": torch.bfloat16}), model)
        self.assertIs(calls[0][1]["metadata"], metadata)
        self.assertEqual(calls[0][1]["model_options"]["dtype"], torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
