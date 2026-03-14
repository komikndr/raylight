from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule

ROOT = Path(__file__).resolve().parents[3]
RAYLIGHT_SRC = ROOT / "custom_nodes" / "raylight" / "src"
KITCHEN_REPO = Path("/home/kxn/comfy-kitchen-distributed")
for candidate in (str(RAYLIGHT_SRC), str(KITCHEN_REPO), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from importlib.util import module_from_spec, spec_from_file_location

from comfy_kitchen.tensor.base import QuantizedTensor


def _load_module(name: str, file_path: Path):
    spec = spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module {name} from {file_path}')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fsdp_utils = _load_module('raylight_fsdp_utils_test', RAYLIGHT_SRC / 'raylight' / 'comfy_dist' / 'fsdp_utils.py')
_nvfp4_patch = _load_module('raylight_nvfp4_patch_test', RAYLIGHT_SRC / 'raylight' / 'comfy_dist' / 'kitchen_patches' / 'nvfp4.py')

fully_shard_bottom_up = _fsdp_utils.fully_shard_bottom_up
load_from_full_model_state_dict = _fsdp_utils.load_from_full_model_state_dict
install_nvfp4_patches = _nvfp4_patch.install_nvfp4_patches
restore_nvfp4_patches = _nvfp4_patch.restore_nvfp4_patches

PAGESIZE = os.sysconf("SC_PAGE_SIZE")


def rss_mb() -> float:
    with open("/proc/self/statm", "r", encoding="utf-8") as f:
        rss_pages = int(f.read().split()[1])
    return rss_pages * PAGESIZE / (1024 * 1024)


class TinyNVFP4MLP(torch.nn.Module):
    def __init__(self, dim: int = 256, bias: bool = True):
        super().__init__()
        self.l1 = torch.nn.Linear(dim, dim, bias=bias, dtype=torch.bfloat16)
        self.l2 = torch.nn.Linear(dim, dim, bias=bias, dtype=torch.bfloat16)
        self.l3 = torch.nn.Linear(dim, dim, bias=bias, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = torch.nn.functional.gelu(x)
        x = self.l2(x)
        x = torch.nn.functional.gelu(x)
        x = self.l3(x)
        return x


class StackedNVFP4MLP(torch.nn.Module):
    def __init__(self, dim: int, num_layers: int, bias: bool = False):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=bias, dtype=torch.bfloat16) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class BuildResult:
    model: torch.nn.Module
    full_sd: dict[str, torch.Tensor]
    num_layers: int
    approx_weight_bytes: int


def _quant_weight(prefix: str, weight: torch.Tensor, full_sd: dict[str, torch.Tensor], *, input_scale: float = 1.0) -> None:
    qt = QuantizedTensor.from_float(weight, 'TensorCoreNVFP4Layout')
    for key, value in qt.state_dict(prefix=f'{prefix}.weight').items():
        full_sd[key] = value.detach().clone()
    full_sd[f'{prefix}.input_scale'] = torch.tensor([input_scale], dtype=torch.float32)


def build_tiny_model(dim: int = 256) -> BuildResult:
    model = TinyNVFP4MLP(dim=dim, bias=True).to(device='meta')
    full_sd: dict[str, torch.Tensor] = {}
    for idx, layer_name in enumerate(('l1', 'l2', 'l3')):
        weight = torch.ones((dim, dim), dtype=torch.bfloat16) * (idx + 1)
        _quant_weight(layer_name, weight, full_sd, input_scale=1.0 + idx)
        full_sd[f'{layer_name}.bias'] = torch.zeros((dim,), dtype=torch.bfloat16)
    return BuildResult(model=model, full_sd=full_sd, num_layers=3, approx_weight_bytes=3 * dim * dim // 2)


def build_large_model(target_gib: float, dim: int = 4096) -> BuildResult:
    bytes_per_weight = dim * dim / 2 + dim * (dim // 16)
    target_bytes = int(target_gib * (1024 ** 3))
    num_layers = int(max(1, target_bytes // bytes_per_weight))
    model = StackedNVFP4MLP(dim=dim, num_layers=num_layers, bias=False).to(device='meta')
    full_sd: dict[str, torch.Tensor] = {}
    base = torch.randn((dim, dim), dtype=torch.bfloat16)
    for i in range(num_layers):
        _quant_weight(f'layers.{i}', base, full_sd, input_scale=1.0)
    return BuildResult(model=model, full_sd=full_sd, num_layers=num_layers, approx_weight_bytes=num_layers * bytes_per_weight)


def _meta_params(model: torch.nn.Module) -> list[str]:
    out = []
    for name, param in model.named_parameters():
        if getattr(param, 'is_meta', False):
            out.append(name)
    return out


def run_case(mode: str, target_gib: float) -> dict[str, object]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if mode == 'tiny':
        built = build_tiny_model()
        x = torch.ones((4, 256), dtype=torch.bfloat16)
    else:
        built = build_large_model(target_gib=target_gib)
        x = torch.ones((1, built.model.layers[0].in_features), dtype=torch.bfloat16)

    report = {
        'rank': rank,
        'world_size': world_size,
        'mode': mode,
        'rss_before_shard_mb': round(rss_mb(), 2),
        'num_layers': built.num_layers,
        'approx_weight_gib': round(built.approx_weight_bytes / (1024 ** 3), 3),
    }

    fully_shard_bottom_up(built.model, fsdp_kwargs={'reshard_after_forward': True}, native_ignore_scale=False)
    report['rss_after_shard_mb'] = round(rss_mb(), 2)
    load_from_full_model_state_dict(
        built.model,
        built.full_sd,
        device=torch.device('cpu'),
        strict=False,
        cpu_offload=False,
        release_sd=True,
    )
    report['rss_after_load_mb'] = round(rss_mb(), 2)
    report['meta_params_after_load'] = _meta_params(built.model)
    assert not report['meta_params_after_load'], report['meta_params_after_load']
    assert isinstance(built.model, FSDPModule)

    if mode == 'tiny':
        with torch.no_grad():
            y = built.model(x)
        report['rss_after_forward_mb'] = round(rss_mb(), 2)
        report['output_shape'] = tuple(y.shape)
        report['output_sum'] = float(y.float().sum().item())
    else:
        report['rss_after_forward_mb'] = None
        report['output_shape'] = None
        report['output_sum'] = None
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('tiny', 'memory'), default='tiny')
    parser.add_argument('--target-gib', type=float, default=2.0)
    args = parser.parse_args()

    dist.init_process_group(backend='gloo')
    install_nvfp4_patches()
    try:
        result = run_case(args.mode, args.target_gib)
        print(json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        restore_nvfp4_patches()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
