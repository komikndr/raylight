from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

try:
    from comfy_kitchen.tensor import QuantizedTensor, get_layout_class
except Exception:  # pragma: no cover
    from comfy.quant_ops import QuantizedTensor, get_layout_class  # type: ignore

try:
    from comfy.quant_ops import QUANT_ALGOS
except Exception:  # pragma: no cover
    QUANT_ALGOS = {}


def freeze_and_detect_qt(model: torch.nn.Module) -> bool:
    has_qt = False
    for param in model.parameters():
        param.requires_grad = False
        local = getattr(param, "_local_tensor", None)
        if isinstance(param, QuantizedTensor) or isinstance(local, QuantizedTensor):
            has_qt = True
    return has_qt


def _mod_name(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _module_has_subtree_params(module: torch.nn.Module) -> bool:
    return any(True for _ in module.named_parameters(recurse=True))


def _module_has_direct_params(module: torch.nn.Module) -> bool:
    return any(True for _ in module.named_parameters(recurse=False))


def _children_with_params(name: str, module: torch.nn.Module, named_map: dict[str, torch.nn.Module]) -> list[str]:
    out: list[str] = []
    for child_name, _child in module.named_children():
        full = _mod_name(name, child_name)
        if _module_has_subtree_params(named_map[full]):
            out.append(full)
    return out


def _is_descendant(path: str, ancestor: str) -> bool:
    if ancestor == "":
        return path != ""
    return path == ancestor or path.startswith(ancestor + ".")


def _collect_leaf_parent_targets(model: torch.nn.Module) -> set[str]:
    named = dict(model.named_modules())
    all_names = [name for name in named.keys() if name != ""]

    structural_groups: set[str] = set()
    for name in all_names:
        children = _children_with_params(name, named[name], named)
        if len(children) >= 2:
            structural_groups.add(name)

    targets: set[str] = set(structural_groups)
    for name in all_names:
        if _module_has_direct_params(named[name]):
            if not any(_is_descendant(name, group_name) for group_name in structural_groups):
                targets.add(name)

    return targets


def _add_ancestors_to_root(targets: set[str]) -> set[str]:
    out = set(targets)
    for target in list(targets):
        cur = target
        while "." in cur:
            cur = cur.rsplit(".", 1)[0]
            out.add(cur)
    out.add("")
    return out


def _depth(name: str) -> int:
    return 0 if name == "" else name.count(".") + 1


def _supports_fully_shard(module: torch.nn.Module) -> bool:
    return type(module).forward is not torch.nn.Module.forward


def collect_bottom_up_shard_order(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    named = dict(model.named_modules())
    leaf_parents = _collect_leaf_parent_targets(model)
    all_targets = _add_ancestors_to_root(leaf_parents)
    ordered_names = sorted(all_targets, key=_depth, reverse=True)

    out: list[tuple[str, torch.nn.Module]] = []
    for name in ordered_names:
        module = model if name == "" else named[name]
        if _supports_fully_shard(module):
            out.append((name, module))
    return out


def collect_scale_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for param_name, param in module.named_parameters(recurse=True):
        if "scale" in param_name:
            ignored.add(param)
    return ignored


def collect_input_scale_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for param_name, param in module.named_parameters(recurse=True):
        if "input_scale" in param_name:
            ignored.add(param)
    return ignored


def collect_scalar_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for _param_name, param in module.named_parameters(recurse=True):
        if param.ndim == 0:
            ignored.add(param)
    return ignored


def fully_shard_bottom_up(
    model: torch.nn.Module,
    fsdp_kwargs: dict[str, Any],
    native_ignore_scale: bool,
) -> int:
    num_layers_sharded = 0
    for _name, module in collect_bottom_up_shard_order(model):
        kwargs = dict(fsdp_kwargs)
        ignored_params: set[torch.nn.Parameter] = set()
        if native_ignore_scale:
            ignored_params |= collect_scale_ignored_params(module)

        ignored_params |= collect_input_scale_ignored_params(module)
        ignored_params |= collect_scalar_ignored_params(module)

        if ignored_params:
            kwargs["ignored_params"] = ignored_params

        fully_shard(module, **kwargs)
        num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError("No layer modules were sharded. Please check if shard conditions are working as expected.")
    return num_layers_sharded


def _decode_comfy_quant(conf: Any) -> dict[str, Any] | None:
    if conf is None:
        return None
    if isinstance(conf, dict):
        return conf
    if isinstance(conf, (bytes, bytearray)):
        return json.loads(conf.decode("utf-8"))
    if isinstance(conf, torch.Tensor):
        raw = conf.detach().cpu().numpy().tobytes()
        if conf.dtype == torch.uint8:
            return json.loads(raw)
        return json.loads(raw.decode("utf-8"))
    if isinstance(conf, str):
        return json.loads(conf)
    raise TypeError(f"Unsupported comfy_quant type: {type(conf)}")


def _shard_tensor(full_tensor: torch.Tensor, sharded_meta_param: Any, device: torch.device) -> torch.Tensor:
    if not hasattr(sharded_meta_param, "device_mesh"):
        return full_tensor.to(device=device)

    mesh = sharded_meta_param.device_mesh
    if mesh.ndim > 1:
        raise NotImplementedError(f"only support 1D FSDP but got {mesh.ndim}")

    shard_mesh_dim = 0
    shard_world_size = mesh.size(shard_mesh_dim)
    shard_rank = cast(torch.distributed.ProcessGroup, mesh.get_group(shard_mesh_dim)).rank()

    chunk = list(torch.chunk(full_tensor, shard_world_size, dim=0))[shard_rank]
    sharded_param = full_tensor.new_zeros(chunk.size(), device=device)
    sharded_param[: chunk.size(0)].copy_(chunk)
    return sharded_param


def _is_quant_param(param_name: str, full_sd: dict[str, Any], sharded_meta_param: Any) -> bool:
    if isinstance(full_sd.get(param_name), QuantizedTensor):
        return True

    prefix = param_name[: -len("weight")] if param_name.endswith("weight") else None
    if prefix is not None and f"{prefix}comfy_quant" in full_sd:
        return True

    if isinstance(sharded_meta_param, QuantizedTensor):
        return True
    if hasattr(sharded_meta_param, "_local_tensor") and isinstance(sharded_meta_param._local_tensor, QuantizedTensor):
        return True

    return False


def _build_quantized_tensor(
    param_name: str,
    full_sd: dict[str, Any],
    sharded_meta_param: Any,
    device: torch.device,
):
    if not param_name.endswith("weight"):
        return None

    full_q = full_sd.get(param_name)
    if isinstance(full_q, QuantizedTensor):
        qt = cast(Any, full_q)
        local_qdata = _shard_tensor(qt._qdata.to(device=device), sharded_meta_param, device)
        local_params = replace(qt._params, orig_shape=tuple(local_qdata.shape))
        return QuantizedTensor(local_qdata, qt._layout_cls, local_params)

    prefix = param_name[: -len("weight")]
    conf = _decode_comfy_quant(full_sd.get(f"{prefix}comfy_quant"))
    if conf is None:
        return None

    quant_format = conf.get("format", None)
    if quant_format is None or quant_format not in QUANT_ALGOS:
        raise ValueError(f"Unknown quantization format for {param_name}: {quant_format}")

    qconfig = QUANT_ALGOS[quant_format]
    layout_name = qconfig["comfy_tensor_layout"]
    layout_cls = get_layout_class(layout_name)
    if layout_cls is None:
        raise ValueError(f"Missing layout class for {layout_name}")

    full_qdata = full_sd.get(param_name)
    if full_qdata is None:
        raise ValueError(f"Missing quantized weight for {param_name}")

    qdata = full_qdata.to(device=device, dtype=qconfig["storage_t"])
    qdata = _shard_tensor(qdata, sharded_meta_param, device)

    params_kwargs: dict[str, Any] = {"orig_shape": tuple(qdata.shape)}

    local_meta = None
    if isinstance(sharded_meta_param, QuantizedTensor):
        local_meta = sharded_meta_param
    elif hasattr(sharded_meta_param, "_local_tensor") and isinstance(sharded_meta_param._local_tensor, QuantizedTensor):
        local_meta = sharded_meta_param._local_tensor
    if local_meta is not None and hasattr(local_meta, "_params"):
        orig_dtype = getattr(local_meta._params, "orig_dtype", None)
        if orig_dtype is not None:
            params_kwargs["orig_dtype"] = orig_dtype

    if quant_format in ("float8_e4m3fn", "float8_e5m2"):
        scale = full_sd.get(f"{prefix}weight_scale")
        if scale is not None:
            scale = scale.to(device=device)
        params_kwargs["scale"] = scale
    elif quant_format == "nvfp4":
        tensor_scale = full_sd.get(f"{prefix}weight_scale_2")
        block_scale = full_sd.get(f"{prefix}weight_scale")
        if tensor_scale is None or block_scale is None:
            raise ValueError(f"Missing NVFP4 scales for {param_name}")
        tensor_scale = tensor_scale.to(device=device)
        block_scale = block_scale.view(dtype=torch.float8_e4m3fn).to(device=device)
        block_scale = _shard_tensor(block_scale, sharded_meta_param, device)
        params_kwargs["scale"] = tensor_scale
        params_kwargs["block_scale"] = block_scale
    else:
        raise ValueError(f"Unsupported quantization format: {quant_format}")

    params = layout_cls.Params(**params_kwargs)
    return QuantizedTensor(qdata, layout_name, params)


def _release_quant_keys(full_sd: dict[str, Any], param_name: str) -> None:
    prefix = param_name[: -len("weight")]
    for key in (param_name, f"{prefix}weight_scale", f"{prefix}weight_scale_2", f"{prefix}comfy_quant"):
        if key in full_sd:
            full_sd[key] = None


def load_from_full_model_state_dict(
    model,
    full_sd,
    device,
    strict=False,
    cpu_offload=False,
    release_sd=True,
):
    meta_sharded_sd = model.state_dict()
    sharded_sd: dict[str, torch.nn.Parameter] = {}
    for param_name, sharded_meta_param in meta_sharded_sd.items():
        if param_name.endswith("input_scale"):
            full_tensor = full_sd.get(param_name)
            if full_tensor is None:
                if strict:
                    raise ValueError(f"Missing parameter {param_name} in state_dict")
                continue
            full_tensor = full_tensor.to(dtype=sharded_meta_param.dtype, device=device)
            if cpu_offload:
                full_tensor = full_tensor.cpu()
            sharded_sd[param_name] = torch.nn.Parameter(full_tensor, requires_grad=sharded_meta_param.requires_grad)
            if release_sd:
                full_sd[param_name] = None
            continue

        if _is_quant_param(param_name, full_sd, sharded_meta_param):
            quant_tensor = _build_quantized_tensor(param_name, full_sd, sharded_meta_param, device)
            if quant_tensor is None:
                raise ValueError(f"Expected quantized tensor for {param_name}, but could not build it")
            if hasattr(sharded_meta_param, "device_mesh"):
                sharded_tensor = DTensor.from_local(
                    quant_tensor,
                    device_mesh=sharded_meta_param.device_mesh,
                    placements=sharded_meta_param.placements,
                )
            else:
                sharded_tensor = quant_tensor
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
            sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
            if release_sd:
                _release_quant_keys(full_sd, param_name)
            continue
        full_tensor = full_sd.get(param_name)
        if full_tensor is None:
            if strict:
                raise ValueError(f"Missing parameter {param_name} in state_dict")
            continue
        full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
        if hasattr(sharded_meta_param, "device_mesh"):
            local_dense = _shard_tensor(full_tensor, sharded_meta_param, device)
            sharded_tensor = DTensor.from_local(
                local_dense,
                device_mesh=sharded_meta_param.device_mesh,
                placements=sharded_meta_param.placements,
            )
        else:
            sharded_tensor = full_tensor
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
        if release_sd:
            full_sd[param_name] = None
    return model.load_state_dict(sharded_sd, strict=strict, assign=True)
