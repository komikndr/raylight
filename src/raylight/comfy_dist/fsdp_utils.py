from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from comfy_kitchen.float_utils import from_blocked, to_blocked

try:
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class
except Exception:  # pragma: no cover
    from comfy_kitchen.tensor import QuantizedTensor, get_layout_class  # type: ignore

    QUANT_ALGOS = {}


"""
 This stuff is for systematically fully_shard from bottom up,
 with "*-1 parents are sharded, then continue up to root*", the def collect_bottom_up_shard_order is the function
 the tree looks like this:

model                          [FSDP]
├── block0                     [FSDP]
│   ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
│   │   ├── q.weight           [SHARDED]
│   │   ├── q.scale            [IGNORED]
│   │   ├── k.weight           [SHARDED]
│   │   ├── k.scale            [IGNORED]
│   │   ├── v.weight           [SHARDED]
│   │   └── v.scale            [IGNORED]
│   ├── ffn                    [FSDP, ignored_params={scale}]
│   │   ├── weight             [SHARDED]
│   │   └── scale              [IGNORED]
│   └── conv                   [FSDP, ignored_params={} ]
│       ├── weight             [SHARDED]
│       └── bias               [SHARDED]
│
├── block1                     [FSDP]
│   ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
│   ├── ffn                    [FSDP, ignored_params={scale}]
│   └── conv                   [FSDP]
│
└── block2                     [FSDP]
    ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
    ├── ffn                    [FSDP, ignored_params={scale}]
    └── conv                   [FSDP]
"""


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


def _has_odd_shard_dim(param: torch.Tensor) -> bool:
    return param.ndim > 0 and (int(param.shape[0]) % 2) == 1


def collect_odd_dim0_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for _param_name, param in module.named_parameters(recurse=True):
        if _has_odd_shard_dim(param):
            ignored.add(param)
    return ignored


def _should_materialize_unsharded_param(
    param_name: str,
    param: torch.Tensor,
    full_sd: dict[str, Any] | None = None,
) -> bool:
    if param_name.endswith("input_scale") or param.ndim == 0:
        return True
    if not _has_odd_shard_dim(param):
        return False
    if full_sd is not None and _is_quant_param(param_name, full_sd, param):
        return False
    return True


def _get_parent_module_and_name(model: torch.nn.Module, param_name: str) -> tuple[torch.nn.Module, str]:
    if "." not in param_name:
        return model, param_name
    parent_name, leaf_name = param_name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf_name


def _maybe_collapse_replicated_leading_dim(full_tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    expected_shape = tuple(target_shape)
    if tuple(full_tensor.shape) == expected_shape:
        return full_tensor
    if full_tensor.ndim == 0 or full_tensor.ndim != len(expected_shape):
        return full_tensor
    if tuple(full_tensor.shape[1:]) != expected_shape[1:]:
        return full_tensor

    actual_leading = full_tensor.shape[0]
    expected_leading = expected_shape[0]
    if expected_leading <= 0 or actual_leading < expected_leading or actual_leading % expected_leading != 0:
        return full_tensor

    replicas = actual_leading // expected_leading
    if replicas <= 1:
        return full_tensor

    collapsed = full_tensor.reshape(expected_leading, replicas, *full_tensor.shape[1:])
    canonical = collapsed[:, 0, ...]
    if torch.equal(collapsed, canonical.unsqueeze(1).expand_as(collapsed)):
        return canonical
    return full_tensor


def _materialize_unsharded_param(
    model: torch.nn.Module,
    param_name: str,
    meta_param: torch.Tensor,
    full_tensor: torch.Tensor,
    device: torch.device,
    cpu_offload: bool,
) -> None:
    full_tensor = _maybe_collapse_replicated_leading_dim(full_tensor, meta_param.shape)
    full_tensor = full_tensor.to(dtype=meta_param.dtype, device=device)
    if cpu_offload:
        full_tensor = full_tensor.cpu()
    parent_module, leaf_name = _get_parent_module_and_name(model, param_name)
    parent_module.register_parameter(
        leaf_name,
        torch.nn.Parameter(full_tensor, requires_grad=meta_param.requires_grad),
    )


def _materialize_missing_ignored_params(
    model: torch.nn.Module,
    full_sd: dict[str, Any],
    device: torch.device,
    strict: bool,
    cpu_offload: bool,
    release_sd: bool,
) -> None:
    for param_name, param in list(model.named_parameters()):
        if not getattr(param, "is_meta", False):
            continue
        if not _should_materialize_unsharded_param(param_name, param, full_sd):
            continue
        full_tensor = full_sd.get(param_name)
        if full_tensor is None:
            if strict:
                raise ValueError(f"Missing parameter {param_name} in state_dict")
            continue
        _materialize_unsharded_param(model, param_name, param, full_tensor, device, cpu_offload)
        if release_sd:
            full_sd[param_name] = None


def _collect_subtree_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    return set(module.parameters())


def fully_shard_bottom_up(
    model: torch.nn.Module,
    fsdp_kwargs: dict[str, Any],
    native_ignore_scale: bool,
    ignored_modules: set[torch.nn.Module] | None = None,
) -> int:
    excluded_params: set[torch.nn.Parameter] = set()
    ignored_module_ids: set[int] = set()
    if ignored_modules:
        for mod in ignored_modules:
            excluded_params |= _collect_subtree_params(mod)
            ignored_module_ids.update(id(child) for child in mod.modules())

    shard_order = collect_bottom_up_shard_order(model)
    if ignored_modules:
        shard_order = [(n, m) for n, m in shard_order if id(m) not in ignored_module_ids]

    num_layers_sharded = 0
    for _name, module in shard_order:
        kwargs = dict(fsdp_kwargs)
        ignored_params: set[torch.nn.Parameter] = set()
        if native_ignore_scale:
            ignored_params |= collect_scale_ignored_params(module)

        ignored_params |= collect_input_scale_ignored_params(module)
        ignored_params |= collect_scalar_ignored_params(module)
        ignored_params |= collect_odd_dim0_ignored_params(module)

        subtree_params = set(module.parameters())
        ignored_params |= (excluded_params & subtree_params)

        if ignored_params:
            kwargs["ignored_params"] = ignored_params

        fully_shard(module, **kwargs)
        num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError("No layer modules were sharded. Please check if shard conditions are working as expected.")
    return num_layers_sharded


def materialize_excluded_params(
    model: torch.nn.Module,
    excluded_modules: set[torch.nn.Module],
    full_sd: dict[str, Any],
    device: torch.device,
    cpu_offload: bool = False,
) -> int:
    """Load parameters of modules excluded from FSDP wrapping from a full state dict.

    After FSDP wrapping, excluded-module parameters remain on meta device.
    This function materializes them onto *device* (or CPU when cpu_offload).
    Returns the number of parameters materialized.
    """
    module_to_prefix: dict[int, str] = {}
    for name, mod in model.named_modules():
        module_to_prefix[id(mod)] = name

    count = 0
    for module in excluded_modules:
        prefix = module_to_prefix.get(id(module), "")
        for param_name, param in module.named_parameters(recurse=True):
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if not getattr(param, "is_meta", False):
                continue
            full_tensor = full_sd.get(full_name)
            if full_tensor is None:
                continue
            _materialize_unsharded_param(model, full_name, param, full_tensor, device, cpu_offload)
            count += 1
    return count


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


def _find_scaled_fp8_key(full_sd: dict[str, Any]) -> str | None:
    if "scaled_fp8" in full_sd:
        return "scaled_fp8"

    for key in full_sd.keys():
        if key.endswith(".scaled_fp8"):
            return key

    return None


def _legacy_scaled_fp8_conf(prefix: str, full_sd: dict[str, Any]) -> dict[str, Any] | None:
    has_legacy_scale = f"{prefix}scale_weight" in full_sd
    has_converted_scale = f"{prefix}weight_scale" in full_sd
    if not has_legacy_scale and not has_converted_scale:
        return None

    conf: dict[str, Any] = {"format": "float8_e4m3fn"}
    scaled_fp8_key = _find_scaled_fp8_key(full_sd)
    if scaled_fp8_key is not None:
        scaled_fp8_weight = full_sd.get(scaled_fp8_key)
        if isinstance(scaled_fp8_weight, torch.Tensor) and scaled_fp8_weight.nelement() == 2:
            conf["full_precision_matrix_mult"] = True

    return conf


def _quant_payload_debug_info(param_name: str, full_sd: dict[str, Any]) -> str:
    prefix = param_name[: -len("weight")] if param_name.endswith("weight") else param_name
    debug_bits = {
        "weight": param_name in full_sd,
        "comfy_quant": f"{prefix}comfy_quant" in full_sd,
        "weight_scale": f"{prefix}weight_scale" in full_sd,
        "weight_scale_2": f"{prefix}weight_scale_2" in full_sd,
        "input_scale": f"{prefix}input_scale" in full_sd,
        "legacy_scale_weight": f"{prefix}scale_weight" in full_sd,
        "legacy_scale_input": f"{prefix}scale_input" in full_sd,
        "scaled_fp8": _find_scaled_fp8_key(full_sd) is not None,
    }
    prefix_keys = sorted(
        key
        for key in full_sd.keys()
        if key.startswith(prefix)
        and (
            key == param_name
            or key.endswith("comfy_quant")
            or key.endswith("weight_scale")
            or key.endswith("weight_scale_2")
            or key.endswith("input_scale")
            or key.endswith("scale_weight")
            or key.endswith("scale_input")
        )
    )
    return f"payload={debug_bits}, prefix_keys={prefix_keys}"


def _shard_tensor(
    full_tensor: torch.Tensor,
    sharded_meta_param: Any,
    device: torch.device,
    *,
    pad_to_local_meta: bool = True,
) -> torch.Tensor:
    if not hasattr(sharded_meta_param, "device_mesh"):
        return full_tensor.to(device=device)

    mesh = sharded_meta_param.device_mesh
    if mesh.ndim > 1:
        raise NotImplementedError(f"only support 1D FSDP but got {mesh.ndim}")

    shard_mesh_dim = 0
    shard_world_size = mesh.size(shard_mesh_dim)
    shard_rank = cast(torch.distributed.ProcessGroup, mesh.get_group(shard_mesh_dim)).rank()

    chunk = torch.tensor_split(full_tensor, shard_world_size, dim=0)[shard_rank].to(device=device)

    local_meta = getattr(sharded_meta_param, "_local_tensor", None)
    if not pad_to_local_meta or not isinstance(local_meta, torch.Tensor):
        return chunk

    local_shape = tuple(local_meta.shape)
    if tuple(chunk.shape) == local_shape:
        return chunk
    if len(local_shape) != chunk.ndim:
        return chunk
    if any(local_dim < chunk_dim for local_dim, chunk_dim in zip(local_shape, chunk.shape)):
        return chunk

    sharded_param = full_tensor.new_zeros(local_shape, device=device)
    if chunk.numel() > 0:
        sharded_param[tuple(slice(0, dim) for dim in chunk.shape)].copy_(chunk)
    return sharded_param


def _is_quant_param(param_name: str, full_sd: dict[str, Any], sharded_meta_param: Any) -> bool:
    if isinstance(full_sd.get(param_name), QuantizedTensor):
        return True

    prefix = param_name[: -len("weight")] if param_name.endswith("weight") else None
    if prefix is not None and (
        f"{prefix}comfy_quant" in full_sd
        or f"{prefix}weight_scale" in full_sd
        or f"{prefix}scale_weight" in full_sd
    ):
        return True

    if isinstance(sharded_meta_param, QuantizedTensor):
        return True
    if hasattr(sharded_meta_param, "_local_tensor") and isinstance(sharded_meta_param._local_tensor, QuantizedTensor):
        return True

    return False


def _shard_block_scale(scale, start, end, padded_rows, scale_cols, device):
    # A contiguous group of 128 rows is an independent Kitchen scale tile.
    tile_cols = ((scale_cols + 3) // 4) * 4
    tiles = scale.view(torch.uint8).reshape(-1, tile_cols)
    if start % 128 == 0 and end - start == padded_rows and padded_rows % 128 == 0:
        return tiles[start:end].to(device=device, copy=True).view(scale.dtype)

    scales = torch.zeros((padded_rows, scale_cols), dtype=torch.uint8, device=device)
    if start < end:
        first = start // 128 * 128
        tiles = tiles[first:((end + 127) // 128) * 128]
        tile_rows = from_blocked(tiles, num_rows=tiles.shape[0], num_cols=scale_cols)
        scales[:end - start].copy_(tile_rows[start - first:end - first])
    return to_blocked(scales, flatten=False).view(scale.dtype)


PACKED_ROW_LAYOUTS = ("TensorCoreConvRotW4A4Layout",)


class _PackedRowWeight(QuantizedTensor):
    """Keep packed FSDP storage views separate from Kitchen's matrix shape."""

    def __new__(cls, qdata, layout_cls, params, shape=None):
        return torch.Tensor._make_wrapper_subclass(
            cls, params.orig_shape if shape is None else shape,
            dtype=params.orig_dtype, device=qdata.device, requires_grad=False,
        )

    def __init__(self, qdata, layout_cls, params, shape=None):
        super().__init__(qdata, layout_cls, params)

    def _copy_with(self, qdata=None, params=None, clone_params=True):
        if params is None:
            params = self._params.clone() if clone_params else self._params
        return type(self)(self._qdata if qdata is None else qdata, self._layout_cls, params, self.shape)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        qt = QuantizedTensor.__tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride)
        return _PackedRowWeight(qt._qdata, qt._layout_cls, qt._params, outer_size)

    def dequantize(self):
        return super().dequantize().view(self.shape)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        weight = args[0]
        if isinstance(weight, cls):
            if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
                return cls(weight._qdata, weight._layout_cls, weight._params, weight.shape)
            # reset_sharded_param saves a flat view; init_unsharded_param
            # takes an identity as_strided before wrapping the gathered weight.
            if func == torch.ops.aten.view.default and tuple(args[1]) == (-1,):
                return cls(weight._qdata, weight._layout_cls, weight._params, (weight.numel(),))
            if func == torch.ops.aten.as_strided.default:
                offset = args[3] if len(args) > 3 else (kwargs or {}).get("storage_offset", 0)
                if tuple(args[1]) == weight.shape and tuple(args[2]) == weight.stride() and offset == 0:
                    return cls(weight._qdata, weight._layout_cls, weight._params, weight.shape)
            if weight.ndim == 1:
                return cls._dequant_and_fallback(func, args, kwargs or {})
        return super().__torch_dispatch__(func, types, args, kwargs)

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        return (self._qdata.contiguous(), self._params.scale.contiguous()), tuple(outer_size)

    def fsdp_post_all_gather(self, outputs, shape, param_dtype, *, out=None):
        rows = shape[0]
        qdata = outputs[0][:rows]
        params = replace(self._params, scale=outputs[1][:rows], orig_shape=shape, orig_dtype=param_dtype)
        if out is not None:
            out._qdata = qdata
            out._params = params
            return
        return _PackedRowWeight(qdata, self._layout_cls, params), outputs


def _shard_kitchen_weight(weight, meta_param, device):
    layout = weight._layout_cls
    nvfp4 = layout == "TensorCoreNVFP4Layout"
    mxfp8 = layout == "TensorCoreMXFP8Layout"
    int8 = layout == "TensorWiseINT8Layout"
    packed_rows = layout in PACKED_ROW_LAYOUTS
    fp8 = layout in ("TensorCoreFP8Layout", "TensorCoreFP8E4M3Layout", "TensorCoreFP8E5M2Layout")
    if not (nvfp4 or mxfp8 or int8 or fp8 or packed_rows):
        raise NotImplementedError(f"Unsupported FSDP quantized weight layout: {layout}")

    params = weight._params
    mesh = getattr(meta_param, "device_mesh", None)
    if mesh is None:
        if params.scale is None:
            return QuantizedTensor(weight._qdata.to(device=device), layout, params)
        return weight.to(device=device)
    if mesh.ndim != 1:
        raise NotImplementedError(f"{layout} requires a 1D FSDP mesh, got {mesh.ndim} dimensions")
    if getattr(params, "transposed", False):
        raise NotImplementedError(f"Cannot row-shard a transposed {layout} weight")

    rows = weight.shape[0]
    chunk = (rows + mesh.size() - 1) // mesh.size()
    start = min(mesh.get_local_rank() * chunk, rows)
    end = min(start + chunk, rows)
    local_shape = (chunk, *weight.shape[1:])

    def copy_rows(tensor, padded_rows):
        shard = tensor[start:end]
        if shard.shape[0] == padded_rows:
            return shard.to(device=device, copy=shard.shape != tensor.shape)
        local = tensor.new_zeros((padded_rows, *tensor.shape[1:]), device=device)
        local[:end - start].copy_(shard)
        return local

    if packed_rows:
        params = replace(params, orig_shape=local_shape, scale=copy_rows(params.scale, chunk))
        return _PackedRowWeight(copy_rows(weight._qdata, chunk), layout, params)
    if nvfp4 or mxfp8:
        padded_rows, padded_cols = weight.layout_cls.get_padded_shape(local_shape)
        qdata = copy_rows(weight._qdata, padded_rows)
        scale = params.block_scale if nvfp4 else params.scale
        scale_cols = padded_cols // (16 if nvfp4 else 32)
        local_scale = _shard_block_scale(scale, start, end, padded_rows, scale_cols, device)
        if nvfp4:
            params = replace(params, orig_shape=local_shape, scale=params.scale.to(device=device), block_scale=local_scale)
        else:
            params = replace(params, orig_shape=local_shape, scale=local_scale)
    else:
        qdata = copy_rows(weight._qdata, chunk)
        scale = params.scale
        if isinstance(scale, torch.Tensor):
            if int8 and scale.ndim > 0 and scale.shape[0] == rows:
                scale = copy_rows(scale, chunk)
            else:
                scale = scale.to(device=device)
                if int8:
                    scale = scale.reshape(())
        params = replace(params, orig_shape=local_shape, scale=scale)
    return QuantizedTensor(qdata, layout, params)


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
        return _shard_kitchen_weight(full_q, sharded_meta_param, device)

    prefix = param_name[: -len("weight")]
    conf = _decode_comfy_quant(full_sd.get(f"{prefix}comfy_quant"))
    if conf is None:
        conf = _legacy_scaled_fp8_conf(prefix, full_sd)
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

    if full_q is None:
        raise ValueError(f"Missing quantized weight for {param_name}")

    qdata = full_q.to(dtype=qconfig["storage_t"])
    params_kwargs: dict[str, Any] = {"orig_shape": tuple(sharded_meta_param.shape), "orig_dtype": sharded_meta_param.dtype}
    scale = full_sd.get(f"{prefix}weight_scale")
    if quant_format in ("float8_e4m3fn", "float8_e5m2"):
        params_kwargs["scale"] = scale if scale is not None else full_sd.get(f"{prefix}scale_weight")
    elif quant_format == "mxfp8":
        if scale is None:
            raise ValueError(f"Missing MXFP8 block scales for {param_name}")
        params_kwargs["scale"] = scale.view(torch.float8_e8m0fnu)
    elif quant_format == "nvfp4":
        tensor_scale = full_sd.get(f"{prefix}weight_scale_2")
        if tensor_scale is None or scale is None:
            raise ValueError(f"Missing NVFP4 scales for {param_name}")
        params_kwargs["scale"] = tensor_scale
        params_kwargs["block_scale"] = scale.view(torch.float8_e4m3fn)
    elif quant_format == "int8_tensorwise":
        if scale is None:
            raise ValueError(f"Missing INT8 weight scale for {param_name}")
        params_kwargs["scale"] = scale
        params_conf = conf.get("params", {})
        if not isinstance(params_conf, dict):
            params_conf = {}
        if conf.get("convrot", params_conf.get("convrot", False)):
            params_kwargs["convrot"] = True
            params_kwargs["convrot_groupsize"] = int(conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)))
    elif quant_format == "convrot_w4a4":
        rows, cols = params_kwargs["orig_shape"]
        if qdata.shape != (rows, cols // 2) or cols % 2 or scale is None or scale.numel() != rows:
            raise ValueError(f"Expected packed W4A4 rows and one weight scale per row for {param_name}")
        params_conf = conf.get("params", {})
        if not isinstance(params_conf, dict):
            params_conf = {}
        params_kwargs.update(scale=scale.reshape(rows), quant_group_size=64,
                             convrot_groupsize=int(conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))),
                             linear_dtype=conf.get("linear_dtype", params_conf.get("linear_dtype", "int4")))
    elif quant_format == "gguf":
        qdata = _shard_tensor(qdata, sharded_meta_param, device, pad_to_local_meta=False)
        params_kwargs["orig_shape"] = tuple(qdata.shape)
        n_blocks_per_superblock = conf.get("n_blocks_per_superblock", 8)
        super_block_scale_scale = full_sd.get(f"{prefix}super_block_scale_scale")
        super_block_min_scale = full_sd.get(f"{prefix}super_block_min_scale")
        quantized_block_scale = full_sd.get(f"{prefix}quantized_block_scale")
        quantized_block_min = full_sd.get(f"{prefix}quantized_block_min")
        if (
            super_block_scale_scale is None
            or super_block_min_scale is None
            or quantized_block_scale is None
            or quantized_block_min is None
        ):
            raise ValueError(f"Missing GGUF scales for {param_name}")
        super_block_scale_scale = super_block_scale_scale.to(device=device)
        super_block_min_scale = super_block_min_scale.to(device=device)
        quantized_block_scale = quantized_block_scale.to(device=device)
        quantized_block_min = quantized_block_min.to(device=device)
        super_block_scale_scale = _shard_tensor(super_block_scale_scale, sharded_meta_param, device, pad_to_local_meta=False)
        super_block_min_scale = _shard_tensor(super_block_min_scale, sharded_meta_param, device, pad_to_local_meta=False)
        quantized_block_scale = _shard_tensor(quantized_block_scale, sharded_meta_param, device, pad_to_local_meta=False)
        quantized_block_min = _shard_tensor(quantized_block_min, sharded_meta_param, device, pad_to_local_meta=False)
        params_kwargs["n_blocks_per_superblock"] = n_blocks_per_superblock
        params_kwargs["super_block_scale_scale"] = super_block_scale_scale
        params_kwargs["super_block_min_scale"] = super_block_min_scale
        params_kwargs["quantized_block_scale"] = quantized_block_scale
        params_kwargs["quantized_block_min"] = quantized_block_min
        if f"{prefix}scale" in full_sd:
            scale = full_sd.get(f"{prefix}scale")
            if scale is not None:
                params_kwargs["scale"] = scale.to(device=device)
        return QuantizedTensor(qdata, layout_name, layout_cls.Params(**params_kwargs))
    else:
        raise ValueError(f"Unsupported quantization format: {quant_format}")

    params = layout_cls.Params(**params_kwargs)
    return _shard_kitchen_weight(QuantizedTensor(qdata, layout_name, params), sharded_meta_param, device)


def _release_quant_keys(full_sd: dict[str, Any], param_name: str, keep: set[str] | None = None) -> None:
    #A payload key can also be a parameter in its own right - a quantized
    #embedding carries weight_scale that way - and it has not necessarily been
    #loaded yet when the weight it belongs to frees the group. Releasing it
    #leaves it on meta with its source gone.
    prefix = param_name[: -len("weight")]
    for key in (
        param_name,
        f"{prefix}weight_scale",
        f"{prefix}weight_scale_2",
        f"{prefix}scale_weight",
        f"{prefix}comfy_quant",
        f"{prefix}super_block_scale_scale",
        f"{prefix}super_block_min_scale",
        f"{prefix}quantized_block_scale",
        f"{prefix}quantized_block_min",
    ):
        if key in full_sd and (keep is None or key not in keep):
            full_sd[key] = None


# Heavily modified from
# https://github.com/meta-pytorch/torchtune/blob/d0f63bb33d00b8bd3905a010b71d8c6324c2e980/torchtune/training/_distributed.py#L336
# Need to be done since dcp loader cause wrong dtype among rank when broadcasting.
def load_from_full_model_state_dict(
    model,
    full_sd,
    device,
    strict=False,
    cpu_offload=False,
    release_sd=True,
):
    meta_sharded_sd = model.state_dict()
    # Unsharded Kitchen modules export packed bytes and synthetic scale keys.
    # Target the actual parameter so its logical dimensions and dtype survive.
    for key in list(meta_sharded_sd):
        if key.rsplit(".", 1)[-1] != "comfy_quant":
            continue
        module, _ = _get_parent_module_and_name(model, key)
        weight = module._parameters.get("weight")
        if isinstance(weight, QuantizedTensor):
            prefix = key[:-len("comfy_quant")]
            meta_sharded_sd[prefix + "weight"] = weight
            del meta_sharded_sd[key]
            for name in ("weight_scale", "weight_scale_2"):
                if name not in module._parameters and name not in module._buffers:
                    meta_sharded_sd.pop(prefix + name, None)
    own_params = set(meta_sharded_sd)
    sharded_sd: dict[str, torch.Tensor] = {}
    unsharded_params = []
    for param_name, sharded_meta_param in meta_sharded_sd.items():
        parent_module, leaf_name = _get_parent_module_and_name(model, param_name)
        is_buffer = leaf_name in parent_module._buffers
        if not is_buffer and _should_materialize_unsharded_param(param_name, sharded_meta_param, full_sd):
            full_tensor = full_sd.get(param_name)
            if full_tensor is None:
                if strict:
                    raise ValueError(f"Missing parameter {param_name} in state_dict")
                continue
            unsharded_params.append((param_name, sharded_meta_param, full_tensor))
            sharded_sd[param_name] = full_tensor
            continue

        if not is_buffer and _is_quant_param(param_name, full_sd, sharded_meta_param):
            quant_tensor = _build_quantized_tensor(param_name, full_sd, sharded_meta_param, device)
            if quant_tensor is None:
                raise ValueError(
                    f"Expected quantized tensor for {param_name}, but could not build it ({_quant_payload_debug_info(param_name, full_sd)})"
                )
            if hasattr(sharded_meta_param, "device_mesh"):
                sharded_tensor = DTensor(
                    quant_tensor,
                    sharded_meta_param._spec,
                    requires_grad=sharded_meta_param.requires_grad,
                )
            else:
                sharded_tensor = quant_tensor
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
            #Parameter defaults requires_grad to True, which undoes the freeze
            #the wrapper applied and, for a quantized tensor that reports an
            #integer dtype, raises outright. The meta parameter already carries
            #the right answer, as it does for the DTensor above.
            sharded_sd[param_name] = torch.nn.Parameter(
                sharded_tensor, requires_grad=sharded_meta_param.requires_grad)
            if release_sd:
                own_params.discard(param_name)
                _release_quant_keys(full_sd, param_name, keep=own_params)
            continue
        full_tensor = full_sd.get(param_name)
        if full_tensor is None:
            if strict:
                raise ValueError(f"Missing parameter {param_name} in state_dict")
            continue
        if not hasattr(sharded_meta_param, "device_mesh"):
            full_tensor = _maybe_collapse_replicated_leading_dim(full_tensor, sharded_meta_param.shape)
            full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
            sharded_tensor = full_tensor
        else:
            full_tensor = full_tensor.to(sharded_meta_param.dtype)
            local_dense = _shard_tensor(full_tensor, sharded_meta_param, device)
            sharded_tensor = DTensor.from_local(
                local_dense,
                device_mesh=sharded_meta_param.device_mesh,
                placements=sharded_meta_param.placements,
            )
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[param_name] = sharded_tensor if is_buffer else torch.nn.Parameter(
            sharded_tensor, requires_grad=sharded_meta_param.requires_grad)
        if release_sd:
            own_params.discard(param_name)
            full_sd[param_name] = None
    out = model.load_state_dict(sharded_sd, strict=strict, assign=True)
    for param_name, meta_param, full_tensor in unsharded_params:
        _materialize_unsharded_param(model, param_name, meta_param, full_tensor, device, cpu_offload)
        if release_sd:
            full_sd[param_name] = None
    _materialize_missing_ignored_params(model, full_sd, device, strict, cpu_offload, release_sd)
    return out
