# INT8 tensorwise patches for Raylight FSDP operations.
from __future__ import annotations

from typing import Any, cast

import torch

_PATCHED = False
_MISSING = object()
_ORIG_LAYOUT_PRE = _MISSING
_ORIG_LAYOUT_POST = _MISSING
_PATCHED_LAYOUT = None
_ORIG_QT_PRE = _MISSING
_ORIG_QT_POST = _MISSING


def _get_op(path: str) -> Any:
    cur = torch
    for part in path.split(".")[1:]:
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def install_int8_patches() -> None:
    global _PATCHED
    global _ORIG_LAYOUT_PRE, _ORIG_LAYOUT_POST, _PATCHED_LAYOUT, _ORIG_QT_PRE, _ORIG_QT_POST
    if _PATCHED:
        return

    from comfy_kitchen.tensor.base import QuantizedTensor, register_layout_op
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as KitchenTensorWiseINT8Layout

    try:
        from comfy.quant_ops import get_layout_class as comfy_get_layout_class
        TensorWiseINT8Layout = comfy_get_layout_class("TensorWiseINT8Layout") or KitchenTensorWiseINT8Layout
    except Exception:  # pragma: no cover
        TensorWiseINT8Layout = KitchenTensorWiseINT8Layout
    _PATCHED_LAYOUT = TensorWiseINT8Layout

    def maybe_register(op):
        def deco(fn):
            if op is not None:
                register_layout_op(op, TensorWiseINT8Layout)(fn)
            return fn

        return deco

    def _is_row_scale(scale: Any, qdata: torch.Tensor) -> bool:
        return isinstance(scale, torch.Tensor) and scale.dim() > 0 and qdata.dim() > 0 and int(scale.shape[0]) == int(qdata.shape[0])

    def _wrap_int8_tensor(qtensor, qdata: torch.Tensor, *, scale=None, orig_shape=None, transposed=None):
        old = qtensor._params
        params = TensorWiseINT8Layout.Params(
            scale=old.scale if scale is None else scale,
            orig_dtype=old.orig_dtype,
            orig_shape=tuple(qdata.shape) if orig_shape is None else tuple(orig_shape),
            is_weight=getattr(old, "is_weight", True),
            convrot=getattr(old, "convrot", False),
            convrot_groupsize=getattr(old, "convrot_groupsize", 256),
            transposed=getattr(old, "transposed", False) if transposed is None else transposed,
        )
        return QuantizedTensor(qdata, qtensor._layout_cls, params)

    def _slice_row_scale(input_tensor, dim, start, end, step):
        scale = input_tensor._params.scale
        if dim != 0 or not _is_row_scale(scale, input_tensor._qdata):
            return scale
        row_slice = [slice(None)] * scale.dim()
        row_slice[0] = slice(start, end, step)
        return scale[tuple(row_slice)]

    def _is_flatten_to_1d(args, kwargs) -> bool:
        shape = None
        if len(args) > 1:
            shape = args[1:] if len(args) > 2 else args[1]
        else:
            shape = kwargs.get("shape", kwargs.get("size", None))
        if isinstance(shape, int):
            return shape == -1
        if isinstance(shape, torch.Size):
            shape = tuple(shape)
        if isinstance(shape, (list, tuple)) and len(shape) == 1:
            try:
                return int(shape[0]) == -1
            except TypeError:
                return False
        return False

    def pre_all_gather(qtensor: QuantizedTensor, mesh, outer_size):
        qdata = qtensor._qdata.contiguous()
        scale = qtensor._params.scale
        scale_is_sharded = _is_row_scale(scale, qtensor._qdata)
        metadata = {
            "scale_is_sharded": scale_is_sharded,
            "orig_dtype": qtensor._params.orig_dtype,
            "orig_shape": tuple(outer_size),
            "is_weight": getattr(qtensor._params, "is_weight", True),
            "convrot": getattr(qtensor._params, "convrot", False),
            "convrot_groupsize": getattr(qtensor._params, "convrot_groupsize", 256),
            "transposed": getattr(qtensor._params, "transposed", False),
        }
        if scale_is_sharded:
            return (qdata, scale.contiguous()), metadata
        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=qdata.device)
        metadata["scale"] = scale
        return (qdata,), metadata

    def post_all_gather(qtensor: QuantizedTensor, all_gather_outputs, metadata: Any, param_dtype: torch.dtype, *, out=None):
        gathered_qdata = all_gather_outputs[0] if isinstance(all_gather_outputs, tuple) else all_gather_outputs
        if metadata.get("scale_is_sharded", False):
            scale = all_gather_outputs[1]
        else:
            scale = metadata["scale"]
        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=gathered_qdata.device)
        orig_shape = metadata["orig_shape"]
        gathered_qdata = gathered_qdata[:orig_shape[0]]
        if metadata["scale_is_sharded"]:
            scale = scale[:orig_shape[0]]
        params = TensorWiseINT8Layout.Params(
            scale=scale,
            orig_dtype=metadata.get("orig_dtype", param_dtype),
            orig_shape=orig_shape,
            is_weight=metadata.get("is_weight", True),
            convrot=metadata.get("convrot", False),
            convrot_groupsize=metadata.get("convrot_groupsize", 256),
            transposed=metadata.get("transposed", False),
        )
        if out is not None:
            if not isinstance(out, QuantizedTensor):
                raise TypeError(f"Expected QuantizedTensor out, got {type(out)}")
            out._qdata = gathered_qdata
            out._params = params
            return None
        return QuantizedTensor(gathered_qdata, qtensor._layout_cls, params), all_gather_outputs

    if _ORIG_LAYOUT_PRE is _MISSING:
        _ORIG_LAYOUT_PRE = getattr(TensorWiseINT8Layout, "pre_all_gather", _MISSING)
    if _ORIG_LAYOUT_POST is _MISSING:
        _ORIG_LAYOUT_POST = getattr(TensorWiseINT8Layout, "post_all_gather", _MISSING)
    setattr(TensorWiseINT8Layout, "pre_all_gather", pre_all_gather)
    setattr(TensorWiseINT8Layout, "post_all_gather", post_all_gather)

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        return self.layout_cls.pre_all_gather(self, mesh, outer_size)

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        return self.layout_cls.post_all_gather(self, all_gather_outputs, metadata, param_dtype, out=out)

    if _ORIG_QT_PRE is _MISSING:
        _ORIG_QT_PRE = getattr(QuantizedTensor, "fsdp_pre_all_gather", _MISSING)
    if _ORIG_QT_POST is _MISSING:
        _ORIG_QT_POST = getattr(QuantizedTensor, "fsdp_post_all_gather", _MISSING)
    setattr(QuantizedTensor, "fsdp_pre_all_gather", fsdp_pre_all_gather)
    setattr(QuantizedTensor, "fsdp_post_all_gather", fsdp_post_all_gather)

    op_all_gather_ops = tuple(
        op
        for op in (
            _get_op("torch.ops.c10d_functional.all_gather_into_tensor.default"),
            _get_op("torch.ops._c10d_functional.all_gather_into_tensor.default"),
        )
        if op is not None
    )
    op_wait_tensor_ops = tuple(
        op
        for op in (
            _get_op("torch.ops.c10d_functional.wait_tensor.default"),
            _get_op("torch.ops._c10d_functional.wait_tensor.default"),
        )
        if op is not None
    )
    op_alias = _get_op("torch.ops.aten.alias.default")
    op_view = _get_op("torch.ops.aten.view.default")
    op_reshape = _get_op("torch.ops.aten.reshape.default")
    op_slice = _get_op("torch.ops.aten.slice.Tensor")
    op_split = _get_op("torch.ops.aten.split.Tensor")
    op_split_with_sizes = _get_op("torch.ops.aten.split_with_sizes.default")
    op_cat = _get_op("torch.ops.aten.cat.default")
    op_new_zeros = _get_op("torch.ops.aten.new_zeros.default")
    op_as_strided = _get_op("torch.ops.aten.as_strided.default")

    def _wait_tensor_if_available(tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor) or len(op_wait_tensor_ops) == 0:
            return tensor
        return op_wait_tensor_ops[0](tensor)

    def _handle_all_gather_impl(op_all_gather, args, kwargs):
        input_tensor = None
        input_idx = None
        for idx, arg in enumerate(args):
            if isinstance(arg, QuantizedTensor):
                input_tensor = arg
                input_idx = idx
                break
        if input_tensor is None:
            return op_all_gather(*args, **kwargs)
        input_idx_i = cast(int, input_idx)

        q_args = list(args)
        q_args[input_idx_i] = input_tensor._qdata.contiguous().view(torch.uint8)
        q_ret = op_all_gather(*q_args, **kwargs)
        q_bytes = q_ret if isinstance(q_ret, torch.Tensor) else q_args[input_idx_i]
        q_bytes = _wait_tensor_if_available(q_bytes)
        gathered_qdata = q_bytes.view(input_tensor._qdata.dtype)

        scale = input_tensor._params.scale
        if _is_row_scale(scale, input_tensor._qdata):
            s_args = list(args)
            s_args[input_idx_i] = scale.contiguous().view(torch.uint8)
            s_ret = op_all_gather(*s_args, **kwargs)
            s_bytes = s_ret if isinstance(s_ret, torch.Tensor) else s_args[input_idx_i]
            s_bytes = _wait_tensor_if_available(s_bytes)
            scale = s_bytes.view(scale.dtype)

        return _wrap_int8_tensor(input_tensor, gathered_qdata, scale=scale, orig_shape=tuple(gathered_qdata.shape))

    for _op_all_gather in op_all_gather_ops:

        @maybe_register(_op_all_gather)
        def handle_all_gather(qt, args, kwargs, _op=_op_all_gather):
            return _handle_all_gather_impl(_op, args, kwargs)

    def _handle_wait_tensor_impl(op_wait_tensor, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_wait_tensor(*args, **kwargs)
        waited_q_bytes = op_wait_tensor(input_tensor._qdata.view(torch.uint8), *args[1:], **kwargs)
        waited_qdata = waited_q_bytes.view(input_tensor._qdata.dtype)
        scale = input_tensor._params.scale
        if _is_row_scale(scale, input_tensor._qdata):
            waited_s_bytes = op_wait_tensor(scale.contiguous().view(torch.uint8), *args[1:], **kwargs)
            scale = waited_s_bytes.view(scale.dtype)
        return _wrap_int8_tensor(input_tensor, waited_qdata, scale=scale, orig_shape=tuple(waited_qdata.shape))

    for _op_wait_tensor in op_wait_tensor_ops:

        @maybe_register(_op_wait_tensor)
        def handle_wait_tensor(qt, args, kwargs, _op=_op_wait_tensor):
            return _handle_wait_tensor_impl(_op, args, kwargs)

    @maybe_register(op_alias)
    def handle_alias(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_alias(*args, **kwargs)
        return _wrap_int8_tensor(input_tensor, op_alias(input_tensor._qdata), orig_shape=input_tensor._params.orig_shape)

    def _handle_shape_op(op, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op(*args, **kwargs)
        new_qdata = op(input_tensor._qdata, *args[1:], **kwargs)
        scale = input_tensor._params.scale
        if _is_row_scale(scale, input_tensor._qdata) and (new_qdata.dim() == 0 or int(new_qdata.shape[0]) != int(scale.shape[0])):
            if _is_flatten_to_1d(args, kwargs):
                return _wrap_int8_tensor(input_tensor, new_qdata, scale=scale)
            raise RuntimeError(
                "INT8 rowwise/ConvRot QuantizedTensor cannot be reshaped without invalidating scale metadata. "
                f"qdata_shape={tuple(input_tensor._qdata.shape)}, new_qdata_shape={tuple(new_qdata.shape)}, "
                f"scale_shape={tuple(scale.shape)}"
            )
        return _wrap_int8_tensor(input_tensor, new_qdata)

    @maybe_register(op_view)
    def handle_view(qt, args, kwargs):
        return _handle_shape_op(op_view, args, kwargs)

    @maybe_register(op_reshape)
    def handle_reshape(qt, args, kwargs):
        return _handle_shape_op(op_reshape, args, kwargs)

    @maybe_register(op_as_strided)
    def handle_as_strided(qt, args, kwargs):
        return _handle_shape_op(op_as_strided, args, kwargs)

    @maybe_register(op_slice)
    def handle_slice(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_slice(*args, **kwargs)
        sliced_qdata = op_slice(input_tensor._qdata, *args[1:], **kwargs)
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        dim = dim if dim >= 0 else dim + input_tensor._qdata.dim()
        start = args[2] if len(args) > 2 else kwargs.get("start", None)
        end = args[3] if len(args) > 3 else kwargs.get("end", None)
        step = args[4] if len(args) > 4 else kwargs.get("step", None)
        scale = _slice_row_scale(input_tensor, dim, start, end, step)
        return _wrap_int8_tensor(input_tensor, sliced_qdata, scale=scale)

    @maybe_register(op_split)
    def handle_split(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_split(*args, **kwargs)
        chunks = op_split(input_tensor._qdata, *args[1:], **kwargs)
        dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
        dim = dim if dim >= 0 else dim + input_tensor._qdata.dim()
        scale = input_tensor._params.scale
        if dim == 0 and _is_row_scale(scale, input_tensor._qdata):
            scale_chunks = op_split(scale, args[1], 0)
            return tuple(_wrap_int8_tensor(input_tensor, chunk, scale=scale_chunk) for chunk, scale_chunk in zip(chunks, scale_chunks))
        return tuple(_wrap_int8_tensor(input_tensor, chunk) for chunk in chunks)

    @maybe_register(op_split_with_sizes)
    def handle_split_with_sizes(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_split_with_sizes(*args, **kwargs)
        chunks = op_split_with_sizes(input_tensor._qdata, *args[1:], **kwargs)
        dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
        dim = dim if dim >= 0 else dim + input_tensor._qdata.dim()
        scale = input_tensor._params.scale
        if dim == 0 and _is_row_scale(scale, input_tensor._qdata):
            scale_chunks = op_split_with_sizes(scale, args[1], 0)
            return tuple(_wrap_int8_tensor(input_tensor, chunk, scale=scale_chunk) for chunk, scale_chunk in zip(chunks, scale_chunks))
        return tuple(_wrap_int8_tensor(input_tensor, chunk) for chunk in chunks)

    @maybe_register(op_cat)
    def handle_cat(qt, args, kwargs):
        tensors = args[0]
        if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
            return op_cat(*args, **kwargs)
        for tensor in tensors:
            if not isinstance(tensor, QuantizedTensor):
                return op_cat(*args, **kwargs)
        dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
        concatenated = op_cat([tensor._qdata for tensor in tensors], *args[1:], **kwargs)
        first = tensors[0]
        scale = first._params.scale
        if dim == 0 and all(_is_row_scale(tensor._params.scale, tensor._qdata) for tensor in tensors):
            scale = op_cat([tensor._params.scale for tensor in tensors], 0)
        return _wrap_int8_tensor(first, concatenated, scale=scale, orig_shape=tuple(concatenated.shape))

    @maybe_register(op_new_zeros)
    def handle_new_zeros(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_new_zeros(*args, **kwargs)
        new_qdata = op_new_zeros(input_tensor._qdata, *args[1:], **kwargs)
        scale = input_tensor._params.scale
        if _is_row_scale(scale, input_tensor._qdata) and new_qdata.dim() > 0 and int(new_qdata.shape[0]) != int(scale.shape[0]):
            scale = scale.new_zeros((int(new_qdata.shape[0]), *tuple(scale.shape[1:])), device=new_qdata.device)
        return _wrap_int8_tensor(input_tensor, new_qdata, scale=scale)

    _PATCHED = True


def restore_int8_patches() -> None:
    global _PATCHED
    global _ORIG_LAYOUT_PRE, _ORIG_LAYOUT_POST, _PATCHED_LAYOUT, _ORIG_QT_PRE, _ORIG_QT_POST

    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as KitchenTensorWiseINT8Layout

    TensorWiseINT8Layout = _PATCHED_LAYOUT or KitchenTensorWiseINT8Layout

    if _ORIG_LAYOUT_PRE is _MISSING:
        if hasattr(TensorWiseINT8Layout, "pre_all_gather"):
            delattr(TensorWiseINT8Layout, "pre_all_gather")
    else:
        setattr(TensorWiseINT8Layout, "pre_all_gather", _ORIG_LAYOUT_PRE)

    if _ORIG_LAYOUT_POST is _MISSING:
        if hasattr(TensorWiseINT8Layout, "post_all_gather"):
            delattr(TensorWiseINT8Layout, "post_all_gather")
    else:
        setattr(TensorWiseINT8Layout, "post_all_gather", _ORIG_LAYOUT_POST)

    if _ORIG_QT_PRE is _MISSING:
        if hasattr(QuantizedTensor, "fsdp_pre_all_gather"):
            delattr(QuantizedTensor, "fsdp_pre_all_gather")
    else:
        setattr(QuantizedTensor, "fsdp_pre_all_gather", _ORIG_QT_PRE)

    if _ORIG_QT_POST is _MISSING:
        if hasattr(QuantizedTensor, "fsdp_post_all_gather"):
            delattr(QuantizedTensor, "fsdp_post_all_gather")
    else:
        setattr(QuantizedTensor, "fsdp_post_all_gather", _ORIG_QT_POST)

    _ORIG_LAYOUT_PRE = _MISSING
    _ORIG_LAYOUT_POST = _MISSING
    _PATCHED_LAYOUT = None
    _ORIG_QT_PRE = _MISSING
    _ORIG_QT_POST = _MISSING
    _PATCHED = False
