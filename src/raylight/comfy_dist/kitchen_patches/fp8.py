# Extremely modified from https://github.com/komikndr/comfy-kitchen-distributed, untill it merge with main
from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist

_PATCHED = False
_MISSING = object()
_ORIG_ALL_GATHER_INTO_TENSOR = None
_ORIG_ALL_GATHER = None
_ORIG_SCATTER = None
_ORIG_LAYOUT_PRE = _MISSING
_ORIG_LAYOUT_POST = _MISSING
_ORIG_QT_PRE = _MISSING
_ORIG_QT_POST = _MISSING


def _get_op(path: str):
    cur = torch
    for part in path.split(".")[1:]:
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def install_fp8_patches() -> None:
    # Doing all of this bullshit since, c10d for all_gather only available only on dynamo somehow and not eager??
    # next to do i guess, meh....
    global _PATCHED
    global _ORIG_ALL_GATHER_INTO_TENSOR, _ORIG_ALL_GATHER, _ORIG_SCATTER
    global _ORIG_LAYOUT_PRE, _ORIG_LAYOUT_POST, _ORIG_QT_PRE, _ORIG_QT_POST
    if _PATCHED:
        return

    from comfy_kitchen.tensor.base import (
        QuantizedTensor,
        _LAYOUT_DISPATCH_TABLE,
        register_layout_op,
    )
    from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout

    def is_registered(op: Any) -> bool:
        if op is None:
            return True
        if op not in _LAYOUT_DISPATCH_TABLE:
            return False
        return TensorCoreFP8Layout in _LAYOUT_DISPATCH_TABLE[op]

    def maybe_register(op):
        def deco(fn):
            if op is not None and not is_registered(op):
                register_layout_op(op, TensorCoreFP8Layout)(fn)
            return fn

        return deco

    def wrap_fp8_tensor(qtensor: QuantizedTensor, qdata: torch.Tensor) -> QuantizedTensor:
        params = TensorCoreFP8Layout.Params(
            scale=qtensor._params.scale,
            orig_dtype=qtensor._params.orig_dtype,
            orig_shape=tuple(qdata.shape),
        )
        return QuantizedTensor(qdata, qtensor._layout_cls, params)

    @classmethod
    def pre_all_gather(cls, qtensor: QuantizedTensor, mesh):
        qdata = qtensor._qdata
        if not qdata.is_contiguous():
            qdata = qdata.contiguous()

        scale = qtensor._params.scale
        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=qdata.device)

        return (qdata,), (scale,)

    @classmethod
    def post_all_gather(
        cls,
        qtensor: QuantizedTensor,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: QuantizedTensor | None = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata

        if out is not None:
            if not isinstance(out, QuantizedTensor):
                raise TypeError(f"Expected QuantizedTensor out, got {type(out)}")
            out._qdata = data
            out._params = cls.Params(
                scale=scale,
                orig_dtype=param_dtype,
                orig_shape=tuple(data.shape),
            )
            return None

        params = cls.Params(
            scale=scale,
            orig_dtype=param_dtype,
            orig_shape=tuple(data.shape),
        )
        return QuantizedTensor(data, qtensor._layout_cls, params), (data,)

    if _ORIG_LAYOUT_PRE is _MISSING:
        _ORIG_LAYOUT_PRE = getattr(TensorCoreFP8Layout, "pre_all_gather", _MISSING)
    if _ORIG_LAYOUT_POST is _MISSING:
        _ORIG_LAYOUT_POST = getattr(TensorCoreFP8Layout, "post_all_gather", _MISSING)
    TensorCoreFP8Layout.pre_all_gather = pre_all_gather
    TensorCoreFP8Layout.post_all_gather = post_all_gather

    def fsdp_pre_all_gather(self, mesh):
        return self.layout_cls.pre_all_gather(self, mesh)

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        return self.layout_cls.post_all_gather(
            self,
            all_gather_outputs,
            metadata,
            param_dtype,
            out=out,
        )

    if _ORIG_QT_PRE is _MISSING:
        _ORIG_QT_PRE = getattr(QuantizedTensor, "fsdp_pre_all_gather", _MISSING)
    if _ORIG_QT_POST is _MISSING:
        _ORIG_QT_POST = getattr(QuantizedTensor, "fsdp_post_all_gather", _MISSING)
    QuantizedTensor.fsdp_pre_all_gather = fsdp_pre_all_gather
    QuantizedTensor.fsdp_post_all_gather = fsdp_post_all_gather

    if _ORIG_ALL_GATHER_INTO_TENSOR is None:
        _ORIG_ALL_GATHER_INTO_TENSOR = dist.all_gather_into_tensor

        def all_gather_into_tensor_patched(output_tensor, input_tensor, group=None, async_op=False):
            output_qt = output_tensor if isinstance(output_tensor, QuantizedTensor) else None
            input_qt = input_tensor if isinstance(input_tensor, QuantizedTensor) else None

            use_byte_transport = output_qt is not None or input_qt is not None

            if not use_byte_transport:
                return _ORIG_ALL_GATHER_INTO_TENSOR(
                    output_tensor,
                    input_tensor,
                    group=group,
                    async_op=async_op,
                )

            output_arg = output_tensor
            input_arg = input_tensor
            if output_qt is not None:
                output_arg = output_qt._qdata.contiguous().view(torch.uint8)
            if input_qt is not None:
                input_arg = input_qt._qdata.contiguous().view(torch.uint8)

            work = _ORIG_ALL_GATHER_INTO_TENSOR(
                output_arg,
                input_arg,
                group=group,
                async_op=async_op,
            )

            if output_qt is not None and not async_op:
                output_qt._params = replace(output_qt._params, orig_shape=tuple(output_qt._qdata.shape))

            return work

        dist.all_gather_into_tensor = all_gather_into_tensor_patched

    if _ORIG_ALL_GATHER is None:
        _ORIG_ALL_GATHER = dist.all_gather

        def all_gather_patched(tensor_list, tensor, group=None, async_op=False):
            input_qt = tensor if isinstance(tensor, QuantizedTensor) else None
            output_qt_flags = [isinstance(t, QuantizedTensor) for t in tensor_list]

            if input_qt is None and not any(output_qt_flags):
                return _ORIG_ALL_GATHER(
                    tensor_list,
                    tensor,
                    group=group,
                    async_op=async_op,
                )

            input_arg = input_qt._qdata.contiguous().view(torch.uint8) if input_qt is not None else tensor
            output_arg = [t._qdata.contiguous().view(torch.uint8) if is_qt else t for t, is_qt in zip(tensor_list, output_qt_flags)]

            work = _ORIG_ALL_GATHER(
                output_arg,
                input_arg,
                group=group,
                async_op=async_op,
            )

            if not async_op:
                for t, is_qt in zip(tensor_list, output_qt_flags):
                    if is_qt:
                        t._params = replace(t._params, orig_shape=tuple(t._qdata.shape))

            return work

        dist.all_gather = all_gather_patched

    if _ORIG_SCATTER is None:
        _ORIG_SCATTER = dist.scatter

        def scatter_patched(
            tensor: torch.Tensor,
            scatter_list=None,
            src=None,
            group=None,
            async_op=False,
            group_src=None,
        ):
            output_qt = tensor if isinstance(tensor, QuantizedTensor) else None
            has_qt_input = bool(scatter_list) and any(isinstance(t, QuantizedTensor) for t in scatter_list)

            if output_qt is None and not has_qt_input:
                return _ORIG_SCATTER(
                    tensor,
                    scatter_list=scatter_list,
                    src=src,
                    group=group,
                    async_op=async_op,
                    group_src=group_src,
                )

            output_arg = output_qt._qdata.contiguous().view(torch.uint8) if output_qt is not None else tensor
            scatter_arg = scatter_list
            if scatter_list is not None:
                scatter_arg = [t._qdata.contiguous().view(torch.uint8) if isinstance(t, QuantizedTensor) else t for t in scatter_list]

            work = _ORIG_SCATTER(
                output_arg,
                scatter_list=scatter_arg,
                src=src,
                group=group,
                async_op=async_op,
                group_src=group_src,
            )

            if output_qt is not None and not async_op:
                output_qt._params = replace(output_qt._params, orig_shape=tuple(output_qt._qdata.shape))

            return work

        dist.scatter = scatter_patched

    op_all_gather = _get_op("torch.ops._c10d_functional.all_gather_into_tensor.default")
    op_wait_tensor = _get_op("torch.ops._c10d_functional.wait_tensor.default")
    op_broadcast = _get_op("torch.ops.c10d.broadcast_.default")
    op_scatter = _get_op("torch.ops.c10d.scatter_.default")

    @maybe_register(op_all_gather)
    def handle_all_gather(qt, args, kwargs):
        if len(args) == 0:
            return op_all_gather(*args, **kwargs)

        new_args = list(args)
        out_idx = None
        in_idx = None
        out_qt = None
        in_qt = None

        for idx, arg in enumerate(args):
            if isinstance(arg, QuantizedTensor):
                if out_qt is None:
                    out_qt = arg
                    out_idx = idx
                elif in_qt is None:
                    in_qt = arg
                    in_idx = idx
                    break

        if in_qt is None and out_qt is not None:
            in_qt = out_qt
            in_idx = out_idx
            out_qt = None
            out_idx = None

        if in_qt is None:
            return op_all_gather(*args, **kwargs)

        new_args[in_idx] = in_qt._qdata.contiguous().view(torch.uint8)
        if out_qt is not None:
            new_args[out_idx] = out_qt._qdata.contiguous().view(torch.uint8)

        ret = op_all_gather(*new_args, **kwargs)
        if isinstance(ret, torch.Tensor):
            gathered_bytes = ret
        elif out_qt is not None:
            gathered_bytes = new_args[out_idx]
        else:
            return ret

        if op_wait_tensor is not None:
            gathered_bytes = op_wait_tensor(gathered_bytes)

        if out_qt is not None:
            gathered_qdata = gathered_bytes.view(out_qt._qdata.dtype)
            out_qt._qdata.copy_(gathered_qdata)
            out_qt._params = replace(out_qt._params, orig_shape=tuple(out_qt._qdata.shape))
            return out_qt

        gathered_qdata = gathered_bytes.view(in_qt._qdata.dtype)
        gathered_params = replace(in_qt._params, orig_shape=tuple(gathered_qdata.shape))
        return QuantizedTensor(gathered_qdata, in_qt._layout_cls, gathered_params)

    @maybe_register(op_wait_tensor)
    def handle_wait_tensor(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_wait_tensor(*args, **kwargs)

        waited_bytes = op_wait_tensor(
            input_tensor._qdata.view(torch.uint8),
            *args[1:],
            **kwargs,
        )
        waited_qdata = waited_bytes.view(input_tensor._qdata.dtype)
        waited_params = replace(input_tensor._params, orig_shape=tuple(waited_qdata.shape))
        return QuantizedTensor(waited_qdata, input_tensor._layout_cls, waited_params)

    @maybe_register(op_broadcast)
    def handle_broadcast(qt, args, kwargs):
        tensor_list = args[0]
        input_tensor = None
        input_idx = None
        for idx, tensor in enumerate(tensor_list):
            if isinstance(tensor, QuantizedTensor):
                input_tensor = tensor
                input_idx = idx
                break

        if input_tensor is None:
            return op_broadcast(*args, **kwargs)

        new_tensor_list = list(tensor_list)
        new_tensor_list[input_idx] = input_tensor._qdata.contiguous().view(torch.uint8)
        new_args = list(args)
        new_args[0] = new_tensor_list

        broadcasted = op_broadcast(*new_args, **kwargs)
        if isinstance(broadcasted, tuple):
            tensor_list_out, work = broadcasted
        else:
            tensor_list_out = broadcasted
            work = None

        tensor_list_out = list(tensor_list_out)
        qdata_out = tensor_list_out[input_idx].view(input_tensor._qdata.dtype)
        tensor_list_out[input_idx] = wrap_fp8_tensor(input_tensor, qdata_out)

        if work is not None:
            return tensor_list_out, work
        return tensor_list_out

    @maybe_register(op_scatter)
    def handle_scatter(qt, args, kwargs):
        output_tensors = args[0]
        input_tensors = args[1]

        quantized_outputs = []
        new_output_tensors = list(output_tensors)
        for idx, tensor in enumerate(output_tensors):
            if isinstance(tensor, QuantizedTensor):
                quantized_outputs.append((idx, tensor))
                new_output_tensors[idx] = tensor._qdata.contiguous().view(torch.uint8)

        has_quantized_input = False
        new_input_tensors = []
        for entry in input_tensors:
            if isinstance(entry, (list, tuple)):
                row = []
                for tensor in entry:
                    if isinstance(tensor, QuantizedTensor):
                        has_quantized_input = True
                        row.append(tensor._qdata.contiguous().view(torch.uint8))
                    else:
                        row.append(tensor)
                new_input_tensors.append(row)
            else:
                new_input_tensors.append(entry)

        if not quantized_outputs and not has_quantized_input:
            return op_scatter(*args, **kwargs)

        result = op_scatter(new_output_tensors, new_input_tensors, *args[2:], **kwargs)
        if isinstance(result, tuple):
            output_list, work = result
        else:
            output_list = result
            work = None

        output_list = list(output_list)
        for idx, original in quantized_outputs:
            qdata = output_list[idx].view(original._qdata.dtype)
            output_list[idx] = wrap_fp8_tensor(original, qdata)

        if work is not None:
            return output_list, work
        return output_list

    op_slice = _get_op("torch.ops.aten.slice.Tensor")
    op_split = _get_op("torch.ops.aten.split.Tensor")
    op_split_with_sizes = _get_op("torch.ops.aten.split_with_sizes.default")
    op_cat = _get_op("torch.ops.aten.cat.default")
    op_new_zeros = _get_op("torch.ops.aten.new_zeros.default")

    @maybe_register(op_slice)
    def handle_slice(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_slice(*args, **kwargs)
        sliced_qdata = op_slice(input_tensor._qdata, *args[1:], **kwargs)
        return wrap_fp8_tensor(input_tensor, sliced_qdata)

    @maybe_register(op_split)
    def handle_split(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_split(*args, **kwargs)
        chunks = op_split(input_tensor._qdata, *args[1:], **kwargs)
        return tuple(wrap_fp8_tensor(input_tensor, chunk) for chunk in chunks)

    @maybe_register(op_split_with_sizes)
    def handle_split_with_sizes(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_split_with_sizes(*args, **kwargs)
        chunks = op_split_with_sizes(input_tensor._qdata, *args[1:], **kwargs)
        return tuple(wrap_fp8_tensor(input_tensor, chunk) for chunk in chunks)

    @maybe_register(op_cat)
    def handle_cat(qt, args, kwargs):
        tensors = args[0]
        if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
            return op_cat(*args, **kwargs)

        for tensor in tensors:
            if not isinstance(tensor, QuantizedTensor):
                return op_cat(*args, **kwargs)

        concatenated = op_cat([tensor._qdata for tensor in tensors], *args[1:], **kwargs)
        return wrap_fp8_tensor(tensors[0], concatenated)

    @maybe_register(op_new_zeros)
    def handle_new_zeros(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_new_zeros(*args, **kwargs)
        new_qdata = op_new_zeros(input_tensor._qdata, *args[1:], **kwargs)
        return wrap_fp8_tensor(input_tensor, new_qdata)

    for path in (
        "torch.ops.aten.view.default",
        "torch.ops.aten.reshape.default",
        "torch.ops.aten.t.default",
        "torch.ops.aten.as_strided.default",
        "torch.ops.aten.alias.default",
    ):
        op = _get_op(path)
        if op is None or is_registered(op):
            continue

        @register_layout_op(op, TensorCoreFP8Layout)
        def shape_handler(qt, args, kwargs, _op=op):
            input_tensor = args[0]
            if not isinstance(input_tensor, QuantizedTensor):
                return _op(*args, **kwargs)
            new_qdata = _op(input_tensor._qdata, *args[1:], **kwargs)
            return wrap_fp8_tensor(input_tensor, new_qdata)

    _PATCHED = True


def restore_fp8_patches() -> None:
    global _PATCHED
    global _ORIG_ALL_GATHER_INTO_TENSOR, _ORIG_ALL_GATHER, _ORIG_SCATTER
    global _ORIG_LAYOUT_PRE, _ORIG_LAYOUT_POST, _ORIG_QT_PRE, _ORIG_QT_POST

    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout

    if _ORIG_ALL_GATHER_INTO_TENSOR is not None:
        dist.all_gather_into_tensor = _ORIG_ALL_GATHER_INTO_TENSOR
        _ORIG_ALL_GATHER_INTO_TENSOR = None
    if _ORIG_ALL_GATHER is not None:
        dist.all_gather = _ORIG_ALL_GATHER
        _ORIG_ALL_GATHER = None
    if _ORIG_SCATTER is not None:
        dist.scatter = _ORIG_SCATTER
        _ORIG_SCATTER = None

    if _ORIG_LAYOUT_PRE is _MISSING:
        if hasattr(TensorCoreFP8Layout, "pre_all_gather"):
            delattr(TensorCoreFP8Layout, "pre_all_gather")
    else:
        TensorCoreFP8Layout.pre_all_gather = _ORIG_LAYOUT_PRE

    if _ORIG_LAYOUT_POST is _MISSING:
        if hasattr(TensorCoreFP8Layout, "post_all_gather"):
            delattr(TensorCoreFP8Layout, "post_all_gather")
    else:
        TensorCoreFP8Layout.post_all_gather = _ORIG_LAYOUT_POST

    if _ORIG_QT_PRE is _MISSING:
        if hasattr(QuantizedTensor, "fsdp_pre_all_gather"):
            delattr(QuantizedTensor, "fsdp_pre_all_gather")
    else:
        QuantizedTensor.fsdp_pre_all_gather = _ORIG_QT_PRE

    if _ORIG_QT_POST is _MISSING:
        if hasattr(QuantizedTensor, "fsdp_post_all_gather"):
            delattr(QuantizedTensor, "fsdp_post_all_gather")
    else:
        QuantizedTensor.fsdp_post_all_gather = _ORIG_QT_POST

    _ORIG_LAYOUT_PRE = _MISSING
    _ORIG_LAYOUT_POST = _MISSING
    _ORIG_QT_PRE = _MISSING
    _ORIG_QT_POST = _MISSING
    _PATCHED = False
