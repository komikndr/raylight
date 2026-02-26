# Extremely modified from https://github.com/komikndr/comfy-kitchen-distributed, untill it merge with main
from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

_PATCHED = False


def _get_op(path: str):
    cur = torch
    for part in path.split(".")[1:]:
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def install_fp8_patches() -> None:
    global _PATCHED
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

    QuantizedTensor.fsdp_pre_all_gather = fsdp_pre_all_gather
    QuantizedTensor.fsdp_post_all_gather = fsdp_post_all_gather

    op_all_gather = _get_op("torch.ops._c10d_functional.all_gather_into_tensor.default")
    op_wait_tensor = _get_op("torch.ops._c10d_functional.wait_tensor.default")
    op_broadcast = _get_op("torch.ops.c10d.broadcast_.default")
    op_scatter = _get_op("torch.ops.c10d.scatter_.default")

    @maybe_register(op_all_gather)
    def handle_all_gather(qt, args, kwargs):
        input_tensor = None
        input_idx = None
        for idx, arg in enumerate(args):
            if isinstance(arg, QuantizedTensor):
                input_tensor = arg
                input_idx = idx
                break

        if input_tensor is None:
            return op_all_gather(*args, **kwargs)

        qdata = input_tensor._qdata
        new_args = list(args)
        new_args[input_idx] = qdata.contiguous().view(torch.uint8)

        gathered_bytes = op_all_gather(*new_args, **kwargs)
        gathered_qdata = gathered_bytes.view(qdata.dtype)
        gathered_params = replace(input_tensor._params, orig_shape=tuple(gathered_qdata.shape))
        return QuantizedTensor(gathered_qdata, input_tensor._layout_cls, gathered_params)

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
