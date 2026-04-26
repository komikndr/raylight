# Extremely modified from https://github.com/komikndr/comfy-kitchen-distributed, untill it merge with main
from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import torch

_PATCHED = False
_MISSING = object()
_ORIG_LAYOUT_PRE = _MISSING
_ORIG_LAYOUT_POST = _MISSING
_ORIG_QT_PRE = _MISSING
_ORIG_QT_POST = _MISSING


def _get_op(path: str) -> Any:
    cur = torch
    for part in path.split(".")[1:]:
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def install_fp8_patches() -> None:
    global _PATCHED
    global _ORIG_LAYOUT_PRE, _ORIG_LAYOUT_POST, _ORIG_QT_PRE, _ORIG_QT_POST
    if _PATCHED:
        return

    from comfy_kitchen.tensor.base import (
        QuantizedTensor,
        _LAYOUT_DISPATCH_TABLE,
        dequantize_args,
        register_layout_op,
    )
    from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout
    from comfy_kitchen.scaled_mm_v2 import scaled_mm_v2

    def maybe_register(op):
        def deco(fn):
            if op is not None:
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

    def _fp8_scaled_mm(input_qdata, weight_qdata, scale_a, scale_b, bias=None, out_dtype=None):
        return scaled_mm_v2(
            input_qdata.contiguous(),
            weight_qdata,
            scale_a=scale_a,
            scale_b=scale_b,
            bias=bias,
            out_dtype=out_dtype,
        )

    @register_layout_op(torch.ops.aten.linear.default, TensorCoreFP8Layout)
    def handle_linear(qt, args, kwargs):
        input_tensor, weight = args[0], args[1]
        bias = args[2] if len(args) > 2 else None
        output_shape = None

        if isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):
            input_qdata, scale_a = TensorCoreFP8Layout.get_plain_tensors(input_tensor)
            weight_qdata, scale_b = TensorCoreFP8Layout.get_plain_tensors(weight)
            out_dtype = kwargs.get("out_dtype", input_tensor._params.orig_dtype)
        elif isinstance(input_tensor, torch.Tensor) and isinstance(weight, QuantizedTensor):
            input_qdata, input_params = TensorCoreFP8Layout.quantize(input_tensor)
            scale_a = input_params.scale
            weight_qdata, scale_b = TensorCoreFP8Layout.get_plain_tensors(weight)
            out_dtype = kwargs.get("out_dtype", input_tensor.dtype)
        else:
            return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

        if input_qdata.ndim > 2:
            output_shape = (*input_qdata.shape[:-1], weight_qdata.shape[0])
            input_qdata = input_qdata.reshape(-1, input_qdata.shape[-1])

        try:
            output = _fp8_scaled_mm(input_qdata, weight_qdata.t(), scale_a, scale_b, bias, out_dtype)
            if output_shape is not None:
                output = output.reshape(output_shape)
            return output
        except (RuntimeError, TypeError) as e:
            if isinstance(e, torch.OutOfMemoryError):
                raise
            return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    @register_layout_op(torch.ops.aten.mm.default, TensorCoreFP8Layout)
    def handle_mm(qt, args, kwargs):
        a, b = args[0], args[1]

        if isinstance(a, QuantizedTensor) and isinstance(b, QuantizedTensor):
            a_qdata, scale_a = TensorCoreFP8Layout.get_plain_tensors(a)
            b_qdata, scale_b = TensorCoreFP8Layout.get_plain_tensors(b)
            out_dtype = kwargs.get("out_dtype", a._params.orig_dtype)
        elif isinstance(a, torch.Tensor) and isinstance(b, QuantizedTensor):
            a_qdata, a_params = TensorCoreFP8Layout.quantize(a)
            scale_a = a_params.scale
            b_qdata, scale_b = TensorCoreFP8Layout.get_plain_tensors(b)
            out_dtype = kwargs.get("out_dtype", a.dtype)
        else:
            return torch.mm(*dequantize_args(args))

        try:
            return _fp8_scaled_mm(a_qdata, b_qdata, scale_a, scale_b, out_dtype=out_dtype)
        except (RuntimeError, TypeError) as e:
            if isinstance(e, torch.OutOfMemoryError):
                raise
            return torch.mm(*dequantize_args(args))

    @register_layout_op(torch.ops.aten.addmm.default, TensorCoreFP8Layout)
    def handle_addmm(qt, args, kwargs):
        bias, a, b = args[0], args[1], args[2]

        if isinstance(a, QuantizedTensor) and isinstance(b, QuantizedTensor):
            a_qdata, scale_a = TensorCoreFP8Layout.get_plain_tensors(a)
            b_qdata, scale_b = TensorCoreFP8Layout.get_plain_tensors(b)
            out_dtype = kwargs.get("out_dtype", a._params.orig_dtype)
        elif isinstance(a, torch.Tensor) and isinstance(b, QuantizedTensor):
            a_qdata, a_params = TensorCoreFP8Layout.quantize(a)
            scale_a = a_params.scale
            b_qdata, scale_b = TensorCoreFP8Layout.get_plain_tensors(b)
            out_dtype = kwargs.get("out_dtype", a.dtype)
        else:
            return torch.addmm(*dequantize_args(args))

        try:
            return _fp8_scaled_mm(a_qdata, b_qdata, scale_a, scale_b, bias, out_dtype)
        except (RuntimeError, TypeError) as e:
            if isinstance(e, torch.OutOfMemoryError):
                raise
            return torch.addmm(*dequantize_args(args))

    def pre_all_gather(qtensor: QuantizedTensor, mesh):
        qdata = qtensor._qdata
        if not qdata.is_contiguous():
            qdata = qdata.contiguous()

        scale = qtensor._params.scale
        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=qdata.device)

        return (qdata,), (scale,)

    def post_all_gather(
        qtensor: QuantizedTensor,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: QuantizedTensor | None = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        orig_shape = tuple(qtensor._params.orig_shape)

        expected_numel = 1
        for dim in orig_shape:
            expected_numel *= int(dim)

        actual_numel = int(data.numel())
        if actual_numel < expected_numel:
            raise RuntimeError(
                "FP8 FSDP post_all_gather produced insufficient storage for the upcoming "
                "as_strided materialization. "
                f"expected_numel={expected_numel}, actual_numel={actual_numel}, "
                f"orig_shape={orig_shape}, gathered_shape={tuple(data.shape)}, "
                f"dtype={data.dtype}, device={data.device}"
            )

        if out is not None:
            if not isinstance(out, QuantizedTensor):
                raise TypeError(f"Expected QuantizedTensor out, got {type(out)}")
            out._qdata = data
            out._params = TensorCoreFP8Layout.Params(
                scale=scale,
                orig_dtype=param_dtype,
                orig_shape=orig_shape,
            )
            return None

        params = TensorCoreFP8Layout.Params(
            scale=scale,
            orig_dtype=param_dtype,
            orig_shape=orig_shape,
        )
        return QuantizedTensor(data, qtensor._layout_cls, params), (data,)

    if _ORIG_LAYOUT_PRE is _MISSING:
        _ORIG_LAYOUT_PRE = getattr(TensorCoreFP8Layout, "pre_all_gather", _MISSING)
    if _ORIG_LAYOUT_POST is _MISSING:
        _ORIG_LAYOUT_POST = getattr(TensorCoreFP8Layout, "post_all_gather", _MISSING)
    setattr(TensorCoreFP8Layout, "pre_all_gather", pre_all_gather)
    setattr(TensorCoreFP8Layout, "post_all_gather", post_all_gather)

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
    op_broadcast = _get_op("torch.ops.c10d.broadcast_.default")
    op_scatter = _get_op("torch.ops.c10d.scatter_.default")

    def _wait_tensor_if_available(tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if len(op_wait_tensor_ops) == 0:
            return tensor
        return op_wait_tensor_ops[0](tensor)

    def _handle_all_gather_impl(op_all_gather, args, kwargs):
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

        assert in_idx is not None
        in_idx_i = cast(int, in_idx)

        new_args[in_idx_i] = in_qt._qdata.contiguous().view(torch.uint8)
        if out_qt is not None:
            assert out_idx is not None
            out_idx_i = cast(int, out_idx)
            new_args[out_idx_i] = out_qt._qdata.contiguous().view(torch.uint8)

        ret = op_all_gather(*new_args, **kwargs)
        if isinstance(ret, torch.Tensor):
            gathered_bytes = ret
        elif out_qt is not None:
            assert out_idx is not None
            out_idx_i = cast(int, out_idx)
            gathered_bytes = new_args[out_idx_i]
        else:
            return ret

        gathered_bytes = _wait_tensor_if_available(gathered_bytes)

        if out_qt is not None:
            gathered_qdata = gathered_bytes.view(out_qt._qdata.dtype)
            out_qt._qdata.copy_(gathered_qdata)
            out_qt._params = replace(out_qt._params, orig_shape=tuple(out_qt._qdata.shape))
            return out_qt

        gathered_qdata = gathered_bytes.view(in_qt._qdata.dtype)
        gathered_params = replace(in_qt._params, orig_shape=tuple(gathered_qdata.shape))
        return QuantizedTensor(gathered_qdata, in_qt._layout_cls, gathered_params)

    for _op_all_gather in op_all_gather_ops:

        @maybe_register(_op_all_gather)
        def handle_all_gather(qt, args, kwargs, _op=_op_all_gather):
            return _handle_all_gather_impl(_op, args, kwargs)

    def _handle_wait_tensor_impl(op_wait_tensor, args, kwargs):
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

    for _op_wait_tensor in op_wait_tensor_ops:

        @maybe_register(_op_wait_tensor)
        def handle_wait_tensor(qt, args, kwargs, _op=_op_wait_tensor):
            return _handle_wait_tensor_impl(_op, args, kwargs)

    @maybe_register(op_broadcast)
    def handle_broadcast(qt, args, kwargs):
        tensor_list = args[0]
        input_tensor = None
        input_idx = -1
        for idx, tensor in enumerate(tensor_list):
            if isinstance(tensor, QuantizedTensor):
                input_tensor = tensor
                input_idx = idx
                break

        if input_tensor is None:
            return op_broadcast(*args, **kwargs)
        if input_idx < 0:
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
    op_as_strided = _get_op("torch.ops.aten.as_strided.default")

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

    @maybe_register(op_as_strided)
    def handle_as_strided(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_as_strided(*args, **kwargs)

        try:
            new_qdata = op_as_strided(input_tensor._qdata, *args[1:], **kwargs)
        except RuntimeError as e:
            msg = str(e)
            if "out of bounds for storage" in msg or "setStorage" in msg:
                size = args[1] if len(args) > 1 else kwargs.get("size")
                stride = args[2] if len(args) > 2 else kwargs.get("stride")
                storage_offset = args[3] if len(args) > 3 else kwargs.get("storage_offset", 0)

                qdata = input_tensor._qdata
                storage_nbytes = qdata.untyped_storage().nbytes()
                itemsize = qdata.element_size()
                storage_elems = storage_nbytes // itemsize if itemsize > 0 else 0

                raise RuntimeError(
                    "FP8 as_strided storage OOB on QuantizedTensor._qdata. "
                    "This usually means FSDP unshard reconstruction produced insufficient "
                    "backing storage for the requested logical parameter view. "
                    f"requested_size={size}, requested_stride={stride}, "
                    f"storage_offset={storage_offset}, qdata_shape={tuple(qdata.shape)}, "
                    f"qdata_dtype={qdata.dtype}, qdata_storage_elems={storage_elems}, "
                    f"qdata_storage_nbytes={storage_nbytes}"
                ) from e
            raise

        return wrap_fp8_tensor(input_tensor, new_qdata)

    for path in (
        "torch.ops.aten.view.default",
        "torch.ops.aten.reshape.default",
        "torch.ops.aten.t.default",
        "torch.ops.aten.alias.default",
    ):
        op = _get_op(path)
        if op is None:
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
    global _ORIG_LAYOUT_PRE, _ORIG_LAYOUT_POST, _ORIG_QT_PRE, _ORIG_QT_POST

    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout

    if _ORIG_LAYOUT_PRE is _MISSING:
        if hasattr(TensorCoreFP8Layout, "pre_all_gather"):
            delattr(TensorCoreFP8Layout, "pre_all_gather")
    else:
        setattr(TensorCoreFP8Layout, "pre_all_gather", _ORIG_LAYOUT_PRE)

    if _ORIG_LAYOUT_POST is _MISSING:
        if hasattr(TensorCoreFP8Layout, "post_all_gather"):
            delattr(TensorCoreFP8Layout, "post_all_gather")
    else:
        setattr(TensorCoreFP8Layout, "post_all_gather", _ORIG_LAYOUT_POST)

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
    _ORIG_QT_PRE = _MISSING
    _ORIG_QT_POST = _MISSING
    _PATCHED = False
