# GGUF quantization patches for Raylight FSDP operations
from __future__ import annotations

from typing import Any, cast

import torch

_QK_K = 256

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


def install_gguf_patches() -> None:
    global _PATCHED
    global _ORIG_LAYOUT_PRE, _ORIG_LAYOUT_POST, _ORIG_QT_PRE, _ORIG_QT_POST
    if _PATCHED:
        return

    from comfy_kitchen.tensor.base import QuantizedTensor, register_layout_op
    from comfy_kitchen.tensor.gguf import TensorCoreGGUFLayout

    def maybe_register(op):
        def deco(fn):
            if op is not None:
                register_layout_op(op, TensorCoreGGUFLayout)(fn)
            return fn

        return deco

    def _extract_collective_result(result):
        if isinstance(result, tuple):
            return result[0], result[1]
        return result, None

    class _CompositeWork:
        def __init__(self, *works):
            self._works = [work for work in works if work is not None]

        def wait(self):
            result = None
            for work in self._works:
                result = work.wait()
            return result

        def is_completed(self):
            return all(getattr(work, "is_completed", lambda: True)() for work in self._works)

    def _wrap_gguf_tensor(
        qtensor,
        qdata: torch.Tensor,
        *,
        super_block_scale_scale=None,
        super_block_min_scale=None,
        quantized_block_scale=None,
        quantized_block_min=None,
        orig_shape=None,
        n_blocks_per_superblock=None,
    ):
        params = TensorCoreGGUFLayout.Params(
            scale=qtensor._params.scale,
            orig_dtype=qtensor._params.orig_dtype,
            orig_shape=orig_shape if orig_shape is not None else qtensor._params.orig_shape,
            n_blocks_per_superblock=n_blocks_per_superblock
            if n_blocks_per_superblock is not None
            else qtensor._params.n_blocks_per_superblock,
            super_block_scale_scale=super_block_scale_scale
            if super_block_scale_scale is not None
            else qtensor._params.super_block_scale_scale,
            super_block_min_scale=super_block_min_scale
            if super_block_min_scale is not None
            else qtensor._params.super_block_min_scale,
            quantized_block_scale=quantized_block_scale
            if quantized_block_scale is not None
            else qtensor._params.quantized_block_scale,
            quantized_block_min=quantized_block_min
            if quantized_block_min is not None
            else qtensor._params.quantized_block_min,
        )
        return QuantizedTensor(qdata, qtensor._layout_cls, params)

    def _normalize_slice_args(size: int, start, end, step) -> tuple[int, int, int]:
        step = 1 if step is None else step
        if step != 1:
            raise NotImplementedError("GGUF only supports slice step=1")
        start = 0 if start is None else start
        end = size if end is None else end
        if start < 0:
            start += size
        if end < 0:
            end += size
        start = max(0, min(start, size))
        end = max(start, min(end, size))
        return start, end, step

    def _dequantize_arg(arg):
        if isinstance(arg, QuantizedTensor):
            return arg.dequantize()
        return arg

    def pre_all_gather(qtensor: QuantizedTensor, mesh):
        qdata = qtensor._qdata.contiguous()
        metadata = {
            "scale": qtensor._params.scale,
            "orig_dtype": qtensor._params.orig_dtype,
            "orig_shape": qtensor._params.orig_shape,
            "n_blocks_per_superblock": qtensor._params.n_blocks_per_superblock,
            "super_block_scale_scale": qtensor._params.super_block_scale_scale,
            "super_block_min_scale": qtensor._params.super_block_min_scale,
            "quantized_block_scale": qtensor._params.quantized_block_scale,
            "quantized_block_min": qtensor._params.quantized_block_min,
            "qdata_shape": tuple(qdata.shape),
        }
        return (qdata,), metadata

    def post_all_gather(
        qtensor: QuantizedTensor, all_gather_outputs, metadata: Any, param_dtype: torch.dtype, *, out=None
    ):
        gathered_qdata = all_gather_outputs[0] if isinstance(all_gather_outputs, tuple) else all_gather_outputs
        orig_shape = metadata.get("orig_shape")
        qdata_shape = metadata.get("qdata_shape")
        if orig_shape is not None and qdata_shape is not None:
            orig_shape = tuple(orig_shape)
            qdata_shape = tuple(qdata_shape)
            if len(orig_shape) == 2 and len(qdata_shape) == 2 and gathered_qdata.dim() == 2:
                old_storage_rows = int(qdata_shape[0])
                new_storage_rows = int(gathered_qdata.shape[0])
                if old_storage_rows > 0 and new_storage_rows != old_storage_rows:
                    new_logical_rows = (int(orig_shape[0]) * new_storage_rows + old_storage_rows - 1) // old_storage_rows
                    orig_shape = (new_logical_rows, int(orig_shape[1]))
        params = TensorCoreGGUFLayout.Params(
            scale=metadata["scale"],
            orig_dtype=metadata.get("orig_dtype", param_dtype),
            orig_shape=orig_shape,
            n_blocks_per_superblock=metadata["n_blocks_per_superblock"],
            super_block_scale_scale=metadata["super_block_scale_scale"],
            super_block_min_scale=metadata["super_block_min_scale"],
            quantized_block_scale=metadata["quantized_block_scale"],
            quantized_block_min=metadata["quantized_block_min"],
        )
        if out is not None:
            if not isinstance(out, QuantizedTensor):
                raise TypeError(f"Expected QuantizedTensor out, got {type(out)}")
            out._qdata = gathered_qdata
            out._params = params
            return None
        return QuantizedTensor(gathered_qdata, qtensor._layout_cls, params), (gathered_qdata,)

    if _ORIG_LAYOUT_PRE is _MISSING:
        _ORIG_LAYOUT_PRE = getattr(TensorCoreGGUFLayout, "pre_all_gather", _MISSING)
    if _ORIG_LAYOUT_POST is _MISSING:
        _ORIG_LAYOUT_POST = getattr(TensorCoreGGUFLayout, "post_all_gather", _MISSING)
    setattr(TensorCoreGGUFLayout, "pre_all_gather", pre_all_gather)
    setattr(TensorCoreGGUFLayout, "post_all_gather", post_all_gather)

    def fsdp_pre_all_gather(self, mesh):
        return self.layout_cls.pre_all_gather(self, mesh)

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
    op_broadcast = _get_op("torch.ops.c10d.broadcast_.default")
    op_scatter = _get_op("torch.ops.c10d.scatter_.default")
    op_alias = _get_op("torch.ops.aten.alias.default")
    op_view = _get_op("torch.ops.aten.view.default")
    op_view_as = _get_op("torch.ops.aten.view_as.default")
    op_reshape = _get_op("torch.ops.aten.reshape.default")
    op_slice = _get_op("torch.ops.aten.slice.Tensor")
    op_split = _get_op("torch.ops.aten.split.Tensor")
    op_split_with_sizes = _get_op("torch.ops.aten.split_with_sizes.default")
    op_cat = _get_op("torch.ops.aten.cat.default")
    op_new_zeros = _get_op("torch.ops.aten.new_zeros.default")
    op_as_strided = _get_op("torch.ops.aten.as_strided.default")

    def _wait_tensor_if_available(tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if len(op_wait_tensor_ops) == 0:
            return tensor
        return op_wait_tensor_ops[0](tensor)

    def _handle_all_gather_impl(op_all_gather, args, kwargs):
        if len(args) == 0:
            return op_all_gather(*args, **kwargs)

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
        return _wrap_gguf_tensor(input_tensor, gathered_qdata)

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
        return _wrap_gguf_tensor(input_tensor, waited_qdata)

    for _op_wait_tensor in op_wait_tensor_ops:

        @maybe_register(_op_wait_tensor)
        def handle_wait_tensor(qt, args, kwargs, _op=_op_wait_tensor):
            return _handle_wait_tensor_impl(_op, args, kwargs)

    @maybe_register(op_broadcast)
    def handle_broadcast(qt, args, kwargs):
        tensor_list = args[0]
        quantized_entries = [(idx, tensor) for idx, tensor in enumerate(tensor_list) if isinstance(tensor, QuantizedTensor)]
        if not quantized_entries:
            return op_broadcast(*args, **kwargs)

        q_tensor_list = list(tensor_list)
        for idx, tensor in quantized_entries:
            q_tensor_list[idx] = tensor._qdata.contiguous().view(torch.uint8)

        q_result = op_broadcast(q_tensor_list, *args[1:], **kwargs)
        q_list, q_work = _extract_collective_result(q_result)

        output_list = list(q_list)
        for idx, original in quantized_entries:
            output_list[idx] = _wrap_gguf_tensor(
                original,
                q_list[idx].view(original._qdata.dtype),
                orig_shape=original._params.orig_shape,
            )
        if q_work is not None:
            return output_list, _CompositeWork(q_work)
        return output_list

    @maybe_register(op_scatter)
    def handle_scatter(qt, args, kwargs):
        output_tensors = args[0]
        input_tensors = args[1]

        quantized_outputs = []
        new_q_outputs = list(output_tensors)
        for idx, tensor in enumerate(output_tensors):
            if isinstance(tensor, QuantizedTensor):
                quantized_outputs.append((idx, tensor))
                new_q_outputs[idx] = tensor._qdata.contiguous().view(torch.uint8)

        has_quantized_input = False
        q_inputs = []
        for entry in input_tensors:
            if isinstance(entry, (list, tuple)):
                q_entry = []
                for tensor in entry:
                    if isinstance(tensor, QuantizedTensor):
                        has_quantized_input = True
                        q_entry.append(tensor._qdata.contiguous().view(torch.uint8))
                    else:
                        q_entry.append(tensor)
                q_inputs.append(q_entry)
            else:
                q_inputs.append(entry)

        if not quantized_outputs and not has_quantized_input:
            return op_scatter(*args, **kwargs)

        q_result = op_scatter(new_q_outputs, q_inputs, *args[2:], **kwargs)
        q_list, q_work = _extract_collective_result(q_result)

        output_list = list(q_list)
        for idx, original in quantized_outputs:
            output_list[idx] = _wrap_gguf_tensor(
                original,
                q_list[idx].view(original._qdata.dtype),
                orig_shape=original._params.orig_shape,
            )
        if q_work is not None:
            return output_list, _CompositeWork(q_work)
        return output_list

    @maybe_register(op_alias)
    def handle_alias(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_alias(*args, **kwargs)
        aliased_qdata = op_alias(input_tensor._qdata)
        return _wrap_gguf_tensor(
            input_tensor,
            aliased_qdata,
            orig_shape=input_tensor._params.orig_shape,
        )

    def _handle_same_shape_view(op, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op(*args, **kwargs)
        return handle_alias(None, (input_tensor,), {})

    def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(shape) == 0:
            return ()
        stride = [1] * len(shape)
        for idx in range(len(shape) - 2, -1, -1):
            stride[idx] = max(1, int(shape[idx + 1])) * stride[idx + 1]
        return tuple(stride)

    def _can_alias_as_logical_shape(
        input_tensor, requested_size: tuple[int, ...], requested_stride: tuple[int, ...], storage_offset: int
    ) -> bool:
        if int(storage_offset) != 0:
            return False
        if requested_stride != _contiguous_stride(requested_size):
            return False
        if len(requested_size) != 2 or input_tensor._qdata.dim() != 2:
            return False
        return tuple(input_tensor._qdata.shape) == TensorCoreGGUFLayout.get_storage_shape(requested_size)

    def _alias_with_orig_shape(input_tensor, orig_shape: tuple[int, ...]):
        aliased_qdata = op_alias(input_tensor._qdata)
        return _wrap_gguf_tensor(
            input_tensor,
            aliased_qdata,
            orig_shape=orig_shape,
        )

    @maybe_register(op_view)
    def handle_view(qt, args, kwargs):
        return _handle_same_shape_view(op_view, args, kwargs)

    @maybe_register(op_view_as)
    def handle_view_as(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_view_as(*args, **kwargs)
        return handle_alias(None, (input_tensor,), {})

    @maybe_register(op_reshape)
    def handle_reshape(qt, args, kwargs):
        return _handle_same_shape_view(op_reshape, args, kwargs)

    @maybe_register(op_as_strided)
    def handle_as_strided(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_as_strided(*args, **kwargs)

        size = args[1] if len(args) > 1 else kwargs.get("size")
        stride = args[2] if len(args) > 2 else kwargs.get("stride")
        storage_offset = args[3] if len(args) > 3 else kwargs.get("storage_offset", 0)

        requested_size = tuple(int(dim) for dim in size)
        requested_stride = tuple(int(dim) for dim in stride)
        orig_shape = tuple(int(dim) for dim in input_tensor._params.orig_shape)

        if requested_size == orig_shape and requested_stride == _contiguous_stride(orig_shape) and int(storage_offset) == 0:
            return _alias_with_orig_shape(input_tensor, orig_shape)

        if _can_alias_as_logical_shape(input_tensor, requested_size, requested_stride, int(storage_offset)):
            return _alias_with_orig_shape(input_tensor, requested_size)

        return op_as_strided(*[_dequantize_arg(arg) for arg in args], **kwargs)

    @maybe_register(op_slice)
    def handle_slice(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_slice(*args, **kwargs)

        dim = args[1] if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else None
        end = args[3] if len(args) > 3 else None
        step = args[4] if len(args) > 4 else None
        dim = dim if dim >= 0 else dim + len(input_tensor._params.orig_shape)

        if dim == 0:
            start, end, _ = _normalize_slice_args(int(input_tensor._params.orig_shape[0]), start, end, step)
            start_storage = (start * input_tensor._qdata.shape[0]) // int(input_tensor._params.orig_shape[0])
            end_storage = (end * input_tensor._qdata.shape[0] + int(input_tensor._params.orig_shape[0]) - 1) // int(
                input_tensor._params.orig_shape[0]
            )
            sliced_qdata = torch.ops.aten.slice.Tensor(input_tensor._qdata, 0, start_storage, end_storage, 1)
            return _wrap_gguf_tensor(input_tensor, sliced_qdata, orig_shape=(end - start, int(input_tensor._params.orig_shape[1])))
        if dim == 1:
            start, end, step = _normalize_slice_args(int(input_tensor._params.orig_shape[1]), start, end, step)
            if start == 0 and end == int(input_tensor._params.orig_shape[1]) and step == 1:
                return handle_alias(qt, (input_tensor,), {})
        return op_slice(*[_dequantize_arg(arg) for arg in args], **kwargs)

    @maybe_register(op_split)
    def handle_split(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_split(*args, **kwargs)

        split_size = args[1]
        dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
        dim = dim if dim >= 0 else dim + len(input_tensor._params.orig_shape)
        if dim != 0 or not isinstance(split_size, int):
            return op_split(*[_dequantize_arg(arg) for arg in args], **kwargs)

        logical_rows = int(input_tensor._params.orig_shape[0])
        chunks = []
        for start in range(0, logical_rows, split_size):
            end = min(start + split_size, logical_rows)
            start_storage = (start * input_tensor._qdata.shape[0]) // logical_rows
            end_storage = (end * input_tensor._qdata.shape[0] + logical_rows - 1) // logical_rows
            sliced_qdata = torch.ops.aten.slice.Tensor(input_tensor._qdata, 0, start_storage, end_storage, 1)
            chunks.append(_wrap_gguf_tensor(input_tensor, sliced_qdata, orig_shape=(end - start, int(input_tensor._params.orig_shape[1]))))
        return tuple(chunks)

    @maybe_register(op_split_with_sizes)
    def handle_split_with_sizes(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_split_with_sizes(*args, **kwargs)

        split_sizes = args[1]
        dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
        dim = dim if dim >= 0 else dim + len(input_tensor._params.orig_shape)
        if dim != 0:
            return op_split_with_sizes(*[_dequantize_arg(arg) for arg in args], **kwargs)

        chunks = []
        start = 0
        for split_size in split_sizes:
            end = start + int(split_size)
            start_storage = (start * input_tensor._qdata.shape[0]) // int(input_tensor._params.orig_shape[0])
            end_storage = (end * input_tensor._qdata.shape[0] + int(input_tensor._params.orig_shape[0]) - 1) // int(
                input_tensor._params.orig_shape[0]
            )
            sliced_qdata = torch.ops.aten.slice.Tensor(input_tensor._qdata, 0, start_storage, end_storage, 1)
            chunks.append(_wrap_gguf_tensor(input_tensor, sliced_qdata, orig_shape=(end - start, int(input_tensor._params.orig_shape[1]))))
            start = end
        return tuple(chunks)

    @maybe_register(op_cat)
    def handle_cat(qt, args, kwargs):
        tensors = args[0]
        dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
        if dim != 0 or not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
            return op_cat(*args, **kwargs)
        for tensor in tensors:
            if not isinstance(tensor, QuantizedTensor):
                return op_cat(*args, **kwargs)
        first = tensors[0]
        if any(tensor._params.orig_shape[1] != first._params.orig_shape[1] for tensor in tensors):
            return op_cat(*[_dequantize_arg(arg) for arg in args], **kwargs)

        concatenated_qdata = op_cat([tensor._qdata for tensor in tensors], 0)
        orig_rows = sum(int(tensor._params.orig_shape[0]) for tensor in tensors)
        return _wrap_gguf_tensor(
            first,
            concatenated_qdata,
            orig_shape=(orig_rows, int(first._params.orig_shape[1])),
        )

    @maybe_register(op_new_zeros)
    def handle_new_zeros(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return op_new_zeros(*args, **kwargs)
        size = tuple(args[1]) if len(args) > 1 else tuple(input_tensor._params.orig_shape)
        if len(size) != 2:
            return op_new_zeros(*[_dequantize_arg(arg) for arg in args], **kwargs)
        device = kwargs.get("device", input_tensor._qdata.device)
        storage_shape = TensorCoreGGUFLayout.get_storage_shape(size)
        qdata = torch.zeros(storage_shape, device=device, dtype=input_tensor._qdata.dtype)
        return _wrap_gguf_tensor(input_tensor, qdata, orig_shape=size)

    _PATCHED = True


def restore_gguf_patches() -> None:
    global _PATCHED
    global _ORIG_LAYOUT_PRE, _ORIG_LAYOUT_POST, _ORIG_QT_PRE, _ORIG_QT_POST

    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.gguf import TensorCoreGGUFLayout

    if _ORIG_LAYOUT_PRE is _MISSING:
        if hasattr(TensorCoreGGUFLayout, "pre_all_gather"):
            delattr(TensorCoreGGUFLayout, "pre_all_gather")
    else:
        setattr(TensorCoreGGUFLayout, "pre_all_gather", _ORIG_LAYOUT_PRE)

    if _ORIG_LAYOUT_POST is _MISSING:
        if hasattr(TensorCoreGGUFLayout, "post_all_gather"):
            delattr(TensorCoreGGUFLayout, "post_all_gather")
    else:
        setattr(TensorCoreGGUFLayout, "post_all_gather", _ORIG_LAYOUT_POST)

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
