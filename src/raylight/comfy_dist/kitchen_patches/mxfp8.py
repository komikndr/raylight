from dataclasses import replace

import torch
from comfy_kitchen.float_utils import from_blocked, to_blocked
from comfy_kitchen.tensor.base import QuantizedTensor, get_layout_class, register_layout_op
from comfy_kitchen.tensor.mxfp8 import TensorCoreMXFP8Layout

_PATCHED = False
_ORIGINALS = []


def install_mxfp8_patches():
    global _PATCHED
    if _PATCHED:
        return
    layout = get_layout_class("TensorCoreMXFP8Layout")

    def pre_all_gather(qt, mesh, outer_size, outer_stride, module, mp_policy):
        rows = outer_size[0]
        chunk_rows = (rows + mesh.size() - 1) // mesh.size()
        local_rows = qt.shape[0]
        scales = from_blocked(qt._params.scale.view(torch.uint8), num_rows=qt._qdata.shape[0], num_cols=qt._qdata.shape[1] // 32)
        # Gather row-major scale bytes, then swizzle the full matrix once.
        data = qt._qdata[:local_rows].view(torch.uint8)
        scales = scales[:local_rows]
        data = torch.nn.functional.pad(data, (0, 0, 0, chunk_rows - local_rows))
        scales = torch.nn.functional.pad(scales, (0, 0, 0, chunk_rows - local_rows))
        return (data, scales), (tuple(outer_size), qt._qdata.dtype)

    def post_all_gather(qt, outputs, metadata, param_dtype, *, out=None):
        orig_shape, storage_dtype = metadata
        rows = orig_shape[0]
        data, scales = outputs
        padded_rows = ((rows + 31) // 32) * 32
        data = torch.nn.functional.pad(data[:rows], (0, 0, 0, padded_rows - rows)).view(storage_dtype)
        scales = torch.nn.functional.pad(scales[:rows], (0, 0, 0, padded_rows - rows))
        scale = to_blocked(scales, flatten=False).view(torch.float8_e8m0fnu)
        params = layout.Params(scale=scale, orig_shape=orig_shape, orig_dtype=param_dtype)
        if out is not None:
            out._qdata.copy_(data)
            out._params.scale.copy_(scale)
            return None
        return QuantizedTensor(data, qt._layout_cls, params), (data, scale)

    old_pre = getattr(QuantizedTensor, "fsdp_pre_all_gather", None)
    old_post = getattr(QuantizedTensor, "fsdp_post_all_gather", None)

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        if self._layout_cls == "TensorCoreMXFP8Layout":
            return pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy)
        return old_pre(self, mesh)

    def fsdp_post_all_gather(self, outputs, metadata, param_dtype, *, out=None):
        if self._layout_cls == "TensorCoreMXFP8Layout":
            return post_all_gather(self, outputs, metadata, param_dtype, out=out)
        return old_post(self, outputs, metadata, param_dtype, out=out)

    for name, method in (("fsdp_pre_all_gather", fsdp_pre_all_gather), ("fsdp_post_all_gather", fsdp_post_all_gather)):
        _ORIGINALS.append((name, getattr(QuantizedTensor, name, None)))
        setattr(QuantizedTensor, name, method)

    def alias(qt, args, kwargs):
        return QuantizedTensor(args[0]._qdata, args[0]._layout_cls, args[0]._params)

    def as_strided(qt, args, kwargs):
        tensor = args[0]
        size = tuple(args[1])
        stride = tuple(args[2])
        offset = args[3] if len(args) > 3 else kwargs.get("storage_offset", 0)
        if len(size) == 2 and stride == (size[1], 1) and offset == 0 and size[1] <= tensor._qdata.shape[1]:
            return QuantizedTensor(tensor._qdata, tensor._layout_cls, replace(tensor._params, orig_shape=size))
        return torch.as_strided(tensor.dequantize(), size, stride, offset)

    def view(qt, args, kwargs):
        tensor = args[0]
        size = tuple(args[1])
        if size == (-1,):
            size = (tensor.numel(),)
        if size == tuple(tensor.shape) or size == (tensor.numel(),):
            return QuantizedTensor(tensor._qdata, tensor._layout_cls, replace(tensor._params, orig_shape=size))
        return torch.reshape(tensor.dequantize(), size)

    for cls in {layout, TensorCoreMXFP8Layout}:
        register_layout_op(torch.ops.aten.alias.default, cls)(alias)
        register_layout_op(torch.ops.aten.as_strided.default, cls)(as_strided)
        register_layout_op(torch.ops.aten.view.default, cls)(view)
    _PATCHED = True


def restore_mxfp8_patches():
    global _PATCHED
    for name, original in reversed(_ORIGINALS):
        if original is None:
            delattr(QuantizedTensor, name)
        else:
            setattr(QuantizedTensor, name, original)
    _ORIGINALS.clear()
    _PATCHED = False
