import torch
import comfy.model_management
import comfy.ops
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple

from raylight.comfy_dist.lora import calculate_weight as ray_calculate_weight

from .lazy_tensor import LazySafetensor


def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    if isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    if isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    return item


class SafetensorLayer(torch.nn.Module):
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight_key = f"{prefix}weight"
        weight = state_dict.get(weight_key)

        if isinstance(weight, LazySafetensor) or isinstance(self, torch.nn.Linear):
            return self.eager_load_params(state_dict, prefix, *args, **kwargs)

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def eager_load_params(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for key, value in state_dict.items():
            if key.startswith(prefix):
                attr_name = key[prefix_len:]
                if attr_name == "weight":
                    self.weight = torch.nn.Parameter(value, requires_grad=False)
                elif attr_name == "bias" and value is not None:
                    self.bias = torch.nn.Parameter(value, requires_grad=False)

        if self.weight is None:
            missing_keys.append(prefix + "weight")

    def get_weight(self, tensor, dtype, device):
        weight = tensor.to(device)

        patches = getattr(tensor, "patches", [])
        if patches:
            patch_list = []
            last_key = None
            for patch, key in patches:
                patch_list += move_patch_to_device(patch, device)
                last_key = key
            weight = ray_calculate_weight(patch_list, weight, last_key)

        return weight

    def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)

        if self.bias is not None:
            bias = self.get_weight(self.bias, bias_dtype, device)
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

        weight = self.get_weight(self.weight, dtype, device)
        weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias


def _init_conv2d_metadata(layer, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, padding_mode):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    dilation = _pair(dilation)

    if isinstance(padding, str):
        padding_value = padding
        reversed_padding = [0, 0] * len(kernel_size)
        if padding == "same":
            for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
                total_padding = d * (k - 1)
                left_pad = total_padding // 2
                reversed_padding[2 * i] = left_pad
                reversed_padding[2 * i + 1] = total_padding - left_pad
    else:
        padding_value = _pair(padding)
        reversed_padding = _reverse_repeat_tuple(padding_value, 2)

    layer.in_channels = in_channels
    layer.out_channels = out_channels
    layer.kernel_size = kernel_size
    layer.stride = stride
    layer.padding = padding_value
    layer.dilation = dilation
    layer.transposed = False
    layer.output_padding = (0,) * len(kernel_size)
    layer.groups = groups
    layer.padding_mode = padding_mode
    layer._reversed_padding_repeated_twice = reversed_padding


class SafetensorOps(comfy.ops.manual_cast):
    class Linear(SafetensorLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(SafetensorLayer, comfy.ops.manual_cast.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None):
            torch.nn.Module.__init__(self)
            _init_conv2d_metadata(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, padding_mode)
            self.weight = None
            self.bias = None

        def forward(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Embedding(SafetensorLayer, comfy.ops.manual_cast.Embedding):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.sparse = sparse
            self.weight = None

        def forward(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight is not None and self.weight.dtype in (torch.float16, torch.bfloat16):
                out_dtype = None

            weight = self.get_weight(self.weight, None, input.device)
            return torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)

    class LayerNorm(SafetensorLayer, comfy.ops.manual_cast.LayerNorm):
        def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            self.weight = None
            self.bias = None

        def forward(self, input):
            if self.weight is None:
                return super().forward(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

    class GroupNorm(SafetensorLayer, comfy.ops.manual_cast.GroupNorm):
        def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.num_groups = num_groups
            self.num_channels = num_channels
            self.eps = eps
            self.affine = affine
            self.weight = None
            self.bias = None

        def forward(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)
