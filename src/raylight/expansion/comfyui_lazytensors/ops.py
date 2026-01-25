"""Custom operations for zero-copy safetensor loading.

Implements custom layers that override _load_from_state_dict to perform
pointer assignment instead of data copying. This allows LazySafetensor
wrappers to replace module parameters without triggering materialization/copy.
"""
import torch
import comfy.ops
import comfy.lora
import comfy.model_management
from .lazy_tensor import LazySafetensor
from raylight.comfy_dist.lora import calculate_weight as ray_calculate_weight

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    return item

class SafetensorLayer(torch.nn.Module):
    """Mixin for layers that handle LazySafetensor assignment."""
    
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight_key = f"{prefix}weight"
        bias_key = f"{prefix}bias"
        weight = state_dict.get(weight_key)
        
        # CRITICAL: If weight is a lazy tensor, assign it directly!
        # Standard _load_from_state_dict performs .copy_() which forces materialization
        if isinstance(weight, LazySafetensor) or isinstance(self, torch.nn.Linear):
             return self.eager_load_params(state_dict, prefix, *args, **kwargs)
             
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def eager_load_params(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Manually assign parameters to avoid copy overhead."""
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k.startswith(prefix):
                 # Remove prefix to identify attribute
                 attr_name = k[prefix_len:]
                 if attr_name == "weight":
                     self.weight = torch.nn.Parameter(v, requires_grad=False)
                 elif attr_name == "bias" and v is not None:
                     self.bias = torch.nn.Parameter(v, requires_grad=False)
                 # Handle internal layers/submodules if necessary? 
                 # Usually state_dict is flat, so this loop mainly hits weight/bias for this layer
        
        # Handle missing keys logic similar to PyTorch
        if self.weight is None:
             if isinstance(self, torch.nn.Linear):
                  # Create dummy if needed, but usually we expect weight
                  pass
             missing_keys.append(prefix + "weight")

    def get_weight(self, tensor, dtype, device):
        """Get weight applying patches on GPU if needed."""
        # Transfer to device (triggers lazy materialization)
        weight = tensor.to(device)
        
        # Check for patches (ComfyUI attaches .patches list to tensor)
        patches = getattr(tensor, "patches", [])
        if len(patches) > 0:
            # Move patches to GPU
            patch_list = []
            for p, key in patches:
                patch_list += move_patch_to_device(p, device)
            
            # Apply patches on GPU
            # Using ray_calculate_weight (same as GGUF) for consistent LoRA math
            weight = ray_calculate_weight(patch_list, weight, key)
            
        return weight

    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)
        
        # Handle bias
        if s.bias is not None:
            # Use get_weight to handle potential lazy + patches on bias (rare but possible)
            bias = s.get_weight(s.bias, bias_dtype, device)
            bias = comfy.ops.cast_to(
                bias, bias_dtype, device, non_blocking=non_blocking, copy=False
            )

        # Handle weight - this triggers LazySafetensorTensor.to() which materializes
        # AND applies patches if present
        weight = s.get_weight(s.weight, dtype, device)
        weight = comfy.ops.cast_to(
            weight, dtype, device, non_blocking=non_blocking, copy=False
        )
        return weight, bias


class SafetensorOps(comfy.ops.manual_cast):
    """Operations factory that produces zero-copy aware layers."""
    
    class Linear(SafetensorLayer, comfy.ops.manual_cast.Linear):
        def forward(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(SafetensorLayer, comfy.ops.manual_cast.Conv2d):
        def forward(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)
            
    class Embedding(SafetensorLayer, comfy.ops.manual_cast.Embedding):
        def forward(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            
            # Use get_weight logic (via cast_bias_weight internal logic or direct)
            # Embedding doesn't use cast_bias_weight standardly in Comfy ops, so we adapt:
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
        def forward(self, input):
            if self.weight is None:
                return super().forward(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

    class GroupNorm(SafetensorLayer, comfy.ops.manual_cast.GroupNorm):
        def forward(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(
                input, self.num_groups, weight, bias, self.eps
            )
