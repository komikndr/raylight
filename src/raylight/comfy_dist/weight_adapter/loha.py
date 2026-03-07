import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
import comfy.model_management
from comfy.weight_adapter.base import WeightAdapterBase, weight_decompose


class LoHaAdapter(WeightAdapterBase):
    name = "loha"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, torch.Tensor],
        alpha: float,
        dora_scale: torch.Tensor,
        loaded_keys: set[str] = None,
    ) -> Optional["LoHaAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()

        hada_w1_a_name = "{}.hada_w1_a".format(x)
        hada_w1_b_name = "{}.hada_w1_b".format(x)
        hada_w2_a_name = "{}.hada_w2_a".format(x)
        hada_w2_b_name = "{}.hada_w2_b".format(x)
        hada_t1_name = "{}.hada_t1".format(x)
        hada_t2_name = "{}.hada_t2".format(x)
        if hada_w1_a_name in lora.keys():
            hada_t1 = None
            hada_t2 = None
            if hada_t1_name in lora.keys():
                hada_t1 = lora[hada_t1_name]
                hada_t2 = lora[hada_t2_name]
                loaded_keys.add(hada_t1_name)
                loaded_keys.add(hada_t2_name)

            weights = (
                lora[hada_w1_a_name],
                lora[hada_w1_b_name],
                alpha,
                lora[hada_w2_a_name],
                lora[hada_w2_b_name],
                hada_t1,
                hada_t2,
                dora_scale,
            )
            loaded_keys.add(hada_w1_a_name)
            loaded_keys.add(hada_w1_b_name)
            loaded_keys.add(hada_w2_a_name)
            loaded_keys.add(hada_w2_b_name)
            return cls(loaded_keys, weights)
        else:
            return None

    def calculate_weight(
        self,
        weight,
        key,
        strength,
        strength_model,
        offset,
        function,
        intermediate_dtype=torch.float32,
        original_weight=None,
        device_mesh=None,
    ):
        v = self.weights
        w1a = v[0]
        w1b = v[1]
        if v[2] is not None:
            alpha = v[2] / w1b.shape[0]
        else:
            alpha = 1.0

        w2a = v[3]
        w2b = v[4]
        dora_scale = v[7]
        if v[5] is not None:  # cp decomposition
            t1 = v[5]
            t2 = v[6]
            m1 = torch.einsum(
                "i j k l, j r, i p -> p r k l",
                comfy.model_management.cast_to_device(t1, weight.device, intermediate_dtype),
                comfy.model_management.cast_to_device(w1b, weight.device, intermediate_dtype),
                comfy.model_management.cast_to_device(w1a, weight.device, intermediate_dtype),
            )

            m2 = torch.einsum(
                "i j k l, j r, i p -> p r k l",
                comfy.model_management.cast_to_device(t2, weight.device, intermediate_dtype),
                comfy.model_management.cast_to_device(w2b, weight.device, intermediate_dtype),
                comfy.model_management.cast_to_device(w2a, weight.device, intermediate_dtype),
            )
        else:
            m1 = torch.mm(
                comfy.model_management.cast_to_device(w1a, weight.device, intermediate_dtype),
                comfy.model_management.cast_to_device(w1b, weight.device, intermediate_dtype),
            )
            m2 = torch.mm(
                comfy.model_management.cast_to_device(w2a, weight.device, intermediate_dtype),
                comfy.model_management.cast_to_device(w2b, weight.device, intermediate_dtype),
            )

        try:
            lora_diff = (m1 * m2).reshape(weight.shape)
            if dora_scale is not None:
                weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function)
            else:
                if isinstance(weight, DTensor):
                    weight += DTensor.from_local(function(((strength * alpha) * lora_diff).type(weight.dtype)), device_mesh)
                else:
                    weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
        return weight

    def h(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        func_list = [None, None, F.linear, F.conv1d, F.conv2d, F.conv3d]

        v = self.weights
        w1a = comfy.model_management.cast_to_device(v[0], x.device, x.dtype)
        w1b = comfy.model_management.cast_to_device(v[1], x.device, x.dtype)
        alpha = v[2]
        w2a = comfy.model_management.cast_to_device(v[3], x.device, x.dtype)
        w2b = comfy.model_management.cast_to_device(v[4], x.device, x.dtype)
        t1 = v[5]
        t2 = v[6]

        rank = w1b.shape[0]
        scale = (alpha / rank if alpha is not None else 1.0) * getattr(self, "multiplier", 1.0)

        is_conv = getattr(self, "is_conv", False)
        conv_dim = getattr(self, "conv_dim", 0)
        kw_dict = getattr(self, "kw_dict", {}) if is_conv else {}

        if t1 is not None and t2 is not None:
            t1 = comfy.model_management.cast_to_device(t1, x.device, x.dtype)
            t2 = comfy.model_management.cast_to_device(t2, x.device, x.dtype)
            m1 = torch.einsum("i j k l, j r, i p -> p r k l", t1, w1b, w1a)
            m2 = torch.einsum("i j k l, j r, i p -> p r k l", t2, w2b, w2a)
            diff_weight = (m1 * m2) * scale
        else:
            diff_weight = (w1a @ w1b) * (w2a @ w2b) * scale

        if is_conv:
            op = func_list[conv_dim + 2]
            kernel_size = getattr(self, "kernel_size", (1,) * conv_dim)
            in_channels = getattr(self, "in_channels", None)
            if diff_weight.dim() == 2:
                if in_channels is not None:
                    diff_weight = diff_weight.view(diff_weight.shape[0], in_channels, *kernel_size)
                else:
                    diff_weight = diff_weight.view(*diff_weight.shape, *([1] * conv_dim))
        else:
            op = F.linear

        return op(x, diff_weight, **kw_dict)
