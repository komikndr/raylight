import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
import comfy.model_management
from comfy.weight_adapter.base import WeightAdapterBase, weight_decompose


class LoKrAdapter(WeightAdapterBase):
    name = "lokr"

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
    ) -> Optional["LoKrAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()
        lokr_w1_name = "{}.lokr_w1".format(x)
        lokr_w2_name = "{}.lokr_w2".format(x)
        lokr_w1_a_name = "{}.lokr_w1_a".format(x)
        lokr_w1_b_name = "{}.lokr_w1_b".format(x)
        lokr_t2_name = "{}.lokr_t2".format(x)
        lokr_w2_a_name = "{}.lokr_w2_a".format(x)
        lokr_w2_b_name = "{}.lokr_w2_b".format(x)

        lokr_w1 = None
        if lokr_w1_name in lora.keys():
            lokr_w1 = lora[lokr_w1_name]
            loaded_keys.add(lokr_w1_name)

        lokr_w2 = None
        if lokr_w2_name in lora.keys():
            lokr_w2 = lora[lokr_w2_name]
            loaded_keys.add(lokr_w2_name)

        lokr_w1_a = None
        if lokr_w1_a_name in lora.keys():
            lokr_w1_a = lora[lokr_w1_a_name]
            loaded_keys.add(lokr_w1_a_name)

        lokr_w1_b = None
        if lokr_w1_b_name in lora.keys():
            lokr_w1_b = lora[lokr_w1_b_name]
            loaded_keys.add(lokr_w1_b_name)

        lokr_w2_a = None
        if lokr_w2_a_name in lora.keys():
            lokr_w2_a = lora[lokr_w2_a_name]
            loaded_keys.add(lokr_w2_a_name)

        lokr_w2_b = None
        if lokr_w2_b_name in lora.keys():
            lokr_w2_b = lora[lokr_w2_b_name]
            loaded_keys.add(lokr_w2_b_name)

        lokr_t2 = None
        if lokr_t2_name in lora.keys():
            lokr_t2 = lora[lokr_t2_name]
            loaded_keys.add(lokr_t2_name)

        if (lokr_w1 is not None) or (lokr_w2 is not None) or (lokr_w1_a is not None) or (lokr_w2_a is not None):
            weights = (lokr_w1, lokr_w2, alpha, lokr_w1_a, lokr_w1_b, lokr_w2_a, lokr_w2_b, lokr_t2, dora_scale)
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
        w1 = v[0]
        w2 = v[1]
        w1_a = v[3]
        w1_b = v[4]
        w2_a = v[5]
        w2_b = v[6]
        t2 = v[7]
        dora_scale = v[8]
        dim = None

        if w1 is None:
            dim = w1_b.shape[0]
            w1 = torch.mm(
                comfy.model_management.cast_to_device(w1_a, weight.device, intermediate_dtype),
                comfy.model_management.cast_to_device(w1_b, weight.device, intermediate_dtype),
            )
        else:
            w1 = comfy.model_management.cast_to_device(w1, weight.device, intermediate_dtype)

        if w2 is None:
            dim = w2_b.shape[0]
            if t2 is None:
                w2 = torch.mm(
                    comfy.model_management.cast_to_device(w2_a, weight.device, intermediate_dtype),
                    comfy.model_management.cast_to_device(w2_b, weight.device, intermediate_dtype),
                )
            else:
                w2 = torch.einsum(
                    "i j k l, j r, i p -> p r k l",
                    comfy.model_management.cast_to_device(t2, weight.device, intermediate_dtype),
                    comfy.model_management.cast_to_device(w2_b, weight.device, intermediate_dtype),
                    comfy.model_management.cast_to_device(w2_a, weight.device, intermediate_dtype),
                )
        else:
            w2 = comfy.model_management.cast_to_device(w2, weight.device, intermediate_dtype)

        if len(w2.shape) == 4:
            w1 = w1.unsqueeze(2).unsqueeze(2)
        if v[2] is not None and dim is not None:
            alpha = v[2] / dim
        else:
            alpha = 1.0

        try:
            lora_diff = torch.kron(w1, w2).reshape(weight.shape)
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
        v = self.weights
        w1 = v[0]
        w2 = v[1]
        alpha = v[2]
        w1_a = v[3]
        w1_b = v[4]
        w2_a = v[5]
        w2_b = v[6]
        t2 = v[7]

        use_w1 = w1 is not None
        use_w2 = w2 is not None
        tucker = t2 is not None

        is_conv = getattr(self, "is_conv", False)
        conv_dim = getattr(self, "conv_dim", 0)
        kw_dict = getattr(self, "kw_dict", {}) if is_conv else {}
        op = (F.conv1d, F.conv2d, F.conv3d)[conv_dim - 1] if is_conv else F.linear

        rank = w1_b.size(0) if not use_w1 else w2_b.size(0) if not use_w2 else alpha
        scale = (alpha / rank if alpha is not None else 1.0) * getattr(self, "multiplier", 1.0)

        if use_w1:
            c = comfy.model_management.cast_to_device(w1, x.device, x.dtype)
        else:
            c = comfy.model_management.cast_to_device(w1_a, x.device, x.dtype) @ comfy.model_management.cast_to_device(
                w1_b, x.device, x.dtype
            )
        uq = c.size(1)

        if use_w2:
            ba = comfy.model_management.cast_to_device(w2, x.device, x.dtype)
        else:
            a = comfy.model_management.cast_to_device(w2_b, x.device, x.dtype)
            b = comfy.model_management.cast_to_device(w2_a, x.device, x.dtype)
            if is_conv:
                if tucker:
                    if a.dim() == 2:
                        a = a.view(*a.shape, *([1] * conv_dim))
                    if b.dim() == 2:
                        b = b.view(*b.shape, *([1] * conv_dim))
                else:
                    if b.dim() == 2:
                        b = b.view(*b.shape, *([1] * conv_dim))

        if is_conv:
            batch, _, *rest = x.shape
            h_in_group = x.reshape(batch * uq, -1, *rest)
        else:
            h_in_group = x.reshape(*x.shape[:-1], uq, -1)

        if use_w2:
            hb = op(h_in_group, ba, **kw_dict)
        else:
            if is_conv:
                if tucker:
                    t = comfy.model_management.cast_to_device(t2, x.device, x.dtype)
                    if t.dim() == 2:
                        t = t.view(*t.shape, *([1] * conv_dim))
                    ha = op(h_in_group, a)
                    ht = op(ha, t, **kw_dict)
                    hb = op(ht, b)
                else:
                    ha = op(h_in_group, a, **kw_dict)
                    hb = op(ha, b)
            else:
                ha = op(h_in_group, a)
                hb = op(ha, b)

        if is_conv:
            hb = hb.view(batch, -1, *hb.shape[1:])
            h_cross_group = hb.transpose(1, -1)
        else:
            h_cross_group = hb.transpose(-1, -2)

        hc = F.linear(h_cross_group, c)

        if is_conv:
            hc = hc.transpose(1, -1)
            out = hc.reshape(batch, -1, *hc.shape[3:])
        else:
            hc = hc.transpose(-1, -2)
            out = hc.reshape(*hc.shape[:-2], -1)

        return out * scale
