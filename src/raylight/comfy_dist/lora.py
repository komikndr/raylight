# This file only contains swapped function from ComfyUI model_patcher.py into raylight equivalent

from __future__ import annotations

import logging

import torch
from torch.distributed.tensor import DTensor

import comfy

import raylight.comfy_dist.weight_adapter as weight_adapter
from comfy.lora import pad_tensor_to_shape


def load_lora(lora, to_load, log_missing=True):
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        dora_scale_name = "{}.dora_scale".format(x)
        dora_scale = None
        if dora_scale_name in lora.keys():
            dora_scale = lora[dora_scale_name]
            loaded_keys.add(dora_scale_name)

        for adapter_cls in weight_adapter.adapters:
            adapter = adapter_cls.load(x, lora, alpha, dora_scale, loaded_keys)
            if adapter is not None:
                patch_dict[to_load[x]] = adapter
                loaded_keys.update(adapter.loaded_keys)
                continue

        w_norm_name = "{}.w_norm".format(x)
        b_norm_name = "{}.b_norm".format(x)
        w_norm = lora.get(w_norm_name, None)
        b_norm = lora.get(b_norm_name, None)

        if w_norm is not None:
            loaded_keys.add(w_norm_name)
            patch_dict[to_load[x]] = ("diff", (w_norm,))
            if b_norm is not None:
                loaded_keys.add(b_norm_name)
                patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = ("diff", (b_norm,))

        diff_name = "{}.diff".format(x)
        diff_weight = lora.get(diff_name, None)
        if diff_weight is not None:
            patch_dict[to_load[x]] = ("diff", (diff_weight,))
            loaded_keys.add(diff_name)

        diff_bias_name = "{}.diff_b".format(x)
        diff_bias = lora.get(diff_bias_name, None)
        if diff_bias is not None:
            patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = ("diff", (diff_bias,))
            loaded_keys.add(diff_bias_name)

        set_weight_name = "{}.set_weight".format(x)
        set_weight = lora.get(set_weight_name, None)
        if set_weight is not None:
            patch_dict[to_load[x]] = ("set", (set_weight,))
            loaded_keys.add(set_weight_name)

    if log_missing:
        for x in lora.keys():
            if x not in loaded_keys:
                logging.warning("lora key not loaded: {}".format(x))

    return patch_dict


def calculate_weight(patches, weight, key, intermediate_dtype=torch.float32, original_weights=None, device_mesh=None):
    for p in patches:
        strength = p[0]
        v = p[1]
        strength_model = p[2]
        offset = p[3]
        function = p[4]
        if function is None:
            function = lambda a: a

        old_weight = None
        if offset is not None:
            old_weight = weight
            weight = weight.narrow(offset[0], offset[1], offset[2])

        if strength_model != 1.0:
            weight *= strength_model

        if isinstance(v, list):
            v = (
                calculate_weight(
                    v[1:],
                    v[0][1](comfy.model_management.cast_to_device(v[0][0], weight.device, intermediate_dtype, copy=True), inplace=True),
                    key,
                    intermediate_dtype=intermediate_dtype,
                    device_mesh=device_mesh,
                ),
            )

        if isinstance(v, weight_adapter.WeightAdapterBase):
            output = v.calculate_weight(
                weight, key, strength, strength_model, offset, function, intermediate_dtype, original_weights, device_mesh
            )
            if output is None:
                logging.warning("Calculate Weight Failed: {} {}".format(v.name, key))
            else:
                weight = output
                if old_weight is not None:
                    weight = old_weight
            continue

        if len(v) == 1:
            patch_type = "diff"
        elif len(v) == 2:
            patch_type = v[0]
            v = v[1]

        if patch_type == "diff":
            diff: torch.Tensor = v[0]
            # An extra flag to pad the weight if the diff's shape is larger than the weight
            do_pad_weight = len(v) > 1 and v[1]["pad_weight"]
            if do_pad_weight and diff.shape != weight.shape:
                logging.info("Pad weight {} from {} to shape: {}".format(key, weight.shape, diff.shape))
                weight = pad_tensor_to_shape(weight, diff.shape)

            if strength != 0.0:
                if diff.shape != weight.shape:
                    logging.warning("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, diff.shape, weight.shape))
                else:
                    add_term = function(strength * comfy.model_management.cast_to_device(diff, weight.device, weight.dtype))
                    if isinstance(weight, DTensor):
                        if device_mesh is not None:
                            weight += DTensor.from_local(add_term, device_mesh)
                        else:
                            weight += add_term
                    else:
                        weight += add_term
        elif patch_type == "set":
            weight.copy_(v[0])
        elif patch_type == "model_as_lora":
            target_weight: torch.Tensor = v[0]
            diff_weight = comfy.model_management.cast_to_device(
                target_weight, weight.device, intermediate_dtype
            ) - comfy.model_management.cast_to_device(original_weights[key][0][0], weight.device, intermediate_dtype)

            add_term = function(strength * comfy.model_management.cast_to_device(diff_weight, weight.device, weight.dtype))

            if isinstance(weight, DTensor):
                if device_mesh is not None:
                    weight += DTensor.from_local(add_term, device_mesh)
                else:
                    weight += add_term
            else:
                weight += add_term
        else:
            logging.warning("patch type not recognized {} {}".format(patch_type, key))

        if old_weight is not None:
            weight = old_weight

    return weight
