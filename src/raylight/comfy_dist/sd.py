import logging

import torch
import torch.nn.functional as F

from raylight import comfy_dist
from raylight.diffusion_models.wan.pipefusion import (
    filter_wan_state_dict_for_stage,
    partition_wan_for_pipefusion,
)
from raylight.distributed_worker.pipefusion_schema import build_stage_plan

from comfy.sd import model_detection_error_hint
from comfy import model_detection, model_management
import comfy


# I dont really want to implement this, 0.2s / step in Chroma TensorCoreFP8 FSDP
# The correct way maybe custom sharded lora and then share it
class OffsetBypassAdapter(comfy.weight_adapter.WeightAdapterBase):
    name = "lora_offset_bypass"

    def __init__(self, entries):
        self.entries = entries
        self.loaded_keys = set()
        self.weights = []
        self._warned_shapes = set()

    def _warn_shape(self, base_key, offset, delta_shape, base_shape):
        warn_key = (base_key, offset, tuple(delta_shape), tuple(base_shape))
        if warn_key in self._warned_shapes:
            return
        self._warned_shapes.add(warn_key)
        logging.warning(
            "BYPASS OFFSET SHAPE MISMATCH key=%s offset=%s delta=%s base=%s",
            base_key,
            offset,
            tuple(delta_shape),
            tuple(base_shape),
        )

    def h(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(base_out)
        out_dim = 1 if getattr(self, "is_conv", False) else base_out.ndim - 1

        for item in self.entries:
            adapter = item["adapter"]
            offset = item["offset"]
            strength = item["strength"]
            base_key = item["key"]

            prev_multiplier = getattr(adapter, "multiplier", 1.0)
            adapter.multiplier = strength
            try:
                delta = adapter.h(x, base_out)
            finally:
                adapter.multiplier = prev_multiplier

            if offset is None:
                if delta.shape == base_out.shape:
                    total = total + delta
                else:
                    self._warn_shape(base_key, offset, delta.shape, base_out.shape)
                continue

            if not isinstance(offset, tuple) or len(offset) != 3:
                self._warn_shape(base_key, offset, delta.shape, base_out.shape)
                continue

            shard_dim, start, length = offset
            if shard_dim != 0:
                self._warn_shape(base_key, offset, delta.shape, base_out.shape)
                continue

            if out_dim >= base_out.ndim:
                self._warn_shape(base_key, offset, delta.shape, base_out.shape)
                continue

            slicer = [slice(None)] * base_out.ndim
            slicer[out_dim] = slice(start, start + length)
            view = total[tuple(slicer)]

            if delta.shape == view.shape:
                view = view + delta
                total[tuple(slicer)] = view
            elif delta.shape == base_out.shape:
                total = total + delta
            else:
                self._warn_shape(base_key, offset, delta.shape, base_out.shape)

        return total


class DirectDiffBypassAdapter(comfy.weight_adapter.WeightAdapterBase):
    name = "diff_bypass"

    def __init__(self, weight_diff=None, bias_diff=None, key=None):
        self.weights = (weight_diff, bias_diff)
        self.key = key
        self._warned = False

    def _warn(self, message, *args):
        if self._warned:
            return
        self._warned = True
        logging.warning(message, *args)

    def _bias_delta(self, base_out, bias):
        if bias is None:
            return torch.zeros_like(base_out)
        bias = bias.to(device=base_out.device, dtype=base_out.dtype)
        if base_out.ndim == 0:
            return bias.reshape_as(base_out)
        shape = [1] * base_out.ndim
        if base_out.ndim == 1:
            shape[0] = bias.shape[0]
        else:
            shape[1 if getattr(self, "is_conv", False) else -1] = bias.shape[0]
        return bias.reshape(shape).expand_as(base_out)

    def h(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        weight_diff, bias_diff = self.weights
        scale = getattr(self, "multiplier", 1.0)
        if weight_diff is not None:
            weight_diff = weight_diff.to(device=x.device, dtype=x.dtype)
        if bias_diff is not None:
            bias_diff = bias_diff.to(device=x.device, dtype=x.dtype)
        if weight_diff is None:
            return self._bias_delta(base_out, bias_diff) * scale

        try:
            if getattr(self, "is_conv", False):
                conv_fn = (F.conv1d, F.conv2d, F.conv3d)[self.conv_dim - 1]
                out = conv_fn(x, weight_diff, bias_diff, **getattr(self, "kw_dict", {}))
            else:
                out = F.linear(x, weight_diff, bias_diff)
        except Exception as e:
            self._warn("DIFF BYPASS FAILED key=%s error=%s", self.key, e)
            return torch.zeros_like(base_out)

        if out.shape != base_out.shape:
            self._warn("DIFF BYPASS SHAPE MISMATCH key=%s delta=%s base=%s", self.key, tuple(out.shape), tuple(base_out.shape))
            return torch.zeros_like(base_out)
        return out * scale


class ParamDiffBypassAdapter(comfy.weight_adapter.WeightAdapterBase):
    name = "param_diff_bypass"

    def __init__(self, weight_diff=None, bias_diff=None, key=None):
        self.weights = (weight_diff, bias_diff)
        self.key = key
        self._warned = False

    def _warn(self, message, *args):
        if self._warned:
            return
        self._warned = True
        logging.warning(message, *args)

    def _patch_param(self, module, param_name, diff, scale):
        param = getattr(module, param_name, None)
        if param is None or diff is None:
            return None
        if tuple(param.shape) != tuple(diff.shape):
            self._warn(
                "PARAM DIFF BYPASS SHAPE MISMATCH key=%s.%s delta=%s base=%s",
                self.key,
                param_name,
                tuple(diff.shape),
                tuple(param.shape),
            )
            return None
        orig_data = param.data
        try:
            param.data = orig_data + (diff.to(device=orig_data.device, dtype=orig_data.dtype) * scale)
        except Exception as e:
            self._warn("PARAM DIFF BYPASS FAILED key=%s.%s error=%s", self.key, param_name, e)
            return None
        return orig_data

    def bypass_forward(self, org_forward, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        module = getattr(org_forward, "__self__", None)
        if module is None:
            self._warn("PARAM DIFF BYPASS FAILED key=%s missing bound module", self.key)
            return org_forward(x, *args, **kwargs)

        weight_diff, bias_diff = self.weights
        scale = getattr(self, "multiplier", 1.0)
        orig_weight = None
        orig_bias = None
        try:
            orig_weight = self._patch_param(module, "weight", weight_diff, scale)
            orig_bias = self._patch_param(module, "bias", bias_diff, scale)
            return org_forward(x, *args, **kwargs)
        finally:
            if orig_weight is not None:
                module.weight.data = orig_weight
            if orig_bias is not None:
                module.bias.data = orig_bias


class MergedWeightBypassAdapter(comfy.weight_adapter.WeightAdapterBase):
    name = "merged_weight_bypass"
    _logged_calls = 0
    _max_logged_calls = 16

    def __init__(self, key, weight_patches=None, bias_patches=None):
        self.key = key
        self.weight_key = f"{key}.weight"
        self.bias_key = f"{key}.bias"
        self.weight_patches = weight_patches or []
        self.bias_patches = bias_patches or []
        self.weights = []
        self._warned = False
        self._logged = False

    def _warn(self, message, *args):
        if self._warned:
            return
        self._warned = True
        logging.warning(message, *args)

    def _tensor_data(self, value):
        if value is None:
            return None
        return getattr(value, "data", value)

    def _materialize_tensor(self, value, device, dtype, label):
        tensor = self._tensor_data(value)
        if tensor is None:
            return None, False

        was_quantized = hasattr(tensor, "_qdata") and hasattr(tensor, "_layout_cls") and hasattr(tensor, "_params")
        try:
            if was_quantized:
                tensor = tensor.dequantize()
            return tensor.to(device=device, dtype=dtype, copy=True), was_quantized
        except Exception as e:
            self._warn("[Raylight LoRA][merged-forward] materialize failed key=%s.%s error=%s", self.key, label, e)
            return None, was_quantized

    def _patched_weight(self, module, x):
        weight, was_quantized = self._materialize_tensor(getattr(module, "weight", None), x.device, x.dtype, "weight")
        if weight is None:
            return None, was_quantized
        if self.weight_patches:
            weight = comfy_dist.lora.calculate_weight(
                self.weight_patches,
                weight,
                self.weight_key,
                intermediate_dtype=weight.dtype,
            )
        return weight, was_quantized

    def _patched_bias(self, module, x, out_features):
        bias_param = getattr(module, "bias", None)
        if bias_param is None:
            if not self.bias_patches:
                return None
            bias = torch.zeros((out_features,), device=x.device, dtype=x.dtype)
        else:
            bias, _ = self._materialize_tensor(bias_param, x.device, x.dtype, "bias")
            if bias is None:
                return None

        if self.bias_patches:
            bias = comfy_dist.lora.calculate_weight(
                self.bias_patches,
                bias,
                self.bias_key,
                intermediate_dtype=bias.dtype,
            )
        return bias

    def _log_first_call(self, module, x, weight, was_quantized):
        if self._logged:
            return
        self._logged = True
        if MergedWeightBypassAdapter._logged_calls >= MergedWeightBypassAdapter._max_logged_calls:
            return
        MergedWeightBypassAdapter._logged_calls += 1
        logging.warning(
            "[Raylight LoRA][merged-forward] first_call key=%s module=%s input=%s weight=%s "
            "weight_patches=%d bias_patches=%d dtype=%s device=%s quantized=%s",
            self.key,
            type(module).__name__,
            tuple(x.shape),
            tuple(weight.shape),
            len(self.weight_patches),
            len(self.bias_patches),
            weight.dtype,
            weight.device,
            was_quantized,
        )

    def bypass_forward(self, org_forward, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        module = getattr(org_forward, "__self__", None)
        if module is None:
            self._warn("[Raylight LoRA][merged-forward] missing bound module key=%s", self.key)
            return org_forward(x, *args, **kwargs)

        weight, was_quantized = self._patched_weight(module, x)
        if weight is None:
            return org_forward(x, *args, **kwargs)

        bias = self._patched_bias(module, x, weight.shape[0])
        self._log_first_call(module, x, weight, was_quantized)

        try:
            if getattr(self, "is_conv", False):
                conv_fn = (F.conv1d, F.conv2d, F.conv3d)[self.conv_dim - 1]
                return conv_fn(x, weight, bias, **getattr(self, "kw_dict", {}))
            return F.linear(x, weight, bias)
        except Exception as e:
            self._warn("[Raylight LoRA][merged-forward] forward failed key=%s error=%s", self.key, e)
            return org_forward(x, *args, **kwargs)


def _get_module_by_key(model, key):
    module = model
    try:
        for part in key.split("."):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def _diff_target_from_key(key):
    if key.endswith(".weight"):
        return key[:-7], "weight"
    if key.endswith(".bias"):
        return key[:-5], "bias"
    return None, None


def _module_key_from_weight_key(key):
    if isinstance(key, str) and key.endswith(".weight"):
        return key[:-7]
    return key


def _is_linear_or_conv_module(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        return True
    class_name = type(module).__name__.lower()
    return "linear" in class_name or "conv" in class_name


def _diff_tensor_from_patch(patch_data):
    if not isinstance(patch_data, tuple) or len(patch_data) != 2:
        return None
    patch_type, patch_payload = patch_data
    if patch_type != "diff" or not isinstance(patch_payload, tuple) or len(patch_payload) < 1:
        return None
    return patch_payload[0]


def _adapter_dora_scale(adapter):
    weights = getattr(adapter, "weights", None)
    name = getattr(adapter, "name", None)
    if not isinstance(weights, (tuple, list)):
        return None
    if name == "lora" and len(weights) > 4:
        return weights[4]
    if name == "loha" and len(weights) > 7:
        return weights[7]
    if name == "lokr" and len(weights) > 8:
        return weights[8]
    return None


def _adapter_has_dora(adapter):
    return _adapter_dora_scale(adapter) is not None


def _adapter_has_reshape(adapter):
    weights = getattr(adapter, "weights", None)
    if getattr(adapter, "name", None) != "lora" or not isinstance(weights, (tuple, list)):
        return False
    return len(weights) > 5 and weights[5] is not None


def _sample_keys(keys, limit=8):
    return [str(key) for key in list(keys)[:limit]]


def load_lora_for_models(model, lora, strength_model):
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)

    lora = comfy.lora_convert.convert_lora(lora)

    # == Change from comfy ==#
    loaded = comfy_dist.lora.load_lora(lora, key_map)
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_patches(loaded, strength_model)
    else:
        k = ()
        new_modelpatcher = None

    k = set(k)
    for x in loaded:
        if x not in k:
            logging.warning("NOT LOADED {}".format(x))

    return new_modelpatcher


def load_lora_for_models_quantized(model, lora, strength_model):
    raw_lora_key_count = len(lora)
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)

    lora = comfy.lora_convert.convert_lora(lora)
    converted_lora_key_count = len(lora)
    loaded = comfy_dist.lora.load_lora(lora, key_map)

    if model is None:
        return None

    new_modelpatcher = model.clone()

    grouped_adapters = {}
    diff_groups = {}
    unsupported_keys = []
    adapter_counts = {}
    direct_diff_counts = {"weight": 0, "bias": 0, "param": 0, "unsupported": 0}
    dora_keys = []
    reshape_keys = []
    offset_keys = []
    function_keys = []
    other_patch_counts = {}
    merged_patch_groups = {}
    sidecar_lora_count = 0

    def merged_group(module_key):
        return merged_patch_groups.setdefault(
            module_key,
            {"weight_patches": [], "bias_patches": [], "handled_keys": set()},
        )

    for loaded_key, patch_data in loaded.items():
        key = loaded_key
        offset = None
        function = None
        if isinstance(loaded_key, tuple):
            key = loaded_key[0]
            if len(loaded_key) > 1:
                offset = loaded_key[1]
            if len(loaded_key) > 2:
                function = loaded_key[2]

        if not isinstance(key, str):
            continue

        patch_name = getattr(patch_data, "name", None)
        if patch_name in {"lora", "loha", "lokr"} and hasattr(patch_data, "h"):
            adapter_counts[patch_name] = adapter_counts.get(patch_name, 0) + 1
            if _adapter_has_dora(patch_data):
                dora_keys.append(key)
            if _adapter_has_reshape(patch_data):
                reshape_keys.append(key)
            if offset is not None:
                offset_keys.append(loaded_key)
            if function is not None:
                function_keys.append(loaded_key)
            if function is not None:
                unsupported_keys.append(loaded_key)
                continue
            module_key = _module_key_from_weight_key(key)
            module = _get_module_by_key(new_modelpatcher.model, module_key)
            if module is None:
                unsupported_keys.append(loaded_key)
                continue

            if _adapter_has_dora(patch_data) or _adapter_has_reshape(patch_data):
                unsupported_keys.append(loaded_key)
                continue

            if _is_linear_or_conv_module(module):
                grouped_adapters.setdefault(module_key, []).append(
                    {
                        "adapter": patch_data,
                        "offset": offset,
                        "strength": strength_model,
                        "key": key,
                        "handled_keys": {key},
                    }
                )
                sidecar_lora_count += 1
                continue

            grouped_adapters.setdefault(module_key, []).append(
                {
                    "adapter": patch_data,
                    "offset": offset,
                    "strength": strength_model,
                    "key": key,
                    "handled_keys": {key},
                }
            )
            sidecar_lora_count += 1
            continue

        diff = _diff_tensor_from_patch(patch_data)
        if diff is not None:
            if function is not None or offset is not None:
                direct_diff_counts["unsupported"] += 1
                if function is not None:
                    function_keys.append(loaded_key)
                if offset is not None:
                    offset_keys.append(loaded_key)
                unsupported_keys.append(loaded_key)
                continue

            module_key, param_name = _diff_target_from_key(key)
            module = _get_module_by_key(new_modelpatcher.model, module_key) if module_key is not None else None
            if module is None or not hasattr(module, "weight"):
                direct_diff_counts["unsupported"] += 1
                unsupported_keys.append(loaded_key)
                continue

            if param_name in direct_diff_counts:
                direct_diff_counts[param_name] += 1
            if _is_linear_or_conv_module(module):
                adapter = DirectDiffBypassAdapter(
                    weight_diff=diff if param_name == "weight" else None,
                    bias_diff=diff if param_name == "bias" else None,
                    key=module_key,
                )
                grouped_adapters.setdefault(module_key, []).append(
                    {
                        "adapter": adapter,
                        "offset": None,
                        "strength": strength_model,
                        "key": key,
                        "handled_keys": {key},
                    }
                )
                continue

            group = diff_groups.setdefault(module_key, {"weight": None, "bias": None, "handled_keys": set()})
            group[param_name] = diff
            group["handled_keys"].add(key)
            continue

        patch_type = patch_name
        if patch_type is None and isinstance(patch_data, tuple) and len(patch_data) > 0:
            patch_type = patch_data[0]
        other_patch_counts[str(patch_type or type(patch_data).__name__)] = other_patch_counts.get(
            str(patch_type or type(patch_data).__name__), 0
        ) + 1

    for module_key, group in diff_groups.items():
        module = _get_module_by_key(new_modelpatcher.model, module_key)
        if module is None:
            unsupported_keys.extend(group["handled_keys"])
            continue

        if _is_linear_or_conv_module(module):
            adapter = DirectDiffBypassAdapter(group.get("weight"), group.get("bias"), key=module_key)
        else:
            direct_diff_counts["param"] += len(group["handled_keys"])
            if module_key in grouped_adapters:
                unsupported_keys.extend(group["handled_keys"])
                continue
            adapter = ParamDiffBypassAdapter(group.get("weight"), group.get("bias"), key=module_key)

        grouped_adapters.setdefault(module_key, []).append(
            {
                "adapter": adapter,
                "offset": None,
                "strength": strength_model,
                "key": module_key,
                "handled_keys": set(group["handled_keys"]),
            }
        )

    merged_forward_count = 0
    merged_weight_patch_count = 0
    merged_bias_patch_count = 0
    for module_key, group in merged_patch_groups.items():
        weight_patches = group["weight_patches"]
        bias_patches = group["bias_patches"]
        if not weight_patches and not bias_patches:
            continue
        adapter = MergedWeightBypassAdapter(module_key, weight_patches, bias_patches)
        grouped_adapters.setdefault(module_key, []).append(
            {
                "adapter": adapter,
                "offset": None,
                "strength": 1.0,
                "key": module_key,
                "handled_keys": set(group["handled_keys"]),
            }
        )
        merged_forward_count += 1
        merged_weight_patch_count += len(weight_patches)
        merged_bias_patch_count += len(bias_patches)

    manager = comfy.weight_adapter.BypassInjectionManager()
    loaded_keys = set()
    for key, entries in grouped_adapters.items():
        for item in entries:
            loaded_keys.update(item.get("handled_keys", {item["key"]}))
        if len(entries) == 1 and entries[0]["offset"] is None:
            manager.add_adapter(key, entries[0]["adapter"], strength=entries[0]["strength"])
            continue

        wrapper_entries = []
        for item in entries:
            wrapper_entries.append(
                {
                    "adapter": item["adapter"],
                    "offset": item["offset"],
                    "strength": item["strength"],
                    "key": item["key"],
                }
            )
        manager.add_adapter(key, OffsetBypassAdapter(wrapper_entries), strength=1.0)

    injections = manager.create_injections(new_modelpatcher.model)
    get_hook_count = getattr(manager, "get_hook_count", None)
    if get_hook_count is not None:
        hook_count = get_hook_count()
    else:
        hook_count = len(getattr(manager, "hooks", ()))

    logging.warning(
        "[Raylight LoRA][quant-bypass] keys raw=%d converted=%d key_map=%d loaded=%d strength=%s",
        raw_lora_key_count,
        converted_lora_key_count,
        len(key_map),
        len(loaded),
        strength_model,
    )
    logging.warning(
        "[Raylight LoRA][quant-bypass] adapters=%s direct_diff=%s other=%s sidecar_lora=%d grouped=%d unsupported=%d hooks=%d",
        adapter_counts,
        direct_diff_counts,
        other_patch_counts,
        sidecar_lora_count,
        len(grouped_adapters),
        len(unsupported_keys),
        hook_count,
    )
    logging.warning(
        "[Raylight LoRA][merged-forward] groups=%d weight_patches=%d bias_patches=%d",
        merged_forward_count,
        merged_weight_patch_count,
        merged_bias_patch_count,
    )
    if dora_keys:
        logging.warning(
            "[Raylight LoRA][merged-forward] DoRA adapters present: count=%d sample=%s",
            len(dora_keys),
            _sample_keys(dora_keys),
        )
    if reshape_keys:
        logging.warning(
            "[Raylight LoRA][merged-forward] reshape adapters present: count=%d sample=%s",
            len(reshape_keys),
            _sample_keys(reshape_keys),
        )
    if offset_keys:
        logging.warning(
            "[Raylight LoRA][quant-bypass] offset LoRA entries present: count=%d sample=%s",
            len(offset_keys),
            _sample_keys(offset_keys),
        )
    if function_keys:
        logging.warning(
            "[Raylight LoRA][quant-bypass] function LoRA entries are unsupported in quantized bypass: count=%d sample=%s",
            len(function_keys),
            _sample_keys(function_keys),
        )
    if len(loaded) > 0 and hook_count == 0:
        logging.warning("[Raylight LoRA][quant-bypass] loaded patches are nonzero but created bypass hooks=0")

    new_modelpatcher.set_injections("quantized_lora_bypass", injections)

    reported_keys = set()
    for key in unsupported_keys:
        normalized_key = key[0] if isinstance(key, tuple) else key
        reported_keys.add(normalized_key)
        logging.warning("SKIP LORA KEY IN QUANTIZED BYPASS MODE {}".format(key))
    for key, patch_data in loaded.items():
        normalized_key = key[0] if isinstance(key, tuple) else key
        if normalized_key not in loaded_keys and normalized_key not in reported_keys:
            patch_type = getattr(patch_data, "name", None)
            if patch_type is None and isinstance(patch_data, tuple) and len(patch_data) > 0:
                patch_type = patch_data[0]
            logging.warning("NOT LOADED IN QUANTIZED BYPASS MODE [%s] %s", patch_type or type(patch_data).__name__, key)

    return new_modelpatcher


def fsdp_load_diffusion_model(unet_path, rank, device_mesh, is_cpu_offload, model_options={}):
    use_mmap = model_options.get("use_mmap", True)
    if use_mmap and unet_path.lower().endswith(".safetensors"):
        sd, metadata = load_safetensors_mmap_with_metadata(unet_path)
    else:
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
    model, state_dict = fsdp_load_diffusion_model_stat_dict(
        sd, rank, device_mesh, is_cpu_offload, model_options=model_options, metadata=metadata
    )
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    return model, state_dict


def load_safetensors_mmap(unet_path):
    import safetensors.torch

    return safetensors.torch.load_file(unet_path, device="cpu")


def load_safetensors_mmap_with_metadata(unet_path):
    from safetensors import safe_open

    sd = load_safetensors_mmap(unet_path)
    metadata = {}
    try:
        with safe_open(unet_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception:
        metadata = {}

    return sd, metadata


def lazy_load_diffusion_model(unet_path, model_options={}):
    import comfy.sd

    from raylight.expansion.comfyui_lazytensors.lazy_tensor import wrap_state_dict_lazy
    from raylight.expansion.comfyui_lazytensors.ops import SafetensorOps

    sd = load_safetensors_mmap(unet_path)
    lazy_sd = wrap_state_dict_lazy(sd)

    load_options = model_options.copy()
    cast_dtype = load_options.pop("dtype", None)
    load_options.setdefault("custom_operations", SafetensorOps)

    model = comfy.sd.load_diffusion_model_state_dict(lazy_sd, model_options=load_options)
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))

    if cast_dtype is not None and hasattr(model, "model"):
        model.model.manual_cast_dtype = cast_dtype

    model.mmap_cache = sd
    return model


def pipefusion_load_diffusion_model_state_dict(sd, pipefusion_config, parallel_context, model_options={}, metadata=None):
    if not pipefusion_config.enabled:
        raise ValueError("PipeFusion loader requires pipefusion_enabled=True")
    if parallel_context is None:
        raise ValueError("PipeFusion loader requires an initialized xFuser parallel context")

    dtype = model_options.get("dtype", None)
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    custom_operations = model_options.get("custom_operations", None)
    if custom_operations is None:
        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)

    load_device = model_management.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "", metadata=metadata)

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else:
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype

    if model_config.quant_config is not None:
        manual_cast_dtype = model_management.unet_manual_cast(None, load_device, model_config.supported_inference_dtypes)
    else:
        manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if custom_operations is not None:
        model_config.custom_operations = custom_operations

    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    stage_plan = build_stage_plan(
        total_blocks=len(model.diffusion_model.blocks),
        rank=parallel_context.pipeline_rank,
        world_size=parallel_context.pipeline_world_size,
        config=pipefusion_config,
        group_ranks=tuple(parallel_context.pp_group().ranks),
    )
    partition_wan_for_pipefusion(model, stage_plan)

    model_patcher = comfy_dist.model_patcher.PipefusionModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        pipefusion_config=pipefusion_config,
        stage_plan=stage_plan,
        parallel_context=parallel_context,
    )
    if not model_management.is_device_cpu(offload_device):
        model.to(offload_device)

    stage_sd = filter_wan_state_dict_for_stage(new_sd, stage_plan)
    model.load_model_weights(stage_sd, "", assign=model_patcher.is_dynamic())
    return model_patcher


def pipefusion_load_diffusion_model(unet_path, pipefusion_config, parallel_context, model_options={}):
    model_options = dict(model_options)
    use_mmap = model_options.get("use_mmap", True)
    if use_mmap and unet_path.lower().endswith(".safetensors"):
        sd, metadata = load_safetensors_mmap_with_metadata(unet_path)
    else:
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)

    model = pipefusion_load_diffusion_model_state_dict(
        sd,
        pipefusion_config=pipefusion_config,
        parallel_context=parallel_context,
        model_options=model_options,
        metadata=metadata,
    )
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    model.cached_patcher_init = (pipefusion_load_diffusion_model, (unet_path, pipefusion_config, parallel_context, dict(model_options)))
    return model


def gguf_load_diffusion_model(unet_path, model_options={}, dequant_dtype=None, patch_dtype=None):
    from raylight.expansion.comfyui_gguf.ops import GGMLOps
    from raylight.expansion.comfyui_gguf.loader import gguf_sd_loader
    from raylight.expansion.comfyui_gguf.nodes import GGUFModelPatcher

    ops = GGMLOps()

    if dequant_dtype in ("default", None):
        ops.Linear.dequant_dtype = None
    elif dequant_dtype in ["target"]:
        ops.Linear.dequant_dtype = dequant_dtype
    else:
        ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

    if patch_dtype in ("default", None):
        ops.Linear.patch_dtype = None
    elif patch_dtype in ["target"]:
        ops.Linear.patch_dtype = patch_dtype
    else:
        ops.Linear.patch_dtype = getattr(torch, patch_dtype)

    use_mmap = model_options.get("use_mmap", True)

    # init model
    sd = gguf_sd_loader(unet_path, use_mmap=use_mmap)
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops})
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    model = GGUFModelPatcher.clone(model)
    # Keep mmap-backed GGUF state so the patcher can restore it on detach.
    # Related upstream idea: https://github.com/avtc/raylight/tree/fix/ram-usage-in-data-parallel-cause-oom
    model.use_mmap = use_mmap
    model.mmap_cache = sd if use_mmap else None
    model.unet_path = unet_path
    model._mmap_param_backup = None
    model.mmap_released = not use_mmap
    return model


# Disabled for now: GGUF FSDP support is blocked at the loader level.
# Keep this helper around for when the path is re-enabled.
def fsdp_gguf_load_diffusion_model(unet_path, rank, device_mesh, is_cpu_offload, model_options={}, dequant_dtype=None, patch_dtype=None):
    from raylight.expansion.comfyui_gguf.ops import GGMLOps
    from raylight.expansion.comfyui_gguf.loader import gguf_sd_loader

    ops = GGMLOps()

    if dequant_dtype in ("default", None):
        ops.Linear.dequant_dtype = None
    elif dequant_dtype in ["target"]:
        ops.Linear.dequant_dtype = dequant_dtype
    else:
        ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

    if patch_dtype in ("default", None):
        ops.Linear.patch_dtype = None
    elif patch_dtype in ["target"]:
        ops.Linear.patch_dtype = patch_dtype
    else:
        ops.Linear.patch_dtype = getattr(torch, patch_dtype)

    load_options = dict(model_options)
    load_options["custom_operations"] = ops

    sd = gguf_sd_loader(unet_path, use_mmap=model_options.get("use_mmap", True))
    model, state_dict = fsdp_load_diffusion_model_stat_dict(
        sd,
        rank,
        device_mesh,
        is_cpu_offload,
        model_options=load_options,
        metadata=None,
    )
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL %s", unet_path)
        raise RuntimeError(f"ERROR: Could not detect model type of: {unet_path}")
    return model, state_dict


def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
    steps = samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
    steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
    steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)
    pbar = comfy.utils.ProgressBar(steps)

    decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
    output = self.process_output(
        (
            comfy_dist.utils.tiled_scale(
                samples,
                decode_fn,
                tile_x // 2,
                tile_y * 2,
                overlap,
                upscale_amount=self.upscale_ratio,
                output_device=self.output_device,
                pbar=pbar,
            )
            + comfy_dist.utils.tiled_scale(
                samples,
                decode_fn,
                tile_x * 2,
                tile_y // 2,
                overlap,
                upscale_amount=self.upscale_ratio,
                output_device=self.output_device,
                pbar=pbar,
            )
            + comfy_dist.utils.tiled_scale(
                samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=self.upscale_ratio, output_device=self.output_device, pbar=pbar
            )
        )
        / 3.0
    )
    return output


def decode_tiled_1d(self, samples, tile_x=128, overlap=32):
    if samples.ndim == 3:
        decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
    else:
        og_shape = samples.shape
        samples = samples.reshape((og_shape[0], og_shape[1] * og_shape[2], -1))
        decode_fn = lambda a: self.first_stage_model.decode(
            a.reshape((-1, og_shape[1], og_shape[2], a.shape[-1])).to(self.vae_dtype).to(self.device)
        ).float()

    return self.process_output(
        comfy_dist.utils.tiled_scale_multidim(
            samples,
            decode_fn,
            tile=(tile_x,),
            overlap=overlap,
            upscale_amount=self.upscale_ratio,
            out_channels=self.output_channels,
            output_device=self.output_device,
        )
    )


def decode_tiled_3d(self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
    decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
    return self.process_output(
        comfy_dist.utils.tiled_scale_multidim(
            samples,
            decode_fn,
            tile=(tile_t, tile_x, tile_y),
            overlap=overlap,
            upscale_amount=self.upscale_ratio,
            out_channels=self.output_channels,
            index_formulas=self.upscale_index_formula,
            output_device=self.output_device,
        )
    )


##################################################
# TEMPORARY FOR PREVIOUS Fp8 forward, MODIFIYING HERE
##################################################
def fsdp_load_diffusion_model_stat_dict(sd, rank, device_mesh, is_cpu_offload, model_options={}, metadata=None):
    dtype = model_options.get("dtype", None)
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    custom_operations = model_options.get("custom_operations", None)
    if custom_operations is None:
        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)

    load_device = model_management.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "", metadata=metadata)

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:  # diffusers mmdit
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else:  # diffusers unet
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype

    if model_config.quant_config is not None:
        manual_cast_dtype = model_management.unet_manual_cast(None, load_device, model_config.supported_inference_dtypes)
    else:
        manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if custom_operations is not None:
        model_config.custom_operations = custom_operations

    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")

    model_patcher = comfy_dist.model_patcher.FSDPModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        rank=rank,
        device_mesh=device_mesh,
        is_cpu_offload=is_cpu_offload,
    )
    model.load_model_weights(new_sd, "", assign=model_patcher.is_dynamic())
    if not model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: {}".format(left_over))
    state_dict = model_patcher.model_state_dict(filter_prefix="diffusion_model.")
    model_patcher.model.diffusion_model.to("meta")
    return model_patcher, state_dict
