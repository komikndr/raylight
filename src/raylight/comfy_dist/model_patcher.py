from __future__ import annotations

import collections
import logging
import gc

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.utils import _free_storage
from torch.distributed.tensor import DTensor

import comfy
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import get_key_weight, string_to_seed, move_weight_functions

from raylight import comfy_dist
from .fsdp_utils import freeze_and_detect_qt, fully_shard_bottom_up, load_from_full_model_state_dict

try:
    from comfy_kitchen.tensor.base import QuantizedTensor as _CKQuantizedTensor
except Exception:
    _CKQuantizedTensor = ()


class LowVramPatch:
    def __init__(self, key, patches, convert_func=None, set_func=None):
        self.key = key
        self.patches = patches
        self.convert_func = convert_func
        self.set_func = set_func

    def __call__(self, weight):
        return comfy_dist.lora.calculate_weight(
            self.patches[self.key],
            weight,
            self.key,
            intermediate_dtype=weight.dtype,
        )


def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights

    if hasattr(m, "weight_function"):
        m.weight_function = []

    if hasattr(m, "bias_function"):
        m.bias_function = []


def _safe_free_storage(tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        return

    try:
        if tensor.device.type == "meta":
            return
    except Exception:
        return

    try:
        _free_storage(tensor)
    except RuntimeError as e:
        msg = str(e)
        if "invalid python storage" in msg:
            return
        if "out of bounds for storage" in msg:
            return
        raise


def _is_quantized_tensor_like(tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    if _CKQuantizedTensor and isinstance(tensor, _CKQuantizedTensor):
        return True
    return hasattr(tensor, "_qdata") and hasattr(tensor, "_layout_cls") and hasattr(tensor, "_params")


def _state_dict_has_quant_payload(state_dict) -> bool:
    if not isinstance(state_dict, dict):
        return False

    for key, value in state_dict.items():
        if key.endswith(".comfy_quant") or key.endswith(".scale_weight") or key.endswith(".weight_scale"):
            return True
        if key == "scaled_fp8" or key.endswith(".scaled_fp8"):
            return True
        if _is_quantized_tensor_like(value):
            return True

    return False


def patch_fsdp(self):
    print(f"[Rank {self.rank}] Applying FSDP to {type(self.model.diffusion_model).__name__}")

    if isinstance(self.model.diffusion_model, FSDPModule):
        print("FSDP already registered, skip wrapping...")
        return self.model

    if self.fsdp_state_dict is None:
        raise ValueError("FSDP state_dict is None. Call set_fsdp_state_dict before patch_fsdp.")

    diffusion_model = self.model.diffusion_model
    fsdp_kwargs = {"reshard_after_forward": True}
    has_qt_runtime = freeze_and_detect_qt(diffusion_model)
    has_quant_sd = _state_dict_has_quant_payload(self.fsdp_state_dict)
    use_quant_loader = has_qt_runtime or has_quant_sd

    fully_shard_bottom_up(diffusion_model, fsdp_kwargs=fsdp_kwargs, native_ignore_scale=not use_quant_loader)

    if use_quant_loader:
        target_device = (
            self.load_device if isinstance(self.load_device, torch.device) else torch.device("cuda", torch.cuda.current_device())
        )
        load_from_full_model_state_dict(
            model=self.model,
            full_sd=self.fsdp_state_dict,
            device=target_device,
            strict=False,
            cpu_offload=self.is_cpu_offload,
            release_sd=False,
        )
    else:
        options = StateDictOptions(
            full_state_dict=True,
            strict=False,
            cpu_offload=self.is_cpu_offload,
            broadcast_from_rank0=True,
        )
        set_model_state_dict(self.model, self.fsdp_state_dict, options=options)

    print("FSDP registered successfully.")
    return self.model


class FSDPModelPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(
        self,
        model,
        load_device,
        offload_device,
        size=0,
        weight_inplace_update=False,
        rank: int = 0,
        fsdp_state_dict: dict | None = None,
        device_mesh=None,
        is_cpu_offload: bool = False,
    ):
        super().__init__(
            model=model,
            load_device=load_device,
            offload_device=offload_device,
            size=size,
            weight_inplace_update=weight_inplace_update,
        )
        self.rank = rank
        self.fsdp_state_dict = fsdp_state_dict
        self.device_mesh = device_mesh
        self.is_cpu_offload = is_cpu_offload
        self._has_quantized_dtensor_shards: bool | None = None
        self.patch_fsdp = patch_fsdp.__get__(self, FSDPModelPatcher)

    def is_dynamic(self):
        return True

    def config_fsdp(self, rank, device_mesh):
        self.rank = rank
        self.device_mesh = device_mesh
        self.model.diffusion_model.to("meta")

    def set_fsdp_state_dict(self, sd):
        self.fsdp_state_dict = sd

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, convert_dtensor=False):
        weight, set_func, convert_func = get_key_weight(self.model, key)
        if key not in self.patches:
            return weight

        inplace_update = self.weight_inplace_update or inplace_update

        if key not in self.backup and not return_weight:
            self.backup[key] = collections.namedtuple("Dimension", ["weight", "inplace_update"])(
                weight.to(device=self.offload_device, copy=inplace_update), inplace_update
            )

        temp_dtype = comfy.model_management.lora_compute_dtype(device_to)
        if device_to is not None:
            temp_weight = comfy.model_management.cast_to_device(weight, device_to, temp_dtype, copy=True)
        else:
            temp_weight = weight.to(temp_dtype, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = comfy_dist.lora.calculate_weight(self.patches[key], temp_weight, key, device_mesh=self.device_mesh)
        if set_func is None:
            out_weight = comfy_dist.float.stochastic_rounding(
                out_weight, weight.dtype, seed=string_to_seed(key), device_mesh=self.device_mesh
            )

            if return_weight:
                return out_weight
            if inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)

        else:
            return set_func(
                out_weight,
                inplace_update=inplace_update,
                seed=string_to_seed(key),
                return_weight=return_weight,
            )

    def clone(self, *args, **kwargs):
        # Call parent clone normally (keeps init signature correct)
        n = super(FSDPModelPatcher, self).clone(*args, **kwargs)

        n.__class__ = FSDPModelPatcher
        n.rank = self.rank
        n.fsdp_state_dict = self.fsdp_state_dict
        n.device_mesh = self.device_mesh
        n.is_cpu_offload = self.is_cpu_offload
        n._has_quantized_dtensor_shards = self._has_quantized_dtensor_shards

        return n

    def _load_list(self, prio_comfy_cast_weights=False):
        loading = []
        for n, m in self.model.named_modules():
            params = []
            skip = False
            for name, param in m.named_parameters(recurse=False):
                params.append(name)
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    skip = True  # skip random weights in non leaf modules
                    break
            if not skip and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                prepend = (not hasattr(m, "comfy_cast_weights"),) if prio_comfy_cast_weights else ()
                loading.append(prepend + (comfy.model_management.module_size(m), n, m, params))
        return loading

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            if not isinstance(self.model.diffusion_model, FSDPModule):
                self.patch_fsdp()
            self.unpatch_hooks()
            mem_counter = 0
            patch_counter = 0
            lowvram_counter = 0
            loading = self._load_list()

            loading.sort(reverse=True)
            for x in loading:
                module_mem, n, m, params = x

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if not full_load and hasattr(m, "comfy_cast_weights"):
                    if mem_counter + module_mem >= lowvram_model_memory:
                        lowvram_counter += 1
                        if hasattr(m, "prev_comfy_cast_weights"):  # Already lowvramed
                            continue

                cast_weight = self.force_cast_weights
                m.comfy_force_cast_weights = self.force_cast_weights

                if hasattr(m, "comfy_cast_weights"):
                    m.weight_function = []
                    m.bias_function = []

                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        _, set_func, convert_func = get_key_weight(self.model, weight_key)
                        m.weight_function = [LowVramPatch(weight_key, self.patches, convert_func, set_func)]
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        _, set_func, convert_func = get_key_weight(self.model, bias_key)
                        m.bias_function = [LowVramPatch(bias_key, self.patches, convert_func, set_func)]
                        patch_counter += 1

                cast_weight = True

                if cast_weight and hasattr(m, "comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True

                if weight_key in self.weight_wrapper_patches:
                    m.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.weight_wrapper_patches:
                    m.bias_function.extend(self.weight_wrapper_patches[bias_key])

                mem_counter += move_weight_functions(m, device_to)

            if lowvram_counter > 0:
                logging.info(
                    "loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter)
                )
                self.model.model_lowvram = True
            else:
                logging.info(
                    "loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load)
                )
                self.model.model_lowvram = False
                if full_load:
                    self.model.to(device_to)
                    mem_counter = self.model_size()

            self.model.lowvram_patch_counter += patch_counter
            self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
            self.model.model_offload_buffer_memory = 0
            self.model.current_weight_patches_uuid = self.patches_uuid

            for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

            self.apply_hooks(self.forced_hooks, force_apply=True)

    def cleanup(self):
        self.clean_hooks()
        if hasattr(self.model, "current_patcher"):
            self.model.current_patcher = None
        for callback in self.get_all_callbacks(CallbacksMP.ON_CLEANUP):
            callback(self)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()
            if self.model.model_lowvram:
                for m in self.model.modules():
                    move_weight_functions(m, device_to)
                    wipe_lowvram_weight(m)

                self.model.model_lowvram = False
                self.model.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            for k in keys:
                bk = self.backup[k]
                if bk.inplace_update:
                    comfy.utils.copy_to_param(self.model, k, bk.weight)
                else:
                    comfy.utils.set_attr_param(self.model, k, bk.weight)

            self.model.current_weight_patches_uuid = None
            self.backup.clear()

            if device_to is not None:
                if next(self.model.parameters()).device == torch.device("meta"):
                    pass
                else:
                    self.model.to(device_to)
                    self.model.device = device_to
            self.model.model_loaded_weight_memory = 0
            self.model.model_offload_buffer_memory = 0

            for m in self.model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            comfy.utils.set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()

    def __del__(self):
        try:
            self.detach(unpatch_all=False)
        except Exception:
            pass

        model = getattr(self, "model", None)
        if model is not None:
            try:
                has_qt_hint = self._has_quantized_dtensor_shards
                for m in model.modules():
                    for p in m.parameters(recurse=False):
                        try:
                            tensor = p.data if isinstance(p, torch.Tensor) else None
                            if tensor is None:
                                continue

                            if isinstance(tensor, DTensor):
                                if has_qt_hint is True:
                                    continue

                                try:
                                    local = getattr(tensor, "_local_tensor", None)
                                    if local is None:
                                        local = tensor.to_local()
                                except Exception:
                                    continue

                                if has_qt_hint is None:
                                    has_qt_hint = _is_quantized_tensor_like(local)
                                    self._has_quantized_dtensor_shards = has_qt_hint

                                if has_qt_hint is True:
                                    continue
                                _safe_free_storage(local.data)
                                continue

                            if _is_quantized_tensor_like(tensor):
                                continue

                            _safe_free_storage(tensor.data)
                        except Exception:
                            continue
            except Exception:
                pass

        self.model = None
        try:
            comfy.model_management.soft_empty_cache()
            gc.collect()
        except Exception:
            pass
