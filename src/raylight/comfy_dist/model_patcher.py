from __future__ import annotations

import collections
import logging

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.utils import _free_storage
from torch.distributed.tensor import DTensor

import comfy
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import (get_key_weight,
                                 string_to_seed,
                                 move_weight_functions)

from raylight import comfy_dist
from .fsdp_registry import patch_fsdp


class LowVramPatch:
    def __init__(self, key, patches):
        self.key = key
        self.patches = patches

    def __call__(self, weight):
        intermediate_dtype = weight.dtype
        if intermediate_dtype not in [torch.float32, torch.float16, torch.bfloat16]:   # intermediate_dtype has to be one that is supported in math ops
            intermediate_dtype = torch.float32
            return comfy.float.stochastic_rounding(comfy.lora.calculate_weight(self.patches[self.key], weight.to(intermediate_dtype), self.key, intermediate_dtype=intermediate_dtype), weight.dtype, seed=string_to_seed(self.key))

        return comfy_dist.lora.calculate_weight(self.patches[self.key], weight, self.key, intermediate_dtype=intermediate_dtype)


def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights

    if hasattr(m, "weight_function"):
        m.weight_function = []

    if hasattr(m, "bias_function"):
        m.bias_function = []


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
        self.patch_fsdp = patch_fsdp.__get__(self, FSDPModelPatcher)

    def config_fsdp(self, rank, device_mesh):
        self.rank = rank
        self.device_mesh = device_mesh
        self.model.to("meta")

    def set_fsdp_state_dict(self, sd):
        self.fsdp_state_dict = sd

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, convert_dtensor=False):
        inplace_update = True
        if key not in self.patches:
            return
        weight, set_func, convert_func = get_key_weight(self.model, key)
        inplace_update = self.weight_inplace_update or inplace_update

        if key not in self.backup:
            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

        if device_to is not None:
            temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = comfy_dist.lora.calculate_weight(self.patches[key], temp_weight, key, device_mesh=self.device_mesh)
        if set_func is None:
            out_weight = comfy_dist.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key), device_mesh=self.device_mesh)

            if inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)

        else:
            set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

    def clone(self, *args, **kwargs):
        # Call parent clone normally (keeps init signature correct)
        n = super(FSDPModelPatcher, self).clone(*args, **kwargs)

        n.__class__ = FSDPModelPatcher
        n.rank = self.rank
        n.fsdp_state_dict = self.fsdp_state_dict
        n.device_mesh = self.device_mesh
        n.is_cpu_offload = self.is_cpu_offload

        return n

    def _load_list(self):
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
                loading.append((comfy.model_management.module_size(m), n, m, params))
        return loading

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            if not isinstance(self.model.diffusion_model, FSDPModule):
                self.patch_fsdp()
            else:
                pass
            self.unpatch_hooks()

            mem_counter = 0
            patch_counter = 0
            load_completely = self._load_list()
            load_completely.sort(reverse=True)
            for x in load_completely:
                n = x[1]
                m = x[2]
                params = x[3]
                module_mem = x[0]

                if hasattr(m, "comfy_patched_weights"):
                    if m.comfy_patched_weights is True:
                        continue

                for param in params:
                    self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)

                logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
                m.comfy_patched_weights = True
                mem_counter += module_mem

            for x in load_completely:
                x[2].to(device_to)

            logging.info("loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
            self.model.model_lowvram = False
            if full_load:
                self.model.to(device_to)
                mem_counter = self.model_size()

            self.model.lowvram_patch_counter += patch_counter
            self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
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

            for m in self.model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            comfy.utils.set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()

    def __del__(self):
        self.detach(unpatch_all=False)
        for m in self.model.modules():
            for p in m.parameters(recurse=False):
                if isinstance(p, DTensor):
                    local = p.to_local()
                    _free_storage(local.data)
                elif isinstance(p, torch.Tensor):
                    _free_storage(p.data)

        self.model.to("meta")
        del self.model
        self.model = None
        comfy.model_management.soft_empty_cache()
