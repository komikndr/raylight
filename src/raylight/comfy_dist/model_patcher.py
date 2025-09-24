from __future__ import annotations

import collections

import torch
import gc
from torch.distributed.utils import _free_storage
from torch.distributed.fsdp import FSDPModule

import comfy
from comfy.model_patcher import (get_key_weight,
                                 string_to_seed,
                                 move_weight_functions)

from raylight import comfy_dist
from comfy.patcher_extension import CallbacksMP
from torch.distributed.tensor import DTensor
import logging

def model_size_mb(model: torch.nn.Module) -> float:
    """
    Return total model size in MB (parameters + buffers).
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**2)


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


    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            self.unpatch_hooks()
            mem_counter = 0
            patch_counter = 0
            lowvram_counter = 0
            loading = self._load_list()

            load_completely = []
            loading.sort(reverse=True)
            for x in loading:
                n = x[1]
                m = x[2]
                params = x[3]
                module_mem = x[0]

                lowvram_weight = False

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if not full_load and hasattr(m, "comfy_cast_weights"):
                    if mem_counter + module_mem >= lowvram_model_memory:
                        lowvram_weight = True
                        lowvram_counter += 1
                        if hasattr(m, "prev_comfy_cast_weights"):  # Already lowvramed
                            continue

                # This single line, take my entire week
                lowvram_weight = True
                cast_weight = self.force_cast_weights
                if lowvram_weight:
                    if hasattr(m, "comfy_cast_weights"):
                        m.weight_function = []
                        m.bias_function = []

                    if weight_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(weight_key)
                        else:
                            m.weight_function = [LowVramPatch(weight_key, self.patches)]
                            patch_counter += 1
                    if bias_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(bias_key)
                        else:
                            m.bias_function = [LowVramPatch(bias_key, self.patches)]
                            patch_counter += 1

                    cast_weight = True
                else:
                    if hasattr(m, "comfy_cast_weights"):
                        wipe_lowvram_weight(m)

                    if full_load or mem_counter + module_mem < lowvram_model_memory:
                        mem_counter += module_mem
                        load_completely.append((module_mem, n, m, params))

                if cast_weight and hasattr(m, "comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True

                if weight_key in self.weight_wrapper_patches:
                    m.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.weight_wrapper_patches:
                    m.bias_function.extend(self.weight_wrapper_patches[bias_key])

                mem_counter += move_weight_functions(m, device_to)

            load_completely.sort(reverse=True)
            for x in load_completely:
                n = x[1]
                m = x[2]
                params = x[3]
                if hasattr(m, "comfy_patched_weights"):
                    if m.comfy_patched_weights == True:
                        continue

                for param in params:
                    self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)

                logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
                m.comfy_patched_weights = True

            for x in load_completely:
                x[2].to(device_to)

            if lowvram_counter > 0:
                logging.info("loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter))
                self.model.model_lowvram = True
            else:
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

    def config_fsdp(self, rank, device_mesh):
        self.rank = rank
        self.device_mesh = device_mesh
        self.model.to("meta")

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = FSDPModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = FSDPModelPatcher
        self.__class__ = src_cls

        # inject fsdp-specific defaults if missing
        if not hasattr(n, "device_mesh"):
            n.device_mesh = None
        if not hasattr(n, "rank"):
            n.rank = None

        if src_cls != FSDPModelPatcher:
            n.size = 0
        return n

    # def __del__(self):
    #     print(self.model.__dict__.keys())
    #     for m in self.model.modules():
    #         if isinstance(m, FSDPModule):
    #             _free_storage(m._modules._handle.data)
    #     del self.model
    #     gc.collect()
    #     comfy.model_management.soft_empty_cache()
    def __del__(self):
        print(f"[DEL] Before cleanup: {model_size_mb(self.model):.2f} MB")
        self.detach(unpatch_all=False)

        for m in self.model.modules():
            for p in m.parameters(recurse=False):
                if isinstance(p, DTensor):
                    local = p.to_local()
                    _free_storage(local.data)
                elif isinstance(p, torch.Tensor):
                    _free_storage(p.data)

        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print("[DEL] After cleanup (gc + cuda.empty_cache done)")
        print(f"[DEL] CUDA mem allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"[DEL] CUDA mem reserved : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
