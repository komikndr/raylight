import logging

import torch

from raylight import comfy_dist

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
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)

    lora = comfy.lora_convert.convert_lora(lora)
    loaded = comfy_dist.lora.load_lora(lora, key_map)

    if model is None:
        return None

    new_modelpatcher = model.clone()

    grouped_adapters = {}
    unsupported_keys = []
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

        if getattr(patch_data, "name", None) in {"lora", "loha", "lokr"} and hasattr(patch_data, "h"):
            if function is not None:
                unsupported_keys.append(loaded_key)
                continue
            grouped_adapters.setdefault(key, []).append(
                {
                    "adapter": patch_data,
                    "offset": offset,
                }
            )

    manager = comfy.weight_adapter.BypassInjectionManager()
    loaded_keys = set()
    for key, entries in grouped_adapters.items():
        loaded_keys.add(key)
        if len(entries) == 1 and entries[0]["offset"] is None:
            manager.add_adapter(key, entries[0]["adapter"], strength=strength_model)
            continue

        wrapper_entries = []
        for item in entries:
            wrapper_entries.append(
                {
                    "adapter": item["adapter"],
                    "offset": item["offset"],
                    "strength": strength_model,
                    "key": key,
                }
            )
        manager.add_adapter(key, OffsetBypassAdapter(wrapper_entries), strength=1.0)

    injections = manager.create_injections(new_modelpatcher.model)
    new_modelpatcher.set_injections("quantized_lora_bypass", injections)

    for key in unsupported_keys:
        logging.warning("SKIP FUNCTIONAL LORA KEY IN QUANTIZED BYPASS MODE {}".format(key))
    for key, patch_data in loaded.items():
        normalized_key = key[0] if isinstance(key, tuple) else key
        if normalized_key not in loaded_keys:
            patch_type = getattr(patch_data, "name", None)
            if patch_type is None and isinstance(patch_data, tuple) and len(patch_data) > 0:
                patch_type = patch_data[0]
            logging.warning("NOT LOADED IN QUANTIZED BYPASS MODE [%s] %s", patch_type or type(patch_data).__name__, key)

    return new_modelpatcher


def fsdp_load_diffusion_model(unet_path, rank, device_mesh, is_cpu_offload, model_options={}):
    use_mmap = model_options.get("use_mmap", True)
    if use_mmap and unet_path.lower().endswith(".safetensors"):
        sd, metadata = load_safetensors_mmap(unet_path, return_metadata=True)
    else:
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
    model, state_dict = fsdp_load_diffusion_model_stat_dict(
        sd, rank, device_mesh, is_cpu_offload, model_options=model_options, metadata=metadata
    )
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    return model, state_dict


def load_safetensors_mmap(unet_path, return_metadata=False):
    import safetensors.torch
    from safetensors import safe_open

    sd = safetensors.torch.load_file(unet_path, device="cpu")
    if not return_metadata:
        return sd

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

    # init model
    sd = gguf_sd_loader(unet_path)
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops})
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    model = GGUFModelPatcher.clone(model)
    return model


def fsdp_bnb_load_diffusion_model(unet_path, rank, device_mesh, is_cpu_offload, model_options={}):
    from raylight.expansion.comfyui_bnb import OPS

    use_mmap = model_options.get("use_mmap", True)
    if use_mmap and unet_path.lower().endswith(".safetensors"):
        sd, metadata = load_safetensors_mmap(unet_path, return_metadata=True)
    else:
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
    model, state_dict = fsdp_load_diffusion_model_stat_dict(sd, rank, device_mesh, is_cpu_offload, model_options={"custom_operations": OPS, "use_mmap": use_mmap}, metadata=metadata)
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    return model, state_dict


def bnb_load_diffusion_model(unet_path, model_options={}):
    from raylight.expansion.comfyui_bnb import OPS

    sd = comfy.utils.load_torch_file(unet_path)
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": OPS})
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    return model


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
