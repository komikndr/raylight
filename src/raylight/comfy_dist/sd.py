import logging

import torch

from raylight import comfy_dist

import comfy
from comfy.sd import model_detection_error_hint
from comfy import model_detection, model_management


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


def fsdp_load_diffusion_model_stat_dict(sd, rank, device_mesh, is_cpu_offload, model_options={}):
    dtype = model_options.get("dtype", None)
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(
        sd, {diffusion_model_prefix: ""}, filter_keys=True
    )
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)

    load_device = model_management.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "")

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
    if model_config.scaled_fp8 is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=unet_weight_dtype,
            weight_dtype=weight_dtype,
        )
    else:
        unet_dtype = dtype

    manual_cast_dtype = model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get(
        "custom_operations", model_config.custom_operations
    )
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: {}".format(left_over))

    model_patcher = comfy_dist.model_patcher.FSDPModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        rank=rank,
        device_mesh=device_mesh,
        is_cpu_offload=is_cpu_offload,
    )
    state_dict = model_patcher.model_state_dict()
    model_patcher.model.to("meta")

    return model_patcher, state_dict


def fsdp_load_diffusion_model(unet_path, rank, device_mesh, is_cpu_offload, model_options={}):
    sd = comfy.utils.load_torch_file(unet_path)
    model, state_dict = fsdp_load_diffusion_model_stat_dict(sd, rank, device_mesh, is_cpu_offload, model_options=model_options)
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    return model, state_dict


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

    sd = comfy.utils.load_torch_file(unet_path)
    model, state_dict = fsdp_load_diffusion_model_stat_dict(sd, rank, device_mesh, is_cpu_offload, model_options={"custom_operations": OPS})
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
            comfy_dist.utils.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=self.upscale_ratio, output_device=self.output_device, pbar=pbar) +
            comfy_dist.utils.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=self.upscale_ratio, output_device=self.output_device, pbar=pbar) +
            comfy_dist.utils.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=self.upscale_ratio, output_device=self.output_device, pbar=pbar))
        / 3.0)
    return output


def decode_tiled_1d(self, samples, tile_x=128, overlap=32):
    if samples.ndim == 3:
        decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
    else:
        og_shape = samples.shape
        samples = samples.reshape((og_shape[0], og_shape[1] * og_shape[2], -1))
        decode_fn = lambda a: self.first_stage_model.decode(a.reshape((-1, og_shape[1], og_shape[2], a.shape[-1])).to(self.vae_dtype).to(self.device)).float()

    return self.process_output(comfy_dist.utils.tiled_scale_multidim(samples, decode_fn, tile=(tile_x,), overlap=overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, output_device=self.output_device))


def decode_tiled_3d(self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
    decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
    return self.process_output(comfy_dist.utils.tiled_scale_multidim(samples, decode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, index_formulas=self.upscale_index_formula, output_device=self.output_device))
