import copy
import math

import comfy.model_patcher
import comfy.model_sampling
import comfy.samplers
import torch
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention
from einops import rearrange

from raylight.comfy_extra_dist.ray_patch_decorator import ray_patch

from .guiders import parse_skip_blocks, parse_stg_layers_indices


def _make_ray_guider(ray_actors, guider_type, **kwargs):
    guider = {"ray_actors": ray_actors, "type": guider_type}
    guider.update(kwargs)
    return guider


def _parse_float_list(values):
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def _parse_int_layers(block_indices, default_layers=None):
    if isinstance(block_indices, dict):
        return [int(layer) for layer in block_indices.get("layers", [])]
    if isinstance(block_indices, (list, tuple, set)):
        return [int(layer) for layer in block_indices]
    if isinstance(block_indices, str) and block_indices.strip():
        return [int(value.strip()) for value in block_indices.split(",") if value.strip()]
    return list(default_layers or [])


def _get_transformer_options_copy(model):
    model_options = model.model_options.copy()
    transformer_options = model_options.get("transformer_options", {}).copy()
    model_options["transformer_options"] = transformer_options
    return model_options, transformer_options


def _adain_batch_normalize(latents, reference, factor, per_frame=False):
    latents_copy = copy.deepcopy(latents)
    reference = copy.deepcopy(reference)
    samples = latents_copy["samples"]

    if per_frame:
        if reference["samples"].size(2) == 1:
            reference["samples"] = reference["samples"].repeat(1, 1, samples.size(2), 1, 1)
        elif samples.size(2) > reference["samples"].size(2):
            raise ValueError("Latents have more frames than reference")

    normalized = samples.clone()
    for batch_index in range(normalized.size(0)):
        for channel_index in range(normalized.size(1)):
            if not per_frame:
                ref_std, ref_mean = torch.std_mean(reference["samples"][batch_index, channel_index], dim=None)
                cur_std, cur_mean = torch.std_mean(normalized[batch_index, channel_index], dim=None)
                normalized[batch_index, channel_index] = ((normalized[batch_index, channel_index] - cur_mean) / cur_std) * ref_std + ref_mean
                continue

            for frame_index in range(normalized.size(2)):
                ref_std, ref_mean = torch.std_mean(reference["samples"][batch_index, channel_index, frame_index], dim=None)
                cur_std, cur_mean = torch.std_mean(normalized[batch_index, channel_index, frame_index], dim=None)
                normalized[batch_index, channel_index, frame_index] = ((normalized[batch_index, channel_index, frame_index] - cur_mean) / cur_std) * ref_std + ref_mean

    latents_copy["samples"] = torch.lerp(latents["samples"], normalized, factor)
    return latents_copy


def _statistical_normalize(latents, target_mean, target_std, percentile, factor, clip_outliers):
    latents_copy = copy.deepcopy(latents)
    normalized = latents_copy["samples"].clone()
    lower_percentile = (100 - percentile) / 2
    upper_percentile = 100 - lower_percentile

    for batch_index in range(normalized.size(0)):
        for channel_index in range(normalized.size(1)):
            channel_data = normalized[batch_index, channel_index]
            original_shape = channel_data.shape
            channel_flat = channel_data.flatten()
            lower_bound = torch.quantile(channel_flat, lower_percentile / 100)
            upper_bound = torch.quantile(channel_flat, upper_percentile / 100)
            mask = (channel_flat >= lower_bound) & (channel_flat <= upper_bound)
            if mask.sum() <= 0:
                continue

            filtered_data = channel_flat[mask]
            current_mean = filtered_data.mean()
            current_std = filtered_data.std()
            if current_std <= 1e-8:
                normalized[batch_index, channel_index] = channel_data - current_mean + target_mean
                continue

            normalized_flat = ((channel_flat - current_mean) / current_std) * target_std + target_mean
            if clip_outliers:
                normalized_lower = ((lower_bound - current_mean) / current_std) * target_std + target_mean
                normalized_upper = ((upper_bound - current_mean) / current_std) * target_std + target_mean
                normalized_flat = torch.where(channel_flat < lower_bound, normalized_lower, normalized_flat)
                normalized_flat = torch.where(channel_flat > upper_bound, normalized_upper, normalized_flat)

            normalized[batch_index, channel_index] = normalized_flat.reshape(original_shape)

    latents_copy["samples"] = torch.lerp(latents["samples"], normalized, factor)
    return latents_copy


class RayLTXVPerStepAdainPatcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "factors": ("STRING", {"default": "0.9, 0.75, 0.0"}),
                "reference": ("LATENT",),
            },
            "optional": {
                "per_frame": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/LTXV"

    @ray_patch
    def patch(self, model, factors, reference, per_frame=False):
        patched = model.clone()
        factor_list = _parse_float_list(factors)
        step = 0

        def norm_fn(args):
            nonlocal step
            latent = {"samples": args["denoised"]}
            factor = factor_list[min(step, len(factor_list) - 1)]
            step += 1
            return _adain_batch_normalize(latent, reference, factor, per_frame)["samples"]

        patched.set_model_sampler_post_cfg_function(norm_fn)
        return patched


class RayLTXVPerStepStatNormPatcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "factors": ("STRING", {"default": "0.9, 0.75, 0.0"}),
                "target_mean": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01, "round": 0.01}),
                "target_std": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01, "round": 0.01}),
                "percentile": ("FLOAT", {"default": 95.0, "min": 50.0, "max": 100.0, "step": 0.1, "round": 0.1}),
                "clip_outliers": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/LTXV"

    @ray_patch
    def patch(self, model, factors, target_mean, target_std, percentile, clip_outliers):
        patched = model.clone()
        factor_list = _parse_float_list(factors)
        step = 0

        def norm_fn(args):
            nonlocal step
            latent = {"samples": args["denoised"]}
            factor = factor_list[min(step, len(factor_list) - 1)]
            step += 1
            return _statistical_normalize(latent, target_mean, target_std, percentile, factor, clip_outliers)["samples"]

        patched.set_model_sampler_post_cfg_function(norm_fn)
        return patched


class InverseCONST:
    def calculate_input(self, sigma, noise):
        del sigma
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        del sigma, model_input
        return model_output

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        del sigma, noise, max_denoise
        return latent_image

    def inverse_noise_scaling(self, sigma, latent):
        del sigma
        return latent


class ReverseCONST:
    def calculate_input(self, sigma, noise):
        del sigma
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        del sigma, model_input
        return model_output

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        del sigma, noise, max_denoise
        return latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent / (1.0 - sigma)


class RayLTXForwardModelSamplingPredNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ray_actors": ("RAY_ACTORS",)}}

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/LTXV"

    @ray_patch
    def patch(self, model):
        patched = model.clone()

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingFlux, InverseCONST):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=1.15)
        patched.add_object_patch("model_sampling", model_sampling)
        return patched


class RayLTXReverseModelSamplingPredNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ray_actors": ("RAY_ACTORS",)}}

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/LTXV"

    @ray_patch
    def patch(self, model):
        patched = model.clone()

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingFlux, ReverseCONST):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=1.15)
        patched.add_object_patch("model_sampling", model_sampling)
        return patched


class RayLTXVApplySTG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "block_indices": ("STRING", {"default": "14, 19"}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/LTXV"

    @ray_patch
    def patch(self, model, block_indices):
        patched = model.clone()
        skip_block_list = _parse_int_layers(block_indices)
        model_options, transformer_options = _get_transformer_options_copy(patched)
        if "skip_block_list" in transformer_options:
            skip_block_list.extend(transformer_options["skip_block_list"])
        transformer_options["skip_block_list"] = skip_block_list
        patched.model_options = model_options
        return patched


class RayLTXFetaEnhanceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "feta_weight": ("FLOAT", {"default": 4.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "block_indices": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/LTXV"

    @ray_patch
    def patch(self, model, feta_weight, block_indices=""):
        patched = model.clone()
        model_options, transformer_options = _get_transformer_options_copy(patched)
        layers = _parse_int_layers(block_indices, default_layers=range(100))
        transformer_options["feta_weight"] = feta_weight
        transformer_options["feta_layers"] = {"layers": layers}
        patched.model_options = model_options
        return patched


class RayDynamicConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "power": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 2.0, "step": 0.01}),
                "only_first_frame": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/LTXV"

    @ray_patch
    def patch(self, model, power, only_first_frame):
        patched = model.clone()

        def find_step(sigma, step_sigmas):
            for index, step_sigma in enumerate(step_sigmas):
                if step_sigma <= sigma:
                    return index
            return len(step_sigmas) - 1

        def forward(sigma, denoise_mask, extra_options):
            sampler_model = extra_options["model"]
            step_sigmas = extra_options["sigmas"]
            step = find_step(sigma, step_sigmas)
            dynamic_power = power ** step
            updated_mask = denoise_mask.clone()
            if only_first_frame:
                num_channels = sampler_model.model_patcher.model.diffusion_model.in_channels
                updated_mask[:, :num_channels, :1] **= dynamic_power
            else:
                updated_mask **= dynamic_power
            for key in sampler_model.conds:
                if "positive" not in key and "negative" not in key:
                    continue
                for cond in sampler_model.conds[key]:
                    if "model_conds" in cond and "denoise_mask" in cond["model_conds"]:
                        cond["model_conds"]["denoise_mask"].cond = updated_mask
            return updated_mask

        patched.set_model_denoise_mask_function(forward)
        return patched


def gaussian_blur_2d(img, kernel_size, sigma):
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5
    axis = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (axis / sigma).pow(2))
    x_kernel = (pdf / pdf.sum()).to(device=img.device, dtype=img.dtype)
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])
    padded = F.pad(img, [kernel_size // 2] * 4, mode="reflect")
    return F.conv2d(padded, kernel2d, groups=img.shape[-3])


class RayLTXPerturbedAttentionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "rescale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "mode": (["PAG", "SEG"], {"default": "PAG"}),
            },
            "optional": {
                "block_indices": ("STRING", {"default": "14"}),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/LTXV"

    @ray_patch
    def patch(self, model, scale, rescale, cfg, mode, block_indices="14"):
        patched = model.clone()
        layers = _parse_int_layers(block_indices, default_layers=[14])

        def pag_fn(q, k, v, heads, attn_precision=None, transformer_options=None):
            del q, k, heads, attn_precision, transformer_options
            return v

        def seg_fn(q, k, v, heads, attn_precision=None, transformer_options=None):
            del k
            _, _, _ = q.shape
            _, _, frames, height, width = transformer_options["original_shape"]
            q = rearrange(q, "b (f h w) d -> b (f d) w h", h=height, w=width)
            kernel_size = math.ceil(6 * scale) + 1 - math.ceil(6 * scale) % 2
            q = gaussian_blur_2d(q, kernel_size, scale)
            q = rearrange(q, "b (f d) w h -> b (f h w) d", f=frames)
            return optimized_attention(q, k, v, heads, attn_precision=attn_precision)

        def post_cfg_function(args):
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond = args["cond"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]
            uncond = args.get("uncond", None)

            if scale == 0:
                if uncond is None:
                    return cond_pred
                return uncond_pred + (cond_pred - uncond_pred)

            attn_fn = pag_fn if mode == "PAG" else seg_fn
            for block_idx in layers:
                model_options = comfy.model_patcher.set_model_options_patch_replace(
                    model_options,
                    attn_fn,
                    "layer",
                    "self_attn",
                    int(block_idx),
                )

            perturbed = comfy.samplers.calc_cond_batch(args["model"], [cond], x, sigma, model_options)[0]
            output = uncond_pred + cfg * (cond_pred - uncond_pred) + scale * (cond_pred - perturbed)
            if rescale > 0:
                factor = cond_pred.std() / output.std()
                factor = rescale * factor + (1 - rescale)
                output = output * factor
            return output

        patched.set_model_sampler_post_cfg_function(post_cfg_function)
        return patched


class RaySTGGuiderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "stg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("RAY_GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "Raylight/LTXV/guiders"

    def get_guider(self, ray_actors, positive, negative, cfg, stg, rescale):
        return (_make_ray_guider(ray_actors, "ltxv_stg", positive=positive, negative=negative, cfg=cfg, stg_scale=stg, rescale_scale=rescale),)


class RaySTGGuiderAdvancedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "skip_steps_sigma_threshold": ("FLOAT", {"default": 0.998, "min": 0.0, "max": 100.0, "step": 0.001}),
                "cfg_star_rescale": ("BOOLEAN", {"default": True}),
                "sigmas": ("STRING", {"default": "1.0, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180"}),
                "cfg_values": ("STRING", {"default": "8, 6, 6, 4, 3, 1"}),
                "stg_scale_values": ("STRING", {"default": "4, 4, 3, 2, 1, 0"}),
                "stg_rescale_values": ("STRING", {"default": "1, 1, 1, 1, 1, 1"}),
                "stg_layers_indices": ("STRING", {"default": "[29], [29], [29], [29], [29], [29]"}),
            },
            "optional": {
                "apply_apg": ("BOOLEAN", {"default": False}),
                "apg_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "norm_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("RAY_GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "Raylight/LTXV/guiders"

    def get_guider(
        self,
        ray_actors,
        positive,
        negative,
        skip_steps_sigma_threshold,
        cfg_star_rescale,
        sigmas,
        cfg_values,
        stg_scale_values,
        stg_rescale_values,
        stg_layers_indices,
        apply_apg=False,
        apg_cfg_scale=1.0,
        eta=1.0,
        norm_threshold=0.0,
    ):
        return (
            _make_ray_guider(
                ray_actors,
                "ltxv_stg_advanced",
                positive=positive,
                negative=negative,
                skip_steps_sigma_threshold=skip_steps_sigma_threshold,
                cfg_star_rescale=cfg_star_rescale,
                sigma_list=_parse_float_list(sigmas),
                cfg_list=_parse_float_list(cfg_values),
                stg_scale_list=_parse_float_list(stg_scale_values),
                stg_rescale_list=_parse_float_list(stg_rescale_values),
                stg_layers_indices_list=parse_stg_layers_indices(stg_layers_indices),
                apply_apg=apply_apg,
                apg_cfg_scale=apg_cfg_scale,
                eta=eta,
                norm_threshold=norm_threshold,
            ),
        )


class RayGuiderParametersNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "modality": (["VIDEO", "AUDIO"], {"default": "VIDEO"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "stg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "perturb_attn": ("BOOLEAN", {"default": True}),
                "rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 100.0, "step": 0.01}),
                "modality_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "skip_step": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "cross_attn": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "parameters": ("GUIDER_PARAMETERS", {"default": None}),
            },
        }

    RETURN_TYPES = ("GUIDER_PARAMETERS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "get_parameters"
    CATEGORY = "Raylight/LTXV/guiders"

    def get_parameters(self, modality, cfg, stg, perturb_attn, rescale, modality_scale, skip_step, cross_attn, parameters=None):
        parameters = parameters.copy() if parameters is not None else {}
        if modality in parameters:
            raise ValueError(f"Modality {modality} already exists in parameters")
        parameters[modality] = {
            "cfg_scale": cfg,
            "stg_scale": stg,
            "perturb_attn": perturb_attn,
            "rescale_scale": rescale,
            "modality_scale": modality_scale,
            "skip_step": skip_step,
            "cross_attn": cross_attn,
            "cfg_zero_star": False,
            "zero_init_sigma": 1.0,
        }
        return (parameters,)


class RayMultimodalGuiderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "parameters": ("GUIDER_PARAMETERS",),
                "skip_blocks": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("RAY_GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "Raylight/LTXV/guiders"

    def get_guider(self, ray_actors, positive, negative, parameters, skip_blocks):
        return (
            _make_ray_guider(
                ray_actors,
                "ltxv_multimodal",
                positive=positive,
                negative=negative,
                parameters=parameters,
                skip_blocks=parse_skip_blocks(skip_blocks),
            ),
        )


NODE_CLASS_MAPPINGS = {
    "RayLTXVPerStepAdainPatcher": RayLTXVPerStepAdainPatcher,
    "RayLTXVPerStepStatNormPatcher": RayLTXVPerStepStatNormPatcher,
    "RayLTXForwardModelSamplingPred": RayLTXForwardModelSamplingPredNode,
    "RayLTXReverseModelSamplingPred": RayLTXReverseModelSamplingPredNode,
    "RayLTXVApplySTG": RayLTXVApplySTG,
    "RayLTXFetaEnhance": RayLTXFetaEnhanceNode,
    "RayDynamicConditioning": RayDynamicConditioning,
    "RayLTXPerturbedAttention": RayLTXPerturbedAttentionNode,
    "RaySTGGuider": RaySTGGuiderNode,
    "RaySTGGuiderAdvanced": RaySTGGuiderAdvancedNode,
    "RayGuiderParameters": RayGuiderParametersNode,
    "RayMultimodalGuider": RayMultimodalGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayLTXVPerStepAdainPatcher": "LTXV PerStep AdaIN Patcher (Ray)",
    "RayLTXVPerStepStatNormPatcher": "LTXV PerStep StatNorm Patcher (Ray)",
    "RayLTXForwardModelSamplingPred": "LTX Forward Model Sampling Pred (Ray)",
    "RayLTXReverseModelSamplingPred": "LTX Reverse Model Sampling Pred (Ray)",
    "RayLTXVApplySTG": "LTXV Apply STG (Ray)",
    "RayLTXFetaEnhance": "LTX Feta Enhance (Ray)",
    "RayDynamicConditioning": "Dynamic Conditioning (Ray)",
    "RayLTXPerturbedAttention": "LTX Perturbed Attention (Ray)",
    "RaySTGGuider": "STG Guider (Ray)",
    "RaySTGGuiderAdvanced": "STG Guider Advanced (Ray)",
    "RayGuiderParameters": "Guider Parameters (Ray)",
    "RayMultimodalGuider": "Multimodal Guider (Ray)",
}
