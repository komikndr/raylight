import math

import comfy.model_sampling
import comfy.samplers
import node_helpers
import ray
import torchaudio

from xfuser.core.distributed import get_classifier_free_guidance_world_size

from .ray_patch_decorator import ray_patch


def _get_cfg_world_size():
    try:
        world_size = int(get_classifier_free_guidance_world_size())
    except Exception:
        return 1
    return 1 if world_size <= 0 else world_size


def _patch_ltx_reference_audio_guidance(model, identity_guidance_scale, start_percent, end_percent):
    m = model.clone()
    scale = identity_guidance_scale
    model_sampling = m.get_model_object("model_sampling")
    sigma_start = model_sampling.percent_to_sigma(start_percent)
    sigma_end = model_sampling.percent_to_sigma(end_percent)

    def post_cfg_function(args):
        if scale == 0:
            return args["denoised"]

        sigma = args["sigma"]
        sigma_ = sigma[0].item()
        if sigma_ > sigma_start or sigma_ < sigma_end:
            return args["denoised"]

        cond_pred = args["cond_denoised"]
        cond = args["cond"]
        cfg_result = args["denoised"]
        model_options = args["model_options"].copy()
        x = args["input"]

        # Strip ref_audio from conditioning for the no-reference pass.
        noref_cond = []
        for entry in cond:
            new_entry = entry.copy()
            mc = new_entry.get("model_conds", {}).copy()
            mc.pop("ref_audio", None)
            new_entry["model_conds"] = mc
            noref_cond.append(new_entry)

        cfg_world_size = _get_cfg_world_size()
        noref_conds = [noref_cond]
        if cfg_world_size > 1 and x.shape[0] != cfg_world_size:
            noref_conds = [list(noref_cond) for _ in range(cfg_world_size)]

        pred_noref = comfy.samplers.calc_cond_batch(
            args["model"],
            noref_conds,
            x,
            sigma,
            model_options,
        )[0]

        return cfg_result + (cond_pred - pred_noref) * scale

    m.set_model_sampler_post_cfg_function(post_cfg_function)
    return m


class RayModelSamplingLTXV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    @ray_patch
    def patch(self, model, max_shift, base_shift, latent=None):
        m = model.clone()

        if latent is None:
            tokens = 4096
        else:
            tokens = math.prod(latent["samples"].shape[2:])

        x1 = 1024
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = tokens * mm + b

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        return m


class RayLTXVReferenceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "reference_audio": ("AUDIO",),
                "audio_vae": ("VAE",),
                "identity_guidance_scale": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01},
                ),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("ray_actors", "positive", "negative")
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    def patch(
        self,
        ray_actors,
        positive,
        negative,
        reference_audio,
        audio_vae,
        identity_guidance_scale,
        start_percent,
        end_percent,
    ):
        sample_rate = reference_audio["sample_rate"]
        vae_sample_rate = getattr(audio_vae, "audio_sample_rate", 44100)
        if vae_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(reference_audio["waveform"], sample_rate, vae_sample_rate)
        else:
            waveform = reference_audio["waveform"]

        audio_latents = audio_vae.encode(waveform.movedim(1, -1))
        b, c, t, f = audio_latents.shape
        ref_tokens = audio_latents.permute(0, 2, 1, 3).reshape(b, t, c * f)
        ref_audio = {"tokens": ref_tokens}

        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref_audio})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref_audio})

        gpu_workers = ray_actors["workers"]
        futures = [
            actor.model_function_runner.remote(
                _patch_ltx_reference_audio_guidance,
                identity_guidance_scale,
                start_percent,
                end_percent,
            )
            for actor in gpu_workers
        ]
        ray.get(futures)

        return (ray_actors, positive, negative)


NODE_CLASS_MAPPINGS = {
    "RayModelSamplingLTXV": RayModelSamplingLTXV,
    "RayLTXVReferenceAudio": RayLTXVReferenceAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayModelSamplingLTXV": "Model Sampling LTXV (Ray)",
    "RayLTXVReferenceAudio": "LTXV Reference Audio (Ray)",
}
