import gc
import ray

import torch

import comfy.samplers
import comfy.sample
from comfy.k_diffusion import sa_solver
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import latent_preview
import comfy.utils
from .ray_patch_decorator import ray_patch_with_return


class RayBasicScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "Raylight/extra/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    @ray_patch_with_return
    def get_sigmas(self, model, scheduler, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = int(steps / denoise)

        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps
        ).cpu()
        sigmas = sigmas[-(steps + 1) :]
        return (sigmas,)


class RayBetaSamplingScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "beta": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "Raylight/extra/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    @ray_patch_with_return
    def get_sigmas(self, model, steps, alpha, beta):
        sigmas = comfy.samplers.beta_scheduler(
            model.get_model_object("model_sampling"), steps, alpha=alpha, beta=beta
        )
        return (sigmas,)


class RaySamplingPercentToSigma:
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "sampling_percent": (
                    IO.FLOAT,
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001},
                ),
                "return_actual_sigma": (
                    IO.BOOLEAN,
                    {
                        "default": False,
                        "tooltip": "Return the actual sigma value instead of the value used for interval checks.\nThis only affects results at 0.0 and 1.0.",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.FLOAT,)
    RETURN_NAMES = ("sigma_value",)
    CATEGORY = "Raylight/extra/custom_sampling/schedulers"

    FUNCTION = "get_sigma"

    @ray_patch_with_return
    def get_sigma(self, model, sampling_percent, return_actual_sigma):
        model_sampling = model.get_model_object("model_sampling")
        sigma_val = model_sampling.percent_to_sigma(sampling_percent)
        if return_actual_sigma:
            if sampling_percent == 0.0:
                sigma_val = model_sampling.sigma_max.item()
            elif sampling_percent == 1.0:
                sigma_val = model_sampling.sigma_min.item()
        return (sigma_val,)


class RaySamplerSASolver(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "eta": (
                    IO.FLOAT,
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "sde_start_percent": (
                    IO.FLOAT,
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "sde_end_percent": (
                    IO.FLOAT,
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "s_noise": (
                    IO.FLOAT,
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "predictor_order": (IO.INT, {"default": 3, "min": 1, "max": 6}),
                "corrector_order": (IO.INT, {"default": 4, "min": 0, "max": 6}),
                "use_pece": (IO.BOOLEAN, {}),
                "simple_order_2": (IO.BOOLEAN, {}),
            }
        }

    RETURN_TYPES = (IO.SAMPLER,)
    CATEGORY = "Raylight/extra/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    @ray_patch_with_return
    def get_sampler(
        self,
        model,
        eta,
        sde_start_percent,
        sde_end_percent,
        s_noise,
        predictor_order,
        corrector_order,
        use_pece,
        simple_order_2,
    ):
        model_sampling = model.get_model_object("model_sampling")
        start_sigma = model_sampling.percent_to_sigma(sde_start_percent)
        end_sigma = model_sampling.percent_to_sigma(sde_end_percent)
        tau_func = sa_solver.get_tau_interval_func(start_sigma, end_sigma, eta=eta)

        sampler_name = "sa_solver"
        sampler = comfy.samplers.ksampler(
            sampler_name,
            {
                "tau_func": tau_func,
                "s_noise": s_noise,
                "predictor_order": predictor_order,
                "corrector_order": corrector_order,
                "use_pece": use_pece,
                "simple_order_2": simple_order_2,
            },
        )
        return (sampler,)


class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(
            latent_image.shape,
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


class XFuserSamplerCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("output",)

    FUNCTION = "ray_sample"

    CATEGORY = "Raylight/extra/custom_sampling/samplers"

    def ray_sample(
        self,
        ray_actors,
        add_noise,
        noise_seed,
        cfg,
        positive,
        negative,
        sampler,
        sigmas,
        latent_image,
    ):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(self.model, latent_image)
        latent["samples"] = latent_image

        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(latent)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        gpu_actors = ray_actors["workers"]

        futures = [
            actor.custom_sampler.remote(
                noise,
                noise_seed,
                noise_mask,
                cfg,
                positive,
                negative,
                sampler,
                sigmas,
                latent_image,

            )
            for actor in gpu_actors
        ]
        samples_list = ray.get(futures)
        samples = samples_list[0]
        out = latent.copy()
        out["samples"] = samples

        return (out,)


class DPSamplerCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_list": ("NOISE",),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("output", "denoised_output")
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "ray_sample"
    CATEGORY = "Raylight/extra/custom_sampling/samplers"

    def ray_sample(
        self,
        ray_actors,
        add_noise,
        noise_list,
        cfg,
        positive,
        negative,
        sampler,
        sigmas,
        latent_image,
    ):
        ray_actors = ray_actors[0]
        add_noise = add_noise[0]
        cfg = cfg[0]
        positive = positive[0]
        negative = negative[0]
        sampler = sampler[0]
        sigmas = sigmas[0]
        latent_image = latent_image[0]
        noise_seed_list = noise_list

        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(self.model, latent_image)
        latent["samples"] = latent_image

        if not add_noise:
            noise_list = [Noise_EmptyNoise().generate_noise(latent) for noise_seed in noise_seed_list]
        else:
            noise_list = [Noise_RandomNoise(noise_seed).generate_noise(latent) for noise_seed in noise_seed_list]

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        gpu_actors = ray_actors["workers"]

        futures = [
            actor.custom_sampler.remote(
                noise_list[i],
                noise_seed_list[i],
                noise_mask,
                cfg,
                positive,
                negative,
                sampler,
                sigmas,
                latent_image,

            )
            for i, actor in enumerate(gpu_actors)
        ]
        samples_list = ray.get(futures)
        samples = samples_list[0]
        out = latent.copy()
        out["samples"] = samples

        return (out,)


class RayAddNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "force_empty_noise": ("BOOLEAN", ),
                "ray_actors": ("RAY_ACTORS",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "add_noise"

    CATEGORY = "Raylight/extra/custom_sampling/samplers"

    @ray_patch_with_return
    def add_noise(self, model, noise_seed, force_empty_noise, sigmas, latent_image):
        if len(sigmas) == 0:
            return latent_image

        latent = latent_image
        latent_image = latent["samples"]

        if force_empty_noise is False:
            noise = Noise_RandomNoise(noise_seed)
        else:
            noise = Noise_EmptyNoise()

        noisy = noise.generate_noise(latent)
        model_sampling = model.get_model_object("model_sampling")
        process_latent_out = model.get_model_object("process_latent_out")
        process_latent_in = model.get_model_object("process_latent_in")

        if len(sigmas) > 1:
            scale = torch.abs(sigmas[0] - sigmas[-1])
        else:
            scale = sigmas[0]

        if torch.count_nonzero(latent_image) > 0:  # Don't shift the empty latent image.
            latent_image = process_latent_in(latent_image)
        noisy = model_sampling.noise_scaling(scale, noisy, latent_image)
        noisy = process_latent_out(noisy)
        noisy = torch.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)

        out = latent.copy()
        out["samples"] = noisy
        return (out,)


NODE_CLASS_MAPPINGS = {
    "XFuserSamplerCustom": XFuserSamplerCustom,
    "DPSamplerCustom": DPSamplerCustom,
    "RayBasicScheduler": RayBasicScheduler,
    "RayBetaSamplingScheduler": RayBetaSamplingScheduler,
    "RaySamplerSASolver": RaySamplerSASolver,
    "RaySamplingPercentToSigma": RaySamplingPercentToSigma,
    "RayAddNoise": RayAddNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XFuserSamplerCustom": "XFuser SamplerCustom",
    "DPSamplerCustom": "Data Parallel SamplerCustom",
}
