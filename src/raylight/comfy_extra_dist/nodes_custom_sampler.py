import gc
import ray

import torch

import comfy.samplers
import comfy.sample
from comfy.k_diffusion import sa_solver
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import comfy.utils
from .ray_patch_decorator import ray_patch_with_return

from raylight.distributed_worker.utils import Noise_EmptyNoise, Noise_RandomNoise


def _normalize_grouped_inputs(values: list, expected_length: int, label: str):
    if len(values) == expected_length:
        return values
    if len(values) == 1:
        return values * expected_length
    if len(values) > expected_length:
        return values[:expected_length]
    raise ValueError(f"{label} must provide 1 item or at least {expected_length} items, got {len(values)}")


def _collect_grouped_results(results: list, expected_length: int, label: str):
    grouped_results = {}
    for result in results:
        if result is None:
            continue
        dp_rank = int(result["dp_rank"])
        if dp_rank in grouped_results:
            raise RuntimeError(f"{label} produced multiple outputs for dp_rank {dp_rank}")
        grouped_results[dp_rank] = result["result"]

    missing = [dp_rank for dp_rank in range(expected_length) if dp_rank not in grouped_results]
    if missing:
        raise RuntimeError(f"{label} missing outputs for dp_rank {missing}")

    return [grouped_results[dp_rank] for dp_rank in range(expected_length)]


def _normalized_degree(value):
    if value is None:
        return 1
    value = int(value)
    return 1 if value <= 0 else value


def _validate_unified_parallel_setup(parallel_dict: dict, group_infos: list[dict], label: str):
    ulysses_degree = _normalized_degree(parallel_dict.get("ulysses_degree"))
    ring_degree = _normalized_degree(parallel_dict.get("ring_degree"))
    cfg_degree = _normalized_degree(parallel_dict.get("cfg_degree"))
    pp_degree = _normalized_degree(parallel_dict.get("pp_degree"))
    dp_degree = max(int(info.get("dp_degree", 1)) for info in group_infos) if group_infos else 1
    effective_parallel_size = ulysses_degree * ring_degree * cfg_degree * pp_degree * dp_degree

    if effective_parallel_size <= 1:
        raise ValueError(
            f"{label} requires an active unified parallel topology with effective degree > 1. "
            f"Got ulysses={ulysses_degree}, ring={ring_degree}, cfg={cfg_degree}, pp={pp_degree}, dp={dp_degree}. "
            "Use DPSamplerCustom for pure DP or single-worker execution."
        )

    if not parallel_dict.get("is_xdit") and not parallel_dict.get("pipefusion_enabled"):
        raise ValueError(
            f"{label} requires xFuser unified topology to be active. "
            f"Got ulysses={ulysses_degree}, ring={ring_degree}, cfg={cfg_degree}, pp={pp_degree}, dp={dp_degree}, "
            f"is_xdit={parallel_dict.get('is_xdit')}, pipefusion_enabled={parallel_dict.get('pipefusion_enabled')}. "
            "DP-only execution should use DPSamplerCustom."
        )


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

        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
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
        sigmas = comfy.samplers.beta_scheduler(model.get_model_object("model_sampling"), steps, alpha=alpha, beta=beta)
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

    RETURN_TYPES = ("LATENT", "RAY_ACTORS")
    RETURN_NAMES = ("latent", "ray_actors")

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
        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        gpu_actors = ray_actors["workers"]
        futures = [
            actor.custom_sampler.remote(
                add_noise,
                noise_seed,
                cfg,
                positive,
                negative,
                sampler,
                sigmas,
                latent_image,
            )
            for actor in gpu_actors
        ]
        results = ray.get(futures)
        out = results[0]
        return (out, ray_actors)


class UnifiedParallelSamplerCustom:
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

    RETURN_TYPES = ("LATENT", "RAY_ACTORS")
    RETURN_NAMES = ("latent", "ray_actors")
    OUTPUT_IS_LIST = (True, False)
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
        sampler = sampler[0]
        sigmas = sigmas[0]

        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        gpu_actors = ray_actors["workers"]
        parallel_dict = ray.get(gpu_actors[0].get_parallel_dict.remote())
        group_infos = ray.get([actor.get_exec_group_info.remote() for actor in gpu_actors])
        _validate_unified_parallel_setup(parallel_dict, group_infos, "Unified Parallel SamplerCustom")
        dp_degree = int(group_infos[0]["dp_degree"])
        noise_list = _normalize_grouped_inputs(noise_list, dp_degree, "noise_list")
        positive = _normalize_grouped_inputs(positive, dp_degree, "positive")
        negative = _normalize_grouped_inputs(negative, dp_degree, "negative")
        latent_image = _normalize_grouped_inputs(latent_image, dp_degree, "latent_image")

        futures = [
            actor.custom_sampler.remote(
                add_noise,
                noise_list[group_info["dp_rank"]],
                cfg,
                positive[group_info["dp_rank"]],
                negative[group_info["dp_rank"]],
                sampler,
                sigmas,
                latent_image[group_info["dp_rank"]],
                grouped_output=True,
            )
            for actor, group_info in zip(gpu_actors, group_infos)
        ]
        results = ray.get(futures)
        out = _collect_grouped_results(results, dp_degree, "Unified Parallel SamplerCustom")
        return (out, ray_actors)


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

    RETURN_TYPES = ("LATENT", "RAY_ACTORS")
    RETURN_NAMES = ("latent", "ray_actors")
    OUTPUT_IS_LIST = (True, False)
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
        sampler = sampler[0]
        sigmas = sigmas[0]

        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        gpu_actors = ray_actors["workers"]
        parallel_dict = ray.get(gpu_actors[0].get_parallel_dict.remote())
        num_gpus = len(gpu_actors)

        # Replicate last item to fill remaining slots, or truncate if too many
        if len(latent_image) < num_gpus:
            latent_image = latent_image + [latent_image[-1]] * (num_gpus - len(latent_image))
        elif len(latent_image) > num_gpus:
            latent_image = latent_image[:num_gpus]
        if len(positive) == 1:
            positive = positive * num_gpus
        if len(negative) == 1:
            negative = negative * num_gpus
        if len(noise_list) < num_gpus:
            noise_list = noise_list + [noise_list[-1]] * (num_gpus - len(noise_list))
        elif len(noise_list) > num_gpus:
            noise_list = noise_list[:num_gpus]

        # Each GPU gets its own noise/conditioning/latent — decoupled from FSDP sharding
        futures = [
            actor.custom_sampler.remote(
                add_noise,
                noise_list[i],
                cfg,
                positive[i],
                negative[i],
                sampler,
                sigmas,
                latent_image[i],
            )
            for i, actor in enumerate(gpu_actors)
        ]
        out = ray.get(futures)
        return (out, ray_actors)


class RayAddNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "force_empty_noise": ("BOOLEAN",),
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
    "UnifiedParallelSamplerCustom": UnifiedParallelSamplerCustom,
    "DPSamplerCustom": DPSamplerCustom,
    "RayBasicScheduler": RayBasicScheduler,
    "RayBetaSamplingScheduler": RayBetaSamplingScheduler,
    "RaySamplerSASolver": RaySamplerSASolver,
    "RaySamplingPercentToSigma": RaySamplingPercentToSigma,
    "RayAddNoise": RayAddNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XFuserSamplerCustom": "XFuser SamplerCustom",
    "UnifiedParallelSamplerCustom": "Unified Parallel SamplerCustom (Advance)",
    "DPSamplerCustom": "Data Parallel SamplerCustom",
}
