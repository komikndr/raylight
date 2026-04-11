import os
import sys
import gc
import functools
from datetime import timedelta

import torch
import torch.distributed as dist
import ray

import comfy.patcher_extension as pe

from raylight.distributed_modules.pipefusion import (
    PipeFusionInjectRegistry,
    pipefusion_diffusion_model_wrapper,
    pipefusion_outer_sample_wrapper,
    pipefusion_predict_noise_wrapper,
)
from raylight.distributed_modules.usp import USPInjectRegistry
from raylight.distributed_modules.cfg import CFGParallelInjectRegistry
from raylight.distributed_worker.pipefusion_schema import (
    PipeFusionConfig,
    build_stage_plan,
)
from raylight.distributed_worker.pipefusion_state import (
    PIPEFUSION_RUNTIME_ATTACHMENT,
    PIPEFUSION_WRAPPER_KEY,
    PipeFusionRuntime,
)
from raylight.distributed_worker.parallel_group_manager import (
    initialize_xfuser_parallel,
    requires_xfuser_parallel,
)
from raylight.distributed_worker.utils import Noise_EmptyNoise, Noise_RandomNoise, patch_ray_tqdm
from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops
from ray.exceptions import RayActorError


# Developer reminder, Checking model parameter outside ray actor is very expensive (e.g Comfy main thread)
# the model need to be serialized, send to object store and can cause OOM !, so setter and getter is the pattern !


# If ray actor function being called from outside, ray.get([task in actor task]) will become sync between rank
# If called from ray actor within. dist.barrier() become the sync.


# Comfy cli args, does not get pass through into ray actor
def patch_enable_comfy_kitchen_fsdp(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        from raylight.comfy_dist.kitchen_distributed import patch_enable_comfy_kitchen_fsdp as patcher

        return patcher(fn)(self, *args, **kwargs)

    return wrapper


class RayWorker:
    def __init__(self, local_rank, device_id, parallel_dict):
        self.model = None
        self.vae_model = None
        self.model_type = None
        self.state_dict = None
        self.lora_list = None
        self.parallel_dict = parallel_dict
        self.overwrite_cast_dtype = None
        self.cached_base_model = None
        self.cached_base_key = None
        self.active_request_key = None

        self.local_rank = local_rank
        self.global_world_size = self.parallel_dict["global_world_size"]

        self.device_id = device_id
        self.parallel_dict = parallel_dict
        self.device = torch.device(f"cuda:{self.device_id}")
        self.device_mesh = None
        self.compute_capability = int("{}{}".format(*torch.cuda.get_device_capability()))
        self.pipefusion_config = PipeFusionConfig.from_parallel_dict(self.parallel_dict)
        self.pipefusion_stage = None
        self.xfuser_parallel = None

        self.is_model_loaded = False
        self.is_cpu_offload = self.parallel_dict.get("fsdp_cpu_offload", False)

        os.environ["XDIT_LOGGING_LEVEL"] = "WARN"
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)

        dist.init_process_group(
            "nccl",
            rank=local_rank,
            world_size=self.global_world_size,
            timeout=timedelta(minutes=1),
            # device_id=self.device
        )

        # (TODO-Komikndr) Should be modified so it can do support DP on top of FSDP
        if self.parallel_dict["is_xdit"] or self.parallel_dict["is_fsdp"]:
            self.device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(self.global_world_size,))
        elif not self.parallel_dict.get("pipefusion_enabled"):
            print(f"Running Ray in normal seperate sampler with: {self.global_world_size} number of workers")

        if requires_xfuser_parallel(self.parallel_dict):
            self.xfuser_parallel = initialize_xfuser_parallel(
                local_rank=self.local_rank,
                world_size=self.global_world_size,
                parallel_dict=self.parallel_dict,
            )
            if self.parallel_dict["is_xdit"]:
                print("XDiT is enable")
            if self.parallel_dict.get("pipefusion_enabled"):
                print("PipeFusion xFuser topology is enable")
            print(
                "Parallel Degree: "
                f"Ulysses={self.xfuser_parallel.config.ulysses_degree}, "
                f"Ring={self.xfuser_parallel.config.ring_degree}, "
                f"CFG={self.xfuser_parallel.config.cfg_degree}, "
                f"PP={self.xfuser_parallel.config.pp_degree}, "
                f"DP={self.xfuser_parallel.config.data_parallel_degree}"
            )

    def get_meta_model(self):
        first_param_device = next(self.model.model.parameters()).device
        if first_param_device == torch.device("meta"):
            return self.model
        else:
            raise ValueError("Model recieved is not meta, can cause OOM in large model")

    def set_meta_model(self, model):
        first_param_device = next(model.model.parameters()).device
        if first_param_device == torch.device("meta"):
            self.state_dict = None
            self.model = model
            self.model.config_fsdp(self.local_rank, self.device_mesh)
        else:
            raise ValueError("Model being set is not meta, can cause OOM in large model")

    def _normalize_model_options(self, model_options):
        if not model_options:
            return ()

        normalized = []
        for key in sorted(model_options.keys()):
            value = model_options[key]
            if isinstance(value, (list, tuple)):
                value = tuple(str(v) for v in value)
            else:
                value = str(value)
            normalized.append((key, value))
        return tuple(normalized)

    def _lora_signature(self, lora_list=None):
        if lora_list is None:
            lora_list = self.lora_list

        if not lora_list:
            return ()

        return tuple((lora["path"], float(lora["strength_model"])) for lora in lora_list)

    def _base_model_key(self, unet_path, model_options):
        return (unet_path, self._normalize_model_options(model_options))

    def _active_model_key(self, unet_path, model_options, lora_list=None):
        return self._base_model_key(unet_path, model_options) + (self._lora_signature(lora_list),)

    def _reset_active_model(self):
        import comfy.model_management as comfy_model_management

        if self.model is not None:
            try:
                self.model.detach()
            except Exception:
                pass
            try:
                self.model.cleanup()
            except Exception:
                pass

        self.model = None
        self.overwrite_cast_dtype = None
        self.active_request_key = None
        comfy_model_management.soft_empty_cache()
        gc.collect()

    def _invalidate_non_fsdp_cache(self):
        self.cached_base_model = None
        self.cached_base_key = None
        self.active_request_key = None

    def _activate_cached_base_model(self, active_key):
        self._reset_active_model()
        self.model = self.cached_base_model.clone()

        if self.lora_list is not None:
            self.load_lora()

        self.overwrite_cast_dtype = self.model.model.manual_cast_dtype
        self.active_request_key = active_key
        self.is_model_loaded = True

    def set_state_dict(self):
        if self.state_dict is None:
            if self.parallel_dict.get("is_fsdp") is True and self.parallel_dict.get("is_quant") is False:
                self.model.set_fsdp_state_dict({})
                return
            raise ValueError("Worker state_dict is None before set_state_dict")
        self.model.set_fsdp_state_dict(self.state_dict)

    def get_compute_capability(self):
        return self.compute_capability

    def get_parallel_dict(self):
        return self.parallel_dict

    def set_parallel_dict(self, parallel_dict):
        self.parallel_dict = parallel_dict
        self.pipefusion_config = PipeFusionConfig.from_parallel_dict(self.parallel_dict)

    def model_function_runner(self, fn, *args, **kwargs):
        self.model = fn(self.model, *args, **kwargs)

    def model_function_runner_get_values(self, fn, *args, **kwargs):
        return fn(self.model, *args, **kwargs)

    def get_local_rank(self):
        return self.local_rank

    def get_is_model_loaded(self):
        return self.is_model_loaded

    def patch_cfg(self):
        self.model.add_wrapper(pe.WrappersMP.DIFFUSION_MODEL, CFGParallelInjectRegistry.inject(self.model))

    def patch_usp(self):
        self.model.add_callback(
            pe.CallbacksMP.ON_LOAD,
            USPInjectRegistry.inject,
        )

    def patch_pipefusion(self):
        if not self.pipefusion_config.enabled:
            return
        if self.parallel_dict.get("is_fsdp"):
            raise ValueError("PipeFusion v1 cannot be enabled together with FSDP")
        if self.xfuser_parallel is None:
            raise RuntimeError("PipeFusion requires xFuser model parallel state to be initialized")
        if self.xfuser_parallel.config.cfg_degree != 1:
            raise NotImplementedError("PipeFusion currently ignores CFG parallel execution; keep cfg_degree at 1")
        if self.xfuser_parallel.sequence_world_size != 1:
            raise NotImplementedError(
                "PipeFusion topology is now initialized through xFuser, but the Wan execution path does not yet combine PP with USP"
            )

        base_model = self.model.model
        if not hasattr(base_model, "diffusion_model") or not hasattr(base_model.diffusion_model, "blocks"):
            raise ValueError(f"PipeFusion requires a Wan diffusion model with blocks, got {type(base_model).__name__}")

        self.pipefusion_stage = getattr(self.model, "pipefusion_stage", None)
        if self.pipefusion_stage is None:
            self.pipefusion_stage = build_stage_plan(
                total_blocks=getattr(
                    base_model.diffusion_model, "_raylight_pipefusion_total_blocks", len(base_model.diffusion_model.blocks)
                ),
                rank=self.xfuser_parallel.pipeline_rank,
                world_size=self.xfuser_parallel.pipeline_world_size,
                config=self.pipefusion_config,
                group_ranks=tuple(self.xfuser_parallel.pp_group().ranks),
            )

        runtime = PipeFusionRuntime(
            config=self.pipefusion_config,
            stage=self.pipefusion_stage,
            model_name=type(base_model).__name__,
            parallel=self.xfuser_parallel,
        )
        if runtime.debug:
            print(
                "[PipeFusion] "
                f"global_rank={self.xfuser_parallel.global_rank} "
                f"pp_group={self.pipefusion_stage.group_ranks} "
                f"stage={self.pipefusion_stage.stage_start}:{self.pipefusion_stage.stage_end} "
                f"patches={self.pipefusion_stage.num_pipeline_patch}"
            )
        self.model.set_attachments(PIPEFUSION_RUNTIME_ATTACHMENT, runtime)

        self.model.remove_callbacks_with_key(pe.CallbacksMP.ON_LOAD, PIPEFUSION_WRAPPER_KEY)
        self.model.remove_wrappers_with_key(pe.WrappersMP.OUTER_SAMPLE, PIPEFUSION_WRAPPER_KEY)
        self.model.remove_wrappers_with_key(pe.WrappersMP.PREDICT_NOISE, PIPEFUSION_WRAPPER_KEY)
        self.model.remove_wrappers_with_key(pe.WrappersMP.DIFFUSION_MODEL, PIPEFUSION_WRAPPER_KEY)

        self.model.add_callback_with_key(
            pe.CallbacksMP.ON_LOAD,
            PIPEFUSION_WRAPPER_KEY,
            PipeFusionInjectRegistry.inject,
        )
        self.model.add_wrapper_with_key(
            pe.WrappersMP.OUTER_SAMPLE,
            PIPEFUSION_WRAPPER_KEY,
            pipefusion_outer_sample_wrapper,
        )
        self.model.add_wrapper_with_key(
            pe.WrappersMP.PREDICT_NOISE,
            PIPEFUSION_WRAPPER_KEY,
            pipefusion_predict_noise_wrapper,
        )
        self.model.add_wrapper_with_key(
            pe.WrappersMP.DIFFUSION_MODEL,
            PIPEFUSION_WRAPPER_KEY,
            pipefusion_diffusion_model_wrapper,
        )

    def load_unet(self, unet_path, model_options):
        if self.parallel_dict["is_fsdp"] is True:
            import comfy.model_management as comfy_model_management

            if self.cached_base_model is not None or self.active_request_key is not None:
                self._reset_active_model()
            self._invalidate_non_fsdp_cache()
            # Monkey patch
            import comfy.model_patcher as model_patcher
            import comfy.model_management as model_management

            # Monkey patch
            from raylight.comfy_dist.model_management import cleanup_models_gc
            from raylight.comfy_dist.model_patcher import LowVramPatch

            from raylight.comfy_dist.sd import fsdp_load_diffusion_model

            fsdp_model_options = dict(model_options)
            fsdp_model_options["use_mmap"] = self.parallel_dict.get("use_mmap", True)

            # Monkey patch
            model_patcher.LowVramPatch = LowVramPatch
            model_management.cleanup_models_gc = cleanup_models_gc

            del self.model
            del self.state_dict
            self.model = None
            self.state_dict = None
            torch.cuda.synchronize()
            comfy_model_management.soft_empty_cache()
            gc.collect()

            self.model, self.state_dict = fsdp_load_diffusion_model(
                unet_path,
                self.local_rank,
                self.device_mesh,
                self.is_cpu_offload,
                model_options=fsdp_model_options,
            )
            torch.cuda.synchronize()
            comfy_model_management.soft_empty_cache()
            gc.collect()
        else:
            import comfy.sd as comfy_sd

            base_key = self._base_model_key(unet_path, model_options)
            active_key = self._active_model_key(unet_path, model_options)
            use_mmap = (
                self.parallel_dict.get("use_mmap", True)
                and not self.parallel_dict.get("is_quant", False)
                and unet_path.lower().endswith(".safetensors")
            )

            if self.model is not None and self.active_request_key == active_key:
                self.overwrite_cast_dtype = self.model.model.manual_cast_dtype
                self.is_model_loaded = True
                return

            if self.cached_base_model is not None and self.cached_base_key == base_key:
                self._activate_cached_base_model(active_key)
                return

            self._reset_active_model()
            self._invalidate_non_fsdp_cache()
            if self.parallel_dict.get("pipefusion_enabled"):
                from raylight.comfy_dist.sd import pipefusion_load_diffusion_model

                if self.xfuser_parallel is None:
                    raise RuntimeError("PipeFusion model loading requires xFuser parallel context")

                pipefusion_model_options = dict(model_options)
                pipefusion_model_options["use_mmap"] = use_mmap
                loaded_model = pipefusion_load_diffusion_model(
                    unet_path,
                    pipefusion_config=self.pipefusion_config,
                    parallel_context=self.xfuser_parallel,
                    model_options=pipefusion_model_options,
                )
            elif use_mmap:
                from raylight.comfy_dist.sd import lazy_load_diffusion_model

                try:
                    loaded_model = lazy_load_diffusion_model(
                        unet_path,
                        model_options=model_options,
                    )
                except Exception as exc:
                    print(f"[RayWorker {self.local_rank}] Lazy safetensor load failed, falling back to eager load: {exc}")
                    loaded_model = comfy_sd.load_diffusion_model(
                        unet_path,
                        model_options=model_options,
                    )
            else:
                loaded_model = comfy_sd.load_diffusion_model(
                    unet_path,
                    model_options=model_options,
                )
            self.cached_base_model = loaded_model
            self.cached_base_key = base_key
            self._activate_cached_base_model(active_key)
            return

        if self.lora_list is not None:
            self.load_lora()

        self.overwrite_cast_dtype = self.model.model.manual_cast_dtype
        self.is_model_loaded = True

    def load_gguf_unet(self, unet_path, dequant_dtype, patch_dtype, use_mmap=None):
        self._reset_active_model()
        self._invalidate_non_fsdp_cache()
        if use_mmap is None:
            use_mmap = self.parallel_dict.get("use_mmap", True)
        if self.parallel_dict["is_fsdp"] is True:
            # GGUF FSDP stays disabled for now.
            raise RuntimeError("FSDP on GGUF is not supported")
        else:
            from raylight.comfy_dist.sd import gguf_load_diffusion_model

            self.model = gguf_load_diffusion_model(
                unet_path,
                model_options={"use_mmap": use_mmap},
                dequant_dtype=dequant_dtype,
                patch_dtype=patch_dtype,
            )

        if self.lora_list is not None:
            self.load_lora()

        self.is_model_loaded = True

    def set_lora_list(self, lora):
        self.lora_list = lora

    def get_lora_list(
        self,
    ):
        return self.lora_list

    def load_lora(
        self,
    ):
        import comfy.sd as comfy_sd
        import comfy.utils as comfy_utils

        for lora in self.lora_list:
            lora_path = lora["path"]
            strength_model = lora["strength_model"]
            lora_model = comfy_utils.load_torch_file(lora_path, safe_load=True)

            if self.parallel_dict["is_fsdp"] is True:
                from raylight.comfy_dist.sd import (
                    load_lora_for_models as ray_load_lora_for_models,
                    load_lora_for_models_quantized as ray_load_lora_for_models_quantized,
                )

                if self.parallel_dict["is_quant"] is True:
                    self.model = ray_load_lora_for_models_quantized(
                        self.model,
                        lora_model,
                        strength_model,
                    )
                else:
                    self.model = ray_load_lora_for_models(
                        self.model,
                        lora_model,
                        strength_model,
                    )
            else:
                self.model = comfy_sd.load_lora_for_models(self.model, None, lora_model, strength_model, 0)[0]
            del lora_model

    def kill(self):
        self._invalidate_non_fsdp_cache()
        self.model = None
        dist.destroy_process_group()
        ray.actor.exit_actor()

    def ray_vae_loader(self, vae_path):
        import comfy.sd as comfy_sd
        import comfy.utils as comfy_utils

        from ..comfy_dist.sd import decode_tiled_1d, decode_tiled_, decode_tiled_3d

        state_dict = {}
        if "pixel_space" in vae_path:
            state_dict["pixel_space_vae"] = torch.tensor(1.0)
        else:
            state_dict = comfy_utils.load_torch_file(vae_path)

        vae_model = comfy_sd.VAE(sd=state_dict)
        vae_model.throw_exception_if_invalid()

        vae_model.decode_tiled_1d = types.MethodType(decode_tiled_1d, vae_model)
        vae_model.decode_tiled_ = types.MethodType(decode_tiled_, vae_model)
        vae_model.decode_tiled_3d = types.MethodType(decode_tiled_3d, vae_model)

        if self.local_rank == 0:
            print(f"VAE loaded in {self.global_world_size} GPUs")
        self.vae_model = vae_model

    @patch_ray_tqdm
    def ray_vae_decode(self, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        temporal_compression = self.vae_model.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None

        compression = self.vae_model.spacial_compression_decode()

        images = self.vae_model.decode_tiled(
            samples["samples"],
            tile_x=tile_size // compression,
            tile_y=tile_size // compression,
            overlap=overlap // compression,
            tile_t=temporal_size,
            overlap_t=temporal_overlap,
        )
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    @patch_enable_comfy_kitchen_fsdp
    def custom_sampler(
        self,
        add_noise,
        noise_seed,
        cfg,
        positive,
        negative,
        sampler,
        sigmas,
        latent_image,
    ):
        import comfy.model_management as comfy_model_management
        import comfy.sample as comfy_sample
        import comfy.utils as comfy_utils

        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy_sample.fix_empty_latent_channels(self.model, latent_image)
        latent["samples"] = latent_image

        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(latent)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        if self.parallel_dict["is_fsdp"] is True:
            self.model.patch_fsdp()
            del self.state_dict
            self.state_dict = None
            torch.cuda.synchronize()
            comfy_model_management.soft_empty_cache()
            gc.collect()

        disable_pbar = comfy_utils.PROGRESS_BAR_ENABLED
        if self.local_rank == 0:
            disable_pbar = not comfy_utils.PROGRESS_BAR_ENABLED

        with torch.no_grad():
            samples = comfy_sample.sample_custom(
                self.model,
                noise,
                cfg,
                sampler,
                sigmas,
                positive,
                negative,
                latent_image,
                noise_mask=noise_mask,
                disable_pbar=disable_pbar,
                seed=noise_seed,
            )
            out = latent.copy()
            out["samples"] = samples

        if ray.get_runtime_context().get_accelerator_ids()["GPU"][0] and self.parallel_dict["is_fsdp"] == "0":
            self.model.detach()
        else:
            self.model.detach()
        comfy_model_management.soft_empty_cache()
        gc.collect()
        return out

    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    @patch_enable_comfy_kitchen_fsdp
    def common_ksampler(
        self,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
    ):
        import comfy.model_management as comfy_model_management
        import comfy.sample as comfy_sample
        import comfy.utils as comfy_utils

        latent_image = latent["samples"]
        latent_image = comfy_sample.fix_empty_latent_channels(self.model, latent_image)

        if self.parallel_dict["is_fsdp"] is True:
            self.model.patch_fsdp()

        if disable_noise:
            noise = torch.zeros(
                latent_image.size(),
                dtype=latent_image.dtype,
                layout=latent_image.layout,
                device="cpu",
            )
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy_sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        disable_pbar = comfy_utils.PROGRESS_BAR_ENABLED
        if self.local_rank == 0:
            disable_pbar = not comfy_utils.PROGRESS_BAR_ENABLED

        with torch.no_grad():
            samples = comfy_sample.sample(
                self.model,
                noise,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=start_step,
                last_step=last_step,
                force_full_denoise=force_full_denoise,
                noise_mask=noise_mask,
                disable_pbar=disable_pbar,
                seed=seed,
            )
            out = latent.copy()
            out["samples"] = samples

        if ray.get_runtime_context().get_accelerator_ids()["GPU"][0] and self.parallel_dict["is_fsdp"] == "0":
            self.model.detach()

        # I haven't implemented for non FSDP detached, so all rank model will be move into RAM
        else:
            self.model.detach()
        comfy_model_management.soft_empty_cache()
        gc.collect()
        return (out,)


class RayCOMMTester:
    def __init__(self, local_rank, world_size, device_id):
        device = torch.device(f"cuda:{device_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        dist.init_process_group(
            "nccl",
            rank=local_rank,
            world_size=world_size,
            timeout=timedelta(minutes=1),
            # device_id=self.device
        )
        print("Running COMM pre-run")

        # Each rank contributes rank+1
        x = torch.ones(1, device=device) * (local_rank + 1)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        result = x.item()

        # Expected sum = N(N+1)/2
        expected = world_size * (world_size + 1) // 2

        if abs(result - expected) > 1e-3:
            raise RuntimeError(f"[Rank {local_rank}] COMM test failed: got {result}, expected {expected}. world_size may be mismatched!")
        else:
            print(f"[Rank {local_rank}] COMM test passed ✅ (result={result})")

    def kill(self):
        dist.destroy_process_group()
        ray.actor.exit_actor()


def ray_nccl_tester(world_size):
    gpu_actor = ray.remote(RayCOMMTester)
    gpu_actors = []

    for local_rank in range(world_size):
        gpu_actors.append(
            gpu_actor.options(num_gpus=1, name=f"RayTest:{local_rank}").remote(
                local_rank=local_rank,
                world_size=world_size,
                device_id=0,
            )
        )
    for actor in gpu_actors:
        ray.get(actor.__ray_ready__.remote())

    for actor in gpu_actors:
        actor.kill.remote()


def make_ray_actor_fn(world_size, parallel_dict):
    def _init_ray_actor(world_size=world_size, parallel_dict=parallel_dict):
        ray_actors = dict()
        gpu_actor = ray.remote(RayWorker)
        gpu_actors = []

        for local_rank in range(world_size):
            gpu_actors.append(
                gpu_actor.options(num_gpus=1, name=f"RayWorker:{local_rank}").remote(
                    local_rank=local_rank,
                    device_id=0,
                    parallel_dict=parallel_dict,
                )
            )
        ray_actors["workers"] = gpu_actors

        for actor in ray_actors["workers"]:
            ray.get(actor.__ray_ready__.remote())
        return ray_actors

    return _init_ray_actor


# (TODO-Komikndr) Should be removed since FSDP can be unloaded properly
def ensure_fresh_actors(ray_actors_init):
    ray_actors, ray_actor_fn = ray_actors_init
    gpu_actors = ray_actors["workers"]

    needs_restart = False
    try:
        is_loaded = ray.get(gpu_actors[0].get_is_model_loaded.remote())
        if is_loaded:
            needs_restart = True
    except RayActorError:
        # Actor already dead or crashed
        needs_restart = True

    needs_restart = False
    if needs_restart:
        for actor in gpu_actors:
            try:
                ray.get(actor.kill.remote())
            except Exception:
                pass  # ignore already dead
        ray_actors = ray_actor_fn()
        gpu_actors = ray_actors["workers"]

    parallel_dict = ray.get(gpu_actors[0].get_parallel_dict.remote())

    return ray_actors, gpu_actors, parallel_dict
