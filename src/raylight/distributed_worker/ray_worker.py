import os
import sys
import gc
import types
from datetime import timedelta

import torch
import torch.distributed as dist
import ray

import comfy
from comfy import (
    sd,
    sample,
    utils,
)  # Must manually insert comfy package or ray cannot import raylight to cluster
import comfy.patcher_extension as pe

import raylight.distributed_modules.attention as xfuser_attn

from raylight.distributed_modules.usp import USPInjectRegistry
from raylight.distributed_modules.cfg import CFGParallelInjectRegistry

from raylight.comfy_dist.sd import load_lora_for_models as ray_load_lora_for_models
from raylight.distributed_worker.utils import Noise_EmptyNoise, Noise_RandomNoise, patch_ray_tqdm
from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops
from ray.exceptions import RayActorError


# Developer reminder, Checking model parameter outside ray actor is very expensive (e.g Comfy main thread)
# the model need to be serialized, send to object store and can cause OOM !, so setter and getter is the pattern !


# If ray actor function being called from outside, ray.get([task in actor task]) will become sync between rank
# If called from ray actor within. dist.barrier() become the sync.

# Comfy cli args, does not get pass through into ray actor
class RayWorker:
    def __init__(self, local_rank, device_id, parallel_dict):
        self.model = None
        self.vae_model = None
        self.model_type = None
        self.state_dict = None
        self.parallel_dict = parallel_dict
        self.overwrite_cast_dtype = None

        self.local_rank = local_rank
        self.global_world_size = self.parallel_dict["global_world_size"]

        self.device_id = device_id
        self.parallel_dict = parallel_dict
        self.device = torch.device(f"cuda:{self.device_id}")
        self.device_mesh = None
        self.compute_capability = int("{}{}".format(*torch.cuda.get_device_capability()))

        self.is_model_loaded = False
        self.is_cpu_offload = self.parallel_dict.get("fsdp_cpu_offload", False)

        os.environ["XDIT_LOGGING_LEVEL"] = "WARN"
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)

        if sys.platform.startswith("linux"):
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=self.global_world_size,
                timeout=timedelta(minutes=1),
                # device_id=self.device
            )
        elif sys.platform.startswith("win"):
            os.environ["USE_LIBUV"] = "0"
            dist.init_process_group(
                "gloo",
                rank=local_rank,
                world_size=self.global_world_size,
                timeout=timedelta(minutes=1),
                # device_id=self.device
            )

        # (TODO-Komikndr) Should be modified so it can do support DP on top of FSDP
        if self.parallel_dict["is_xdit"] or self.parallel_dict["is_fsdp"]:
            self.device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(self.global_world_size,))
        else:
            print(f"Running Ray in normal seperate sampler with: {self.global_world_size} number of workers")

        # From mochi-xdit, xdit, pipelines.py
        if self.parallel_dict["is_xdit"]:
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            xfuser_attn.set_attn_type(self.parallel_dict["attention"])
            xfuser_attn.set_sync_ulysses(self.parallel_dict["sync_ulysses"])

            self.cp_degree = self.parallel_dict["ulysses_degree"] * parallel_dict["ring_degree"]
            self.cfg_degree = self.parallel_dict["cfg_degree"]
            self.ulysses_degree = self.parallel_dict["ulysses_degree"]
            self.ring_degree = self.parallel_dict["ring_degree"]
            self.cfg_degree = self.parallel_dict["cfg_degree"]

            init_distributed_environment(rank=self.local_rank, world_size=self.global_world_size)
            print("XDiT is enable")

            initialize_model_parallel(
                sequence_parallel_degree=self.cp_degree,
                classifier_free_guidance_degree=self.cfg_degree,
                ring_degree=self.ring_degree,
                ulysses_degree=self.ulysses_degree
            )
            print(
                f"Parallel Degree: Ulysses={self.ulysses_degree}, Ring={self.ring_degree}, CFG={self.cfg_degree}"
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
            self.model = model
            self.model.config_fsdp(self.local_rank, self.device_mesh)
        else:
            raise ValueError("Model being set is not meta, can cause OOM in large model")

    def set_state_dict(self):
        self.model.set_fsdp_state_dict(self.state_dict)

    def get_compute_capability(self):
        return self.compute_capability

    def get_parallel_dict(self):
        return self.parallel_dict

    def set_parallel_dict(self, parallel_dict):
        self.parallel_dict = parallel_dict

    def model_function_runner(self, fn, *args, **kwargs):
        self.model = fn(self.model, *args, **kwargs)

    def model_function_runner_get_values(self, fn, *args, **kwargs):
        return fn(self.model, *args, **kwargs)

    def get_local_rank(self):
        return self.local_rank

    def get_is_model_loaded(self):
        return self.is_model_loaded

    def patch_cfg(self):
        self.model.add_wrapper(
            pe.WrappersMP.DIFFUSION_MODEL,
            CFGParallelInjectRegistry.inject(self.model)
        )

    def patch_usp(self):
        self.model.add_callback(
            pe.CallbacksMP.ON_LOAD,
            USPInjectRegistry.inject,
        )

    def load_unet(self, unet_path, model_options):
        if self.parallel_dict["is_fsdp"] is True:
            # Monkey patch
            import comfy.model_patcher as model_patcher
            import comfy.model_management as model_management

            # Monkey patch
            from raylight.comfy_dist.model_management import cleanup_models_gc
            from raylight.comfy_dist.model_patcher import LowVramPatch

            from raylight.comfy_dist.sd import fsdp_load_diffusion_model

            # Monkey patch
            model_patcher.LowVramPatch = LowVramPatch
            model_management.cleanup_models_gc = cleanup_models_gc

            del self.model
            del self.state_dict
            self.model = None
            self.state_dict = None
            torch.cuda.synchronize()
            comfy.model_management.soft_empty_cache()
            gc.collect()

            self.model, self.state_dict = fsdp_load_diffusion_model(
                unet_path,
                self.local_rank,
                self.device_mesh,
                self.is_cpu_offload,
                model_options=model_options,
            )
            torch.cuda.synchronize()
            comfy.model_management.soft_empty_cache()
            gc.collect()
        else:
            self.model = comfy.sd.load_diffusion_model(
                unet_path, model_options={},
            )

        if self.lora_list is not None:
            self.load_lora()

        self.overwrite_cast_dtype = self.model.model.manual_cast_dtype
        self.is_model_loaded = True

    def apply_model_sampling(self, model_sampling_patch):
         self.model.add_object_patch("model_sampling", model_sampling_patch)


    def load_gguf_unet(self, unet_path, dequant_dtype, patch_dtype):
        if self.parallel_dict["is_fsdp"] is True:
            raise ValueError("FSDP Sharding for GGUF is not supported")
        else:
            from raylight.comfy_dist.sd import gguf_load_diffusion_model
            self.model = gguf_load_diffusion_model(
                unet_path,
                dequant_dtype=dequant_dtype,
                patch_dtype=patch_dtype
            )

        if self.lora_list is not None:
            self.load_lora()

        self.is_model_loaded = True

    def load_bnb_unet(self, unet_path):
        if self.parallel_dict["is_fsdp"] is True:
            import comfy.model_patcher as model_patcher
            import comfy.model_management as model_management

            from raylight.comfy_dist.model_management import cleanup_models_gc
            from raylight.comfy_dist.model_patcher import LowVramPatch

            from raylight.comfy_dist.sd import fsdp_bnb_load_diffusion_model
            from torch.distributed.fsdp import FSDPModule
            model_patcher.LowVramPatch = LowVramPatch
            model_management.cleanup_models_gc = cleanup_models_gc

            m = getattr(self.model, "model", None)
            if m is not None and isinstance(getattr(m, "diffusion_model", None), FSDPModule):
                del self.model
                self.model = None
            self.model, self.state_dict = fsdp_bnb_load_diffusion_model(
                unet_path,
                self.local_rank,
                self.device_mesh,
                self.is_cpu_offload,
            )
        else:
            from raylight.comfy_dist.sd import bnb_load_diffusion_model
            self.model = bnb_load_diffusion_model(
                unet_path,
            )

        if self.lora_list is not None:
            self.load_lora()

        self.is_model_loaded = True

    def set_lora_list(self, lora):
        self.lora_list = lora

    def get_lora_list(self,):
        return self.lora_list

    def load_lora(self,):
        for lora in self.lora_list:
            lora_path = lora["path"]
            strength_model = lora["strength_model"]
            lora_model = comfy.utils.load_torch_file(lora_path, safe_load=True)

            if self.parallel_dict["is_fsdp"] is True:
                self.model = ray_load_lora_for_models(
                    self.model, lora_model, strength_model
                )
            else:
                self.model = comfy.sd.load_lora_for_models(
                    self.model, None, lora_model, strength_model, 0
                )[0]
            del lora_model

    def kill(self):
        self.model = None
        dist.destroy_process_group()
        ray.actor.exit_actor()

    def ray_vae_loader(self, vae_path):
        from ..comfy_dist.sd import decode_tiled_1d, decode_tiled_, decode_tiled_3d
        state_dict = {}
        if "pixel_space" in vae_path:
            state_dict["pixel_space_vae"] = torch.tensor(1.0)
        else:
            state_dict = comfy.utils.load_torch_file(vae_path)

        vae_model = comfy.sd.VAE(sd=state_dict)
        vae_model.throw_exception_if_invalid()

        vae_model.decode_tiled_1d = types.MethodType(decode_tiled_1d, vae_model)
        vae_model.decode_tiled_ = types.MethodType(decode_tiled_, vae_model)
        vae_model.decode_tiled_3d = types.MethodType(decode_tiled_3d, vae_model)

        if self.local_rank == 0:
            print(f"VAE loaded in {self.global_world_size} GPUs")
        self.vae_model = vae_model

    @patch_ray_tqdm
    def ray_vae_decode(
        self,
        samples,
        tile_size,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8
    ):
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        temporal_compression = self.vae_model.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(
                1, min(temporal_size // 2, temporal_overlap // temporal_compression)
            )
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
            images = images.reshape(
                -1, images.shape[-3], images.shape[-2], images.shape[-1]
            )
        return images

    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
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

        if self.parallel_dict["is_fsdp"] is True:
            self.model.patch_fsdp()
            del self.state_dict
            self.state_dict = None
            torch.cuda.synchronize()
            comfy.model_management.soft_empty_cache()
            gc.collect()

        disable_pbar = comfy.utils.PROGRESS_BAR_ENABLED
        if self.local_rank == 0:
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        with torch.no_grad():
            samples = comfy.sample.sample_custom(
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
        comfy.model_management.soft_empty_cache()
        gc.collect()
        return out

    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
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
        sigmas=None,
    ):
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(self.model, latent_image)

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
            noise = comfy.sample.prepare_noise(
                latent_image, seed, batch_inds
            )

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        disable_pbar = comfy.utils.PROGRESS_BAR_ENABLED
        if self.local_rank == 0:
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        # Sampler resolution logic for custom sigmas
        sampler_obj = sampler_name
        if sigmas is not None:
             if isinstance(sampler_name, str):
                 sampler_obj = comfy.samplers.ksampler(sampler_name)

        with torch.no_grad():
            if sigmas is None:
                samples = comfy.sample.sample(
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
            else:
                 samples = comfy.sample.sample_custom(
                    self.model,
                    noise,
                    cfg,
                    sampler_obj,
                    sigmas,
                    positive,
                    negative,
                    latent_image,
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
        comfy.model_management.soft_empty_cache()
        gc.collect()
        return (out,)


class RayCOMMTester:
    def __init__(self, local_rank, world_size, device_id):
        device = torch.device(f"cuda:{device_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        if sys.platform.startswith("linux"):
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=world_size,
                timeout=timedelta(minutes=1),
                # device_id=self.device
            )
        elif sys.platform.startswith("win"):
            os.environ["USE_LIBUV"] = "0"
            if local_rank == 0:
                print("Windows detected, falling back to GLOO backend, consider using WSL, GLOO is slower than NCCL")
            dist.init_process_group(
                "gloo",
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
            raise RuntimeError(
                f"[Rank {local_rank}] COMM test failed: "
                f"got {result}, expected {expected}. "
                f"world_size may be mismatched!"
            )
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


def make_ray_actor_fn(
    world_size,
    parallel_dict
):
    def _init_ray_actor(
        world_size=world_size,
        parallel_dict=parallel_dict
    ):
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
