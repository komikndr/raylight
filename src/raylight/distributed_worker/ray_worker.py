import types
import os
import sys
import gc
from datetime import timedelta

import torch
import torch.distributed as dist
import ray
from torch.distributed.fsdp import FSDPModule

import comfy
from comfy import (
    sd,
    sample,
    utils,
)  # Must manually insert comfy package or ray cannot import raylight to cluster
import comfy.patcher_extension as pe
from comfy import model_base

import raylight.distributed_worker.context_parallel as cp
import raylight.distributed_modules.attention as xfuser_attn
from raylight.comfy_dist.model_patcher import FSDPModelPatcher
from raylight.comfy_dist.sd import load_lora_for_models as ray_load_lora_for_models
from ray.exceptions import RayActorError


def usp_inject_callback(
    model_patcher, device_to, lowvram_model_memory, force_patch_weights, full_load
):
    base_model = model_patcher.model

    if isinstance(base_model, model_base.WAN22_S2V):
        from ..wan.distributed.xdit_context_parallel import (
            usp_audio_dit_forward,
            usp_self_attn_forward,
            usp_t2v_cross_attn_forward,
            usp_audio_injector
        )

        model = base_model.diffusion_model
        print("Initializing USP")
        for block in model.blocks:
            block.self_attn.forward = types.MethodType(usp_self_attn_forward, block.self_attn)
            block.cross_attn.forward = types.MethodType(usp_t2v_cross_attn_forward, block.cross_attn)

        model.audio_injector.forward = types.MethodType(usp_audio_injector, model.audio_injector)
        for inject in model.audio_injector.injector:
            inject.forward = types.MethodType(usp_t2v_cross_attn_forward, inject)

        model.forward_orig = types.MethodType(usp_audio_dit_forward, model)

    elif isinstance(base_model, model_base.WAN21):
        from ..wan.distributed.xdit_context_parallel import (
            usp_self_attn_forward,
            usp_dit_forward,
            usp_i2v_cross_attn_forward,
            usp_t2v_cross_attn_forward
        )
        from comfy.ldm.wan.model import WanT2VCrossAttention, WanI2VCrossAttention

        model = base_model.diffusion_model
        print("Initializing USP")
        for block in model.blocks:
            block.self_attn.forward = types.MethodType(usp_self_attn_forward, block.self_attn)
            if isinstance(block.cross_attn, WanT2VCrossAttention):
                block.cross_attn.forward = types.MethodType(usp_t2v_cross_attn_forward, block.cross_attn)
            elif isinstance(block.cross_attn, WanI2VCrossAttention):
                block.cross_attn.forward = types.MethodType(usp_i2v_cross_attn_forward, block.cross_attn)
        model.forward_orig = types.MethodType(usp_dit_forward, model)

    elif isinstance(base_model, model_base.Flux):
        from ..flux.distributed.xdit_context_parallel import (
            usp_dit_forward,
            usp_single_stream_forward,
            usp_double_stream_forward
        )

        model = base_model.diffusion_model
        print("Initializing USP")
        for block in model.double_blocks:
            block.forward = types.MethodType(usp_double_stream_forward, block)
        for block in model.single_blocks:
            block.forward = types.MethodType(usp_single_stream_forward, block)
        model.forward_orig = types.MethodType(usp_dit_forward, model)

    elif isinstance(base_model, model_base.HunyuanVideo):
        from ..flux.distributed.xdit_context_parallel import (
            usp_single_stream_forward,
            usp_double_stream_forward
        )
        from ..hunyuan_video.distributed.xdit_context_paralllel import (
            usp_dit_forward
        )

        model = base_model.diffusion_model
        print("Initializing USP")
        for block in model.double_blocks:
            block.forward = types.MethodType(usp_double_stream_forward, block)
        for block in model.single_blocks:
            block.forward = types.MethodType(usp_single_stream_forward, block)
        model.forward_orig = types.MethodType(usp_dit_forward, model)

    elif isinstance(base_model, model_base.QwenImage):
        from ..qwen_image.distributed.xdit_context_parallel import (
            usp_dit_forward,
            usp_attn_forward,
        )
        model = base_model.diffusion_model
        print("Initializing USP")
        for block in model.transformer_blocks:
            block.attn.forward = types.MethodType(usp_attn_forward, block.attn)

        model._forward = types.MethodType(usp_dit_forward, model)

    else:
        raise ValueError(
            f"Model: {type(base_model).__name__}, is not yet supported for USP Parallelism"
        )

# Developer reminder, Checking model parameter outside ray actor is very expensive (e.g Comfy main thread)
# the model need to be serialized, send to object store and can cause OOM !, so setter and getter is the pattern !


# If ray actor function being called from outside, ray.get([task in actor task]) will become sync between rank
# If called from ray actor within. dist.barrier() become the sync.

# Comfy cli args, does not get pass through into ray actor
class RayWorker:
    def __init__(self, local_rank, world_size, device_id, parallel_dict):
        self.model = None
        self.model_type = None
        self.noise_add = 0
        self.state_dict = None

        self.local_rank = local_rank
        self.world_size = world_size
        self.device_id = device_id
        self.parallel_dict = parallel_dict
        self.device = torch.device(f"cuda:{self.device_id}")
        self.device_mesh = None
        self.is_model_loaded = False

        if self.parallel_dict["is_xdit"] or self.parallel_dict["is_fsdp"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
            if sys.platform.startswith("linux"):
                dist.init_process_group(
                    "nccl",
                    rank=local_rank,
                    world_size=self.world_size,
                    timeout=timedelta(minutes=1),
                    # device_id=self.device
                )
            elif sys.platform.startswith("win"):
                os.environ["USE_LIBUV"] = "0"
                dist.init_process_group(
                    "gloo",
                    rank=local_rank,
                    world_size=self.world_size,
                    timeout=timedelta(minutes=1),
                    # device_id=self.device
                )
            self.device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(self.world_size,))
            pg = dist.group.WORLD
            cp.set_cp_group(pg, list(range(self.world_size)), local_rank)
        else:
            print(
                f"Running Ray in normal seperate sampler with: {self.world_size} number of workers"
            )
            self.noise_add = self.local_rank

        # From mochi-xdit, xdit, pipelines.py
        if self.parallel_dict["is_xdit"]:
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            xfuser_attn.set_attn_type(self.parallel_dict["attention"])
            cp_rank, cp_size = cp.get_cp_rank_size()
            ulysses_degree = self.parallel_dict["ulysses_degree"]
            ring_degree = self.parallel_dict["ring_degree"]

            print("XDiT is enable")
            init_distributed_environment(rank=cp_rank, world_size=cp_size)

            if ulysses_degree is None and ring_degree is None:
                print(
                    f"No usp config, use default config: ulysses_degree={cp_size}, ring_degree=1"
                )
                initialize_model_parallel(
                    sequence_parallel_degree=self.world_size,
                    ring_degree=1,
                    ulysses_degree=cp_size,
                )
            else:
                print(
                    f"Use usp config: ulysses_degree={ulysses_degree}, ring_degree={ring_degree}"
                )
                initialize_model_parallel(
                    sequence_parallel_degree=self.world_size,
                    ring_degree=ring_degree,
                    ulysses_degree=ulysses_degree,
                )

    def get_parallel_dict(self):
        return self.parallel_dict

    def set_parallel_dict(self, parallel_dict):
        self.parallel_dict = parallel_dict

    def model_function_runner(self, fn, *args, **kwargs):
        self.model = fn(self.model, *args, **kwargs)

    def get_local_rank(self):
        return self.local_rank

    def get_is_model_loaded(self):
        return self.is_model_loaded

    def patch_usp(self):
        self.model.add_callback(
            pe.CallbacksMP.ON_LOAD,
            usp_inject_callback,
        )
        print("USP registered")

    def set_meta_model(self, model):
        first_param_device = next(model.model.parameters()).device
        if first_param_device == torch.device("meta"):
            self.model = model
            self.model.config_fsdp(self.local_rank, self.device_mesh)
        else:
            raise ValueError("Model being set is not meta, can cause OOM in large model")

    def get_meta_model(self):
        first_param_device = next(self.model.model.parameters()).device
        if first_param_device == torch.device("meta"):
            return self.model
        else:
            raise ValueError("Model recieved is not meta, can cause OOM in large model")

    def load_unet(self, unet_path, model_options):
        try:
            if isinstance(self.model.model.diffusion_model, FSDPModule):
                print("WARNING FSDP MODULE ALREADY LOADED, DELETE")
                del self.model
                self.model = None
        except:
            print("NO FSDP MODULE DETECTED")

        self.model = comfy.sd.load_diffusion_model(
            unet_path, model_options=model_options,
        )
        self.model = comfy.sd.load_diffusion_model(
            unet_path, model_options=model_options,
        )
        if self.lora_list is not None:
            self.load_lora()

        if self.parallel_dict["is_fsdp"] is True:
            import comfy.model_patcher as model_patcher
            from raylight.comfy_dist.model_patcher import LowVramPatch
            model_patcher.LowVramPatch = LowVramPatch

            self.model = FSDPModelPatcher.clone(self.model)
            self.state_dict = self.model.model_state_dict()
            self.model.config_fsdp(self.local_rank, self.device_mesh)

        self.is_model_loaded = True

    def set_lora_list(self, lora):
        self.lora_list = lora

    def get_lora_list(self,):
        return self.lora_list

    def patch_fsdp(self,):
        from torch.distributed.fsdp import FSDPModule
        print(f"[Rank {dist.get_rank()}] Applying FSDP to {type(self.model.model.diffusion_model).__name__}")

        if not isinstance(self.model.model.diffusion_model, FSDPModule):
            print("============= BEFORE WRAP ==============")
            print(type(self.model.model.diffusion_model))
            print("THIS IS NOT FSDP MODULE")
            if isinstance(self.model.model, model_base.WAN21) or isinstance(self.model.model, model_base.WAN22):
                from ..wan.distributed.fsdp import shard_model_fsdp2
                self.model.model = shard_model_fsdp2(self.model.model, self.state_dict, self.is_cpu_offload)

            elif isinstance(self.model.model, model_base.Flux):
                from ..flux.distributed.fsdp import shard_model_fsdp2
                self.model.model = shard_model_fsdp2(self.model.model, self.state_dict, self.is_cpu_offload)

            elif isinstance(self.model.model, model_base.QwenImage):
                from ..qwen_image.distributed.fsdp import shard_model_fsdp2
                self.model.model = shard_model_fsdp2(self.model.model, self.state_dict, self.is_cpu_offload)

            elif isinstance(self.model.model, model_base.HunyuanVideo):
                from ..hunyuan_video.distributed.fsdp import shard_model_fsdp2
                self.model.model = shard_model_fsdp2(self.model.model, self.state_dict, self.is_cpu_offload)

            else:
                raise ValueError(f"{type(self.model.model.diffusion_model).__name__} IS CURRENTLY NOT SUPPORTED FOR FSDP")

            self.state_dict = None
            comfy.model_management.soft_empty_cache()
            gc.collect()
            print("============= AFTER WRAP ==============")
            print(type(self.model.model.diffusion_model))
            print("FSDP registered")
        else:
            print("============= NO NEED TO WRAP ==============")
            print("FSDP already registered, skip wrapping...")
            print(type(self.model.model.diffusion_model))

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
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(self.model, latent_image)

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
                latent_image, seed + self.noise_add, batch_inds
            )

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        disable_pbar = comfy.utils.PROGRESS_BAR_ENABLED
        if self.local_rank == 0:
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        if self.parallel_dict["is_fsdp"] is True:
            self.is_cpu_offload = self.parallel_dict["fsdp_cpu_offload"]
            self.patch_fsdp()

        with torch.no_grad():
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
        pg = dist.group.WORLD
        cp.set_cp_group(pg, list(range(world_size)), local_rank)

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


# Just kill the damn actor when lora change, so no dangling memory leak BS when FSDP.
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
                    world_size=world_size,
                    device_id=0,
                    parallel_dict=parallel_dict,
                )
            )
        ray_actors["workers"] = gpu_actors

        for actor in ray_actors["workers"]:
            ray.get(actor.__ray_ready__.remote())
        return ray_actors

    return _init_ray_actor


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

    # ONLY FOR TESTING #
    needs_restart = False
    ####################

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
