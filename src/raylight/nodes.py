import raylight
import os
import gc
import math
from typing import Any
from pathlib import Path
from copy import deepcopy

import ray
import torch
import comfy
import folder_paths
import tempfile
import uuid
from yunchang.kernels import AttnType

# Must manually insert comfy package or ray cannot import raylight to cluster
from comfy import sd, sample, utils # type: ignore

from .distributed_worker.ray_worker import (
    make_ray_actor_fn,
    ensure_fresh_actors,
    ray_nccl_tester,
)
from .comfy_dist.utils import cancellable_get
from .utils_memory import monitor_memory

# Workaround https://github.com/comfyanonymous/ComfyUI/pull/11134
# since in FSDPModelPatcher mode, ray cannot pickle None type cause by getattr
def _monkey():
    from raylight.comfy_dist.supported_models_base import BASE as PatchedBASE
    import comfy.supported_models_base as supported_models_base
    OriginalBASE = supported_models_base.BASE

    if hasattr(PatchedBASE, "__getattr__"):
        setattr(OriginalBASE, "__getattr__", PatchedBASE.__getattr__)


def _resolve_module_dir(module):
    module_file = getattr(module, '__file__', None)
    if module_file:
        path = Path(module_file).resolve()
        if path.is_file():
            return path.parent

    module_paths = getattr(module, '__path__', None)
    if module_paths:
        for path in module_paths:
            if path:
                resolved = Path(path).resolve()
                if resolved.exists():
                    return resolved

    raise RuntimeError(f"Unable to determine module path for {getattr(module, '__name__', module)}")


def _resolve_repo_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'main.py').exists() and (parent / 'execution.py').exists():
            return parent
    raise RuntimeError('Unable to locate ComfyUI repository root')


def _ensure_runtime_workdir(module_dir: Path) -> Path:
    runtime_dir = module_dir.parent / '_ray_runtime_env'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _build_local_runtime_env(module_dir: Path, repo_root: Path, runtime_workdir: Path):
    python_path_entries = [str(repo_root)]
    existing = os.environ.get('PYTHONPATH')
    if existing:
        python_path_entries.extend(part for part in existing.split(os.pathsep) if part)
    python_path = os.pathsep.join(dict.fromkeys(python_path_entries))

    env_vars = {
        'PYTHONPATH': python_path,
        'COMFYUI_BASE_DIRECTORY': str(repo_root),
        'RAY_enable_metrics_collection': '0',
        'RAY_USAGE_STATS_ENABLED': '0',
        'RAY_METRICS_EXPORT_INTERVAL_MS': '0',  # Fully disable metrics export
    }

    return {
        'py_modules': [str(module_dir)],
        'working_dir': str(runtime_workdir),
        'env_vars': env_vars,
        'config': {'eager_install': False},  # Defer module install until actors spawn
    }


def _build_remote_runtime_env(module_dir: Path, repo_root: Path):
    excludes = [
        '.git',
        '.git/**',
        '__pycache__',
        '**/__pycache__',
        '*.pyc',
    ]

    return {
        'py_modules': [str(module_dir)],
        'working_dir': str(repo_root),
        'env_vars': {
            'COMFYUI_BASE_DIRECTORY': '.',
            'RAY_enable_metrics_collection': '0',
            'RAY_USAGE_STATS_ENABLED': '0',
            'RAY_METRICS_EXPORT_INTERVAL_MS': '0',  # Fully disable metrics export
        },
        'excludes': excludes,
        'config': {'eager_install': False},  # Defer module install until actors spawn
    }


_RAYLIGHT_MODULE_PATH = _resolve_module_dir(raylight)
_COMFY_ROOT_PATH = _resolve_repo_root()
_RAYLIGHT_RUNTIME_WORKDIR = _ensure_runtime_workdir(_RAYLIGHT_MODULE_PATH)
_RAY_RUNTIME_ENV_LOCAL = _build_local_runtime_env(
    _RAYLIGHT_MODULE_PATH, _COMFY_ROOT_PATH, _RAYLIGHT_RUNTIME_WORKDIR
)
_RAY_RUNTIME_ENV_REMOTE = _build_remote_runtime_env(_RAYLIGHT_MODULE_PATH, _COMFY_ROOT_PATH)
_LOCAL_CLUSTER_ADDRESSES = {None, '', 'local', 'LOCAL'}


class RayInitializer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_cluster_address": ("STRING", {"default": "local"}),
                "ray_cluster_namespace": ("STRING", {"default": "default"}),
                "GPU": ("INT", {"default": 2}),
                "ulysses_degree": ("INT", {"default": 2}),
                "ring_degree": ("INT", {"default": 1}),
                "cfg_degree": ("INT", {"default": 1}),
                "sync_ulysses": ("BOOLEAN", {"default": False}),
                "FSDP": ("BOOLEAN", {"default": False}),
                "FSDP_CPU_OFFLOAD": ("BOOLEAN", {"default": False}),
                "XFuser_attention": (
                    [member.name for member in AttnType],
                    {"default": "TORCH"},
                ),
                "ray_object_store_gb": ("FLOAT", {
                    "default": 0.0,
                    "tooltip": "Ray shared memory object store size in GB. 0.0 = Auto (Use Ray default ~30% of System RAM). Increase if you see spilling to disk."}),
            },
            "optional": {
                "gpu_indices": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated list of GPU indices to use (e.g., '0,1'). Overrides automatic selection."
                }),
                "skip_comm_test": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip NCCL communication test at startup. Saves ~10-15s but won't detect comm issues early."
                }),
                "pack_qkv": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable combined QKV all-to-all optimization. Only works for self-attention models. Disable for cross-attention (LTXV, SD3, etc.)."
                }),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS_INIT",)
    RETURN_NAMES = ("ray_actors_init",)

    FUNCTION = "spawn_actor"
    CATEGORY = "Raylight"

    def spawn_actor(
        self,
        ray_cluster_address: str,
        ray_cluster_namespace: str,
        GPU: int,
        ulysses_degree: int,
        ring_degree: int ,
        cfg_degree: int,
        sync_ulysses: bool,
        FSDP: bool,
        FSDP_CPU_OFFLOAD: bool,
        XFuser_attention: int,
        ray_object_store_gb: float = 0.0,
        ray_dashboard_address: str = "None",
        torch_dist_address: str = "None",
        gpu_indices: str = "",
        skip_comm_test: bool = True,
        pack_qkv: bool = True,
    ):
        with monitor_memory("RayInitializer.spawn_actor"):
            # THIS IS PYTORCH DIST ADDRESS
            # (TODO) Change so it can be use in cluster of nodes. but it is long waaaaay down in the priority list
            # os.environ['TORCH_CUDA_ARCH_LIST'] = ""
            if torch_dist_address != "None":
                torch_host, torch_port = torch_dist_address.rsplit(":", 1)
                os.environ.setdefault("MASTER_ADDR", torch_host)
                os.environ.setdefault("MASTER_PORT", torch_port)
            else:
                torch_host, torch_port = "127.0.0.1", "29500"
                os.environ.setdefault("MASTER_ADDR", torch_host)
                os.environ.setdefault("MASTER_PORT", torch_port)

            # HF Tokenizer warning when forking
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.parallel_dict: dict[str, Any] = dict()
            _monkey()

            world_size = GPU
            max_world_size = torch.cuda.device_count()
            if world_size > max_world_size:
                raise ValueError("Too many gpus")
            if world_size == 0:
                raise ValueError("Num of cuda/cudalike device is 0")
            if world_size < ulysses_degree * ring_degree * cfg_degree:
                raise ValueError(
                    f"ERROR, num_gpus: {world_size}, is lower than {ulysses_degree=} x {ring_degree=} x {cfg_degree=}"
                )
            if cfg_degree > 2:
                raise ValueError(
                    "CFG batch only can be divided into 2 degree of parallelism, since its dimension is only 2"
                )

            self.parallel_dict["is_xdit"] = False
            self.parallel_dict["is_fsdp"] = False
            self.parallel_dict["sync_ulysses"] = False
            self.parallel_dict["global_world_size"] = world_size

            if (
                ulysses_degree > 0
                or ring_degree > 0
                or cfg_degree > 0
            ):
                if ulysses_degree * ring_degree * cfg_degree == 0:
                    raise ValueError(f"""ERROR, parallel product of {ulysses_degree=} x {ring_degree=} x {cfg_degree=} is 0.
                     Please make sure to set any parallel degree to be greater than 0,
                     or switch into DPKSampler and set 0 to all parallel degree""")
                self.parallel_dict["attention"] = XFuser_attention
                self.parallel_dict["is_xdit"] = True
                self.parallel_dict["ulysses_degree"] = ulysses_degree
                self.parallel_dict["ring_degree"] = ring_degree
                self.parallel_dict["cfg_degree"] = cfg_degree
                self.parallel_dict["sync_ulysses"] = sync_ulysses
                self.parallel_dict["pack_qkv"] = pack_qkv

            if FSDP:
                self.parallel_dict["fsdp_cpu_offload"] = FSDP_CPU_OFFLOAD
                self.parallel_dict["is_fsdp"] = True

            if ray_dashboard_address != "None":
                dashboard_host, dashboard_port = ray_dashboard_address.rsplit(":", 1)
                dashboard_port = int(dashboard_port)
                enable_dashboard = True
            else:
                dashboard_host, dashboard_port = "127.0.0.1", None
                enable_dashboard = False

            if ray_object_store_gb <= 0:
                ray_object_store_memory = None
                print("[Raylight] object_store_memory set to Auto (Ray default).")
            else:
                ray_object_store_memory = int(ray_object_store_gb * 1024**3)
                print(f"[Raylight] object_store_memory set to {ray_object_store_gb} GB.")

            runtime_env_base = _RAY_RUNTIME_ENV_LOCAL
            if ray_cluster_address not in _LOCAL_CLUSTER_ADDRESSES:
                runtime_env_base = _RAY_RUNTIME_ENV_REMOTE

            # GPU Pinning Logic
            original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                if gpu_indices.strip():
                    # Validate and set
                    indices = [x.strip() for x in gpu_indices.split(",") if x.strip()]
                    if len(indices) < world_size:
                         raise ValueError(f"gpu_indices contains {len(indices)} GPUs, but {world_size} (GPU input) were requested.")
                    
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)
                    print(f"[Raylight] Pinning Ray Cluster to GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

                # ===== OPTIMIZATION: Cluster Reuse =====
                # Check if Ray is already initialized with matching config to skip expensive re-init
                is_local = ray_cluster_address in _LOCAL_CLUSTER_ADDRESSES
                should_reuse = False
                
                if ray.is_initialized():
                    try:
                        # Check if the cluster has matching GPU count
                        existing_resources = ray.cluster_resources()
                        existing_gpus = int(existing_resources.get('GPU', 0))
                        if existing_gpus >= world_size:
                            print(f"[Raylight] Reusing existing Ray cluster (GPUs available: {existing_gpus})")
                            should_reuse = True
                    except Exception:
                        pass
                
                if not should_reuse:
                    # Shut down so if comfy user try another workflow it will not cause error
                    ray.shutdown()
                    
                    # Build init kwargs - disable metrics agent for faster startup
                    init_kwargs = {
                        'namespace': ray_cluster_namespace,
                        'runtime_env': deepcopy(runtime_env_base),
                        'include_dashboard': enable_dashboard,
                        'dashboard_host': dashboard_host,
                        'dashboard_port': dashboard_port,
                        '_metrics_export_port': None,  # Disable metrics agent to avoid connection retries
                    }
                    
                    # Only set object_store_memory if explicitly configured (not for reused clusters)
                    if ray_object_store_memory is not None:
                        init_kwargs['object_store_memory'] = ray_object_store_memory
                    
                    ray.init(ray_cluster_address, **init_kwargs)
                    print(f"[Raylight] Ray cluster initialized (new instance)")
                
            except Exception as e:
                ray.shutdown()
                ray.init(
                    runtime_env=deepcopy(runtime_env_base),
                    _metrics_export_port=None,
                )
                raise RuntimeError(f"Ray connection failed: {e}")
            # Restore original environment to avoid affecting other nodes
            if original_visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices
            elif "CUDA_VISIBLE_DEVICES" in os.environ and gpu_indices.strip():
                 del os.environ["CUDA_VISIBLE_DEVICES"]

            # ===== OPTIMIZATION: Skip NCCL Test =====
            # NCCL test spawns/kills separate actors before real workers - saves ~10-15s
            if not skip_comm_test:
                print("[Raylight] Running NCCL communication test...")
                ray_nccl_tester(world_size)
            else:
                print("[Raylight] Skipping NCCL test (skip_comm_test=True)")
            
            ray_actor_fn = make_ray_actor_fn(world_size, self.parallel_dict)
            ray_actors = ray_actor_fn()
            
            # Store GPU indices for later use in samplers (for partial offload matching)
            if gpu_indices.strip():
                ray_actors["gpu_indices"] = [int(x.strip()) for x in gpu_indices.split(",") if x.strip()]
            else:
                # Default: 0, 1, 2, ...
                ray_actors["gpu_indices"] = list(range(world_size))
            
            return ([ray_actors, ray_actor_fn],)


class RayInitializerAdvanced(RayInitializer):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_cluster_address": ("STRING", {
                    "default": "local",
                    "tooltip": "Address of Ray cluster different than torch distributed address"}),
                "ray_cluster_namespace": ("STRING", {"default": "default"}),
                "ray_object_store_gb": ("FLOAT", {
                    "default": 0.0,
                    "tooltip": "Ray shared memory object store size in GB. 0.0 = Auto (Use Ray default ~30% of System RAM). Increase if you see spilling to disk."}),
                "ray_dashboard_address": ("STRING", {
                    "default": "None",
                    "tooltip": "Same format as torch_dist_address, you need to install ray dashboard to monitor"}),
                "torch_dist_address": ("STRING", {
                    "default": "127.0.0.1:29500",
                    "tooltip": "Might need to restart ComfyUI to apply"}),
                "GPU": ("INT", {"default": 2}),
                "ulysses_degree": ("INT", {"default": 2}),
                "ring_degree": ("INT", {"default": 1}),
                "cfg_degree": ("INT", {"default": 1}),
                "sync_ulysses": ("BOOLEAN", {"default": False}),
                "FSDP": ("BOOLEAN", {"default": False}),
                "FSDP_CPU_OFFLOAD": ("BOOLEAN", {"default": False}),
                "XFuser_attention": (
                    [
                        "TORCH",
                        "FLASH_ATTN",
                        "FLASH_ATTN_3",
                        "SAGE_AUTO_DETECT",
                        "SAGE_FP16_TRITON",
                        "SAGE_FP16_CUDA",
                        "SAGE_FP8_CUDA",
                        "SAGE_FP8_SM90",
                        "AITER_ROCM",
                    ],
                    {"default": "TORCH"},
                ),
            },
            "optional": {
                "gpu_indices": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated list of GPU indices to use (e.g., '0,1'). Overrides automatic selection."
                }),
                "skip_comm_test": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip NCCL communication test at startup. Saves ~10-15s but won't detect comm issues early."
                }),
                "pack_qkv": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable combined QKV all-to-all optimization. Only works for self-attention models. Disable for cross-attention (LTXV, SD3, etc.)."
                }),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS_INIT",)
    RETURN_NAMES = ("ray_actors_init",)

    FUNCTION = "spawn_actor"
    CATEGORY = "Raylight"


class RayUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models")
                              + folder_paths.get_filename_list("checkpoints"),),
                "weight_dtype": (
                    [
                        "default",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e5m2",
                        "bf16",
                        "fp16",
                    ],
                ),
                "ray_actors_init": (
                    "RAY_ACTORS_INIT",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "load_ray_unet"

    CATEGORY = "Raylight"

    def load_ray_unet(self, ray_actors_init, unet_name, weight_dtype):
        with monitor_memory("RayUNETLoader.load_ray_unet"):
            ray_actors, gpu_actors, parallel_dict = ensure_fresh_actors(ray_actors_init)

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        try:
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        except:
            unet_path = folder_paths.get_full_path_or_raise("checkpoints", unet_name)


        loaded_futures = []
        patched_futures = []

        if parallel_dict["is_fsdp"] is True:
            worker0 = ray.get_actor("RayWorker:0")
            cancellable_get(worker0.load_unet.remote(unet_path, model_options=model_options))
            meta_model = cancellable_get(worker0.get_meta_model.remote())

            for actor in gpu_actors:
                if actor != worker0:
                    loaded_futures.append(actor.set_meta_model.remote(meta_model))

            cancellable_get(loaded_futures)
            loaded_futures = []

            for actor in gpu_actors:
                loaded_futures.append(actor.set_state_dict.remote())

            cancellable_get(loaded_futures)
            loaded_futures = []
        else:
            # Parallel Loading Mode: Trigger load_unet on all workers.
            # RayWorker is responsible for "Lightweight Ref" logic if needed to save RAM.
            for actor in gpu_actors:
                loaded_futures.append(
                    actor.load_unet.remote(unet_path, model_options=model_options)
                )
            cancellable_get(loaded_futures)
            loaded_futures = []

        for actor in gpu_actors:
            if parallel_dict["is_xdit"]:
                if (parallel_dict["ulysses_degree"]) > 1 or (parallel_dict["ring_degree"] > 1):
                    patched_futures.append(actor.patch_usp.remote())
                if parallel_dict["cfg_degree"] > 1:
                    patched_futures.append(actor.patch_cfg.remote())

        cancellable_get(patched_futures)

        return (ray_actors,)





class XFuserKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
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
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "ray_actors": (
                    "RAY_ACTORS",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "ray_sample"

    CATEGORY = "Raylight"

    def ray_sample(
        self,
        ray_actors,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
        sigmas=None,
    ):
        with monitor_memory("XFuserKSampler.ray_sample"):
            # Clean VRAM for preparation to load model
            pass
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        gpu_actors = ray_actors["workers"]

        futures = [
            actor.common_ksampler.remote(
                noise_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=force_full_denoise,
                sigmas=sigmas,
            )
            for actor in gpu_actors
        ]

        results = cancellable_get(futures)
        return (results[0][0],)


class DPKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "add_noise": (["enable", "disable"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
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
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "ray_actors": (
                    "RAY_ACTORS",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
                "noise_list": ("NOISE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "ray_sample"

    CATEGORY = "Raylight"

    def ray_sample(
        self,
        ray_actors,
        add_noise,
        noise_list,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):

        ray_actors = ray_actors[0]
        add_noise = add_noise[0]
        steps = steps[0]
        cfg = cfg[0]
        sampler_name = sampler_name[0]
        scheduler = scheduler[0]
        positive = positive[0]
        negative = negative[0]
        start_at_step = start_at_step[0]
        end_at_step = end_at_step[0]
        return_with_leftover_noise = return_with_leftover_noise[0]

        gpu_actors = ray_actors["workers"]

        parallel_dict = cancellable_get(gpu_actors[0].get_parallel_dict.remote())
        if parallel_dict["is_xdit"] is True:
            raise ValueError(
                """
            Data Parallel KSampler only supports FSDP or standard Data Parallel (DP).
            Please set both 'ulysses_degree' and 'ring_degree' to 0,
            or use the XFuser KSampler instead. More info on Raylight mode https://github.com/komikndr/raylight
            """
            )

        if len(latent_image) != len(gpu_actors):
            latent_image = [latent_image[0]] * len(gpu_actors)

        # Clean VRAM for preparation to load model
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        futures = [
            actor.common_ksampler.remote(
                noise_list[i],
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image[i],
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=force_full_denoise,
            )
            for i, actor in enumerate(gpu_actors)
        ]

        results = cancellable_get(futures)
        results = [result[0] for result in results]
        return (results,)


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


class DPNoiseList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **{
                    f"noise_seed_{i}": (
                        "INT",
                        {
                            "default": 0,
                            "min": 0,
                            "max": 0xFFFFFFFFFFFFFFFF,
                            "control_after_generate": True,
                        },
                    )
                    for i in range(8)
                }
            }
        }

    RETURN_TYPES = ("NOISE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "get_noise"
    CATEGORY = "Raylight"

    def get_noise(self, **kwargs):
        noise_list = []
        for key, seed in kwargs.items():
            if key.startswith("noise_seed_"):
                noise_list.append(seed)
        return (noise_list,)


class RayVAEDecodeDistributed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS", {"tooltip": "Ray Actor to submit the model into"}),
                "samples": ("LATENT",),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "vae_dtype": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "VAE precision: auto=bf16 on RTX3000+/fp32 fallback, bf16=bfloat16 (recommended), fp16=half (may cause NaN), fp32=full (stable but 2x memory)"
                }),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32},),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": (
                    "INT",
                    {
                        "default": 64,
                        "min": 8,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Only used for video VAEs: Amount of frames to decode at a time.",
                    },
                ),
                "temporal_overlap": (
                    "INT",
                    {
                        "default": 8,
                        "min": 4,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Only used for video VAEs: Amount of frames to overlap.",
                    },
                ),
                "release_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Release VAE from worker RAM after decode. Recommended True to free ~2GB per worker. Set False if you need the VAE for multiple decodes."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ray_decode"

    CATEGORY = "Raylight"

    def ray_decode(self, ray_actors, vae_name, samples, tile_size, vae_dtype="auto", overlap=64, temporal_size=64, temporal_overlap=8, release_vae=True):
        gpu_actors = ray_actors["workers"]
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)

        # 1. Load VAE on all workers
        for actor in gpu_actors:
            cancellable_get(actor.ray_vae_loader.remote(vae_path))

        # 2. Shard samples temporally
        latents = samples["samples"]
        num_workers = len(gpu_actors)
        total_frames = latents.shape[2]

        # Core ComfyUI VAEDecodeTiled safety checks
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2

        # Pre-calculate total output size and compression factors early
        # Retrieve compression factors directly from the worker that just loaded the VAE
        temporal_compression = cancellable_get(gpu_actors[0].get_vae_temporal_compression.remote()) or 1
        spatial_compression = cancellable_get(gpu_actors[0].get_vae_spatial_compression.remote()) or 1

        # Causal VAE output formula: (Latent_T - 1) * compression + 1
        if temporal_compression > 1:
            total_output_frames = (total_frames - 1) * temporal_compression + 1
            overlap_latent_frames = 1 # 1 frame overlap is sufficient for continuity in causal VAE
        else:
            total_output_frames = total_frames
            overlap_latent_frames = 0
            
        # Calculate Master Output Shape (B, T, H, W, C)
        # We assume Batch=1 for video usually, but we handle standard (B, T, H, W, C) structure 
        # or (T, H, W, C) if squeezed. ComfyUI usually expects (TotalFrames, H, W, 3) for video batch.
        H_out = latents.shape[3] * spatial_compression
        W_out = latents.shape[4] * spatial_compression
        # Final shape: (TotalFrames, Height, Width, 3)
        master_shape = (total_output_frames, H_out, W_out, 3)
        
        # 3. Create Shared Memory File (Pre-allocation)
        mmap_path = f"/dev/shm/raylight_vae_out_{uuid.uuid4().hex}.bin"
        num_elements = 1
        for dim in master_shape: num_elements *= dim
        file_size_bytes = num_elements * 4 # float32 = 4 bytes
        
        print(f"[RayVAEDecode] Pre-allocating shared output buffer: {mmap_path} ({file_size_bytes/1024**3:.2f} GB)")
        print(f"[RayVAEDecode] Output Shape: {master_shape}")
        
        full_image = None # Initialize outside try block for finally access
        try:
            with open(mmap_path, "wb") as f:
                f.seek(file_size_bytes - 1)
                f.write(b"\0")
                
            # Create the tensor wrapper immediately
            full_image = torch.from_file(mmap_path, shared=True, size=num_elements, dtype=torch.float32).reshape(master_shape)
            
            # 4. Dispatch Workers
            # NOTE: Do NOT pre-divide temporal parameters by compression factor.
            # Original ComfyUI's decode_tiled passes tile_t and overlap_t directly to decode_tiled_3d.
            
            frames_per_shard = (total_frames + num_workers - 1) // num_workers
            
            futures = []
            for i, actor in enumerate(gpu_actors):
                start = i * frames_per_shard
                end = min((i + 1) * frames_per_shard, total_frames)
                
                if start >= total_frames:
                    continue # No work for this worker
                    
                # context_start: Provide 1 frame of context to ensure continuity
                context_start = max(0, start - 1)
                # actual_end: Each shard must produce frames up to the START of the next shard.
                actual_end = min(end + (1 if i < num_workers - 1 else 0), total_frames)
                
                shard_samples = {
                    "samples": latents[:, :, context_start:actual_end].clone()
                }
                
                # Pass how many latent frames to discard from the beginning of the result
                discard_latent_frames = start - context_start
                
                # Calculate Output Offset due to temporal compression
                # shard index 'start' corresponds to video frame:
                start_video_frame = start * temporal_compression
                if temporal_compression == 1:
                     start_video_frame = start
                
                print(f"[RayVAEDecode] Shard {i}: Latents {context_start} to {end} (discard {discard_latent_frames}) -> Video Frame {start_video_frame}")
                
                futures.append(actor.ray_vae_decode.remote(
                    i, # shard_index
                    shard_samples,
                    tile_size,
                    overlap=overlap,
                    temporal_size=temporal_size,  # Pass raw value, decode_tiled handles scaling
                    temporal_overlap=temporal_overlap,  # Pass raw value, decode_tiled handles scaling
                    discard_latent_frames=discard_latent_frames,
                    vae_dtype=vae_dtype,
                    mmap_path=mmap_path,     # DIRECT WRITING
                    mmap_shape=master_shape, # DIRECT WRITING
                    output_offset=start_video_frame # DIRECT WRITING
                ))
    
            print(f"[RayVAEDecode] Dispatched {len(futures)} shards. Direct-to-disk mode enabled.")
            
            # 5. Gather Results (Stats Only)
            remaining = list(futures)
            futures_count = len(futures)
            received_count = 0
            
            while remaining:
                # Check ComfyUI cancel status
                if comfy.model_management.processing_interrupted():
                    print("[Raylight] Cancellation detected during VAE decoding! Force-canceling Ray tasks...")
                    for ref in remaining:
                        try:
                            ray.cancel(ref, force=True, recursive=True)
                        except:
                            pass
                    raise Exception("Raylight: VAE Decode canceled by user.")

                ready, remaining = ray.wait(remaining, num_returns=1, timeout=1.0)
                if not ready:
                    continue
                    
                for ray_ref in ready:
                    shard_index, result = cancellable_get(ray_ref)
                    received_count += 1
                    
                    if isinstance(result, dict) and result.get("mmap", False):
                        # Direct write success
                        shape = result["shape"]
                        stats = result["stats"]
                        print(f"[RayVAEDecode] Shard {shard_index} wrote {shape} to mmap. Stats: {stats}")
                    else:
                        # Fallback (Legacy / Failure fallback)
                        shard_data = result
                        print(f"[RayVAEDecode] Shard {shard_index} returned tensor {shard_data.shape} (Fallback path)")
                        
                        # Write to mmap manually
                        start = shard_index * frames_per_shard
                        start_video_frame = start * temporal_compression
                        s_len = shard_data.shape[0]
                        end_video_frame = start_video_frame + s_len
                        
                        # Handle truncation if needed
                        if end_video_frame > total_output_frames:
                            s_len = max(0, total_output_frames - start_video_frame)
                            shard_data = shard_data[:s_len]
                            end_video_frame = start_video_frame + s_len
                        
                        if s_len > 0:
                            full_image[start_video_frame:end_video_frame] = shard_data.to(torch.float32)

                        del shard_data

            del remaining
            del futures # Ensure all futures are deleted to release Ray object store references

            # Add batch dimension if needed to match ComfyUI expectation (1, T, H, W, C)?
            # Usually images are just a list of frames. Tensor shape (T, H, W, C).
            # ComfyUI "IMAGE" type is (BATCH, H, W, C). For video, BATCH = Frames.
            # So (T, H, W, 3) is correct.
            
            # 6. Cleanup shared file?
            # If we delete the file now, full_image relies on valid fd/mapping.
            # Since torch.from_file uses Mmap, we should keep the file until we don't need it?
            # Or reliance on unlinked file logic (Linux only).
            # To be safe and cross-platform compatible, we should ideally keep it.
            # BUT we want to ensure it gets cleaned up.
            # We can register it with a cleaner or rely on /tmp cleaners.
            # For now, let's unlink it. On Linux, unlinking an open file keeps it alive for the process.
            # Torch likely holds the fd.
            try:
                os.unlink(mmap_path)
                print(f"[RayVAEDecode] Unlinked temp mmap file: {mmap_path}")
            except Exception as e:
                print(f"[RayVAEDecode] Warning: Could not unlink mmap file: {e}")

            # 7. Release VAE from workers to free RAM for downstream operations (e.g., AudioVAE)
            if release_vae:
                print("[RayVAEDecode] Releasing VAE from workers to free RAM...")
                release_futures = [actor.ray_vae_release.remote() for actor in gpu_actors]
                cancellable_get(release_futures)
                print("[RayVAEDecode] VAE released from all workers.")

            return (full_image,)
            
        except Exception as e:
            # Cleanup on failure
            if os.path.exists(mmap_path):
                try: 
                    os.unlink(mmap_path) 
                except: pass
            raise e


class RayOffloadModel:
    """
    Offloads the diffusion model from all Ray workers' VRAM.
    Place this node after the sampler to free GPU memory.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
            },
            "optional": {
                "latent": ("LATENT", {"default": None, "tooltip": "Passthrough for workflow chaining"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "offload"
    CATEGORY = "Raylight"

    def offload(self, ray_actors, latent=None):
        gpu_actors = ray_actors["workers"]
        
        # Offload from all workers
        offload_futures = [actor.offload_and_clear.remote() for actor in gpu_actors]
        cancellable_get(offload_futures)
        
        print("[RayOffloadModel] All workers offloaded.")
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "XFuserKSamplerAdvanced": XFuserKSamplerAdvanced,
    "DPKSamplerAdvanced": DPKSamplerAdvanced,
    "RayUNETLoader": RayUNETLoader,

    "RayInitializer": RayInitializer,
    "RayInitializerAdvanced": RayInitializerAdvanced,
    "DPNoiseList": DPNoiseList,
    "RayVAEDecodeDistributed": RayVAEDecodeDistributed,
    "RayOffloadModel": RayOffloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XFuserKSamplerAdvanced": "XFuser KSampler (Advanced)",
    "DPKSamplerAdvanced": "Data Parallel KSampler (Advanced)",
    "RayUNETLoader": "Load Diffusion Model (Ray)",

    "RayInitializer": "Ray Init Actor",
    "RayInitializerAdvanced": "Ray Init Actor (Advanced)",
    "DPNoiseList": "Data Parallel Noise List",
    "RayVAEDecodeDistributed": "Distributed VAE (Ray)",
    "RayOffloadModel": "Offload Model (Ray)",
}
