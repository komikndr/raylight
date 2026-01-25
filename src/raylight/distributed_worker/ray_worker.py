import os
import sys
import gc
import types
import ctypes
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


from raylight.utils.common import Noise_EmptyNoise, Noise_RandomNoise, patch_ray_tqdm, cleanup_memory, force_malloc_trim
from raylight.utils.gguf import evict_page_cache, check_mmap_leak
from raylight.utils.memory import monitor_memory
from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops
from raylight.comfy_dist.utils import cancellable_get
from ray.exceptions import RayActorError
from contextlib import contextmanager
import inspect

from raylight.distributed_worker.managers.lora_manager import LoraManager
from raylight.distributed_worker.managers.vae_manager import VaeManager
from raylight.distributed_worker.managers.sampler_manager import SamplerManager


# Developer reminder, Checking model parameter outside ray actor is very expensive (e.g Comfy main thread)
# the model need to be serialized, send to object store and can cause OOM !, so setter and getter is the pattern !


# If ray actor function being called from outside, ray.get([task in actor task]) will become sync between rank
# If called from ray actor within. dist.barrier() become the sync.



# Comfy cli args, does not get pass through into ray actor

class RayWorker:



    def __init__(self, local_rank, device_id, parallel_dict):
        self.local_rank = local_rank
        self.device_id = device_id
        self.parallel_dict = parallel_dict
        self.global_world_size = self.parallel_dict["global_world_size"]

        self.model = None
        self.model_type = None
        self.state_dict = None
        self.overwrite_cast_dtype = None
        
        # Managers
        self.lora_manager = LoraManager(self)
        self.vae_manager = VaeManager(self)
        self.sampler_manager = SamplerManager(self)
        
        # LRU cache for mmap'd state dicts (replaces worker_mmap_cache)
        from raylight.utils.cache import LRUStateCache
        cache_size = parallel_dict.get("mmap_cache_size", 2)
        self.state_cache = LRUStateCache(max_size=cache_size)
        
        # Legacy alias for backward compatibility
        self.worker_mmap_cache = self.state_cache

        # CRITICAL: Apply LowVramPatch monkey-patch early to ensure ALL model types
        # (GGUF, standard, FSDP) use Raylight's LoRA-compatible patch handler.
        # This must happen before any model loading.
        self._apply_global_patches()

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
            xfuser_attn.set_pack_qkv(self.parallel_dict.get("pack_qkv", False))

            self.cp_degree = self.parallel_dict["ulysses_degree"] * parallel_dict["ring_degree"]
            self.cfg_degree = self.parallel_dict["cfg_degree"]
            self.ulysses_degree = self.parallel_dict["ulysses_degree"]
            self.ring_degree = self.parallel_dict["ring_degree"]
            self.cfg_degree = self.parallel_dict["cfg_degree"]
            
            # Cleanup previous state if any (Robustness for restarts)
            try:
                from xfuser.core.distributed import parallel_state
                if parallel_state.get_world_size() > 1:
                    parallel_state.destroy_model_parallel()
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as e:
                print(f"[Raylight] Cleanup warning: {e}")

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

    # Legacy debug_memory_leaks removed


    def model_function_runner_get_values(self, fn, *args, **kwargs):
        return fn(self.model, *args, **kwargs)

    def get_local_rank(self):
        return self.local_rank

    def get_device_id(self):
        return self.device_id

    def get_is_model_loaded(self):
        return self.is_model_loaded

    def _offload_gguf_soft(self):
        """Zero-Copy Offload: Restores mmap references to clear VRAM without copies."""
        if self.model is not None:
            print(f"[RayWorker {self.local_rank}] GGUF Offload: Performing Zero-Copy Pointer Swap...")
            
            # FIRST: Clear all LoRA/patch GPU references before pointer swap
            # This releases GPU tensors so they can be GC'd after unpatch
            self._clear_lora_gpu_refs()
            
            # Use our optimized unpatch_model which restores mmap pointers
            self.model.unpatch_model(device_to=torch.device('cpu'), unpatch_weights=True)
            self.model.current_device = torch.device('cpu')
            
            # Store in worker-level cache to survive model deletions/swaps
            if hasattr(self.model, "mmap_cache") and self.model.mmap_cache:
                unet_path = getattr(self.model, "unet_path", None)
                if unet_path:
                    self.worker_mmap_cache[unet_path] = self.model.mmap_cache
            
            cleanup_memory()
            print(f"[RayWorker {self.local_rank}] GGUF Soft-Offload Complete (VRAM freed, Mapping preserved).")
        else:
            self.is_model_loaded = False

    def _clear_lora_gpu_refs(self):
        """Clear all GPU tensor references from LoRA/patches to allow proper VRAM release."""
        self.lora_manager.clear_gpu_refs()

    def _offload_standard(self):
        print(f"[RayWorker {self.local_rank}] Offloading model (releasing VRAM)...")
        model_to_remove = self.model
        self.model = None
        
        # Clear tracking vars to force reload next time
        self.current_unet_path = None
        self.current_sd_ref = None
        
        cleanup_memory()
        print(f"[RayWorker {self.local_rank}] Offload complete.")

        # Manually deregister from ComfyUI's cache
        try:
            mm = comfy.model_management
            if model_to_remove in mm.current_loaded_models:
                print(f"[RayWorker {self.local_rank}] Removing model from ComfyUI cache: {model_to_remove}")
                mm.current_loaded_models.remove(model_to_remove)
        except Exception as e:
            print(f"[RayWorker {self.local_rank}] Warning in cache removal: {e}")

        del model_to_remove
        
        cleanup_memory()
        
        print(f"[RayWorker {self.local_rank}] Model offloaded successfully.")

    def offload_and_clear(self):
        """
        Offloads the model from VRAM and clears tracking state.
        Uses ModelContext for format-specific offload, with shared cleanup.
        """
        if self.model is None:
            print(f"[RayWorker {self.local_rank}] No model to offload.")
            return
        
        # Delegate format-specific offload to context
        from raylight.distributed_worker.model_context import get_context
        unet_path = getattr(self.model, "unet_path", "")
        ctx = get_context(self, unet_path)
        print(f"[RayWorker {self.local_rank}] Offloading via {ctx.__class__.__name__}...")
        ctx.offload()
        
        # Common cleanup for all formats
        self._common_offload_cleanup()
    
    def _common_offload_cleanup(self):
        """Shared post-offload cleanup for all model formats."""
        # Clear LoRA tracking for fresh re-application after reload
        self.lora_manager.clear_tracking()
        print(f"[RayWorker {self.local_rank}] LoRA tracking cleared.")
        
        # Move buffers to CPU if model still exists
        if self.model is not None:
            diffusion_model = getattr(self.model, "model", None)
            if diffusion_model is not None:
                try:
                    for name, buf in diffusion_model.named_buffers():
                        if buf is not None and buf.device.type == 'cuda':
                            buf.data = buf.data.to('cpu')
                except Exception as e:
                    print(f"[RayWorker {self.local_rank}] Buffer cleanup warning: {e}")
                
                # Delete cached PE tensors
                for attr in ['_cached_pe', 'cached_pe', 'pe_cache', '_pe']:
                    if hasattr(diffusion_model, attr):
                        delattr(diffusion_model, attr)
                    if hasattr(diffusion_model, 'diffusion_model'):
                        inner = diffusion_model.diffusion_model
                        if hasattr(inner, attr):
                            delattr(inner, attr)
        
        # Force garbage collection and CUDA cleanup
        cleanup_memory()
        mem_now = torch.cuda.memory_allocated(self.device) / 1e9
        print(f"[RayWorker {self.local_rank}] Post-Offload VRAM: {mem_now:.2f}GB")


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

    def apply_ffn_chunking(self, num_chunks: int = 8, verbose: bool = True):
        """
        Apply FFN chunking to reduce peak VRAM for LTX-2 models.
        
        This wraps FeedForward layers with ChunkedFFN to process sequences
        in smaller batches, reducing peak memory by ~8x.
        """
        if self.model is None:
            print(f"[RayWorker {self.local_rank}] No model loaded, skipping FFN chunking.")
            return {"ffn_found": 0, "ffn_wrapped": 0}
        
        from raylight.comfy_extra_dist.nodes_ltx_ffn_chunker import wrap_ffn_layers
        
        if verbose:
            print(f"[RayWorker {self.local_rank}] Applying FFN chunking with {num_chunks} chunks...")
        
        info = wrap_ffn_layers(self.model, num_chunks, verbose)
        
        if verbose:
            print(f"[RayWorker {self.local_rank}] FFN chunking complete: {info}")
        
        return info

    def _apply_global_patches(self):
        """Apply monkey patches that are needed for ALL model types.
        
        This is called once during worker initialization to ensure consistent
        behavior across GGUF, Standard, and FSDP models.
        """
        import comfy.model_patcher as model_patcher
        from raylight.comfy_dist.model_patcher import LowVramPatch
        import comfy.lora
        from raylight.comfy_dist.lora import calculate_weight
        
        # Patch LowVramPatch to use Raylight's LoRA-compatible calculate_weight
        # This is critical for LoRA support across all model types.
        model_patcher.LowVramPatch = LowVramPatch
        
        # Patch comfy.lora.calculate_weight to support LoRAAdapter objects
        comfy.lora.calculate_weight = calculate_weight
        
        print(f"[RayWorker {self.local_rank}] Applied global LowVramPatch and LoRA monkey-patches.")

    def _apply_fsdp_patches(self):
        # Monkey patch
        import comfy.model_patcher as model_patcher
        import comfy.model_management as model_management

        # Monkey patch
        from raylight.comfy_dist.model_management import cleanup_models_gc
        from raylight.comfy_dist.model_patcher import LowVramPatch
        
        # Monkey patch
        model_patcher.LowVramPatch = LowVramPatch
        model_management.cleanup_models_gc = cleanup_models_gc

    def _load_unet_fsdp(self, unet_path, model_options):
        # Apply One-Time Patches
        self._apply_fsdp_patches()
        from raylight.comfy_dist.sd import fsdp_load_diffusion_model

        del self.model
        del self.state_dict
        self.model = None
        self.state_dict = None
        cleanup_memory()

        self.model, self.state_dict = fsdp_load_diffusion_model(
            unet_path,
            self.local_rank,
            self.device_mesh,
            self.is_cpu_offload,
            model_options=model_options,
        )
        cleanup_memory()

        import ray
        if self.state_dict is not None:
            print(f"[RayWorker {self.local_rank}] FSDP Mode: Saving base_sd_ref to Object Store...")
            self.base_sd_ref = ray.put(self.state_dict)

    def _load_model_generic(self, unet_path, model_options, dequant_dtype=None, patch_dtype=None):
        """
        Unified loader for Standard (FP16/32) and GGUF models.
        Uses ModelContext pattern with optional mmap and LRU caching.
        """
        from raylight.distributed_worker.model_context import get_context, ModelState
        
        # Idempotency Check: Skip reload if nothing changed
        if (self.model is not None and 
            hasattr(self, "reload_params") and 
            self.reload_params.get("unet_path") == unet_path and
            self.reload_params.get("dequant_dtype") == dequant_dtype and
            self.reload_params.get("patch_dtype") == patch_dtype):
            print(f"[RayWorker {self.local_rank}] Idempotent Load: Model {os.path.basename(unet_path)} already loaded. Skipping.")
            
            # Re-apply manual cast if needed (cheap)
            load_options = model_options.copy()
            cast_dtype = load_options.pop("dtype", None)
            if cast_dtype and hasattr(self.model, "model"):
                self.model.model.manual_cast_dtype = cast_dtype
            return
        
        # Build state and delegate to appropriate context
        state = ModelState(unet_path, model_options, dequant_dtype, patch_dtype)
        ctx = get_context(self, unet_path)
        ctx.load(state)
        
        # Set format flag based on file type
        self.is_gguf = unet_path.lower().endswith(".gguf") or dequant_dtype is not None
        
        # Store cast dtype for overwrite scenarios
        if hasattr(self.model, "model") and self.model.model is not None:
            self.overwrite_cast_dtype = getattr(self.model.model, 'manual_cast_dtype', None)
        
        # GGUF: Force immediate VRAM load
        if self.is_gguf and hasattr(self.model, "load"):
            print(f"[RayWorker {self.local_rank}] GGUF: Forcing immediate VRAM load...")
            self.model.load(self.device, force_patch_weights=True)
            self.model.current_device = self.device
            cleanup_memory()
            check_mmap_leak(unet_path)
        
        # Unified Coordination Ref (for Ray checks)
        import ray
        self.base_sd_ref = ray.put({
            "mode": "disk_parallel",
            "path": unet_path,
            "is_gguf": self.is_gguf
        })

    def load_unet(self, unet_path, model_options):
        with monitor_memory(f"RayWorker {self.local_rank} - load_unet", device=self.device):
            # Smart Reload: Skip if already loaded
            if self.model is not None and getattr(self, "reload_params", {}).get("unet_path") == unet_path:
                    print(f"[RayWorker {self.local_rank}] Smart Reload: Model {unet_path} already loaded. Skipping.")
                    self.is_model_loaded = True
                    return

        if self.parallel_dict["is_fsdp"] is True:
            self._load_unet_fsdp(unet_path, model_options)
            # FSDP tracking
            self.reload_params = {
                "unet_path": unet_path,
                "model_options": model_options,
                "is_fsdp": True
            }
        if self.parallel_dict["is_fsdp"] is True:
            self._load_unet_fsdp(unet_path, model_options)
            # FSDP tracking
            self.reload_params = {
                "unet_path": unet_path,
                "model_options": model_options,
                "is_fsdp": True
            }
        else:
            self._load_model_generic(unet_path, model_options)

        # Store cast dtype for overwrite scenarios
        if hasattr(self.model, "model") and self.model.model is not None:
            self.overwrite_cast_dtype = getattr(self.model.model, 'manual_cast_dtype', None)
        else:
            self.overwrite_cast_dtype = None
        self.is_model_loaded = True



    def apply_model_sampling(self, model_sampling_patch):
         self.model.add_object_patch("model_sampling", model_sampling_patch)


    
    def set_base_ref(self, ref):
        self.base_sd_ref = ref



    def create_patched_ref(self):
        if not hasattr(self, 'base_sd_ref') or self.base_sd_ref is None:
            raise RuntimeError("Base Ref not set!")
        
        return self.base_sd_ref


    def _inject_gguf_ops(self, model_options, state_dict):
        print(f"[RayWorker {self.local_rank}] Detected GGUF Reload. Injecting GGMLOps...")
        try:
            from raylight.expansion.comfyui_gguf.ops import GGMLOps
            # Ensure model_options has custom_operations
            if "custom_operations" not in model_options:
                 model_options["custom_operations"] = GGMLOps()
            
            # Check if we have metadata to pass
            if hasattr(self, "gguf_metadata"):
                 import comfy.sd
                 # Check if load_diffusion_model_state_dict accepts metadata argument
                 sig = inspect.signature(comfy.sd.load_diffusion_model_state_dict)
                 if "metadata" in sig.parameters:
                      print("[RayWorker] Passing GGUF metadata to loader...")
                      self._temp_loader_kwargs = {"metadata": self.gguf_metadata}
        except ImportError:
            print("[RayWorker] Warning: GGUF detected but failed to import GGMLOps!")

        # Manual Prefix Stripping: GGUF reloads often require "clean" keys (no prefix)
        keys = list(state_dict.keys())
        print(f"[RayWorker Debug] Original State Dict Keys [0-5]: {keys[:5]}")
        
        # The auto-detection (unet_prefix_from_state_dict) failed in previous runs (detected 'model.'), so we force 'diffusion_model.'
        if any(k.startswith("diffusion_model.") for k in keys[:5]):
             print("[RayWorker] Detected 'diffusion_model.' prefix. Force-stripping for detection...")
             import comfy.utils
             state_dict = comfy.utils.state_dict_prefix_replace(state_dict, {"diffusion_model.": ""}, filter_keys=True)
             print(f"[RayWorker Debug] Stripped State Dict Keys [0-5]: {list(state_dict.keys())[:5]}")
        else:
             print("[RayWorker Debug] No 'diffusion_model.' prefix detected.")
        
        # Print metadata for debug
        if hasattr(self, "gguf_metadata"):
             print(f"[RayWorker Debug] GGUF Metadata Present: {bool(self.gguf_metadata)}")
             if self.gguf_metadata:
                 print(f"[RayWorker Debug] GGUF Metadata Keys: {list(self.gguf_metadata.keys())}")
                 # Print specific architecture keys if present
                 for k in ["general.architecture", "general.type", "arch_str"]:
                     if k in self.gguf_metadata:
                         print(f"[RayWorker Debug] GGUF Metadata '{k}': {self.gguf_metadata[k]}")
        else:
             print("[RayWorker Debug] No GGUF Metadata found on object.")

        print(f"[RayWorker Debug] Model Options Keys: {list(model_options.keys())}")
        if "custom_operations" in model_options:
             print(f"[RayWorker Debug] Custom Operations injected: {model_options['custom_operations']}")

        return state_dict

    def _enforce_zero_copy(self, state_dict):
        # ENFORCE Memory Sharing (Fix for OOM)
        # Even if load_state_dict copied data (e.g. due to casting), we force it back to shared tensor.
        try:
             count = 0
             cast_count = 0
             # Optimized prefix search
             prefixes = ["", "model.diffusion_model.", "diffusion_model."]
             
             # Iterate directly - no dict() materialization (saves ~50KB + avoids temp allocations)
             for name, param in self.model.model.named_parameters():
                 key_found = None
                 # Quick lookup (exact match)
                 if name in state_dict:
                     key_found = name
                 else:
                     # Prefix fallback (Additive)
                     for p in prefixes:
                         if (p + name) in state_dict:
                             key_found = p + name
                             break
                 
                 # NEW: Prefix Stripping Fallback (Subtractive)
                 # If state_dict has stripped keys (e.g. "foo") but param is "diffusion_model.foo"
                 if not key_found:
                     for p in ["diffusion_model.", "model.diffusion_model."]:
                         if name.startswith(p):
                             stripped_name = name[len(p):]
                             if stripped_name in state_dict:
                                 key_found = stripped_name
                                 break
                 
                 if key_found:
                     shared_tensor = state_dict[key_found]
                     # Check if data pointers differ (implies copy exists)
                     if param.data_ptr() != shared_tensor.data_ptr():
                         # FORCE SHARE: Replace the data tensor with the shared one
                         if param.dtype == shared_tensor.dtype and param.shape == shared_tensor.shape:
                             param.data = shared_tensor
                             count += 1
                         else:
                             # If shapes match but dtypes differ, we might be safer to keep the copy 
                             # OR force cast. Casting creates a copy usually, but if we assign 
                             # the shared_tensor directly, we change the model's precision.
                             # For inference, changing to the stored precision (usually FP16) is desirable.
                             if param.shape == shared_tensor.shape:
                                  param.data = shared_tensor
                                  cast_count += 1

             if count > 0 or cast_count > 0:
                 print(f"[Raylight] Memory Sharing ENFORCED: {count} exact + {cast_count} casted parameters linked back to Shared Object Store.")
                 # Trigger GC to free the temporary copies we just orphaned
                 cleanup_memory()
             else:
                 print("[Raylight] Zero-Copy check passed (or no matching keys found).")

        except Exception as e:
            print(f"[Raylight] Memory Optimization Error: {e}")
            import traceback
            traceback.print_exc()

    def load_unet_from_state_dict(self, state_dict, model_options):
        """Loads the UNET from a provided state dict (passed via shared memory)."""
        import ray
        if isinstance(state_dict, ray.ObjectRef):
            # Smart Reload: Check Ref Match
            if self.model is not None and getattr(self, "current_sd_ref", None) == state_dict:
                 print(f"[RayWorker {self.local_rank}] Smart Reload: State Dict Ref match. Skipping.")
                 self.is_model_loaded = True
                 return

            self.current_sd_ref = state_dict

            print(f"[RayWorker {self.local_rank}] Resolving base_sd_ref from Shared Object Store...")
            state_dict = ray.get(state_dict)

            # Check for Parallel Load Mode (Lightweight Ref)
            if isinstance(state_dict, dict) and state_dict.get("parallel_load_mode", False):
                 print(f"[RayWorker {self.local_rank}] Optimization within Load: Parallel Disk Loading detected...")
                 unet_path = state_dict.get("unet_path")
                 
                 # Mmap Optimization: Strip dtype
                 load_options = state_dict.get("model_options", {}).copy()
                 cast_dtype = load_options.pop("dtype", None)
                 
                 import comfy.sd
                 self.model = comfy.sd.load_diffusion_model(
                     unet_path, model_options=load_options
                 )
                 
                 if cast_dtype:
                      print(f"[RayWorker] Setting manual_cast_dtype to {cast_dtype}")
                      self.model.model.manual_cast_dtype = cast_dtype
                 else:
                      self.overwrite_cast_dtype = self.model.model.manual_cast_dtype
                      
                 self.is_model_loaded = True
                 return

            # Note: GGUF models use init_gguf_from_ref instead, so we should not reach here for GGUF
            # in the current architecture.
            
            # Fallback for unexpected direct state_dict passing:
            print(f"[RayWorker {self.local_rank}] Fallback: Standard Load from State Dict...")
            loader_kwargs = {}
            if self.overwrite_cast_dtype is not None:
                loader_kwargs["manual_cast_dtype"] = self.overwrite_cast_dtype
                
            import comfy.sd
            self.model = comfy.sd.load_diffusion_model_state_dict(
                 state_dict, model_options=model_options, **loader_kwargs
            )
            
            # Since the model was init on Meta, we need to ensure it's usable.
            # load_diffusion_model_state_dict calls load_state_dict(assign=True).
            # If successful, the tensors are now Plasma Views (CPU).
            # We must verify they are not Meta anymore.
            if next(self.model.model.parameters()).device.type == 'meta':
                 print(f"[RayWorker {self.local_rank}] Warning: Model parameters are still on Meta device after load. Moving to CPU...")
                 self.model.model.to("cpu")

        # GGUF Support: Inject Custom Ops if needed
        if getattr(self, "is_gguf", False):
            state_dict = self._inject_gguf_ops(model_options, state_dict)

        loader_kwargs = getattr(self, "_temp_loader_kwargs", {})
        self._temp_loader_kwargs = None # Clear
        
        import comfy.sd
        self.model = comfy.sd.load_diffusion_model_state_dict(
            state_dict, model_options=model_options, **loader_kwargs
        )


        # GGUF Patcher Upgrade
        if getattr(self, "is_gguf", False):
            print(f"[RayWorker {self.local_rank}] Upgrading to GGUFModelPatcher...")
            try:
                from raylight.expansion.comfyui_gguf.nodes import GGUFModelPatcher
                self.model = GGUFModelPatcher.clone(self.model)
                if hasattr(self, "gguf_metadata"):
                     self.model.gguf_metadata = self.gguf_metadata
            except ImportError:
                 print("[RayWorker] Warning: Failed to import GGUFModelPatcher for reload.")
        
        if self.model is None:
             print(f"[RayWorker {self.local_rank}] ERROR: load_diffusion_model_state_dict returned None.")
             if hasattr(state_dict, "keys"):
                 keys = list(state_dict.keys())
                 print(f"[RayWorker Debug] State Dict Keys Sample: {keys[:5]}")
                 import comfy.model_detection
                 print("[RayWorker Debug] Testing detection manually...")
                 conf = comfy.model_detection.model_config_from_unet(state_dict, "", metadata=getattr(self, "gguf_metadata", None))
                 print(f"[RayWorker Debug] Manual Detection Result: {conf}")
             raise RuntimeError("Failed to load model from state dict")

        self._enforce_zero_copy(state_dict)

        if self.model is None:
             raise RuntimeError("Failed to load model from state dict")
        
        if self.model is None:
             raise RuntimeError("Failed to load model from state dict")

        self.overwrite_cast_dtype = self.model.model.manual_cast_dtype
        self.is_model_loaded = True

        # Explicitly free memory
        # Explicitly free memory
        cleanup_memory()


    def load_gguf_unet(self, unet_path, dequant_dtype, patch_dtype):
        with monitor_memory(f"RayWorker {self.local_rank} - load_gguf_unet", device=self.device):
            # Smart Reload: Skip if already loaded (using new reload_params)
            if self.model is not None and getattr(self, "reload_params", {}).get("unet_path") == unet_path:
                 print(f"[RayWorker] Smart Reload: GGUF Model {unet_path} already loaded. Skipping.")
                 self.is_model_loaded = True
                 return

            if self.parallel_dict["is_fsdp"] is True:
                raise ValueError("FSDP Sharding for GGUF is not supported")
            else:
                self._load_model_generic(
                    unet_path,
                    model_options={}, # GGUF generic loader handles options internally if needed or via extra args
                    dequant_dtype=dequant_dtype,
                    patch_dtype=patch_dtype
                )

            
            self.is_model_loaded = True
            return True

    def load_lora(self, lora_path, strength_model, lora_config_hash=None):
        return self.lora_manager.load_lora(lora_path, strength_model, lora_config_hash)

    def reset_loras(self):
        self.lora_manager.reset_loras()

    def reapply_loras_for_config(self, config_hash):
        return self.lora_manager.reapply_loras_for_config(config_hash)

    def _apply_lora_core(self, lora_path, strength_model):
        self.lora_manager._apply_lora_core(lora_path, strength_model)

    def barrier_and_evict(self, path):
         """Synchronizes all workers and then evicts page cache."""
         print(f"[RayWorker {self.local_rank}] Waiting at barrier for synchronized eviction...")
         if dist.is_initialized():
             dist.barrier()
         
         evict_page_cache(path)
         return True


    def get_base_ref(self):
        # Return as list to prevent Ray from auto-resolving the ObjectRef when returned
        return [getattr(self, "base_sd_ref", None)]

    def get_gguf_metadata(self):
        return getattr(self, "gguf_metadata", {})

    def init_gguf_from_ref(self, ref, metadata, reload_params=None):
        print(f"[RayWorker {self.local_rank}] Initializing GGUF from Shared Reference (Follower Mode)...")
        # Save reload params first so we can use them for fallback
        if reload_params:
             self.reload_params = reload_params
             
        self.gguf_metadata = metadata
        self.is_gguf = True
        
        # 1. OPTIMIZATION: Prefer Disk/Mmap Loading (Parallel)
        # Ray Shared Memory transfer for GGUF proved too heavy (OOM) even with deconstruction.
        # Running parallel disk loads allows the OS to use mmap sharing, which is the most memory efficient.
        if reload_params is not None:
             print(f"[RayWorker {self.local_rank}] GGUF Optimization: Using Unified Parallel Disk Loading (Mmap)...")
             self._load_model_generic(
                 reload_params["unet_path"],
                 reload_params["model_options"],
                 reload_params.get("dequant_dtype"),
                 reload_params.get("patch_dtype")
             )
             
             self.is_model_loaded = True
             return
        else:
             raise ValueError("[RayWorker] Cannot RELOAD GGUF: Missing reload params for parallel loading.")

    def reload_model_if_needed(self, load_device=None):
        """Auto-reloads the model if it was offloaded.
        
        Uses ModelContext for unified reload across all formats:
        - GGUF Soft-Reload: hot_load() via mmap cache
        - Standard Reload: full load() from cache or disk
        """
        from raylight.distributed_worker.model_context import get_context, ModelState
        
        target_device = load_device if load_device is not None else self.device
        
        # Quick check: already on target device?
        if self.model is not None:
            curr = getattr(self.model, "current_device", None)
            if str(curr) == str(target_device):
                print(f"[RayWorker] Idempotency: Model already on {target_device}. Skipping reload.")
                return
        
        # Check if reload is needed
        needs_reload = self.model is None
        if self.model is not None and hasattr(self.model, 'current_device'):
            if str(self.model.current_device) != str(target_device):
                needs_reload = True
        
        if not needs_reload:
            return
        
        print("[RayWorker] Model offloaded or device mismatch. Auto-reloading...")
        
        # Delegate to context
        if hasattr(self, 'reload_params') and self.reload_params:
            unet_path = self.reload_params.get("unet_path", "")
            ctx = get_context(self, unet_path)
            ctx.reload_if_needed(target_device)
        elif hasattr(self, 'base_sd_ref') and self.base_sd_ref is not None:
            print("[RayWorker] Reloading from Shared base_sd_ref...")
            self.load_unet_from_state_dict(self.base_sd_ref, model_options={})
        else:
            raise RuntimeError("Model is offloaded but no reload source available!")

    def load_bnb_unet(self, unet_path):
        if self.parallel_dict["is_fsdp"] is True:
            self._apply_fsdp_patches()
            from raylight.comfy_dist.sd import fsdp_bnb_load_diffusion_model
            from torch.distributed.fsdp import FSDPModule
            
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

    def kill(self):
        self.model = None
        dist.destroy_process_group()
        ray.actor.exit_actor()

    def ray_vae_loader(self, vae_path):
        self.vae_manager.load_vae(vae_path)
        self.vae_model = self.vae_manager.vae_model

    def ray_vae_release(self):
        result = self.vae_manager.release_vae()
        self.vae_model = None
        return result

    def get_vae_temporal_compression(self):
        return self.vae_manager.get_temporal_compression()

    def get_vae_spatial_compression(self):
        return self.vae_manager.get_spatial_compression()

    def _check_vae_health(self, samples, shard_index):
        self.vae_manager.check_health(samples, shard_index)

    def ray_vae_decode(
        self,
        shard_index,
        samples,
        tile_size,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
        discard_latent_frames=0,
        vae_dtype="auto",
        mmap_path=None,
        mmap_shape=None,
        output_offset=0,
    ):
        return self.vae_manager.decode(
            shard_index,
            samples,
            tile_size,
            overlap,
            temporal_size,
            temporal_overlap,
            discard_latent_frames,
            vae_dtype,
            mmap_path,
            mmap_shape,
            output_offset,
        )


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
        return self.sampler_manager.custom_sampler(
            add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image
        )

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
        return self.sampler_manager.common_ksampler(
            seed, steps, cfg, sampler_name, scheduler, positive, negative,
            latent, denoise, disable_noise, start_step, last_step,
            force_full_denoise, sigmas
        )


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
        
        # Parallel initialization check (faster startup)
        # Instead of waiting sequentially, we fire all check tasks and wait for them in parallel
        ready_refs = [actor.__ray_ready__.remote() for actor in ray_actors["workers"]]
        cancellable_get(ready_refs)
        
        return ray_actors

    return _init_ray_actor


# (TODO-Komikndr) Should be removed since FSDP can be unloaded properly
def ensure_fresh_actors(ray_actors_init):
    ray_actors, ray_actor_fn = ray_actors_init
    gpu_actors = ray_actors["workers"]

    needs_restart = False
    try:
        is_loaded = cancellable_get(gpu_actors[0].get_is_model_loaded.remote())
        if is_loaded:
            needs_restart = True
    except RayActorError:
        # Actor already dead or crashed
        needs_restart = True

    needs_restart = False
    if needs_restart:
        for actor in gpu_actors:
            try:
                cancellable_get(actor.kill.remote())
            except Exception:
                pass  # ignore already dead
        ray_actors = ray_actor_fn()
        gpu_actors = ray_actors["workers"]

    parallel_dict = cancellable_get(gpu_actors[0].get_parallel_dict.remote())

    return ray_actors, gpu_actors, parallel_dict
