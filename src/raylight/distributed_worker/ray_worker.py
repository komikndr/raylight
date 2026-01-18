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

from raylight.comfy_dist.sd import load_lora_for_models as ray_load_lora_for_models
from raylight.distributed_worker.utils import Noise_EmptyNoise, Noise_RandomNoise, patch_ray_tqdm
from raylight.utils_memory import monitor_memory
from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops
from raylight.comfy_dist.utils import cancellable_get
from ray.exceptions import RayActorError
from contextlib import contextmanager
import inspect


# Developer reminder, Checking model parameter outside ray actor is very expensive (e.g Comfy main thread)
# the model need to be serialized, send to object store and can cause OOM !, so setter and getter is the pattern !


# If ray actor function being called from outside, ray.get([task in actor task]) will become sync between rank
# If called from ray actor within. dist.barrier() become the sync.


# Comfy cli args, does not get pass through into ray actor
def evict_page_cache(path):
    """Tells the kernel to evict pages of the given file from the Page Cache."""
    if not path or not os.path.exists(path):
        return
    
    try:
        if sys.platform.startswith("linux"):
            fd = os.open(path, os.O_RDONLY)
            file_size = os.path.getsize(path)
            # POSIX_FADV_DONTNEED = 4 - tells kernel to evict pages from cache
            os.posix_fadvise(fd, 0, file_size, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            print(f"[Raylight] SUCCESS: Evicted {os.path.basename(path)} from page cache ({file_size / 1024**3:.2f} GB)")
    except (OSError, AttributeError) as e:
        # posix_fadvise not available on all platforms
        print(f"[Raylight] Note: Could not evict {os.path.basename(path)} from page cache: {e}")

class RayWorker:



    def __init__(self, local_rank, device_id, parallel_dict):
        self.model = None
        self.vae_model = None
        self.model_type = None
        self.state_dict = None
        self.parallel_dict = parallel_dict
        self.overwrite_cast_dtype = None
        
        self.lora_cache = {} # Cache for Runtime LoRAs

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

    def _handle_gguf_cache(self):
        """Create /dev/shm cache of CUDA weights before nuking VRAM.
        
        STREAMING VERSION: Writes tensors one at a time to avoid 15GB RAM spike.
        """
        import hashlib
        import ray
        import torch
        import pickle

        # Unique ID per Node + Model (prevents cross-node /dev/shm collisions)
        node_id = ray.get_runtime_context().get_node_id()[:8]
        model_hash = hashlib.md5(self.current_unet_path.encode()).hexdigest()[:8] if self.current_unet_path else "unknown"
        cache_path = f"/dev/shm/raylight_{node_id}_{model_hash}.pt"

        if self.local_rank == 0 and not os.path.exists(cache_path):
            print(f"[RayWorker 0] Streaming VRAM state to SHM (true streaming mode)...")
            
            # TRUE STREAMING: Write tensors one at a time using pickle protocol
            # This avoids building a 15GB state_dict in RAM before saving
            param_count = 0
            buf_count = 0
            bytes_written = 0
            
            # Open file for incremental pickle writes  
            import pickle
            import io
            
            # We'll build a streaming dict by writing key-value pairs one at a time
            # Using a temporary approach: collect keys/shapes first, then stream values
            tensor_index = {}  # name -> (offset, dtype, shape) for verification
            
            # Use buffered I/O and protocol 5 for speed (protocol 5 is optimized for large arrays)
            with open(cache_path, 'wb', buffering=8*1024*1024) as f:  # 8MB buffer
                pickler = pickle.Pickler(f, protocol=5)
                
                # Write header placeholder (will be dict with metadata)
                header_pos = f.tell()
                pickler.dump({"version": 2, "streaming": True, "count": 0})
                data_start = f.tell()
                
                # Stream parameters one at a time (NO gc.collect during loop - major speedup)
                for name, param in self.model.model.named_parameters():
                    if param.device.type == "cuda":
                        # Copy single tensor to CPU and write immediately
                        cpu_tensor = param.detach().cpu()
                        tensor_index[name] = (f.tell() - data_start, str(cpu_tensor.dtype), tuple(cpu_tensor.shape))
                        pickler.dump((name, cpu_tensor))
                        param_count += 1
                        del cpu_tensor
                
                # Stream buffers one at a time  
                for name, buf in self.model.model.named_buffers():
                    if buf.device.type == "cuda":
                        cpu_tensor = buf.detach().cpu()
                        key = f"buf_{name}"
                        tensor_index[key] = (f.tell() - data_start, str(cpu_tensor.dtype), tuple(cpu_tensor.shape))
                        pickler.dump((key, cpu_tensor))
                        buf_count += 1
                        del cpu_tensor
                
                # Write end marker
                pickler.dump(None)
                bytes_written = f.tell()
            
            cache_size_mb = bytes_written / 1024**2
            print(f"[RayWorker 0] Cache created: {cache_path} ({cache_size_mb:.1f} MB)")
            print(f"[RayWorker 0] Cache contains: {param_count} params + {buf_count} buffers (streamed)")
            
            # Cleanup only at end - much faster than periodic GC
            gc.collect()
            torch.cuda.empty_cache()
            
            # Force libc to release freed memory back to OS
            try:
                libc = ctypes.CDLL('libc.so.6')
                libc.malloc_trim(0)
            except Exception:
                pass
            
            # Evict original GGUF file from page cache - we'll use /dev/shm cache instead
            if self.current_unet_path:
                evict_page_cache(self.current_unet_path)
        elif self.local_rank != 0:
            print(f"[RayWorker {self.local_rank}] Waiting for cache from Worker 0...")
        else:
            print(f"[RayWorker 0] Cache already exists: {cache_path}")
        
        # CRITICAL: Barrier to ensure Worker 0 finishes cache before others proceed
        # Workers 1-3 must wait for cache file to be fully written
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
            print(f"[RayWorker {self.local_rank}] Cache sync barrier passed")
        
        # All workers need the cache path for restore
        self.model.cache_path = cache_path

    def _offload_gguf_soft(self):
        print(f"[RayWorker {self.local_rank}] GGUF Soft-Offload: Releasing VRAM but retaining System RAM mmap...")
        
        # === CREATE /dev/shm CACHE BEFORE UNPATCH (weights still on GPU!) ===
        self._handle_gguf_cache()
        
        # NOW unpatch to remove VRAM copies
        if hasattr(self.model, "unpatch_model"):
            self.model.unpatch_model(device_to=self.model.offload_device, unpatch_weights=False)
            self.model.current_device = torch.device("cpu")
        
        # FAILSAFE: Manually strip ALL GPU tensors from the model to free VRAM
        try:
            print(f"[RayWorker {self.local_rank}] GGUF Soft-Offload: Failsafe GPU strip...")
            nuked = 0
            for name, param in self.model.model.named_parameters():
                if param.device.type == "cuda":
                    param.data = torch.empty(0, dtype=param.dtype, device="cpu")
                    nuked += 1
            for name, buf in self.model.model.named_buffers():
                if buf.device.type == "cuda":
                    buf.data = torch.empty(0, dtype=buf.dtype, device="cpu")
                    nuked += 1
            print(f"[RayWorker {self.local_rank}] Nuked {nuked} CUDA tensors")
        except Exception as e:
            print(f"[RayWorker {self.local_rank}] Warning during GPU strip: {e}")

        # Force VRAM cleanup
        import gc
        gc.collect()
        gc.collect() 
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        print(f"[RayWorker {self.local_rank}] GGUF Soft-Offload complete. Model retained with nuked weights.")

    def _offload_standard(self):
        print(f"[RayWorker {self.local_rank}] Offloading model (releasing VRAM)...")
        model_to_remove = self.model
        self.model = None
        
        # Clear tracking vars to force reload next time
        self.current_unet_path = None
        self.current_sd_ref = None
        
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
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
        
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        print(f"[RayWorker {self.local_rank}] Model offloaded successfully.")

    def offload_and_clear(self):
        """
        Offloads the model from VRAM and clears tracking state.
        """
        if self.model is not None:
            if getattr(self, "is_gguf", False):
                self._offload_gguf_soft()
            else:
                self._offload_standard()
        else:
            print(f"[RayWorker {self.local_rank}] No model to offload.")


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

        import ray
        if self.state_dict is not None:
            print("[RayWorker] FSDP Mode: Saving base_sd_ref to Object Store...")
            self.base_sd_ref = ray.put(self.state_dict)

    def _load_unet_standard(self, unet_path, model_options):
        # Parallel Loading Optimization & Dtype Preservation
        # 1. Strip 'dtype' to force Native Mmap (1x RAM) instead of Private Cast (4x RAM)
        load_options = model_options.copy()
        cast_dtype = load_options.pop("dtype", None)
        
        if cast_dtype:
                print(f"[RayWorker] Mmap Preservation: Loading Native (skipping cast to {cast_dtype})")
        
        self.model = comfy.sd.load_diffusion_model(
            unet_path, model_options=load_options,
        )
        
        # Apply Cast for Execution (on-the-fly / GPU)
        if cast_dtype:
                print(f"[RayWorker] Setting manual_cast_dtype to {cast_dtype}")
                self.model.model.manual_cast_dtype = cast_dtype
        else:
                self.overwrite_cast_dtype = self.model.model.manual_cast_dtype

        import ray
        print("[RayWorker] Standard Mode: Storing Lightweight Reference for Parallel Load...")
        # Store metadata instead of full model to save Leader/Ray RAM
        self.base_sd_ref = ray.put({
            "parallel_load_mode": True,
            "unet_path": unet_path,
            "model_options": model_options # Keep original options (with dtype) for reference
        })
        self.is_gguf = False

    def load_unet(self, unet_path, model_options):
        with monitor_memory(f"RayWorker {self.local_rank} - load_unet", device=self.device):
            # Smart Reload: Skip if already loaded
            if self.model is not None and getattr(self, "current_unet_path", None) == unet_path:
                    print(f"[RayWorker] Smart Reload: Model {unet_path} already loaded. Skipping.")
                    self.is_model_loaded = True
                    return

            self.current_unet_path = unet_path
        
        if self.parallel_dict["is_fsdp"] is True:
            self._load_unet_fsdp(unet_path, model_options)
        else:
            self._load_unet_standard(unet_path, model_options)

        if self.lora_list is not None:
            self.load_lora()

        self.overwrite_cast_dtype = self.model.model.manual_cast_dtype
        self.is_model_loaded = True



    def apply_model_sampling(self, model_sampling_patch):
         self.model.add_object_patch("model_sampling", model_sampling_patch)


    
    def set_base_ref(self, ref):
        self.base_sd_ref = ref

    def get_base_ref(self):
        return getattr(self, "base_sd_ref", None)

    def create_patched_ref(self, lora_list):
        import copy
        if not hasattr(self, 'base_sd_ref') or self.base_sd_ref is None:
            raise RuntimeError("Base Ref not set!")
        
        if not lora_list:
            return self.base_sd_ref
            
        print("[RayWorker] Creating Isolated Model Copy for Patching...")
        # 1. Ensure we have the base model loaded (should be zero-copy)
        if hasattr(self, 'gguf_reload_params') and self.gguf_reload_params is not None:
             print("[RayWorker] LoRA Patching: Reloading GGUF Base from Disk...")
             self.load_gguf_unet(**self.gguf_reload_params)
        else:
             self.load_unet_from_state_dict(self.base_sd_ref, model_options={})
        
        # 2. Setup Scratchpad Model (Deep Copy to isolate from Shared Base)
        # We perform a Deep Copy of the underlying DiffusionModel.
        # This allocates ~15GB Private RAM (temporary).
        # We do this to avoid modifying the Shared Memory pages in-place, which would
        # trigger Copy-on-Write and permanently bloat Worker 0's memory usage with dirty pages.
        scratch_unet = copy.deepcopy(self.model.model)
        
        # Create a new Patcher wrapping this private model
        work_model = self.model.clone()
        work_model.model = scratch_unet
        
        # 3. Apply LoRAs to Scratchpad
        for lora in lora_list:
             lora_path = lora["path"]
             strength_model = lora["strength_model"]
             strength_clip = 1.0 
             try:
                 lora_model = comfy.utils.load_torch_file(lora_path, safe_load=True)
             except Exception as e:
                 print(f"[Raylight] Failed to load LoRA {lora_path}: {e}")
                 continue
             
             work_model, _ = comfy.sd.load_lora_for_models(
                 work_model, None, lora_model, strength_model, strength_clip
             )
             del lora_model
             gc.collect()

        # 4. Flatten to State Dict
        print("[RayWorker] Compiling Patched State Dict (Target: Object Store)...")
        # patch_model(cpu) modifies scratch_unet in-place.
        work_model.patch_model(device_to=torch.device("cpu"))
        
        patched_sd = work_model.model.state_dict()
        
        # 5. Cleanup
        # We don't need to unpatch because we're discarding scratch_unet.
        del work_model
        del scratch_unet
        gc.collect()
        
        return patched_sd

    def _inject_gguf_ops(self, model_options, state_dict):
        print("[RayWorker] Detected GGUF Reload. Injecting GGMLOps...")
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
                 gc.collect()
                 torch.cuda.empty_cache()
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
                 print("[RayWorker] Smart Reload: State Dict Ref match. Skipping.")
                 self.is_model_loaded = True
                 return

            self.current_sd_ref = state_dict

            print("[RayWorker] Resolving base_sd_ref from Shared Object Store...")
            state_dict = ray.get(state_dict)

            # Check for Parallel Load Mode (Lightweight Ref)
            if isinstance(state_dict, dict) and state_dict.get("parallel_load_mode", False):
                 print("[RayWorker] Optimization within Load: Parallel Disk Loading detected...")
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
            print("[RayWorker] Fallback: Standard Load from State Dict...")
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
                 print("[RayWorker] Warning: Model parameters are still on Meta device after load. Moving to CPU...")
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
            print("[RayWorker] Upgrading to GGUFModelPatcher...")
            try:
                from raylight.expansion.comfyui_gguf.nodes import GGUFModelPatcher
                self.model = GGUFModelPatcher.clone(self.model)
                if hasattr(self, "gguf_metadata"):
                     self.model.gguf_metadata = self.gguf_metadata
            except ImportError:
                 print("[RayWorker] Warning: Failed to import GGUFModelPatcher for reload.")
        
        if self.model is None:
             print("[RayWorker] ERROR: load_diffusion_model_state_dict returned None.")
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

        if self.lora_list is not None:
            self.load_lora()

        self.overwrite_cast_dtype = self.model.model.manual_cast_dtype
        self.is_model_loaded = True

        # Explicitly free memory
        import gc
        gc.collect()
        comfy.model_management.soft_empty_cache()


    def load_gguf_unet(self, unet_path, dequant_dtype, patch_dtype):
        with monitor_memory(f"RayWorker {self.local_rank} - load_gguf_unet", device=self.device):
            # Smart Reload: Skip if already loaded
            if self.model is not None and getattr(self, "current_unet_path", None) == unet_path:
                 print(f"[RayWorker] Smart Reload: GGUF Model {unet_path} already loaded. Skipping.")
                 self.is_model_loaded = True
                 return

            self.current_unet_path = unet_path

            if self.parallel_dict["is_fsdp"] is True:
                raise ValueError("FSDP Sharding for GGUF is not supported")
            else:
                from raylight.comfy_dist.sd import gguf_load_diffusion_model
                self.model = gguf_load_diffusion_model(
                    unet_path,
                    dequant_dtype=dequant_dtype,
                    patch_dtype=patch_dtype
                )
                import ray
                print("[RayWorker] GGUF Mode: Storing Lightweight Reference to Object Store...")
                # Optimization: We do NOT store heavier GGUF tensors in Ray Object Store as it causes OOM.
                # We rely on parallel mmap loading (disk) which the OS handles efficiently.
                # We store a dummy object to satisfy "is stored" checks.
                self.base_sd_ref = ray.put({"gguf_mmap_mode": True})
                
                self.is_gguf = True
                self.gguf_metadata = getattr(self.model, "gguf_metadata", {})
                self.gguf_reload_params = {
                    "unet_path": unet_path,
                    "dequant_dtype": dequant_dtype,
                    "patch_dtype": patch_dtype
                }
            
            if self.lora_list is not None:
                 self.load_lora()
                 
            self.is_model_loaded = True

            # NEW: FORCED immediate VRAM load and Page Cache eviction
            # We bypass reload_model_if_needed() checks to ensure GGUFModelPatcher.load()
            # executes, clears its backup, and calls evict_page_cache().
            print(f"[RayWorker {self.local_rank}] Loader: FORCING immediate VRAM load and eviction...")
            if hasattr(self.model, "load"):
                self.model.load(self.device, force_patch_weights=True)
                self.model.current_device = self.device
            
            # CRITICAL: Clear legacy backup dict to release mmap tensor references.
            # Raylight uses /dev/shm caching for reload, not the backup dict.
            if hasattr(self.model, "backup"):
                self.model.backup.clear()
            
            # Aggressive GC
            import gc
            for _ in range(3):
                gc.collect()
            comfy.model_management.soft_empty_cache()


    def get_base_ref(self):
        # Return as list to prevent Ray from auto-resolving the ObjectRef when returned
        return [getattr(self, "base_sd_ref", None)]

    def get_gguf_metadata(self):
        return getattr(self, "gguf_metadata", {})

    def init_gguf_from_ref(self, ref, metadata, reload_params=None):
        print("[RayWorker] Initializing GGUF from Shared Reference (Follower Mode)...")
        # Save reload params first so we can use them for fallback
        self.gguf_reload_params = reload_params
        self.gguf_metadata = metadata
        self.is_gguf = True
        
        # 1. OPTIMIZATION: Prefer Disk/Mmap Loading (Parallel)
        # Ray Shared Memory transfer for GGUF proved too heavy (OOM) even with deconstruction.
        # Running parallel disk loads allows the OS to use mmap sharing, which is the most memory efficient.
        if reload_params is not None:
             print("[RayWorker] GGUF Optimization: Using Parallel Disk Loading (Mmap)...")
             self.load_gguf_unet(**reload_params)
             
             if self.lora_list is not None:
                self.load_lora()
             self.is_model_loaded = True
             return
        else:
             raise ValueError("[RayWorker] Cannot RELOAD GGUF: Missing reload params for parallel loading.")

    def reload_model_if_needed(self):
        """Auto-reloads the model if it was offloaded.
        
        Supports two reload modes:
        - GGUF Soft-Reload: model exists but current_device=cpu (nuked weights) → uses /dev/shm cache
        - Standard Reload: model is None → uses disk or shared memory
        """
        # Check if model needs reload:
        # - For standard offload: self.model is None
        # - For GGUF soft-offload: self.model exists but current_device is CPU (nuked weights)
        needs_reload = self.model is None
        if self.model is not None and hasattr(self.model, 'current_device'):
            if self.model.current_device != self.model.load_device:
                needs_reload = True
        
        if not needs_reload:
            return  # Model is loaded, nothing to do
        
        print("[RayWorker] Model offloaded. Auto-reloading...")
        
        # GGUF SOFT-RELOAD: Use /dev/shm cache via model.load()
        if self.model is not None and getattr(self, "is_gguf", False):
            print("[RayWorker] GGUF Soft-Reload: Triggering Self-Heal via model.load()...")
            self.model.load(self.device, force_patch_weights=True)
            self.model.current_device = self.model.load_device
            
            # Clear legacy backup dict to release mmap tensor references
            if hasattr(self.model, "backup"):
                self.model.backup.clear()
            
            print("[RayWorker] GGUF Soft-Reload: Complete.")
            return
        
        # Standard reload for model=None cases
        if hasattr(self, 'gguf_reload_params') and self.gguf_reload_params is not None:
             print("[RayWorker] Reloading GGUF from disk parameters (mmap)...")
             self.load_gguf_unet(**self.gguf_reload_params)
             return

        if hasattr(self, 'base_sd_ref') and self.base_sd_ref is not None:
             print("[RayWorker] Reloading from Shared base_sd_ref...")
             self.load_unet_from_state_dict(self.base_sd_ref, model_options={})
             return
             
        raise RuntimeError("Model is offloaded but no reload source (base_sd_ref or gguf_reload_params) is available!")

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

        if self.lora_list is not None:
            self.load_lora()

        self.is_model_loaded = True

    def set_lora_list(self, lora):
        self.lora_list = lora

    def get_lora_list(self,):
        return self.lora_list

    def _apply_runtime_loras(self, model, lora_list):
        """Helper to apply runtime LoRAs to a model clone."""
        if not lora_list:
            return model
            
        work_model = model.clone()
        for lora in lora_list:
            lora_path = lora["path"]
            strength_model = lora["strength_model"]
            strength_clip = 1.0 
            try:
                lora_model = comfy.utils.load_torch_file(lora_path, safe_load=True)
            except Exception as e:
                print(f"[Raylight] Failed to load LoRA {lora_path}: {e}")
                continue
                
            work_model, _ = comfy.sd.load_lora_for_models(
                work_model, None, lora_model, strength_model, strength_clip
            )
            del lora_model # Release raw dict immediately
        
        gc.collect()
        return work_model



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
        metadata = None
        if "pixel_space" in vae_path:
            state_dict["pixel_space_vae"] = torch.tensor(1.0)
        else:
            # Use return_metadata=True to ensure correct versioning (LTX-1 vs LTX-2)
            res = comfy.utils.load_torch_file(vae_path, return_metadata=True)
            if isinstance(res, tuple):
                state_dict, metadata = res
            else:
                state_dict = res
                metadata = None

        # Diagnostics for VAE identification
        key_check = "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"
        if key_check in state_dict:
            channels = state_dict[key_check].shape[0]
            print(f"[RayWorker {self.local_rank}] VAE Identification: {key_check} has {channels} channels.")
            # Replicate ComfyUI logic for clarity in logs
            version = 0
            if channels == 512: version = 0
            elif channels == 1024:
                version = 1
                if "encoder.down_blocks.1.conv.conv.bias" in state_dict: version = 2
            print(f"[RayWorker {self.local_rank}] VAE Identification: Guessed version {version}")
        
        if metadata:
            print(f"[RayWorker {self.local_rank}] VAE Identification: Metadata found, contains 'config': {'config' in metadata}")

        import logging
        original_level = logging.getLogger().getEffectiveLevel()
        # logging.getLogger().setLevel(logging.ERROR) # Let's see the warnings for now
        try:
            # Pass metadata to VAE constructor to correctly handle configs
            vae_model = comfy.sd.VAE(sd=state_dict, metadata=metadata)
            vae_model.throw_exception_if_invalid()
        finally:
            logging.getLogger().setLevel(original_level)

        vae_model.decode_tiled_1d = types.MethodType(decode_tiled_1d, vae_model)
        vae_model.decode_tiled_ = types.MethodType(decode_tiled_, vae_model)
        vae_model.decode_tiled_3d = types.MethodType(decode_tiled_3d, vae_model)

        if self.local_rank == 0:
            print(f"VAE loaded in {self.global_world_size} GPUs")
        self.vae_model = vae_model

    def get_vae_temporal_compression(self):
        if self.vae_model is None: return None
        return self.vae_model.temporal_compression_decode()

    def get_vae_spatial_compression(self):
        if self.vae_model is None: return None
        return self.vae_model.spacial_compression_decode()

    def _check_vae_health(self, samples, shard_index):
        # Diagnostic: Check input latent statistics
        l_min, l_max, l_mean = samples["samples"].min().item(), samples["samples"].max().item(), samples["samples"].mean().item()
        print(f"[RayWorker {self.local_rank}] Input Latent Stats (shard {shard_index}): min={l_min:.4f}, max={l_max:.4f}, mean={l_mean:.4f}")
        if torch.isnan(samples["samples"]).any():
            print(f"[RayWorker {self.local_rank}] CRITICAL: Input latents for shard {shard_index} contain NaNs!")
        
        # Check if overwrite_cast_dtype is set (from patch_temp_fix_ck_ops)
        if getattr(self, "overwrite_cast_dtype", None) is not None:
            print(f"[RayWorker {self.local_rank}] Debug: overwrite_cast_dtype is set to {self.overwrite_cast_dtype}")
        
        # Check VAE weights for NaNs BEFORE decoding
        if self.vae_model is not None:
            nan_params = []
            for name, p in self.vae_model.first_stage_model.named_parameters():
                if torch.isnan(p).any():
                    nan_params.append(name)
            if nan_params:
                print(f"[RayWorker {self.local_rank}] CRITICAL: VAE parameters contain NaNs: {nan_params[:5]}...")
            else:
                print(f"[RayWorker {self.local_rank}] VAE parameters verified healthy (no NaNs).")

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
    ):
        with monitor_memory(f"RayWorker {self.local_rank} - ray_vae_decode", device=self.device):
            print(f"[RayWorker {self.local_rank}] Entering ray_vae_decode (direct method call) for shard {shard_index}...")
            


            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
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

        # Determine VAE dtype based on user selection
        if vae_dtype == "auto":
            # Auto: Use bfloat16 if supported (RTX 3000+), else fall back to float32
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                print(f"[RayWorker {self.local_rank}] Using bfloat16 VAE (auto: bf16 supported)")
            else:
                dtype = torch.float32
                print(f"[RayWorker {self.local_rank}] Using float32 VAE (auto: bf16 not supported)")
        elif vae_dtype == "bf16":
            dtype = torch.bfloat16
            print(f"[RayWorker {self.local_rank}] Using bfloat16 VAE (user selected)")
        elif vae_dtype == "fp16":
            dtype = torch.float16
            print(f"[RayWorker {self.local_rank}] Using float16 VAE (user selected - may cause NaN on some models)")
        else:  # fp32
            dtype = torch.float32
            print(f"[RayWorker {self.local_rank}] Using float32 VAE (user selected - stable but 2x memory)")
        
        self.vae_model.first_stage_model.to(dtype)
        self.vae_model.vae_dtype = dtype
        
        latents_to_decode = samples["samples"].to(dtype)
        print(f"[RayWorker {self.local_rank}] latents_to_decode shape: {latents_to_decode.shape}")
        print(f"[RayWorker {self.local_rank}] tiling: tile_t={temporal_size}, overlap_t={temporal_overlap}")

        images = self.vae_model.decode_tiled(
            latents_to_decode,
            tile_x=tile_size // compression,
            tile_y=tile_size // compression,
            overlap=overlap // compression,
            tile_t=temporal_size,
            overlap_t=temporal_overlap,
        )
        print(f"[RayWorker {self.local_rank}] decode_tiled complete. Shape: {images.shape}")
        
        if torch.isnan(images).any():
            print(f"[RayWorker {self.local_rank}] CRITICAL: VAE output STILL contains NaNs even in float32!")
        
        if len(images.shape) == 5:
            # VAE.decode_tiled returns [B, T, H, W, C]
            # Reshape to [B*T, H, W, C] and squeeze leading dim if B=1
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        elif len(images.shape) == 4:
            # Already [T, H, W, C]
            pass
        else:
            # Fallback for unexpected shapes
            while len(images.shape) > 4 and images.shape[0] == 1:
                images = images.squeeze(0)
            if len(images.shape) == 3: # H, W, C -> 1, H, W, C
                images = images.unsqueeze(0)
            
        # If we have overlap, discard the warmup frames
        if discard_latent_frames > 0:
            # 1 latent frame = 8 video frames
            temporal_compression = self.vae_model.temporal_compression_decode() or 8
            discard_video_frames = discard_latent_frames * temporal_compression
            print(f"[RayWorker {self.local_rank}] Discarding {discard_video_frames} redundant warmup frames")
            images = images[discard_video_frames:]
            
        # Move to CPU explicitly before transport to avoid Ray GPU serialization issues
        # and print stats to troubleshoot "Black Video" issues.
        print(f"[RayWorker {self.local_rank}] Shard {shard_index} statistics: min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}")
        
        print(f"[RayWorker {self.local_rank}] Moving to CPU and converting to float16 for transport...")
        images = images.cpu().to(torch.float16)
        
        # Proactive memory cleanup
        print(f"[RayWorker {self.local_rank}] Shard {shard_index} complete. Cleanup and return.")
        import gc
        for _ in range(2):
            gc.collect()
        torch.cuda.empty_cache()

        # Force OS to reclaim freed memory (fixes RSS creep on Worker 0)
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception as e:
            print(f"[RayWorker {self.local_rank}] Warning: malloc_trim failed: {e}")

        # Diagnostics removed

            
        return (shard_index, images)

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
        lora_list=None,
    ):
        # NOTE: Reload is now handled by coordinator (sampler node) BEFORE dispatching.
        # This ensures all workers are ready before any collective operations begin.
             
        # Apply Runtime LoRAs locally
        work_model = self._apply_runtime_loras(self.model, lora_list)


        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(work_model, latent_image)
        latent["samples"] = latent_image

        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(latent)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        if self.parallel_dict["is_fsdp"] is True:
            work_model.patch_fsdp()
            del self.state_dict
            self.state_dict = None
            torch.cuda.synchronize()
            comfy.model_management.soft_empty_cache()
            gc.collect()

        disable_pbar = comfy.utils.PROGRESS_BAR_ENABLED
        if self.local_rank == 0:
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # CRITICAL: Ensure model device references are valid for this worker
        # This guards against stale device references from previous contexts
        if hasattr(work_model, 'load_device') and work_model.load_device != self.device:
            work_model.load_device = self.device

        with torch.no_grad():
            samples = comfy.sample.sample_custom(
                work_model,
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

            # Legacy debug_memory_leaks call removed

        
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
        lora_list=None,
    ):
        with monitor_memory(f"RayWorker {self.local_rank} - common_ksampler", device=self.device):
            # NOTE: Reload is now handled by coordinator (sampler node) BEFORE dispatching.
            # This ensures all workers are ready before any collective operations begin.

            # Apply Runtime LoRAs locally (Fallback for FSDP/No-Base-Ref)
            work_model = self._apply_runtime_loras(self.model, lora_list)


            latent_image = latent["samples"]
            latent_image = comfy.sample.fix_empty_latent_channels(work_model, latent_image)

            if self.parallel_dict["is_fsdp"] is True:
                work_model.patch_fsdp()

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

            # CRITICAL: Ensure model device references are valid for this worker
            # This guards against stale device references from previous contexts
            if hasattr(work_model, 'load_device') and work_model.load_device != self.device:
                work_model.load_device = self.device

            with torch.no_grad():
                if sigmas is None:
                    samples = comfy.sample.sample(
                        work_model,
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
                        work_model,
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

        # Legacy debug_memory_leaks call removed

        #comfy.model_management.soft_empty_cache()
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
            cancellable_get(actor.__ray_ready__.remote())
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
