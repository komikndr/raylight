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
from contextlib import contextmanager
import inspect


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

    def debug_memory_leaks(self):
        import gc
        import sys
        print("[RayWorker] --- Starting Memory Leak Debug ---")
        
        # 1. Check for ModelPatcher instances
        patchers = [obj for obj in gc.get_objects() if hasattr(obj, '__class__') and 'ModelPatcher' in obj.__class__.__name__]
        if patchers:
            print(f"[RayWorker] WARNING: Found {len(patchers)} ModelPatcher instances alive:")
            for p in patchers:
                print(f"  - {p} (Refcount: {sys.getrefcount(p)})")
                print(f"    Referrers: {gc.get_referrers(p)}")
        else:
            print("[RayWorker] No ModelPatcher instances found.")

        # 2. Check for Large Tensors on GPU
        print("[RayWorker] Scanning for large GPU tensors...")
        large_tensors = []
        try:
            for obj in gc.get_objects():
                if torch.is_tensor(obj) and obj.is_cuda:
                    mem_mb = obj.element_size() * obj.nelement() / (1024 * 1024)
                    if mem_mb > 10: # Only report > 10MB
                        large_tensors.append((mem_mb, obj.shape, obj.dtype))
        except Exception:
            pass # GC can change during iteration
        
        if large_tensors:
            print(f"[RayWorker] Found {len(large_tensors)} large GPU tensors (>10MB):")
            # Sort by size desc
            large_tensors.sort(key=lambda x: x[0], reverse=True)
            for mb, shape, dtype in large_tensors[:10]: # Top 10
                 print(f"  - {mb:.2f} MB | {shape} | {dtype}")
        else:
             print("[RayWorker] No large GPU tensors found via GC tracking.")
             
        print("[RayWorker] --- End Memory Leak Debug ---")

    def model_function_runner_get_values(self, fn, *args, **kwargs):
        return fn(self.model, *args, **kwargs)

    def get_local_rank(self):
        return self.local_rank

    def get_device_id(self):
        return self.device_id

    def get_is_model_loaded(self):
        return self.is_model_loaded

    def offload_and_clear(self):
        """
        Offloads the model from VRAM and clears tracking state.
        This uses the intricate logic previously found inline in samplers to ensure
        maximum VRAM recovery.
        """
        if self.model is not None:
            # GGUF Soft-Offload: Keep mmap active in System RAM, only clear VRAM
            if getattr(self, "is_gguf", False):
                print(f"[RayWorker {self.local_rank}] GGUF Soft-Offload: Releasing VRAM but retaining System RAM mmap...")
                
                # Unpatch to remove VRAM copies (weights moved to GPU)
                if hasattr(self.model, "unpatch_model"):
                    # device_to=offload_device usually ensures we go back to CPU
                    self.model.unpatch_model(device_to=self.model.offload_device, unpatch_weights=False)
                    self.model.current_device = torch.device("cpu")
                
                # FAILSAFE: Manually strip any remaining GPU tensors from the model
                # unpatch_model might miss internal buffers or non-parameter tensors
                try:
                    print(f"[RayWorker {self.local_rank}] GGUF Soft-Offload: Failsafe GPU strip...")
                    for param in self.model.model.parameters():
                        if param.device.type == "cuda":
                            # Replace with tiny CPU tensor to free VRAM without RAM bloat
                            # to("meta") fails due to type mismatch. to("cpu") copies data.
                            # resetting to empty CPU tensor is the robust solution.
                            param.data = torch.empty(0, dtype=param.dtype, device="cpu")
                    for buf in self.model.model.buffers():
                         if buf.device.type == "cuda":
                            buf.data = torch.empty(0, dtype=buf.dtype, device="cpu")
                except Exception as e:
                    print(f"[RayWorker {self.local_rank}] Warning during GPU strip: {e}")

                # Force VRAM cleanup - GC is critical to destroy detached GPU tensors
                import gc
                gc.collect()
                gc.collect() 
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                # Do NOT clear self.model or tracking vars (current_unet_path)
                # This enables instant "Smart Reload"
                print(f"[RayWorker {self.local_rank}] GGUF Soft-Offload complete.")
                return

            print(f"[RayWorker {self.local_rank}] Offloading model (releasing VRAM)...")
            model_to_remove = self.model
            self.model = None
            
            # Clear tracking vars to force reload next time
            self.current_unet_path = None
            self.current_sd_ref = None
            
            gc.collect()
            gc.collect()

            # Manually deregister from ComfyUI's cache to prevent it from holding refs
            try:
                mm = comfy.model_management
                if model_to_remove in mm.current_loaded_models:
                    print(f"[RayWorker {self.local_rank}] Removing model from ComfyUI cache: {model_to_remove}")
                    mm.current_loaded_models.remove(model_to_remove)
            except Exception as e:
                print(f"[RayWorker {self.local_rank}] Warning in cache removal: {e}")

            # CRITICAL: Delete this local reference too, otherwise it stays in the stack frame
            # causing the model to survive until the function returns (blocking empty_cache)
            del model_to_remove
            
            gc.collect()
            
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() # Aggressive distributed cleanup
            
            self.debug_memory_leaks()
            print(f"[RayWorker {self.local_rank}] Model offloaded successfully.")
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

    def load_unet(self, unet_path, model_options):
        # Smart Reload: Skip if already loaded
        if self.model is not None and getattr(self, "current_unet_path", None) == unet_path:
             print(f"[RayWorker] Smart Reload: Model {unet_path} already loaded. Skipping.")
             self.is_model_loaded = True
             return

        self.current_unet_path = unet_path
        
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

            import ray
            if self.state_dict is not None:
                print("[RayWorker] FSDP Mode: Saving base_sd_ref to Object Store...")
                self.base_sd_ref = ray.put(self.state_dict)

        else:
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
                          # We need to pass this as a kwarg to the loader call, not put it in model_options?
                          # ComfyUI-GGUF loader passes it as `metadata=...` kwarg.
                          # I will add it to a local dict to unpack later.
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

        @contextmanager
        def patch_torch_load_state_dict():
            original_load = torch.nn.Module.load_state_dict
            
            # Check if assign is supported (torch >= 2.1)
            sig = inspect.signature(original_load)
            supports_assign = "assign" in sig.parameters

            if not supports_assign:
                print("[Raylight] Warning: torch version does not support assign=True in load_state_dict. Shared RAM optimization disabled.")
                yield
                return

            def patched_load(self, state_dict, strict=True, assign=False):
                # Force assign=True to share memory from Object Store
                return original_load(self, state_dict, strict=strict, assign=True)
            
            try:
                torch.nn.Module.load_state_dict = patched_load
                yield
            finally:
                torch.nn.Module.load_state_dict = original_load

        loader_kwargs = getattr(self, "_temp_loader_kwargs", {})
        self._temp_loader_kwargs = None # Clear
        
        with patch_torch_load_state_dict():
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

        # ENFORCE Memory Sharing (Fix for OOM)
        # Even if load_state_dict copied data (e.g. due to casting), we force it back to shared tensor.
        try:
             count = 0
             cast_count = 0
             # Optimized prefix search
             prefixes = ["", "model.diffusion_model.", "diffusion_model."]
             
             model_params = dict(self.model.model.named_parameters())
             
             for name, param in model_params.items():
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

        if reload_params is not None:
             print("[RayWorker] GGUF Optimization: Using Parallel Disk Loading (Mmap)...")
             self.load_gguf_unet(**reload_params)
             
             if self.lora_list is not None:
                self.load_lora()
             self.is_model_loaded = True
             return
        else:
             raise ValueError("[RayWorker] Cannot RELOAD GGUF: Missing reload params for parallel loading.")
        
        if self.lora_list is not None:
            self.load_lora()
        
        self.is_model_loaded = True 

    def reload_model_if_needed(self):
        """Auto-reloads the model if it was offloaded."""
        # Check if model needs reload:
        # - For standard offload: self.model is None
        # - For GGUF soft-offload: self.model exists but current_device is CPU (nuked weights)
        needs_reload = self.model is None
        if self.model is not None and hasattr(self.model, 'current_device'):
            if self.model.current_device != self.model.load_device:
                needs_reload = True
                print(f"[RayWorker] GGUF Soft-Reload detected: current_device={self.model.current_device}, load_device={self.model.load_device}")
        
        if not needs_reload:
            return

        print("[RayWorker] Model offloaded. Auto-reloading...")
        
        # 0. GGUF SOFT-RELOAD: If model exists but is offloaded, use Self-Heal path
        # This is faster than disk reload because we use Shared RAM backup
        if self.model is not None and getattr(self, "is_gguf", False):
            print("[RayWorker] GGUF Soft-Reload: Triggering Self-Heal via model.load()...")
            # Directly call the patcher's load method to trigger Self-Heal
            self.model.load(self.device, force_patch_weights=True)
            print("[RayWorker] GGUF Soft-Reload: Complete.")
            return
        
        # 1. SPECIAL CASE: GGUF Full Disk Reload (mmap) - when model was fully cleared
        if hasattr(self, 'gguf_reload_params') and self.gguf_reload_params is not None:
             print("[RayWorker] Reloading GGUF from disk parameters (mmap)...")
             self.load_gguf_unet(**self.gguf_reload_params)
             return

        # 2. Try Shared Memory (Fastest, for Standard/FSDP)
        if hasattr(self, 'base_sd_ref') and self.base_sd_ref is not None:
             print("[RayWorker] Reloading from Shared base_sd_ref...")
             self.load_unet_from_state_dict(self.base_sd_ref, model_options={})
             return
             
        raise RuntimeError("Model is offloaded but no reload source (base_sd_ref or gguf_reload_params) is available!")

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
        lora_list=None,
    ):
        # NOTE: Reload is now handled by coordinator (sampler node) BEFORE dispatching.
        # This ensures all workers are ready before any collective operations begin.
             
        work_model = self.model

        # Apply Runtime LoRAs locally (Fallback for FSDP/No-Base-Ref)
        if lora_list:
             # Clone the model wrapper (lightweight copy of the patcher)
             work_model = work_model.clone()
             
             for lora in lora_list:
                 lora_path = lora["path"]
                 strength_model = lora["strength_model"]
                 strength_clip = 1.0 
                 try:
                     # Load LoRA from disk (No Cache to save RAM)
                     lora_model = comfy.utils.load_torch_file(lora_path, safe_load=True)
                 except Exception as e:
                     print(f"[Raylight] Failed to load LoRA {lora_path}: {e}")
                     continue
                 
                 work_model, _ = comfy.sd.load_lora_for_models(
                     work_model, None, lora_model, strength_model, strength_clip
                 )
                 del lora_model # Release raw dict immediately
             
             gc.collect()

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

            self.debug_memory_leaks()
        
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
        # NOTE: Reload is now handled by coordinator (sampler node) BEFORE dispatching.
        # This ensures all workers are ready before any collective operations begin.

        # Apply Runtime LoRAs locally (Fallback for FSDP/No-Base-Ref)
        work_model = self.model
        if lora_list:
             # Clone the model wrapper (lightweight copy of the patcher)
             work_model = work_model.clone()
             
             for lora in lora_list:
                 lora_path = lora["path"]
                 strength_model = lora["strength_model"]
                 strength_clip = 1.0 
                 try:
                     # Load LoRA from disk (No Cache to save RAM)
                     lora_model = comfy.utils.load_torch_file(lora_path, safe_load=True)
                 except Exception as e:
                     print(f"[Raylight] Failed to load LoRA {lora_path}: {e}")
                     continue
                 
                 work_model, _ = comfy.sd.load_lora_for_models(
                     work_model, None, lora_model, strength_model, strength_clip
                 )
                 del lora_model # Release raw dict immediately
             
             gc.collect()

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

        self.debug_memory_leaks()
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
