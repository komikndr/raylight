# LTXV Integration and Fixes Summary

This document summarizes the changes made to support LTXV and other models in Raylight.

## 1. LTXV Support via Standard Nodes (Architecture Update)

*Note: The initial internal integration of LTXV scheduler and Model Sampling node was reverted as per user request, in favor of using standard ComfyUI nodes.*

### `src/raylight/distributed_worker/ray_worker.py`
- Updated `common_ksampler` to accept an optional `sigmas` argument.
- **Impact**: Enables `XFuserKSampler` and `XFuserKSamplerAdvanced` to accept external `SIGMAS` (e.g., from the standard `LTXVScheduler` node). This allows full LTXV scheduler support without custom Raylight nodes.

## 2. GGUF Metadata Backport (Fix for LTXAV)

### `src/raylight/expansion/comfyui_gguf/loader.py`
- **Backport**: Ported the metadata extraction fix from ComfyUI-GGUF PR #399.
- **Changes**:
  - Added `get_gguf_metadata` helper.
  - Updated `gguf_sd_loader` to return `(state_dict, extra_info)` tuple, where `extra_info` contains the extracted metadata.

### `src/raylight/comfy_dist/sd.py`
- Updated `gguf_load_diffusion_model` to handle the new tuple return from the loader.
- Extracts `metadata` from the loader and passes it to `comfy.sd.load_diffusion_model_state_dict`. This ensures that model-specific metadata (needed for LTXAV) is correctly registered.

### GGUF Loader Diff

```diff
diff --git a/src/raylight/comfy_dist/sd.py b/src/raylight/comfy_dist/sd.py
index 69f9e56..dd4b1aa 100644
--- a/src/raylight/comfy_dist/sd.py
+++ b/src/raylight/comfy_dist/sd.py
@@ -127,6 +127,7 @@ def gguf_load_diffusion_model(unet_path, model_options={}, dequant_dtype=None, p
     from raylight.expansion.comfyui_gguf.ops import GGMLOps
     from raylight.expansion.comfyui_gguf.loader import gguf_sd_loader
     from raylight.expansion.comfyui_gguf.nodes import GGUFModelPatcher
+    import inspect
 
     ops = GGMLOps()
 
@@ -145,8 +146,14 @@ def gguf_load_diffusion_model(unet_path, model_options={}, dequant_dtype=None, p
         ops.Linear.patch_dtype = getattr(torch, patch_dtype)
 
     # init model
-    sd = gguf_sd_loader(unet_path)
-    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops})
+    sd, extra = gguf_sd_loader(unet_path)
+    
+    kwargs = {}
+    valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
+    if "metadata" in valid_params:
+        kwargs["metadata"] = extra.get("metadata", {})
+        
+    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops}, **kwargs)
     if model is None:
         logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
         raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
diff --git a/src/raylight/expansion/comfyui_gguf/loader.py b/src/raylight/expansion/comfyui_gguf/loader.py
index 810b1a6..afdc5ca 100644
--- a/src/raylight/expansion/comfyui_gguf/loader.py
+++ b/src/raylight/expansion/comfyui_gguf/loader.py
@@ -48,7 +48,26 @@ def get_list_field(reader, field_name, field_type):
     else:
         raise TypeError(f"Unknown field type {field_type}")
 
-def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", return_arch=False, is_text_model=False):
+def get_gguf_metadata(reader):
+    """Extract all simple metadata fields like safetensors"""
+    metadata = {}
+    for field_name in reader.fields:
+        try:
+            field = reader.get_field(field_name)
+            if len(field.types) == 1:  # Simple scalar fields only
+                if field.types[0] == gguf.GGUFValueType.STRING:
+                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
+                elif field.types[0] == gguf.GGUFValueType.INT32:
+                    metadata[field_name] = int(field.parts[field.data[-1]])
+                elif field.types[0] == gguf.GGUFValueType.F32:
+                    metadata[field_name] = float(field.parts[field.data[-1]])
+                elif field.types[0] == gguf.GGUFValueType.BOOL:
+                    metadata[field_name] = bool(field.parts[field.data[-1]])
+        except:
+            continue
+    return metadata
+
+def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", is_text_model=False):
     """
     Read state dict as fake tensors
     """
@@ -136,9 +155,12 @@ def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", return_arch=Fal
         max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
         state_dict[max_key].is_largest_weight = True
 
-    if return_arch:
-        return (state_dict, arch_str)
-    return state_dict
+    # extra info to return
+    extra = {
+        "arch_str": arch_str,
+        "metadata": get_gguf_metadata(reader)
+    }
+    return (state_dict, extra)
 
 # for remapping llama.cpp -> original key names
 T5_SD_MAP = {
@@ -246,7 +268,7 @@ def gguf_mmproj_loader(path):
 
     logging.info(f"Using mmproj '{target[0]}' for text encoder '{tenc_fname}'.")
     target = os.path.join(root, target[0])
-    vsd = gguf_sd_loader(target, is_text_model=True)
+    vsd, _ = gguf_sd_loader(target, is_text_model=True)
 
     # concat 4D to 5D
     if "v.patch_embd.weight.1" in vsd:
@@ -375,7 +397,8 @@ def gguf_tekken_tokenizer_loader(path, temb_shape):
     return torch.ByteTensor(list(json.dumps(data).encode('utf-8')))
 
 def gguf_clip_loader(path):
-    sd, arch = gguf_sd_loader(path, return_arch=True, is_text_model=True)
+    sd, extra = gguf_sd_loader(path, is_text_model=True)
+    arch = extra.get("arch_str", None)
     if arch in {"t5", "t5encoder"}:
         temb_key = "token_embd.weight"
         if temb_key in sd and sd[temb_key].shape == (256384, 4096):
```

## 3. LTXV Multi-GPU Fix (Ulysses/Sequence Parallel)

### `src/raylight/diffusion_models/lightricks/xdit_context_parallel.py`
- **Issue**: `LTXVModel` uses a `CompressedTimestep` object for optimization. In distributed sequence-parallel (Ulysses) mode, this object was not being split across GPUs, causing a shape mismatch between the full-size timestep embeddings and the split video tokens.
- **Fix**:
  - Implemented `process_usp_timestep` helper function.
  - This function recursively inspects the timestep arguments.
  - If it encounters a `CompressedTimestep` (detected via `expand` method), it expands it to a full tensor, pads it to world size, and splits (chunks) it according to the rank.
  - Handles `torch.Tensor` objects correctly (checking `is_tensor` before duck-typing `expand`).
- **Result**: Ensures modulation tensors (`vscale`, `vshift`) correctly match the local token count on each GPU during distributed inference.

## 4. GPU Pinning via `RayInitializer`

- **Feature**: Added `gpu_indices` optional input to `RayInitializer` and `RayInitializerAdvanced`.
- **Logic**:
  - Allows user to specify a comma-separated list of GPU indices (e.g., `"0,2"`) to dedicate to the Ray cluster.
  - Can be used to isolate Raylight to specific GPUs on a multi-GPU node.
  - Uses `CUDA_VISIBLE_DEVICES` temporarily during `ray.init` to enforce the constraint without permanently affecting the main ComfyUI process.

## 5. RAM Optimization (Zero-Copy Model Loading)
- **Problem**: Previously, each Ray worker loaded the model checkpoint from disk into its own process memory. With N GPUs, this caused N x ModelSize usage (e.g., 30GB for 2 GPUs for a 15GB model).
- **Solution**: Implemented a "Load Once, Share Everywhere" strategy using Ray's Plasma Object Store.
- **Mechanism**:
  - `RayUNETLoader` instructs **Worker 0** to load the model state dict.
  - Worker 0 returns the state dict, which Ray automatically puts into the shared memory Object Store on the node.
  - `RayUNETLoader` receives a reference (`ObjectRef`) to this shared memory object.
  - It passes this reference to all workers.
  - All workers (including Worker 0) access the weights via zero-copy reads from the shared store.
  - **Eager Freeing**: Immediately after usage, the Object Store reference is freed using `ray.internal.free([sd_ref])` to prevent RAM bloat.
  - **Garbage Collection**: Explicit `gc.collect()` is triggered on workers to minimize transient memory usage.
- **Impact**: RAM usage for model loading is now ~1x ModelSize regardless of the number of GPUs on the same node.

## 6. Dynamic Object Store Sizing
- **Feature**: Added `ray_object_store_gb` input to `RayInitializer` with a default of **0.0 (Auto)**.
- **Adjustment**: 
    - Setting this to `0.0` allows Ray to automatically manage the object store size (typically 30% of system RAM), preventing "Disk Spilling" errors when loading large models (>15GB) if the store was set too small (previous default was 2.0GB).
    - Users can still manually override this with a specific GB value if needed.

## 7. Advanced Memory Sharing (Experimental)
- **Feature**: Implemented `assign=True` patching for `torch.load_state_dict` in Workers.
- ** Mechanism**:
    - Uses a custom context manager `patch_torch_load_state_dict` inside `RayWorker`.
    - Forces `torch` to use `assign=True` when loading the model state dict from the shared memory object.
    - This instructs PyTorch to initialize the model parameters as *views* of the existing shared memory tensors, rather than creating new copies.
- **Impact**: Should dramatically reduce Runtime RAM usage essentially allowing N workers to share a single physical copy of the model weights in RAM (approx 1x Model Size total for all workers).
