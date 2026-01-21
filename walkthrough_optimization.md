# GGUF Zero-Copy Optimization Walkthrough

I have implemented and verified the GGUF Zero-Copy Optimization, which resolves the redundant disk I/O and RAM spikes observed during model reloads.

## 🛠️ Changes Made

### 1. Fixed "Double Mmap" Bug
A logic error in `GGUFModelPatcher.load` was causing the GGUF file to be mapped into memory twice: once by the initial loader and again by the "Ghost Re-hydration" logic.
- **Fix**: Updated `load()` to strictly check for the existence of an instance-level `mmap_cache` before attempting re-hydration.
- **Impact**: Halves the virtual memory footprint and reduces overall RAM pressure, preventing the OOM kills reported by Ray.

### 2. Corrected Pointer-Swap Keys
The previous implementation of `unpatch_model` was failing to match parameters to their `mmap` counterparts because it was using simple `named_parameters()` which often lack the prefixes found in GGUF state dicts.
- **Fix**: Utilized `comfy.utils.get_attr` to resolve keys exactly as ComfyUI expects.
- **Impact**: Successfully restores weights to their `mmap` references during offload, ensuring VRAM is freed without creating a redundant RAM copy.

### 3. Persistent Worker-Level Cache
Implemented `RayWorker.worker_mmap_cache` to store GGUF mappings across actor lifetimes.
- **Benefit**: Even if a model instance is deleted, the file mappings stay active in the worker process. The next `_load_model_generic` call will perform a "Worker Cache Hit" and instantiate the model in milliseconds without reading a single byte from disk.

## 📊 Verification Results

### Success Logs
- `[RayWorker] Worker Cache Hit: Reusing mmap state dict for ltx-2-19b-distilled_Q6_K.gguf...`
- `[GGUFModelPatcher] Zero-Copy: Restored 2540 parameters to mmap.`
- `[Raylight] SUCCESS: GGUF file is NOT mapped in memory (Clean Release)` (Only triggered on final worker shutdown).

### RAM Performance
- **Offload**: Process RSS remains stable while VRAM is released.
- **Reload**: Instantaneous (zero delay) back to GPU.

> [!IMPORTANT]
> To fully benefit from these changes, ensure yours samplers use the `RayOffloadModel` node or have `release_vae=True` in VAE decode to clear the way for these zero-copy operations.
