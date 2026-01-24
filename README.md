# Raylight

## Fork Enhancements (hiyazakite)

The following enhancements were added on top of komikndr's v0.16.0:

## Issue

The original Komikondr Raylight implementation does not utilize zero-copy RAM to VRAM transfer for model weights, leading to excessive RAM usage for Multi-GPU inference. The RAM requirements multiplies with the number of GPUs used, leading to high VRAM usage and potential OOM errors. There were also several issues with RAM and VRAM spikes during initial loading which is resolved in this fork.

### Memory Optimization
- **Safetensors and GGUF Lazy Loading / MMAP VRAM load and RAM Offload**: Implemented efficient mmap GPU loading. Models can be offloaded to RAM and reloaded without duplicating memory across workers. The memory footprint will now only equal the size of 1X the model size without memory spikes during initial loading and hot reloads.
- **Smart Model Reload**: Workers now detect if the requested model is already loaded and skip redundant reloads, significantly reducing iteration time for repeated sampling.

### GGUF Improvements
- **GGUF Metadata Backport**: Ported metadata extraction fix from ComfyUI-GGUF PR #399 to support LTXAV (LTX Audio-Video) models.

### Distributed VAE Decoding
- **Fixed Distributed VAE**: Added sharding of latents, corrected stitching logic for Causal VAEs to eliminate "black blinks" and visual artifacts in decoded video output.
- **Streaming**: Implemented streaming of latents to reduce RAM spike build ups during decoding.

### LTXV/LTX-2 Multi-GPU Support
- **CompressedTimestep Fix**: Fixed shape mismatch in Ulysses/Sequence Parallel mode by correctly splitting `CompressedTimestep` objects across GPU ranks to avoid excessive initial VRAM usage.
- **External Sigmas Support**: `XFuserKSamplerAdvanced` now accepts optional `SIGMAS` input for full LTXV scheduler compatibility.
- **FFN Chunking Node**: Added `RayLTXFFNChunker` node to reduce peak VRAM for LTX-2/LTXAV models by processing FeedForward layers in chunks.

### Worker Lifecycle
- **NCCL Test Skip Option**: Added `skip_comm_test` parameter to skip the NCCL communication test at startup, saving ~10-15 seconds.
- **Cluster Reuse**: Ray now attempts to reuse an existing cluster connection instead of always reinitializing.

### Additional Features
- **Model Offload Node**: Models now stay per default in VRAM after sampling. Added `RayOffloadModel` node to explicitly trigger full model offload across all Ray workers, releasing VRAM if needed with RAM MMAP backed hot reload for fast reinitialization.
- **GPU Pinning**: Added `gpu_indices` parameter to `RayInitializer` for dedicating specific GPUs to the Ray cluster.
- **Combined QKV All-to-All**: Added `pack_qkv` option for reduced NCCL communication overhead (experimental, for self-attention models).
- **Cancellable Ray Operations**: Replaced blocking `ray.get()` calls with `cancellable_get()` for better interrupt handling.
- **Branching LORA support**: LORAs are no longer added to the Loader Node but instead can be injected between the Loader and Sampler node like the regular ComfyUI usage.

### Fixes 

- **Arbitrary number of GPU support for Lumina models**: Fixed issue with Lumina models not supporting arbitrary number of GPUs.


### Note

This fork is not compatible with the original Raylight repository due to major refactoring. The code base has been tested on Linux running a dockerized ComfyUI environment using 4 x 3090 GPUs. 