# LTX-2 & LTXAV VRAM Optimizations for Distributed Inference

This document outlines the memory optimizations implemented to enable long-form video generation (800-900+ frames) on consumer hardware with 24GB VRAM.

## 1. Positional Embedding (PE) Chunking
**Problem**: The standard LTX model generates the full Positional Embedding (PE) tensor for the entire sequence (e.g., 900 frames) on every Ray worker. For long sequences, this temporary tensor can exceed 50GB, causing immediate OOM before computation even begins.

**Solution**: 
- In `xdit_context_parallel.py`, we now slice the `pixel_coords` to match the local worker's chunk *before* calling `_prepare_positional_embeddings`.
- Each worker only generates its local slice of the PE.
- **Result**: Peak VRAM during PE generation dropped from 50GB+ to <2GB for a 900-frame sequence.

## 2. local Timestep Expansion
**Problem**: The `LTXAVModel` uses `CompressedTimestep` objects. Calling `.expand()` on these objects on every worker materializes the full expanded timestep tensor simultaneously. This creates a massive VRAM spike at the start of inference.

**Solution**:
- Overrode `process_usp_timestep` in `xdit_context_parallel.py` to handle `CompressedTimestep` optimized for Sequence Parallelism.
- The expansion now happens locally using indexing: `ts.data[:, frame_indices, :]`. 
- Only the specific chunk of time/spatial steps needed for the local worker is materialized.
- **Result**: Eliminated the ~4GB+ VRAM spike per worker during startup.

## 3. FFN Chunker Monkey-Patching
**Problem**: Replacing modules (the previous strategy) broke ComfyUI's weight patching and LoRA compatibility. 

**Solution**:
- Implemented a **monkey-patching** strategy in `nodes_ltx_ffn_chunker.py`.
- We now patch the `.forward()` method of the Feed-Forward modules (`ff` and `audio_ff`) instead of replacing the entire module.
- The chunking logic processes the sequence dimension in smaller segments (e.g., 16 or 32 chunks), keeping intermediate activations (the 4x hidden dimension expansions) small.
- **Result**: Reduced FFN activation peak memory by 8x or more without breaking LoRA or weight pathing.

## 4. Instrumentation and Diagnostics
- Added `[RayWorker {sp_rank}] PE Generated. VRAM: ...` to track memory floor on every worker.
- Added `[RayWorker {sp_rank}] Blocks complete. Peak VRAM: ...` to capture the maximum memory used during the most intensive part of the inference.
- Added sequence-specific logging to identify exactly when and where optimizations are applied.

## Usage Note
Ensure the `FFN Chunker` node in your workflow has `num_chunks` set to at least **16** for sequences over 512 frames. My logs will confirm if `Video FFN Chunking Active` is triggered.
