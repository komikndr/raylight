# MiniMax H3 Ray USP acceleration nodes (DEV)

This DEV package adds two native MiniMax H3 acceleration nodes for the Raylight USP path:

- `MiniMax H3 Block Cache (Ray USP)`
- `MiniMax H3 SLA Attention (Ray USP)`

Both nodes configure the Ray actors directly and keep their runtime state scoped to a single sampling run. Disabling either node preserves the existing Raylight path.

## MiniMax H3 Block Cache

The Block Cache runs all transformer blocks on `FULL` steps. On eligible `CACHE` steps it runs an initial block prefix and applies the residual captured by the previous `FULL` step. Each USP rank owns only its local residual; the implementation adds no distributed communication.

The cache is limited to MiniMax H3 with Raylight USP. It is intentionally incompatible with EasyCache and TeaCache in the same model path.

Default configuration:

```text
sigma_threshold = 0.12
start_percent = 0.10
end_percent = 0.90
max_cached_steps = 2
cache_depth = 0.75
```

An initial 20-step hardware test executed 8 `FULL` and 12 `CACHE` steps. With 50 transformer blocks and a 13-block cached prefix, it executed 556 of the baseline 1,000 block-steps and skipped 444. The measured denoising time in that test changed from approximately 4:08 to 2:24. This timing is indicative and depends on the workflow, model format, warm-up state, and memory conditions.

## MiniMax H3 SLA Attention

The SLA node ports the original PlagueKind `H3 SLA Attention` block-sparse attention to Raylight USP. The sparse kernel and block-map implementation are vendored verbatim from `ComfyUI-H3-SLA-Attention`.

With Ulysses sequence parallelism, SLA runs after the sequence all-to-all, where every rank has the full packed sequence for its local head slice. Ring parallelism greater than one and unsupported attention configurations fall back to the existing dense Raylight attention backend. The dense fallback remains XFuser `SAGE_AUTO` when that backend is selected.

Validated configuration:

```text
Ulysses = 2
Ring = 1
CFG = 1
Pipeline parallel = 1
Data parallel = 1

sparsity_ratio = 0.90
block_size = 64
min_seq_len = 8192
dense_last_steps = 0
protect_audio = true
```

The protected packed prefix covers text, conditioning, reference, and audio tokens before the video segment.

## Hardware and workflow validation

Validated on two NVIDIA GeForce RTX 3090 GPUs with MiniMax H3 video and audio generation.

The first pass used:

```text
6 denoising steps
4-step LoRA
MiniMax H3 Sigma Shift: video shift = 12
audio generation enabled
MiniMax H3 SLA Attention enabled
```

For a 15-second sample with a packed sequence length of 36,731 tokens, the six-step first pass completed in approximately 1:28. The run executed 300 sparse attention calls per rank with 28 local heads and no dense fall-throughs.

The latent-upscale pass used the ER-SDE Beta sampler without Sigma Shift. The tested schedules were:

```text
3 steps:
0.9035, 0.6316, 0.3158, 0.0000

4 steps:
0.9035, 0.8000, 0.6316, 0.3158, 0.0000

5 steps:
0.9231, 0.8780, 0.8000, 0.6316, 0.3158, 0.0000
```

The three-step upscale pass used a packed sequence length of 132,710 tokens and completed in approximately 3:55. Video quality remained visually valid across repeated runs, without the temporal grey-frame collapse found during development, and audio remained correct.

## Validation

The focused Block Cache and SLA suite passes 48 tests, covering disabled pass-through, cache lifecycle, worker dispatch, sparse selection, protected-prefix semantics, per-head Ulysses consistency, dense fallbacks, hook idempotence, logging, the Triton kernel, and the original block-map pooling semantics.

This remains a DEV feature. Performance and quality should be validated for each model format, LoRA, resolution, duration, sampler, and cache configuration before production use.
