"""Block selection: which key blocks each query block is allowed to see.

Vendored from LightX2V (Apache-2.0):
  lightx2v/common/ops/attn/utils/sla_util_blhd.py
  https://github.com/ModelTC/LightX2V

This is the whole of what "SLA" does at inference time. Mean-pool Q into blocks,
mean-pool a smoothed K into blocks, score the two against each other with one
small matmul, and keep the top ``topk_ratio`` fraction of key blocks per query
block. No weights, nothing trained, nothing to load -- the published SLA LoRA
adapts the *model* to tolerate the resulting sparsity, it does not parameterise
this step.

Three changes against upstream, marked FIX:

1. ``other=0.0`` on the masked load. The masked lanes feed ``tl.sum`` two lines
   later, and Triton leaves them undefined, so the final (partial) block of the
   sequence pooled whatever was in memory. Upstream fixed exactly this in the
   BHSD twin of this file and did not port the fix here.
2. ``max(1, ...)`` on the top-k count, so a short sequence keeps one key block
   rather than zero. Also upstream's BHSD-only fix.
3. Smooth-k is folded into the pooled result instead of being materialised.
   Pooling is a mean over L and the correction is constant along L, so
   ``pool(k - mu) == pool(k) - mu`` exactly. Upstream builds the whole smoothed
   copy of K, which at 768p/15s is a needless ~1.3 GB allocation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compress_kernel(
    X,
    XM,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    idx_b = idx_bh // H
    idx_h = idx_bh - idx_b * H

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_b * L * H * D + idx_h * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    # FIX vs upstream: other=0.0 -- these lanes are summed below.
    x = tl.load(
        X + x_offset + offs_l[:, None] * (H * D) + offs_d[None, :],
        mask=offs_l[:, None] < L,
        other=0.0,
    )

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x, BLK):
    """``(B, L, H, D)`` -> ``(B, H, ceil(L/BLK), D)``, mean over each L block."""
    assert x.is_contiguous()
    B, L, H, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    # fp32, not x.dtype. The Triton reduction already accumulates in fp32; only
    # the store rounds. Upstream stores bf16, which quantises the block *scores*
    # and makes top-k pick different blocks than exact arithmetic would -- and
    # at 85-90% sparsity only 3-4 blocks per query survive, so one wrong pick is
    # a large error. Keeping fp32 here costs a few MB and no measurable time.
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=torch.float32)

    grid = (L_BLOCKS, B * H)
    _compress_kernel[grid](x, x_mean, L, H, D, BLK)
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ=128, BLKK=128, protect_upto=0):
    """Return ``(lut, topk)``: the key blocks each query block should attend to.

    ``q``/``k`` are ``(B, L, H, D)`` contiguous. ``lut`` comes back as
    ``(B, H, ceil(LQ/BLKQ), topk)`` int32, contiguous, ready for the kernel.

    ``protect_upto`` pins the first N tokens into every query block's selection.
    For H3 that is the ``[text | cond | audio]`` prefix, and it exists because
    plain top-k starves audio: at 768p/15s the audio is ~1% of the packed
    sequence (19 key blocks of 1794), so nothing makes a query keep any of it,
    and the smooth-k mean it is scored against is 99% video. The pinned blocks
    are added on top of the top-k budget rather than displacing video, so video
    coverage is unchanged and the extra cost is the prefix itself (~7%).
    """
    pooled_q = mean_pool(q, BLKQ)
    # Smooth-k (SageAttention's trick), folded in rather than materialised.
    mu = k.mean(dim=1, dtype=torch.float32)                  # (B, H, D)
    pooled_k = mean_pool(k, BLKK) - mu[:, :, None, :]

    # GQA, for completeness -- H3 is MHA so this is a no-op there.
    num_q_heads, num_kv_heads = pooled_q.shape[1], pooled_k.shape[1]
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0
        pooled_k = pooled_k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

    pooled_score = pooled_q @ pooled_k.transpose(-1, -2)      # (B, H, NQ, NK)

    NK = pooled_score.shape[-1]
    # FIX vs upstream: keep at least one key block.
    topk = max(1, min(NK, int(topk_ratio * NK)))

    n_pinned = 0
    if protect_upto > 0:
        n_pinned = min((int(protect_upto) + BLKK - 1) // BLKK, NK)
        # Ranking them above everything else is what pins them; widening topk by
        # the same amount is what stops them evicting the blocks top-k chose.
        if n_pinned > 0:
            pooled_score[..., :n_pinned] = float("inf")
            topk = min(NK, topk + n_pinned)

    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    return lut.to(torch.int32).contiguous(), topk
