"""Triton selected-block H3 SLA attention.

Selection follows ComfyUI-PlagueKind-Nodes (MIT), with internals derived
from LightX2V (Apache-2.0). This kernel consumes BLHD tensors and never
materializes a sequence-by-sequence score matrix.
"""

import torch

_H3_HEAD_DIM = 128

try:
    import triton
    import triton.language as tl
except ImportError:
    raise ImportError("Triton is required for H3 SLA")

from ._h3_sla_block_map import get_block_map


_LAUNCH_CONFIGS = {}
_LAUNCH_LADDER = {
    (128, 64): ((8, 3), (4, 3), (8, 2), (4, 1)),
    (128, 128): ((8, 2), (4, 2), (8, 1), (4, 1)),
    (64, 128): ((4, 2), (8, 2), (4, 1)),
    (64, 64): ((4, 1), (4, 3), (8, 3), (8, 1)),
    (32, 32): ((4, 2), (2, 1), (4, 1), (2, 2)),
}


@triton.jit
def _sla_forward_kernel(
        q_ptr, k_ptr, v_ptr, lut_ptr, out_ptr,
        q_sb, q_ss, q_sh, q_sd, k_sb, k_ss, k_sh, k_sd,
        v_sb, v_ss, v_sh, v_sd, o_sb, o_ss, o_sh, o_sd,
        scale,
        H: tl.constexpr, LQ: tl.constexpr, LK: tl.constexpr, M_BLOCKS: tl.constexpr,
        D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TOPK: tl.constexpr):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // H
    head = batch_head % H
    q_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, D)
    q_mask = q_offsets[:, None] < LQ
    q = tl.load(q_ptr + batch * q_sb + q_offsets[:, None] * q_ss + head * q_sh + d_offsets[None, :] * q_sd,
                mask=q_mask, other=0.0)
    max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    norm = tl.zeros((BLOCK_M,), tl.float32)
    output = tl.zeros((BLOCK_M, D), tl.float32)
    for lut_index in tl.range(0, TOPK):
        key_block = tl.load(lut_ptr + batch * H * M_BLOCKS * TOPK + head * M_BLOCKS * TOPK
                            + query_block * TOPK + lut_index)
        key_offsets = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        k_mask = key_offsets[:, None] < LK
        k = tl.load(k_ptr + batch * k_sb + key_offsets[:, None] * k_ss + head * k_sh + d_offsets[None, :] * k_sd,
                    mask=k_mask, other=0.0)
        v = tl.load(v_ptr + batch * v_sb + key_offsets[:, None] * v_ss + head * v_sh + d_offsets[None, :] * v_sd,
                    mask=k_mask, other=0.0)
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        scores = tl.where(key_offsets[None, :] < LK, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(max_score, block_max)
        old_scale = tl.exp(max_score - new_max)
        weights = tl.exp(scores - new_max[:, None])
        output = output * old_scale[:, None] + tl.dot(weights.to(v.dtype), v).to(tl.float32)
        norm = norm * old_scale + tl.sum(weights, axis=1)
        max_score = new_max
    result = output / norm[:, None]
    out_mask = q_offsets[:, None] < LQ
    tl.store(out_ptr + batch * o_sb + q_offsets[:, None] * o_ss + head * o_sh + d_offsets[None, :] * o_sd,
             result, mask=out_mask)


def forward(q, k, v, *, sparsity_ratio, block_size, protected_ranges, **kwargs):
    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("H3 SLA requires CUDA fp16 or bf16 tensors")
    if (block_size not in (32, 64, 128) or q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or
            q.shape[-1] != _H3_HEAD_DIM):
        raise ValueError("unsupported H3 SLA shape or block size")
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    blkk = 64 if block_size == 128 else block_size
    lut, topk = get_block_map(q, k, blkq=block_size, blkk=blkk,
                              sparsity_ratio=sparsity_ratio, protected_ranges=protected_ranges)
    batch, sequence, heads, _ = q.shape
    block_q, block_k = block_size, blkk
    output = torch.empty_like(q)
    m_blocks = (sequence + block_size - 1) // block_size
    grid = (m_blocks, batch * heads)
    scale = kwargs.get("softmax_scale")
    if scale is None:
        scale = _H3_HEAD_DIM ** -0.5
    arguments = (q, k, v, lut, output, *q.stride(), *k.stride(), *v.stride(), *output.stride(), scale)
    constants = dict(H=heads, LQ=sequence, LK=sequence, M_BLOCKS=m_blocks,
                     D=_H3_HEAD_DIM, BLOCK_M=block_q, BLOCK_N=block_k, TOPK=topk)
    cache_key = (block_q, block_k, _H3_HEAD_DIM)
    cached = _LAUNCH_CONFIGS.get(cache_key)
    candidates = _LAUNCH_LADDER[(block_q, block_k)]
    if cached is not None:
        candidates = (cached,) + tuple(config for config in candidates if config != cached)
    last_error = None
    for num_warps, num_stages in candidates:
        try:
            _sla_forward_kernel[grid](*arguments, num_warps=num_warps, num_stages=num_stages, **constants)
        except triton.runtime.errors.OutOfResources as error:
            last_error = error
            continue
        _LAUNCH_CONFIGS[cache_key] = (num_warps, num_stages)
        break
    else:
        raise last_error
    return output
