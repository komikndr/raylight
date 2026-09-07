"""Block selection for MiniMax H3 SLA.

The layout follows ComfyUI-PlagueKind-Nodes (MIT), whose internals derive
from LightX2V (Apache-2.0).
"""

import torch


_TRITON_POOL = None
_TRITON_POOL_CHECKED = False


def _torch_mean_pool(value, block_size):
    sequence = value.shape[1]
    return torch.stack(
        [value[:, start:min(start + block_size, sequence)].float().mean(dim=1)
         for start in range(0, sequence, block_size)], dim=1
    )


def _load_triton_pool():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit
    def mean_pool_kernel(x, out, sequence, heads, dim,
                         x_sb, x_ss, x_sh, x_sd, o_sb, o_ss, o_sh, o_sd,
                         BLOCK_D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        blocks = (sequence + BLOCK_SIZE - 1) // BLOCK_SIZE
        batch = pid // (blocks * heads)
        block = (pid // heads) % blocks
        head = pid % heads
        offsets = tl.arange(0, BLOCK_D)
        values = tl.zeros((BLOCK_D,), tl.float32)
        for index in range(0, BLOCK_SIZE):
            token = block * BLOCK_SIZE + index
            values += tl.load(x + batch * x_sb + token * x_ss + head * x_sh + offsets * x_sd,
                              mask=(token < sequence) & (offsets < dim), other=0.0).to(tl.float32)
        count = tl.minimum(BLOCK_SIZE, sequence - block * BLOCK_SIZE)
        tl.store(out + batch * o_sb + block * o_ss + head * o_sh + offsets * o_sd,
                 values / count, mask=offsets < dim)

    def pool(value, block_size):
        batch, sequence, heads, dim = value.shape
        blocks = (sequence + block_size - 1) // block_size
        output = torch.empty((batch, blocks, heads, dim), dtype=torch.float32, device=value.device)
        grid = (batch * blocks * heads,)
        mean_pool_kernel[grid](value, output, sequence, heads, dim, *value.stride(), *output.stride(),
                               BLOCK_D=triton.next_power_of_2(dim), BLOCK_SIZE=block_size)
        return output

    return pool


def mean_pool(value, block_size):
    if value.device.type == "cuda":
        global _TRITON_POOL, _TRITON_POOL_CHECKED
        if not _TRITON_POOL_CHECKED:
            _TRITON_POOL = _load_triton_pool()
            _TRITON_POOL_CHECKED = True
        pool = _TRITON_POOL
        if pool is not None:
            return pool(value, block_size)
    return _torch_mean_pool(value, block_size)


def get_block_map(q, k, *, blkq, blkk, sparsity_ratio=None, topk=None, protected_ranges=()):
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("H3 SLA expects matching BLHD MHA tensors")
    if blkq not in (32, 64, 128) or blkk not in (32, 64, 128):
        raise ValueError("unsupported H3 SLA block size")
    q_pool = mean_pool(q, blkq)
    k_pool = mean_pool(k, blkk)
    k_mean = k.mean(dim=1, dtype=torch.float32).unsqueeze(1)
    k_pool = k_pool - k_mean
    scores = torch.einsum("bqhd,bkhd->bhqk", q_pool, k_pool)
    key_blocks = scores.shape[-1]
    if topk is None:
        ordinary = max(1, int(key_blocks * (1.0 - sparsity_ratio)))
    else:
        ordinary = max(1, int(key_blocks * topk) if isinstance(topk, float) and topk <= 1 else int(topk))
    pinned = set()
    for start, stop in protected_ranges:
        pinned.update(range(max(0, start // blkk), min(key_blocks, (stop + blkk - 1) // blkk)))
    for block in pinned:
        scores[..., block] = float("inf")
    ordinary = min(key_blocks, ordinary + len(pinned))
    lut = scores.topk(ordinary, dim=-1).indices.to(torch.int32)
    return lut.contiguous(), ordinary
