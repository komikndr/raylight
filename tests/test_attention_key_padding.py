"""Padded keys stay out of the distributed attention.

Four CPU ranks run the real xFuser Ulysses exchange and ring loop over a sequence padded
to the world size. Every real query must equal attention over the real keys alone.
"""
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

SRC = str(Path(__file__).resolve().parents[1] / "src")

WORLD = 4
HEADS = 4
DIM = 8


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reference_step(q, k, v, softmax_scale=None, **kwargs):
    # exact per-step attention with its log-sum-exp, [b, s, h, d] in and out
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    scores = torch.einsum("bqhd,bkhd->bhqk", q.double(), k.double()) * scale
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.double())
    return out.to(q.dtype), lse.to(q.dtype)


class _ExactProcessor:
    supports_ring = True

    def __call__(self, dense_attention, q, k, v, *, transformer_options, **kwargs):
        options = dict(kwargs)
        options["attn_type"] = None
        options["attn_processor"] = _reference_step
        return dense_attention(q, k, v, **options)


def _rank(rank, port, ulysses, ring, length, results):
    sys.path.insert(0, SRC)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(WORLD), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    # xFuser and yunchang only define their ring pieces when CUDA is visible; the
    # tensors here stay on the CPU.
    torch.cuda.is_available = lambda: True
    torch.distributed.init_process_group("gloo", rank=rank, world_size=WORLD)
    # the groups xFuser builds through yunchang for ulysses x ring
    from yunchang import set_seq_parallel_pg
    set_seq_parallel_pg(ulysses, ring, rank, WORLD)
    sp = rank

    from raylight.distributed_modules import attention as xfuser_attn
    from raylight.distributed_modules.inner_attention import INNER_ATTENTION_KEY
    attend = xfuser_attn.make_xfuser_attention("TORCH_EFFICIENT", False)

    torch.manual_seed(0)
    local = -(-length // WORLD)
    full = [torch.randn(1, HEADS, WORLD * local, DIM, dtype=torch.float64) for _ in range(3)]
    for t in full:
        t[:, :, length:] = 0  # padding, as the model pads it
    q, k, v = (t[:, :, sp * local:(sp + 1) * local].contiguous() for t in full)

    options = {INNER_ATTENTION_KEY: _ExactProcessor()}
    out = attend(q, k, v, HEADS, skip_reshape=True, skip_output_reshape=True,
                 transformer_options=options, key_length=length)
    unmasked = attend(q, k, v, HEADS, skip_reshape=True, skip_output_reshape=True,
                      transformer_options=options)

    real = slice(0, max(0, min(local, length - sp * local)))
    ref = torch.nn.functional.scaled_dot_product_attention(
        full[0][:, :, sp * local:(sp + 1) * local], full[1][:, :, :length], full[2][:, :, :length])
    results[rank] = (
        (out[:, :, real] - ref[:, :, real]).abs().max().item(),
        (unmasked[:, :, real] - ref[:, :, real]).abs().max().item(),
    )
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("ulysses,ring", [(2, 2), (1, 4), (4, 1)])
@pytest.mark.parametrize("length", [37, 38, 39])
def test_padded_keys_are_left_out(ulysses, ring, length):
    results = mp.Manager().dict()
    mp.spawn(_rank, args=(_free_port(), ulysses, ring, length, results), nprocs=WORLD)
    for rank in range(WORLD):
        masked, unmasked = results[rank]
        # the ring merges its steps in float32
        assert masked < 1e-6, (rank, masked)
        # the padding would otherwise reach every query
        assert unmasked > 1e-3, (rank, unmasked)


def test_unpadded_length_leaves_keys_alone():
    from raylight.distributed_modules.key_padding import exclude_padded_keys

    k = torch.randn(1, 10, 2, 4)
    fn = object()
    assert exclude_padded_keys(fn, k, k, None) == (fn, k, k)


def test_block_rows():
    from raylight.distributed_modules.key_padding import block_key_rows

    # 37 rows padded to 40, cut into blocks of 10
    assert [block_key_rows(37, b, 10) for b in range(4)] == [10, 10, 10, 7]
