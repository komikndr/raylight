"""Keep the rows a sequence-parallel split pads on out of the attention.

A sequence that does not divide by the sequence-parallel world size is padded at its end.
After the Ulysses exchange every rank of a ring group holds one contiguous block of the
padded sequence, ring rank j holding block j, so the padding is the tail of the last
block. Each ring step attends to one block's keys; the step attending to the last block
uses only its real rows. Every query still sees exactly the real keys, through any
attention kernel, without gathering the sequence onto one rank.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_KEY_LENGTH = ContextVar("raylight_key_length", default=None)


@contextmanager
def key_length_scope(length):
    """Attention inside this scope ignores keys past `length` rows of the full sequence."""
    token = _KEY_LENGTH.set(length)
    try:
        yield
    finally:
        _KEY_LENGTH.reset(token)


def active_key_length():
    return _KEY_LENGTH.get()


def _ring_position(group):
    if group is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0, 1
    return torch.distributed.get_rank(group), torch.distributed.get_world_size(group)


def _check_layout(ring_group):
    # The blocks are in sequence order only when each Ulysses group is a run of
    # consecutive ranks and each ring group steps across them, yunchang's default layout.
    from yunchang.globals import PROCESS_GROUP
    ulysses = torch.distributed.get_process_group_ranks(PROCESS_GROUP.ULYSSES_PG)
    ring = torch.distributed.get_process_group_ranks(ring_group)
    if (ulysses != list(range(ulysses[0], ulysses[0] + len(ulysses))) or
            ring != list(range(ring[0], ring[0] + len(ring) * len(ulysses), len(ulysses)))):
        raise RuntimeError("key padding needs Ulysses groups of consecutive ranks; "
                           "Ulysses group {}, ring group {}".format(ulysses, ring))


def block_key_rows(length, block, block_rows):
    """Real rows of block `block` of a sequence of `length` rows cut into `block_rows` blocks."""
    return max(0, min(block_rows, length - block * block_rows))


def key_slices(lengths, world, ring):
    """Valid rows per ring block when each SP rank concatenates its stream shards."""
    widths = [-(n // -world) for n in lengths]
    blocks = []
    for first in range(0, world, world // ring):
        runs = []
        offset = 0
        for rank in range(first, first + world // ring):
            for n, width in zip(lengths, widths):
                end = offset + block_key_rows(n, rank, width)
                if end > offset:
                    if runs and runs[-1][1] == offset:
                        runs[-1] = (runs[-1][0], end)
                    else:
                        runs.append((offset, end))
                offset += width
        blocks.append(runs)
    return blocks


def select_rows(x, runs):
    if len(runs) == 1:
        start, end = runs[0]
        return x[:, start:end]
    return torch.cat([x[:, start:end] for start, end in runs], dim=1)


def exclude_padded_keys(ring_attn_fn, k, v, group):
    """Return the ring attention, keys and values that leave the active scope's padding out.

    Unchanged when no scope is active. Without a ring the keys and values are cut here, so
    any local attention sees only the real ones. With a ring the per-step attention is
    wrapped instead: the ring loop calls it once per step in order, and step s on ring rank
    r attends to the block of ring rank (r - s) mod n.
    """
    length = active_key_length()
    if length is None:
        return ring_attn_fn, k, v
    block_rows = k.shape[1]
    ring_rank, ring_size = _ring_position(group)
    if isinstance(length, list):
        blocks = length
    elif length > ring_size * block_rows:
        raise ValueError("key length {} is past the padded sequence of {} rows".format(
            length, ring_size * block_rows))
    elif length <= (ring_size - 1) * block_rows:
        raise ValueError("a whole ring block of {} rows is padding (key length {})".format(
            block_rows, length))
    else:
        blocks = [[(0, block_key_rows(length, r, block_rows))] for r in range(ring_size)]
    if ring_size == 1:
        return ring_attn_fn, select_rows(k, blocks[0]), select_rows(v, blocks[0])
    _check_layout(group)

    from yunchang.kernels import AttnType, select_flash_attn_impl

    def ring_with_real_keys(q, k, v, *args, **kwargs):
        attn_type = kwargs.get("attn_type")
        if attn_type in (AttnType.FA3, AttnType.SPARSE_SAGE):
            raise NotImplementedError("key padding is not supported with {}".format(attn_type))
        step_attention = select_flash_attn_impl(attn_type, stage="fwd-only",
                                                attn_processor=kwargs.get("attn_processor"))
        steps = iter(range(ring_size))

        def step(query, key, value, *step_args, **step_kwargs):
            runs = blocks[(ring_rank - next(steps)) % ring_size]
            key, value = select_rows(key, runs), select_rows(value, runs)
            return step_attention(query, key, value, *step_args, **step_kwargs)

        kwargs["attn_type"] = None
        kwargs["attn_processor"] = step
        return ring_attn_fn(q, k, v, *args, **kwargs)

    return ring_with_real_keys, k, v
