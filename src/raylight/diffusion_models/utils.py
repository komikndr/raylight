import torch
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)


def pad_to_size(t: torch.Tensor, size: int, dim: int = 1):
    if size <= 0:
        raise ValueError(f"pad_to_size requires a positive size, got {size}")

    orig_size = t.size(dim)

    # Amount of padding needed so that orig_size % size == 0
    pad = (size - orig_size % size) % size

    if pad == 0:
        return t, orig_size

    pad_shape = list(t.shape)
    pad_shape[dim] = pad

    pad_tensor = torch.zeros(
        pad_shape,
        dtype=t.dtype,
        device=t.device,
    )

    t = torch.cat([t, pad_tensor], dim=dim)
    return t, orig_size


# To handle odd num gpus
def pad_to_world_size(t: torch.Tensor, dim: int = 1):
    return pad_to_size(t, get_sequence_parallel_world_size(), dim=dim)


def shard_ranges(ranges, width):
    if ranges is None:
        return None
    offset = get_sequence_parallel_rank() * width
    return [(max(0, start - offset), min(width, end - offset) if end is not None else width, *rest)
            for start, end, *rest in ranges
            if start < offset + width and (end is None or end > offset)]


def joint_rope_shards(embedder, txt_ids, img_ids, txt_len, img_len):
    world, rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
    # Double blocks shard each stream separately; single blocks shard their unpadded concatenation.
    local_ids = torch.cat((txt_ids.chunk(world, dim=1)[rank], img_ids.chunk(world, dim=1)[rank]), dim=1)
    joint_ids, _ = pad_to_world_size(torch.cat((txt_ids[:, :txt_len], img_ids[:, :img_len]), dim=1))
    return embedder(local_ids), embedder(joint_ids.chunk(world, dim=1)[rank])
