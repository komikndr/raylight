import torch
from xfuser.core.distributed import (
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
