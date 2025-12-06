import torch
from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
)


# To handle odd num gpus
def pad_to_world_size(t: torch.Tensor, dim: int = 1):
    world_size = get_sequence_parallel_world_size()
    orig_size = t.size(dim)

    # Amount of padding needed so that orig_size % world_size == 0
    pad = (world_size - orig_size % world_size) % world_size

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
