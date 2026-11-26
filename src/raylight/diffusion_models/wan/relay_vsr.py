import torch
from comfy.ldm.wan.model_vsr import RELAY_WAN_WINDOW, RelayVSRAttentionBlock
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group


def usp_relay_block(self, x, e, freqs, context, cache, start, store, transformer_options={}):
    frames, height, width = transformer_options["grid_sizes"]
    world = get_sequence_parallel_world_size()
    rank = get_sequence_parallel_rank()
    rows = (height + world - 1) // world
    first = min(rank * rows, height)
    last = min(first + rows, height)

    # Use the same shifted windows as Kitchen at both frame edges. Halo
    # outputs are discarded; only the owned rows go into the next block.
    window = min(height, RELAY_WAN_WINDOW[0])
    top = min(max(0, first - window // 2), height - window)
    bottom = min(max(0, last - 1 - window // 2), height - window) + window
    local_grid = (frames, bottom - top, width)
    def stripe(t):
        return t.unflatten(1, (frames, height, width))[:, :, top:bottom].flatten(1, 3)

    if cache is not None:
        cache = tuple(t[:, :, :, top:bottom] for t in cache)
    opts = dict(transformer_options, grid_sizes=local_grid)
    local = RelayVSRAttentionBlock.forward(self, stripe(x), e, stripe(freqs), context, cache, start, store, transformer_options=opts)
    local = local.unflatten(1, local_grid)[:, :, first - top:last - top]
    if last - first < rows:
        local = torch.nn.functional.pad(local, (0, 0, 0, 0, 0, rows - (last - first)))
    full = get_sp_group().all_gather(local.contiguous(), dim=2)
    return full[:, :, :height].flatten(1, 3)
