import torch

import comfy.ldm.common_dit
import comfy.model_prefetch
import comfy.ops
from comfy.ldm.pan2.model import PAN2
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

import raylight.distributed_modules.attention as xfuser_attn


xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(xfuser_attn.get_attn_type(), xfuser_attn.get_sync_ulysses())


def usp_attention(original_attention, q, k, v, heads, transformer_options={}, **kwargs):
    return xfuser_optimized_attention(q, k, v, heads, transformer_options=transformer_options, **kwargs)


def usp_dit_forward(self, x, timestep, context, transformer_options={}, **kwargs):
    world_size = get_sequence_parallel_world_size()
    t, h, w = ((s + p - 1) // p for s, p in zip(x.shape[2:], self.patch_size))
    # Padding either stream would introduce extra keys into native joint attention.
    if ((t * h * w) % world_size or context.shape[1] % world_size
            or self.raylight_native_attention or "optimized_attention_override" in transformer_options
            or transformer_options.get("patches_replace", {}).get("dit")):
        return PAN2._forward(self, x, timestep, context, transformer_options, **kwargs)

    B, _, T, H, W = x.shape
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()
    # the refiner and joint blocks both reach attention through the module-level
    # attention helper, so the shared override replaces every attention hop
    transformer_options["optimized_attention_override"] = usp_attention
    x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
    # allocated outside the memory compiler region, the blocks update both streams in place
    img = self.x_embedder(x).flatten(2).transpose(1, 2).contiguous()
    del x
    vec = self.time_embedding(timestep, dtype)
    pe = self.rope(t, h, w, device)

    txt = self.context_embedder(context)

    # ===================== SP SPLIT ====================== #
    rank = get_sequence_parallel_rank()
    img = torch.chunk(img, world_size, dim=1)[rank].clone()
    pe = torch.chunk(pe, world_size, dim=1)[rank]
    txt = torch.chunk(txt, world_size, dim=1)[rank].clone()

    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.context_refiner), device, transformer_options)
    for block in self.context_refiner:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype)
        txt = block(txt, transformer_options)
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
    txt = txt + comfy.ops.cast_to_input(self.context_type_embedding, txt)

    transformer_options["total_blocks"] = len(self.transformer_blocks)
    transformer_options["block_type"] = "double"
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.transformer_blocks), device, transformer_options)
    comfy.model_prefetch.malloc_graph_begin(device)
    for i, block in enumerate(self.transformer_blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
        transformer_options["block_index"] = i
        img, txt = block(img, txt, vec, pe, transformer_options)
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
    comfy.model_prefetch.malloc_graph_end()

    img = self.proj_out(self.norm_out(img, vec))
    # ===================== SP GATHER ===================== #
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    pt, ph, pw = self.patch_size
    img = img.view(B, t, h, w, self.out_channels, pt, ph, pw).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, self.out_channels, t * pt, h * ph, w * pw)
    return img[:, :, :T, :H, :W]
