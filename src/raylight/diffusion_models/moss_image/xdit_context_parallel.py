import torch

import comfy.model_prefetch
import comfy.utils
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.moss_image.model import MOSSImage, modulated_rms_norm
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

import raylight.distributed_modules.attention as xfuser_attn
from ..utils import pad_to_world_size


xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(xfuser_attn.get_attn_type(), xfuser_attn.get_sync_ulysses())


def usp_attention(original_attention, q, k, v, heads, transformer_options={}, **kwargs):
    return xfuser_optimized_attention(q, k, v, heads, transformer_options=transformer_options, key_lengths=transformer_options["raylight_key_lengths"], **kwargs)


def usp_moss_forward(self, x, timestep, context, transformer_options={}, ref_latents=[], ref_latents_method=None, ref_kv=None, context_attention_bias=None, **kwargs):
    if self.raylight_native_attention or "optimized_attention_override" in transformer_options:
        # checkpoint-selected attention or a caller override already owns the
        # attention hop: run the full native forward on every rank instead of sharding
        return MOSSImage._forward(
            self, x, timestep, context, transformer_options,
            ref_latents=ref_latents, ref_latents_method=ref_latents_method,
            ref_kv=ref_kv, context_attention_bias=context_attention_bias, **kwargs,
        )

    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()
    text_len = context.shape[1]
    if len(ref_latents) > 0 and ref_latents_method not in ("index", "index_timestep_zero"):
        ref_kv, ref_latents = self.preprocess_refs(ref_latents, transformer_options), []
    if ref_kv is not None:
        ref_kv = ref_kv.expand(B, -1, -1, -1, -1) if ref_kv.shape[0] == 1 else comfy.utils.repeat_to_batch_size(ref_kv, B)
    refs = [comfy.utils.repeat_to_batch_size(ref, B) for ref in ref_latents]
    f_offset = -text_len if (len(refs) > 0 or ref_kv is not None) and ref_latents_method != "index_timestep_zero_text_start" else 0
    pe = self.rope(H, W, text_len, device, transformer_options, [ref.shape[-2:] for ref in refs], f_offset)

    vec = self.time_embedding(timestep_embedding(timestep, 256).to(dtype))
    mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    head_shift, head_scale = self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
    ref = None
    if len(refs) > 0 and ref_latents_method != "index":
        ref_vec = self.time_embedding(timestep_embedding(torch.zeros_like(timestep), 256).to(dtype))
        ref = (H * W + text_len, self.shared_block_modulation(ref_vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1))

    # Keep prompt bias in the native text embedder; the shared xFuser path does
    # not consume the joint-block attention bias.
    text_mask = context_attention_bias[:, None, None] if context_attention_bias is not None else None
    txt = self.text_embedder(context, pe[:, H * W:H * W + text_len], transformer_options, text_mask)
    x = torch.cat([self.video_embedder(x.flatten(2).transpose(1, 2)), txt] + [self.video_embedder(r.flatten(2).transpose(1, 2)) for r in refs], dim=1)
    del txt, refs

    transformer_options = dict(transformer_options or {}, raylight_key_lengths=(x.shape[1],) if ref_kv is None else (x.shape[1], ref_kv.shape[3]))
    world_size = get_sequence_parallel_world_size()
    rank = get_sequence_parallel_rank()
    local_len = -(-x.shape[1] // world_size)
    # the t=0 reference modulation follows the tokens it modulates into this rank's window
    split = 0 if ref is None else min(max(ref[0] - rank * local_len, 0), local_len)
    # ===================== SP SPLIT ====================== #
    x, _ = pad_to_world_size(x, dim=1)
    pe, _ = pad_to_world_size(pe, dim=1)
    x = x.chunk(world_size, dim=1)[rank].clone()
    pe = pe.chunk(world_size, dim=1)[rank]
    if ref_kv is not None:
        # each rank appends only its reference slice to the local K/V; the ring
        # exposes every slice to every query, so the K/V is never allgathered
        ref_kv, _ = pad_to_world_size(ref_kv, dim=3)
        ref_kv = ref_kv.chunk(world_size, dim=3)[rank].clone()
    if ref is not None:
        if split == 0:
            mod, ref = ref[1], None
        elif split == local_len:
            ref = None
        else:
            ref = (split, ref[1])

    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    transformer_options["total_blocks"] = len(self.layers)
    transformer_options["block_type"] = "single"
    transformer_options["optimized_attention_override"] = usp_attention
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), device, transformer_options)
    comfy.model_prefetch.malloc_graph_begin(device)
    for i, block in enumerate(self.layers):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
        transformer_options["block_index"] = i
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                return {"img": block(args["img"], args["mod"], args["pe"], args["transformer_options"], ref, None if ref_kv is None else ref_kv[:, i])}
            x = blocks_replace[("single_block", i)]({"img": x, "vec": vec, "mod": mod, "pe": pe, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
        else:
            x = block(x, mod, pe, transformer_options, ref, None if ref_kv is None else ref_kv[:, i])
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
    comfy.model_prefetch.malloc_graph_end()

    # ===================== SP GATHER ===================== #
    x = get_sp_group().all_gather(x.contiguous(), dim=1)
    out = self.final_linear_video(modulated_rms_norm(x[:, :H * W], self.final_norm_video, head_scale, head_shift))
    return out.transpose(1, 2).reshape(B, self.latent_channels, H, W)
