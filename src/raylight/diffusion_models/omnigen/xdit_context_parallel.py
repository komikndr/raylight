import torch
from einops import rearrange
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

import comfy.ldm.common_dit
import comfy.model_management
import raylight.distributed_modules.attention as xfuser_attn
from comfy.ldm.omnigen.omnigen2 import apply_rotary_emb
from ..utils import pad_to_world_size


attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def _split_sequence(tensor):
    tensor, orig_size = pad_to_world_size(tensor, dim=1)
    tensor = torch.chunk(tensor, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    return tensor, orig_size


def _gather_sequence(tensor, orig_size):
    tensor = get_sp_group().all_gather(tensor.contiguous(), dim=1)
    return tensor[:, :orig_size]


def _run_refiner_layers(hidden_states, rotary_emb, layers, temb=None, transformer_options={}):
    hidden_states, orig_size = _split_sequence(hidden_states)
    rotary_emb, _ = _split_sequence(rotary_emb)

    for layer in layers:
        hidden_states = layer(hidden_states, None, rotary_emb, temb, transformer_options=transformer_options)

    return _gather_sequence(hidden_states, orig_size)


def usp_dit_forward(
    self,
    x,
    timesteps,
    context,
    num_tokens,
    ref_latents=None,
    attention_mask=None,
    transformer_options={},
    **kwargs,
):
    _, _, H, W = x.shape
    hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
    _, _, H_padded, W_padded = hidden_states.shape
    timestep = 1.0 - timesteps
    text_hidden_states = context
    ref_image_hidden_states = ref_latents
    device = hidden_states.device

    temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states[0].dtype)

    (
        hidden_states,
        ref_image_hidden_states,
        _img_mask,
        _ref_img_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
    ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

    (
        context_rotary_emb,
        ref_img_rotary_emb,
        noise_rotary_emb,
        rotary_emb,
        _encoder_seq_lengths,
        _seq_lengths,
    ) = self.rope_embedder(
        hidden_states.shape[0],
        text_hidden_states.shape[1],
        [num_tokens] * text_hidden_states.shape[0],
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
        device,
    )

    # ===================== SP SPLIT ====================== #
    text_hidden_states = _run_refiner_layers(
        text_hidden_states,
        context_rotary_emb,
        self.context_refiner,
        transformer_options=transformer_options,
    )
    # ===================== SP GATHER ===================== #

    img_len = hidden_states.shape[1]
    hidden_states = self.x_embedder(hidden_states)
    # ===================== SP SPLIT ====================== #
    hidden_states = _run_refiner_layers(
        hidden_states,
        noise_rotary_emb,
        self.noise_refiner,
        temb,
        transformer_options=transformer_options,
    )
    # ===================== SP GATHER ===================== #

    if ref_image_hidden_states is not None:
        ref_image_hidden_states = self.ref_image_patch_embedder(ref_image_hidden_states)
        image_index_embedding = comfy.model_management.cast_to(
            self.image_index_embedding,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for i in range(len(ref_image_hidden_states)):
            shift = 0
            for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
                ref_image_hidden_states[i, shift:shift + ref_img_len] += image_index_embedding[j]
                shift += ref_img_len

        # ===================== SP SPLIT ====================== #
        ref_image_hidden_states = _run_refiner_layers(
            ref_image_hidden_states,
            ref_img_rotary_emb,
            self.ref_image_refiner,
            temb,
            transformer_options=transformer_options,
        )
        # ===================== SP GATHER ===================== #
        hidden_states = torch.cat([ref_image_hidden_states, hidden_states], dim=1)

    hidden_states = torch.cat([text_hidden_states, hidden_states], dim=1)

    # ===================== SP SPLIT ====================== #
    hidden_states, hidden_states_orig_size = _split_sequence(hidden_states)
    rotary_emb, _ = _split_sequence(rotary_emb)

    for layer in self.layers:
        hidden_states = layer(hidden_states, None, rotary_emb, temb, transformer_options=transformer_options)

    # ===================== SP GATHER ===================== #
    hidden_states = _gather_sequence(hidden_states, hidden_states_orig_size)
    hidden_states = self.norm_out(hidden_states, temb)

    p = self.patch_size
    output = rearrange(
        hidden_states[:, -img_len:],
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h=H_padded // p,
        w=W_padded // p,
        p1=p,
        p2=p,
    )[:, :, :H, :W]

    return -output


def usp_attention_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    attention_mask=None,
    image_rotary_emb=None,
    transformer_options={},
    **kwargs,
):
    batch_size = hidden_states.shape[0]

    query = self.to_q(hidden_states)
    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    query = query.view(batch_size, -1, self.heads, self.dim_head)
    key = key.view(batch_size, -1, self.kv_heads, self.dim_head)
    value = value.view(batch_size, -1, self.kv_heads, self.dim_head)

    query = self.norm_q(query)
    key = self.norm_k(key)

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if self.kv_heads < self.heads:
        key = key.repeat_interleave(self.heads // self.kv_heads, dim=1)
        value = value.repeat_interleave(self.heads // self.kv_heads, dim=1)

    hidden_states = xfuser_optimized_attention(
        query,
        key,
        value,
        self.heads,
        mask=None,
        skip_reshape=True,
    )
    return self.to_out[0](hidden_states)
