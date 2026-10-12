import torch
from einops import rearrange

import comfy.ldm.common_dit
import comfy.model_management
from comfy.ldm.omnigen.omnigen2 import apply_rotary_emb
from ..omnigen.xdit_context_parallel import (
    _gather_sequence,
    _run_refiner_layers,
    _split_sequence,
    usp_attention_forward,
    xfuser_optimized_attention,
)


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

        for i in range(hidden_states.shape[0]):
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

    combined_img_hidden_states = hidden_states
    instruct_size = text_hidden_states.shape[1]
    combined_img_rotary_emb = rotary_emb[:, instruct_size:]
    instruct_rotary_emb = rotary_emb[:, :instruct_size]

    # ===================== SP SPLIT ====================== #
    text_hidden_states, text_orig_size = _split_sequence(text_hidden_states)
    combined_img_hidden_states, img_orig_size = _split_sequence(combined_img_hidden_states)
    instruct_rotary_emb, _ = _split_sequence(instruct_rotary_emb)
    combined_img_rotary_emb, _ = _split_sequence(combined_img_rotary_emb)
    local_rotary_emb = torch.cat([instruct_rotary_emb, combined_img_rotary_emb], dim=1)

    for layer in self.double_stream_layers:
        combined_img_hidden_states, text_hidden_states = layer(
            combined_img_hidden_states,
            text_hidden_states,
            local_rotary_emb,
            combined_img_rotary_emb,
            temb,
            joint_attention_mask=None,
            img_attention_mask=None,
            transformer_options=transformer_options,
        )

    # ===================== SP GATHER ===================== #
    text_hidden_states = _gather_sequence(text_hidden_states, text_orig_size)
    combined_img_hidden_states = _gather_sequence(combined_img_hidden_states, img_orig_size)

    hidden_states = torch.cat([text_hidden_states, combined_img_hidden_states], dim=1)
    # ===================== SP SPLIT ====================== #
    hidden_states, hidden_states_orig_size = _split_sequence(hidden_states)
    rotary_emb, _ = _split_sequence(rotary_emb)

    for layer in self.single_stream_layers:
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


def usp_joint_attention_forward(
    self,
    img_hidden_states,
    instruct_hidden_states,
    rotary_emb,
    attention_mask=None,
    transformer_options={},
    **kwargs,
):
    batch_size = img_hidden_states.shape[0]
    instruct_size = instruct_hidden_states.shape[1]
    processor = self.processor

    img_q = processor.img_to_q(img_hidden_states)
    img_k = processor.img_to_k(img_hidden_states)
    img_v = processor.img_to_v(img_hidden_states)

    instruct_q = processor.instruct_to_q(instruct_hidden_states)
    instruct_k = processor.instruct_to_k(instruct_hidden_states)
    instruct_v = processor.instruct_to_v(instruct_hidden_states)

    query = torch.cat([instruct_q, img_q], dim=1)
    key = torch.cat([instruct_k, img_k], dim=1)
    value = torch.cat([instruct_v, img_v], dim=1)

    query = query.view(batch_size, -1, self.heads, self.dim_head)
    key = key.view(batch_size, -1, self.kv_heads, self.dim_head)
    value = value.view(batch_size, -1, self.kv_heads, self.dim_head)

    query = self.norm_q(query)
    key = self.norm_k(key)

    if rotary_emb is not None:
        query = apply_rotary_emb(query, rotary_emb)
        key = apply_rotary_emb(key, rotary_emb)

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

    instruct_hidden_states = processor.instruct_out(hidden_states[:, :instruct_size])
    img_hidden_states = processor.img_out(hidden_states[:, instruct_size:])
    hidden_states = torch.cat([instruct_hidden_states, img_hidden_states], dim=1)
    return self.to_out[0](hidden_states)


def usp_img_self_attention_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    attention_mask=None,
    image_rotary_emb=None,
    transformer_options={},
    **kwargs,
):
    return usp_attention_forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        image_rotary_emb,
        transformer_options,
    )


def usp_double_stream_forward(
    self,
    img_hidden_states,
    instruct_hidden_states,
    joint_rotary_emb,
    img_rotary_emb,
    temb,
    joint_attention_mask=None,
    img_attention_mask=None,
    transformer_options={},
    **kwargs,
):
    instruct_size = instruct_hidden_states.shape[1]

    img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(img_hidden_states, temb)
    img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
    img_norm3_out, img_gate_self, _, _ = self.img_norm3(img_hidden_states, temb)

    instruct_norm1_out, instruct_gate_msa, instruct_scale_mlp, instruct_gate_mlp = self.instruct_norm1(instruct_hidden_states, temb)
    instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(instruct_hidden_states, temb)

    joint_attn_out = self.img_instruct_attn(
        img_norm1_out,
        instruct_norm1_out,
        joint_rotary_emb,
        joint_attention_mask,
        transformer_options=transformer_options,
    )
    instruct_attn_out = joint_attn_out[:, :instruct_size]
    img_attn_out = joint_attn_out[:, instruct_size:]

    img_self_attn_out = self.img_self_attn(
        img_norm3_out,
        img_norm3_out,
        img_attention_mask,
        img_rotary_emb,
        transformer_options=transformer_options,
    )

    img_hidden_states = img_hidden_states + img_gate_msa.unsqueeze(1).tanh() * self.img_attn_norm(img_attn_out)
    img_hidden_states = img_hidden_states + img_gate_self.unsqueeze(1).tanh() * self.img_self_attn_norm(img_self_attn_out)
    img_mlp_input = (1 + img_scale_mlp.unsqueeze(1)) * img_norm2_out + img_shift_mlp.unsqueeze(1)
    img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
    img_hidden_states = img_hidden_states + img_gate_mlp.unsqueeze(1).tanh() * self.img_ffn_norm2(img_mlp_out)

    instruct_hidden_states = instruct_hidden_states + instruct_gate_msa.unsqueeze(1).tanh() * self.instruct_attn_norm(instruct_attn_out)
    instruct_mlp_input = (1 + instruct_scale_mlp.unsqueeze(1)) * instruct_norm2_out + instruct_shift_mlp.unsqueeze(1)
    instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_mlp_input))
    instruct_hidden_states = instruct_hidden_states + instruct_gate_mlp.unsqueeze(1).tanh() * self.instruct_ffn_norm2(instruct_mlp_out)

    return img_hidden_states, instruct_hidden_states


usp_img_self_attention_forward = usp_attention_forward
