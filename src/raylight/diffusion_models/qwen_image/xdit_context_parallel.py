import torch
from torch import Tensor
from typing import Optional, Tuple
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from ..utils import pad_to_world_size
from comfy.ldm.flux.math import apply_rope
import raylight.distributed_modules.attention as xfuser_attn
attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)



def usp_dit_forward(
    self,
    x,
    timesteps,
    context,
    attention_mask=None,
    ref_latents=None,
    additional_t_cond=None,
    transformer_options={},
    control=None,
    *args,
    **kwargs
):
    timestep = timesteps
    encoder_hidden_states = context
    encoder_hidden_states_mask = attention_mask

    hidden_states, img_ids, orig_shape = self.process_img(x)
    num_embeds = hidden_states.shape[1]

    timestep_zero_index = None
    ref_num_tokens = None
    if ref_latents is not None:
        ref_num_tokens = []
        h = 0
        w = 0
        index = 0
        ref_method = kwargs.get("ref_latents_method", self.default_ref_method)
        index_ref_method = (ref_method == "index") or (ref_method == "index_timestep_zero")
        negative_ref_method = ref_method == "negative_index"
        timestep_zero = ref_method == "index_timestep_zero"
        for ref in ref_latents:
            if index_ref_method:
                index += 1
                h_offset = 0
                w_offset = 0
            elif negative_ref_method:
                index -= 1
                h_offset = 0
                w_offset = 0
            else:
                index = 1
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

            kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
            hidden_states = torch.cat([hidden_states, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)
            ref_num_tokens.append(kontext.shape[1])
        if timestep_zero:
            if index > 0:
                timestep = torch.cat([timestep, timestep * 0], dim=0)
                timestep_zero_index = num_embeds

    txt_start = round(max(
        ((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2,
        ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2,))

    txt_ids = (
        torch.arange(txt_start, txt_start + context.shape[1], device=x.device)
        .reshape(1, -1, 1)
        .repeat(x.shape[0], 1, 3))

    hidden_states = self.img_in(hidden_states)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)

    if ref_num_tokens is not None:
        transformer_options = transformer_options.copy()
        transformer_options["reference_image_num_tokens"] = ref_num_tokens

    hidden_states = _parallel_blocks(self, x, hidden_states, img_ids, encoder_hidden_states, txt_ids,
                                     encoder_hidden_states_mask, temb, timestep_zero_index, control, transformer_options, x.dtype)
    hidden_states = hidden_states[:, :num_embeds].view(
        orig_shape[0], orig_shape[-3], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2
    )
    hidden_states = hidden_states.permute(0, 4, 1, 2, 5, 3, 6)
    return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]


def usp_mage_forward(self, x, timestep, context, attention_mask=None, ref_latents=None, transformer_options={}, control=None, **kwargs):
    hidden_states, img_ids, (h, w) = self.process_img(x)
    transformer_options = transformer_options.copy()
    if ref_latents is not None:
        ref_sizes = []
        for index, ref in enumerate(ref_latents, 1):
            tokens, ids, _ = self.process_img(ref, index=index)
            hidden_states = torch.cat((hidden_states, tokens), dim=1)
            img_ids = torch.cat((img_ids, ids), dim=1)
            ref_sizes.append(tokens.shape[1])
        transformer_options["reference_image_num_tokens"] = ref_sizes

    if attention_mask is not None and not torch.is_floating_point(attention_mask):
        attention_mask = (attention_mask - 1).to(x.dtype) * torch.finfo(x.dtype).max
    txt_ids = img_ids.new_zeros((x.shape[0], context.shape[1], 3))
    hidden_states = self.img_in(hidden_states)
    temb = self.time_text_embed(timestep, hidden_states)
    hidden_states = _parallel_blocks(self, x, hidden_states, img_ids, self.txt_in(self.txt_norm(context)), txt_ids,
                                     attention_mask, temb, None, control, transformer_options)
    return hidden_states[:, :h * w].reshape(x.shape[0], h, w, self.out_channels).movedim(-1, 1)


def _parallel_blocks(self, x, hidden_states, img_ids, encoder_hidden_states, txt_ids, encoder_hidden_states_mask,
                     temb, timestep_zero_index, control, transformer_options, rope_dtype=None):
    patches = transformer_options.get("patches", {})
    if "post_input" in patches:
        for p in patches["post_input"]:
            out = p({"img": hidden_states, "txt": encoder_hidden_states, "img_ids": img_ids, "txt_ids": txt_ids, "transformer_options": transformer_options})
            hidden_states = out["img"]
            encoder_hidden_states = out["txt"]
            img_ids = out["img_ids"]
            txt_ids = out["txt_ids"]

    image_rotary_emb = [self.pe_embedder(ids).transpose(1, 2).to(dtype=rope_dtype).contiguous()
                        for ids in (img_ids, txt_ids)]

    del txt_ids, img_ids

    # ===================== SP SPLIT ====================== #
    hidden_states, hidden_states_orig_size = pad_to_world_size(hidden_states, dim=1)
    encoder_hidden_states, _ = pad_to_world_size(encoder_hidden_states, dim=1)
    image_rotary_emb[0], _ = pad_to_world_size(image_rotary_emb[0], dim=1)
    image_rotary_emb[1], _ = pad_to_world_size(image_rotary_emb[1], dim=1)

    sp_rank = get_sequence_parallel_rank()
    sp_world_size = get_sequence_parallel_world_size()

    hidden_states = torch.chunk(hidden_states, sp_world_size, dim=1)[sp_rank]
    encoder_hidden_states = torch.chunk(encoder_hidden_states, sp_world_size, dim=1)[sp_rank]
    if timestep_zero_index is not None:
        width = hidden_states.shape[1]
        timestep_zero_index = max(0, min(width, timestep_zero_index - sp_rank * width))

    # Modified, freq rope : List[txt, img]
    image_rotary_emb[0] = torch.chunk(image_rotary_emb[0], sp_world_size, dim=1)[sp_rank]
    image_rotary_emb[1] = torch.chunk(image_rotary_emb[1], sp_world_size, dim=1)[sp_rank]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})

    transformer_options["total_blocks"] = len(self.transformer_blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.transformer_blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["txt"], out["img"] = block(hidden_states=args["img"],
                                               encoder_hidden_states=args["txt"],
                                               encoder_hidden_states_mask=encoder_hidden_states_mask,
                                               temb=args["vec"],
                                               image_rotary_emb=args["pe"],
                                               timestep_zero_index=timestep_zero_index,
                                               transformer_options=args["transformer_options"])
                return out
            out = blocks_replace[("double_block", i)]({"img": hidden_states,
                                                       "txt": encoder_hidden_states,
                                                       "vec": temb,
                                                       "pe": image_rotary_emb,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap})
            hidden_states = out["img"]
            encoder_hidden_states = out["txt"]
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                timestep_zero_index=timestep_zero_index,
                transformer_options=transformer_options,
            )

        if "double_block" in patches:
            for p in patches["double_block"]:
                out = p({"img": hidden_states,
                         "txt": encoder_hidden_states,
                         "x": x,
                         "block_index": i,
                         "transformer_options": transformer_options})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    # Pad + chunk control signal to match USP sequence split
                    start = sp_rank * hidden_states.shape[1]
                    add = add[:, start:start + hidden_states.shape[1]]
                    hidden_states[:, :add.shape[1]] += add

    if timestep_zero_index is not None:
        temb = temb.chunk(2, dim=0)[0]

    # ===================== SP GATHER ===================== #
    hidden_states = get_sp_group().all_gather(hidden_states.contiguous(), dim=1)
    hidden_states = hidden_states[:, :hidden_states_orig_size, :]

    hidden_states = self.norm_out(hidden_states, temb)
    return self.proj_out(hidden_states)


def usp_attn_forward(
    self,
    hidden_states: torch.FloatTensor,  # Image stream
    encoder_hidden_states: torch.FloatTensor = None,  # Text stream
    encoder_hidden_states_mask: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_txt = encoder_hidden_states.shape[1]
    transformer_options = kwargs.get("transformer_options", {})
    transformer_patches = transformer_options.get("patches", {})
    extra_options = transformer_options.copy()

    img_query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
    img_key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
    img_value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

    txt_query = self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
    txt_key = self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
    txt_value = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))

    img_query = self.norm_q(img_query)
    img_key = self.norm_k(img_key)

    txt_query = self.norm_added_q(txt_query)
    txt_key = self.norm_added_k(txt_key)

    joint_query = torch.cat([txt_query, img_query], dim=1)
    joint_key = torch.cat([txt_key, img_key], dim=1)
    joint_value = torch.cat([txt_value, img_value], dim=1)

    extra_options["img_slice"] = [txt_query.shape[1], joint_query.shape[1]]
    if "attn1_patch" in transformer_patches:
        patch = transformer_patches["attn1_patch"]
        for p in patch:
            out = p(joint_query, joint_key, joint_value, pe=image_rotary_emb, attn_mask=encoder_hidden_states_mask, extra_options=extra_options)
            joint_query, joint_key, joint_value, image_rotary_emb, encoder_hidden_states_mask = out.get("q", joint_query), out.get("k", joint_key), out.get("v", joint_value), out.get("pe", image_rotary_emb), out.get("attn_mask", encoder_hidden_states_mask)

    if image_rotary_emb is not None:
        txt_len = seq_txt
        txt_query, img_query = joint_query[:, :txt_len], joint_query[:, txt_len:]
        txt_key, img_key = joint_key[:, :txt_len], joint_key[:, txt_len:]
        img_rotary_emb = image_rotary_emb[0]
        txt_rotary_emb = image_rotary_emb[1]
        img_query, img_key = apply_rope(img_query, img_key, img_rotary_emb)
        txt_query, txt_key = apply_rope(txt_query, txt_key, txt_rotary_emb)
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)

    joint_query = joint_query.flatten(start_dim=2)
    joint_key = joint_key.flatten(start_dim=2)
    joint_value = joint_value.flatten(start_dim=2)

    joint_hidden_states = xfuser_optimized_attention(
        joint_query,
        joint_key,
        joint_value,
        self.heads,
        mask=attention_mask,
    )

    txt_attn_output = joint_hidden_states[:, :seq_txt, :]
    img_attn_output = joint_hidden_states[:, seq_txt:, :]

    img_attn_output = self.to_out[0](img_attn_output)
    img_attn_output = self.to_out[1](img_attn_output)
    txt_attn_output = self.to_add_out(txt_attn_output)

    return img_attn_output, txt_attn_output
