import torch
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

from comfy.ldm.flux.math import apply_rope
from comfy.ldm.lens.model import _lens_position_ids
from ..utils import pad_to_world_size
import raylight.distributed_modules.attention as xfuser_attn

attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def _pad_and_chunk(tensor, dim=1):
    tensor, orig_size = pad_to_world_size(tensor, dim=dim)
    tensor = torch.chunk(tensor, get_sequence_parallel_world_size(), dim=dim)[get_sequence_parallel_rank()]
    return tensor, orig_size


def usp_lens_attention_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    freqs_cis,
    attention_mask=None,
    transformer_options=None,
):
    bsz, seq_img, _ = hidden_states.shape
    seq_txt = encoder_hidden_states.shape[1]

    img_qkv = self.img_qkv(hidden_states).view(bsz, seq_img, 3, self.heads, self.dim_head)
    img_q, img_k, img_v = img_qkv.unbind(dim=2)
    img_q = self.norm_q(img_q)
    img_k = self.norm_k(img_k)
    del img_qkv

    txt_qkv = self.txt_qkv(encoder_hidden_states).view(bsz, seq_txt, 3, self.heads, self.dim_head)
    txt_q, txt_k, txt_v = txt_qkv.unbind(dim=2)
    txt_q = self.norm_added_q(txt_q)
    txt_k = self.norm_added_k(txt_k)
    del txt_qkv

    if isinstance(freqs_cis, (list, tuple)):
        freqs_img, freqs_txt = freqs_cis
    else:
        freqs_img = freqs_cis[:, :seq_img]
        freqs_txt = freqs_cis[:, seq_img:seq_img + seq_txt]

    img_q, img_k = apply_rope(img_q, img_k, freqs_img)
    txt_q, txt_k = apply_rope(txt_q, txt_k, freqs_txt)

    q = torch.cat([img_q, txt_q], dim=1).flatten(start_dim=2)
    k = torch.cat([img_k, txt_k], dim=1).flatten(start_dim=2)
    v = torch.cat([img_v, txt_v], dim=1).flatten(start_dim=2)

    out = xfuser_optimized_attention(q, k, v, self.heads, mask=None)
    img_out = self.to_out[1](self.to_out[0](out[:, :seq_img, :]))
    txt_out = self.to_add_out(out[:, seq_img:, :])
    return img_out, txt_out


def usp_lens_forward(
    self,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    attention_mask=None,
    transformer_options=None,
    control=None,
    **kwargs,
) -> torch.Tensor:
    if transformer_options is None:
        transformer_options = {}
    transformer_options = transformer_options.copy()
    patches = transformer_options.get("patches", {})
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})

    B, C, h, w = x.shape
    hidden_states = x.permute(0, 2, 3, 1).reshape(B, h * w, C)

    if self.multi_layer_encoder_feature:
        L = len(self.selected_layer_index)
        enc_dim = context.shape[-1] // L
        encoder_hidden_states = list(context.reshape(B, -1, L, enc_dim).unbind(dim=2))
        text_seq_len = encoder_hidden_states[0].shape[1]
    else:
        encoder_hidden_states = context
        text_seq_len = context.shape[1]

    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)

    if self.multi_layer_encoder_feature:
        normed = [self.txt_norm[i](encoder_hidden_states[i]) for i in range(L)]
        encoder_hidden_states = torch.cat(normed, dim=-1)
    else:
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if "post_input" in patches:
        for p in patches["post_input"]:
            out = p({
                "img": hidden_states,
                "txt": encoder_hidden_states,
                "transformer_options": transformer_options,
            })
            hidden_states = out["img"]
            encoder_hidden_states = out["txt"]

    temb = self.time_text_embed(timestep, hidden_states)
    img_len = hidden_states.shape[1]
    ids = _lens_position_ids(1, h, w, text_seq_len, device=hidden_states.device).unsqueeze(0)
    freqs_cis = self.pos_embed(ids)
    freqs_img = freqs_cis[:, :img_len]
    freqs_txt = freqs_cis[:, img_len:]

    hidden_states, hidden_states_orig_size = _pad_and_chunk(hidden_states, dim=1)
    encoder_hidden_states, _ = _pad_and_chunk(encoder_hidden_states, dim=1)
    freqs_img, _ = _pad_and_chunk(freqs_img, dim=1)
    freqs_txt, _ = _pad_and_chunk(freqs_txt, dim=1)
    freqs_payload = [freqs_img, freqs_txt]

    sp_rank = get_sequence_parallel_rank()
    sp_world_size = get_sequence_parallel_world_size()

    transformer_options["total_blocks"] = len(self.transformer_blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.transformer_blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["txt"], out["img"] = block(
                    hidden_states=args["img"],
                    encoder_hidden_states=args["txt"],
                    temb=args["vec"],
                    freqs_cis=args["pe"],
                    attention_mask=None,
                    transformer_options=args.get("transformer_options"),
                )
                return out

            out = blocks_replace[("double_block", i)](
                {
                    "img": hidden_states,
                    "txt": encoder_hidden_states,
                    "vec": temb,
                    "pe": freqs_payload,
                    "attn_mask": None,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )
            encoder_hidden_states = out["txt"]
            hidden_states = out["img"]
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                freqs_cis=freqs_payload,
                attention_mask=None,
                transformer_options=transformer_options,
            )

        if "double_block" in patches:
            for p in patches["double_block"]:
                out = p({
                    "img": hidden_states,
                    "txt": encoder_hidden_states,
                    "x": x,
                    "block_index": i,
                    "transformer_options": transformer_options,
                })
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]

        if control is not None:
            control_i = control.get("input")
            if control_i is not None and i < len(control_i):
                add = control_i[i]
                if add is not None:
                    add, _ = pad_to_world_size(add, dim=1)
                    add = torch.chunk(add, sp_world_size, dim=1)[sp_rank]
                    hidden_states[:, :add.shape[1]] += add

    hidden_states = get_sp_group().all_gather(hidden_states.contiguous(), dim=1)
    hidden_states = hidden_states[:, :hidden_states_orig_size, :]

    hidden_states = self.norm_out(hidden_states, temb)
    out = self.proj_out(hidden_states)
    return out.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()
