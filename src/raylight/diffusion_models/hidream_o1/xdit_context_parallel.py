import einops
import torch

import comfy.ops
from comfy.ldm.hidream_o1.model import IMAGE_TOKEN_ID
import raylight.distributed_modules.attention as xfuser_attn
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

from ..utils import pad_to_world_size

attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def _split_suffix(t, prefix_len, dim=1):
    prefix = t.narrow(dim, 0, prefix_len)
    suffix = t.narrow(dim, prefix_len, t.shape[dim] - prefix_len)
    suffix, orig_size = pad_to_world_size(suffix, dim=dim)
    suffix = torch.chunk(suffix, get_sequence_parallel_world_size(), dim=dim)[get_sequence_parallel_rank()]
    return torch.cat([prefix, suffix], dim=dim), orig_size


def _split_freqs_suffix(freqs_cis, prefix_len):
    out = []
    orig_size = None
    for freq in freqs_cis:
        prefix = freq[..., :prefix_len, :]
        suffix = freq[..., prefix_len:, :]
        suffix, this_orig_size = pad_to_world_size(suffix, dim=-2)
        suffix = torch.chunk(suffix, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
        out.append(torch.cat([prefix, suffix], dim=-2))
        orig_size = this_orig_size
    return tuple(out), orig_size


def usp_two_pass_attention(ar_len: int, transformer_options=None):
    def two_pass_attention(q, k, v, heads, **kwargs):
        B, H, T, D = q.shape

        if T < k.shape[2]:
            out = comfy.ops.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        elif ar_len >= T:
            out = comfy.ops.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        elif ar_len <= 0:
            out = xfuser_optimized_attention(
                q, k, v, heads,
                skip_reshape=True,
                skip_output_reshape=True,
            )
        else:
            out_ar = comfy.ops.scaled_dot_product_attention(
                q[:, :, :ar_len], k[:, :, :ar_len], v[:, :, :ar_len],
                attn_mask=None, dropout_p=0.0, is_causal=True,
            )
            full_k_gen = get_sp_group().all_gather(k[:, :, ar_len:].contiguous(), dim=2)
            full_v_gen = get_sp_group().all_gather(v[:, :, ar_len:].contiguous(), dim=2)
            full_k = torch.cat([k[:, :, :ar_len], full_k_gen], dim=2)
            full_v = torch.cat([v[:, :, :ar_len], full_v_gen], dim=2)
            out_gen = xfuser_optimized_attention(
                q[:, :, ar_len:], full_k, full_v, heads,
                skip_reshape=True,
                skip_output_reshape=True,
            )
            out = torch.cat([out_ar, out_gen], dim=2)

        return out.transpose(1, 2).reshape(B, T, H * D)

    return two_pass_attention


def usp_dit_forward(self, x, timesteps, context=None, transformer_options={}, input_ids=None, attention_mask=None,
                    position_ids=None, vinput_mask=None, ar_len=None, ref_pixel_values=None,
                    ref_image_grid_thw=None, ref_patches=None, **kwargs):
    if input_ids is None or position_ids is None:
        raise ValueError("HiDreamO1Transformer requires input_ids and position_ids in conditioning")

    B, _, H, W = x.shape
    h_p, w_p = H // self.patch_size, W // self.patch_size
    tgt_image_len = h_p * w_p

    z = einops.rearrange(
        x, 'B C (H p1) (W p2) -> B (H W) (C p1 p2)',
        p1=self.patch_size, p2=self.patch_size,
    )
    vinputs = torch.cat([z, ref_patches.to(z.dtype)], dim=1) if ref_patches is not None else z

    inputs_embeds = self.language_model.embed_tokens(input_ids).to(x.dtype)

    if ref_pixel_values is not None and ref_image_grid_thw is not None:
        cached = self._visual_cache
        if cached is not None and cached[0] is ref_pixel_values:
            image_embeds = cached[1]
        else:
            ref_pv = ref_pixel_values.to(inputs_embeds.device)
            ref_grid = ref_image_grid_thw.to(inputs_embeds.device).long()
            if ref_pv.dim() == 3:
                ref_pv = ref_pv[0]
            if ref_grid.dim() == 3:
                ref_grid = ref_grid[0]
            image_embeds = self.visual(ref_pv, ref_grid).to(inputs_embeds.dtype)
            self._visual_cache = (ref_pixel_values, image_embeds)
        image_idx = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        if image_idx.shape[0] != image_embeds.shape[0]:
            raise ValueError(
                f"Image-token count {image_idx.shape[0]} != ViT output count "
                f"{image_embeds.shape[0]}; check tokenizer/processor alignment."
            )
        inputs_embeds[:, image_idx] = image_embeds.unsqueeze(0).expand(B, -1, -1)

    sigma = timesteps.float() / 1000.0
    t_pixeldit = 1.0 - sigma
    t_emb = self.t_embedder1(t_pixeldit * 1000, inputs_embeds.dtype)
    tms_mask_3d = (input_ids == self.tms_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = torch.where(tms_mask_3d, t_emb.unsqueeze(1).expand_as(inputs_embeds), inputs_embeds)

    vinputs_embedded = self.x_embedder(vinputs.to(inputs_embeds.dtype))
    inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)

    freqs_cis = self.language_model.compute_freqs_cis(position_ids[0].to(x.device), x.device)
    freqs_cis = tuple(t.to(x.dtype) for t in freqs_cis)

    prefix_len = int(ar_len.item() if torch.is_tensor(ar_len) else (ar_len or 0))
    inputs_embeds, gen_orig_size = _split_suffix(inputs_embeds, prefix_len, dim=1)
    freqs_cis, _ = _split_freqs_suffix(freqs_cis, prefix_len)

    two_pass_attn = usp_two_pass_attention(prefix_len, transformer_options=transformer_options)
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.language_model.layers)
    transformer_options["block_type"] = "double"

    hidden_states = inputs_embeds
    for i, layer in enumerate(self.language_model.layers):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args, _layer=layer):
                out = {}
                out["x"], _ = _layer(
                    x=args["x"], attention_mask=args.get("attention_mask"),
                    freqs_cis=args["freqs_cis"], optimized_attention=args["optimized_attention"],
                    past_key_value=None,
                )
                return out
            out = blocks_replace[("double_block", i)](
                {"x": hidden_states, "attention_mask": None,
                 "freqs_cis": freqs_cis, "optimized_attention": two_pass_attn,
                 "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            hidden_states = out["x"]
        else:
            hidden_states, _ = layer(
                x=hidden_states, attention_mask=None,
                freqs_cis=freqs_cis, optimized_attention=two_pass_attn,
                past_key_value=None,
            )

    full_gen = get_sp_group().all_gather(hidden_states[:, prefix_len:].contiguous(), dim=1)[:, :gen_orig_size]
    hidden_states = torch.cat([hidden_states[:, :prefix_len], full_gen], dim=1)

    if self.language_model.norm is not None:
        hidden_states = self.language_model.norm(hidden_states)

    if vinput_mask is not None:
        vmask = vinput_mask.to(x.device).bool()
        target_hidden = hidden_states[vmask].view(B, -1, hidden_states.shape[-1])[:, :tgt_image_len]
    else:
        txt_seq_len = input_ids.shape[1]
        target_hidden = hidden_states[:, txt_seq_len:txt_seq_len + tgt_image_len]
    x_pred_tgt = self.final_layer2(target_hidden)

    x_pred_img = einops.rearrange(
        x_pred_tgt, 'B (H W) (C p1 p2) -> B C (H p1) (W p2)',
        H=h_p, W=w_p, p1=self.patch_size, p2=self.patch_size,
    )
    return (x.float() - x_pred_img.float()) / sigma.view(B, 1, 1, 1).clamp_min(1e-3)
