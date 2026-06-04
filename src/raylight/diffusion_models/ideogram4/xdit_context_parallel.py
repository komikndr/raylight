import torch
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

import raylight.distributed_modules.attention as xfuser_attn
from comfy.ldm.ideogram4.model import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
from comfy.text_encoders.llama import apply_rope, precompute_freqs_cis

from ..utils import pad_to_world_size


attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def usp_attention_forward(self, x, attn_mask, freqs_cis, transformer_options={}):
    batch_size, seq_len, _ = x.shape
    qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    q, k, v = qkv.unbind(dim=2)

    q = self.norm_q(q)
    k = self.norm_k(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q, k = apply_rope(q, k, freqs_cis)

    out = xfuser_optimized_attention(
        q,
        k,
        v,
        self.num_heads,
        mask=attn_mask,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    return self.o(out)


def _run_usp_backbone(self, h, position_ids, adaln_input, transformer_options):
    h, orig_size = pad_to_world_size(h, dim=1)
    position_ids, _ = pad_to_world_size(position_ids, dim=1)

    sp_rank = get_sequence_parallel_rank()
    sp_world_size = get_sequence_parallel_world_size()
    h = torch.chunk(h, sp_world_size, dim=1)[sp_rank]
    position_ids = torch.chunk(position_ids, sp_world_size, dim=1)[sp_rank]

    freqs_cis = precompute_freqs_cis(
        self.head_dim,
        position_ids[0].transpose(0, 1),
        self.rope_theta,
        rope_dims=self.mrope_section,
        interleaved_mrope=True,
        device=position_ids.device,
    )

    transformer_options["total_blocks"] = len(self.layers)
    transformer_options["block_type"] = "single"
    for i, layer in enumerate(self.layers):
        transformer_options["block_index"] = i
        h = layer(h, None, freqs_cis, adaln_input, transformer_options=transformer_options)

    h = get_sp_group().all_gather(h.contiguous(), dim=1)
    return h[:, :orig_size]


def usp_dit_forward(self, x, timesteps, context=None, attention_mask=None, transformer_options={}, **kwargs):
    bs, c, gh, gw = x.shape
    timesteps = 1.0 - timesteps

    t_cond = self.t_embedding(timesteps)
    if timesteps.dim() == 1:
        t_cond = t_cond.unsqueeze(1)
    adaln_input = torch.nn.functional.silu(self.adaln_proj(t_cond))

    img_tokens = self._img_to_tokens(x).to(self.dtype)
    latent_dim = img_tokens.shape[-1]
    device = x.device

    if context is None:
        seq_len = img_tokens.shape[1]
        position_ids = self._image_position_ids(gh, gw, device).unsqueeze(0).expand(bs, seq_len, 3)
        h = self.input_proj(img_tokens)
        h = h + self.embed_image_indicator(torch.ones((bs, seq_len), dtype=torch.long, device=device))

        h = _run_usp_backbone(self, h, position_ids, adaln_input, transformer_options)
        out = self.final_layer(h, adaln_input)
        return -self._tokens_to_img(out, gh, gw)

    l_text = context.shape[1]
    l_img = img_tokens.shape[1]
    seq_len = l_text + l_img

    x_full = torch.zeros(bs, seq_len, latent_dim, dtype=img_tokens.dtype, device=device)
    x_full[:, l_text:] = img_tokens

    text_pos = torch.arange(l_text, device=device).view(-1, 1).expand(l_text, 3)
    img_pos = self._image_position_ids(gh, gw, device)
    position_ids = torch.cat([text_pos, img_pos], dim=0).unsqueeze(0).expand(bs, seq_len, 3)

    indicator = torch.empty(bs, seq_len, dtype=torch.long, device=device)
    indicator[:, :l_text] = LLM_TOKEN_INDICATOR
    indicator[:, l_text:] = OUTPUT_IMAGE_INDICATOR

    if attention_mask is not None:
        pad = attention_mask == 0
        indicator[:, :l_text][pad] = 0

    output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x_full.dtype).unsqueeze(-1)
    h = self.input_proj(x_full) * output_image_mask

    text_mask = (indicator[:, :l_text] == LLM_TOKEN_INDICATOR).to(x_full.dtype).unsqueeze(-1)
    llm = self.llm_cond_norm(context * text_mask)
    llm = self.llm_cond_proj(llm) * text_mask
    h[:, :l_text] = h[:, :l_text] + llm

    h = h + self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
    h = _run_usp_backbone(self, h, position_ids, adaln_input, transformer_options)
    out = self.final_layer(h, adaln_input)
    return -self._tokens_to_img(out[:, l_text:], gh, gw)
