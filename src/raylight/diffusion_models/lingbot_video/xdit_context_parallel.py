import torch

from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.lingbot_video.model import make_joint_position_ids
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

from ..utils import pad_to_world_size
import raylight.distributed_modules.attention as xfuser_attn


attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def usp_attention_forward(self, x, rotary_emb, attention_mask=None, transformer_options={}, **kwargs):
    q = self.to_q(x).unflatten(2, (self.num_heads, self.head_dim))
    k = self.to_k(x).unflatten(2, (self.num_heads, self.head_dim))
    v = self.to_v(x).unflatten(2, (self.num_heads, self.head_dim))
    q = apply_rope1(self.norm_q(q), rotary_emb)
    k = apply_rope1(self.norm_k(k), rotary_emb)
    out = xfuser_optimized_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        heads=self.num_heads,
        mask=attention_mask,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    return self.to_out(out)


def usp_dit_forward(
    self,
    hidden_states,
    timestep,
    context=None,
    encoder_attention_mask=None,
    attention_mask=None,
    transformer_options={},
    **kwargs,
):
    encoder_hidden_states = context
    if encoder_hidden_states is None:
        raise ValueError("LingBotVideo requires text conditioning.")
    if encoder_attention_mask is None:
        encoder_attention_mask = attention_mask

    B, C, T, H, W = hidden_states.shape
    pF, pH, pW = self.patch_size
    gt, gh, gw = T // pF, H // pH, W // pW
    n_video = gt * gh * gw
    L = encoder_hidden_states.shape[1]
    device = hidden_states.device
    if encoder_attention_mask is not None:
        text_lens = encoder_attention_mask.sum(dim=-1).long()
    else:
        text_lens = torch.full((B,), L, dtype=torch.long, device=device)
    text_lens_list = [int(v) for v in text_lens.detach().cpu().tolist()]

    patch_tokens = hidden_states.reshape(B, C, gt, pF, gh, pH, gw, pW)
    patch_tokens = patch_tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, n_video, pF * pH * pW * C)
    x = self.patch_embedder(patch_tokens)
    text = self.text_embedder(encoder_hidden_states)
    joint = torch.cat([x, text], dim=1)
    joint_seq_len = joint.shape[1]

    rotary = torch.stack(
        [self.rope(make_joint_position_ids(text_lens_list[i], gt, gh, gw, device, L)) for i in range(B)],
        dim=0,
    ).unsqueeze(2)

    key_mask = None
    if encoder_attention_mask is not None and bool((text_lens < L).any()):
        key_mask = torch.cat(
            [torch.ones(B, n_video, dtype=torch.bool, device=device), encoder_attention_mask.bool()],
            dim=1,
        )

    timestep_proj = timestep_embedding(timestep.to(hidden_states.dtype), self.freq_dim, time_factor=1.0)
    t_emb = self.time_embedder(timestep_proj)
    temb_input = t_emb.unsqueeze(1).expand(B, joint_seq_len, -1)

    joint, joint_orig_size = pad_to_world_size(joint, dim=1)
    rotary, _ = pad_to_world_size(rotary, dim=1)
    local_temb_input, _ = pad_to_world_size(temb_input, dim=1)
    if key_mask is not None:
        key_mask, _ = pad_to_world_size(key_mask, dim=1)

    sp_world_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    # ===================== SP SPLIT ====================== #
    joint = torch.chunk(joint, sp_world_size, dim=1)[sp_rank]
    rotary = torch.chunk(rotary, sp_world_size, dim=1)[sp_rank]
    local_temb_input = torch.chunk(local_temb_input, sp_world_size, dim=1)[sp_rank]
    moe_padding_mask = None if key_mask is None else torch.chunk(key_mask, sp_world_size, dim=1)[sp_rank].reshape(-1)
    temb6 = self.time_modulation(local_temb_input.reshape(-1, local_temb_input.shape[-1]))

    # xFuser does not support LingBot's arbitrary joint attention mask after sharding.
    for block in self.blocks:
        joint = block(
            joint,
            temb6,
            rotary,
            moe_padding_mask=moe_padding_mask,
            transformer_options=transformer_options,
        )

    # ===================== SP GATHER ===================== #
    joint = get_sp_group().all_gather(joint.contiguous(), dim=1)
    joint = joint[:, :joint_orig_size, :]

    final_mod = self.norm_out_modulation(temb_input.reshape(joint.shape[0] * joint.shape[1], -1))
    shift, scale = final_mod.reshape(joint.shape[0], joint.shape[1], -1).chunk(2, dim=-1)
    final_hidden = self.norm_out(joint) * (1.0 + scale) + shift
    projected = self.proj_out(final_hidden)
    x = projected[:, :n_video]

    x = x.reshape(B, gt, gh, gw, pF, pH, pW, self.out_channels)
    return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, self.out_channels, T, H, W)
