import torch
from torch import Tensor

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
import comfy

attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.to(dtype=freqs_cis.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_cis.dtype).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def pad_if_odd(t: torch.Tensor, dim: int = 1):
    if t.size(dim) % 2 != 0:
        pad_shape = list(t.shape)
        pad_shape[dim] = 1  # add one element along target dim
        pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad_tensor], dim=dim)
    return t


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


def usp_dit_forward(self, x, timesteps, context, num_tokens, attention_mask=None, **kwargs):
    t = 1.0 - timesteps
    cap_feats = context
    cap_mask = attention_mask
    bs, c, h, w = x.shape
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
    """
    Forward pass of NextDiT.
    t: (N,) tensor of diffusion timesteps
    y: (N,) tensor of text tokens/features
    """

    t = self.t_embedder(t * self.time_scale, dtype=x.dtype)  # (N, D)
    adaln_input = t

    cap_feats = self.cap_embedder(
        cap_feats
    )  # (N, L, D)  # todo check if able to batchify w.o. redundant compute

    transformer_options = kwargs.get("transformer_options", {})
    x_is_tensor = isinstance(x, torch.Tensor)
    x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(
        x, cap_feats, cap_mask, t, num_tokens, transformer_options=transformer_options
    )
    freqs_cis = freqs_cis.to(x.device)
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = pad_if_odd(x, dim=1)
    freqs_cis = pad_if_odd(freqs_cis, dim=1)

    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    freqs_cis = torch.chunk(freqs_cis, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    for layer in self.layers:
        x = layer(
            x, mask, freqs_cis, adaln_input, transformer_options=transformer_options
        )

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    x = get_sp_group().all_gather(x, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    x = self.final_layer(x, adaln_input)
    x = self.unpatchify(x, img_size, cap_size, return_tensor=x_is_tensor)[:, :, :h, :w]

    return -x


def usp_joint_attention_forward(
    self,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    freqs_cis: torch.Tensor,
    transformer_options={},
) -> torch.Tensor:
    bsz, seqlen, _ = x.shape

    xq, xk, xv = torch.split(
        self.qkv(x),
        [
            self.n_local_heads * self.head_dim,
            self.n_local_kv_heads * self.head_dim,
            self.n_local_kv_heads * self.head_dim,
        ],
        dim=-1,
    )
    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

    xq = self.q_norm(xq)
    xk = self.k_norm(xk)

    xq, xk = apply_rope(xq, xk, freqs_cis)

    n_rep = self.n_local_heads // self.n_local_kv_heads
    if n_rep >= 1:
        xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

    output = xfuser_optimized_attention(xq.movedim(1, 2),
                                        xk.movedim(1, 2),
                                        xv.movedim(1, 2),
                                        self.n_local_heads,
                                        x_mask,
                                        skip_reshape=True)

    return self.out(output)
