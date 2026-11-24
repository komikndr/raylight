import torch
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
import comfy
from ..utils import pad_to_world_size, shard_ranges
from comfy.ldm.flux.math import apply_rope
attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


def usp_dit_forward(
        self,
        x,
        timesteps,
        context,
        num_tokens,
        attention_mask=None,
        ref_latents=[],
        ref_contexts=[],
        siglip_feats=[],
        direct_context=None,
        ref_frames=(),
        transformer_options={},
        *args,
        **kwargs
):
    omni = len(ref_latents) > 0
    if omni:
        timesteps = torch.cat([timesteps * 0, timesteps], dim=0)

    t = 1.0 - timesteps
    cap_feats = context
    cap_mask = attention_mask
    h, w = x.shape[-2:]
    frames = x.shape[2] if x.ndim == 5 else None
    patch = self.patch_size
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (1, patch, patch) if frames is not None else (patch, patch))

    t = self.t_embedder(t * self.time_scale, dtype=x.dtype)  # (N, D)
    adaln_input = t

    if self.clip_text_pooled_proj is not None:
        pooled = kwargs.get("clip_text_pooled", None)
        if pooled is not None:
            pooled = self.clip_text_pooled_proj(pooled)
        else:
            pooled = torch.zeros((x.shape[0], self.clip_text_dim), device=x.device, dtype=x.dtype)

        adaln_input = self.time_text_embed(torch.cat((t, pooled), dim=-1))

    patches = transformer_options.get("patches", {})
    ming_conds = {"direct_context": direct_context, "ref_frames": ref_frames} if direct_context is not None or ref_frames else {}
    x_is_tensor = isinstance(x, torch.Tensor)
    img, mask, img_size, cap_size, freqs_cis, timestep_zero_index = self.patchify_and_embed(
        x,
        cap_feats,
        cap_mask,
        adaln_input,
        num_tokens,
        ref_latents=ref_latents,
        ref_contexts=ref_contexts,
        siglip_feats=siglip_feats,
        transformer_options=transformer_options,
        **ming_conds,
    )

    freqs_cis = freqs_cis.to(img.device)
    # ===================== SP SPLIT ====================== #
    transformer_options = dict(transformer_options or {}, raylight_key_lengths=(img.shape[1],))
    img, img_orig_size = pad_to_world_size(img, dim=1)
    freqs_cis, _ = pad_to_world_size(freqs_cis, dim=1)

    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    freqs_cis = torch.chunk(freqs_cis, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    local_zero_index = shard_ranges(timestep_zero_index, img.shape[1])
    local_cap = max(0, min(img.shape[1], cap_size[0] - get_sequence_parallel_rank() * img.shape[1]))

    transformer_options["total_blocks"] = len(self.layers)
    transformer_options["block_type"] = "double"
    img_input = img
    for i, layer in enumerate(self.layers):
        transformer_options["block_index"] = i
        img = layer(img,
                    mask,
                    freqs_cis,
                    adaln_input,
                    timestep_zero_index=local_zero_index,
                    transformer_options=transformer_options)

        if "double_block" in patches:
            for p in patches["double_block"]:
                out = p({"img": img[:, local_cap:],
                         "img_input": img_input[:, local_cap:],
                         "txt": img[:, :local_cap],
                         "pe": freqs_cis[:, local_cap:],
                         "vec": adaln_input,
                         "x": x,
                         "block_index": i,
                         "transformer_options": transformer_options})

                if "img" in out:
                    img[:, local_cap:] = out["img"]
                if "txt" in out:
                    img[:, :local_cap] = out["txt"]

    # ===================== SP GATHER ===================== #

    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    img = img[:, :img_orig_size, :]

    if hasattr(self, "dec_net"):
        b, c, hp, wp = x.shape
        n = hp * wp // (patch * patch)
        pixels = x.reshape(b, c, hp // patch, patch, wp // patch, patch).permute(0, 2, 4, 3, 5, 1)
        cond = img[:, cap_size[0]:cap_size[0] + n].reshape(b * n, self.dim)
        img = self.dec_net(pixels.reshape(b * n, 1, -1), cond).reshape(b, n, -1)
        cap_size = [0] * b
    else:
        img = self.final_layer(img, adaln_input, timestep_zero_index=timestep_zero_index)
    frame_arg = {"frames": frames} if frames is not None else {}
    img = self.unpatchify(img, img_size, cap_size, return_tensor=x_is_tensor, **frame_arg)[..., :h, :w]
    return -img


def usp_joint_attention_forward(
    self,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    freqs_cis: torch.Tensor,
    transformer_options={},
    *args,
    **kwargs,
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
                                         mask=x_mask,
                                         skip_reshape=True, key_lengths=transformer_options.get("raylight_key_lengths"))

    return self.out(output)
