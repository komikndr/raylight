import torch
import torch.nn.functional as F
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

import comfy.ldm.common_dit
from comfy.ldm.flux.math import apply_rope
from ..utils import pad_to_world_size
import raylight.distributed_modules.attention as xfuser_attn

attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def _pad_and_chunk(tensor, dim=1):
    tensor, orig_size = pad_to_world_size(tensor, dim=dim)
    tensor = torch.chunk(tensor, get_sequence_parallel_world_size(), dim=dim)[get_sequence_parallel_rank()]
    return tensor, orig_size


def _pad_and_chunk_pos(pos):
    pos = pos.unsqueeze(0)
    pos, orig_size = _pad_and_chunk(pos, dim=1)
    return pos.squeeze(0), orig_size


def usp_joint_attention_forward(self, x, y, pos_img, pos_txt=None, attn_mask=None, transformer_options={}):
    B, Nx, _ = x.shape
    _, Ny, _ = y.shape
    H = self.num_heads
    D = self.head_dim

    qkv_x = self.qkv_x(x).reshape(B, Nx, 3, H, D).permute(2, 0, 3, 1, 4)
    qx, kx, vx = qkv_x.unbind(0)
    qx = self.q_norm_x(qx)
    kx = self.k_norm_x(kx)

    qkv_y = self.qkv_y(y).reshape(B, Ny, 3, H, D).permute(2, 0, 3, 1, 4)
    qy, ky, vy = qkv_y.unbind(0)
    qy = self.q_norm_y(qy)
    ky = self.k_norm_y(ky)

    qx, kx = apply_rope(qx, kx, pos_img[None, None])
    if pos_txt is not None:
        qy, ky = apply_rope(qy, ky, pos_txt[None, None])

    q_joint = torch.cat([qy, qx], dim=2)
    k_joint = torch.cat([ky, kx], dim=2)
    v_joint = torch.cat([vy, vx], dim=2)

    out_joint = xfuser_optimized_attention(
        q_joint,
        k_joint,
        v_joint,
        H,
        mask=None,
        skip_reshape=True,
        skip_output_reshape=True,
    )

    out_y = out_joint[:, :, :Ny, :].transpose(1, 2).reshape(B, Ny, H * D)
    out_x = out_joint[:, :, Ny:, :].transpose(1, 2).reshape(B, Nx, H * D)
    return self.proj_x(out_x), self.proj_y(out_y)


def usp_rotary_attention_forward(self, x, pos, mask=None, transformer_options={}):
    B, N, _ = x.shape
    H = self.num_heads
    D = self.head_dim
    qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = apply_rope(self.q_norm(q), self.k_norm(k), pos[None, None])
    x = xfuser_optimized_attention(q, k, v, H, mask=None, skip_reshape=True)
    return self.proj(x)


def usp_pit_block_forward(self, x, s_cond, image_height, image_width, patch_size, mask=None, transformer_options={}):
    BL, P2, _ = x.shape
    B = transformer_options.get("_raylight_pixeldit_batch")
    local_l = transformer_options.get("_raylight_pixeldit_local_l")
    pos_comp = transformer_options.get("_raylight_pixeldit_pos")
    if B is None or local_l is None:
        Hs, Ws = image_height // patch_size, image_width // patch_size
        local_l = Hs * Ws
        B = BL // local_l
    if pos_comp is None:
        Hs, Ws = image_height // patch_size, image_width // patch_size
        pos_comp = self._fetch_pos(Hs, Ws, x.device, x.dtype, **(transformer_options.get("rope_options") or {}))

    msa_params = self.adaLN_modulation_msa(s_cond).view(BL, P2, 3 * self.pixel_dim)
    shift_msa, scale_msa, gate_msa = msa_params.chunk(3, dim=-1)

    x_norm = self.norm1(x)
    x_norm = x_norm.addcmul(x_norm, scale_msa).add_(shift_msa)
    x_flat = x_norm.reshape(BL, P2 * self.pixel_dim)

    x_comp = self.compress_to_attn(x_flat).view(B, local_l, self.attn_dim)
    attn_out = self.attn(x_comp, pos_comp, mask=None, transformer_options=transformer_options)
    attn_flat = self.expand_from_attn(attn_out.reshape(BL, self.attn_dim))
    attn_exp = attn_flat.view(BL, P2, self.pixel_dim)
    x = torch.addcmul(x, gate_msa, attn_exp)
    del msa_params, shift_msa, scale_msa, gate_msa

    mlp_params = self.adaLN_modulation_mlp(s_cond).view(BL, P2, 3 * self.pixel_dim)
    shift_mlp, scale_mlp, gate_mlp = mlp_params.chunk(3, dim=-1)
    gate_mlp = gate_mlp.contiguous()
    mlp_input = self.norm2(x)
    mlp_input = mlp_input.addcmul(mlp_input, scale_mlp).add_(shift_mlp)
    del mlp_params, shift_mlp, scale_mlp

    chunk_size = (BL + self.mlp_chunks - 1) // self.mlp_chunks
    for start in range(0, BL, chunk_size):
        end = min(start + chunk_size, BL)
        x[start:end].addcmul_(gate_mlp[start:end], self.mlp(mlp_input[start:end]))
    return x


def usp_pixdit_forward(self, x, timesteps, context=None, attention_mask=None, transformer_options={}, **kwargs):
    transformer_options = transformer_options.copy()
    H_orig, W_orig = x.shape[2], x.shape[3]
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
    B, _, H, W = x.shape
    Hs = H // self.patch_size
    Ws = W // self.patch_size
    L = Hs * Ws

    pos_img = self._fetch_patch_pos(Hs, Ws, x.device, x.dtype, **(transformer_options.get("rope_options") or {}))
    x_patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

    t_emb = self.t_embedder(timesteps.view(-1), x.dtype).view(B, -1, self.hidden_size)

    if context is None or context.dim() != 3:
        raise ValueError("PixDiT_T2I requires context (text embeddings) of shape [B, L, D]")
    Ltxt = min(context.shape[1], self.txt_max_length)
    y = context[:, :Ltxt, :]
    y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
    y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb)

    condition = F.silu(t_emb)
    pos_txt = self._fetch_text_pos(Ltxt, x.device, x.dtype) if self.use_text_rope else None

    s = self.s_embedder(x_patches)
    s, s_orig_size = _pad_and_chunk(s, dim=1)
    y_emb, _ = _pad_and_chunk(y_emb, dim=1)
    pos_img, _ = _pad_and_chunk_pos(pos_img)
    if pos_txt is not None:
        pos_txt, _ = _pad_and_chunk_pos(pos_txt)

    local_kwargs = dict(kwargs)
    pid_lq_features = local_kwargs.get("pid_lq_features")
    if pid_lq_features is not None:
        local_kwargs["pid_lq_features"] = [_pad_and_chunk(feature, dim=1)[0] for feature in pid_lq_features]

    for i, blk in enumerate(self.patch_blocks):
        s = self._pre_patch_block(s, i, **local_kwargs)
        s, y_emb = blk(s, y_emb, condition, pos_img, pos_txt, None, transformer_options=transformer_options)
    s = F.silu(t_emb + s)

    local_l = s.shape[1]
    s_cond = s.reshape(B * local_l, self.hidden_size)
    x_pixels = self.pixel_embedder(x, patch_size=self.patch_size)
    P2 = x_pixels.shape[1]
    x_pixels = x_pixels.view(B, L, P2, self.pixel_hidden_size)
    x_pixels, _ = _pad_and_chunk(x_pixels, dim=1)
    local_l = x_pixels.shape[1]
    x_pixels = x_pixels.reshape(B * local_l, P2, self.pixel_hidden_size)

    transformer_options["_raylight_pixeldit_batch"] = B
    transformer_options["_raylight_pixeldit_local_l"] = local_l
    transformer_options["_raylight_pixeldit_pos"] = pos_img
    try:
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask=None, transformer_options=transformer_options)
    finally:
        transformer_options.pop("_raylight_pixeldit_batch", None)
        transformer_options.pop("_raylight_pixeldit_local_l", None)
        transformer_options.pop("_raylight_pixeldit_pos", None)

    x_pixels = self.final_layer(x_pixels)
    C_out = self.out_channels
    x_pixels = x_pixels.view(B, local_l, P2, C_out)
    x_pixels = get_sp_group().all_gather(x_pixels.contiguous(), dim=1)
    x_pixels = x_pixels[:, :s_orig_size, :, :]
    x_pixels = x_pixels.permute(0, 3, 2, 1).reshape(B, C_out * P2, L)
    out = F.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
    return out[:, :, :H_orig, :W_orig]


def usp_pid_forward(
    self,
    x,
    timesteps,
    context=None,
    attention_mask=None,
    transformer_options={},
    lq_latent=None,
    degrade_sigma=None,
    **kwargs,
):
    if lq_latent is None:
        raise ValueError("PidNet requires lq_latent — attach via PiDConditioning")
    expected_c = self.lq_proj.latent_channels
    if lq_latent.shape[1] != expected_c:
        raise ValueError(
            f"Input latent has {lq_latent.shape[1]} channels, this model variant expects {expected_c}. "
            f"Flux1/SD3 = 16 channels, Flux2 = 128 channels."
        )
    B = x.shape[0]
    Hs = -(-x.shape[2] // self.patch_size)
    Ws = -(-x.shape[3] // self.patch_size)

    degrade_sigma = degrade_sigma.to(device=x.device, dtype=torch.float32).reshape(-1)
    if degrade_sigma.numel() == 1 and B > 1:
        degrade_sigma = degrade_sigma.expand(B).contiguous()

    lq_features = self.lq_proj(lq_latent=lq_latent.to(x), target_pH=Hs, target_pW=Ws)
    return usp_pixdit_forward(
        self,
        x,
        timesteps,
        context=context,
        attention_mask=attention_mask,
        transformer_options=transformer_options,
        pid_lq_features=lq_features,
        pid_degrade_sigma=degrade_sigma,
        **kwargs,
    )
