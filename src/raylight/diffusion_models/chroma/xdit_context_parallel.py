import math

import torch
from torch import Tensor

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
from ..utils import pad_to_world_size
attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def apply_mod(tensor, m_mult, m_add=None, modulation_dims=None):
    if modulation_dims is None:
        if m_add is not None:
            return torch.addcmul(m_add, tensor, m_mult)
        else:
            return tensor * m_mult
    else:
        for d in modulation_dims:
            tensor[:, d[0]:d[1]] *= m_mult[:, d[2]]
            if m_add is not None:
                tensor[:, d[0]:d[1]] += m_add[:, d[2]]
        return tensor


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.to(dtype=freqs_cis.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_cis.dtype).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q, k, v, pe, mask=None) -> Tensor:
    if pe is not None:
        q, k = apply_rope(q, k, pe)

    heads = q.shape[1]
    x = xfuser_optimized_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True
    )
    return x


def usp_dit_forward(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})

    # running on sequences img
    img = self.img_in(img)

    # distilled vector guidance
    mod_index_length = 344
    distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
    # guidance = guidance *
    distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)

    # get all modulation index
    modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(img.device, img.dtype)
    # we need to broadcast the modulation index here so each batch has all of the index
    modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
    # and we need to broadcast timestep and guidance along too
    timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
    # then and only then we could concatenate it together
    input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)

    mod_vectors = self.distilled_guidance_layer(input_vec)

    txt = self.txt_in(txt)

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    # Seq is odd (idk how) if the w == h, so just pad 0 to the end
    img, img_orig_size = pad_to_world_size(img, dim=1)
    img_ids, _ = pad_to_world_size(img_ids, dim=1)
    txt, txt_orig_size= pad_to_world_size(txt, dim=1)
    txt_ids, _ = pad_to_world_size(txt_ids, dim=1)
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe_combine = self.pe_embedder(ids)
    pe_image = self.pe_embedder(img_ids)

    pe_combine = torch.chunk(pe_combine, get_sequence_parallel_world_size(), dim=2)[get_sequence_parallel_rank()]
    pe_image = torch.chunk(pe_image, get_sequence_parallel_world_size(), dim=2)[get_sequence_parallel_rank()]

    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    txt = torch.chunk(txt, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if i not in self.skip_mmdit:
            double_mod = (
                self.get_modulations(mod_vectors, "double_img", idx=i),
                self.get_modulations(mod_vectors, "double_txt", idx=i),
            )
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"),
                                                   transformer_options=args.get("transformer_options"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": double_mod,
                                                           "pe": pe_image,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=double_mod,
                                 pe=pe_image,
                                 attn_mask=attn_mask,
                                 transformer_options=transformer_options)

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    txt = get_sp_group().all_gather(txt.contiguous(), dim=1)

    img = img[:, :img_orig_size, :]
    txt = txt[:, :txt_orig_size, :]

    img = torch.cat((txt, img), 1)
    img, img_orig_size = pad_to_world_size(img, dim=1)
    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    for i, block in enumerate(self.single_blocks):
        if i not in self.skip_dit:
            single_mod = self.get_modulations(mod_vectors, "single", idx=i)
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": single_mod,
                                                           "pe": pe_combine,
                                                           "attn_mask": attn_mask},
                                                          {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=single_mod, pe=pe_combine, attn_mask=attn_mask, transformer_options=transformer_options)

            if control is not None:  # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1]:, ...] += add

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    img = img[:, :img_orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = img[:, txt.shape[1]:, ...]
    if hasattr(self, "final_layer"):
        final_mod = self.get_modulations(mod_vectors, "final")
        img = self.final_layer(img, vec=final_mod)  # (N, T, patch_size ** 2 * out_channels)
    return img


def usp_single_stream_forward(self, x: Tensor, pe: Tensor, vec: Tensor, attn_mask=None, **kwargs) -> Tensor:
    mod = vec
    x_mod = torch.addcmul(mod.shift, 1 + mod.scale, self.pre_norm(x))
    qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = self.norm(q, k, v)

    # compute attention
    attn = attention(q, k, v, pe=pe, mask=attn_mask)
    # compute activation in mlp stream, cat again and run second linear layer
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    x.addcmul_(mod.gate, output)
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def usp_double_stream_forward(self, img: Tensor, txt: Tensor, pe: Tensor, vec: Tensor, attn_mask=None, **kwargs):
    (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

    # prepare image for attention
    img_modulated = torch.addcmul(img_mod1.shift, 1 + img_mod1.scale, self.img_norm1(img))
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = torch.addcmul(txt_mod1.shift, 1 + txt_mod1.scale, self.txt_norm1(txt))
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    # run actual attention
    img_q, img_k = apply_rope(img_q, img_k, pe)
    attn = attention(torch.cat((txt_q, img_q), dim=2),
                     torch.cat((txt_k, img_k), dim=2),
                     torch.cat((txt_v, img_v), dim=2),
                     pe=None, mask=attn_mask)

    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

    # calculate the img bloks
    img.addcmul_(img_mod1.gate, self.img_attn.proj(img_attn))
    img.addcmul_(img_mod2.gate, self.img_mlp(torch.addcmul(img_mod2.shift, 1 + img_mod2.scale, self.img_norm2(img))))

    # calculate the txt bloks
    txt.addcmul_(txt_mod1.gate, self.txt_attn.proj(txt_attn))
    txt.addcmul_(txt_mod2.gate, self.txt_mlp(torch.addcmul(txt_mod2.shift, 1 + txt_mod2.scale, self.txt_norm2(txt))))

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    return img, txt
