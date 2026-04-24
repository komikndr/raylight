import math

import torch
from torch import Tensor

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


def invert_slices(slices, length):
    if slices is None:
        return [(0, length)]

    output = []
    current = 0
    for start, end in slices:
        if start > current:
            output.append((current, start))
        current = end
    if current < length:
        output.append((current, length))
    return output


def attention(q, k, v, pe, mask=None, transformer_options={}) -> Tensor:
    if pe is not None:
        q, k = apply_rope(q, k, pe)

    heads = q.shape[1]
    x = xfuser_optimized_attention(q, k, v, heads, skip_reshape=True)
    return x


def usp_dit_forward(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    timestep_zero_index=None,
    transformer_options={},
    attn_mask: Tensor = None,
    *args,
    **kwargs,
) -> Tensor:
    transformer_options = transformer_options.copy()
    patches = transformer_options.get("patches", {})
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img, img_orig_size = pad_to_world_size(img, dim=1)
    img_ids, _ = pad_to_world_size(img_ids, dim=1)
    txt, txt_orig_size = pad_to_world_size(txt, dim=1)
    txt_ids, _ = pad_to_world_size(txt_ids, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    if self.vector_in is not None:
        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

    if self.txt_norm is not None:
        txt = self.txt_norm(txt)
    txt = self.txt_in(txt)

    if "post_input" in patches:
        for p in patches["post_input"]:
            out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids, "transformer_options": transformer_options})
            img = out["img"]
            txt = out["txt"]
            img_ids = out["img_ids"]
            txt_ids = out["txt_ids"]

    vec_orig = vec
    txt_vec = vec
    modulation_dims = None
    extra_kwargs = {}
    if timestep_zero_index is not None:
        modulation_dims = []
        batch = vec.shape[0] // 2
        vec_orig = vec_orig.reshape(2, batch, vec.shape[1]).movedim(0, 1)
        invert = invert_slices(timestep_zero_index, img.shape[1])
        for start, end in invert:
            modulation_dims.append((start, end, 0))
        for start, end in timestep_zero_index:
            modulation_dims.append((start, end, 1))
        extra_kwargs["modulation_dims_img"] = modulation_dims
        txt_vec = vec[:batch]
    if self.params.global_modulation:
        vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(txt_vec))

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe_combine = self.pe_embedder(ids)
        pe_image = self.pe_embedder(img_ids)
        # seq parallel
        pe_combine = torch.chunk(pe_combine, get_sequence_parallel_world_size(), dim=2)[get_sequence_parallel_rank()]
        pe_image = torch.chunk(pe_image, get_sequence_parallel_world_size(), dim=2)[get_sequence_parallel_rank()]
    else:
        pe_combine = None
        pe_image = None

    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    txt = torch.chunk(txt, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.double_blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.double_blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"),
                                               transformer_options=args.get("transformer_options"),
                                               **extra_kwargs)
                return out

            out = blocks_replace[("double_block", i)]({
                "img": img,
                "txt": txt, "vec": vec,
                "pe": pe_image,
                "attn_mask": attn_mask,
                "transformer_options": transformer_options}, {
                "original_block": block_wrap}
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                             txt=txt,
                             vec=vec,
                             pe=pe_image,
                             attn_mask=attn_mask,
                             transformer_options=transformer_options,
                             **extra_kwargs)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    # Pad + chunk control signal to match USP sequence split
                    add, _ = pad_to_world_size(add, dim=1)
                    add = torch.chunk(add, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    img[:, :add.shape[1]] += add
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    txt = get_sp_group().all_gather(txt.contiguous(), dim=1)
    img = img[:, :img_orig_size, :]
    txt = txt[:, :txt_orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    img = torch.cat((txt, img), 1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img, img_orig_size = pad_to_world_size(img, dim=1)

    if self.params.global_modulation:
        vec, _ = self.single_stream_modulation(vec_orig)
    extra_kwargs = {}
    if timestep_zero_index is not None:
        assert modulation_dims is not None
        extra_kwargs["modulation_dims"] = list(
            map(lambda x: (0 if x[0] == 0 else x[0] + txt.shape[1], x[1] + txt.shape[1], x[2]), modulation_dims)
        )
    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    transformer_options["total_blocks"] = len(self.single_blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [txt.shape[1], img.shape[1]]
    for i, block in enumerate(self.single_blocks):
        transformer_options["block_index"] = i
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"),
                                   transformer_options=args.get("transformer_options"),
                                   **extra_kwargs)
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe_combine,
                                                       "attn_mask": attn_mask,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe_combine, attn_mask=attn_mask, transformer_options=transformer_options, **extra_kwargs)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    # Embed control signal in full concatenated [txt, img] space, then pad+chunk
                    sp_ws = get_sequence_parallel_world_size()
                    sp_r = get_sequence_parallel_rank()
                    txt_len = txt.shape[1]
                    full_add = torch.zeros(img.shape[0], img_orig_size, add.shape[-1],
                                           device=add.device, dtype=add.dtype)
                    full_add[:, txt_len:txt_len + add.shape[1]] = add
                    full_add, _ = pad_to_world_size(full_add, dim=1)
                    img += torch.chunk(full_add, sp_ws, dim=1)[sp_r]

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    img = img[:, :img_orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = img[:, txt.shape[1]:, ...]

    extra_kwargs = {}
    if timestep_zero_index is not None:
        extra_kwargs["modulation_dims"] = modulation_dims

    img = self.final_layer(img, vec_orig, **extra_kwargs)  # (N, T, patch_size ** 2 * out_channels)
    return img


def usp_single_stream_forward(
    self,
    x: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask=None,
    modulation_dims=None,
    transformer_options={},
    **kwargs
) -> Tensor:
    if self.modulation:
        mod, _ = self.modulation(vec)
    else:
        mod = vec

    transformer_patches = transformer_options.get("patches", {})
    extra_options = transformer_options.copy()

    qkv, mlp = torch.split(self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    del qkv
    q, k = self.norm(q, k, v)

    if "attn1_patch" in transformer_patches:
        patch = transformer_patches["attn1_patch"]
        for p in patch:
            out = p(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
            q, k, v, pe, attn_mask = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe), out.get("attn_mask", attn_mask)

    # compute attention
    attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
    del q, k, v

    if "attn1_output_patch" in transformer_patches:
        patch = transformer_patches["attn1_output_patch"]
        for p in patch:
            attn = p(attn, extra_options)

    # compute activation in mlp stream, cat again and run second linear layer
    if self.yak_mlp:
        mlp = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2:]) * mlp[..., :self.mlp_hidden_dim_first // 2]
    else:
        mlp = self.mlp_act(mlp)

    output = self.linear2(torch.cat((attn, mlp), 2))
    x += apply_mod(output, mod.gate, None, modulation_dims)
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def usp_double_stream_forward(
    self,
    img: Tensor,
    txt: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask=None,
    modulation_dims_img=None,
    modulation_dims_txt=None,
    transformer_options={},
    **kwargs
):
    if self.modulation:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
    else:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

    transformer_patches = transformer_options.get("patches", {})
    extra_options = transformer_options.copy()

    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
    img_qkv = self.img_attn.qkv(img_modulated)
    del img_modulated
    img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    del img_qkv
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    del txt_modulated
    txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    del txt_qkv
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    img_q, img_k = apply_rope(img_q, img_k, pe)
    q = torch.cat((txt_q, img_q), dim=2)
    del txt_q, img_q
    k = torch.cat((txt_k, img_k), dim=2)
    del txt_k, img_k
    v = torch.cat((txt_v, img_v), dim=2)
    del txt_v, img_v
    extra_options["img_slice"] = [txt.shape[1], q.shape[2]]
    if "attn1_patch" in transformer_patches:
        patch = transformer_patches["attn1_patch"]
        for p in patch:
            out = p(q, k, v, pe=None, attn_mask=attn_mask, extra_options=extra_options)
            q, k, v, pe, attn_mask = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe), out.get("attn_mask", attn_mask)

    # run actual attention
    attn = attention(q, k, v, pe=None, mask=attn_mask, transformer_options=transformer_options)
    del q, k, v

    if "attn1_output_patch" in transformer_patches:
        extra_options["img_slice"] = [txt.shape[1], attn.shape[1]]
        patch = transformer_patches["attn1_output_patch"]
        for p in patch:
            attn = p(attn, extra_options)

    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

    # calculate the img bloks
    img += apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
    del img_attn
    img += apply_mod(self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)), img_mod2.gate, None, modulation_dims_img)

    # calculate the txt bloks
    txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
    del txt_attn
    txt += apply_mod(self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, None, modulation_dims_txt)

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    return img, txt
