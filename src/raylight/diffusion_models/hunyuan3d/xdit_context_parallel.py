import math
import torch
from torch import Tensor

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn

attn_type = xfuser_attn.get_attn_type()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type)

# HunyuanVid use the same DiT single/double block as Flux


def pad_if_odd(t: torch.Tensor, dim: int = 1):
    if t.size(dim) % 2 != 0:
        pad_shape = list(t.shape)
        pad_shape[dim] = 1  # add one element along target dim
        pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad_tensor], dim=dim)
    return t


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
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def usp_dit_forward(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    txt_mask: Tensor,
    timesteps: Tensor,
    y: Tensor = None,
    txt_byt5=None,
    guidance: Tensor = None,
    guiding_frame_index=None,
    ref_latent=None,
    disable_time_r=False,
    control=None,
    transformer_options={},
) -> Tensor:
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    # Seq is odd (idk how) if the w == h, so just pad 0 to the end
    img = pad_if_odd(img, dim=1)
    img_ids = pad_if_odd(img_ids, dim=1)
    txt = pad_if_odd(txt, dim=1)
    txt_ids = pad_if_odd(txt_ids, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    patches_replace = transformer_options.get("patches_replace", {})

    initial_shape = list(img.shape)
    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(
        timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype)
    )

    if (self.time_r_in is not None) and (not disable_time_r):
        w = torch.where(
            transformer_options["sigmas"][0] == transformer_options["sample_sigmas"]
        )[
            0
        ]  # This most likely could be improved
        if len(w) > 0:
            timesteps_r = transformer_options["sample_sigmas"][w[0] + 1]
            timesteps_r = timesteps_r.unsqueeze(0).to(
                device=timesteps.device, dtype=timesteps.dtype
            )
            vec_r = self.time_r_in(
                timestep_embedding(timesteps_r, 256, time_factor=1000.0).to(img.dtype)
            )
            vec = (vec + vec_r) / 2

    if ref_latent is not None:
        ref_latent_ids = self.img_ids(ref_latent)
        ref_latent = self.img_in(ref_latent)
        img = torch.cat([ref_latent, img], dim=-2)
        ref_latent_ids[..., 0] = -1
        ref_latent_ids[..., 2] += initial_shape[-1] // self.patch_size[-1]
        img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

    if guiding_frame_index is not None:
        token_replace_vec = self.time_in(
            timestep_embedding(guiding_frame_index, 256, time_factor=1.0)
        )
        if self.vector_in is not None:
            vec_ = self.vector_in(y[:, : self.params.vec_in_dim])
            vec = torch.cat(
                [(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)],
                dim=1,
            )
        else:
            vec = torch.cat(
                [(token_replace_vec).unsqueeze(1), (vec).unsqueeze(1)], dim=1
            )
        frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (
            initial_shape[-2] // self.patch_size[-2]
        )
        modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
        modulation_dims_txt = [(0, None, 1)]
    else:
        if self.vector_in is not None:
            vec = vec + self.vector_in(y[:, : self.params.vec_in_dim])
        modulation_dims = None
        modulation_dims_txt = None

    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(
                timestep_embedding(guidance, 256).to(img.dtype)
            )

    if txt_mask is not None and not torch.is_floating_point(txt_mask):
        txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

    txt = self.txt_in(txt, timesteps, txt_mask, transformer_options=transformer_options)

    if self.byt5_in is not None and txt_byt5 is not None:
        txt_byt5 = self.byt5_in(txt_byt5)
        txt_byt5_ids = torch.zeros(
            (txt_ids.shape[0], txt_byt5.shape[1], txt_ids.shape[-1]),
            device=txt_ids.device,
            dtype=txt_ids.dtype,
        )
        txt = torch.cat((txt, txt_byt5), dim=1)
        txt_ids = torch.cat((txt_ids, txt_byt5_ids), dim=1)

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe_combine = self.pe_embedder(ids)
    pe_image = self.pe_embedder(img_ids)
    # seq parallel
    pe_combine = torch.chunk(pe_combine, get_sequence_parallel_world_size(), dim=2)[get_sequence_parallel_rank()]
    pe_image = torch.chunk(pe_image, get_sequence_parallel_world_size(), dim=2)[get_sequence_parallel_rank()]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    img_len = img.shape[1]
    if txt_mask is not None:
        attn_mask_len = img_len + txt.shape[1]
        attn_mask = torch.zeros(
            (1, 1, attn_mask_len), dtype=img.dtype, device=img.device
        )
        attn_mask[:, 0, img_len:] = txt_mask
    else:
        attn_mask = None

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    txt = torch.chunk(txt, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(
                    img=args["img"],
                    txt=args["txt"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args["attention_mask"],
                    modulation_dims_img=args["modulation_dims_img"],
                    modulation_dims_txt=args["modulation_dims_txt"],
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("double_block", i)](
                {
                    "img": img,
                    "txt": txt,
                    "vec": vec,
                    "pe": pe_image,
                    "attention_mask": attn_mask,
                    "modulation_dims_img": modulation_dims,
                    "modulation_dims_txt": modulation_dims_txt,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(
                img=img,
                txt=txt,
                vec=vec,
                pe=pe_image,
                attn_mask=attn_mask,
                modulation_dims_img=modulation_dims,
                modulation_dims_txt=modulation_dims_txt,
                transformer_options=transformer_options,
            )

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = get_sp_group().all_gather(img, dim=1)
    txt = get_sp_group().all_gather(txt, dim=1)

    img = torch.cat((txt, img), 1)

    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args["attention_mask"],
                    modulation_dims=args["modulation_dims"],
                    transformer_options=args["transformer_options"],
                )
                return out

            out = blocks_replace[("single_block", i)](
                {
                    "img": img,
                    "vec": vec,
                    "pe": pe_combine,
                    "attention_mask": attn_mask,
                    "modulation_dims": modulation_dims,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )
            img = out["img"]
        else:
            img = block(
                img,
                vec=vec,
                pe=pe_combine,
                attn_mask=attn_mask,
                modulation_dims=modulation_dims,
                transformer_options=transformer_options,
            )

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, :img_len] += add

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = get_sp_group().all_gather(img, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = img[:, :img_len]
    if ref_latent is not None:
        img = img[:, ref_latent.shape[1]:]

    img = self.final_layer(
        img, vec, modulation_dims=modulation_dims
    )  # (N, T, patch_size ** 2 * out_channels)

    shape = initial_shape[-len(self.patch_size):]
    for i in range(len(shape)):
        shape[i] = shape[i] // self.patch_size[i]
    img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
    if img.ndim == 8:
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(
            initial_shape[0],
            self.out_channels,
            initial_shape[2],
            initial_shape[3],
            initial_shape[4],
        )
    else:
        img = img.permute(0, 3, 1, 4, 2, 5)
        img = img.reshape(
            initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3]
        )
    return img
