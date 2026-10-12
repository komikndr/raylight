import comfy.ldm.common_dit
import comfy_kitchen
import torch

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


def usp_attention_forward(self, img, txt, image_rotary_emb, transformer_options=None, **kwargs):
    heads = self.num_attention_heads

    img_q, img_k, img_v = self.img_attn_qkv(img).chunk(3, dim=-1)
    txt_q, txt_k, txt_v = self.txt_attn_qkv(txt).chunk(3, dim=-1)

    img_q = img_q.unflatten(-1, (heads, -1))
    img_k = img_k.unflatten(-1, (heads, -1))
    img_v = img_v.unflatten(-1, (heads, -1))
    txt_q = txt_q.unflatten(-1, (heads, -1))
    txt_k = txt_k.unflatten(-1, (heads, -1))
    txt_v = txt_v.unflatten(-1, (heads, -1))

    img_q = self.img_attn_q_norm(img_q)
    img_k = self.img_attn_k_norm(img_k)
    txt_q = self.txt_attn_q_norm(txt_q)
    txt_k = self.txt_attn_k_norm(txt_k)

    img_q, img_k = comfy_kitchen.apply_rope(img_q, img_k, image_rotary_emb)

    joint_q = torch.cat([img_q, txt_q], dim=1).flatten(2, 3)
    joint_k = torch.cat([img_k, txt_k], dim=1).flatten(2, 3)
    joint_v = torch.cat([img_v, txt_v], dim=1).flatten(2, 3)
    joint_out = xfuser_optimized_attention(
        joint_q,
        joint_k,
        joint_v,
        heads=heads,
        transformer_options=transformer_options,
    )

    seq_img = img.shape[1]
    img_out = self.img_attn_proj(joint_out[:, :seq_img, :])
    txt_out = self.txt_attn_proj(joint_out[:, seq_img:, :])
    return img_out, txt_out


def usp_dit_forward(
    self,
    hidden_states,
    timestep,
    context,
    ref_latents=None,
    transformer_options=None,
    **kwargs,
):
    pt, ph, pw = self.patch_size
    _, _, ot, oh, ow = hidden_states.shape

    components = [hidden_states, *(ref_latents or [])]
    component_sizes = []
    img_tokens = []
    for comp in components:
        comp = comfy.ldm.common_dit.pad_to_patch_size(comp, self.patch_size)
        _, _, ct, ch, cw = comp.shape
        component_sizes.append((ct // pt, ch // ph, cw // pw))
        img_tokens.append(self.img_in(comp).flatten(2).transpose(1, 2))

    img = torch.cat(img_tokens, dim=1)
    _, vec, txt = self.condition_embedder(timestep, context)
    vec = vec.unflatten(1, (6, -1))
    image_rotary_emb = self.get_rotary_pos_embed_for_components(component_sizes, device=hidden_states.device)

    img, img_orig_size = pad_to_world_size(img, dim=1)
    txt, _ = pad_to_world_size(txt, dim=1)
    image_rotary_emb, _ = pad_to_world_size(image_rotary_emb, dim=1)

    sp_world_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    # ===================== SP SPLIT ====================== #
    img = torch.chunk(img, sp_world_size, dim=1)[sp_rank]
    txt = torch.chunk(txt, sp_world_size, dim=1)[sp_rank]
    image_rotary_emb = torch.chunk(image_rotary_emb, sp_world_size, dim=1)[sp_rank]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.double_blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.double_blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(
                    hidden_states=args["img"],
                    encoder_hidden_states=args["txt"],
                    temb=args["vec"],
                    image_rotary_emb=args["pe"],
                    transformer_options=args.get("transformer_options"),
                )
                return out

            out = blocks_replace[("double_block", i)](
                {
                    "img": img,
                    "txt": txt,
                    "vec": vec,
                    "pe": image_rotary_emb,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(
                hidden_states=img,
                encoder_hidden_states=txt,
                temb=vec,
                image_rotary_emb=image_rotary_emb,
                transformer_options=transformer_options,
            )

    # ===================== SP GATHER ===================== #
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    img = img[:, :img_orig_size, :]

    tt, th, tw = component_sizes[0]
    target_tokens = tt * th * tw
    img = self.proj_out(self.norm_out(img[:, :target_tokens, :]))
    img = self.unpatchify(img, tt, th, tw)
    return img[:, :, :ot, :oh, :ow]
