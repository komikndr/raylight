# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
from enum import Enum

from einops import rearrange
import torch
import comfy.ldm.common_dit

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


def pad_if_odd(t: torch.Tensor, dim: int = 1):
    if t.size(dim) % 2 != 0:
        pad_shape = list(t.shape)
        pad_shape[dim] = 1  # add one element along target dim
        pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad_tensor], dim=dim)
        return t, True
    return t, False


class DataType(Enum):
    IMAGE = "image"
    VIDEO = "video"


# ============================ GENERAL MODEL ======================== #
def usp_general_attention_forward(
    self,
    x,
    context=None,
    mask=None,
    rope_emb=None,
    transformer_options={},
    *args,
    **kwargs,
):
    q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)
    out = xfuser_optimized_attention(q, k, v, self.heads, skip_reshape=True, mask=mask, skip_output_reshape=True)
    del q, k, v
    out = rearrange(out, " b n s c -> s b (n c)")
    return self.to_out(out)


def usp_general_dit_forward(
    self,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    fps: Optional[torch.Tensor] = None,
    image_size: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    scalar_feature: Optional[torch.Tensor] = None,
    data_type: Optional[DataType] = DataType.VIDEO,
    latent_condition: Optional[torch.Tensor] = None,
    latent_condition_sigma: Optional[torch.Tensor] = None,
    condition_video_augment_sigma: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
):
    crossattn_emb = context
    crossattn_mask = attention_mask

    inputs = self.forward_before_blocks(
        x=x,
        timesteps=timesteps,
        crossattn_emb=crossattn_emb,
        crossattn_mask=crossattn_mask,
        fps=fps,
        image_size=image_size,
        padding_mask=padding_mask,
        scalar_feature=scalar_feature,
        data_type=data_type,
        latent_condition=latent_condition,
        latent_condition_sigma=latent_condition_sigma,
        condition_video_augment_sigma=condition_video_augment_sigma,
        **kwargs,
    )
    x, affline_emb_B_D, crossattn_emb, crossattn_mask, rope_emb_L_1_1_D, adaln_lora_B_3D, original_shape = (
        inputs["x"],
        inputs["affline_emb_B_D"],
        inputs["crossattn_emb"],
        inputs["crossattn_mask"],
        inputs["rope_emb_L_1_1_D"],
        inputs["adaln_lora_B_3D"],
        inputs["original_shape"],
    )
    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = inputs["extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D"]
    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.to(x.dtype)
    del inputs

    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert x.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
            f"{x.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape} {original_shape}"
        )

    if self.blocks["block0"].x_format != "THWBD":
        raise ValueError(f"CosmosVideo USP only supports THWBD, got {self.blocks['block0'].x_format}")

    # ================ SEQUENCE PARALLEL ================== #
    sp_world = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    T, H, W, B, D = x.shape
    rope_emb_L_1_1_D = rearrange(rope_emb_L_1_1_D, "(t h w) s c d -> t h w s c d", t=T, h=H, w=W)

    x, t_orig_size = pad_to_world_size(x, dim=0)
    rope_emb_L_1_1_D, _ = pad_to_world_size(rope_emb_L_1_1_D, dim=0)
    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, _ = pad_to_world_size(extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, dim=0)

    x = torch.chunk(x, sp_world, dim=0)[sp_rank]
    rope_emb_L_1_1_D = torch.chunk(rope_emb_L_1_1_D, sp_world, dim=0)[sp_rank]
    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = torch.chunk(extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, sp_world, dim=0)[sp_rank]

    rope_emb_L_1_1_D = rearrange(rope_emb_L_1_1_D, "t h w s c d -> (t h w) s c d")
    # ================ SEQUENCE PARALLEL ================== #
    transformer_options = kwargs.get("transformer_options", {})
    for _, block in self.blocks.items():
        assert self.blocks["block0"].x_format == block.x_format, f"First block has x_format {self.blocks[0].x_format}, got {block.x_format}"

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            x += extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D
        x = block(
            x,
            affline_emb_B_D,
            crossattn_emb,
            crossattn_mask,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_3D=adaln_lora_B_3D,
            transformer_options=transformer_options,
        )

    x = get_sp_group().all_gather(x, dim=0)
    x = x[:t_orig_size, ...]
    x_B_T_H_W_D = rearrange(x, "T H W B D -> B T H W D")

    x_B_D_T_H_W = self.decoder_head(
        x_B_T_H_W_D=x_B_T_H_W_D,
        emb_B_D=affline_emb_B_D,
        crossattn_emb=None,
        origin_shape=original_shape,
        crossattn_mask=None,
        adaln_lora_B_3D=adaln_lora_B_3D,
    )

    return x_B_D_T_H_W


# ============================ PREDICT2 ======================== #
# original code from: https://github.com/nvidia-cosmos/cosmos-predict2
def usp_xfuser_attention_op(
    q_B_S_H_D: torch.Tensor, k_B_S_H_D: torch.Tensor, v_B_S_H_D: torch.Tensor, transformer_options: Optional[dict] = {}
) -> torch.Tensor:
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    return xfuser_optimized_attention(q_B_H_S_D, k_B_H_S_D, v_B_H_S_D, in_q_shape[-2], skip_reshape=True)


def usp_mini_train_dit_forward(
    self,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    fps: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
):
    orig_shape = list(x.shape)
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))
    x_B_C_T_H_W = x
    timesteps_B_T = timesteps
    crossattn_emb = context
    """
    Args:
        x: (B, C, T, H, W) tensor of spatial-temp inputs
        timesteps: (B, ) tensor of timesteps
        crossattn_emb: (B, N, D) tensor of cross-attention embeddings
    """
    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
        x_B_C_T_H_W,
        fps=fps,
        padding_mask=padding_mask,
    )

    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)
    t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](self.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype))
    t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

    # for logging purpose
    affline_scale_log_info = {}
    affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
    self.affline_scale_log_info = affline_scale_log_info
    self.affline_emb = t_embedding_B_T_D
    self.crossattn_emb = crossattn_emb

    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
            f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
        )

    # The residual stream for this model has large values. To make fp16 compute_dtype work, we keep the residual stream
    # in fp32, but run attention and MLP modules in fp16.
    # An alternate method that clamps fp16 values "works" in the sense that it makes coherent images, but there is noticeable
    # quality degradation and visual artifacts.
    if x_B_T_H_W_D.dtype == torch.float16:
        x_B_T_H_W_D = x_B_T_H_W_D.float()

    # ================ SEQUENCE PARALLEL ================== #
    sp_world = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()

    B, T, H, W, D = x_B_T_H_W_D.shape
    x_B_T_H_W_D, h_orig_size = pad_to_world_size(x_B_T_H_W_D, dim=2)

    rope_emb_L_1_1_D = rearrange(rope_emb_L_1_1_D, "(t h w) s c d -> t h w s c d", t=T, h=H, w=W)
    rope_emb_L_1_1_D, _ = pad_to_world_size(rope_emb_L_1_1_D, dim=1)
    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, _ = pad_to_world_size(extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, dim=2)

    x_B_T_H_W_D = torch.chunk(x_B_T_H_W_D, sp_world, dim=2)[sp_rank]
    rope_emb_L_1_1_D = torch.chunk(rope_emb_L_1_1_D, sp_world, dim=1)[sp_rank]
    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = torch.chunk(extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, sp_world, dim=2)[sp_rank]

    rope_emb_L_1_1_D = rearrange(rope_emb_L_1_1_D, "t h w s c d -> (t h w) s c d")
    # ================ SEQUENCE PARALLEL ================== #

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        "transformer_options": kwargs.get("transformer_options", {}),
    }
    for block in self.blocks:
        x_B_T_H_W_D = block(
            x_B_T_H_W_D,
            t_embedding_B_T_D,
            crossattn_emb,
            **block_kwargs,
        )

    x_B_T_H_W_D = get_sp_group().all_gather(x_B_T_H_W_D, dim=2)
    x_B_T_H_W_D = x_B_T_H_W_D[:, :, :h_orig_size, :, :]

    x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D.to(crossattn_emb.dtype), t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
    x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]
    return x_B_C_Tt_Hp_Wp
