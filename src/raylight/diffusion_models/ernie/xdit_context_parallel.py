import torch

import comfy

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

import raylight.distributed_modules.attention as xfuser_attn

from comfy.ldm.ernie.model import apply_rotary_emb

from ..utils import pad_to_world_size

attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def usp_forward(self, x, timesteps, context, **kwargs):
    transformer_options = kwargs.get("transformer_options", {})
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
        self._forward,
        self,
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
    ).execute(x, timesteps, context, **kwargs)


def usp_attention_forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, image_rotary_emb: torch.Tensor = None) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape

    q_flat = self.to_q(x)
    k_flat = self.to_k(x)
    v_flat = self.to_v(x)

    query = self.norm_q(q_flat.view(batch_size, seq_len, self.heads, self.head_dim))
    key = self.norm_k(k_flat.view(batch_size, seq_len, self.heads, self.head_dim))

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    hidden_states = xfuser_optimized_attention(
        query.reshape(batch_size, seq_len, -1).to(x.dtype),
        key.reshape(batch_size, seq_len, -1).to(x.dtype),
        v_flat,
        self.heads,
        mask=attention_mask,
    )
    return self.to_out[0](hidden_states)


def usp_dit_forward(self, x, timesteps, context, **kwargs):
    device, dtype = x.device, x.dtype
    batch_size, _, height, width = x.shape
    patch_size = self.patch_size
    patches_h = height // patch_size
    patches_w = width // patch_size
    num_image_tokens = patches_h * patches_w

    image_tokens = self.x_embedder(x)

    text_tokens = context
    if self.text_proj is not None and text_tokens.numel() > 0:
        text_tokens = self.text_proj(text_tokens)
    text_seq_len = text_tokens.shape[1]

    hidden_states = torch.cat([image_tokens, text_tokens], dim=1)

    text_ids = torch.zeros((batch_size, text_seq_len, 3), device=device, dtype=torch.float32)
    text_ids[:, :, 0] = torch.linspace(0, text_seq_len - 1, steps=text_seq_len, device=device, dtype=torch.float32)
    index = float(text_seq_len)

    transformer_options = kwargs.get("transformer_options", {})
    rope_options = transformer_options.get("rope_options")

    h_len = float(patches_h)
    w_len = float(patches_w)
    h_offset = 0.0
    w_offset = 0.0

    if rope_options is not None:
        h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
        w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0
        index += rope_options.get("shift_t", 0.0)
        h_offset += rope_options.get("shift_y", 0.0)
        w_offset += rope_options.get("shift_x", 0.0)

    image_ids = torch.zeros((patches_h, patches_w, 3), device=device, dtype=torch.float32)
    image_ids[:, :, 0] = index
    image_ids[:, :, 1] = torch.linspace(h_offset, h_len - 1 + h_offset, steps=patches_h, device=device, dtype=torch.float32).unsqueeze(1)
    image_ids[:, :, 2] = torch.linspace(w_offset, w_len - 1 + w_offset, steps=patches_w, device=device, dtype=torch.float32).unsqueeze(0)
    image_ids = image_ids.view(1, num_image_tokens, 3).expand(batch_size, -1, -1)

    rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1)).to(dtype)

    sample = self.time_proj(timesteps).to(dtype)
    conditioning = self.time_embedding(sample)
    temb = [chunk.unsqueeze(1).contiguous() for chunk in self.adaLN_modulation(conditioning).chunk(6, dim=-1)]

    sp_world_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()

    hidden_states, hidden_states_orig_size = pad_to_world_size(hidden_states, dim=1)
    rotary_pos_emb, _ = pad_to_world_size(rotary_pos_emb, dim=2)

    hidden_states = torch.chunk(hidden_states, sp_world_size, dim=1)[sp_rank]
    rotary_pos_emb = torch.chunk(rotary_pos_emb, sp_world_size, dim=2)[sp_rank]

    # Ernie attends over a single joint image+text sequence, so shard both together.
    for layer in self.layers:
        hidden_states = layer(hidden_states, rotary_pos_emb, temb)

    hidden_states = get_sp_group().all_gather(hidden_states.contiguous(), dim=1)
    hidden_states = hidden_states[:, :hidden_states_orig_size, :]

    hidden_states = self.final_norm(hidden_states, conditioning).type_as(hidden_states)

    patches = self.final_linear(hidden_states)[:, :num_image_tokens, :]
    output = patches.view(batch_size, patches_h, patches_w, patch_size, patch_size, self.out_channels)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch_size, self.out_channels, height, width)
    return output
