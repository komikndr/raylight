import torch
import torch.nn.functional as F
from einops import rearrange
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

import comfy
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from ..utils import pad_to_world_size
import raylight.distributed_modules.attention as xfuser_attn


attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def usp_dit_forward(self, x, timesteps, context, attention_mask=None, ref_latents=None, transformer_options={}, **kwargs):
    transformer_options = transformer_options.copy()
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, _, h_orig, w_orig = x.shape

    context = self._unpack_context(context)
    img, imgpos, h_, w_ = self.process_img(x)
    img_tokens = img.shape[1]
    timestep_zero_index = None
    ref_method = kwargs.get("ref_latents_method", self.default_ref_method)
    if ref_method is not None and ref_latents is not None and len(ref_latents) > 0:
        ref_tokens = []
        ref_pos = []
        ref_num_tokens = []
        for index, ref in enumerate(ref_latents, 1):
            if ref.ndim == 5:
                rb, rc, rt, rh5, rw5 = ref.shape
                ref = ref.reshape(rb * rt, rc, rh5, rw5)
            ref = comfy.utils.repeat_to_batch_size(ref, bs)
            ref_img, ref_imgpos, _, _ = self.process_img(ref, index=index)
            ref_tokens.append(ref_img)
            ref_pos.append(ref_imgpos)
            ref_num_tokens.append(ref_img.shape[1])
        img = torch.cat([img] + ref_tokens, dim=1)
        imgpos = torch.cat([imgpos] + ref_pos, dim=1)
        if ref_method == "index_timestep_zero":
            timestep_zero_index = img_tokens
        transformer_options["reference_image_num_tokens"] = ref_num_tokens

    img_total_tokens = img.shape[1]
    img = self.first(img)
    t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec = self.tproj(t)
    if timestep_zero_index is not None:
        t0 = self.tmlp(timestep_embedding(torch.zeros_like(timesteps), self.tdim).unsqueeze(1).to(img.dtype))
        tvec = torch.cat((tvec, self.tproj(t0)), dim=0)

    txtpos = torch.zeros(bs, context.shape[1], 3, device=img.device, dtype=torch.float32)
    context, _ = pad_to_world_size(context, dim=1)
    txtpos, _ = pad_to_world_size(txtpos, dim=1)
    img, _ = pad_to_world_size(img, dim=1)
    imgpos, _ = pad_to_world_size(imgpos, dim=1)

    sp_rank = get_sequence_parallel_rank()
    sp_world_size = get_sequence_parallel_world_size()
    local_img_tokens = img.shape[1] // sp_world_size
    local_img_start = sp_rank * local_img_tokens
    context = torch.chunk(context, sp_world_size, dim=1)[sp_rank]
    txtpos = torch.chunk(txtpos, sp_world_size, dim=1)[sp_rank]
    img = torch.chunk(img, sp_world_size, dim=1)[sp_rank]
    imgpos = torch.chunk(imgpos, sp_world_size, dim=1)[sp_rank]

    context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = self.txtmlp(context)

    patches = transformer_options.get("patches", {})
    if "post_input" in patches:
        for patch in patches["post_input"]:
            out = patch({"img": img, "txt": context, "img_ids": imgpos, "txt_ids": txtpos, "transformer_options": transformer_options})
            img, context = out["img"], out["txt"]
            imgpos, txtpos = out["img_ids"], out["txt_ids"]

    txtlen = context.shape[1]
    combined = torch.cat((context, img), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    freqs = self.pe_embedder(pos)
    if timestep_zero_index is not None:
        timestep_zero_index = txtlen + min(max(img_tokens - local_img_start, 0), img.shape[1])

    transformer_options["total_blocks"] = len(self.blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [txtlen, combined.shape[1]]
    for i, block in enumerate(self.blocks):
        transformer_options["block_index"] = i
        combined = block(combined, tvec, freqs, None, timestep_zero_index=timestep_zero_index, transformer_options=transformer_options)

    out = self.last(combined, t)[:, txtlen:, :]
    out = get_sp_group().all_gather(out.contiguous(), dim=1)
    out = out[:, :img_total_tokens, :]
    out = out[:, :img_tokens, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=self.patch, pw=self.patch, c=self.channels)
    out = out[:, :, :h_orig, :w_orig]
    if temporal:
        out = out.reshape(b5, t5, self.channels, h_orig, w_orig).movedim(1, 2)
    return out


def usp_attention_forward(self, x, freqs=None, mask=None, transformer_options={}, **kwargs):
    transformer_patches = transformer_options.get("patches", {})
    extra_options = transformer_options.copy()
    q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
    q, k = self.qknorm(q, k)

    if "block_index" in transformer_options and "attn1_patch" in transformer_patches:
        for patch in transformer_patches["attn1_patch"]:
            patched = patch(q, k, v, pe=freqs, attn_mask=mask, extra_options=extra_options)
            q = patched.get("q", q)
            k = patched.get("k", k)
            v = patched.get("v", v)
            freqs = patched.get("pe", freqs)
            mask = patched.get("attn_mask", mask)

    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if self.kvheads != self.heads:
        repeats = self.heads // self.kvheads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    out = xfuser_optimized_attention(q, k, v, self.heads, mask=mask, skip_reshape=True)

    if "block_index" in transformer_options and "attn1_output_patch" in transformer_patches:
        for patch in transformer_patches["attn1_output_patch"]:
            out = patch(out, extra_options)
    return self.wo(out * F.sigmoid(gate))
