import contextlib
import math
from dataclasses import dataclass

import comfy.ldm.modules.attention
import comfy.samplers
import comfy.utils
import node_helpers
import torch
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import CallbacksMP


def _sigma_scalar(sigma):
    if isinstance(sigma, torch.Tensor):
        return float(sigma.reshape(-1)[0].item())
    return float(sigma)


def parse_skip_blocks(skip_blocks):
    if skip_blocks is None:
        return []
    if isinstance(skip_blocks, str):
        return [int(n.strip()) for n in skip_blocks.split(",") if n.strip()]
    return [int(n) for n in skip_blocks]


def parse_stg_layers_indices(stg_layers_indices):
    if isinstance(stg_layers_indices, list):
        return [[int(n) for n in indices] for indices in stg_layers_indices]

    chunks = [s + "]" for s in stg_layers_indices.split("],")[:-1]]
    if stg_layers_indices.strip():
        chunks.append(stg_layers_indices.split("],")[-1])

    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk == "[]":
            result.append([])
            continue

        chunk = chunk.strip("[]").strip()
        if not chunk:
            result.append([])
            continue

        result.append([int(value.strip()) for value in chunk.split(",") if value.strip()])
    return result


def stg(noise_pred_pos, noise_pred_neg, noise_pred_perturbed, cfg_scale, stg_scale, rescale_scale):
    noise_pred = (
        noise_pred_pos
        + (cfg_scale - 1) * (noise_pred_pos - noise_pred_neg)
        + stg_scale * (noise_pred_pos - noise_pred_perturbed)
    )
    if rescale_scale != 0:
        factor = noise_pred_pos.std() / noise_pred.std()
        factor = rescale_scale * factor + (1 - rescale_scale)
        noise_pred = noise_pred * factor
    return noise_pred


def project(v0, v1):
    dtype = v0.dtype
    v0 = v0.double()
    v1 = torch.nn.functional.normalize(v1.double(), dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def apg(noise_pred_pos, noise_pred_neg, cfg_scale, momentum_buffer=None, eta=1.0, norm_threshold=0.0):
    del momentum_buffer
    diff = noise_pred_pos - noise_pred_neg
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, noise_pred_pos)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return noise_pred_pos + (cfg_scale - 1) * normalized_update


@dataclass
class STGFlag:
    do_skip: bool = False
    skip_layers: list[int] | None = None


class PatchAttention(contextlib.AbstractContextManager):
    def __init__(self, attn_idx=None):
        self.current_idx = -1
        if isinstance(attn_idx, int):
            self.attn_idx = [attn_idx]
        elif attn_idx is None:
            self.attn_idx = [0]
        else:
            self.attn_idx = list(attn_idx)

    def __enter__(self):
        self.original_attention = comfy.ldm.modules.attention.optimized_attention
        self.original_attention_masked = comfy.ldm.modules.attention.optimized_attention_masked
        comfy.ldm.modules.attention.optimized_attention = self.stg_attention
        comfy.ldm.modules.attention.optimized_attention_masked = self.stg_attention_masked

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        comfy.ldm.modules.attention.optimized_attention = self.original_attention
        comfy.ldm.modules.attention.optimized_attention_masked = self.original_attention_masked
        self.original_attention = None
        self.original_attention_masked = None

    def stg_attention(self, q, k, v, heads, *args, **kwargs):
        self.current_idx += 1
        if self.current_idx in self.attn_idx:
            return v
        return self.original_attention(q, k, v, heads, *args, **kwargs)

    def stg_attention_masked(self, q, k, v, heads, *args, **kwargs):
        self.current_idx += 1
        if self.current_idx in self.attn_idx:
            return v
        return self.original_attention_masked(q, k, v, heads, *args, **kwargs)


class STGBlockWrapper:
    def __init__(self, block, stg_flag, idx):
        self.block = block
        self.flag = stg_flag
        self.idx = idx

    def __call__(self, args, extra_args):
        context_manager = contextlib.nullcontext()
        stg_indexes = args["transformer_options"].get("stg_indexes", [0])
        if self.flag.do_skip and self.idx in (self.flag.skip_layers or []):
            context_manager = PatchAttention(stg_indexes)
        with context_manager:
            return extra_args["original_block"](args)


def get_transformer_blocks(model):
    diffusion_model = model.get_model_object("diffusion_model")
    key = "diffusion_model.transformer_blocks"
    if diffusion_model.__class__.__name__ == "LTXVTransformer3D":
        key = "diffusion_model.transformer.transformer_blocks"
    return model.get_model_object(key)


def patch_stg_model(model, stg_flag):
    transformer_blocks = get_transformer_blocks(model)
    for index, block in enumerate(transformer_blocks):
        model.set_model_patch_replace(STGBlockWrapper(block, stg_flag, index), "dit", "double_block", index)


class RaySTGGuider(comfy.samplers.CFGGuider):
    def __init__(self, model, cfg, stg_scale, rescale_scale):
        model = model.clone()
        super().__init__(model)
        self.stg_flag = STGFlag(do_skip=False, skip_layers=model.model_options.get("transformer_options", {}).get("skip_block_list", []))
        patch_stg_model(model, self.stg_flag)
        self.cfg = cfg
        self.stg_scale = stg_scale
        self.rescale_scale = rescale_scale

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        del seed
        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        if model_options.get("sigma_to_params_mapping") is not None:
            cfg_value, stg_scale, stg_layer_indices, stg_rescale = model_options["sigma_to_params_mapping"](_sigma_scalar(timestep))
            self.stg_flag.skip_layers = stg_layer_indices
            patch_stg_model(self.model_patcher, self.stg_flag)
        else:
            cfg_value = self.cfg
            stg_scale = self.stg_scale
            stg_rescale = self.rescale_scale

        noise_pred_pos = comfy.samplers.calc_cond_batch(self.inner_model, [positive_cond], x, timestep, model_options)[0]

        noise_pred_neg = 0
        noise_pred_perturbed = 0

        if not math.isclose(cfg_value, 1.0):
            noise_pred_neg = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond], x, timestep, model_options)[0]

        if not math.isclose(stg_scale, 0.0):
            try:
                model_options["transformer_options"]["ptb_index"] = 0
                self.stg_flag.do_skip = True
                noise_pred_perturbed = comfy.samplers.calc_cond_batch(self.inner_model, [positive_cond], x, timestep, model_options)[0]
            finally:
                self.stg_flag.do_skip = False
                del model_options["transformer_options"]["ptb_index"]

        stg_result = stg(noise_pred_pos, noise_pred_neg, noise_pred_perturbed, cfg_value, stg_scale, stg_rescale)

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": stg_result,
                "cond": positive_cond,
                "uncond": negative_cond,
                "model": self.inner_model,
                "uncond_denoised": noise_pred_neg,
                "cond_denoised": noise_pred_pos,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
                "perturbed_cond": positive_cond,
                "perturbed_cond_denoised": noise_pred_perturbed,
            }
            stg_result = fn(args)

        return stg_result


class RaySTGGuiderAdvanced(comfy.samplers.CFGGuider):
    def __init__(
        self,
        model,
        sigma_list,
        cfg_list,
        stg_scale_list,
        stg_rescale_list,
        stg_layers_indices_list,
        skip_steps_sigma_threshold,
        cfg_star_rescale,
        apply_apg,
        apg_cfg_scale,
        eta,
        norm_threshold,
    ):
        model = model.clone()
        super().__init__(model)
        self.stg_flag = STGFlag(do_skip=False, skip_layers=model.model_options.get("transformer_options", {}).get("skip_block_list", []))
        self.sigma_list = sigma_list
        self.cfg_list = cfg_list
        self.stg_scale_list = stg_scale_list
        self.stg_rescale_list = stg_rescale_list
        self.stg_layers_indices_list = stg_layers_indices_list
        self.skip_steps_sigma_threshold = skip_steps_sigma_threshold
        self.cfg_star_rescale = cfg_star_rescale
        self.apply_apg = apply_apg
        self.apg_cfg_scale = apg_cfg_scale
        self.eta = eta
        self.norm_threshold = norm_threshold
        patch_stg_model(model, self.stg_flag)

    def sigma_to_params_mapping(self, sigma):
        higher_sigmas = [value for value in self.sigma_list if value >= sigma]
        if not higher_sigmas:
            closest_idx = -1
        else:
            closest_higher = min(higher_sigmas)
            closest_idx = self.sigma_list.index(closest_higher)
        return (
            self.cfg_list[closest_idx],
            self.stg_scale_list[closest_idx],
            self.stg_rescale_list[closest_idx],
            self.stg_layers_indices_list[closest_idx],
        )

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        del seed
        sigma = _sigma_scalar(timestep)
        if sigma > self.skip_steps_sigma_threshold:
            return torch.zeros_like(x)

        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        cfg_value, stg_scale, stg_rescale, stg_layer_indices = self.sigma_to_params_mapping(sigma)
        if stg_layer_indices is not None:
            self.stg_flag.skip_layers = stg_layer_indices
            patch_stg_model(self.model_patcher, self.stg_flag)

        noise_pred_pos = comfy.samplers.calc_cond_batch(self.inner_model, [positive_cond], x, timestep, model_options)[0]

        noise_pred_neg = 0
        noise_pred_perturbed = 0

        if not math.isclose(cfg_value, 1.0) or (self.apply_apg and not math.isclose(self.apg_cfg_scale, 1.0)):
            noise_pred_neg = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond], x, timestep, model_options)[0]
            if self.cfg_star_rescale:
                batch_size = noise_pred_pos.shape[0]
                positive_flat = noise_pred_pos.view(batch_size, -1)
                negative_flat = noise_pred_neg.view(batch_size, -1)
                dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
                alpha = dot_product / squared_norm
                noise_pred_neg = alpha * noise_pred_neg

        if not math.isclose(stg_scale, 0.0):
            try:
                model_options["transformer_options"]["ptb_index"] = 0
                self.stg_flag.do_skip = True
                noise_pred_perturbed = comfy.samplers.calc_cond_batch(self.inner_model, [positive_cond], x, timestep, model_options)[0]
            finally:
                self.stg_flag.do_skip = False
                del model_options["transformer_options"]["ptb_index"]

        stg_result = stg(noise_pred_pos, noise_pred_neg, noise_pred_perturbed, cfg_value, stg_scale, stg_rescale)

        if self.apply_apg:
            stg_result = apg(
                stg_result,
                noise_pred_neg,
                cfg_scale=self.apg_cfg_scale,
                momentum_buffer=None,
                eta=self.eta,
                norm_threshold=self.norm_threshold,
            )

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": stg_result,
                "cond": positive_cond,
                "uncond": negative_cond,
                "model": self.inner_model,
                "uncond_denoised": noise_pred_neg,
                "cond_denoised": noise_pred_pos,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
                "perturbed_cond": positive_cond,
                "perturbed_cond_denoised": noise_pred_perturbed,
            }
            stg_result = fn(args)

        return stg_result


class RayLTXVGuiderParameters:
    def __init__(
        self,
        cfg_scale=1.0,
        stg_scale=0.0,
        perturb_attn=True,
        rescale_scale=0.0,
        modality_scale=1.0,
        skip_step=0,
        cross_attn=True,
        cfg_zero_star=False,
        zero_init_sigma=1.0,
    ):
        self.cfg_scale = cfg_scale
        self.stg_scale = stg_scale
        self.perturb_attn = perturb_attn
        self.rescale_scale = rescale_scale
        self.modality_scale = modality_scale
        self.skip_step = skip_step
        self.cross_attn = cross_attn
        self.cfg_zero_star = cfg_zero_star
        self.zero_init_sigma = zero_init_sigma

    def calculate(self, noise_pred_pos, noise_pred_neg, noise_pred_perturbed, noise_pred_modality):
        noise_pred = (
            noise_pred_pos
            + (self.cfg_scale - 1) * (noise_pred_pos - noise_pred_neg)
            + self.stg_scale * (noise_pred_pos - noise_pred_perturbed)
            + (self.modality_scale - 1) * (noise_pred_pos - noise_pred_modality)
        )
        if self.rescale_scale != 0:
            factor = noise_pred_pos.std() / noise_pred.std()
            factor = self.rescale_scale * factor + (1 - self.rescale_scale)
            noise_pred = noise_pred * factor
        return noise_pred

    def do_uncond(self):
        return not math.isclose(self.cfg_scale, 1.0)

    def do_perturbed(self):
        return not math.isclose(self.stg_scale, 0.0)

    def do_modality(self):
        return not math.isclose(self.modality_scale, 1.0)

    def do_skip(self, step):
        if self.skip_step == 0:
            return False
        return step % (self.skip_step + 1) != 0

    def do_cross_attn(self, step):
        return self.cross_attn and not self.do_skip(step)


def _coerce_guider_parameter(value):
    if isinstance(value, RayLTXVGuiderParameters):
        return value
    if isinstance(value, dict):
        return RayLTXVGuiderParameters(
            cfg_scale=float(value.get("cfg_scale", value.get("cfg", 1.0))),
            stg_scale=float(value.get("stg_scale", value.get("stg", 0.0))),
            perturb_attn=bool(value.get("perturb_attn", True)),
            rescale_scale=float(value.get("rescale_scale", value.get("rescale", 0.0))),
            modality_scale=float(value.get("modality_scale", 1.0)),
            skip_step=int(value.get("skip_step", 0)),
            cross_attn=bool(value.get("cross_attn", True)),
            cfg_zero_star=bool(value.get("cfg_zero_star", False)),
            zero_init_sigma=float(value.get("zero_init_sigma", 1.0)),
        )
    return RayLTXVGuiderParameters(
        cfg_scale=float(getattr(value, "cfg_scale", getattr(value, "cfg", 1.0))),
        stg_scale=float(getattr(value, "stg_scale", getattr(value, "stg", 0.0))),
        perturb_attn=bool(getattr(value, "perturb_attn", True)),
        rescale_scale=float(getattr(value, "rescale_scale", getattr(value, "rescale", 0.0))),
        modality_scale=float(getattr(value, "modality_scale", 1.0)),
        skip_step=int(getattr(value, "skip_step", 0)),
        cross_attn=bool(getattr(value, "cross_attn", True)),
        cfg_zero_star=bool(getattr(value, "cfg_zero_star", False)),
        zero_init_sigma=float(getattr(value, "zero_init_sigma", 1.0)),
    )


class RayMultimodalGuider(comfy.samplers.CFGGuider):
    def __init__(self, model, parameters, skip_blocks):
        model = model.clone()
        self.current_step = 0
        self.last_denoised_v = None
        self.last_denoised_a = None
        model.add_callback_with_key(CallbacksMP.ON_PRE_RUN, "mm_guider_on_pre_run", self.reset_current_step)
        super().__init__(model)
        self.stg_flag = STGFlag(do_skip=False, skip_layers=skip_blocks)
        patch_stg_model(model, self.stg_flag)
        self.parameters = {key: _coerce_guider_parameter(value) for key, value in parameters.items()}

    def reset_current_step(self, model_patcher=None):
        del model_patcher
        self.current_step = 0
        self.last_denoised_v = None
        self.last_denoised_a = None

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def calc_stg_indexes(self, run_vx, run_ax, audio_ptb, video_ptb):
        stg_indexes = set()
        num_self_attns = int(run_vx) + int(run_ax)
        video_attn_idx = 0
        audio_attn_idx = 0 if num_self_attns == 1 else 2
        if video_ptb:
            stg_indexes.add(video_attn_idx)
        if audio_ptb:
            stg_indexes.add(audio_attn_idx)
        return list(stg_indexes)

    def unpack_latents(self, x):
        latent_shapes = self.conds.get("positive", {})[0].get("model_conds", {}).get("latent_shapes", None).cond
        return comfy.utils.unpack_latents(x, latent_shapes)

    def pack_latents(self, vx, ax):
        return comfy.utils.pack_latents([vx, ax])

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        del seed
        current_step = self.current_step
        self.current_step += 1

        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        audio_params = self.parameters.get("AUDIO", RayLTXVGuiderParameters())
        video_params = self.parameters.get("VIDEO", RayLTXVGuiderParameters())

        run_vx = not video_params.do_skip(current_step)
        run_ax = not audio_params.do_skip(current_step)
        run_a2v = video_params.do_cross_attn(current_step)
        run_v2a = audio_params.do_cross_attn(current_step)

        vx, ax = self.unpack_latents(x)
        if not run_vx:
            vx = self.last_denoised_v
        if not run_ax:
            ax = self.last_denoised_a
        x, _ = self.pack_latents(vx, ax)

        if not run_vx and not run_ax:
            return x

        try:
            model_options["transformer_options"]["run_vx"] = run_vx
            model_options["transformer_options"]["run_ax"] = run_ax
            model_options["transformer_options"]["a2v_cross_attn"] = run_a2v
            model_options["transformer_options"]["v2a_cross_attn"] = run_v2a
            noise_pred_pos = comfy.samplers.calc_cond_batch(self.inner_model, [positive_cond], x, timestep, model_options)[0]
        finally:
            del model_options["transformer_options"]["run_vx"]
            del model_options["transformer_options"]["run_ax"]
            del model_options["transformer_options"]["a2v_cross_attn"]
            del model_options["transformer_options"]["v2a_cross_attn"]
        v_noise_pred_pos, a_noise_pred_pos = self.unpack_latents(noise_pred_pos)

        a_noise_pred_neg = v_noise_pred_neg = 0
        a_noise_pred_perturbed = v_noise_pred_perturbed = 0
        a_noise_pred_modality = v_noise_pred_modality = 0

        if any(params.do_uncond() for params in [audio_params, video_params]):
            try:
                model_options["transformer_options"]["run_vx"] = run_vx
                model_options["transformer_options"]["run_ax"] = run_ax
                model_options["transformer_options"]["a2v_cross_attn"] = run_a2v
                model_options["transformer_options"]["v2a_cross_attn"] = run_v2a
                noise_pred_neg = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond], x, timestep, model_options)[0]
                v_noise_pred_neg, a_noise_pred_neg = self.unpack_latents(noise_pred_neg)
            finally:
                del model_options["transformer_options"]["run_vx"]
                del model_options["transformer_options"]["run_ax"]
                del model_options["transformer_options"]["a2v_cross_attn"]
                del model_options["transformer_options"]["v2a_cross_attn"]

        if any(params.do_perturbed() for params in [audio_params, video_params]):
            try:
                stg_indexes = self.calc_stg_indexes(run_vx, run_ax and ax.numel() > 0, audio_params.perturb_attn, video_params.perturb_attn)
                model_options["transformer_options"]["run_vx"] = run_vx
                model_options["transformer_options"]["run_ax"] = run_ax
                model_options["transformer_options"]["a2v_cross_attn"] = run_a2v
                model_options["transformer_options"]["v2a_cross_attn"] = run_v2a
                model_options["transformer_options"]["ptb_index"] = 0
                model_options["transformer_options"]["stg_indexes"] = stg_indexes
                self.stg_flag.do_skip = True
                noise_pred_perturbed = comfy.samplers.calc_cond_batch(self.inner_model, [positive_cond], x, timestep, model_options)[0]
                v_noise_pred_perturbed, a_noise_pred_perturbed = self.unpack_latents(noise_pred_perturbed)
            finally:
                self.stg_flag.do_skip = False
                del model_options["transformer_options"]["ptb_index"]
                del model_options["transformer_options"]["run_vx"]
                del model_options["transformer_options"]["run_ax"]
                del model_options["transformer_options"]["a2v_cross_attn"]
                del model_options["transformer_options"]["v2a_cross_attn"]
                del model_options["transformer_options"]["stg_indexes"]

        if any(params.do_modality() for params in [audio_params, video_params]):
            try:
                model_options["transformer_options"]["run_vx"] = run_vx
                model_options["transformer_options"]["run_ax"] = run_ax
                model_options["transformer_options"]["a2v_cross_attn"] = False
                model_options["transformer_options"]["v2a_cross_attn"] = False
                noise_pred_modality = comfy.samplers.calc_cond_batch(self.inner_model, [positive_cond], x, timestep, model_options)[0]
                v_noise_pred_modality, a_noise_pred_modality = self.unpack_latents(noise_pred_modality)
            finally:
                del model_options["transformer_options"]["a2v_cross_attn"]
                del model_options["transformer_options"]["v2a_cross_attn"]
                del model_options["transformer_options"]["run_vx"]
                del model_options["transformer_options"]["run_ax"]

        if run_vx:
            vx = video_params.calculate(v_noise_pred_pos, v_noise_pred_neg, v_noise_pred_perturbed, v_noise_pred_modality)
        else:
            vx = self.last_denoised_v

        if run_ax:
            ax = audio_params.calculate(a_noise_pred_pos, a_noise_pred_neg, a_noise_pred_perturbed, a_noise_pred_modality)
        else:
            ax = self.last_denoised_a

        x, _ = self.pack_latents(vx, ax)

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": x,
                "cond": positive_cond,
                "uncond": negative_cond,
                "model": self.inner_model,
                "uncond_denoised": noise_pred_neg,
                "cond_denoised": noise_pred_pos,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
                "perturbed_cond": positive_cond,
                "perturbed_cond_denoised": noise_pred_perturbed,
            }
            x = fn(args)

        self.last_denoised_v, self.last_denoised_a = self.unpack_latents(x)
        return x


def build_ltxv_ray_guider(model, guider_spec):
    guider_type = guider_spec.get("type")

    if guider_type == "ltxv_stg":
        guider = RaySTGGuider(
            model,
            guider_spec["cfg"],
            guider_spec["stg_scale"],
            guider_spec["rescale_scale"],
        )
        guider.set_conds(guider_spec["positive"], guider_spec["negative"])
        return guider

    if guider_type == "ltxv_stg_advanced":
        guider = RaySTGGuiderAdvanced(
            model,
            guider_spec["sigma_list"],
            guider_spec["cfg_list"],
            guider_spec["stg_scale_list"],
            guider_spec["stg_rescale_list"],
            guider_spec["stg_layers_indices_list"],
            guider_spec["skip_steps_sigma_threshold"],
            guider_spec["cfg_star_rescale"],
            guider_spec["apply_apg"],
            guider_spec["apg_cfg_scale"],
            guider_spec["eta"],
            guider_spec["norm_threshold"],
        )
        guider.set_conds(guider_spec["positive"], guider_spec["negative"])
        return guider

    if guider_type == "ltxv_multimodal":
        guider = RayMultimodalGuider(
            model,
            guider_spec["parameters"],
            guider_spec["skip_blocks"],
        )
        guider.set_conds(guider_spec["positive"], guider_spec["negative"])
        return guider

    return None
