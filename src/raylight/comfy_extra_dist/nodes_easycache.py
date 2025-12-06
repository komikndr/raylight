import logging
import math
from typing import Any, Dict, Optional, Sequence, List

import torch

import comfy
import comfy.model_patcher
from comfy.patcher_extension import WrappersMP

from comfy_extras.nodes_easycache import EasyCacheHolder

from .ray_patch_decorator import ray_patch


class DistributedCacheMixin:
    def __init__(self, enable_sync: bool = True) -> None:
        self._enable_sync = enable_sync
        self._distributed = (
            enable_sync
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        self._dist_backend: Optional[str] = None
        if self._distributed:
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
            try:
                self._dist_backend = str(torch.distributed.get_backend())
            except Exception:
                self._dist_backend = None
        else:
            self._rank = 0
            self._world_size = 1

        self._sync_device = torch.device("cpu")
        if self._distributed:
            backend = str(self._dist_backend or "").lower()
            if backend in {"nccl", "cuda"} and torch.cuda.is_available():
                try:
                    current_device = torch.cuda.current_device()
                except RuntimeError:
                    current_device = 0
                self._sync_device = torch.device("cuda", current_device)
            else:
                self._sync_device = torch.device("cpu")

    @property
    def distributed_sync_enabled(self) -> bool:
        return self._distributed and self._enable_sync

    @property
    def is_log_rank(self) -> bool:
        return (not self._distributed) or (self._rank == 0)

    def _ensure_scalar_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.ndimension() > 0:
                tensor = tensor.reshape(())
            target_device = self._sync_device if self.distributed_sync_enabled else tensor.device
            if tensor.device != target_device:
                tensor = tensor.to(target_device)
            return tensor.to(dtype=torch.float32).clone()
        device = self._sync_device if self.distributed_sync_enabled else torch.device("cpu")
        return torch.tensor(float(value), device=device, dtype=torch.float32)

    def sync_scalar(self, value: Optional[Any], op: str = "max") -> Optional[float]:
        if value is None:
            return None
        if not self.distributed_sync_enabled:
            return float(value.item() if isinstance(value, torch.Tensor) else value)
        tensor = self._ensure_scalar_tensor(value)
        if op == "max":
            reduce_op = torch.distributed.ReduceOp.MAX
        elif op == "sum":
            reduce_op = torch.distributed.ReduceOp.SUM
        elif op == "mean":
            reduce_op = torch.distributed.ReduceOp.SUM
        else:
            raise ValueError(f"Unsupported reduction op '{op}'")
        torch.distributed.all_reduce(tensor, op=reduce_op)
        if op == "mean":
            tensor /= float(self._world_size)
        return float(tensor.item())

    def sync_bool(self, flag: bool, mode: str = "all") -> bool:
        if not self.distributed_sync_enabled:
            return flag
        t = torch.tensor(1 if flag else 0, device=self._sync_device, dtype=torch.int32)
        op = torch.distributed.ReduceOp.MIN if mode == "all" else torch.distributed.ReduceOp.MAX
        torch.distributed.all_reduce(t, op=op)
        return bool(int(t.item()) != 0)


class DistributedEasyCacheHolder(EasyCacheHolder, DistributedCacheMixin):
    def __init__(
        self,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        subsample_factor: int = 8,
        offload_cache_diff: bool = False,
        verbose: bool = False,
        distributed_sync: bool = True,
    ) -> None:
        EasyCacheHolder.__init__(
            self,
            reuse_threshold,
            start_percent,
            end_percent,
            subsample_factor,
            offload_cache_diff,
            verbose,
        )
        DistributedCacheMixin.__init__(self, enable_sync=distributed_sync)
        self.distributed_sync = distributed_sync

    def check_metadata(self, x: torch.Tensor) -> None:
        return

    def apply_cache_diff(self, x: torch.Tensor, uuids: List) -> torch.Tensor:
        if self.first_cond_uuid in uuids:
            self.total_steps_skipped += 1

        out = x
        batch_offset = out.shape[0] // max(len(uuids), 1)

        for i, uuid in enumerate(uuids):
            if uuid not in self.uuid_cache_diffs:
                continue
            xi = out[i * batch_offset: (i + 1) * batch_offset]
            di = self.uuid_cache_diffs[uuid].to(device=xi.device, dtype=xi.dtype)

            if xi.shape[1:] != di.shape[1:]:
                min_shape = tuple(min(a, b) for a, b in zip(xi.shape[1:], di.shape[1:]))
                crop = (slice(None),) + tuple(slice(0, s) for s in min_shape)
                xi_c = xi[crop]
                di_c = di[(slice(None),) + tuple(slice(0, s) for s in min_shape)]
                xi_c += di_c
            else:
                xi += di

            out[i * batch_offset: (i + 1) * batch_offset] = xi
        return out

    def update_cache_diff(self, output: torch.Tensor, x: torch.Tensor, uuids: List) -> None:
        batch_offset = output.shape[0] // max(len(uuids), 1)
        for i, uuid in enumerate(uuids):
            yo = output[i * batch_offset: (i + 1) * batch_offset]
            xi = x[i * batch_offset: (i + 1) * batch_offset]

            if yo.shape[1:] != xi.shape[1:]:
                min_shape = tuple(min(a, b) for a, b in zip(yo.shape[1:], xi.shape[1:]))
                slice_y = (slice(None),) + tuple(slice(0, s) for s in min_shape)
                slice_x = (slice(None),) + tuple(slice(0, s) for s in min_shape)
                yo = yo[slice_y]
                xi = xi[slice_x]
            self.uuid_cache_diffs[uuid] = (yo - xi).detach().clone()

    def clone(self):
        return DistributedEasyCacheHolder(
            self.reuse_threshold,
            self.start_percent,
            self.end_percent,
            self.subsample_factor,
            self.offload_cache_diff,
            self.verbose,
            distributed_sync=self.distributed_sync,
        )


def _extract_transformer_options(args: Sequence[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    transformer_options = args[-1]
    if not isinstance(transformer_options, dict):
        transformer_options = kwargs.get("transformer_options")
        if transformer_options is None:
            transformer_options = args[-2]
    return transformer_options


def distributed_easycache_forward_wrapper(executor, *args, **kwargs):
    x: torch.Tensor = args[0]
    transformer_options = _extract_transformer_options(args, kwargs)
    easycache: DistributedEasyCacheHolder = transformer_options["easycache"]
    sigmas = transformer_options["sigmas"]
    uuids = transformer_options["uuids"]

    if not uuids:
        return executor(*args, **kwargs)

    easycache.check_metadata(x)

    if easycache.first_cond_uuid is None:
        easycache.first_cond_uuid = uuids[0]
        easycache.initial_step = False
    elif easycache.first_cond_uuid not in uuids:
        easycache.first_cond_uuid = uuids[0]
        easycache.uuid_cache_diffs = {}
        easycache.x_prev_subsampled = None
        easycache.output_prev_subsampled = None
        easycache.output_prev_norm = None
        easycache.relative_transformation_rate = None
        easycache.cumulative_change_rate = 0.0
        easycache.skip_current_step = False

    if sigmas is not None and easycache.is_past_end_timestep(sigmas):
        return executor(*args, **kwargs)

    has_first_cond_uuid = easycache.has_first_cond_uuid(uuids)
    next_x_prev = x
    input_change_value: Optional[float] = None
    do_easycache = easycache.should_do_easycache(sigmas)

    if do_easycache and easycache.skip_current_step:

        x_slice = easycache.subsample(x, uuids, clone=False)
        easycache.x_prev_subsampled = x_slice.clone()
        if has_first_cond_uuid and easycache.first_cond_uuid in easycache.uuid_cache_diffs:
            diff_slice = easycache.uuid_cache_diffs[easycache.first_cond_uuid]

            xs = list(x_slice.shape); ds = list(diff_slice.shape)
            if xs[1:] != ds[1:]:
                min_shape = tuple(min(a, b) for a, b in zip(xs[1:], ds[1:]))
                crop = (slice(None),) + tuple(slice(0, s) for s in min_shape)
                approx_out = x_slice[crop] + diff_slice[crop].to(x_slice.device, x_slice.dtype)
            else:
                approx_out = x_slice + diff_slice.to(x_slice.device, x_slice.dtype)
            easycache.output_prev_subsampled = approx_out.detach().clone()
            easycache.output_prev_norm = easycache.sync_scalar(
                approx_out.flatten().abs().mean(), op="max"
            )
        if easycache.verbose and easycache.is_log_rank:
            logging.info("[EasyCache] SKIP(carry) — updated prev-step refs; returning cached diff.")
        return easycache.apply_cache_diff(x, uuids)

    if do_easycache and has_first_cond_uuid:
        if easycache.initial_step:
            easycache.first_cond_uuid = uuids[0]
            has_first_cond_uuid = easycache.has_first_cond_uuid(uuids)
            easycache.initial_step = False

        if easycache.has_x_prev_subsampled():
            diff = (easycache.subsample(x, uuids, clone=False) - easycache.x_prev_subsampled).flatten().abs().mean()
            input_change_value = easycache.sync_scalar(diff, op="max")

        if (
            easycache.has_output_prev_norm()
            and easycache.has_relative_transformation_rate()
            and input_change_value is not None
            and input_change_value > 0.0
        ):
            approx_output_change_rate = (
                easycache.relative_transformation_rate * input_change_value
            ) / (easycache.output_prev_norm + 1e-8)
            approx_output_change_rate = easycache.sync_scalar(approx_output_change_rate, op="max")
            easycache.cumulative_change_rate += approx_output_change_rate

            should_skip = (easycache.cumulative_change_rate < easycache.reuse_threshold)
            should_skip = easycache.sync_bool(should_skip, mode="all")

            if should_skip:
                easycache.skip_current_step = True

                x_slice = easycache.subsample(x, uuids, clone=False)
                easycache.x_prev_subsampled = x_slice.clone()
                if has_first_cond_uuid and easycache.first_cond_uuid in easycache.uuid_cache_diffs:
                    diff_slice = easycache.uuid_cache_diffs[easycache.first_cond_uuid]
                    xs = list(x_slice.shape); ds = list(diff_slice.shape)
                    if xs[1:] != ds[1:]:
                        min_shape = tuple(min(a, b) for a, b in zip(xs[1:], ds[1:]))
                        crop = (slice(None),) + tuple(slice(0, s) for s in min_shape)
                        approx_out = x_slice[crop] + diff_slice[crop].to(x_slice.device, x_slice.dtype)
                    else:
                        approx_out = x_slice + diff_slice.to(x_slice.device, x_slice.dtype)
                    easycache.output_prev_subsampled = approx_out.detach().clone()
                    easycache.output_prev_norm = easycache.sync_scalar(
                        approx_out.flatten().abs().mean(), op="max"
                    )

                if easycache.verbose and easycache.is_log_rank:
                    logging.info(
                        "[EasyCache] SKIP(decide) — cum=%.6f < thr=%.6f",
                        easycache.cumulative_change_rate,
                        easycache.reuse_threshold,
                    )
                return easycache.apply_cache_diff(x, uuids)
            else:
                easycache.cumulative_change_rate = 0.0

    output: torch.Tensor = executor(*args, **kwargs)

    if has_first_cond_uuid:
        out_sub = easycache.subsample(output, uuids, clone=False)
        if easycache.has_output_prev_norm():
            output_change = (out_sub - easycache.output_prev_subsampled).flatten().abs().mean()
            output_change_value = easycache.sync_scalar(output_change, op="max")
            output_change_rate = output_change_value / (easycache.output_prev_norm + 1e-8)
            if easycache.verbose and easycache.is_log_rank:
                logging.info("[EasyCache] compute — output_change_rate=%.6f", output_change_rate)

            if input_change_value is not None and input_change_value > 0.0:
                easycache.relative_transformation_rate = output_change_value / (input_change_value + 1e-8)
            else:
                easycache.relative_transformation_rate = None

    easycache.update_cache_diff(output, next_x_prev, uuids)
    if has_first_cond_uuid:
        easycache.x_prev_subsampled = easycache.subsample(next_x_prev, uuids)
        easycache.output_prev_subsampled = easycache.subsample(output, uuids)

        easycache.output_prev_norm = easycache.sync_scalar(
            easycache.output_prev_subsampled.flatten().abs().mean(), op="max"
        )
        if easycache.verbose and easycache.is_log_rank:
            logging.info("EasyCache — updated prev refs; x_prev_subsampled shape=%s",
                         tuple(easycache.x_prev_subsampled.shape))
    return output


def distributed_easycache_calc_cond_batch_wrapper(executor, *args, **kwargs):
    model_options = args[-1]
    if not isinstance(model_options, dict):
        model_options = kwargs.get("model_options")
        if model_options is None:
            model_options = args[-2]
    easycache: DistributedEasyCacheHolder = model_options["transformer_options"]["easycache"]
    easycache.skip_current_step = False
    return executor(*args, **kwargs)


def distributed_easycache_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    orig_model_options = guider.model_options
    guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
    try:
        easycache: DistributedEasyCacheHolder = (
            guider.model_options["transformer_options"]["easycache"]
            .clone()
            .prepare_timesteps(guider.model_patcher.model.model_sampling)
        )
        guider.model_options["transformer_options"]["easycache"] = easycache
        if easycache.is_log_rank:
            logging.info(
                "EasyCache enabled — thr=%.3f start=%.2f end=%.2f",
                easycache.reuse_threshold, easycache.start_percent, easycache.end_percent,
            )
        return executor(*args, **kwargs)
    finally:
        easycache: DistributedEasyCacheHolder = guider.model_options["transformer_options"]["easycache"]
        try:
            total_steps = max(len(args[3]) - 1, 1)
        except Exception:
            total_steps = 1
        denom = max(total_steps - easycache.total_steps_skipped, 1)
        speedup = total_steps / denom
        if easycache.is_log_rank:
            logging.info("EasyCache — skipped %d/%d steps (%.2fx).",
                         easycache.total_steps_skipped, total_steps, speedup)
        easycache.reset()
        guider.model_options = orig_model_options


class DistributedTeaCacheHolder(DistributedEasyCacheHolder):
    def __init__(
        self,
        threshold: float,
        start_percent: float,
        end_percent: float,
        warmup_percent: float,
        retention_interval: int,
        subsample_factor: int,
        verbose: bool,
        distributed_sync: bool,
    ) -> None:
        super().__init__(
            threshold,
            start_percent,
            end_percent,
            subsample_factor=subsample_factor,
            offload_cache_diff=False,
            verbose=verbose,
            distributed_sync=distributed_sync,
        )
        self.name = "TeaCache"
        self.warmup_percent = float(warmup_percent)
        self.retention_interval = max(1, int(retention_interval))
        self.total_steps = 0
        self.step_index = 0
        self.warmup_steps = 0
        self.steps_since_compute = 0
        self.accumulated_change = 0.0
        self.prev_modulated: Optional[torch.Tensor] = None
        self.relative_change_history = []

    def clone(self):
        return DistributedTeaCacheHolder(
            self.reuse_threshold,
            self.start_percent,
            self.end_percent,
            self.warmup_percent,
            self.retention_interval,
            self.subsample_factor,
            self.verbose,
            self.distributed_sync,
        )

    def reset(self):
        super().reset()
        self.total_steps = 0
        self.step_index = 0
        self.warmup_steps = 0
        self.steps_since_compute = 0
        self.accumulated_change = 0.0
        self.prev_modulated = None
        self.relative_change_history = []
        return self

    def prepare_timesteps(self, model_sampling, total_steps: int) -> "DistributedTeaCacheHolder":
        super().prepare_timesteps(model_sampling)
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(1, math.ceil(self.total_steps * self.warmup_percent))
        self.step_index = 0
        self.steps_since_compute = 0
        self.accumulated_change = 0.0
        self.prev_modulated = None
        self.relative_change_history = []
        return self

    def begin_step(self):
        self.step_index += 1
        self.skip_current_step = False

    def _extract_sigma_value(self, sigmas) -> float:
        if isinstance(sigmas, torch.Tensor):
            if sigmas.numel() == 0:
                return 0.0
            return float(sigmas.reshape(-1)[0].item())
        if isinstance(sigmas, (list, tuple)):
            if not sigmas:
                return 0.0
            return float(sigmas[0])
        return float(sigmas)

    def _modulation_weight(self, sigma_value: float) -> float:
        sigma_sq = sigma_value * sigma_value
        alpha_cumprod = 1.0 / (sigma_sq + 1.0)
        return (1.0 - alpha_cumprod) / (alpha_cumprod + 1e-8)

    def modulate_input(self, tensor: torch.Tensor, sigma_value: float) -> torch.Tensor:
        return tensor * self._modulation_weight(sigma_value)

    def in_warmup(self) -> bool:
        return self.step_index <= self.warmup_steps

    def needs_retention(self) -> bool:
        return self.steps_since_compute >= self.retention_interval

    def record_relative_change(self, change: float) -> float:
        self.accumulated_change += change
        return self.accumulated_change

    def store_modulated(self, tensor: torch.Tensor):
        self.prev_modulated = tensor.detach().clone()

    def mark_skip(self):
        self.skip_current_step = True
        self.steps_since_compute += 1

    def mark_compute(self):
        self.accumulated_change = 0.0
        self.steps_since_compute = 0
        self.skip_current_step = False


def teacache_forward_wrapper(executor, *args, **kwargs):
    x: torch.Tensor = args[0]
    transformer_options = _extract_transformer_options(args, kwargs)
    teacache: DistributedTeaCacheHolder = transformer_options["teacache"]
    sigmas = transformer_options["sigmas"]
    uuids = transformer_options["uuids"]

    if not uuids:
        return executor(*args, **kwargs)

    teacache.check_metadata(x)

    if teacache.first_cond_uuid is None:
        teacache.first_cond_uuid = uuids[0]
        teacache.initial_step = False
    elif teacache.first_cond_uuid not in uuids:
        teacache.first_cond_uuid = uuids[0]
        teacache.uuid_cache_diffs = {}
        teacache.x_prev_subsampled = None
        teacache.output_prev_subsampled = None
        teacache.output_prev_norm = None
        teacache.prev_modulated = None
        teacache.relative_transformation_rate = None
        teacache.cumulative_change_rate = 0.0
        teacache.accumulated_change = 0.0
        teacache.steps_since_compute = 0
        teacache.skip_current_step = False

    sigma_value = teacache._extract_sigma_value(sigmas)
    consider_cache = teacache.should_do_easycache(sigmas) and not teacache.is_past_end_timestep(sigmas)

    cond_slice = teacache.subsample(x, uuids, clone=False)
    modulated_input = teacache.modulate_input(cond_slice, sigma_value)

    if teacache.prev_modulated is None:
        teacache.store_modulated(modulated_input)
        output = executor(*args, **kwargs)
        teacache.update_cache_diff(output, x, uuids)
        teacache.x_prev_subsampled = teacache.subsample(x, uuids)
        teacache.output_prev_subsampled = teacache.subsample(output, uuids)
        teacache.output_prev_norm = teacache.sync_scalar(
            teacache.output_prev_subsampled.flatten().abs().mean(), op="max"
        )
        teacache.mark_compute()
        return output

    curr, prev = modulated_input, teacache.prev_modulated
    if curr.shape != prev.shape:
        min_shape = tuple(min(a, b) for a, b in zip(curr.shape, prev.shape))
        slices = tuple(slice(0, s) for s in min_shape)
        curr = curr[slices]
        prev = prev[slices]

    diff = torch.abs(curr - prev)
    ref = torch.abs(prev)
    relative_l1 = diff.sum() / (ref.sum() + 1e-8)
    relative_change = teacache.sync_scalar(relative_l1, op="max")
    if teacache.verbose and teacache.is_log_rank:
        teacache.relative_change_history.append(relative_change)

    force_compute = (not consider_cache) or teacache.in_warmup() or teacache.needs_retention()

    if not force_compute:
        acc = teacache.record_relative_change(relative_change)
        should_skip = (acc < teacache.reuse_threshold)
        should_skip = teacache.sync_bool(should_skip, mode="all")
        if should_skip:
            teacache.store_modulated(modulated_input)
            teacache.mark_skip()
            if teacache.verbose and teacache.is_log_rank:
                logging.info(
                    "TeaCache — SKIP; acc=%.6f < thr=%.6f; since=%d",
                    acc, teacache.reuse_threshold, teacache.steps_since_compute,
                )
            return teacache.apply_cache_diff(x, uuids)
        force_compute = True

    output: torch.Tensor = executor(*args, **kwargs)
    teacache.update_cache_diff(output, x, uuids)
    teacache.x_prev_subsampled = teacache.subsample(x, uuids)
    teacache.output_prev_subsampled = teacache.subsample(output, uuids)
    teacache.output_prev_norm = teacache.sync_scalar(
        teacache.output_prev_subsampled.flatten().abs().mean(), op="max"
    )
    teacache.store_modulated(modulated_input)
    teacache.mark_compute()
    if teacache.verbose and teacache.is_log_rank:
        logging.info("TeaCache — COMPUTE; accumulated_change reset.")
    return output


def teacache_calc_cond_batch_wrapper(executor, *args, **kwargs):
    model_options = args[-1]
    if not isinstance(model_options, dict):
        model_options = kwargs.get("model_options")
        if model_options is None:
            model_options = args[-2]
    teacache: DistributedTeaCacheHolder = model_options["transformer_options"]["teacache"]
    teacache.begin_step()
    return executor(*args, **kwargs)


def teacache_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    orig_model_options = guider.model_options
    guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
    sigmas = args[3] if len(args) > 3 else ()
    total_steps = max(len(sigmas) - 1, 1) if hasattr(sigmas, "__len__") else 1
    try:
        teacache: DistributedTeaCacheHolder = (
            guider.model_options["transformer_options"]["teacache"]
            .clone()
            .prepare_timesteps(guider.model_patcher.model.model_sampling, total_steps)
        )
        guider.model_options["transformer_options"]["teacache"] = teacache
        if teacache.is_log_rank:
            logging.info(
                "TeaCache enabled — thr=%.3f warmup=%.2f retain=%d start=%.2f end=%.2f",
                teacache.reuse_threshold,
                teacache.warmup_percent,
                teacache.retention_interval,
                teacache.start_percent,
                teacache.end_percent,
            )
        return executor(*args, **kwargs)
    finally:
        teacache: DistributedTeaCacheHolder = guider.model_options["transformer_options"]["teacache"]
        denom = max(total_steps - teacache.total_steps_skipped, 1)
        speedup = total_steps / denom
        if teacache.is_log_rank:
            logging.info("TeaCache — skipped %d/%d steps (%.2fx).",
                         teacache.total_steps_skipped, total_steps, speedup)
        teacache.reset()
        guider.model_options = orig_model_options


class RayEasyCacheNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "reuse_threshold": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 3.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "subsample_factor": ("INT", {"default": 8, "min": 1, "max": 16}),
                "verbose": ("BOOLEAN", {"default": False}),
                "distributed_sync": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    @ray_patch
    def patch(
        self,
        model,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        subsample_factor: int = 8,
        verbose: bool = False,
        distributed_sync: bool = True,
    ):
        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})
        transformer_options["easycache"] = DistributedEasyCacheHolder(
            reuse_threshold,
            start_percent,
            end_percent,
            subsample_factor=subsample_factor,
            offload_cache_diff=False,
            verbose=verbose,
            distributed_sync=distributed_sync,
        )

        m.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, "easycache")
        m.remove_wrappers_with_key(WrappersMP.CALC_COND_BATCH, "easycache")
        m.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, "easycache")

        m.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "easycache", distributed_easycache_sample_wrapper)
        m.add_wrapper_with_key(WrappersMP.CALC_COND_BATCH, "easycache", distributed_easycache_calc_cond_batch_wrapper)
        m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "easycache", distributed_easycache_forward_wrapper)
        return m


class RayTeaCacheNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 5.0, "step": 0.01}),
                "warmup_percent": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.5, "step": 0.01}),
                "retention_interval": ("INT", {"default": 8, "min": 1, "max": 64}),
                "start_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "subsample_factor": ("INT", {"default": 8, "min": 1, "max": 16}),
                "verbose": ("BOOLEAN", {"default": False}),
                "distributed_sync": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    @ray_patch
    def patch(
        self,
        model,
        threshold: float,
        warmup_percent: float,
        retention_interval: int,
        start_percent: float,
        end_percent: float,
        subsample_factor: int = 8,
        verbose: bool = False,
        distributed_sync: bool = True,
    ):
        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})
        transformer_options["teacache"] = DistributedTeaCacheHolder(
            threshold,
            start_percent,
            end_percent,
            warmup_percent,
            retention_interval,
            subsample_factor,
            verbose,
            distributed_sync,
        )

        m.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, "teacache")
        m.remove_wrappers_with_key(WrappersMP.CALC_COND_BATCH, "teacache")
        m.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, "teacache")

        m.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "teacache", teacache_sample_wrapper)
        m.add_wrapper_with_key(WrappersMP.CALC_COND_BATCH, "teacache", teacache_calc_cond_batch_wrapper)
        m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "teacache", teacache_forward_wrapper)
        return m


NODE_CLASS_MAPPINGS = {
    "RayEasyCache": RayEasyCacheNode,
    "RayTeaCache": RayTeaCacheNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayEasyCache": "EasyCache (Ray)",
    "RayTeaCache": "TeaCache (Ray)",
}
