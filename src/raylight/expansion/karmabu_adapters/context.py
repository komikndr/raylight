from queue import Empty
from types import SimpleNamespace
import logging

import ray
import torch
from ray.util.queue import Queue

import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.utils
import latent_preview

from raylight.distributed_worker.ray_worker import _generate_advanced_noise

from .registry import PROXY_ATTACHMENT, RAY_ACTORS_ATTACHMENT


def get_proxy_metadata(model):
    base_model = model.model
    return {
        "model_class": type(base_model).__name__,
        "latent_format": base_model.latent_format,
        "latent_format_class": type(base_model.latent_format).__name__,
    }


class DriverModelShell(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.latent_format = metadata["latent_format"]
        self.karmabu_metadata = metadata

    def forward(self, *args, **kwargs):
        raise RuntimeError("K3U Export MODEL is a driver-only proxy and cannot execute a model forward.")


def create_model_proxy(metadata, ray_actors=None):
    shell = DriverModelShell(metadata)
    load_device = comfy.model_management.get_torch_device()
    proxy = comfy.model_patcher.ModelPatcher(shell, load_device, torch.device("cpu"), size=0)
    proxy.set_attachments(PROXY_ATTACHMENT, dict(metadata))
    proxy.set_attachments(RAY_ACTORS_ATTACHMENT, ray_actors)
    return proxy


def prepare_wrapper_inputs(add_noise, noise_seed, latent):
    samples = latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise ValueError("the first K3U Adapter version requires a nested MiniMax H3 latent")
    streams = samples.unbind()
    if len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24:
        raise ValueError("the first K3U Adapter version supports only MiniMax H3 video+audio latents")

    # _generate_advanced_noise is the exact worker function. fork_rng restores
    # the driver's CPU RNG state after torch.manual_seed inside prepare_noise.
    with torch.random.fork_rng(devices=[]):
        noise, sampling_seed = _generate_advanced_noise(add_noise, noise_seed, latent)
    if not getattr(noise, "is_nested", False):
        raise ValueError("MiniMax H3 noise did not preserve the nested latent structure")

    packed_noise, latent_shapes = comfy.utils.pack_latents(noise.unbind())
    packed_latent, packed_shapes = comfy.utils.pack_latents(streams)
    if latent_shapes != packed_shapes:
        raise RuntimeError("noise and latent shapes differ")
    return packed_noise, packed_latent, latent_shapes, sampling_seed


class KarmabuAdapterContext:
    def __init__(self, model_proxy, spec, wrapper, queue_factory=Queue):
        self.model_proxy = model_proxy
        self.spec = spec
        self.wrapper = wrapper
        self.queue_factory = queue_factory
        self.preview_warning_logged = False

    def _warn_preview_once(self, exc):
        if not self.preview_warning_logged:
            self.preview_warning_logged = True
            logging.warning("[K3U Adapter] Preview disabled; sampling continues: %s", exc)

    def _consume_events(self, futures, event_queue, callback):
        pending = list(futures)
        preview_enabled = True
        worker_error = None
        while pending:
            try:
                event = event_queue.get(timeout=0.1)
            except Empty:
                ready, _ = ray.wait(pending, num_returns=1, timeout=0)
                for ref in ready:
                    try:
                        ray.get(ref)
                    except Exception as exc:
                        if worker_error is None:
                            worker_error = exc
                    pending.remove(ref)
                continue

            if event.get("kind") == "done":
                continue
            if event.get("kind") != "step":
                continue

            try:
                if preview_enabled:
                    x0_ref = event["x0_ref"][0]
                    x_ref = event["x_ref"][0]
                    x0, x = ray.get([x0_ref, x_ref])
                    callback(event["step"], x0, x, event["total_steps"])
            except Exception as exc:
                preview_enabled = False
                self._warn_preview_once(exc)
            finally:
                event.clear()

        if worker_error is not None:
            raise worker_error
        return ray.get(futures)

    def sample_xfuser_advanced(self, gpu_actors, add_noise, noise_seed, guider, sampler, sigmas, latent_image):
        metadata = self.model_proxy.get_attachment(PROXY_ATTACHMENT)
        if metadata.get("model_class") != "MiniMaxH3":
            self._warn_preview_once(
                ValueError(f"unsupported model {metadata.get('model_class')!r}; expected MiniMaxH3")
            )
            futures = [
                actor.custom_sampler_advanced.remote(add_noise, noise_seed, guider, sampler, sigmas, latent_image)
                for actor in gpu_actors
            ]
            return ray.get(futures)

        try:
            noise, packed_latent, latent_shapes, sampling_seed = prepare_wrapper_inputs(
                add_noise, noise_seed, latent_image
            )
        except Exception as exc:
            self._warn_preview_once(exc)
            futures = [
                actor.custom_sampler_advanced.remote(add_noise, noise_seed, guider, sampler, sigmas, latent_image)
                for actor in gpu_actors
            ]
            return ray.get(futures)

        try:
            event_queue = self.queue_factory(maxsize=1)
        except Exception as exc:
            self._warn_preview_once(exc)
            futures = [
                actor.custom_sampler_advanced.remote(add_noise, noise_seed, guider, sampler, sigmas, latent_image)
                for actor in gpu_actors
            ]
            return ray.get(futures)
        state = {"started": False, "finished": False, "futures": (), "results": None}

        def dispatch(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
            del noise, latent_image, denoise_mask, disable_pbar, seed, latent_shapes
            state["started"] = True
            futures = [
                actor.custom_sampler_advanced.remote(
                    add_noise,
                    noise_seed,
                    guider,
                    sampler,
                    sigmas,
                    latent_image_original,
                    adapter_event_queue=event_queue,
                )
                for actor in gpu_actors
            ]
            state["futures"] = futures
            try:
                state["results"] = self._consume_events(futures, event_queue, callback)
            except Exception:
                state["finished"] = True
                raise
            state["finished"] = True
            return state["results"]

        latent_image_original = latent_image
        guider_shell = SimpleNamespace(model_patcher=self.model_proxy)
        default_callback = latent_preview.prepare_callback(self.model_proxy, sigmas.shape[-1] - 1)
        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            dispatch, guider_shell, [self.wrapper]
        )

        try:
            return executor.execute(
                noise,
                packed_latent,
                sampler,
                sigmas,
                None,
                default_callback,
                False,
                sampling_seed,
                latent_shapes=latent_shapes,
            )
        except Exception as exc:
            if state["results"] is not None:
                self._warn_preview_once(exc)
                return state["results"]
            if state["started"]:
                raise
            self._warn_preview_once(exc)
            futures = [
                actor.custom_sampler_advanced.remote(add_noise, noise_seed, guider, sampler, sigmas, latent_image_original)
                for actor in gpu_actors
            ]
            return ray.get(futures)
        finally:
            if state["started"] and not state["finished"]:
                for future in state["futures"]:
                    try:
                        ray.cancel(future, force=False)
                    except Exception:
                        pass
            try:
                event_queue.shutdown(force=True)
            except Exception as exc:
                self._warn_preview_once(exc)
