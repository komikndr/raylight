from pathlib import Path
from queue import Empty
import logging
import types

import pytest
import torch

import comfy.nested_tensor
from comfy.latent_formats import MiniMaxH3AV
from comfy.patcher_extension import WrappersMP

from raylight.expansion.karmabu_adapters import context as adapter_context
from raylight.expansion.karmabu_adapters import nodes as adapter_nodes
from raylight.expansion.karmabu_adapters import worker as adapter_worker
from raylight.expansion.karmabu_adapters.registry import (
    KJNODES_PREVIEW_OVERRIDE,
    PROXY_ATTACHMENT,
    RAY_ACTORS_ATTACHMENT,
    resolve_adapter,
)


ROOT = Path(__file__).parents[1]


def _metadata():
    return {
        "model_class": "MiniMaxH3",
        "latent_format": MiniMaxH3AV(),
        "latent_format_class": "MiniMaxH3AV",
    }


def _proxy(monkeypatch):
    monkeypatch.setattr(adapter_context.comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    return adapter_context.create_model_proxy(_metadata())


def _kj_wrapper(executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes):
    def new_callback(step, x0, x, total_steps):
        return callback(step, x0, x, total_steps)

    return executor(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask,
        new_callback,
        disable_pbar,
        seed,
        latent_shapes=latent_shapes,
    )


def _h3_latent():
    video = torch.zeros((1, 24, 2, 3, 4))
    audio = torch.zeros((1, 32, 2, 8))
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def test_export_creates_weightless_model_proxy_and_clone(monkeypatch):
    monkeypatch.setattr(adapter_context.comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))

    class RemoteMetadata:
        def remote(self, fn):
            return _metadata()

    actor = types.SimpleNamespace(model_function_runner_get_values=RemoteMetadata())
    monkeypatch.setattr(adapter_nodes.ray, "get", lambda value: value)
    ray_actors = {"workers": [actor]}
    proxy = adapter_nodes.KarmabuExport().export_model(ray_actors)[0]

    assert list(proxy.model.parameters()) == []
    assert proxy.get_attachment(PROXY_ATTACHMENT)["model_class"] == "MiniMaxH3"
    clone = proxy.clone()
    assert clone.model is proxy.model
    assert clone.get_attachment(PROXY_ATTACHMENT)["latent_format_class"] == "MiniMaxH3AV"
    assert clone.get_attachment(RAY_ACTORS_ATTACHMENT) is ray_actors
    with pytest.raises(RuntimeError, match="driver-only proxy"):
        clone.model(torch.zeros(1))


def test_k3u_user_facing_names_and_sampler_connector():
    from raylight.comfy_extra_dist.nodes_custom_sampler import XFuserSamplerCustomAdvanced

    assert adapter_nodes.NODE_DISPLAY_NAME_MAPPINGS["KarmabuExport"] == "K3U Export"
    assert adapter_nodes.NODE_DISPLAY_NAME_MAPPINGS["KarmabuImport"] == "K3U Import"
    assert adapter_nodes.KarmabuImport.RETURN_TYPES[0] == "K3U_ADAPTER_CONTEXT"
    assert adapter_nodes.KarmabuImport.RETURN_NAMES[0] == "k3u_adapter_context"
    assert XFuserSamplerCustomAdvanced.INPUT_TYPES()["optional"] == {
        "k3u_adapter_context": ("K3U_ADAPTER_CONTEXT",)
    }


def test_kj_wrapper_detection_and_import(monkeypatch):
    proxy = _proxy(monkeypatch)
    ray_actors = {"workers": [object()]}
    proxy.set_attachments(RAY_ACTORS_ATTACHMENT, ray_actors)
    proxy.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "kj_preview_override", _kj_wrapper)

    spec, wrapper = resolve_adapter(proxy)
    assert spec == KJNODES_PREVIEW_OVERRIDE
    assert wrapper is _kj_wrapper
    imported, imported_ray_actors = adapter_nodes.KarmabuImport().import_model(proxy)
    assert imported.spec.id == "kjnodes.preview_override"
    assert imported.wrapper is _kj_wrapper
    assert imported_ray_actors is ray_actors


def test_unknown_wrapper_is_warned_and_rejected(monkeypatch, caplog):
    proxy = _proxy(monkeypatch)
    proxy.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "unknown_preview", _kj_wrapper)

    with caplog.at_level(logging.WARNING):
        assert adapter_nodes.KarmabuImport().import_model(proxy) == (None, None)
    assert "Unsupported MODEL modification" in caplog.text
    assert "outer_sample:unknown_preview" in caplog.text


def test_no_supported_wrapper_returns_none_without_warning(monkeypatch, caplog):
    proxy = _proxy(monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert adapter_nodes.KarmabuImport().import_model(proxy) == (None, None)
    assert "K3U Adapter" not in caplog.text


def test_worker_emits_only_dp_zero_group_leader_and_keeps_x0(monkeypatch):
    monkeypatch.setattr(adapter_worker.ray, "put", lambda tensor: tensor)

    class FakeQueue:
        def __init__(self):
            self.events = []

        def put(self, event):
            self.events.append(event)

    video_x0 = torch.ones((1, 24, 2, 3, 4), requires_grad=True)
    audio_x0 = torch.full((1, 32, 2, 8), 5.0)
    video_x = torch.full_like(video_x0, 2.0)
    audio_x = torch.full_like(audio_x0, 6.0)
    x0 = comfy.nested_tensor.NestedTensor((video_x0, audio_x0))
    x = comfy.nested_tensor.NestedTensor((video_x, audio_x))

    primary = types.SimpleNamespace(
        get_exec_group_info=lambda: {"is_group_leader": True, "dp_rank": 0}
    )
    queue = FakeQueue()
    x0_output = {}
    emitter = adapter_worker.WorkerPreviewEmitter(primary, queue, x0_output)
    emitter.callback(2, x0, x, 10)
    video_x0.detach().fill_(9.0)
    emitter.close()

    assert x0_output["x0"] is x0
    assert [event["kind"] for event in queue.events] == ["step", "done"]
    event = queue.events[0]
    assert event["x0_ref"][0].shape == (1, 24, 2, 3, 4)
    assert event["x_ref"][0].shape == (1, 24, 2, 3, 4)
    assert not event["x0_ref"][0].requires_grad
    assert torch.all(event["x0_ref"][0] == 1.0)

    other_queue = FakeQueue()
    other_output = {}
    other = types.SimpleNamespace(
        get_exec_group_info=lambda: {"is_group_leader": True, "dp_rank": 1}
    )
    other_emitter = adapter_worker.WorkerPreviewEmitter(other, other_queue, other_output)
    other_emitter.callback(2, x0, x, 10)
    other_emitter.close()
    assert other_output["x0"] is x0
    assert other_queue.events == []


def test_h3_noise_uses_worker_generator_and_preserves_nested_latent():
    latent = _h3_latent()
    latent["batch_index"] = [2]
    rng_before = torch.random.get_rng_state().clone()
    packed_noise, packed_latent, shapes, sampling_seed = adapter_context.prepare_wrapper_inputs(
        True, 1234, latent
    )
    rng_after = torch.random.get_rng_state()

    with torch.random.fork_rng(devices=[]):
        expected, expected_seed = adapter_context._generate_advanced_noise(True, 1234, latent)
    expected_packed, expected_shapes = comfy.utils.pack_latents(expected.unbind())

    assert sampling_seed == expected_seed
    assert shapes == expected_shapes
    assert torch.equal(packed_noise, expected_packed)
    assert torch.equal(rng_before, rng_after)
    assert packed_latent.shape[-1] == sum(t.numel() for t in latent["samples"].unbind())


def test_non_nested_worker_noise_falls_back_to_legacy_sampling(monkeypatch, caplog):
    context = _adapter(monkeypatch)
    expected = ({"samples": "out"}, {"samples": "x0"})
    calls = []
    guider = {}
    sampler = object()
    sigmas = torch.tensor([1.0, 0.0])
    latent = _h3_latent()

    class Remote:
        def remote(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "future-0"

    monkeypatch.setattr(adapter_context.ray, "get", lambda value: [expected] if isinstance(value, list) else value)
    actor = types.SimpleNamespace(custom_sampler_advanced=Remote())

    with caplog.at_level(logging.WARNING):
        results = context.sample_xfuser_advanced(
            [actor], False, 7, guider, sampler, sigmas, latent
        )

    assert results == [expected]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:4] == (False, 7, guider, sampler)
    assert args[4] is sigmas
    assert args[5] is latent
    assert kwargs == {}
    assert "noise did not preserve the nested latent structure" in caplog.text


class _FakeQueue:
    instances = []

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.events = []
        self.shutdown_called = False
        self.__class__.instances.append(self)

    def put(self, event):
        self.events.append(event)

    def get(self, timeout):
        if self.events:
            return self.events.pop(0)
        raise Empty

    def shutdown(self, force=False):
        self.shutdown_called = force
        self.events.clear()


class _FakeRemoteSampler:
    def __init__(self, token, steps=2):
        self.token = token
        self.steps = steps

    def remote(self, *args, **kwargs):
        queue = kwargs.get("adapter_event_queue")
        if queue is not None:
            for step in range(self.steps):
                queue.put(
                    {
                        "kind": "step",
                        "step": step,
                        "total_steps": self.steps,
                        "x0_ref": [f"x0-{step}"],
                        "x_ref": [f"x-{step}"],
                    }
                )
            queue.put({"kind": "done"})
        return self.token


def _patch_fake_ray(monkeypatch, results, tensors, worker_error=None):
    def fake_get(value):
        if isinstance(value, list):
            if value and isinstance(value[0], str) and value[0].startswith(("x0-", "x-")):
                return [tensors[item] for item in value]
            return [fake_get(item) for item in value]
        if worker_error is not None and value == "future-0":
            raise worker_error
        return results.get(value, value)

    monkeypatch.setattr(adapter_context.ray, "get", fake_get)
    monkeypatch.setattr(
        adapter_context.ray,
        "wait",
        lambda pending, num_returns, timeout: ([pending[0]], pending[1:]),
    )


def _adapter(monkeypatch):
    proxy = _proxy(monkeypatch)
    return adapter_context.KarmabuAdapterContext(
        proxy, KJNODES_PREVIEW_OVERRIDE, _kj_wrapper, queue_factory=_FakeQueue
    )


def test_callback_exception_does_not_stop_sampling_and_queue_is_cleaned(monkeypatch, caplog):
    _FakeQueue.instances.clear()
    context = _adapter(monkeypatch)
    callback_calls = []

    def failing_callback(step, x0, x, total):
        callback_calls.append(step)
        raise RuntimeError("preview exploded")

    monkeypatch.setattr(adapter_context.latent_preview, "prepare_callback", lambda *args: failing_callback)
    tensors = {
        "x0-0": torch.zeros((1, 24, 2, 3, 4)),
        "x-0": torch.ones((1, 24, 2, 3, 4)),
        "x0-1": torch.zeros((1, 24, 2, 3, 4)),
        "x-1": torch.ones((1, 24, 2, 3, 4)),
    }
    expected = ({"samples": "out"}, {"samples": "x0"})
    _patch_fake_ray(monkeypatch, {"future-0": expected}, tensors)
    actor = types.SimpleNamespace(custom_sampler_advanced=_FakeRemoteSampler("future-0"))

    with caplog.at_level(logging.WARNING):
        results = context.sample_xfuser_advanced(
            [actor], True, 7, {}, object(), torch.tensor([1.0, 0.0]), _h3_latent()
        )

    assert results == [expected]
    assert callback_calls == [0]
    assert caplog.text.count("Preview disabled; sampling continues") == 1
    assert _FakeQueue.instances[-1].maxsize == 1
    assert _FakeQueue.instances[-1].shutdown_called
    assert _FakeQueue.instances[-1].events == []


def test_worker_exception_propagates_and_queue_is_cleaned(monkeypatch):
    _FakeQueue.instances.clear()
    context = _adapter(monkeypatch)
    monkeypatch.setattr(adapter_context.latent_preview, "prepare_callback", lambda *args: lambda *args: None)
    tensors = {
        "x0-0": torch.zeros((1, 24, 2, 3, 4)),
        "x-0": torch.ones((1, 24, 2, 3, 4)),
        "x0-1": torch.zeros((1, 24, 2, 3, 4)),
        "x-1": torch.ones((1, 24, 2, 3, 4)),
    }
    _patch_fake_ray(monkeypatch, {}, tensors, worker_error=RuntimeError("worker failed"))
    actor = types.SimpleNamespace(custom_sampler_advanced=_FakeRemoteSampler("future-0"))

    with pytest.raises(RuntimeError, match="worker failed"):
        context.sample_xfuser_advanced(
            [actor], True, 7, {}, object(), torch.tensor([1.0, 0.0]), _h3_latent()
        )
    assert _FakeQueue.instances[-1].shutdown_called
    assert _FakeQueue.instances[-1].events == []


def test_execution_cancellation_cancels_future_and_cleans_queue(monkeypatch):
    _FakeQueue.instances.clear()
    context = _adapter(monkeypatch)
    cancelled = []

    def interrupting_callback(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(adapter_context.latent_preview, "prepare_callback", lambda *args: interrupting_callback)
    tensors = {
        "x0-0": torch.zeros((1, 24, 2, 3, 4)),
        "x-0": torch.ones((1, 24, 2, 3, 4)),
        "x0-1": torch.zeros((1, 24, 2, 3, 4)),
        "x-1": torch.ones((1, 24, 2, 3, 4)),
    }
    _patch_fake_ray(monkeypatch, {"future-0": ({}, {})}, tensors)
    monkeypatch.setattr(
        adapter_context.ray,
        "cancel",
        lambda future, force: cancelled.append((future, force)),
    )
    actor = types.SimpleNamespace(custom_sampler_advanced=_FakeRemoteSampler("future-0"))

    with pytest.raises(KeyboardInterrupt):
        context.sample_xfuser_advanced(
            [actor], True, 7, {}, object(), torch.tensor([1.0, 0.0]), _h3_latent()
        )
    assert cancelled == [("future-0", False)]
    assert _FakeQueue.instances[-1].shutdown_called
    assert _FakeQueue.instances[-1].events == []


def test_xfuser_advanced_none_context_keeps_legacy_remote_call(monkeypatch):
    from raylight.comfy_extra_dist import nodes_custom_sampler

    calls = []

    class Remote:
        def remote(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "future"

    actor = types.SimpleNamespace(custom_sampler_advanced=Remote())
    ray_actors = {"workers": [actor]}
    guider = {"ray_actors": ray_actors, "type": "basic", "positive": []}
    expected = ({"samples": "out"}, {"samples": "x0"})
    monkeypatch.setattr(nodes_custom_sampler.ray, "get", lambda futures: [expected])
    monkeypatch.setattr(nodes_custom_sampler, "configure_sampling_features", lambda actors: None)
    monkeypatch.setattr(nodes_custom_sampler, "_clear_ray_worker_vram_after_sampling", lambda actors: None)
    monkeypatch.setattr(nodes_custom_sampler.gc, "collect", lambda: None)
    monkeypatch.setattr(nodes_custom_sampler.comfy.model_management, "unload_all_models", lambda: None)
    monkeypatch.setattr(nodes_custom_sampler.comfy.model_management, "soft_empty_cache", lambda: None)

    output = nodes_custom_sampler.XFuserSamplerCustomAdvanced().ray_sample(
        True, 9, guider, "sampler", "sigmas", "latent", k3u_adapter_context=None
    )
    assert output == (expected[0], expected[1], ray_actors)
    assert calls == [((True, 9, guider, "sampler", "sigmas", "latent"), {})]


def test_worker_has_no_kjnodes_or_tae_imports():
    source = (ROOT / "src/raylight/distributed_worker/ray_worker.py").read_text().lower()
    assert "kjnodes" not in source
    assert "taeh3" not in source
    assert "tiny_vae" not in source
