import logging

import ray
import torch


def extract_video_tensor(value):
    if getattr(value, "is_nested", False):
        tensors = value.unbind()
        if not tensors:
            raise ValueError("empty nested latent")
        value = tensors[0]
    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise ValueError(f"expected a 5D video latent, got {type(value).__name__}")
    return value


def is_preview_producer(worker):
    group_info = worker.get_exec_group_info()
    return bool(group_info["is_group_leader"] and group_info["dp_rank"] == 0)


class WorkerPreviewEmitter:
    def __init__(self, worker, event_queue, x0_output):
        self.event_queue = event_queue
        self.x0_output = x0_output
        self.producer = is_preview_producer(worker)
        self.transport_enabled = self.producer
        self.warning_logged = False

    def _warn_once(self, exc):
        if not self.warning_logged:
            self.warning_logged = True
            logging.warning("[K3U Adapter][worker] Preview transport disabled: %s", exc)

    def _put(self, event):
        if not self.transport_enabled:
            return
        try:
            self.event_queue.put(event)
        except Exception as exc:
            self.transport_enabled = False
            self._warn_once(exc)

    def callback(self, step, x0, x, total_steps):
        self.x0_output["x0"] = x0
        if not self.transport_enabled:
            return
        try:
            x0_cpu = extract_video_tensor(x0).detach().to(device="cpu", copy=True)
            x_cpu = extract_video_tensor(x).detach().to(device="cpu", copy=True)
            # Nested ObjectRefs keep the large tensors out of the queue actor. The
            # bounded queue owns at most one event; the driver resolves and drops
            # both refs before accepting the next step.
            x0_ref = ray.put(x0_cpu)
            x_ref = ray.put(x_cpu)
            self._put(
                {
                    "kind": "step",
                    "step": int(step),
                    "total_steps": int(total_steps),
                    "x0_ref": [x0_ref],
                    "x_ref": [x_ref],
                }
            )
        except Exception as exc:
            self.transport_enabled = False
            self._warn_once(exc)

    def close(self):
        self._put({"kind": "done"})
