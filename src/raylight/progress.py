"""Side channel carrying sampler progress from Ray workers back to the host.

Ray actors are single threaded, so while rank 0 is sampling it cannot answer a
status call. Progress goes through a small board actor instead: the initializer
starts one per cluster, which does nothing but hold values and so answers at once
while every worker is busy, and hands its name to the workers in their
environment. Calls from one process run on it in the order they were made, so a
late update never lands after a clear. Rank 0 posts to it and the host reads it while it
waits on the sampling futures. Nothing touches the filesystem, so it works the
same with workers on other nodes, and two clusters never share a board.

Every call swallows its errors: progress reporting must never break a render.
"""

import io
import os
import sys
import time
import uuid

BOARD_ENV = "RAYLIGHT_PROGRESS_BOARD"

_PROGRESS = "sampler_progress"
_PREVIEW = "sampler_preview"
_MIN_INTERVAL = 0.25

_last_write = 0.0
_handle = None
_local = None


class Board:
    """Values by key. Runs as a Ray actor; tests use it in process."""

    def __init__(self):
        self._values = {}

    def put(self, key, value):
        self._values[key] = value

    def get(self, key):
        return self._values.get(key)

    def items(self, prefix):
        return {key: value for key, value in list(self._values.items()) if key.startswith(prefix)}

    def clear(self, prefix):
        for key in [key for key in list(self._values) if key.startswith(prefix)]:
            self._values.pop(key, None)


def new_board_name():
    return f"raylight-progress-{uuid.uuid4().hex[:12]}"


def create_board(name):
    """Start the board under `name`. Host side, once ray.init() has run."""
    global _handle
    try:
        import ray
        from ray import cloudpickle

        # ship the class by value: imported by reference, the board process would load
        # raylight, and torch with it, just to hold a few values
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
        handle = ray.remote(Board).options(name=name, num_cpus=0).remote()
    except Exception:
        return None
    os.environ[BOARD_ENV] = name
    _handle = (name, handle)
    return handle


def use_local_board(board):
    """Send every call to an in-process board instead of the actor; None undoes it."""
    global _local
    _local = board


def _call(method, *args, wait=True):
    if _local is not None:
        return getattr(_local, method)(*args)
    name = os.environ.get(BOARD_ENV)
    if not name:
        return None
    global _handle
    try:
        import ray

        if _handle is None or _handle[0] != name:
            _handle = (name, ray.get_actor(name))
        ref = getattr(_handle[1], method).remote(*args)
        return ray.get(ref, timeout=5) if wait else None
    except Exception:
        _handle = None
        return None


def put(key, value):
    """Post a value without waiting for the board to take it."""
    _call("put", key, value, wait=False)


def get(key):
    return _call("get", key)


def items(prefix):
    return _call("items", prefix) or {}


def clear_keys(prefix):
    _call("clear", prefix)


def clear():
    clear_keys("sampler_")


def write(value, total, force=False, preview_seq=0):
    """Publish a progress fraction. Throttled unless force is set.

    preview_seq lets the host tell whether a new preview image accompanies
    this update, so it only pays to decode one when there is something new.
    """
    global _last_write
    now = time.monotonic()
    if not force and now - _last_write < _MIN_INTERVAL:
        return
    _last_write = now
    try:
        put(_PROGRESS, (int(value), int(total), int(preview_seq)))
    except (TypeError, ValueError):
        pass


def read():
    """Return (value, total, preview_seq) or None."""
    try:
        value, total, preview_seq = get(_PROGRESS)
        return int(value), int(total), int(preview_seq)
    except (TypeError, ValueError):
        return None


def write_preview(image, image_format="JPEG"):
    """Store a preview image produced by ComfyUI's previewer."""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format=image_format, quality=85)
        put(_PREVIEW, buffer.getvalue())
    except Exception:
        pass


def read_preview():
    try:
        from PIL import Image

        data = get(_PREVIEW)
        if not data:
            return None
        return Image.open(io.BytesIO(data)).copy()
    except Exception:
        return None
