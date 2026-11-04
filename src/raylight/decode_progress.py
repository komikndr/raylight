import comfy.model_management

from raylight import progress


_PREFIX = "decode/"
_SHARE = 1000


def clear():
    progress.clear_keys(_PREFIX)


def check_cancel(force=False):
    comfy.model_management.throw_exception_if_processing_interrupted()
    if progress.cancel_requested(force=force):
        raise comfy.model_management.InterruptProcessingException()


def write(part, done, total):
    value = min(_SHARE, done * _SHARE // total) if total else _SHARE
    progress.put(_PREFIX + part, value)


def read(parts):
    # Each planned task owns a fixed share, including tasks that have not started.
    values = progress.items(_PREFIX, timeout=0.1)
    if not values:
        return None
    return sum(values.get(_PREFIX + part, 0) for part in parts), len(parts) * _SHARE, 0
