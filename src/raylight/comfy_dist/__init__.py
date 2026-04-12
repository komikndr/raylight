# Distributed version of comfy library
import logging

from . import lora
from . import sd
from . import utils
from . import model_patcher
from . import float
from . import supported_models_base


def patch_base_getattr():
    """Patch BASE.__getattr__ to raise AttributeError for dunder methods."""
    import comfy.supported_models_base

    def _safe_getattr(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        logging.warning(f"WARNING: Missing config attribute '{name}'")

        def _dummy(*args, **kwargs):
            logging.warning(f"Called missing attribute {name}")
            return None

        return _dummy

    setattr(comfy.supported_models_base.BASE, "__getattr__", _safe_getattr)


__all__ = [
    "lora",
    "sd",
    "model_patcher",
    "float",
    "utils",
    "supported_models_base",
    "patch_base_getattr",
]
