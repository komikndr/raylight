from dataclasses import dataclass
from enum import Enum
import inspect
import logging

from comfy.patcher_extension import WrappersMP


PROXY_ATTACHMENT = "karmabu_adapter_proxy"
RAY_ACTORS_ATTACHMENT = "karmabu_ray_actors"


class AdapterMode(str, Enum):
    DRIVER_HOOK = "driver_hook"


@dataclass(frozen=True)
class AdapterSpec:
    id: str
    wrapper_type: str
    wrapper_key: str
    mode: AdapterMode
    needs_intermediate_x0: bool
    needs_current_x: bool
    requires_worker_patch: bool


KJNODES_PREVIEW_OVERRIDE = AdapterSpec(
    id="kjnodes.preview_override",
    wrapper_type=WrappersMP.OUTER_SAMPLE,
    wrapper_key="kj_preview_override",
    mode=AdapterMode.DRIVER_HOOK,
    needs_intermediate_x0=True,
    needs_current_x=True,
    requires_worker_patch=False,
)

ADAPTER_SPECS = {
    (KJNODES_PREVIEW_OVERRIDE.wrapper_type, KJNODES_PREVIEW_OVERRIDE.wrapper_key): KJNODES_PREVIEW_OVERRIDE,
}

_KJ_WRAPPER_PARAMETERS = (
    "executor",
    "noise",
    "latent_image",
    "sampler",
    "sigmas",
    "denoise_mask",
    "callback",
    "disable_pbar",
    "seed",
    "latent_shapes",
)


def _has_unregistered_model_changes(model):
    return bool(
        model.patches
        or model.object_patches
        or model.weight_wrapper_patches
        or model.callbacks
        or model.injections
        or model.additional_models
    )


def resolve_adapter(model):
    if model.get_attachment(PROXY_ATTACHMENT) is None:
        logging.warning("[K3U Adapter] Import expects a MODEL created by K3U Export; adapter disabled.")
        return None

    unsupported = []
    supported = []
    for wrapper_type, keyed_wrappers in model.wrappers.items():
        for wrapper_key, wrappers in keyed_wrappers.items():
            spec = ADAPTER_SPECS.get((wrapper_type, wrapper_key))
            if spec is None:
                unsupported.append(f"{wrapper_type}:{wrapper_key}")
            else:
                supported.extend((spec, wrapper) for wrapper in wrappers)

    if _has_unregistered_model_changes(model):
        unsupported.append("MODEL patches/callbacks")

    if unsupported:
        logging.warning(
            "[K3U Adapter] Unsupported MODEL modification(s): %s; adapter disabled.",
            ", ".join(unsupported),
        )
        return None

    if not supported:
        return None
    if len(supported) != 1:
        logging.warning("[K3U Adapter] Exactly one supported wrapper is required; adapter disabled.")
        return None

    spec, wrapper = supported[0]
    parameters = tuple(inspect.signature(wrapper).parameters)
    if parameters != _KJ_WRAPPER_PARAMETERS:
        logging.warning(
            "[K3U Adapter] Unsupported signature for %s: %s; adapter disabled.",
            spec.id,
            parameters,
        )
        return None
    return spec, wrapper
