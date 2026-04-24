import torch


class _RayControlNetRef:
    """Lightweight placeholder for a ControlNet in conditioning.

    Created by RayControlNetApply in the main process.  Carries only the
    hint image and apply settings -- no model weights.  Workers replace
    this with a real ControlNet loaded from their local cache.
    """

    def __init__(self, strength, timestep_percent_range, cond_hint_original,
                 extra_concat_orig=None, previous_controlnet=None, needs_vae=False):
        self.strength = strength
        self.timestep_percent_range = timestep_percent_range
        self.cond_hint_original = cond_hint_original
        self.extra_concat_orig = list(extra_concat_orig or [])
        self.previous_controlnet = previous_controlnet
        self.needs_vae = needs_vae

        # ControlBase interface stubs needed by ComfyUI internals
        self.cond_hint = None
        self.global_average_pooling = False
        self.compression_ratio = 1
        self.upscale_algorithm = "nearest-exact"
        self.extra_args = {}
        self.extra_conds = []
        self.strength_type = None
        self.concat_mask = False
        self.extra_hooks = None
        self.preprocess_image = lambda a: a
        self.model_sampling_current = None
        self.latent_format = None
        self.vae = None

    def set_cond_hint(self, *args, **kwargs):
        pass

    def set_previous_controlnet(self, prev):
        self.previous_controlnet = prev

    def copy(self):
        c = _RayControlNetRef(
            self.strength,
            self.timestep_percent_range,
            self.cond_hint_original,
            self.extra_concat_orig,
            self.previous_controlnet,
            self.needs_vae,
        )
        return c

    def pre_run(self, model, percent_to_timestep_function):
        pass

    def get_models(self):
        return []

    def get_extra_hooks(self):
        hooks = []
        if self.previous_controlnet is not None:
            hooks += self.previous_controlnet.get_extra_hooks()
        return hooks

    def cleanup(self):
        pass

    def inference_memory_requirements(self, dtype):
        return 0


def _materialize_controlnet_ref(control, cnet_template, worker_vae=None):
    if not isinstance(control, _RayControlNetRef):
        return control

    cnet = cnet_template.copy()
    cnet.cond_hint_original = control.cond_hint_original
    cnet.strength = control.strength
    cnet.timestep_percent_range = control.timestep_percent_range
    if control.extra_concat_orig:
        cnet.extra_concat_orig = list(control.extra_concat_orig)
    if control.needs_vae and worker_vae is not None:
        cnet.vae = worker_vae
    if control.previous_controlnet is not None:
        cnet.set_previous_controlnet(
            _materialize_controlnet_ref(control.previous_controlnet, cnet_template, worker_vae)
        )
    return cnet


def _restore_controlnet_refs(cond_list, cached_controlnet, worker_vae=None):
    """Replace _RayControlNetRef placeholders with real ControlNet objects."""
    if cond_list is None or cached_controlnet is None:
        return

    _, cnet_template = cached_controlnet
    for item in cond_list:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            d = item[1]
        elif isinstance(item, dict):
            d = item
        else:
            continue
        if not isinstance(d, dict):
            continue

        control = d.get("control")
        if not isinstance(control, _RayControlNetRef):
            continue

        d["control"] = _materialize_controlnet_ref(control, cnet_template, worker_vae)


def _remap_conditioning_devices(positive, negative):
    """Remap CUDA device references in conditioning to cuda:0.

    Conditioning is created in the main ComfyUI process where CUDA device
    indices map to physical GPUs.  Inside a ray worker, CUDA_VISIBLE_DEVICES
    is set to a single physical GPU, so only cuda:0 is valid.  Any model
    device (VAE, ControlNet, etc.) that references cuda:N (N>0) will fail.
    """
    target = torch.device("cuda:0")
    for cond_list in (positive, negative):
        if cond_list is None:
            continue
        for item in cond_list:
            # Conditioning items are [tensor, dict] -- the dict is at index 1.
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                cond = item[1]
            elif isinstance(item, dict):
                cond = item
            else:
                continue
            if not isinstance(cond, dict):
                continue
            control = cond.get("control")
            if control is not None:
                _remap_control_devices(control, target)


def _remap_control_devices(control, target):
    vae = getattr(control, "vae", None)
    if vae is not None:
        _remap_cuda_device(vae, "device", target)
        _remap_cuda_device(vae, "output_device", target)
        patcher = getattr(vae, "patcher", None)
        if patcher is not None:
            _remap_patcher_device(patcher, target)
    model_wrapped = getattr(control, "control_model_wrapped", None)
    if model_wrapped is not None:
        _remap_patcher_device(model_wrapped, target)
    _remap_cuda_device(control, "load_device", target)
    prev = getattr(control, "previous_controlnet", None)
    if prev is not None:
        _remap_control_devices(prev, target)


def _move_control_to_device(control, device):
    """Move ControlNet model weights to the worker's GPU.

    Ray deserializes the ControlNet model on CPU.  Without this, only rank 0
    loads the model via load_models_gpu(), leaving other ranks unable to run
    the ControlNet forward.  This ensures every rank has the model on its GPU.
    """
    model_wrapped = getattr(control, "control_model_wrapped", None)
    if model_wrapped is not None:
        model = getattr(model_wrapped, "model", None)
        if model is not None:
            try:
                model.to(device)
            except Exception:
                pass
    # Also move the hint tensor if it's already materialized
    cond_hint = getattr(control, "cond_hint", None)
    if isinstance(cond_hint, torch.Tensor) and cond_hint.device.type == "cpu":
        try:
            control.cond_hint = cond_hint.to(device)
        except Exception:
            pass
    prev = getattr(control, "previous_controlnet", None)
    if prev is not None:
        _move_control_to_device(prev, device)


def _prepare_control_models(positive, negative):
    """Remap devices AND move ControlNet model weights to the worker's GPU."""
    target = torch.device("cuda:0")
    for cond_list in (positive, negative):
        if cond_list is None:
            continue
        for item in cond_list:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                cond = item[1]
            elif isinstance(item, dict):
                cond = item
            else:
                continue
            if not isinstance(cond, dict):
                continue
            control = cond.get("control")
            if control is not None:
                _move_control_to_device(control, target)


def _remap_patcher_device(patcher, target):
    _remap_cuda_device(patcher, "load_device", target)
    _remap_cuda_device(patcher, "offload_device", target)


def _remap_cuda_device(obj, attr, target):
    val = getattr(obj, attr, None)
    if isinstance(val, torch.device) and val.type == "cuda":
        setattr(obj, attr, target)
