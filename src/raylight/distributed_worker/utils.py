import torch
import torch.distributed as dist
import functools
from ray.experimental import tqdm_ray as ray_tqdm_module
from ray.experimental.tqdm_ray import tqdm as ray_tqdm
import tqdm.auto as tqdm_auto


class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(
            latent_image.shape,
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        import comfy.sample as comfy_sample

        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return comfy_sample.prepare_noise(latent_image, self.seed, batch_inds)


# Monkey patch-unpatch tqdm and trange so it does not broke the progress bar
def _patch_ray_tqdm_close_once():
    if getattr(ray_tqdm_module, "_raylight_clear_on_close", False):
        return

    original_close = ray_tqdm_module._Bar.close

    def close_and_clear(self):
        try:
            self.bar.clear()
        except Exception:
            pass
        original_close(self)

    ray_tqdm_module._Bar.close = close_and_clear
    ray_tqdm_module._raylight_clear_on_close = True


def patch_ray_tqdm(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _patch_ray_tqdm_close_once()

        rank = dist.get_rank()
        orig_tqdm = tqdm_auto.tqdm
        orig_trange = tqdm_auto.trange
        if rank == 0:
            def ray_tqdm_absorb_disable(*a, **k):
                k.pop("disable", None)
                return ray_tqdm(*a, **k)

            def ray_trange_absorb_disable(*a, **k):
                k.pop("disable", None)
                return ray_tqdm(range(*a), **k)

            tqdm_auto.tqdm = ray_tqdm_absorb_disable
            tqdm_auto.trange = ray_trange_absorb_disable

        try:
            return fn(*args, **kwargs)
        finally:
            tqdm_auto.tqdm = orig_tqdm
            tqdm_auto.trange = orig_trange

    return wrapper
