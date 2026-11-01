import ray
from functools import wraps


def ray_patch(patch_func):
    @wraps(patch_func)
    def wrapper(self, ray_actors, *args, **kwargs):
        def _patch(model):
            return patch_func(self, model, *args, **kwargs)

        return ({**ray_actors, "sampling_patches": (*ray_actors.get("sampling_patches", ()), _patch)},)
    return wrapper


def ray_model_loader(load_func):
    @wraps(load_func)
    def wrapper(self, *args, **kwargs):
        def load_model():
            return load_func(self, *args, **kwargs)[0]

        ray_actors = load_model()
        return ({**ray_actors, "load_model": load_model},)
    return wrapper


def prepare_ray_model(ray_actors):
    load_model = ray_actors.get("load_model")
    if load_model is not None:
        return {**ray_actors, **load_model()}
    return ray_actors


# For nodes with return value, like produce float, int, or latent that still require model patcher.
def ray_patch_with_return(patch_func):
    @wraps(patch_func)
    def wrapper(self, ray_actors, *args, **kwargs):
        ray_actors = prepare_ray_model(ray_actors)
        def _patch(model, *inner_args, **inner_kwargs):
            # call the original patch on each model
            return patch_func(self, model, *inner_args, **inner_kwargs)

        # Just need rank 0
        actor = ray_actors["workers"][0]
        value = ray.get(actor.model_function_runner_get_values.remote(
            _patch, *args, sampling_patches=ray_actors.get("sampling_patches", ()), **kwargs))
        return value
    return wrapper
