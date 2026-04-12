import ray
from functools import wraps

from raylight.distributed_worker.ray_worker import ensure_fresh_actors


# Decorator to make a patch function Ray-distributable.
# Handles wrapping into _patch and Ray actor execution,
def ray_patch(patch_func):
    @wraps(patch_func)
    def wrapper(self, ray_actors, *args, **kwargs):
        ray_actors, gpu_workers, _ = ensure_fresh_actors(ray_actors)

        def _patch(model, *inner_args, **inner_kwargs):
            # call the original patch on each model
            return patch_func(self, model, *inner_args, **inner_kwargs)

        futures = [actor.model_function_runner.remote(_patch, *args, **kwargs) for actor in gpu_workers]

        ray.get(futures)
        return (ray_actors,)

    return wrapper


# For nodes with return value, like produce float, int, or latent that still require model patcher.
def ray_patch_with_return(patch_func):
    @wraps(patch_func)
    def wrapper(self, ray_actors, *args, **kwargs):
        ray_actors, gpu_workers, _ = ensure_fresh_actors(ray_actors)

        def _patch(model, *inner_args, **inner_kwargs):
            # call the original patch on each model
            return patch_func(self, model, *inner_args, **inner_kwargs)

        # Just need rank 0
        actor = gpu_workers[0]
        value = ray.get(actor.model_function_runner_get_values.remote(_patch, *args, **kwargs))
        return value

    return wrapper
