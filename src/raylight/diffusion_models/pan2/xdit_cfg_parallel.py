from xfuser.core.distributed import get_classifier_free_guidance_world_size

from raylight.distributed_modules.cfg_utils import cfg_parallel_forward


def cfg_parallel_forward_wrapper(executor, x, timestep, context, *args, **kwargs):
    # PAN2 keeps unequal prompt lengths in separate calls; there is no combined
    # guidance batch to split in that case.
    if x.shape[0] != get_classifier_free_guidance_world_size():
        return executor(x, timestep, context, *args, **kwargs)
    return cfg_parallel_forward(
        executor,
        x, timestep, context,
        *args,
        chunk_names=("x", "timestep", "context"),
        **kwargs,
    )
