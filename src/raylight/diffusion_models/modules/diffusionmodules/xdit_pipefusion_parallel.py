import torch
from xfuser.core.distributed import (
    get_pipeline_parallel_world_size,
    get_runtime_state,
    get_pp_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    is_dp_last_group,
    get_world_group,
    get_vae_parallel_group,
    get_dit_world_size,
)


def pipefusion_wrapper(executor, *args, **kwargs):
    noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed = args
    print(f"{latent_image.shape=}==================")
    b, c, h, w = latent_image.shape
    num_inference_steps = sigmas.shape[0] - 1
    get_runtime_state().set_input_parameters(
        height=h,
        width=w,
        batch_size=b,
        num_inference_steps=num_inference_steps,
    )
    result = executor(*args, **kwargs)
    return result
