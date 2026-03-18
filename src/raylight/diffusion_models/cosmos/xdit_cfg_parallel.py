from raylight.distributed_modules.cfg_utils import cfg_parallel_forward


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    return cfg_parallel_forward(
        executor,
        *args,
        chunk_names=(
            "x",
            "timesteps",
            "context",
            "attention_mask",
            "fps",
            "image_size",
            "padding_mask",
            "scalar_feature",
            "latent_condition",
            "latent_condition_sigma",
            "condition_video_augment_sigma",
        ),
        **kwargs,
    )
