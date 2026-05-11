import types

import torch


def load_vae_model(vae_path):
    import comfy.sd as comfy_sd
    import comfy.utils as comfy_utils

    from ..comfy_dist.sd import decode_tiled_1d, decode_tiled_, decode_tiled_3d

    state_dict = {}
    if "pixel_space" in vae_path:
        state_dict["pixel_space_vae"] = torch.tensor(1.0)
    else:
        state_dict = comfy_utils.load_torch_file(vae_path)

    vae_model = comfy_sd.VAE(sd=state_dict)
    vae_model.throw_exception_if_invalid()

    vae_model.decode_tiled_1d = types.MethodType(decode_tiled_1d, vae_model)
    vae_model.decode_tiled_ = types.MethodType(decode_tiled_, vae_model)
    vae_model.decode_tiled_3d = types.MethodType(decode_tiled_3d, vae_model)
    return vae_model


def ray_vae_decode_impl(worker, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
    vae = worker.vae_model
    if tile_size < overlap * 4:
        overlap = tile_size // 4
    if temporal_size < temporal_overlap * 2:
        temporal_overlap = temporal_overlap // 2
    temporal_compression = vae.temporal_compression_decode()
    if temporal_compression is not None:
        temporal_size = max(2, temporal_size // temporal_compression)
        temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
    else:
        temporal_size = None
        temporal_overlap = None

    compression = vae.spacial_compression_decode()

    images = vae.decode_tiled(
        samples["samples"],
        tile_x=tile_size // compression,
        tile_y=tile_size // compression,
        overlap=overlap // compression,
        tile_t=temporal_size,
        overlap_t=temporal_overlap,
    )
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def ray_vae_decode_partial_impl(worker, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8, job_rank=0, job_world_size=1):
    import comfy.model_management as model_management
    from raylight import comfy_dist

    vae = worker.vae_model
    if tile_size < overlap * 4:
        overlap = tile_size // 4
    if temporal_size < temporal_overlap * 2:
        temporal_overlap = temporal_overlap // 2
    temporal_compression = vae.temporal_compression_decode()
    if temporal_compression is not None:
        temporal_size = max(2, temporal_size // temporal_compression)
        temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
    else:
        temporal_size = None
        temporal_overlap = None

    latent = samples["samples"]
    compression = vae.spacial_compression_decode()
    tile_x = tile_size // compression
    tile_y = tile_size // compression
    overlap = overlap // compression

    memory_used = vae.memory_used_decode(latent.shape, vae.vae_dtype)
    model_management.load_models_gpu([vae.patcher], memory_required=memory_used, force_full_load=vae.disable_offload)

    partials = []
    dims = latent.ndim - 2
    if dims == 1 or vae.extra_1d_channel is not None:
        if latent.ndim == 3:
            partial_latent = latent
            decode_fn = lambda a: vae.first_stage_model.decode(a.to(vae.vae_dtype).to(vae.device)).float()
        else:
            og_shape = latent.shape
            partial_latent = latent.reshape((og_shape[0], og_shape[1] * og_shape[2], -1))
            decode_fn = lambda a: vae.first_stage_model.decode(
                a.reshape((-1, og_shape[1], og_shape[2], a.shape[-1])).to(vae.vae_dtype).to(vae.device)
            ).float()

        partials.append(
            comfy_dist.utils.tiled_scale_multidim_partial(
                partial_latent,
                decode_fn,
                tile=(tile_x,),
                overlap=overlap,
                upscale_amount=vae.upscale_ratio,
                out_channels=vae.output_channels,
                output_device="cpu",
                job_rank=job_rank,
                job_world_size=job_world_size,
            )
        )
    elif dims == 2:
        decode_fn = lambda a: vae.first_stage_model.decode(a.to(vae.vae_dtype).to(vae.device)).float()
        for pass_tile_x, pass_tile_y in ((tile_x // 2, tile_y * 2), (tile_x * 2, tile_y // 2), (tile_x, tile_y)):
            partials.append(
                comfy_dist.utils.tiled_scale_multidim_partial(
                    latent,
                    decode_fn,
                    tile=(pass_tile_y, pass_tile_x),
                    overlap=overlap,
                    upscale_amount=vae.upscale_ratio,
                    out_channels=vae.output_channels,
                    output_device="cpu",
                    job_rank=job_rank,
                    job_world_size=job_world_size,
                )
            )
    elif dims == 3:
        decode_fn = lambda a: vae.first_stage_model.decode(a.to(vae.vae_dtype).to(vae.device)).float()
        if temporal_overlap is None:
            pass_overlap = (1, overlap, overlap)
        else:
            pass_overlap = (max(1, temporal_overlap), overlap, overlap)
        partials.append(
            comfy_dist.utils.tiled_scale_multidim_partial(
                latent,
                decode_fn,
                tile=(max(2, temporal_size) if temporal_size is not None else 999, tile_x, tile_y),
                overlap=pass_overlap,
                upscale_amount=vae.upscale_ratio,
                out_channels=vae.output_channels,
                index_formulas=vae.upscale_index_formula,
                output_device="cpu",
                job_rank=job_rank,
                job_world_size=job_world_size,
            )
        )
    else:
        raise ValueError(f"Unsupported VAE latent dimensions: {dims}")

    return partials


def ray_vae_decode_finalize_impl(worker, decoded):
    images = worker.vae_model.process_output(decoded).movedim(1, -1)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images.cpu()
