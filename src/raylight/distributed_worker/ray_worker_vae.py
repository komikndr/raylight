import torch


def load_vae_model(vae_path):
    import comfy.sd as comfy_sd
    import comfy.utils as comfy_utils

    state_dict = {}
    metadata = None
    if "pixel_space" in vae_path:
        state_dict["pixel_space_vae"] = torch.tensor(1.0)
    else:
        state_dict, metadata = comfy_utils.load_torch_file(vae_path, return_metadata=True)

    vae_model = comfy_sd.VAE(sd=state_dict, metadata=metadata)
    vae_model.throw_exception_if_invalid()


    return vae_model


def _validate_job_rank(job_rank, job_world_size):
    if job_world_size < 1 or not 0 <= job_rank < job_world_size:
        raise ValueError(f"Invalid distributed VAE rank {job_rank} for world size {job_world_size}.")


def _is_nested_tensor(latent):
    if not isinstance(latent, torch.Tensor):
        return True
    return getattr(latent, "is_nested", False)


def _compute_1d_tile_ranges(latent_size, tile_size, overlap):
    tile_size = max(1, int(tile_size))
    overlap = min(max(0, int(overlap)), tile_size - 1)
    stride = max(1, tile_size - overlap)
    if latent_size <= tile_size:
        return [(0, latent_size)]
    ranges = []
    for pos in range(0, latent_size - overlap, stride):
        end = min(pos + tile_size, latent_size)
        ranges.append((pos, end))
    return ranges


def _compute_2d_tile_ranges(height, width, tile_y, tile_x, overlap):
    tile_y = max(1, int(tile_y))
    tile_x = max(1, int(tile_x))
    overlap_y = min(max(0, int(overlap)), tile_y - 1)
    overlap_x = min(max(0, int(overlap)), tile_x - 1)
    stride_y = max(1, tile_y - overlap_y)
    stride_x = max(1, tile_x - overlap_x)

    y_positions = list(range(0, height - overlap_y, stride_y)) if height > tile_y else [0]
    x_positions = list(range(0, width - overlap_x, stride_x)) if width > tile_x else [0]

    ranges = []
    for y in y_positions:
        y_end = min(y + tile_y, height)
        for x in x_positions:
            x_end = min(x + tile_x, width)
            ranges.append((y, y_end, x, x_end))
    return ranges


def _get_upscale_func(upscale, dim):
    if isinstance(upscale, (tuple, list)):
        if dim >= len(upscale):
            return lambda val: val
        upscale = upscale[dim]
    if callable(upscale):
        return upscale
    return lambda val: upscale * val


def _round_upscale(upscale_func, val):
    return round(upscale_func(val))


def _linear_feather(size, feather):
    weight = torch.ones(size, dtype=torch.float32)
    if feather > 0 and feather < size:
        ramp = torch.arange(1, feather + 1, dtype=torch.float32) / feather
        weight[:feather] = ramp
        weight[-feather:] = ramp.flip(0)
    return weight


def _formula_func(formula, fallback):
    if formula is None:
        return fallback
    if callable(formula):
        return formula
    return lambda position: formula * position


def _get_pos_func(index_formula, upscale, dim):
    fallback = _get_upscale_func(upscale, dim)
    if index_formula is None:
        return fallback
    if isinstance(index_formula, (tuple, list)):
        if dim >= len(index_formula):
            return fallback
        return _formula_func(index_formula[dim], fallback)
    return _formula_func(index_formula, fallback)


def _compute_feather(upscale, overlap_latent, spatial_dims):
    if spatial_dims == 1:
        feather = round(_get_upscale_func(upscale, 0)(overlap_latent))
        return (feather,)
    feather_h = round(_get_upscale_func(upscale, 0)(overlap_latent))
    feather_w = round(_get_upscale_func(upscale, 1)(overlap_latent))
    if spatial_dims == 2:
        return (feather_h, feather_w)
    feather_h = round(_get_upscale_func(upscale, 1)(overlap_latent))
    feather_w = round(_get_upscale_func(upscale, 2)(overlap_latent))
    return (feather_h, feather_w)


def ray_vae_decode_partial_impl(worker, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8, job_rank=0, job_world_size=1):
    import comfy.model_management as model_management

    _validate_job_rank(job_rank, job_world_size)

    vae = worker.vae_model

    latent = samples["samples"]
    if _is_nested_tensor(latent):
        raise ValueError(
            "Distributed VAE (Ray) cannot decode structured/nested latents. "
            "Use LTXVSeparateAVLatent to prepare latents, then use normal ComfyUI VAE Decode."
        )
    if vae.handles_tiling:
        raise ValueError(
            "Distributed VAE (Ray) cannot externally tile this VAE. "
            "Use SeedVR2 VAE Decode Distributed (Ray) for SeedVR2."
        )
    if tile_size < overlap * 4:
        overlap = tile_size // 4

    compression = vae.spacial_compression_decode()
    tile_x = tile_size // compression
    tile_y = tile_size // compression
    overlap_latent = overlap // compression

    memory_used = vae.memory_used_decode(latent.shape, vae.vae_dtype)
    model_management.load_models_gpu([vae.patcher], memory_required=memory_used, force_full_load=vae.disable_offload)

    decode_dtype = vae.vae_output_dtype()
    upscale = vae.upscale_ratio
    index_formula = getattr(vae, "upscale_index_formula", None)
    spatial_dims = latent.ndim - 2

    extra_channels = getattr(vae, "extra_1d_channel", None)

    upscale_0 = _get_upscale_func(upscale, 0)
    upscale_1 = _get_upscale_func(upscale, 1) if spatial_dims >= 2 else None
    upscale_2 = _get_upscale_func(upscale, 2) if spatial_dims >= 3 else None

    pos_0 = _get_pos_func(index_formula, upscale, 0)
    pos_1 = _get_pos_func(index_formula, upscale, 1) if spatial_dims >= 2 else None
    pos_2 = _get_pos_func(index_formula, upscale, 2) if spatial_dims >= 3 else None

    partials = []

    if spatial_dims == 1 or extra_channels is not None:
        if latent.ndim == 3:
            partial_latent = latent
        else:
            og_shape = latent.shape
            partial_latent = latent.reshape((og_shape[0], og_shape[1] * og_shape[2], -1))

        if extra_channels is not None:
            og_C = partial_latent.shape[1] // extra_channels
        else:
            og_C = partial_latent.shape[1]

        tile_ranges = _compute_1d_tile_ranges(partial_latent.shape[2], tile_x, overlap_latent)

        tiles = []
        job_index = 0
        with model_management.cuda_device_context(vae.device):
            for x, x_end in tile_ranges:
                if job_index % job_world_size == job_rank:
                    tile = partial_latent[:, :, x:x_end]
                    if extra_channels is not None:
                        tile = tile.reshape(tile.shape[0], og_C, extra_channels, -1)
                    with torch.inference_mode():  # ray workers run outside comfy's inference context; in-place custom ops (RoPE in LTX-2.x/MiniMax causal VAEs) raise under autograd
                        decoded = vae.first_stage_model.decode(tile.to(vae.device, dtype=vae.vae_dtype))
                    if extra_channels is not None:
                        decoded = decoded.reshape(decoded.shape[0], -1, decoded.shape[-1])
                    x_out = _round_upscale(pos_0, x)
                    tiles.append((x, x_out, decoded.to(device="cpu", dtype=decode_dtype, copy=True)))
                job_index += 1

        output_shape = [partial_latent.shape[0], vae.output_channels]
        output_shape.append(_round_upscale(upscale_0, partial_latent.shape[2]))

        partials.append({
            "output_shape": tuple(output_shape),
            "spatial_dims": 1,
            "pass_index": 0,
            "latent_spatial_shape": (partial_latent.shape[2],),
            "overlap_latent": overlap_latent,
            "feather": _compute_feather(upscale, overlap_latent, 1),
            "tiles": tiles,
        })

    elif spatial_dims == 2:
        for pass_idx, (pass_tile_x, pass_tile_y) in enumerate(((tile_x // 2, tile_y * 2), (tile_x * 2, tile_y // 2), (tile_x, tile_y))):
            tile_ranges = _compute_2d_tile_ranges(latent.shape[2], latent.shape[3], pass_tile_y, pass_tile_x, overlap_latent)

            tiles = []
            job_index = 0
            with model_management.cuda_device_context(vae.device):
                for y, y_end, x, x_end in tile_ranges:
                    if job_index % job_world_size == job_rank:
                        tile = latent[:, :, y:y_end, x:x_end]
                        with torch.inference_mode():  # ray workers run outside comfy's inference context; in-place custom ops (RoPE in LTX-2.x/MiniMax causal VAEs) raise under autograd
                            decoded = vae.first_stage_model.decode(tile.to(vae.device, dtype=vae.vae_dtype))
                        y_out = _round_upscale(pos_0, y)
                        x_out = _round_upscale(pos_1, x)
                        tiles.append((y, x, y_out, x_out, decoded.to(device="cpu", dtype=decode_dtype, copy=True)))
                    job_index += 1

            output_shape = [latent.shape[0], vae.output_channels]
            output_shape.append(_round_upscale(upscale_0, latent.shape[2]))
            output_shape.append(_round_upscale(upscale_1, latent.shape[3]))

            partials.append({
                "output_shape": tuple(output_shape),
                "spatial_dims": 2,
                "pass_index": pass_idx,
                "latent_spatial_shape": (latent.shape[2], latent.shape[3]),
                "overlap_latent": overlap_latent,
                "feather": _compute_feather(upscale, overlap_latent, 2),
                "tiles": tiles,
            })

    elif spatial_dims == 3:
        tile_ranges = _compute_2d_tile_ranges(latent.shape[3], latent.shape[4], tile_y, tile_x, overlap_latent)

        tiles = []
        job_index = 0
        with model_management.cuda_device_context(vae.device):
            for y, y_end, x, x_end in tile_ranges:
                if job_index % job_world_size == job_rank:
                    tile = latent[:, :, :, y:y_end, x:x_end]
                    with torch.inference_mode():  # ray workers run outside comfy's inference context; in-place custom ops (RoPE in LTX-2.x/MiniMax causal VAEs) raise under autograd
                        decoded = vae.first_stage_model.decode(tile.to(vae.device, dtype=vae.vae_dtype))
                    y_out = _round_upscale(pos_1, y)
                    x_out = _round_upscale(pos_2, x)
                    tiles.append((y, x, y_out, x_out, decoded.to(device="cpu", dtype=decode_dtype, copy=True)))
                job_index += 1

        output_shape = [latent.shape[0], vae.output_channels]
        output_shape.append(_round_upscale(upscale_0, latent.shape[2]))
        output_shape.append(_round_upscale(upscale_1, latent.shape[3]))
        output_shape.append(_round_upscale(upscale_2, latent.shape[4]))

        partials.append({
            "output_shape": tuple(output_shape),
            "spatial_dims": 3,
            "pass_index": 0,
            "latent_spatial_shape": (latent.shape[3], latent.shape[4]),
            "overlap_latent": overlap_latent,
            "feather": _compute_feather(upscale, overlap_latent, 3),
            "tiles": tiles,
        })

    else:
        raise ValueError(f"Unsupported VAE latent dimensions: {spatial_dims}")

    return partials


def _normalize_worker_result(worker_result):
    if worker_result is None:
        raise ValueError("Distributed VAE decode received a None worker result.")
    if isinstance(worker_result, dict):
        return [worker_result]
    if not isinstance(worker_result, (list, tuple)):
        raise ValueError(f"Distributed VAE decode worker returned invalid result type {type(worker_result).__name__}.")
    if len(worker_result) == 0:
        raise ValueError("Distributed VAE decode worker returned an empty result list.")
    return list(worker_result)


def _validate_worker_passes(worker_partials, expected_passes, spatial_dims):
    for i, worker_result in enumerate(worker_partials):
        partials = _normalize_worker_result(worker_result)
        pass_indices = set()
        for partial in partials:
            if partial["spatial_dims"] != spatial_dims:
                raise ValueError(
                    f"Distributed VAE decode worker {i} returned inconsistent spatial dimensions "
                    f"(expected {spatial_dims}, got {partial['spatial_dims']})."
                )
            idx = partial["pass_index"]
            if idx in pass_indices:
                raise ValueError(f"Distributed VAE decode worker {i} has duplicate pass index {idx}.")
            pass_indices.add(idx)
        if pass_indices != expected_passes:
            missing = expected_passes - pass_indices
            extra = pass_indices - expected_passes
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"unexpected {sorted(extra)}")
            raise ValueError(f"Distributed VAE decode worker {i} has incorrect pass(es): {', '.join(parts)}.")


def combine_dist_vae_partials(worker_partials):
    if not worker_partials:
        raise ValueError("Distributed VAE decode received no worker results.")

    first_result = _normalize_worker_result(worker_partials[0])
    first_partial = first_result[0]
    spatial_dims = first_partial["spatial_dims"]
    if spatial_dims == 2:
        expected_passes = {0, 1, 2}
    else:
        expected_passes = {0}

    _validate_worker_passes(worker_partials, expected_passes, spatial_dims)

    passes = {}
    for worker_result in worker_partials:
        partials = _normalize_worker_result(worker_result)
        for partial in partials:
            idx = partial["pass_index"]
            passes.setdefault(idx, []).append(partial)

    decoded_passes = []
    for pass_index in sorted(passes.keys()):
        group = passes[pass_index]
        first = group[0]
        output_shape = tuple(first["output_shape"])
        overlap_latent = first["overlap_latent"]
        latent_spatial_shape = first["latent_spatial_shape"]
        feather = first["feather"]

        for partial in group:
            if tuple(partial["output_shape"]) != output_shape:
                raise ValueError("Distributed VAE decode workers returned different output shapes.")
            if partial["spatial_dims"] != spatial_dims:
                raise ValueError("Distributed VAE decode workers returned inconsistent spatial dimensions.")
            if partial["overlap_latent"] != overlap_latent:
                raise ValueError("Distributed VAE decode workers returned inconsistent overlap metadata.")
            if tuple(partial["latent_spatial_shape"]) != tuple(latent_spatial_shape):
                raise ValueError("Distributed VAE decode workers returned inconsistent latent spatial shapes.")
            if partial["feather"] != feather:
                raise ValueError("Distributed VAE decode workers returned inconsistent feather metadata.")

        if spatial_dims == 1:
            output = torch.zeros(output_shape, dtype=torch.float32)
            output_div = torch.zeros((1, 1, output_shape[-1]), dtype=torch.float32)
        elif spatial_dims == 2:
            output = torch.zeros(output_shape, dtype=torch.float32)
            output_div = torch.zeros((1, 1, output_shape[-2], output_shape[-1]), dtype=torch.float32)
        else:
            output = torch.zeros(output_shape, dtype=torch.float32)
            output_div = torch.zeros((1, 1, 1, output_shape[-2], output_shape[-1]), dtype=torch.float32)

        for partial in group:
            for tile in partial["tiles"]:
                decoded_tile = tile[-1]

                if spatial_dims == 1:
                    x, x_out = tile[0], tile[1]

                    if decoded_tile.ndim != 3:
                        raise ValueError(f"Distributed VAE decode returned invalid 1D tile shape {tuple(decoded_tile.shape)}.")
                    if decoded_tile.shape[0] != output_shape[0]:
                        raise ValueError(f"Distributed VAE decode returned invalid batch dimension {decoded_tile.shape[0]}.")
                    if decoded_tile.shape[1] != output_shape[1]:
                        raise ValueError(f"Distributed VAE decode returned invalid channel dimension {decoded_tile.shape[1]}.")

                    tile_w_out = decoded_tile.shape[2]
                    if x_out < 0 or x_out + tile_w_out > output_shape[-1]:
                        raise ValueError("Distributed VAE decode returned a tile outside the output bounds.")

                    feather_axis = feather[0]
                    effective_feather = feather_axis if 0 < feather_axis < tile_w_out else 0
                    weight = _linear_feather(tile_w_out, effective_feather)

                    out_slice = output[:, :, x_out:x_out + tile_w_out]
                    div_slice = output_div[:, :, x_out:x_out + tile_w_out]
                    out_slice.add_(decoded_tile.float() * weight)
                    div_slice.add_(weight)

                elif spatial_dims == 2:
                    y, x, y_out, x_out = tile[0], tile[1], tile[2], tile[3]

                    if decoded_tile.ndim != 4:
                        raise ValueError(f"Distributed VAE decode returned invalid 2D tile shape {tuple(decoded_tile.shape)}.")
                    if decoded_tile.shape[0] != output_shape[0]:
                        raise ValueError(f"Distributed VAE decode returned invalid batch dimension {decoded_tile.shape[0]}.")
                    if decoded_tile.shape[1] != output_shape[1]:
                        raise ValueError(f"Distributed VAE decode returned invalid channel dimension {decoded_tile.shape[1]}.")

                    tile_h_out = decoded_tile.shape[2]
                    tile_w_out = decoded_tile.shape[3]
                    if y_out < 0 or y_out + tile_h_out > output_shape[-2]:
                        raise ValueError("Distributed VAE decode returned a tile outside the output bounds.")
                    if x_out < 0 or x_out + tile_w_out > output_shape[-1]:
                        raise ValueError("Distributed VAE decode returned a tile outside the output bounds.")

                    feather_y = feather[0] if 0 < feather[0] < tile_h_out else 0
                    feather_x = feather[1] if 0 < feather[1] < tile_w_out else 0
                    weight_y = _linear_feather(tile_h_out, feather_y)
                    weight_x = _linear_feather(tile_w_out, feather_x)

                    weight = weight_y.view(1, 1, -1, 1) * weight_x.view(1, 1, 1, -1)
                    out_slice = output[:, :, y_out:y_out + tile_h_out, x_out:x_out + tile_w_out]
                    div_slice = output_div[:, :, y_out:y_out + tile_h_out, x_out:x_out + tile_w_out]
                    out_slice.add_(decoded_tile.float() * weight)
                    div_slice.add_(weight)

                elif spatial_dims == 3:
                    y, x, y_out, x_out = tile[0], tile[1], tile[2], tile[3]

                    if decoded_tile.ndim != 5:
                        raise ValueError(f"Distributed VAE decode returned invalid 3D tile shape {tuple(decoded_tile.shape)}.")
                    if decoded_tile.shape[0] != output_shape[0]:
                        raise ValueError(f"Distributed VAE decode returned invalid batch dimension {decoded_tile.shape[0]}.")
                    if decoded_tile.shape[1] != output_shape[1]:
                        raise ValueError(f"Distributed VAE decode returned invalid channel dimension {decoded_tile.shape[1]}.")
                    if decoded_tile.shape[2] != output_shape[2]:
                        raise ValueError(f"Distributed VAE decode returned invalid temporal dimension {decoded_tile.shape[2]}, expected {output_shape[2]}.")

                    tile_h_out = decoded_tile.shape[3]
                    tile_w_out = decoded_tile.shape[4]
                    if y_out < 0 or y_out + tile_h_out > output_shape[-2]:
                        raise ValueError("Distributed VAE decode returned a tile outside the output bounds.")
                    if x_out < 0 or x_out + tile_w_out > output_shape[-1]:
                        raise ValueError("Distributed VAE decode returned a tile outside the output bounds.")

                    feather_y = feather[0] if 0 < feather[0] < tile_h_out else 0
                    feather_x = feather[1] if 0 < feather[1] < tile_w_out else 0
                    weight_y = _linear_feather(tile_h_out, feather_y)
                    weight_x = _linear_feather(tile_w_out, feather_x)

                    weight = weight_y.view(1, 1, 1, -1, 1) * weight_x.view(1, 1, 1, 1, -1)
                    out_slice = output[:, :, :, y_out:y_out + tile_h_out, x_out:x_out + tile_w_out]
                    div_slice = output_div[:, :, :, y_out:y_out + tile_h_out, x_out:x_out + tile_w_out]
                    out_slice.add_(decoded_tile.float() * weight)
                    div_slice.add_(weight)

        if torch.any(output_div == 0):
            raise RuntimeError("Distributed VAE decode did not cover the complete output.")
        decoded_passes.append(output / output_div)

    decoded = decoded_passes[0]
    if len(decoded_passes) > 1:
        decoded = sum(decoded_passes) / len(decoded_passes)
    return decoded


def normalize_seedvr2_latent(latent, latent_channels):
    if latent.ndim == 5:
        if latent.shape[1] != latent_channels:
            raise ValueError(
                f"SeedVR2 distributed decode expected {latent_channels} latent channels, got shape {tuple(latent.shape)}."
            )
        return latent
    if latent.ndim == 4:
        if latent.shape[1] % latent_channels != 0:
            raise ValueError(
                "SeedVR2 distributed decode expected a collapsed latent shaped "
                f"(B, {latent_channels}*T, H, W), got shape {tuple(latent.shape)}."
            )
        return latent.reshape(latent.shape[0], latent_channels, -1, latent.shape[2], latent.shape[3])
    raise ValueError(
        "SeedVR2 distributed decode expected a 5-D (B, C, T, H, W) latent or "
        f"a collapsed 4-D latent, got shape {tuple(latent.shape)}."
    )


def seedvr2_spatial_tile_ranges(height, width, tile_size, overlap):
    tile_size = max(1, int(tile_size))
    overlap = min(max(0, int(overlap)), tile_size - 1)
    stride = max(1, tile_size - overlap)
    ranges = []
    for y in range(0, height, stride):
        y_end = min(y + tile_size, height)
        if y > 0 and y_end - y <= overlap:
            continue
        for x in range(0, width, stride):
            x_end = min(x + tile_size, width)
            if x > 0 and x_end - x <= overlap:
                continue
            ranges.append((y, y_end, x, x_end))
    return ranges, overlap


def ray_seedvr2_vae_decode_partial_impl(worker, samples, tile_size, overlap=64, job_rank=0, job_world_size=1):
    import comfy.model_management as model_management
    from comfy.ldm.seedvr.vae import VideoAutoencoderKLWrapper

    vae = worker.vae_model
    model = vae.first_stage_model
    if not isinstance(model, VideoAutoencoderKLWrapper):
        raise ValueError("SeedVR2 VAE Decode Distributed (Ray) requires a SeedVR2 VAE.")
    _validate_job_rank(job_rank, job_world_size)

    latent = normalize_seedvr2_latent(samples["samples"], vae.latent_channels)
    spatial_scale = model.spatial_downsample_factor
    temporal_scale = model.temporal_downsample_factor
    tile_latent = max(1, tile_size // spatial_scale)
    overlap_latent = max(0, overlap // spatial_scale)
    tile_ranges, overlap_latent = seedvr2_spatial_tile_ranges(
        latent.shape[3], latent.shape[4], tile_latent, overlap_latent
    )

    memory_shape = (
        1,
        latent.shape[1],
        latent.shape[2],
        min(tile_latent, latent.shape[3]),
        min(tile_latent, latent.shape[4]),
    )
    memory_used = vae.memory_used_decode(memory_shape, vae.vae_dtype)
    model_management.load_models_gpu([vae.patcher], memory_required=memory_used, force_full_load=vae.disable_offload)

    tiles = []
    job_index = 0
    with model_management.cuda_device_context(vae.device):
        for batch_index in range(latent.shape[0]):
            for y, y_end, x, x_end in tile_ranges:
                if job_index % job_world_size == job_rank:
                    tile = latent[batch_index:batch_index + 1, :, :, y:y_end, x:x_end]
                    with torch.inference_mode():  # ray workers run outside comfy's inference context; in-place custom ops (RoPE in LTX-2.x/MiniMax causal VAEs) raise under autograd
                        decoded = model.decode(tile.to(device=vae.device, dtype=vae.vae_dtype))
                    tiles.append((batch_index, y, y_end, x, x_end, decoded.to(device="cpu", dtype=torch.float32, copy=True)))
                job_index += 1

    output_shape = (
        latent.shape[0],
        vae.output_channels,
        max(1, latent.shape[2] * temporal_scale - (temporal_scale - 1)),
        latent.shape[3] * spatial_scale,
        latent.shape[4] * spatial_scale,
    )
    return {
        "output_shape": output_shape,
        "latent_spatial_shape": (latent.shape[3], latent.shape[4]),
        "spatial_scale": spatial_scale,
        "overlap": overlap_latent,
        "feather": (round(spatial_scale * overlap_latent),) * 2,
        "tiles": tiles,
    }


def combine_seedvr2_vae_partials(worker_partials):
    if not worker_partials:
        raise ValueError("SeedVR2 distributed decode received no worker results.")

    first = worker_partials[0]
    output_shape = tuple(first["output_shape"])
    latent_height, latent_width = first["latent_spatial_shape"]
    spatial_scale = first["spatial_scale"]
    blend_overlap = first["overlap"] * spatial_scale
    expected_metadata = (
        tuple(first["latent_spatial_shape"]),
        spatial_scale,
        first["overlap"],
    )
    output = torch.zeros(output_shape, dtype=torch.float32)
    output_div = torch.zeros((output_shape[0], 1, 1, output_shape[3], output_shape[4]), dtype=torch.float32)
    ramp_cache = {}

    def get_ramp(steps):
        if steps not in ramp_cache:
            t = torch.linspace(0, 1, steps=steps, dtype=torch.float32)
            ramp_cache[steps] = 0.5 - 0.5 * torch.cos(t * torch.pi)
        return ramp_cache[steps]

    for partial in worker_partials:
        if tuple(partial["output_shape"]) != output_shape:
            raise ValueError("SeedVR2 distributed decode workers returned different output shapes.")
        metadata = (
            tuple(partial["latent_spatial_shape"]),
            partial["spatial_scale"],
            partial["overlap"],
        )
        if metadata != expected_metadata:
            raise ValueError("SeedVR2 distributed decode workers returned inconsistent tiling metadata.")
        for batch_index, y, y_end, x, x_end, tile in partial["tiles"]:
            if not 0 <= batch_index < output_shape[0]:
                raise ValueError(f"SeedVR2 distributed decode returned invalid batch index {batch_index}.")
            if tile.ndim != 5 or tile.shape[0] != 1 or tile.shape[1] != output_shape[1] or tile.shape[2] != output_shape[2]:
                raise ValueError(f"SeedVR2 distributed decode returned invalid tile shape {tuple(tile.shape)}.")
            tile = tile.float()
            overlap_y = min(blend_overlap, tile.shape[3] // 2)
            overlap_x = min(blend_overlap, tile.shape[4] // 2)
            weight_y = torch.ones(tile.shape[3], dtype=torch.float32)
            weight_x = torch.ones(tile.shape[4], dtype=torch.float32)
            if overlap_y > 0:
                ramp = get_ramp(overlap_y)
                if y > 0:
                    weight_y[:overlap_y] = ramp
                if y_end < latent_height:
                    weight_y[-overlap_y:] = 1.0 - ramp
            if overlap_x > 0:
                ramp = get_ramp(overlap_x)
                if x > 0:
                    weight_x[:overlap_x] = ramp
                if x_end < latent_width:
                    weight_x[-overlap_x:] = 1.0 - ramp

            weight = weight_y.view(1, 1, 1, -1, 1) * weight_x.view(1, 1, 1, 1, -1)
            y_out = y * spatial_scale
            x_out = x * spatial_scale
            if y_out + tile.shape[3] > output_shape[3] or x_out + tile.shape[4] > output_shape[4]:
                raise ValueError("SeedVR2 distributed decode returned a tile outside the output bounds.")
            output[batch_index:batch_index + 1, :, :, y_out:y_out + tile.shape[3], x_out:x_out + tile.shape[4]].add_(tile * weight)
            output_div[batch_index:batch_index + 1, :, :, y_out:y_out + tile.shape[3], x_out:x_out + tile.shape[4]].add_(weight)

    if torch.any(output_div == 0):
        raise RuntimeError("SeedVR2 distributed decode did not cover the complete output.")
    return output / output_div


def ray_vae_decode_finalize_impl(worker, decoded):
    images = worker.vae_model.process_output(decoded).movedim(1, -1)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images.cpu()
