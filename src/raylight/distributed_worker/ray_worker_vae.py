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
