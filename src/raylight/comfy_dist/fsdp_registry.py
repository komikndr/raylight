from torch.distributed.fsdp import FSDPModule
from comfy import model_base

from ..diffusion_models.wan.fsdp import shard_model_fsdp2 as wan_shard
from ..diffusion_models.flux.fsdp import shard_model_fsdp2 as flux_shard
from ..diffusion_models.chroma.fsdp import shard_model_fsdp2 as chroma_shard
from ..diffusion_models.chroma_radiance.fsdp import shard_model_fsdp2 as chroma_radiance_shard
from ..diffusion_models.qwen_image.fsdp import shard_model_fsdp2 as qwen_shard
from ..diffusion_models.hunyuan_video.fsdp import shard_model_fsdp2 as hunyuan_shard
from ..diffusion_models.lumina.fsdp import shard_model_fsdp2 as lumina_shard


class FSDPShardRegistry:
    _REGISTRY = {}

    @classmethod
    def register(cls, model_class):
        """Register a model class and its FSDP shard handler.

        IMPORTANT:
        Always register the **most specific (child)** model classes FIRST,
        followed by their **parent/base** classes.

        Example:
            @FSDPShardRegistry.register(model_base.Chroma)  # subclass of Flux
            def _wrap_chroma(...): ...

            @FSDPShardRegistry.register(model_base.Flux)
            def _wrap_flux(...): ...

        This ensures isinstance() checks correctly dispatch to the subclass handler
        before falling back to the base handler.
        """
        def decorator(shard_func):
            cls._REGISTRY[model_class] = shard_func
            return shard_func
        return decorator

    @classmethod
    def wrap(cls, model, fsdp_state_dict=None, cpu_offload=False):
        """Find the right shard function based on model type."""
        for registered_cls, shard_func in cls._REGISTRY.items():
            if isinstance(model, registered_cls):
                print(f"[FSDPRegistry] Wrapping {registered_cls.__name__}")
                return shard_func(model, fsdp_state_dict, cpu_offload)


# Register per-model handlers

@FSDPShardRegistry.register(model_base.WAN21)
@FSDPShardRegistry.register(model_base.WAN22)
def _wrap_wan(model, sd, cpu_offload):
    return wan_shard(model, sd, cpu_offload)


@FSDPShardRegistry.register(model_base.ChromaRadiance)
def _wrap_chroma_radiance(model, sd, cpu_offload):
    return chroma_radiance_shard(model, sd, cpu_offload)


@FSDPShardRegistry.register(model_base.Chroma)
def _wrap_chroma(model, sd, cpu_offload):
    return chroma_shard(model, sd, cpu_offload)


@FSDPShardRegistry.register(model_base.Flux)
def _wrap_flux(model, sd, cpu_offload):
    return flux_shard(model, sd, cpu_offload)


@FSDPShardRegistry.register(model_base.QwenImage)
def _wrap_qwen(model, sd, cpu_offload):
    return qwen_shard(model, sd, cpu_offload)


@FSDPShardRegistry.register(model_base.Hunyuan3Dv2)
def _wrap_hunyuan_3dv2(model, sd, cpu_offload):
    return flux_shard(model, sd, cpu_offload)


@FSDPShardRegistry.register(model_base.HunyuanVideo)
def _wrap_hunyuan(model, sd, cpu_offload):
    return hunyuan_shard(model, sd, cpu_offload)


@FSDPShardRegistry.register(model_base.Lumina2)
def _wrap_lumina(model, sd, cpu_offload):
    return lumina_shard(model, sd, True)


def patch_fsdp(self):
    print(f"[Rank {self.rank}] Applying FSDP to {type(self.model.diffusion_model).__name__}")

    if isinstance(self.model.diffusion_model, FSDPModule):
        print("FSDP already registered, skip wrapping...")
        return self.model

    try:
        self.model = FSDPShardRegistry.wrap(
            self.model,
            fsdp_state_dict=self.fsdp_state_dict,
            cpu_offload=self.is_cpu_offload,
        )
        print("FSDP registered successfully.")
    except ValueError as e:
        raise ValueError(f"{type(self.model.diffusion_model).__name__} IS CURRENTLY NOT SUPPORTED FOR FSDP") from e

    return self.model
