import ray

from .context import KarmabuAdapterContext, create_model_proxy, get_proxy_metadata
from .registry import RAY_ACTORS_ATTACHMENT, resolve_adapter


class KarmabuExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ray_actors": ("RAY_ACTORS",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model_proxy",)
    FUNCTION = "export_model"
    CATEGORY = "Raylight/expansion/K3U Adapter"

    def export_model(self, ray_actors):
        worker = ray_actors["workers"][0]
        metadata = ray.get(worker.model_function_runner_get_values.remote(get_proxy_metadata))
        return (create_model_proxy(metadata, ray_actors),)


class KarmabuImport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("K3U_ADAPTER_CONTEXT", "RAY_ACTORS")
    RETURN_NAMES = ("k3u_adapter_context", "ray_actors")
    FUNCTION = "import_model"
    CATEGORY = "Raylight/expansion/K3U Adapter"

    def import_model(self, model):
        ray_actors = model.get_attachment(RAY_ACTORS_ATTACHMENT)
        resolved = resolve_adapter(model)
        if resolved is None:
            return (None, ray_actors)
        spec, wrapper = resolved
        return (KarmabuAdapterContext(model, spec, wrapper), ray_actors)


NODE_CLASS_MAPPINGS = {
    "KarmabuExport": KarmabuExport,
    "KarmabuImport": KarmabuImport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KarmabuExport": "K3U Export",
    "KarmabuImport": "K3U Import",
}
