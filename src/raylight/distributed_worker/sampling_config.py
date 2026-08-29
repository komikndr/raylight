import ray

from raylight.diffusion_models.minimax.block_cache import ACTORS_CONFIG_KEY
from raylight.diffusion_models.minimax.sla import ACTORS_CONFIG_KEY as SLA_ACTORS_CONFIG_KEY


DISABLED_H3_BLOCK_CACHE_CONFIG = {
    "enabled": False,
    "sigma_threshold": 0.0,
    "start_percent": 0.0,
    "end_percent": 1.0,
    "max_cached_steps": 0,
    "cache_depth": 0.0,
    "debug": False,
}


DISABLED_H3_SLA_CONFIG = {
    "enabled": False,
    "sparsity_ratio": 0.90,
    "block_size": 64,
    "min_seq_len": 8192,
    "dense_last_steps": 0,
    "protect_audio": True,
    "debug": False,
}


def configure_sampling_features(ray_actors):
    config = dict(ray_actors.get(ACTORS_CONFIG_KEY, DISABLED_H3_BLOCK_CACHE_CONFIG))
    sla_config = dict(ray_actors.get(SLA_ACTORS_CONFIG_KEY, DISABLED_H3_SLA_CONFIG))
    if config.get("debug", False):
        print(f"[H3 Block Cache][host] dispatch enabled={config.get('enabled', False)} workers={len(ray_actors['workers'])}", flush=True)
    if sla_config.get("debug", False):
        print(f"[H3 SLA][host] dispatch enabled={sla_config.get('enabled', False)} sparsity={sla_config.get('sparsity_ratio', 0.90)} workers={len(ray_actors['workers'])}", flush=True)
    futures = []
    for actor in ray_actors["workers"]:
        futures.append(actor.configure_minimax_h3_block_cache.remote(config))
        futures.append(actor.configure_minimax_h3_sla.remote(sla_config))
    ray.get(futures)
