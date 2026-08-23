from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_every_ray_sampler_pushes_fresh_sampling_config():
    nodes_source = (ROOT / "src/raylight/nodes.py").read_text()
    custom_source = (ROOT / "src/raylight/comfy_extra_dist/nodes_custom_sampler.py").read_text()

    assert nodes_source.count("configure_sampling_features(ray_actors)") == nodes_source.count("actor.common_ksampler.remote(") == 3
    assert custom_source.count("configure_sampling_features(ray_actors)") == (
        custom_source.count("actor.custom_sampler.remote(") + custom_source.count("actor.custom_sampler_advanced.remote(")
    ) == 6
