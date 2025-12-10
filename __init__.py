import sys
import os

# Comfy dynamic lib loader cause the module imported as a path instead of normal python path
# e.g /home/user/ComfyUI/custom_nodes/raylight.src.raylight.etc
# this code will change it into :
# raylight.wanvideo or raylight.flux
# this is done since ray cluster needs libs to run in this case raylight.
# ray.init(runtime_env={"py_modules":[raylight]})
this_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(this_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

comfy_dir = os.path.abspath(os.path.join(this_dir, "../../comfy"))
if comfy_dir not in sys.path:
    sys.path.insert(0, comfy_dir)

from raylight.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Extra nodes
from raylight.comfy_extra_dist.nodes_torch_compile import NODE_CLASS_MAPPINGS as COMPILE_NODE_CLASS_MAPPINGS
from raylight.comfy_extra_dist.nodes_torch_compile import NODE_DISPLAY_NAME_MAPPINGS as COMPILE_DISPLAY_NAME_MAPPINGS

from raylight.comfy_extra_dist.nodes_model_advanced import NODE_CLASS_MAPPINGS as MODEL_ADV_CLASS_MAPPINGS
from raylight.comfy_extra_dist.nodes_easycache import NODE_CLASS_MAPPINGS as EASY_CACHE_NODE_CLASS_MAPPINGS
from raylight.comfy_extra_dist.nodes_easycache import NODE_DISPLAY_NAME_MAPPINGS as EASY_CACHE_DISPLAY_NAME_MAPPINGS

from raylight.comfy_extra_dist.nodes_custom_sampler import NODE_CLASS_MAPPINGS as SAMPLER_CLASS_MAPPINGS
from raylight.comfy_extra_dist.nodes_custom_sampler import NODE_DISPLAY_NAME_MAPPINGS as SAMPLER_DISPLAY_MAPPINGS

if os.getenv("debug_raylight") == "1":
    print("RAYLIGHT DEBUG MODE")
    from raylight.nodes_debug import NODE_CLASS_MAPPINGS as DEBUG_NODE_CLASS_NAME_MAPPINGS
    from raylight.nodes_debug import NODE_DISPLAY_NAME_MAPPINGS as DEBUG_NODE_DISPLAY_NAME_MAPPINGS
    NODE_CLASS_MAPPINGS.update(DEBUG_NODE_CLASS_NAME_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(DEBUG_NODE_DISPLAY_NAME_MAPPINGS)

gguf_dir = os.path.join(this_dir, "..", "ComfyUI-GGUF")
print(gguf_dir)
gguf_dir = os.path.abspath(gguf_dir)
print(gguf_dir)

if os.path.isdir(gguf_dir):
    from raylight.expansion.comfyui_gguf.nodes import NODE_CLASS_MAPPINGS as GGUF_NODE_CLASS_MAPPINGS
    from raylight.expansion.comfyui_gguf.nodes import NODE_DISPLAY_NAME_MAPPINGS as GGUF_NODE_DISPLAY_NAME_MAPPINGS
    NODE_CLASS_MAPPINGS.update(GGUF_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(GGUF_NODE_DISPLAY_NAME_MAPPINGS)
else:
    print("City96 GGUF not found, GGUF ray loader disable")


# CLASS
NODE_CLASS_MAPPINGS.update(COMPILE_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(MODEL_ADV_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(EASY_CACHE_NODE_CLASS_MAPPINGS)

NODE_CLASS_MAPPINGS.update(SAMPLER_CLASS_MAPPINGS)


# DISPLAY
NODE_DISPLAY_NAME_MAPPINGS.update(COMPILE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(EASY_CACHE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SAMPLER_DISPLAY_MAPPINGS)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__author__ = """Micko Lesmana"""
__email__ = "mickolesmana@gmail.com"
__version__ = "0.14.2"
