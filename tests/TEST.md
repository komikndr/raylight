# Tests

Run from the ComfyUI repository root with the Python environment that contains the project dependencies.

Examples below assume `python` and `torchrun` are already on `PATH`.

Main test script:

```bash
custom_nodes/raylight/tests/fsdp_quant_compare.py
```

CPU / gloo uses eager patch files:
- `custom_nodes/raylight/src/raylight/comfy_dist/kitchen_patches/fp8_eager.py`
- `custom_nodes/raylight/src/raylight/comfy_dist/kitchen_patches/nvfp4_eager.py`

CUDA / nccl uses active patch files:
- `custom_nodes/raylight/src/raylight/comfy_dist/kitchen_patches/fp8.py`
- `custom_nodes/raylight/src/raylight/comfy_dist/kitchen_patches/nvfp4.py`

Tiny smoke tests:

```bash
torchrun --standalone --nproc_per_node=2 custom_nodes/raylight/tests/fsdp_quant_compare.py --device cpu --layout fp8 --mode tiny --dim 256
torchrun --standalone --nproc_per_node=2 custom_nodes/raylight/tests/fsdp_quant_compare.py --device cpu --layout nvfp4 --mode tiny --dim 256
torchrun --standalone --nproc_per_node=2 custom_nodes/raylight/tests/fsdp_quant_compare.py --device cuda --layout fp8 --mode tiny --dim 256
torchrun --standalone --nproc_per_node=2 custom_nodes/raylight/tests/fsdp_quant_compare.py --device cuda --layout nvfp4 --mode tiny --dim 256
```

Memory profile examples:

```bash
torchrun --standalone --nproc_per_node=2 custom_nodes/raylight/tests/fsdp_quant_compare.py --device cpu --layout fp8 --mode memory --target-gib 2.0 --dim 1024
torchrun --standalone --nproc_per_node=2 custom_nodes/raylight/tests/fsdp_quant_compare.py --device cpu --layout nvfp4 --mode memory --target-gib 2.0 --dim 1024
```

Arguments:
- `--layout {fp8,nvfp4}`
- `--device {cpu,cuda}`
- `--mode {tiny,memory}`
- `--target-gib FLOAT`
- `--dim INT`

Output checks:
- `input_scale_report` must have no `meta`, no `missing`, no `mismatches`
- `meta_params_after_load` must be empty
- `mem_*` contains RSS for CPU and RSS + VRAM stats for CUDA
