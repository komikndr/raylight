"""A full load leaves a CPU-offloaded FSDP model's shards on the host.

ComfyUI asks for a full load whenever its budget says the model fits. For a
CPU-offloaded FSDP model that used to move every shard onto the card, on
whichever ranks cleared the threshold - one rank of two, in the case that found
it, which then ran out of memory while its peer did not.
"""
import pytest
import torch

if not torch.cuda.is_available():
    # comfy picks its device at import and needs telling there is no card
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

from raylight.comfy_dist.model_patcher import FSDPModelPatcher


class _Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Linear(4, 4)
        self.moved_to = []

    def to(self, *args, **kwargs):
        self.moved_to.append(args[0] if args else kwargs.get("device"))
        return self


def _patcher(cpu_offload):
    root = _Root()
    patcher = FSDPModelPatcher(root, load_device=torch.device("cpu"), offload_device=torch.device("cpu"),
                               is_cpu_offload=cpu_offload)
    patcher.patch_fsdp = lambda: root
    return patcher, root


@pytest.mark.parametrize("cpu_offload, moved", [(True, False), (False, True)])
def test_a_full_load_moves_only_a_model_that_is_not_cpu_offloaded(cpu_offload, moved):
    patcher, root = _patcher(cpu_offload)

    patcher.load(torch.device("cpu"), lowvram_model_memory=1 << 40, full_load=True)

    assert bool(root.moved_to) is moved
