# From XDiT, mochi-xdit project
# https://github.com/xdit-project/mochi-xdit
import torch
import torch.distributed as dist

_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_RANK = None
_CONTEXT_PARALLEL_GROUP_SIZE = None
_CONTEXT_PARALLEL_GROUP_RANKS = None

_CFG_PARALLEL_GROUP = None
_CFG_PARALLEL_RANK = None
_CFG_PARALLEL_GROUP_SIZE = None
_CFG_PARALLEL_GROUP_RANKS = None


def get_cp_rank_size():
    if _CONTEXT_PARALLEL_GROUP:
        return _CONTEXT_PARALLEL_RANK, _CONTEXT_PARALLEL_GROUP_SIZE
    else:
        return 0, 1


def local_shard(x: torch.Tensor, dim: int = 2) -> torch.Tensor:
    if not _CONTEXT_PARALLEL_GROUP:
        return x

    cp_rank, cp_size = get_cp_rank_size()
    return x.tensor_split(cp_size, dim=dim)[cp_rank]


def set_cp_group(cp_group, ranks, global_rank):
    global _CONTEXT_PARALLEL_GROUP, _CONTEXT_PARALLEL_RANK, _CONTEXT_PARALLEL_GROUP_SIZE, _CONTEXT_PARALLEL_GROUP_RANKS
    if _CONTEXT_PARALLEL_GROUP is not None:
        raise RuntimeError("CP group already initialized.")
    _CONTEXT_PARALLEL_GROUP = cp_group
    _CONTEXT_PARALLEL_RANK = dist.get_rank(cp_group)
    _CONTEXT_PARALLEL_GROUP_SIZE = dist.get_world_size(cp_group)
    _CONTEXT_PARALLEL_GROUP_RANKS = ranks

    assert _CONTEXT_PARALLEL_RANK == ranks.index(
        global_rank
    ), f"Rank mismatch: {global_rank} in {ranks} does not have position {_CONTEXT_PARALLEL_RANK} "
    assert _CONTEXT_PARALLEL_GROUP_SIZE == len(
        ranks
    ), f"Group size mismatch: {_CONTEXT_PARALLEL_GROUP_SIZE} != len({ranks})"


def get_cp_group():
    if _CONTEXT_PARALLEL_GROUP is None:
        raise RuntimeError("CP group not initialized")
    return _CONTEXT_PARALLEL_GROUP


def is_cp_active():
    return _CONTEXT_PARALLEL_GROUP is not None


# ========================== CFG ========================== #
def set_cfg_group(cfg_group, ranks, global_rank):
    global _CFG_PARALLEL_GROUP, _CFG_PARALLEL_RANK, _CFG_PARALLEL_GROUP_SIZE, _CFG_PARALLEL_GROUP_RANKS
    if _CFG_PARALLEL_GROUP is not None:
        raise RuntimeError("CFG group already initialized.")

    _CFG_PARALLEL_GROUP = cfg_group
    _CFG_PARALLEL_RANK = dist.get_rank(cfg_group)
    _CFG_PARALLEL_GROUP_SIZE = dist.get_world_size(cfg_group)
    _CFG_PARALLEL_GROUP_RANKS = ranks

    assert _CFG_PARALLEL_RANK == ranks.index(
        global_rank
    ), f"Rank mismatch (CFG): {global_rank} not at position {_CFG_PARALLEL_RANK} in {ranks}"

    assert _CFG_PARALLEL_GROUP_SIZE == len(
        ranks
    ), f"CFG group size mismatch: {_CFG_PARALLEL_GROUP_SIZE} != {len(ranks)}"


def get_cfg_group():
    if _CFG_PARALLEL_GROUP is None:
        raise RuntimeError("CFG group not initialized")
    return _CFG_PARALLEL_GROUP


def is_cfg_active():
    return _CFG_PARALLEL_GROUP is not None


def get_cfg_rank_size():
    if _CFG_PARALLEL_GROUP:
        return _CFG_PARALLEL_RANK, _CFG_PARALLEL_GROUP_SIZE
    else:
        return 0, 1


def cfg_local_shard(x: torch.Tensor, dim: int = 2) -> torch.Tensor:
    if not _CFG_PARALLEL_GROUP:
        return x

    cfg_rank, cfg_size = get_cfg_rank_size()
    return x.tensor_split(cfg_size, dim=dim)[cfg_rank]
