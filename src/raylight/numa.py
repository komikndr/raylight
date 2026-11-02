"""Place a worker on the NUMA node its GPU is attached to.

A worker whose threads run on one socket while its GPU hangs off the other
pays the interconnect twice: once for the host side of every transfer, and
again for NCCL's staging buffers. Affinity and memory policy are inherited by
threads created afterwards, so this has to run early in worker startup, before
the process group is built and the model is loaded.

Memory is preferred, not bound: the local node may not have room for the
model, and spilling to another node is better than failing to allocate.
"""

import ctypes
import logging
import os

import torch


def _gpu_numa_node(device_index):
    """The NUMA node a GPU is attached to, or None when unknown."""
    props = torch.cuda.get_device_properties(device_index)
    try:
        bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        with open(f"/sys/bus/pci/devices/{bdf}/numa_node") as fh:
            node = int(fh.read().strip())
    except (AttributeError, OSError, ValueError):
        return None
    return node if node >= 0 else None


def _node_cpus(node):
    """The CPUs belonging to a NUMA node, parsed from its cpulist."""
    with open(f"/sys/devices/system/node/node{node}/cpulist") as fh:
        spec = fh.read().strip()
    cpus = set()
    for part in spec.split(","):
        if "-" in part:
            first, last = part.split("-", 1)
            cpus.update(range(int(first), int(last) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _prefer_node(node):
    lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
    lib.numa_available.argtypes = []
    lib.numa_available.restype = ctypes.c_int
    if lib.numa_available() < 0:
        return False
    lib.numa_set_preferred.argtypes = [ctypes.c_int]
    lib.numa_set_preferred.restype = None
    ctypes.set_errno(0)
    lib.numa_set_preferred(node)
    return ctypes.get_errno() == 0


def bind_to_gpu_node(device_index=0):
    """Place the worker's startup thread on its GPU's NUMA node. Returns a description of
    what was applied, or None when it was skipped or unavailable."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        logging.info("[Raylight][NUMA] sched affinity unsupported here, leaving placement alone")
        return None

    node = _gpu_numa_node(device_index)
    if node is None:
        logging.info("[Raylight][NUMA] GPU reports no NUMA node, leaving placement alone")
        return None

    try:
        # Keep inherited restrictions; an empty overlap means they win.
        cpus = _node_cpus(node) & os.sched_getaffinity(0)
    except (OSError, ValueError) as e:
        logging.warning("[Raylight][NUMA] could not read CPU affinity for node %s: %s", node, e)
        return None

    if not cpus:
        logging.warning("[Raylight][NUMA] node %s has no CPUs inside the inherited affinity, leaving placement alone", node)
        return None

    try:
        os.sched_setaffinity(0, cpus)
    except OSError as e:
        logging.warning("[Raylight][NUMA] sched_setaffinity failed: %s", e)
        return None

    try:
        preferred = _prefer_node(node)
    except OSError as e:
        logging.info("[Raylight][NUMA] memory policy unavailable: %s", e)
        preferred = False

    return f"node={node} cpus={len(cpus)} mem={'preferred' if preferred else 'unchanged'}"
