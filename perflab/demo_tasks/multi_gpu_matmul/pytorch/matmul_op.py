"""Editable multi-GPU matmul operation for PyTorch.

Tensor-parallel style: the K (reduction) dimension is split across every
visible CUDA device, each device computes a partial GEMM on its own
K-shard, and an NCCL all-reduce sums the partial results into the final
output. The agent can edit how work is split across devices and how the
partial results are combined -- for example, balancing the split evenly
instead of the deliberately lopsided one below.
"""
from __future__ import annotations

import torch
import torch.cuda.nccl as nccl


def _split_sizes(K: int, num_devices: int) -> list[int]:
    """Naive split: device 0 gets the lion's share of K, everyone else
    splits what's left evenly. On a 2-GPU box this hands ~90% of the
    compute to GPU 0 and ~10% to GPU 1 -- a deliberate load-imbalance bug
    for the agent to find (perflab's gpu_active_pct_by_device /
    "GPU load imbalance" bottleneck rule) and fix.
    """
    if num_devices == 1:
        return [K]
    first = int(K * 0.9)
    rest = K - first
    base = rest // (num_devices - 1)
    sizes = [first] + [base] * (num_devices - 1)
    sizes[-1] += rest - base * (num_devices - 1)
    return sizes


def matmul_op(A: torch.Tensor, B: torch.Tensor, devices: list[torch.device]) -> torch.Tensor:
    """Compute A @ B, splitting the K (contraction) dimension across `devices`.

    A: (M, K) on devices[0]. B: (K, N) on devices[0]. Each device computes
    A_shard @ B_shard for its own K-slice (shape (M, N) on every device
    regardless of shard size, since K is contracted away), then
    torch.cuda.nccl.all_reduce sums those partials in place -- a genuine
    NCCL collective, not torch.cuda.comm's P2P-copy-based fallback, so it
    shows up as an "nccl"-named kernel in a trace the way real multi-GPU
    tensor-parallel code does.
    """
    num_devices = len(devices)
    K = A.shape[-1]
    sizes = _split_sizes(K, num_devices)

    partials = []
    offset = 0
    for dev, size in zip(devices, sizes, strict=True):
        A_shard = A[:, offset:offset + size].to(dev, non_blocking=True)
        B_shard = B[offset:offset + size, :].to(dev, non_blocking=True)
        partials.append(A_shard @ B_shard)
        offset += size

    if num_devices == 1:
        return partials[0]

    nccl.all_reduce(partials)  # in-place: every entry now holds the summed result
    return partials[0]
