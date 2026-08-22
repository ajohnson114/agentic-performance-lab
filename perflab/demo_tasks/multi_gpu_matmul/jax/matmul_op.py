"""Editable multi-GPU matmul operation for JAX.

Tensor-parallel style: the K (reduction) dimension is split across every
visible JAX device, each device computes a partial GEMM on its own
K-shard, and a jax.shard_map + jax.lax.psum all-reduce sums the partial
results into the final output. On GPU backends XLA compiles psum to a real
NCCL collective -- the JAX equivalent of torch.cuda.nccl.all_reduce, not a
host round-trip. The agent can edit how work is split across devices and
how the partial results are combined -- for example, balancing the split
evenly instead of the deliberately lopsided one below.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


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


def matmul_op(A: jnp.ndarray, B: jnp.ndarray, devices: list) -> jnp.ndarray:
    """Compute A @ B, splitting the K (contraction) dimension across `devices`.

    A: (M, K), B: (K, N), host-resident. Each device computes
    A_shard @ B_shard for its own K-slice (shape (M, N) on every device
    regardless of shard size, since K is contracted away), then a
    shard_map'd jax.lax.psum sums those partials in place across devices.
    """
    num_devices = len(devices)
    M, K = A.shape
    _, N = B.shape
    sizes = _split_sizes(K, num_devices)

    partials = []
    offset = 0
    for dev, size in zip(devices, sizes, strict=True):
        A_shard = jax.device_put(A[:, offset:offset + size], dev)
        B_shard = jax.device_put(B[offset:offset + size, :], dev)
        partials.append(jnp.matmul(A_shard, B_shard))
        offset += size

    if num_devices == 1:
        return partials[0]

    # Assemble the per-device partials (each already resident on its own
    # device) into one logically-sharded array, then combine with a real
    # cross-device collective -- no host round-trip.
    mesh = Mesh(np.array(devices), axis_names=("dev",))
    sharding = NamedSharding(mesh, PartitionSpec("dev"))
    stacked = jax.make_array_from_single_device_arrays(
        shape=(num_devices, M, N), sharding=sharding,
        arrays=[p.reshape(1, M, N) for p in partials],
    )

    def _combine(x):
        return jax.lax.psum(x[0], axis_name="dev")

    combine = jax.shard_map(
        _combine, mesh=mesh, in_specs=PartitionSpec("dev"), out_specs=PartitionSpec(),
    )
    return combine(stacked)
