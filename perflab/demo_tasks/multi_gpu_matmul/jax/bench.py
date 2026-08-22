from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
from matmul_op import matmul_op


def _devices() -> list:
    devices = jax.devices()
    if len(devices) < 2:
        raise RuntimeError(
            f"multi_gpu_matmul requires >=2 JAX devices, found {len(devices)}. "
            "This task demonstrates multi-GPU profiling and is meant to run "
            "on a multi-GPU box (e.g. a 2x+ GPU RunPod instance)."
        )
    return devices


def _tflops(M: int, N: int, K: int, seconds: float) -> float:
    flops = 2.0 * M * N * K
    return flops / seconds / 1e12


def _percentile_index(fraction: float, n: int) -> int:
    """Ceiling-indexed position of `fraction` into a sorted list of length n.
    Mirrors perflab.harness.precision._ceil_percentile_index -- see
    matmul/jax/bench.py for the full rationale.
    """
    return min(n - 1, math.ceil(fraction * (n - 1)))


def _true_median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    return sorted_values[mid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    # GPU-scale problem, pinned by contract.fixed_params -- see that comment
    # in task.yaml for why (prevents a candidate from shrinking the problem
    # to manufacture a "win").
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    args = ap.parse_args()
    M, N, K = args.M, args.N, args.K

    devices = _devices()

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (M, K), dtype=jnp.float32)
    B = jax.random.normal(k2, (K, N), dtype=jnp.float32)

    warmup = max(1, int(os.environ.get("PERFLAB_BENCH_WARMUP", 3)))
    for _ in range(warmup):
        matmul_op(A, B, devices).block_until_ready()

    times = []
    trace_dir = os.environ.get("PERFLAB_JAX_TRACE_DIR")
    ctx = jax.profiler.trace(trace_dir) if trace_dir else nullcontext()

    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 10))
    with ctx:
        for _ in range(repeats):
            t0 = time.perf_counter()
            matmul_op(A, B, devices).block_until_ready()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times_sorted = sorted(times)
    p50 = _true_median(times_sorted)
    p95 = times_sorted[_percentile_index(0.95, len(times_sorted))]
    tflops_med = _tflops(M, N, K, p50 / 1000.0)
    tflops_list = [_tflops(M, N, K, t / 1000.0) for t in times]

    out = {
        "meta": {
            "device": str(devices[0]), "num_gpus": len(devices),
            "M": M, "N": N, "K": K,
            "repeats": repeats, "warmup": warmup,
        },
        "times_ms": times,
        "latency_ms": {"p50": p50, "p95": p95, "raw_values": times},
        "tflops": {"median": tflops_med, "raw_values": tflops_list},
        "num_gpus": len(devices),
        "ok": True,
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"tflops_median": tflops_med, "lat_ms_p50": p50, "num_gpus": len(devices)}, indent=2))


if __name__ == "__main__":
    main()
