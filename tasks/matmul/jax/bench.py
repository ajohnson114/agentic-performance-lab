"""Benchmark JAX jnp.matmul (no jit — agent must discover and add it)."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml
from matmul_op import matmul_op


def tflops(M: int, N: int, K: int, seconds: float) -> float:
    flops = 2.0 * M * N * K
    return flops / seconds / 1e12


def _percentile_index(fraction: float, n: int) -> int:
    """Ceiling-indexed position of `fraction` into a sorted list of length n.

    Floor indexing (int(fraction * (n - 1))) rounds toward the middle of the
    distribution, so at small sample counts (n=2 during fast screening) the
    extreme value falls just past the computed index and is never reported --
    e.g. n=2: int(0.95*1)=0 returns the MINIMUM as "p95". Mirrors
    perflab.harness.precision._ceil_percentile_index.
    """
    return min(n - 1, math.ceil(fraction * (n - 1)))


def _true_median(sorted_values: list[float]) -> float:
    """Median of an already-sorted list, averaging the middle pair when n is even.

    sorted_values[n // 2] alone (the previous implementation here) is biased
    high for even n -- at n=2 it always returns the larger of the two samples.
    Matches perflab.analyzers.bench_stats.compute_bench_stats's median.
    """
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    return sorted_values[mid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Output JSON path")
    args = ap.parse_args()

    knobs = yaml.safe_load(Path("tuning.yaml").read_text(encoding="utf-8"))
    M = int(knobs.get("M", 2048))
    N = int(knobs.get("N", 2048))
    K = int(knobs.get("K", 2048))

    jax_dtype = jnp.float32

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (M, K), dtype=jax_dtype)
    B = jax.random.normal(k2, (K, N), dtype=jax_dtype)

    # Warmup (at least 1 for JIT compilation cost during fast screening)
    warmup = max(1, int(os.environ.get("PERFLAB_BENCH_WARMUP", 1)))
    for _ in range(warmup):
        C = matmul_op(A, B)
        C.block_until_ready()

    # Benchmark
    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 20))
    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        C = matmul_op(A, B)
        C.block_until_ready()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    sorted_times = sorted(times_ms)
    p50 = _true_median(sorted_times)
    p95 = sorted_times[_percentile_index(0.95, len(sorted_times))]
    tflops_med = tflops(M, N, K, p50 / 1000.0)
    # Per-repeat tflops, in measurement order -- what the accept gate's
    # variance check (extract_repeated_values) needs alongside the aggregate.
    tflops_list = [tflops(M, N, K, t / 1000.0) for t in times_ms]

    device = str(jax.devices()[0])

    out = {
        "meta": {
            "device": device, "dtype": "float32", "M": M, "N": N, "K": K,
            "warmup": warmup, "repeats": repeats,
        },
        "times_ms": times_ms,
        "latency_ms": {"p50": p50, "p95": p95, "raw_values": times_ms},
        "tflops": {"median": tflops_med, "raw_values": tflops_list},
        "ok": True,
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"tflops_median": tflops_med, "lat_ms_p50": p50, "device": device}, indent=2))


if __name__ == "__main__":
    main()
