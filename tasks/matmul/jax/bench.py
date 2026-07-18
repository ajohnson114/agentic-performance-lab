"""Benchmark JAX jnp.matmul (no jit — agent must discover and add it)."""
from __future__ import annotations

import argparse
import json
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
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(0.95 * (len(sorted_times) - 1))]
    tflops_med = tflops(M, N, K, p50 / 1000.0)

    device = str(jax.devices()[0])

    out = {
        "meta": {"device": device, "dtype": "float32", "M": M, "N": N, "K": K},
        "times_ms": times_ms,
        "latency_ms": {"p50": p50, "p95": p95},
        "tflops": {"median": tflops_med},
        "ok": True,
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"tflops_median": tflops_med, "lat_ms_p50": p50, "device": device}, indent=2))


if __name__ == "__main__":
    main()
