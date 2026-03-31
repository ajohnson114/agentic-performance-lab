"""Benchmark the pure-Python matmul."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import yaml

from matmul import matmul, random_matrix


def tflops(M: int, N: int, K: int, seconds: float) -> float:
    flops = 2.0 * M * N * K
    return flops / seconds / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Output JSON path")
    args = ap.parse_args()

    knobs = yaml.safe_load(Path("tuning.yaml").read_text(encoding="utf-8"))
    M = int(knobs.get("M", 128))
    N = int(knobs.get("N", 128))
    K = int(knobs.get("K", 128))

    A = random_matrix(M, K, seed=42)
    B = random_matrix(K, N, seed=123)

    # Warmup
    warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 1))
    for _ in range(warmup):
        _ = matmul(A, B)

    # Benchmark
    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 3))
    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        C = matmul(A, B)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    sorted_times = sorted(times_ms)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(0.95 * (len(sorted_times) - 1))]
    tflops_med = tflops(M, N, K, p50 / 1000.0)

    out = {
        "meta": {"M": M, "N": N, "K": K},
        "times_ms": times_ms,
        "latency_ms": {"p50": p50, "p95": p95},
        "tflops": {"median": tflops_med},
        "ok": True,
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"tflops_median": tflops_med, "lat_ms_p50": p50}, indent=2))


if __name__ == "__main__":
    main()
