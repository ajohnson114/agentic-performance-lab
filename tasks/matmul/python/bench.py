"""Benchmark the pure-Python matmul.

Protected file: bench.py is NOT in task.yaml's edit_policy.allowed_paths, so a
candidate cannot edit it, and the workspace is hash-verified after every
iteration. That is what lets the reporting below be trusted:

  * meta.M / meta.N / meta.K are the shapes actually multiplied (not the
    tuning.yaml knobs), which is what makes contract.fixed_params bind — see
    _measured_shape below.
  * meta.repeats / meta.warmup are the counts actually used, which is what
    makes contract.min_repeats / min_warmup bind.
  * The timed output is checked against an fp64 numpy reference, so a kernel
    that only computes correctly at the 64x64x64 shape tests.py uses cannot
    coast through the benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import yaml
from matmul import matmul, random_matrix

# Slack for fp64 accumulation-order differences between a Python left-to-right
# sum and BLAS's blocked/SIMD reassociation (~1e-13 at K=256). Four orders of
# magnitude tighter than the ~1e-6 error a float32 kernel would show, so a
# silent precision downgrade fails here too.
MAX_ABS_ERR = 1e-9


def tflops(M: int, N: int, K: int, seconds: float) -> float:
    flops = 2.0 * M * N * K
    return flops / seconds / 1e12


def _percentile_index(fraction: float, n: int) -> int:
    """Ceiling-indexed position of `fraction` into a sorted list of length n.

    Floor indexing (int(fraction * (n - 1))) rounds toward the middle of the
    distribution, so at the small sample counts this task actually uses
    (n=2 during fast screening, n=3 by default) the extreme value falls just
    past the computed index and is never reported -- e.g. n=2:
    int(0.95*1)=0 returns the MINIMUM as "p95". Mirrors
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


def _measured_shape(A, B) -> tuple[int, int, int]:
    """Return (M, N, K) as actually laid out by the operands.

    random_matrix() lives in the editable matmul.py, so the tuning.yaml knobs
    are a request, not a guarantee. Reporting the real shapes is what turns
    contract.fixed_params into an enforced floor on problem size.
    """
    return len(A), len(B[0]), len(A[0])


def _check_result(C, A, B) -> float:
    """Validate the timed output against an fp64 numpy reference.

    Runs outside the timed region. Returns the max absolute error so it can be
    published in bench.json alongside the timings.
    """
    out = np.asarray(C, dtype=np.float64)
    ref = np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64)
    if out.shape != ref.shape:
        raise SystemExit(
            f"benchmarked matmul returned shape {out.shape}, expected {ref.shape}"
        )
    if not np.isfinite(out).all():
        raise SystemExit("benchmarked matmul returned non-finite values")
    max_abs = float(np.abs(out - ref).max())
    if not max_abs < MAX_ABS_ERR:
        raise SystemExit(
            f"benchmarked matmul is wrong: max_abs={max_abs:.3e} >= {MAX_ABS_ERR:g}"
        )
    return max_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Output JSON path")
    args = ap.parse_args()

    knobs = yaml.safe_load(Path("tuning.yaml").read_text(encoding="utf-8"))
    M = int(knobs.get("M", 256))
    N = int(knobs.get("N", 256))
    K = int(knobs.get("K", 256))

    A = random_matrix(M, K, seed=42)
    B = random_matrix(K, N, seed=123)
    M, N, K = _measured_shape(A, B)

    # Warmup
    warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 1))
    for _ in range(warmup):
        _ = matmul(A, B)

    # Benchmark
    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 3))
    times_ms = []
    C = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        C = matmul(A, B)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    max_abs_err = _check_result(C, A, B)

    sorted_times = sorted(times_ms)
    p50 = _true_median(sorted_times)
    p95 = sorted_times[_percentile_index(0.95, len(sorted_times))]
    tflops_med = tflops(M, N, K, p50 / 1000.0)
    # Per-repeat tflops, in measurement order -- what the accept gate's
    # variance check (extract_repeated_values) needs alongside the aggregate.
    tflops_list = [tflops(M, N, K, t / 1000.0) for t in times_ms]

    out = {
        "meta": {
            "M": M,
            "N": N,
            "K": K,
            # The counts genuinely used above, after the PERFLAB_BENCH_*
            # overrides the fast screen applies. Without these,
            # contract.min_repeats/min_warmup are inert.
            "repeats": repeats,
            "warmup": warmup,
            "max_abs_err": max_abs_err,
        },
        "times_ms": times_ms,
        "latency_ms": {"p50": p50, "p95": p95, "raw_values": times_ms},
        "tflops": {"median": tflops_med, "raw_values": tflops_list},
        "ok": True,
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"tflops_median": tflops_med, "lat_ms_p50": p50}, indent=2))


if __name__ == "__main__":
    main()
