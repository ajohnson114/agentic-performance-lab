"""Benchmark harness for stream operations.

Buffer ownership lives HERE, not in stream.py, and deliberately so. This file
is protected (PROTECTED_FILENAMES), stream.py is agent-editable.

Originally run_stream() allocated three 134 MB arrays and called
np.random.randn twice on every invocation, all inside the timed region. Once
the scalar loops are vectorized that setup *dominates*: measured 0.299 s of
alloc+RNG against ~0.295 s total for an honestly-vectorized run. The metric
then rewards caching the buffers in module globals -- worth ~10x -- which is
not a streaming optimization at all. Allocating once here, and restoring the
inputs between repeats OUTSIDE the timer, leaves the timed region measuring
only the four stream kernels.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from stream import N, run_stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Output JSON path")
    args = parser.parse_args()

    warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", "1"))
    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", "5"))

    # Allocated once, outside every timed region.
    rng = np.random.default_rng(0xC0FFEE)
    A = np.zeros((N, N), dtype=np.float64)
    B = np.empty((N, N), dtype=np.float64)
    C = np.empty((N, N), dtype=np.float64)
    # Pristine sources: the kernels overwrite A/B/C, so inputs are restored
    # before each repeat to keep every measurement identical.
    b_src = rng.standard_normal((N, N), dtype=np.float64)
    c_src = rng.standard_normal((N, N), dtype=np.float64)

    def reset():
        np.copyto(B, b_src)
        np.copyto(C, c_src)
        A.fill(0.0)

    # Warmup
    for _ in range(warmup):
        reset()
        run_stream(A, B, C)

    # Timed runs
    times = []
    total_bytes = None
    for _ in range(repeats):
        reset()  # outside the timer
        t0 = time.perf_counter()
        total_bytes = run_stream(A, B, C)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    # Compute throughput in GB/s, in measurement order -- raw_values must stay
    # per-repeat (consumed by the accept gate's variance check,
    # extract_repeated_values), so the median is computed from a separate
    # sorted copy rather than sorting this list in place.
    throughputs = [total_bytes / t / 1e9 for t in times]
    sorted_throughputs = sorted(throughputs)
    n = len(sorted_throughputs)
    median_tp = (
        sorted_throughputs[n // 2]
        if n % 2 == 1
        else (sorted_throughputs[n // 2 - 1] + sorted_throughputs[n // 2]) / 2
    )

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "ok": True,
        "throughput": {
            "median": round(median_tp, 4),
            "raw_values": [round(t, 4) for t in throughputs],
            "unit": "GB/s",
        },
        "meta": {
            "N": N,
            "dtype": "float64",
            "total_bytes": total_bytes,
            "warmup": warmup,
            "repeats": repeats,
        },
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"throughput.median = {median_tp:.4f} GB/s")


if __name__ == "__main__":
    main()
