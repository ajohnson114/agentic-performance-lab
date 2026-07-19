"""Benchmark harness for stream operations."""
import argparse
import json
import os
import time
from pathlib import Path

from stream import N, run_stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Output JSON path")
    args = parser.parse_args()

    warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", "1"))
    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", "5"))

    # Warmup
    for _ in range(warmup):
        run_stream()

    # Timed runs
    times = []
    total_bytes = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        total_bytes, _ = run_stream()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    # Compute throughput in GB/s
    throughputs = [total_bytes / t / 1e9 for t in times]
    throughputs.sort()
    n = len(throughputs)
    median_tp = throughputs[n // 2] if n % 2 == 1 else (throughputs[n // 2 - 1] + throughputs[n // 2]) / 2

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
        },
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"throughput.median = {median_tp:.4f} GB/s")


if __name__ == "__main__":
    main()
