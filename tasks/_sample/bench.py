"""Benchmark harness for the sample task.

Every PerfLab bench.py must:
  1. Accept --json <path> to write results
  2. Honor PERFLAB_BENCH_WARMUP and PERFLAB_BENCH_REPEATS env vars
  3. Write a JSON file with at least the metric referenced in task.yaml

The metric path in task.yaml (e.g. "throughput.median") is a dotted path
into this JSON, so the structure must match.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import median

import yaml

from sample import sum_of_squares


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Output JSON path")
    args = parser.parse_args()

    # Load tunable parameters
    knobs = yaml.safe_load(Path("tuning.yaml").read_text(encoding="utf-8"))
    size = int(knobs.get("size", 512))

    # Warmup — respect env var for fast screening
    warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 2))
    for _ in range(warmup):
        sum_of_squares(size)

    # Timed runs — respect env var for fast screening
    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 10))
    times_s = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sum_of_squares(size)
        t1 = time.perf_counter()
        times_s.append(t1 - t0)

    # Compute throughput: elements processed per second
    throughput_list = [size / t for t in times_s]
    med = median(throughput_list)

    out = {
        "throughput": {
            "median": med,
            "all": throughput_list,
        },
        "latency_ms": {
            "median": median(times_s) * 1000,
        },
        "meta": {
            "size": size,
            "warmup": warmup,
            "repeats": repeats,
        },
        "ok": True,
    }

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"throughput.median = {med:.1f} elements/sec")


if __name__ == "__main__":
    main()
