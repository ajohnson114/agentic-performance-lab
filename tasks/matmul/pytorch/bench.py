from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml
from matmul_op import matmul_op


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _sync(dev: torch.device):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()

def _tflops(M: int, N: int, K: int, batch: int, seconds: float) -> float:
    # GEMM FLOPs ~ 2*M*N*K per matmul
    flops = 2.0 * M * N * K * batch
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

def maybe_torch_profiler_enabled() -> tuple[bool, str | None]:
    if os.environ.get("PERFLAB_TORCH_PROFILE", "0") != "1":
        return False, None
    return True, os.environ.get("PERFLAB_TORCH_TRACE_PATH")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    # GPU-scale problem: 8192^3 fp16 is ~1.1 TFLOP/iteration (~400 MB of
    # operands) -- a real workload for an H100 and still tractable on Apple
    # Silicon MPS. These sizes live here, in the tamper-protected bench.py,
    # rather than in the agent-editable tuning.yaml, and are additionally
    # pinned by contract.fixed_params so a candidate cannot shrink the
    # problem to manufacture a "win".
    ap.add_argument("--M", type=int, default=8192)
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--K", type=int, default=8192)
    args = ap.parse_args()

    knobs = yaml.safe_load(Path("tuning.yaml").read_text(encoding="utf-8"))
    dtype = knobs.get("dtype", "fp16")
    batch = int(knobs.get("batch", 1))

    dev = _device()
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32

    # Setup
    torch.manual_seed(0)
    A = torch.randn(batch, args.M, args.K, device=dev, dtype=torch_dtype)
    B = torch.randn(batch, args.K, args.N, device=dev, dtype=torch_dtype)

    # Warmup
    _sync(dev)
    warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
    for _ in range(warmup):
        matmul_op(A, B)
    _sync(dev)

    # Benchmark
    times = []
    enabled, trace_path = maybe_torch_profiler_enabled()
    prof = None
    if enabled:
        from torch.profiler import ProfilerActivity, profile
        activities = [ProfilerActivity.CPU]
        # CUDA activity only when available
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        prof = profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True)

    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 20))
    if prof is None:
        for _ in range(repeats):
            t0 = time.perf_counter()
            matmul_op(A, B)
            _sync(dev)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    else:
        with prof:
            for _ in range(repeats):
                t0 = time.perf_counter()
                matmul_op(A, B)
                _sync(dev)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(trace_path)

    times_sorted = sorted(times)
    p50 = _true_median(times_sorted)
    p95 = times_sorted[_percentile_index(0.95, len(times_sorted))]
    med_ms = p50
    tflops_med = _tflops(args.M, args.N, args.K, batch, med_ms/1000.0)
    # Per-repeat tflops, in measurement order -- what the accept gate's
    # variance check (extract_repeated_values) needs alongside the aggregate.
    tflops_list = [_tflops(args.M, args.N, args.K, batch, t / 1000.0) for t in times]

    out = {
        # repeats/warmup report the counts ACTUALLY used (after any
        # PERFLAB_BENCH_* override) so contract.min_repeats/min_warmup can be
        # enforced instead of silently skipped.
        "meta": {
            "device": dev.type, "dtype": dtype, "batch": batch,
            "M": args.M, "N": args.N, "K": args.K,
            "repeats": repeats, "warmup": warmup,
        },
        "times_ms": times,
        "latency_ms": {"p50": p50, "p95": p95, "raw_values": times},
        "tflops": {"median": tflops_med, "raw_values": tflops_list},
        "ok": True,
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"tflops_median": tflops_med, "lat_ms_p50": p50, "device": dev.type}, indent=2))

if __name__ == "__main__":
    main()
