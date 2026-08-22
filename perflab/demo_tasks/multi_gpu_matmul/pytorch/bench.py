from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from matmul_op import matmul_op


def _devices() -> list[torch.device]:
    n = torch.cuda.device_count()
    if n < 2:
        raise RuntimeError(
            f"multi_gpu_matmul requires >=2 CUDA devices, found {n}. "
            "This task demonstrates multi-GPU profiling and is meant to run "
            "on a multi-GPU box (e.g. a 2x+ GPU RunPod instance)."
        )
    return [torch.device(f"cuda:{i}") for i in range(n)]


def _sync(devices: list[torch.device]):
    for d in devices:
        torch.cuda.synchronize(d)


def _tflops(M: int, N: int, K: int, seconds: float) -> float:
    flops = 2.0 * M * N * K
    return flops / seconds / 1e12


def _percentile_index(fraction: float, n: int) -> int:
    """Ceiling-indexed position of `fraction` into a sorted list of length n.
    Mirrors perflab.harness.precision._ceil_percentile_index -- see
    matmul/pytorch/bench.py for the full rationale.
    """
    return min(n - 1, math.ceil(fraction * (n - 1)))


def _true_median(sorted_values: list[float]) -> float:
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
    # GPU-scale problem. Pinned by contract.fixed_params in task.yaml so a
    # candidate can't "optimize" by shrinking the problem or reporting a
    # bigger tflops number for a smaller one -- see that file's comment.
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    args = ap.parse_args()

    devices = _devices()
    dev0 = devices[0]

    torch.manual_seed(0)
    A = torch.randn(args.M, args.K, device=dev0, dtype=torch.float16)
    B = torch.randn(args.K, args.N, device=dev0, dtype=torch.float16)

    warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
    for _ in range(warmup):
        matmul_op(A, B, devices)
    _sync(devices)

    times = []
    enabled, trace_path = maybe_torch_profiler_enabled()
    prof = None
    if enabled:
        from torch.profiler import ProfilerActivity, profile
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True, with_stack=True,
        )

    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 10))
    with prof if prof is not None else nullcontext():
        for _ in range(repeats):
            t0 = time.perf_counter()
            matmul_op(A, B, devices)
            _sync(devices)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    if prof is not None and trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(trace_path)

    times_sorted = sorted(times)
    p50 = _true_median(times_sorted)
    p95 = times_sorted[_percentile_index(0.95, len(times_sorted))]
    tflops_med = _tflops(args.M, args.N, args.K, p50 / 1000.0)
    tflops_list = [_tflops(args.M, args.N, args.K, t / 1000.0) for t in times]

    out = {
        # repeats/warmup report the counts ACTUALLY used (after any
        # PERFLAB_BENCH_* override) so contract.min_repeats can be enforced
        # instead of silently skipped.
        "meta": {
            "device": "cuda", "num_gpus": len(devices),
            "M": args.M, "N": args.N, "K": args.K,
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
