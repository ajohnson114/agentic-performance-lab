from __future__ import annotations

import argparse
import json
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

def maybe_torch_profiler_enabled() -> tuple[bool, str | None]:
    if os.environ.get("PERFLAB_TORCH_PROFILE", "0") != "1":
        return False, None
    return True, os.environ.get("PERFLAB_TORCH_TRACE_PATH")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--K", type=int, default=2048)
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
    p50 = times_sorted[len(times_sorted)//2]
    p95 = times_sorted[int(0.95 * (len(times_sorted)-1))]
    med_ms = p50
    tflops_med = _tflops(args.M, args.N, args.K, batch, med_ms/1000.0)

    out = {
        "meta": {"device": dev.type, "dtype": dtype, "batch": batch, "M": args.M, "N": args.N, "K": args.K},
        "times_ms": times,
        "latency_ms": {"p50": p50, "p95": p95},
        "tflops": {"median": tflops_med},
        "ok": True,
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"tflops_median": tflops_med, "lat_ms_p50": p50, "device": dev.type}, indent=2))

if __name__ == "__main__":
    main()
