"""Inference throughput benchmark for ResNet-50 pipeline.

Measures images_per_sec over repeated runs of the full inference pipeline.
The agent should discover GPU-side tensor creation, batched preprocessing,
torch.compile, half precision, and other optimizations by editing pipeline.py.
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from statistics import median

import torch
import yaml
from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Output JSON path")
    args = parser.parse_args()

    knobs = yaml.safe_load(Path("tuning.yaml").read_text())
    num_images = int(knobs.get("num_images", 1024))
    batch_size = int(knobs.get("batch_size", 32))

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        warnings.warn(
            "CUDA not available — running on CPU. "
            "Results will not reflect H100 performance.",
            stacklevel=2,
        )

    # Optional torch profiler
    do_profile = os.environ.get("PERFLAB_TORCH_PROFILE", "").lower() in (
        "1",
        "true",
    )
    trace_path = os.environ.get("PERFLAB_TORCH_TRACE_PATH")
    prof = None
    if do_profile:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        prof = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    # Warmup
    warmup_runs = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
    for _ in range(warmup_runs):
        run_pipeline(num_images, batch_size, device)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    n_repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 10))
    times_s = []

    ctx = prof if prof is not None else _nullcontext()
    with ctx:
        for _ in range(n_repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_pipeline(num_images, batch_size, device)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_s.append(t1 - t0)

    if prof is not None and trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(trace_path)

    images_per_sec_list = [num_images / t for t in times_s]
    latency_ms_list = [t * 1000 for t in times_s]
    med_ips = median(images_per_sec_list)
    med_lat = median(latency_ms_list)

    # ResNet-50 FLOPs: ~4.1 GFLOPs per image (forward pass, 224x224 input)
    # 2x for multiply-accumulate = ~8.2 GFLOPs per image
    # At FP16: H100 can do 1979 TFLOPS vs 989 TFLOPS at FP32
    flops_per_image = 8.2e9
    tflops_list = [(ips * flops_per_image) / 1e12 for ips in images_per_sec_list]
    med_tflops = median(tflops_list)

    out = {
        "images_per_sec": {"median": med_ips, "all": images_per_sec_list},
        "tflops": {"median": med_tflops, "all": tflops_list},
        "latency_ms": {"median": med_lat, "all": latency_ms_list},
        "meta": {
            "num_images": num_images,
            "batch_size": batch_size,
            "device": device,
            "n_repeats": n_repeats,
            "warmup_runs": warmup_runs,
            "flops_per_image": flops_per_image,
            "flops": num_images * flops_per_image,
            "bytes_moved": 25.5e6 * 4 + num_images * (3 * 224 * 224 * 4 + 1000 * 4),
        },
        "ok": True,
    }

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"images_per_sec.median = {med_ips:.1f}  ({med_tflops:.2f} TFLOPS)")


class _nullcontext:
    """Minimal no-op context manager for Python 3.10 compat."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
