"""Inference throughput benchmark for SmallCNN pipeline.

Measures images_per_sec over repeated runs of the full inference pipeline
(preprocessing, model forward, postprocessing). The agent should discover
batching, GPU preprocessing, torch.compile, half precision, and other
optimizations by editing pipeline.py.
"""

import argparse
import json
import os
import time
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
    num_images = int(knobs.get("num_images", 64))
    batch_size = int(knobs.get("batch_size", 1))

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Optional torch profiler
    do_profile = os.environ.get("PERFLAB_TORCH_PROFILE", "").lower() in ("1", "true")
    trace_path = os.environ.get("PERFLAB_TORCH_TRACE_PATH")
    prof = None
    if do_profile:
        from torch.profiler import ProfilerActivity, profile
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        prof = profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True)

    # Warmup
    warmup_runs = int(os.environ.get("PERFLAB_BENCH_WARMUP", 2))
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
    med = median(images_per_sec_list)

    out = {
        "images_per_sec": {"median": med, "all": images_per_sec_list},
        "meta": {
            "num_images": num_images,
            "batch_size": batch_size,
            "device": device,
            "n_repeats": n_repeats,
        },
        "ok": True,
    }

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"images_per_sec.median = {med:.1f}")


class _nullcontext:
    """Minimal no-op context manager for Python 3.10 compat."""
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
