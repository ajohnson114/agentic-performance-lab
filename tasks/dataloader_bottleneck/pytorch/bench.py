"""Benchmark: measures samples/sec through the DataLoader + model forward pass.

The default tuning.yaml has num_workers=0, pin_memory=false — deliberately slow.
Fix: num_workers=4+, pin_memory=true → throughput jumps dramatically.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from torch.profiler import record_function

from dataset import SyntheticImageDataset
from model import SmallClassifier


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Output path for bench.json")
    args = ap.parse_args()

    knobs = yaml.safe_load(Path("tuning.yaml").read_text(encoding="utf-8"))
    num_workers = int(knobs.get("num_workers", 0))
    pin_memory = bool(knobs.get("pin_memory", False))
    prefetch_factor = int(knobs.get("prefetch_factor", 2))
    batch_size = int(knobs.get("batch_size", 32))

    device = _device()
    model = SmallClassifier(num_classes=10).to(device)
    model.eval()

    dataset = SyntheticImageDataset(size=1024, simulate_io_ms=1.0)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    # prefetch_factor only valid when num_workers > 0
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    # Optional torch profiler integration
    enabled = os.environ.get("PERFLAB_TORCH_PROFILE", "0") == "1"
    trace_path = os.environ.get("PERFLAB_TORCH_TRACE_PATH")
    prof = None
    if enabled:
        from torch.profiler import profile, ProfilerActivity
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        prof = profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True)

    # Warmup
    _sync(device)
    for images, labels in loader:
        images = images.to(device, non_blocking=pin_memory)
        with torch.no_grad():
            _ = model(images)
        break
    _sync(device)

    # Benchmark: iterate through entire dataset
    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 3))
    samples_per_sec_list = []

    ctx = prof if prof is not None else _nullcontext()
    with ctx:
        for rep in range(repeats):
            total_samples = 0
            t0 = time.perf_counter()
            for images, labels in loader:
                with record_function("## data_loading ##"):
                    images = images.to(device, non_blocking=pin_memory)
                with record_function("## forward ##"):
                    with torch.no_grad():
                        _ = model(images)
                total_samples += images.shape[0]
            _sync(device)
            elapsed = time.perf_counter() - t0
            samples_per_sec_list.append(total_samples / elapsed if elapsed > 0 else 0)

    if prof is not None and trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(trace_path)

    samples_per_sec_list.sort()
    median = samples_per_sec_list[len(samples_per_sec_list) // 2]
    p95_idx = min(int(0.95 * (len(samples_per_sec_list) - 1)), len(samples_per_sec_list) - 1)

    out = {
        "meta": {
            "device": str(device),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "prefetch_factor": prefetch_factor,
            "batch_size": batch_size,
        },
        "samples_per_sec": {
            "median": median,
            "values": samples_per_sec_list,
        },
        "ok": True,
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"samples_per_sec_median": median, "device": str(device)}, indent=2))


class _nullcontext:
    """Minimal no-op context manager for Python 3.10 compat."""
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
