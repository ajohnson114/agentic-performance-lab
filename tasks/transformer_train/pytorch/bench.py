"""Training throughput benchmark for small PyTorch transformer.

Runs in fp32 with naive attention and no torch.compile. An optimizing agent
should discover and apply AMP, SDPA, and torch.compile via code edits.
"""

import argparse
import json
import os
import time
from pathlib import Path
from statistics import median

import torch
import yaml
from model import SmallTransformer
from torch.profiler import record_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Output JSON path")
    args = parser.parse_args()

    knobs = yaml.safe_load(Path("tuning.yaml").read_text())
    batch_size = int(knobs.get("batch_size", 8))
    seq_len = int(knobs.get("seq_len", 128))
    lr = float(knobs.get("lr", 1e-3))

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    vocab_size = 1024
    model = SmallTransformer(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Synthetic data (no external dataset needed)
    def make_batch():
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        return tokens[:, :-1], tokens[:, 1:]

    def train_step(inputs, targets):
        with record_function("## optimizer_zero_grad ##"):
            optimizer.zero_grad()
        with record_function("## forward ##"):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size), targets.reshape(-1)
            )
        with record_function("## backward ##"):
            loss.backward()
        with record_function("## optimizer ##"):
            optimizer.step()
        return loss.item()

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
    warmup_steps = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
    for _ in range(warmup_steps):
        inputs, targets = make_batch()
        train_step(inputs, targets)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed steps
    n_steps = int(os.environ.get("PERFLAB_BENCH_REPEATS", 10))
    tokens_per_step = batch_size * (seq_len - 1)
    times_s = []

    ctx = prof if prof is not None else _nullcontext()
    with ctx:
        for _ in range(n_steps):
            inputs, targets = make_batch()
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            train_step(inputs, targets)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_s.append(t1 - t0)

    if prof is not None and trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(trace_path)

    tokens_per_sec_list = [tokens_per_step / t for t in times_s]
    med = median(tokens_per_sec_list)

    out = {
        "tokens_per_sec": {"median": med, "all": tokens_per_sec_list},
        "meta": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "n_steps": n_steps,
        },
        "ok": True,
    }

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"tokens_per_sec.median = {med:.1f}")


class _nullcontext:
    """Minimal no-op context manager for Python 3.10 compat."""
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
