"""Training throughput benchmark for small PyTorch transformer.

Runs in fp32 with naive attention and no torch.compile. An optimizing agent
should discover and apply AMP, SDPA, and torch.compile via code edits.

Anti-gaming wiring (see perflab/harness/):
  * SyncTimer     — full device synchronization on BOTH sides of every timed
                    step, so async GPU work cannot be pushed outside the
                    timing window and reported as speedup. Works on CUDA, MPS
                    and CPU (SyncTimer dispatches on the device type and is a
                    no-op sync for CPU). Note this also fixes a real gap in
                    the previous version of this file, which synchronized only
                    when device == "cuda" and therefore mis-timed MPS runs.
  * cuda_sync_guard — drains warmup work before the thread snapshot and the
                    first timed step.
  * ThreadGuard   — snapshots the thread count AFTER warmup and reports the
                    delta across the timed region as bench.thread_delta, so
                    the framework can reject candidates that offload work to
                    background threads while returning "instantly".
"""

import argparse
import json
import os
from pathlib import Path
from statistics import median

import torch
import yaml
from model import SmallTransformer
from torch.profiler import record_function

from perflab.harness import SyncTimer, ThreadGuard, cuda_sync_guard


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

    # Warmup. cuda_sync_guard drains the device on both sides so that any
    # lazily-created thread pools and JIT/inductor compilation triggered by
    # the first steps are finished before the thread snapshot below.
    warmup_steps = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
    with cuda_sync_guard(device):
        for _ in range(warmup_steps):
            inputs, targets = make_batch()
            train_step(inputs, targets)

    # Timed steps
    n_steps = int(os.environ.get("PERFLAB_BENCH_REPEATS", 10))
    tokens_per_step = batch_size * (seq_len - 1)
    times_s = []

    timer = SyncTimer(device)
    # Snapshot AFTER warmup on purpose: torch.compile / inductor and cuDNN
    # start worker pools on first use, and those are legitimate. What must not
    # happen is new threads appearing during the steady-state timed region.
    guard = ThreadGuard()
    guard.snapshot()

    ctx = prof if prof is not None else _nullcontext()
    with ctx:
        for _ in range(n_steps):
            inputs, targets = make_batch()
            timer.start()
            train_step(inputs, targets)
            times_s.append(timer.stop())

    thread_delta = guard.thread_delta

    if prof is not None and trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(trace_path)

    tokens_per_sec_list = [tokens_per_step / t for t in times_s]
    med = median(tokens_per_sec_list)

    out = {
        "tokens_per_sec": {
            "median": med,
            "all": tokens_per_sec_list,
            "raw_values": tokens_per_sec_list,
        },
        # Read by the framework's thread_count_check (anti_gaming in task.yaml).
        "thread_delta": thread_delta,
        "meta": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "n_steps": n_steps,
            # repeats/warmup report the counts ACTUALLY used (after any
            # PERFLAB_BENCH_* override) so that contract.min_repeats /
            # min_warmup are enforced instead of silently skipped.
            "repeats": n_steps,
            "warmup": warmup_steps,
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
