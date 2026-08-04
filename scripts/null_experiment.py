#!/usr/bin/env python3
"""Measure how much of a "win" your machine invents out of nothing.

Runs a **null experiment**: both arms execute byte-identical code, so the true
effect is exactly zero. Anything the comparison reports is measurement
artifact. The same measurements are then read two ways -- sequentially (all of
arm A, then all of arm B, which is how PerfLab compared a candidate against the
incumbent by default) and ABBA-interleaved -- so the two designs are scored on
the *same* data and the difference is the design, not luck.

Why this script exists
----------------------
The rationale doc used to quote a "+42% apparent win" and a "1.1-1.8x variance
reduction" from a one-off run on a dev machine. Those numbers were real when
observed and completely unsourced afterwards -- the same unfalsifiable decimals
an external reviewer had just (correctly) objected to elsewhere in that
document. A number worth citing is a number someone else can reproduce, and
these are hardware-specific: the whole point is that the answer differs wildly
between a quiet laptop and a thermally-throttled GPU box.

    python scripts/null_experiment.py tasks/matmul/python/task.yaml
    python scripts/null_experiment.py <task.yaml> --pairs 8 --repeats 3

Interpreting it: a large sequential bias means paired measurement is worth its
2.6-10x cost on this machine. A small one means your host is quiet enough that
the unpaired gate is fine, and interleaving would only buy you a longer wait.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perflab.runners.benchmark import metric_value, run_benchmark  # noqa: E402
from perflab.runners.paired import interleave_order  # noqa: E402
from perflab.task_spec import TaskSpec  # noqa: E402


def measure(task: TaskSpec, repeats: int) -> float:
    """One spawn of the task's real benchmark; returns the primary metric."""
    _, bench = run_benchmark(
        task.benchmark.cmd,
        cwd=task.workspace,
        program_type=task.program_type,
        rlimit_as_gb=task.constraints.rlimit_as_gb,
        env_passthrough=task.constraints.env_passthrough,
        warmup=task.benchmark.warmup,
        repeats=repeats,
    )
    return metric_value(bench, task.benchmark.metric.name)


def relative(candidate: float, incumbent: float, mode: str) -> float:
    """Apparent improvement as a fraction; positive always means 'better'."""
    if incumbent == 0:
        return 0.0
    ratio = candidate / incumbent - 1.0
    return ratio if mode == "maximize" else -ratio


def cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_yaml")
    ap.add_argument("--pairs", type=int, default=6, help="A/B pairs (default 6)")
    ap.add_argument("--repeats", type=int, default=None,
                    help="repeats per spawn (default: the task's own)")
    args = ap.parse_args()

    task = TaskSpec.load(Path(args.task_yaml))
    repeats = args.repeats or task.benchmark.repeats
    mode = task.benchmark.metric.mode

    order = interleave_order(args.pairs, counterbalanced=True)
    n = len(order)
    print(f"task    : {task.name}  ({task.benchmark.metric.name}, {mode})")
    print(f"design  : {''.join(order)}   ({args.pairs} pairs, {repeats} repeats/spawn)")
    print("Both arms run the SAME code. Any reported win is measurement artifact.\n")

    samples: list[tuple[str, float]] = []
    for i, arm in enumerate(order, 1):
        value = measure(task, repeats)
        samples.append((arm, value))
        print(f"  spawn {i:>2}/{n}  arm {arm}  {value:.6g}")

    a_vals = [v for arm, v in samples if arm == "A"]
    b_vals = [v for arm, v in samples if arm == "B"]

    # Sequential reading: the arms as PerfLab used to compare them -- every A
    # measured before every B. Same numbers, re-ordered in time.
    seq_a = [v for _, v in samples[: n // 2]]
    seq_b = [v for _, v in samples[n // 2:]]
    seq_bias = relative(statistics.median(seq_b), statistics.median(seq_a), mode)

    # Paired reading: differences within each adjacent A/B pair.
    # Normalize by ARM, not by position. Under ABBA the pairs alternate
    # (A,B),(B,A),... so pairing positionally would flip the sign on half of
    # them and quietly cancel the very bias this is measuring.
    diffs = []
    for (arm_x, x), (arm_y, y) in zip(samples[::2], samples[1::2], strict=False):
        if arm_x == arm_y:
            continue
        a_val, b_val = (x, y) if arm_x == "A" else (y, x)
        diffs.append(relative(b_val, a_val, mode))
    paired_bias = statistics.fmean(diffs) if diffs else 0.0

    print("\n--- results (true effect is 0.0%) ---")
    print(f"  sequential  apparent win : {seq_bias * 100:+7.2f}%")
    print(f"  ABBA paired apparent win : {paired_bias * 100:+7.2f}%")
    print(f"  per-arm CV               : A {cv(a_vals) * 100:5.2f}%   B {cv(b_vals) * 100:5.2f}%")
    # NOT a CV: paired differences centre on zero, so dividing by their mean
    # is meaningless (and sign-flips). The spread itself is the quantity that
    # matters -- it is what a paired test must overcome.
    if len(diffs) > 1:
        print(f"  spread of paired diffs   : {statistics.stdev(diffs) * 100:5.2f}%"
              "  (sd, not CV: these centre on zero)")

    worst = max((abs(d) for d in diffs), default=0.0)
    print(f"  worst single pair        : {worst * 100:7.2f}%")
    print(
        "\nA large sequential bias means paired measurement earns its 2.6-10x cost\n"
        "on this machine. A small one means the host is quiet enough that the\n"
        "unpaired gate is adequate and interleaving would only cost you time."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
