from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from perflab.analyzers.decision import (
    NON_OVERLAPPING_CI,
    TOLERANCE_ONLY,
    Comparison,
    ImprovementVerdict,
)

# `ImprovementVerdict` and the accept rule itself now live in
# `perflab.analyzers.decision`, which is the single module every accept/reject
# call site depends on. Re-exported here because this module's names are public
# and used across the codebase; see decision.py for the rule and its rationale.
__all__ = [
    "Decision",
    "ImprovementVerdict",
    "RunSummary",
    "assess_improvement",
    "calc_speedup",
    "compute_run_summary",
    "improvement_factor",
    "is_improvement",
]


@dataclass
class Decision:
    accepted: bool
    notes: str


@dataclass
class RunSummary:
    baseline_value: float
    best_value: float
    median_speedup: float        # median of speedup for accepted iters
    p90_speedup: float           # 90th percentile speedup
    time_to_first_improvement: int | None  # iteration of first accepted improvement
    success_rate: float          # fraction of non-baseline iters with accepted improvement
    total_iterations: int


def assess_improvement(
    new: float,
    best: float,
    mode: Literal["maximize", "minimize"],
    tol: float,
    *,
    new_samples: list[float] | None = None,
    best_samples: list[float] | None = None,
    noise_gate: bool = True,
) -> ImprovementVerdict:
    """Decide whether ``new`` is a real improvement over ``best``.

    Compatibility wrapper: the rule lives in
    :mod:`perflab.analyzers.decision`, which every accept/reject call site now
    depends on directly. This signature is kept because it is public and
    already used elsewhere; new code should build a
    :class:`~perflab.analyzers.decision.Comparison` and pick a rule explicitly.

    ``noise_gate=True`` selects
    :class:`~perflab.analyzers.decision.NonOverlappingCI` (materiality floor
    AND non-overlapping 95% CIs); ``noise_gate=False`` selects
    :class:`~perflab.analyzers.decision.ToleranceOnly` (the bare ratio). See
    those classes for the rule and its rationale.
    """
    rule = NON_OVERLAPPING_CI if noise_gate else TOLERANCE_ONLY
    return rule.decide(Comparison(
        candidate=new,
        incumbent=best,
        mode=mode,
        tolerance=tol,
        candidate_samples=new_samples,
        incumbent_samples=best_samples,
    ))


def is_improvement(
    new: float,
    best: float,
    mode: Literal["maximize", "minimize"],
    tol: float,
    *,
    new_samples: list[float] | None = None,
    best_samples: list[float] | None = None,
    noise_gate: bool = True,
) -> bool:
    """Boolean form of `assess_improvement` (see it for the rule).

    Backward compatible: called with the original four positional arguments and
    no samples it is exactly the old ratio test,
    ``new > best * (1 + tol)`` / ``new < best * (1 - tol)``.
    Callers that need to explain a rejection should use `assess_improvement`
    directly — the boolean cannot distinguish "did not beat the incumbent"
    from "beat it by less than the machine can measure".
    """
    return assess_improvement(
        new, best, mode, tol,
        new_samples=new_samples, best_samples=best_samples, noise_gate=noise_gate,
    ).improved


def calc_speedup(value: float, baseline: float) -> float:
    """Compute speedup ratio (value / baseline), returning 1.0 if baseline is zero."""
    return value / baseline if baseline != 0 else 1.0


def improvement_factor(
    new: float, old: float, mode: Literal["maximize", "minimize"],
) -> float:
    """How many times better ``new`` is than ``old`` under the metric mode.

    Mode-aware, unlike calc_speedup: >1.0 always means "better" (a latency
    drop from 10ms to 1ms is 10.0, as is a throughput rise from 1 to 10).
    Returns 1.0 when either value is zero, since no meaningful ratio exists.

    The neutral 1.0 for a zero value is correct for history/reporting, but
    callers policing benchmark gaming must special-case zero themselves: a
    candidate reporting exactly 0.0 (a stubbed/no-op kernel) is the most
    extreme gaming case, yet it looks neutral here.
    """
    if new == 0 or old == 0:
        return 1.0
    return new / old if mode == "maximize" else old / new


def compute_run_summary(
    history: list[dict],
    baseline_value: float,
    mode: str,
) -> RunSummary:
    """Compute aggregate summary metrics for a completed optimization run.

    Args:
        history: list of dicts with keys: iteration, value, accepted
        baseline_value: the metric value at iteration 0
        mode: "maximize" or "minimize"

    Returns:
        RunSummary with aggregate statistics.
    """
    if not history:
        return RunSummary(
            baseline_value=baseline_value,
            best_value=baseline_value,
            median_speedup=1.0,
            p90_speedup=1.0,
            time_to_first_improvement=None,
            success_rate=0.0,
            total_iterations=0,
        )

    # Compute speedups for all entries
    speedups: list[float] = []
    for entry in history:
        val = entry.get("value", baseline_value)
        if baseline_value != 0:
            if mode == "maximize":
                speedups.append(val / baseline_value)
            else:
                speedups.append(baseline_value / val if val != 0 else 1.0)
        else:
            speedups.append(1.0)

    # Best value
    values = [e.get("value", baseline_value) for e in history]
    if mode == "maximize":
        best_value = max(values)
    else:
        best_value = min(values)

    # Median and p90 of speedups for accepted iterations (including baseline)
    # zip() strict= needs Python 3.10+ (this codebase still runs on 3.9); speedups and history
    # are always the same length (one appended per history entry above), so a plain zip is safe.
    accepted_speedups = [
        s for s, e in zip(speedups, history) if e.get("accepted", False)  # noqa: B905
    ]
    if not accepted_speedups:
        accepted_speedups = [1.0]

    accepted_speedups_sorted = sorted(accepted_speedups)
    median_speedup = accepted_speedups_sorted[len(accepted_speedups_sorted) // 2]
    p90_idx = min(int(0.9 * (len(accepted_speedups_sorted) - 1)), len(accepted_speedups_sorted) - 1)
    p90_speedup = accepted_speedups_sorted[p90_idx]

    # Time to first improvement: first iteration > 0 where accepted=True
    time_to_first: int | None = None
    for entry in history:
        it = entry.get("iteration", 0)
        if it > 0 and entry.get("accepted", False):
            time_to_first = it
            break

    # Success rate: fraction of non-baseline iterations with accepted improvement
    non_baseline = [e for e in history if e.get("iteration", 0) > 0]
    if non_baseline:
        success_rate = sum(1 for e in non_baseline if e.get("accepted", False)) / len(non_baseline)
    else:
        success_rate = 0.0

    total_iterations = max((e.get("iteration", 0) for e in history), default=0)

    return RunSummary(
        baseline_value=baseline_value,
        best_value=best_value,
        median_speedup=median_speedup,
        p90_speedup=p90_speedup,
        time_to_first_improvement=time_to_first,
        success_rate=success_rate,
        total_iterations=total_iterations,
    )
