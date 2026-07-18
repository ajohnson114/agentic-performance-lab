from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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


def is_statistically_significant(
    baseline_times: list[float],
    candidate_times: list[float],
    alpha: float = 0.05,
) -> tuple[bool, float]:
    """Test whether candidate times differ significantly from baseline.

    Uses Mann-Whitney U test. Returns (is_significant, p_value).
    Falls back to (True, 0.0) if scipy is not installed.
    """
    if len(baseline_times) < 2 or len(candidate_times) < 2:
        return True, 0.0
    try:
        from scipy.stats import mannwhitneyu
        stat, p_value = mannwhitneyu(
            baseline_times, candidate_times, alternative="two-sided"
        )
        return p_value < alpha, p_value
    except ImportError:
        return True, 0.0


def is_improvement(new: float, best: float, mode: Literal["maximize","minimize"], tol: float) -> bool:
    if mode == "maximize":
        return new > best * (1.0 + tol)
    else:
        return new < best * (1.0 - tol)


def calc_speedup(value: float, baseline: float) -> float:
    """Compute speedup ratio (value / baseline), returning 1.0 if baseline is zero."""
    return value / baseline if baseline != 0 else 1.0


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
