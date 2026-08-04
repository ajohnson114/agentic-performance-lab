"""Benchmark statistical analysis — noise detection and confidence intervals.

Analyzes repeated benchmark measurements to detect noisy results and warn
the agent when measurement variance is too high for reliable speedup claims.

How the two noise knobs relate (they used to be unrelated magic numbers)
-----------------------------------------------------------------------
``constraints.regression_tolerance`` (default 2%) is a *materiality* floor:
"a win smaller than this is not worth keeping". ``cv_threshold`` (default 10%)
is a *dispersion alarm*: "this measurement is too jittery to trust". Comparing
them directly is meaningless because a CV says nothing on its own — what a
benchmark can actually *resolve* is the CV shrunk by the sample count:

    relative 95% CI half-width   m = t * CV / sqrt(n)

so the smallest relative improvement that can be told apart from noise, when
both the candidate and the incumbent are measured this way, is

    resolvable ≈ (1 + m) / (1 - m) - 1  ≈  2 * t * CV / sqrt(n)

`cv_budget_for_gate` inverts that: it answers "how quiet must this machine be
for my regression_tolerance gate to mean anything at n repeats?". With the
defaults (tol=2%, repeats=20, t=2) the answer is CV <= 2.2% — i.e. the old
10% cv_threshold was ~5x looser than the gate it was supposed to protect,
which is exactly how noise got accepted as a win. The accept gate therefore
does not compare against a fixed CV at all (see
`perflab.analyzers.metrics_rollup.assess_improvement`); it uses the n-aware
interval directly, and `cv_budget_for_gate` / `repeats_needed_for_gate` exist
to tell the user *why* their gate is unreachable and what to change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Coarse "this measurement is obviously jittery" alarm used by reporting.
# NOT the accept gate — see the module docstring and cv_budget_for_gate.
DEFAULT_CV_THRESHOLD = 0.10

# t-multiplier for a 95% interval: 1.96 asymptotically, 2.0 as a deliberately
# conservative stand-in for the small-n t-distribution (benchmarks routinely
# run n=3..20, where the true t is 4.30..2.09).
_T_SMALL_SAMPLE = 2.0
_T_LARGE_SAMPLE = 1.96
_LARGE_SAMPLE_N = 30

# Top-level bench.json keys holding a per-repeat measurement array. Used only
# as a fallback by extract_repeated_values (see its docstring).
_TOP_LEVEL_SAMPLE_KEYS = ("times_ms", "times", "raw_values", "samples")

# Keys checked inside the metric's own parent dict (the primary path).
_METRIC_SAMPLE_KEYS = ("raw_values", "samples", "values", "all")


def t_value(n: int) -> float:
    """95% t-multiplier for a sample of size ``n`` (conservative for small n)."""
    return _T_LARGE_SAMPLE if n >= _LARGE_SAMPLE_N else _T_SMALL_SAMPLE


@dataclass
class BenchStats:
    """Statistical summary of benchmark measurements."""
    n: int
    mean: float
    median: float
    std: float
    cv: float  # coefficient of variation (std / mean)
    ci_95_low: float  # 95% confidence interval lower bound
    ci_95_high: float  # 95% confidence interval upper bound
    is_noisy: bool  # True if CV exceeds threshold
    warning: str = ""


def compute_bench_stats(
    values: list[float],
    cv_threshold: float = DEFAULT_CV_THRESHOLD,
) -> BenchStats | None:
    """Compute statistical summary from repeated benchmark measurements.

    Args:
        values: List of benchmark metric values from repeated runs.
        cv_threshold: Coefficient of variation threshold above which results
            are flagged noisy (default 10%). This is a reporting alarm, not
            the accept gate; for the gate-coherent value see
            `cv_budget_for_gate`.

    Returns:
        BenchStats or None if fewer than 2 values.
    """
    if len(values) < 2:
        return None

    n = len(values)
    mean = sum(values) / n
    if mean == 0:
        return None

    sorted_vals = sorted(values)
    if n % 2 == 0:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    else:
        median = sorted_vals[n // 2]

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    cv = std / abs(mean)

    # 95% CI using t-distribution approximation (t ≈ 1.96 for large n,
    # use 2.0 as conservative estimate for small samples)
    t_val = t_value(n)
    margin = t_val * std / math.sqrt(n)
    ci_95_low = mean - margin
    ci_95_high = mean + margin

    is_noisy = cv > cv_threshold
    warning = ""
    if is_noisy:
        warning = (
            f"High measurement variance detected: CV={cv:.1%} "
            f"(threshold={cv_threshold:.0%}). "
            f"Speedup claims may be unreliable. "
            f"Consider increasing repeats or reducing background load."
        )

    return BenchStats(
        n=n,
        mean=mean,
        median=median,
        std=std,
        cv=cv,
        ci_95_low=ci_95_low,
        ci_95_high=ci_95_high,
        is_noisy=is_noisy,
        warning=warning,
    )


def relative_ci_margin(stats: BenchStats) -> float:
    """95% CI half-width as a fraction of the mean: ``t * CV / sqrt(n)``.

    Working in *relative* terms is what makes a sample array usable even when
    it is not in the metric's units — see `extract_repeated_values`.
    """
    if stats.mean == 0:
        return math.inf
    return (stats.ci_95_high - stats.mean) / abs(stats.mean)


def cv_budget_for_gate(
    regression_tolerance: float, repeats: int, t: float | None = None,
) -> float:
    """Max CV at which a ``regression_tolerance`` gate is actually resolvable.

    Inverts ``resolvable ≈ (1 + m)/(1 - m) - 1`` with ``m = t*CV/sqrt(n)``,
    assuming the candidate and the incumbent are measured with the same CV and
    the same number of repeats (the usual case — same harness, same machine):

        CV_budget = sqrt(n) * tol / (t * (2 + tol))

    Defaults (tol=0.02, repeats=20) give 2.2%. A machine noisier than this
    cannot tell a 2% win from noise no matter how confidently the ratio is
    printed.
    """
    n = max(int(repeats), 1)
    tol = max(regression_tolerance, 0.0)
    t_mult = t if t is not None else t_value(n)
    if tol <= 0 or t_mult <= 0:
        return math.inf
    return math.sqrt(n) * tol / (t_mult * (2.0 + tol))


def repeats_needed_for_gate(
    cv: float, regression_tolerance: float, t: float | None = None,
) -> int:
    """Repeats needed for a ``regression_tolerance`` gate to resolve at this CV.

    The inverse of `cv_budget_for_gate`. Answers the only actionable question
    a noisy environment raises: "how many samples would make my gate mean
    something?" (8% CV against a 2% gate needs ~261 repeats — which is the
    honest signal that the machine, not the kernel, is the bottleneck).
    """
    tol = max(regression_tolerance, 0.0)
    if tol <= 0 or cv <= 0:
        return 1
    t_mult = t if t is not None else _T_SMALL_SAMPLE
    return max(1, math.ceil((t_mult * cv * (2.0 + tol) / tol) ** 2))


def _samples_from(container: dict, keys: tuple[str, ...]) -> list[float]:
    """Return the first key in ``keys`` holding a numeric list of length >= 2."""
    for key in keys:
        raw = container.get(key)
        if isinstance(raw, list) and len(raw) >= 2:
            try:
                vals = [float(v) for v in raw]
            except (ValueError, TypeError):
                continue
            if all(math.isfinite(v) for v in vals):
                return vals
    return []


def extract_repeated_values(bench: dict, metric_name: str) -> list[float]:
    """Extract individual repeated measurement values from bench.json.

    Primary path (preferred, and what every first-party task emits): a
    per-repeat array alongside the metric itself. For a metric like
    "throughput.median" that means one of
      - bench["throughput"]["raw_values"]
      - bench["throughput"]["samples"]
      - bench["throughput"]["values"]
      - bench["throughput"]["all"]      (the shape the task templates emit)

    Fallback: the top-level per-repeat timing array (`times_ms`, then `times`,
    `raw_values`, `samples`) that most harnesses already write. A task whose
    bench.json has not been updated to publish per-metric samples still gets
    variance this way.

    The fallback array is frequently NOT in the metric's units — `times_ms`
    next to a `tflops.median` metric, for example. That is deliberate and
    safe, because every consumer of these samples uses them for *relative*
    dispersion only (`relative_ci_margin`, i.e. t*CV/sqrt(n)): CV is
    unit-free, and to first order it is preserved under the reciprocal
    transform that relates a rate to a time (CV[c/t] ≈ CV[t] for small CV).
    Do not use the returned values as measurements of the metric itself.

    Returns [] only when no usable array exists anywhere.
    """
    if not isinstance(bench, dict):
        return []

    parts = metric_name.split(".")
    if len(parts) >= 2:
        cur: object = bench
        for part in parts[:-1]:
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if isinstance(cur, dict):
            vals = _samples_from(cur, _METRIC_SAMPLE_KEYS)
            if vals:
                return vals

    return _samples_from(bench, _TOP_LEVEL_SAMPLE_KEYS)


def format_bench_stats_for_prompt(stats: BenchStats) -> str:
    """Format bench stats as a concise string for the LLM prompt."""
    parts = [
        f"Benchmark stats (n={stats.n}): "
        f"mean={stats.mean:.4g}, median={stats.median:.4g}, "
        f"std={stats.std:.4g}, CV={stats.cv:.1%}"
    ]
    if stats.is_noisy:
        parts.append(f"WARNING: {stats.warning}")
    else:
        parts.append(f"95% CI: [{stats.ci_95_low:.4g}, {stats.ci_95_high:.4g}]")
    return "\n".join(parts)
