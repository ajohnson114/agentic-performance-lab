"""Benchmark statistical analysis — noise detection and confidence intervals.

Analyzes repeated benchmark measurements to detect noisy results and warn
the agent when measurement variance is too high for reliable speedup claims.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


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
    cv_threshold: float = 0.10,
) -> BenchStats | None:
    """Compute statistical summary from repeated benchmark measurements.

    Args:
        values: List of benchmark metric values from repeated runs.
        cv_threshold: Coefficient of variation threshold above which results
            are considered noisy (default 10%).

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
    t_val = 2.0 if n < 30 else 1.96
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


def extract_repeated_values(bench: dict, metric_name: str) -> list[float]:
    """Extract individual repeated measurement values from bench.json.

    Looks for a `raw_values` or `samples` list alongside the metric.
    Falls back to an empty list if only the aggregate is available.

    For a metric like "throughput.median", looks for:
      - bench["throughput"]["raw_values"]
      - bench["throughput"]["samples"]
    """
    parts = metric_name.split(".")
    if len(parts) < 2:
        return []

    # Navigate to the parent dict (e.g., bench["throughput"])
    cur = bench
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return []
        cur = cur[part]

    if not isinstance(cur, dict):
        return []

    # Look for raw values
    for key in ("raw_values", "samples", "values"):
        raw = cur.get(key)
        if isinstance(raw, list) and len(raw) >= 2:
            try:
                return [float(v) for v in raw]
            except (ValueError, TypeError):
                continue

    return []


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
