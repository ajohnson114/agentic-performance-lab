"""Shared constructor for optimizer history entries.

History entries are consumed downstream (reporting, run summaries,
dashboards) as plain dicts, so this helper produces exactly the same
shape the hand-rolled call sites used to build.
"""
from __future__ import annotations

from typing import Literal

from perflab.analyzers.metrics_rollup import calc_speedup


def make_history_entry(
    iteration: int,
    description: str,
    value: float,
    baseline: float,
    *,
    accepted: bool,
    mode: Literal["maximize", "minimize"] = "maximize",
    **extra,
) -> dict:
    """Build one history-entry dict for the optimization loop.

    ``delta`` and ``speedup`` are computed against ``baseline``. ``speedup``
    is mode-aware so >1.0 always means "better": value/baseline when
    maximizing, baseline/value when minimizing (1.0 when the denominator is
    zero, matching compute_run_summary). Extra keyword arguments are appended
    in order; entries whose value is None are dropped, so callers can pass
    conditional fields unconditionally.
    """
    if mode == "minimize":
        speedup = baseline / value if value != 0 else 1.0
    else:
        speedup = calc_speedup(value, baseline)
    entry = {
        "iteration": iteration,
        "description": description,
        "value": value,
        "accepted": accepted,
        "delta": value - baseline,
        "speedup": speedup,
    }
    for key, val in extra.items():
        if val is not None:
            entry[key] = val
    return entry
