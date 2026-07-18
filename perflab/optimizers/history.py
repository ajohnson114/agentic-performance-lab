"""Shared constructor for optimizer history entries.

History entries are consumed downstream (reporting, run summaries,
dashboards) as plain dicts, so this helper produces exactly the same
shape the hand-rolled call sites used to build.
"""
from __future__ import annotations

from perflab.analyzers.metrics_rollup import calc_speedup


def make_history_entry(
    iteration: int,
    description: str,
    value: float,
    baseline: float,
    *,
    accepted: bool,
    **extra,
) -> dict:
    """Build one history-entry dict for the optimization loop.

    ``delta`` and ``speedup`` are computed against ``baseline``
    (``calc_speedup`` returns 1.0 when baseline is zero). Extra keyword
    arguments are appended in order; entries whose value is None are
    dropped, so callers can pass conditional fields unconditionally.
    """
    entry = {
        "iteration": iteration,
        "description": description,
        "value": value,
        "accepted": accepted,
        "delta": value - baseline,
        "speedup": calc_speedup(value, baseline),
    }
    for key, val in extra.items():
        if val is not None:
            entry[key] = val
    return entry
