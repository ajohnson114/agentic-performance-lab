"""Interval-union helper for wall-clock busy-time metrics.

Summing per-event durations double-counts wall-clock time whenever events
overlap — e.g. kernels running concurrently on multiple CUDA streams, or
nested operator spans. Metrics that claim a fraction of wall time
("GPU active %", host-vs-device split) must instead measure the length of
the union of busy intervals.
"""
from __future__ import annotations

from collections.abc import Iterable


def union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    """Return the total length covered by the union of ``(start, end)`` intervals.

    Overlapping intervals are merged so concurrent/nested events are counted
    once. Empty or inverted intervals (``end <= start``) are ignored.
    Works with int (ns) or float (us) endpoints.
    """
    total = 0.0
    prev_end: float | None = None
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if prev_end is None or start >= prev_end:
            total += end - start
            prev_end = end
        elif end > prev_end:
            total += end - prev_end
            prev_end = end
    return total
