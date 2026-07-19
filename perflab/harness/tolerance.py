"""Task-declared accuracy tolerance for harness comparison helpers.

The optimizer exports task.yaml ``constraints.accuracy_tolerance`` to
correctness subprocesses as ``PERFLAB_ACCURACY_TOLERANCE`` (see
perflab.runners.correctness). Harness helpers resolve their default
atol/rtol through this module so a task author's declared bound applies
without editing every tests.py call site; explicit atol/rtol arguments
always win.
"""
from __future__ import annotations

import os

_ENV_VAR = "PERFLAB_ACCURACY_TOLERANCE"


def env_accuracy_tolerance(default: float = 1e-5) -> float:
    """Resolve the accuracy tolerance from PERFLAB_ACCURACY_TOLERANCE.

    "exact" maps to 0.0, a numeric string ("1e-3") to its float value, and
    unset or unparsable values to ``default``.
    """
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw:
        return default
    if raw.lower() == "exact":
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return default
