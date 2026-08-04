"""Task-declared accuracy tolerance for harness comparison helpers.

The optimizer exports task.yaml ``constraints.accuracy_tolerance`` to
correctness subprocesses as ``PERFLAB_ACCURACY_TOLERANCE`` (see
perflab.runners.correctness). Harness helpers resolve their default
atol/rtol through this module so a task author's declared bound applies
without editing every tests.py call site; explicit atol/rtol arguments
always win.
"""
from __future__ import annotations

import math
import os
import warnings

_ENV_VAR = "PERFLAB_ACCURACY_TOLERANCE"


def env_accuracy_tolerance(default: float = 1e-5) -> float:
    """Resolve the accuracy tolerance from PERFLAB_ACCURACY_TOLERANCE.

    "exact" maps to 0.0, a numeric string ("1e-3") to its float value, and
    unset or unparsable values to ``default``.

    A parsed value is only honored if it is finite and non-negative. That
    guard is the whole point of routing this through one function: this
    tolerance becomes the atol/rtol of every accuracy comparison that does not
    pass an explicit one, so a value of ``inf`` makes ``allclose`` return True
    for *any* pair of arrays -- accuracy checking silently switches off across
    the harness. ``float()`` reaches infinity from more spellings than it
    looks: "inf", "Infinity", and any overflowing literal such as "1e400".
    "nan" is equally poisonous (comparisons against NaN are meaningless) and a
    negative tolerance is not a tolerance at all. All of them fall back to
    ``default`` -- the strict direction -- with a warning, matching how an
    unparsable value is already handled.
    """
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw:
        return default
    if raw.lower() == "exact":
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value) or value < 0:
        warnings.warn(
            f"{_ENV_VAR}={raw!r} parses to {value!r}, which is not a usable "
            f"tolerance (it must be finite and non-negative; {value!r} would "
            f"disable or invalidate every accuracy comparison). Falling back to "
            f"{default!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default
    return value
