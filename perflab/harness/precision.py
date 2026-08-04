"""Mitigation #5: ULP Precision Guard (Precision Downgrade Detection).

Prevents LLM-generated code from computing in lower precision (e.g., fp16)
then casting to fp32 to achieve speed gains while degrading accuracy near
tolerance thresholds.

The fix: compare kernel output against an fp64 reference using ULP (units in
last place) distance, which is more sensitive than simple allclose checks.
Also checks that the output dtype matches the expected dtype.

Backend-neutral: actual/reference may be torch tensors, numpy arrays, JAX
arrays or nested Python lists, in any combination. Both operands are flattened
to float32-rounded Python floats before the distances are taken -- the same
normalization the torch-only version applied via ``.float()``, and the reason
an fp32 kernel can pass against an fp64 reference at all (unrounded, a
correctly-rounded fp32 value sits ~2**29 float64-ULPs from its fp64 original).

``expected_dtype`` accepts a native dtype (``torch.float32``, ``numpy.float32``)
or its name as a string, so the check is usable from a task that has no torch.

Usage in tests.py:
    from perflab.harness.precision import assert_ulp_close

    output = kernel(A.float(), B.float())
    reference = (A.double() @ B.double())
    assert_ulp_close(output, reference, max_ulp=4, expected_dtype=torch.float32)
"""
from __future__ import annotations

import math
import random
from typing import Any

from perflab.harness import _array

# A single corrupted element (as opposed to ordinary heavy-tail fp32
# rounding) can blow the max ULP distance up far beyond max_ulp even when
# the p99 distance still passes -- see the max_ulp_observed gate below.
HARD_MAX_ULP_FACTOR = 64


def _ulp_distance(
    a_float: float, b_float: float, precision: str = "float32"
) -> float:
    """Number of representable ``precision`` values between two floats.

    ULP = Unit in the Last Place: how many representable steps apart two values
    are. Sharper than an absolute or relative tolerance for catching a
    precision downgrade, because it is scale-free.

    The distance is measured in the WORKING precision -- the one the kernel
    actually computed in -- not float64. Measuring float64 ULPs over values
    that have been rounded to float32 inflates every distance by ~2**29: one
    float32 step reads as 536,870,912, so a default budget of 16 silently
    means "must be bit-identical" and rejects every honest fp32 kernel. (An
    fp32 matmul against an fp64 reference measures p50=1, p99=6 float32-ULPs,
    which is exactly what a budget of 16 was written for.)

    Implemented by mapping IEEE bit patterns onto a monotonic integer ordering
    and subtracting, which is exact at every magnitude including subnormals --
    unlike dividing by a local ULP size, which is only an approximation once
    the two values straddle an exponent boundary.
    """
    if math.isnan(a_float) or math.isnan(b_float):
        return float("inf")
    nbits = _array.FLOAT_PRECISIONS[precision][0]
    key_a = _monotonic_key(_array.float_bits(a_float, precision), nbits)
    key_b = _monotonic_key(_array.float_bits(b_float, precision), nbits)
    return float(abs(key_a - key_b))


def _monotonic_key(bits: int, nbits: int) -> int:
    """Map an IEEE bit pattern onto a monotonically increasing integer.

    Raw bits are not orderable across zero: negative floats ascend in magnitude
    as their bit patterns ascend, so subtracting raw patterns is meaningless
    for a mixed-sign pair. Flipping the negative half fixes that, and makes
    +0.0 and -0.0 land on the same key (they are numerically equal).
    """
    sign = 1 << (nbits - 1)
    return (1 << nbits) - bits if bits & sign else bits + sign


def _float_ulp(x: float) -> float:
    """Return the float64 ULP (unit in last place) for float x."""
    return math.ulp(abs(x)) if x != 0 else math.ulp(0.0)


def _infer_precision(value, fallback: str = "float32") -> str:
    """Working precision of ``value``, defaulting when it is not a float type.

    Inferring from the kernel's own output is what makes the check usable under
    mixed precision: a bf16 autocast result is compared in bf16 ULPs against a
    bf16-rounded reference, so AMP -- the intended optimization for several
    tasks -- passes, while a kernel materially worse than bf16 still fails.
    """
    try:
        name = _array.dtype_name(value)
    except Exception:  # noqa: BLE001 - dtype is advisory; fall back
        return fallback
    return name if name in _array.FLOAT_PRECISIONS else fallback


def _ceil_percentile_index(fraction: float, n_s: int) -> int:
    """Ceiling-indexed position of `fraction` into a sorted list of n_s values.

    Floor indexing (int(fraction * (n_s - 1))) rounds toward the middle of
    the distribution, so for small samples the single most extreme value can
    fall just past the computed index and never get checked -- e.g. n_s=4,
    fraction=0.99: floor(0.99*3)=2 picks the 3rd-largest value and misses the
    outlier at index 3 entirely. Ceiling indexing always includes at least
    the top `1 - fraction` fraction of the tail.
    """
    return min(n_s - 1, math.ceil(fraction * (n_s - 1)))


def assert_ulp_close(
    actual,
    reference,
    max_ulp: float = 16.0,
    expected_dtype=None,
    precision: str | None = None,
    sample_fraction: float = 0.01,
    min_samples: int = 1000,
    max_samples: int = 100000,
) -> dict:
    """Assert that actual and reference tensors are close in ULP distance.

    Computes the ULP distance element-wise (on a random sample for large
    tensors) and asserts that the p99 ULP distance is within max_ulp.

    Args:
        actual: Output array from the kernel (any supported backend).
        reference: Reference array computed in fp64.
        max_ulp: Maximum allowed ULP distance (p99). Default 16 allows
                 for normal fp32 rounding but catches fp16→fp32 casts.
        expected_dtype: If set, assert actual's dtype matches this. Accepts a
                 native dtype object or its name as a string.
        sample_fraction: Fraction of elements to sample for large arrays.
        min_samples: Minimum number of elements to check.
        max_samples: Maximum number of elements to check.

    Returns:
        Dict with statistics: {mean_ulp, p50_ulp, p95_ulp, p99_ulp, max_ulp}.

    Raises:
        AssertionError if precision check fails, or if either operand has a
        type the harness cannot inspect.
    """
    _array.require_backend(actual, "actual")
    _array.require_backend(reference, "reference")

    # Dtype check
    if expected_dtype is not None and not _array.dtype_matches(actual, expected_dtype):
        raise AssertionError(
            f"Precision downgrade detected: output dtype is "
            f"{_array.dtype_repr(actual)}, expected {expected_dtype}. The kernel "
            f"may be computing in lower precision and casting up."
        )

    a_shape = _array.shape_of(actual, "actual")
    r_shape = _array.shape_of(reference, "reference")
    if a_shape != r_shape:
        raise AssertionError(
            f"Shape mismatch: actual {a_shape} vs reference {r_shape}"
        )

    n = math.prod(a_shape)
    n_samples = min(max(int(n * sample_fraction), min(min_samples, n)), min(max_samples, n))

    # Sampled indices are drawn in pure Python (no torch.randperm), so the
    # check works on a backend-free install; sorting keeps the subsequent
    # gather sequential on every backend.
    indices = sorted(random.sample(range(n), n_samples)) if n_samples < n else None
    # The kernel's own dtype is the working precision; the reference is
    # rounded down to it so an fp64 reference can be compared at all.
    prec = precision or _infer_precision(actual)
    a_list = _array.flat_float_list(actual, prec, indices, "actual")
    r_list = _array.flat_float_list(reference, prec, indices, "reference")

    # Compute ULP distances. a_list/r_list are always the same length (same
    # shape, sampled with the same indices).
    ulp_dists = []
    for a_val, r_val in zip(a_list, r_list, strict=True):
        ulp_dists.append(_ulp_distance(a_val, r_val, prec))

    ulp_dists.sort()
    n_s = len(ulp_dists)

    stats: dict[str, Any] = {
        "mean_ulp": sum(ulp_dists) / n_s if n_s > 0 else 0,
        "p50_ulp": ulp_dists[n_s // 2] if n_s > 0 else 0,
        "p95_ulp": ulp_dists[_ceil_percentile_index(0.95, n_s)] if n_s > 1 else (ulp_dists[0] if n_s else 0),
        "p99_ulp": ulp_dists[_ceil_percentile_index(0.99, n_s)] if n_s > 1 else (ulp_dists[0] if n_s else 0),
        "max_ulp_observed": ulp_dists[-1] if n_s > 0 else 0,
        "n_samples": n_s,
        "precision": prec,
    }

    # Hard ceiling: a single corrupted element can leave p99 well within
    # bounds (the percentile window covers only ~1% of samples) while the
    # max ULP distance is catastrophic. Heavy-tail fp32 rounding is tolerated
    # up to HARD_MAX_ULP_FACTOR x max_ulp; a corrupted element is not.
    hard_max = max_ulp * HARD_MAX_ULP_FACTOR
    if stats["max_ulp_observed"] > hard_max:
        raise AssertionError(
            f"Precision downgrade detected: max ULP distance is "
            f"{stats['max_ulp_observed']:.1f}, which exceeds the hard ceiling of "
            f"{hard_max:.1f} ({HARD_MAX_ULP_FACTOR}x max_ulp={max_ulp}). "
            f"Heavy-tail fp32 rounding is tolerated up to that ceiling, but a "
            f"single element this far off is almost certainly corrupted, not "
            f"just imprecise. Checked {n_s}/{n} elements."
        )

    if stats["p99_ulp"] > max_ulp:
        raise AssertionError(
            f"Precision downgrade detected: p99 ULP distance is "
            f"{stats['p99_ulp']:.1f} (max allowed: {max_ulp}). "
            f"Mean ULP: {stats['mean_ulp']:.1f}, Max ULP: {stats['max_ulp_observed']:.1f}. "
            f"The kernel may be computing in lower precision (e.g., fp16) "
            f"and casting to fp32. Checked {n_s}/{n} elements."
        )

    return stats
