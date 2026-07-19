"""Mitigation #5: ULP Precision Guard (Precision Downgrade Detection).

Prevents LLM-generated code from computing in lower precision (e.g., fp16)
then casting to fp32 to achieve speed gains while degrading accuracy near
tolerance thresholds.

The fix: compare kernel output against an fp64 reference using ULP (units in
last place) distance, which is more sensitive than simple allclose checks.
Also checks that the output dtype matches the expected dtype.

Usage in tests.py:
    from perflab.harness.precision import assert_ulp_close

    output = kernel(A.float(), B.float())
    reference = (A.double() @ B.double())
    assert_ulp_close(output, reference, max_ulp=4, expected_dtype=torch.float32)
"""
from __future__ import annotations

import math

# A single corrupted element (as opposed to ordinary heavy-tail fp32
# rounding) can blow the max ULP distance up far beyond max_ulp even when
# the p99 distance still passes -- see the max_ulp_observed gate below.
HARD_MAX_ULP_FACTOR = 64


def _ulp_distance(a_float: float, b_float: float) -> float:
    """Compute ULP distance between two floats.

    ULP = Unit in the Last Place. Measures how many representable floats
    apart two values are. More precise than absolute/relative tolerance
    for catching precision downgrades.
    """
    if math.isnan(a_float) or math.isnan(b_float):
        return float("inf")
    if a_float == b_float:
        return 0.0
    # Use the smaller exponent's ULP as the reference
    if a_float == 0.0:
        return abs(b_float) / _float_ulp(b_float)
    if b_float == 0.0:
        return abs(a_float) / _float_ulp(a_float)
    ulp_size = min(_float_ulp(a_float), _float_ulp(b_float))
    if ulp_size == 0:
        return float("inf")
    return abs(a_float - b_float) / ulp_size


def _float_ulp(x: float) -> float:
    """Return the ULP (unit in last place) for float x."""
    return math.ulp(abs(x)) if x != 0 else math.ulp(0.0)


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
    sample_fraction: float = 0.01,
    min_samples: int = 1000,
    max_samples: int = 100000,
) -> dict:
    """Assert that actual and reference tensors are close in ULP distance.

    Computes the ULP distance element-wise (on a random sample for large
    tensors) and asserts that the p99 ULP distance is within max_ulp.

    Args:
        actual: Output tensor from the kernel.
        reference: Reference tensor computed in fp64.
        max_ulp: Maximum allowed ULP distance (p99). Default 16 allows
                 for normal fp32 rounding but catches fp16→fp32 casts.
        expected_dtype: If set, assert actual.dtype matches this.
        sample_fraction: Fraction of elements to sample for large tensors.
        min_samples: Minimum number of elements to check.
        max_samples: Maximum number of elements to check.

    Returns:
        Dict with statistics: {mean_ulp, p50_ulp, p95_ulp, p99_ulp, max_ulp}.

    Raises:
        AssertionError if precision check fails.
    """
    import torch

    # Dtype check
    if expected_dtype is not None and actual.dtype != expected_dtype:
        raise AssertionError(
            f"Precision downgrade detected: output dtype is {actual.dtype}, "
            f"expected {expected_dtype}. The kernel may be computing in lower "
            f"precision and casting up."
        )

    # Flatten for sampling
    a_flat = actual.detach().float().cpu().flatten()
    r_flat = reference.detach().float().cpu().flatten()

    if a_flat.shape != r_flat.shape:
        raise AssertionError(
            f"Shape mismatch: actual {actual.shape} vs reference {reference.shape}"
        )

    n = a_flat.numel()
    n_samples = min(max(int(n * sample_fraction), min(min_samples, n)), min(max_samples, n))

    if n_samples < n:
        indices = torch.randperm(n)[:n_samples]
        a_sampled = a_flat[indices]
        r_sampled = r_flat[indices]
    else:
        a_sampled = a_flat
        r_sampled = r_flat

    # Compute ULP distances
    ulp_dists = []
    a_list = a_sampled.tolist()
    r_list = r_sampled.tolist()
    # a_list/r_list are always the same length (sliced/derived together above).
    for a_val, r_val in zip(a_list, r_list, strict=True):
        ulp_dists.append(_ulp_distance(a_val, r_val))

    ulp_dists.sort()
    n_s = len(ulp_dists)

    stats = {
        "mean_ulp": sum(ulp_dists) / n_s if n_s > 0 else 0,
        "p50_ulp": ulp_dists[n_s // 2] if n_s > 0 else 0,
        "p95_ulp": ulp_dists[_ceil_percentile_index(0.95, n_s)] if n_s > 1 else (ulp_dists[0] if n_s else 0),
        "p99_ulp": ulp_dists[_ceil_percentile_index(0.99, n_s)] if n_s > 1 else (ulp_dists[0] if n_s else 0),
        "max_ulp_observed": ulp_dists[-1] if n_s > 0 else 0,
        "n_samples": n_s,
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
