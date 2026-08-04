"""Cross-backend semantic parity for `perflab.harness._array`.

These guard three defects found by differential testing -- running identical
data through the list, numpy and torch paths and diffing the verdicts. Each one
was invisible to single-backend testing, because each backend was individually
self-consistent; only the comparison exposed them.
"""
from __future__ import annotations

import math

import pytest

from perflab.harness import _array

numpy = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

INF = float("inf")


class TestNonFiniteParity:
    """The pure-Python leaf inverted infinity in both directions.

    It fell through to `abs(a-b) <= atol + rtol*abs(b)`, where
    `|inf - -inf| = inf` and `inf <= inf` is True (so +inf "matched" -inf --
    a wrong-answer kernel passing its correctness check), while
    `|inf - inf| = nan` and `nan <= inf` is False (so a correct kernel was
    rejected). numpy and torch were always right; only the list path was not.
    """

    @pytest.mark.parametrize(("a", "b", "expected"), [
        ([INF], [-INF], False),
        ([-INF], [INF], False),
        ([INF], [INF], True),
        ([-INF], [-INF], True),
        ([INF], [1.0], False),
        ([1.0], [INF], False),
        ([0.0, INF], [0.0, INF], True),
        ([0.0, INF], [0.0, -INF], False),
    ])
    def test_all_backends_agree(self, a, b, expected):
        as_list = _array.allclose(a, b)[0]
        as_numpy = _array.allclose(numpy.array(a), numpy.array(b))[0]
        as_torch = _array.allclose(torch.tensor(a), torch.tensor(b))[0]
        assert as_list == as_numpy == as_torch == expected

    def test_infinite_match_reports_zero_not_nan(self):
        """A passing comparison must not report max_diff=nan."""
        ok, diff = _array.allclose([INF], [INF])
        assert ok is True
        assert diff == 0.0 and not math.isnan(diff)

    def test_nan_still_never_matches(self):
        nan = float("nan")
        for a, b in (([nan], [nan]), ([nan], [1.0]), ([nan], [INF])):
            assert _array.allclose(a, b)[0] is False


class TestMixedPrecisionTorch:
    """torch raised RuntimeError on a dtype mismatch.

    RuntimeError is not an AssertionError, so it escaped a task's correctness
    handling as an unrelated crash rather than a check failure. Mixed precision
    is the *intended* optimization for several tasks, so a bf16 kernel output
    compared against an fp32 reference is the normal case, not an error -- and
    the numpy path already handled it, so this was pure backend divergence.
    """

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
    def test_equal_values_across_dtypes_compare_equal(self, dtype):
        ok, _ = _array.allclose(torch.zeros(4, dtype=dtype), torch.zeros(4))
        assert ok is True

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
    def test_unequal_values_across_dtypes_still_caught(self, dtype):
        """Promotion must not paper over a genuine difference."""
        ok, _ = _array.allclose(torch.ones(4, dtype=dtype), torch.zeros(4))
        assert ok is False

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
    def test_exact_equal_across_dtypes(self, dtype):
        assert _array.exact_equal(torch.zeros(4, dtype=dtype), torch.zeros(4)) is True
        assert _array.exact_equal(torch.ones(4, dtype=dtype), torch.zeros(4)) is False

    def test_no_runtime_error_leaks(self):
        """Whatever happens, it must not be a bare RuntimeError."""
        try:
            _array.allclose(torch.zeros(4, dtype=torch.bfloat16), torch.zeros(4))
        except RuntimeError as exc:  # pragma: no cover - the bug being guarded
            pytest.fail(f"RuntimeError escaped instead of comparing: {exc}")


class TestMultiOutputStructures:
    """Heterogeneous multi-output kernels were rejected outright.

    `shape_of` called (out, lse) "ragged" because it tries to derive ONE shape,
    so attention (out, lse), top-k (values, indices) and (out, scalar_loss)
    could not be checked at all. As a structure they compare fine: pair by
    position, compare each.
    """

    def test_matching_multi_output_passes(self):
        pair = (torch.zeros(4, 4), torch.zeros(4))
        ok, _ = _array.allclose(pair, (torch.zeros(4, 4), torch.zeros(4)))
        assert ok is True

    def test_difference_in_any_output_is_caught(self):
        ok, diff = _array.allclose(
            (torch.zeros(4, 4), torch.zeros(4)),
            (torch.zeros(4, 4), torch.ones(4)),
        )
        assert ok is False and diff == 1.0

    def test_first_output_difference_is_caught(self):
        ok, _ = _array.allclose(
            (torch.ones(4, 4), torch.zeros(4)),
            (torch.zeros(4, 4), torch.zeros(4)),
        )
        assert ok is False

    def test_works_for_numpy_and_list_structures_too(self):
        a = (numpy.zeros((4, 4)), numpy.zeros(4))
        assert _array.allclose(a, (numpy.zeros((4, 4)), numpy.zeros(4)))[0] is True
        assert _array.allclose(a, (numpy.zeros((4, 4)), numpy.ones(4)))[0] is False

    def test_genuinely_mismatched_leaf_shapes_still_fail(self):
        """Standing the single-shape guard down must not disable shape checking."""
        with pytest.raises(AssertionError):
            _array.allclose(
                (torch.zeros(4, 4), torch.zeros(4)),
                (torch.zeros(4, 4), torch.zeros(8)),
            )
