"""The no-op probe must not accuse an honest low-cardinality kernel.

Phase 2 of `assert_deterministic` infers "no-op" from "different inputs
produced the same output". That inference is unsound at low cardinality: an
8-element boolean output has 256 possible values, and a reduction over booleans
has two, so an honest kernel collides by chance. Failing on the first collision
rejected correct kernels ~2.5% of the time -- measured as a genuine flake, not
a hypothetical.

A real no-op collides on *every* draw, so retrying costs a true positive
nothing. These tests pin both directions deterministically, with scripted
inputs rather than by sampling, so they cannot themselves be flaky.
"""
from __future__ import annotations

import pytest

from perflab.harness import assert_deterministic


def scripted_factory(draws):
    """An input_factory that yields a fixed sequence, then repeats the last."""
    state = {"i": 0}

    def factory():
        i = state["i"]
        state["i"] += 1
        return draws[min(i, len(draws) - 1)]

    return factory


class TestHonestLowCardinalityKernelAccepted:
    def test_collision_on_first_probe_then_differs_is_accepted(self):
        """The exact shape of the observed flake, made deterministic.

        `any()` over booleans has a two-value output space. Draws 1 and 2 are
        different inputs that both reduce to True -- a legitimate collision.
        The old single-shot probe raised here; the retry sees draw 3 differ.
        """
        factory = scripted_factory([
            ([True, False],),   # phase 1
            ([True, False],),   # phase 2 input A
            ([False, True],),   # phase 2 input B -> different input, same output
            ([False, False],),  # retry -> output finally differs
        ])
        assert_deterministic(fn=lambda x: any(x), input_factory=factory, n_runs=2)

    def test_single_element_bool_kernel_is_accepted(self):
        """One bool in, one bool out: collides half the time by construction."""
        factory = scripted_factory([
            ([True],), ([True],), ([False],), ([False],), ([True],),
        ])
        assert_deterministic(fn=lambda x: [not x[0]], input_factory=factory, n_runs=2)


class TestGenuineNoOpStillCaught:
    def test_constant_kernel_caught_despite_retries(self):
        """A real no-op collides on every draw, so retries never rescue it."""
        counter = {"i": 0}

        def varying_factory():
            counter["i"] += 1
            return ([float(counter["i"]), 2.0],)

        with pytest.raises(AssertionError, match="No-op kernel detected"):
            assert_deterministic(
                fn=lambda x: [42.0, 42.0], input_factory=varying_factory, n_runs=2
            )

    def test_error_message_reports_how_many_draws_were_tried(self):
        """The message must say it retried, or it reads as a one-shot verdict."""
        counter = {"i": 0}

        def varying_factory():
            counter["i"] += 1
            return ([float(counter["i"])],)

        with pytest.raises(AssertionError, match="independent draws"):
            assert_deterministic(
                fn=lambda x: [7.0], input_factory=varying_factory, n_runs=2
            )

    def test_stale_buffer_kernel_still_caught(self):
        """Compute once, return the same object forever."""
        cache: dict[str, list[float]] = {}
        counter = {"i": 0}

        def varying_factory():
            counter["i"] += 1
            return ([float(counter["i"]), 1.0],)

        def stale(x):
            if "v" not in cache:
                cache["v"] = [x[0] * 2, x[1] * 2]
            return cache["v"]

        with pytest.raises(AssertionError, match="No-op kernel detected"):
            assert_deterministic(fn=stale, input_factory=varying_factory, n_runs=2)


class TestPoisonDataAlwaysDiffers:
    """`random_like` produces poison for the memoization check.

    Redrawing a value to its original leaves that element un-poisoned, so a
    cached result still matches there. At bool cardinality that happened 50% of
    the time per element; retrying only shrank the odds. It is now guaranteed
    by construction.
    """

    @pytest.mark.parametrize("value", [True, False, 0, 7, -3])
    def test_scalars_always_change(self, value):
        from perflab.harness import _array

        for _ in range(200):
            assert _array.random_like(value) != value

    def test_numpy_low_cardinality_and_boundary_dtypes(self):
        numpy = pytest.importorskip("numpy")
        from perflab.harness import _array

        cases = [
            numpy.array([True]),
            numpy.zeros(8, dtype=bool),
            numpy.full(4, 127, dtype=numpy.int8),    # +1 must wrap, not saturate
            numpy.full(4, 255, dtype=numpy.uint8),
        ]
        for original in cases:
            for _ in range(200):
                poisoned = _array.random_like(original)
                assert not bool((poisoned == original).any()), (
                    f"poison collided for dtype {original.dtype}"
                )

    def test_torch_low_cardinality_dtypes(self):
        torch = pytest.importorskip("torch")
        from perflab.harness import _array

        for original in (torch.zeros(8, dtype=torch.bool), torch.ones(4, dtype=torch.int64)):
            for _ in range(200):
                poisoned = _array.random_like(original)
                assert not bool((poisoned == original).any())
