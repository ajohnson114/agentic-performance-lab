"""Tests for PairedDifference and the stdlib exact signed-rank test.

Two things are being pinned here, and the second matters at least as much as
the first.

1. The statistics are *right*: the dynamic-programming null distribution is
   checked against brute-force enumeration of all 2^n sign assignments,
   including ties and zeros, and against the published critical values.

2. The gate FAILS CLOSED. There is direct history behind this: an earlier
   ``is_statistically_significant()`` helper depended on scipy and returned
   "significant" when scipy was missing. Every degradation path below -- no
   pairs, too few pairs, non-finite differences -- must land on a rule at least
   as strict as the one the project already shipped, and must be visible in the
   verdict rather than silent.
"""
from __future__ import annotations

import itertools
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from perflab.analyzers.decision import (
    NON_OVERLAPPING_CI,
    PAIRED_DIFFERENCE,
    TOLERANCE_ONLY,
    Comparison,
    PairedDifference,
    _doubled_midranks,
    decide,
    get_rule,
    is_paired_rule,
    min_attainable_p,
    rule_for_constraints,
    rule_names,
    signed_rank_p_greater,
)


def _brute_force_p(diffs: list[float]) -> float:
    """P(W+ >= observed) by enumerating every sign assignment. Ground truth."""
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    ranks = _doubled_midranks([abs(d) for d in nonzero])
    observed = sum(r for r, d in zip(ranks, nonzero, strict=True) if d > 0)
    hits = sum(
        1 for signs in itertools.product((0, 1), repeat=n)
        if sum(r for r, s in zip(ranks, signs, strict=True) if s) >= observed
    )
    return hits / 2 ** n


def _pairs_from_diffs(diffs: list[float], base: float = 100.0):
    """(candidate, incumbent) pairs realising the given *minimize-mode* diffs.

    Positive ``d`` means better, so in minimize mode the candidate is *lower*.
    """
    return [(base - d, base) for d in diffs]


class TestExactSignedRank:
    def test_matches_brute_force_enumeration(self):
        """Including ties and zeros, which is where rank tests usually break."""
        rng = random.Random(20260803)
        for _ in range(300):
            n = rng.randint(1, 11)
            diffs = [
                rng.choice([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0])
                for _ in range(n)
            ]
            result = signed_rank_p_greater(diffs)
            assert result is not None
            assert result[0] == pytest.approx(_brute_force_p(diffs), abs=1e-12)

    def test_all_positive_gives_the_floor_p_value(self):
        for n in (5, 6, 7, 8):
            p, nonzero = signed_rank_p_greater([float(i + 1) for i in range(n)])
            assert nonzero == n
            assert p == pytest.approx(0.5 ** n)
            assert p == pytest.approx(min_attainable_p(n))

    def test_published_critical_values(self):
        """One-sided alpha=0.05 rejects at W+ >= 19 for n=6 (W- <= 2)."""
        # One negative at the smallest magnitude: W+ = 2+3+4+5+6 = 20 -> reject.
        p, _ = signed_rank_p_greater([-1, 2, 3, 4, 5, 6])
        assert p == pytest.approx(0.03125)
        assert p < 0.05
        # One negative at the largest magnitude: W+ = 1+2+3+4+5 = 15 -> no.
        p, _ = signed_rank_p_greater([1, 2, 3, 4, 5, -6])
        assert p == pytest.approx(0.21875)
        assert p > 0.05

    def test_ties_get_averaged_ranks(self):
        assert _doubled_midranks([5.0, 5.0, 9.0]) == [3, 3, 6]
        assert _doubled_midranks([1.0, 2.0, 3.0]) == [2, 4, 6]

    def test_zeros_are_dropped_and_reported(self):
        p, nonzero = signed_rank_p_greater([0.0, 0.0, 1.0, 2.0, 3.0])
        assert nonzero == 3
        assert p == pytest.approx(0.125)

    def test_all_zeros_is_no_evidence_not_evidence(self):
        """Identical measurements must never read as an improvement."""
        assert signed_rank_p_greater([0.0] * 8) == (1.0, 0)

    def test_non_finite_differences_produce_no_test_at_all(self):
        """None means "could not test" -- callers must not read it as a p-value."""
        assert signed_rank_p_greater([1.0, float("nan"), 3.0]) is None
        assert signed_rank_p_greater([1.0, float("inf")]) is None

    def test_all_negative_is_never_significant(self):
        p, _ = signed_rank_p_greater([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
        assert p == 1.0

    def test_normal_approximation_takes_over_for_large_n_conservatively(self):
        """Above the exact cutoff the tail must not be *smaller* than exact."""
        exact_p, _ = signed_rank_p_greater([float(i + 1) for i in range(50)])
        approx_p, _ = signed_rank_p_greater([float(i + 1) for i in range(51)])
        assert exact_p == pytest.approx(0.5 ** 50)
        # A one-pair-larger, equally decisive sample must still reject easily,
        # and the approximation errs toward a larger (safer) p in the far tail.
        assert approx_p < 1e-6
        assert approx_p > exact_p

    def test_min_attainable_p_explains_the_block_floor(self):
        assert min_attainable_p(4) == pytest.approx(0.0625)   # cannot reach 0.05
        assert min_attainable_p(5) == pytest.approx(0.03125)  # can
        assert min_attainable_p(4) > 0.05 > min_attainable_p(5)


class TestPairedDifferenceAccepts:
    def test_a_consistent_win_across_blocks_is_accepted(self):
        pairs = _pairs_from_diffs([6.0, 5.0, 7.0, 4.0, 6.5, 5.5])
        verdict = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=94.0, incumbent=100.0, mode="minimize",
            tolerance=0.02, pairs=pairs,
        ))
        assert verdict.improved is True
        assert verdict.verified is True
        assert verdict.paired is True
        assert verdict.p_value == pytest.approx(0.015625)
        assert verdict.reason == ""
        assert verdict.rule == "paired_difference"

    def test_maximize_mode_uses_the_shared_sign_normalisation(self):
        """Comparison.paired_diffs owns the sign; the rule must not re-derive it."""
        pairs = [(110.0 + i, 100.0) for i in (0, 1, 2, 3, 4, 5)]  # candidate higher
        verdict = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=112.5, incumbent=100.0, mode="maximize",
            tolerance=0.02, pairs=pairs,
        ))
        assert verdict.improved is True
        assert verdict.p_value == pytest.approx(0.015625)

    def test_the_same_pairs_reversed_are_a_regression_not_a_win(self):
        pairs = [(90.0, 100.0)] * 6
        better = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=90.0, incumbent=100.0, mode="minimize",
            tolerance=0.02, pairs=pairs,
        ))
        worse = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=90.0, incumbent=100.0, mode="maximize",
            tolerance=0.02, pairs=pairs,
        ))
        assert better.improved is True
        assert worse.improved is False

    def test_verdict_reports_the_paired_difference_cv_not_an_arm_cv(self):
        pairs = _pairs_from_diffs([6.0, 5.0, 7.0, 4.0, 6.5, 5.5])
        verdict = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=94.0, incumbent=100.0, mode="minimize",
            tolerance=0.02, pairs=pairs,
        ))
        assert verdict.cv is not None
        assert verdict.cv == pytest.approx(
            (sum((d - 5.6666666666666667) ** 2 for d in
                 [6.0, 5.0, 7.0, 4.0, 6.5, 5.5]) / 5) ** 0.5 / 100.0,
            rel=1e-6,
        )
        assert verdict.n == 6

    def test_second_bar_is_significance_so_there_is_no_magnitude_to_report(self):
        """`required` must not carry a number the rule never actually used."""
        pairs = _pairs_from_diffs([6.0, 5.0, 7.0, 4.0, 6.5, 5.5])
        verdict = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=94.0, incumbent=100.0, mode="minimize",
            tolerance=0.02, pairs=pairs,
        ))
        assert verdict.noise_required is None
        assert verdict.required == verdict.tolerance


class TestPairedDifferenceRejects:
    def test_the_materiality_floor_still_applies(self):
        """Perfectly consistent, perfectly significant, and far too small."""
        pairs = _pairs_from_diffs([0.5] * 6)      # 0.5% win, tolerance is 2%
        verdict = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=99.5, incumbent=100.0, mode="minimize",
            tolerance=0.02, pairs=pairs,
        ))
        assert verdict.p_value == pytest.approx(0.015625)   # significant
        assert verdict.beats_tolerance is False
        assert verdict.improved is False
        assert "regression tolerance" in verdict.reason

    def test_a_large_but_inconsistent_win_is_rejected(self):
        """Half the blocks say better, half say worse -- that is not a win."""
        pairs = _pairs_from_diffs([20.0, -18.0, 22.0, -19.0, 21.0, -17.0])
        verdict = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=95.0, incumbent=100.0, mode="minimize",
            tolerance=0.02, pairs=pairs,
        ))
        assert verdict.beats_tolerance is True
        assert verdict.improved is False
        assert verdict.paired is True
        assert "not significant under pairing" in verdict.reason

    def test_reject_reasons_are_always_populated(self):
        for pairs, cand in (
            (_pairs_from_diffs([0.5] * 6), 99.5),
            (_pairs_from_diffs([20.0, -18.0, 22.0, -19.0, 21.0, -17.0]), 95.0),
        ):
            verdict = PAIRED_DIFFERENCE.decide(Comparison(
                candidate=cand, incumbent=100.0, mode="minimize",
                tolerance=0.02, pairs=pairs,
            ))
            assert verdict.improved is False
            assert verdict.reason


class TestFailsClosed:
    """No pairs, too few pairs, or a broken test must never mean "accept"."""

    def _no_pairs(self, **kw) -> Comparison:
        base = dict(
            candidate=95.0, incumbent=100.0, mode="minimize", tolerance=0.02,
        )
        base.update(kw)
        return Comparison(**base)  # type: ignore[arg-type]

    def test_no_pairs_degrades_to_the_unpaired_gate_not_to_acceptance(self):
        # A 5% "win" on a machine with a wide spread: the unpaired interval
        # rule rejects it, and so must the paired rule when it has no pairs.
        noisy = self._no_pairs(
            candidate_samples=[95.0, 87.0, 103.0, 91.0, 99.0],
            incumbent_samples=[100.0, 92.0, 108.0, 96.0, 104.0],
        )
        assert NON_OVERLAPPING_CI.decide(noisy).improved is False
        verdict = PAIRED_DIFFERENCE.decide(noisy)
        assert verdict.improved is False
        assert verdict.paired is False
        assert verdict.verified is False
        assert verdict.p_value is None

    def test_the_fallback_is_never_more_permissive_than_the_unpaired_rule(self):
        """Exhaustive over a grid: fail-closed is a property, not an example."""
        for observed in (0.0, 0.01, 0.03, 0.10, 0.40):
            for spread in (0.0, 0.02, 0.10):
                candidate = 100.0 * (1 - observed)
                samples = [candidate * (1 + spread * s) for s in (-1, 0, 1)]
                comparison = self._no_pairs(
                    candidate=candidate,
                    candidate_samples=samples,
                    incumbent_samples=[100.0 * (1 + spread * s) for s in (-1, 0, 1)],
                )
                paired = PAIRED_DIFFERENCE.decide(comparison)
                unpaired = NON_OVERLAPPING_CI.decide(comparison)
                assert paired.improved == unpaired.improved
                assert paired.verified is False

    def test_a_rejection_says_why_the_paired_test_did_not_run(self):
        """The note rides on `reason`, which the protocol reserves for rejects.

        On the accept side there is no reason field to use (the protocol says
        it is "" when improved), so the signal there is verified=False /
        paired=False -- which is what phases/evaluate.py branches on to print
        its "accepted WITHOUT the paired test" note.
        """
        verdict = PAIRED_DIFFERENCE.decide(self._no_pairs(candidate=99.9))
        assert verdict.improved is False
        assert "paired test not run" in verdict.reason
        assert "0 pair(s)" in verdict.reason

    def test_too_few_pairs_to_ever_reject_falls_back_rather_than_pretending(self):
        """At 4 pairs the best attainable p is 0.0625 -- the gate can't open."""
        pairs = _pairs_from_diffs([6.0, 5.0, 7.0, 4.0])
        verdict = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=94.0, incumbent=100.0, mode="minimize",
            tolerance=0.02, pairs=pairs,
        ))
        assert verdict.paired is False
        assert verdict.verified is False
        # ... and on a rejecting comparison it explains the shortfall.
        rejected = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=99.9, incumbent=100.0, mode="minimize",
            tolerance=0.02, pairs=_pairs_from_diffs([0.1] * 4),
        ))
        assert "needs 5" in rejected.reason
        assert "0.062" in rejected.reason

    def test_five_pairs_is_the_first_count_that_runs_the_paired_test(self):
        for n, expect_paired in ((4, False), (5, True), (6, True)):
            pairs = _pairs_from_diffs([6.0 + i * 0.1 for i in range(n)])
            verdict = PAIRED_DIFFERENCE.decide(Comparison(
                candidate=94.0, incumbent=100.0, mode="minimize",
                tolerance=0.02, pairs=pairs,
            ))
            assert verdict.paired is expect_paired

    def test_a_non_finite_pair_disables_the_test_without_accepting(self):
        pairs = _pairs_from_diffs([6.0, 5.0, 7.0, 4.0, 6.5, 5.5])
        pairs[2] = (float("nan"), 100.0)
        verdict = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=94.0, incumbent=100.0, mode="minimize",
            tolerance=0.02, pairs=pairs,
        ))
        assert verdict.paired is False
        assert verdict.verified is False
        assert "non-finite" in verdict.reason or verdict.improved is True

    def test_verified_is_true_only_when_the_paired_test_actually_ran(self):
        ran = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=94.0, incumbent=100.0, mode="minimize", tolerance=0.02,
            pairs=_pairs_from_diffs([6.0, 5.0, 7.0, 4.0, 6.5, 5.5]),
        ))
        did_not = PAIRED_DIFFERENCE.decide(self._no_pairs())
        assert ran.verified is True
        assert did_not.verified is False

    def test_an_explicit_fallback_can_be_swapped_in(self):
        rule = PairedDifference(fallback=TOLERANCE_ONLY)
        verdict = rule.decide(self._no_pairs())
        assert verdict.rule == "paired_difference"
        assert verdict.improved is True          # tolerance_only accepts a 5% win
        assert verdict.verified is False         # but never claims verification

    def test_no_third_party_statistics_package_is_involved(self):
        """The scipy-shaped failure mode cannot recur if scipy is never used.

        An earlier ``is_statistically_significant()`` helper imported scipy in a
        try/except and returned "significant" from the except branch, so a
        machine without scipy silently accepted everything. Checking
        ``sys.modules`` alone would not catch a reintroduction (a lazy import
        inside ``decide`` would not show up until it ran), so this reads the
        source of both modules.
        """
        import perflab.analyzers.decision as decision_mod
        import perflab.runners.paired as paired_mod

        for module in (decision_mod, paired_mod):
            source = Path(module.__file__).read_text(encoding="utf-8")
            code = "\n".join(
                line for line in source.splitlines()
                if line.lstrip().startswith(("import ", "from "))
            )
            for banned in ("scipy", "numpy", "pandas", "statsmodels", "sklearn"):
                assert banned not in code, f"{module.__name__} imports {banned}"
        assert "scipy" not in sys.modules

    def test_the_test_runs_with_no_optional_dependency_available(self, monkeypatch):
        """Simulate the exact historical failure: the package is simply absent.

        Blocking the import must change nothing, because nothing imports it.
        """
        import builtins

        real_import = builtins.__import__

        def no_scipy(name, *args, **kwargs):
            if name.split(".")[0] in {"scipy", "numpy"}:
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_scipy)
        verdict = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=94.0, incumbent=100.0, mode="minimize", tolerance=0.02,
            pairs=_pairs_from_diffs([6.0, 5.0, 7.0, 4.0, 6.5, 5.5]),
        ))
        assert verdict.improved is True
        assert verdict.p_value == pytest.approx(0.015625)


class TestRegistryIntegration:
    def test_registered_and_selectable_by_name(self):
        assert "paired_difference" in rule_names()
        assert get_rule("paired_difference") is PAIRED_DIFFERENCE

    def test_selected_through_task_constraints(self):
        constraints = SimpleNamespace(
            noise_gate=True, decision_rule="paired_difference",
        )
        assert rule_for_constraints(constraints) is PAIRED_DIFFERENCE

    def test_noise_gate_off_still_wins(self):
        constraints = SimpleNamespace(
            noise_gate=False, decision_rule="paired_difference",
        )
        assert rule_for_constraints(constraints) is TOLERANCE_ONLY

    def test_it_is_not_the_default(self):
        """Pairing roughly doubles the authoritative benchmark: opt in."""
        assert rule_for_constraints(SimpleNamespace()) is NON_OVERLAPPING_CI

    def test_is_paired_rule_identifies_only_the_paired_rule(self):
        assert is_paired_rule(PAIRED_DIFFERENCE) is True
        assert is_paired_rule(NON_OVERLAPPING_CI) is False
        assert is_paired_rule(TOLERANCE_ONLY) is False

    def test_decides_through_the_shared_entry_point(self):
        verdict = decide(
            Comparison(
                candidate=94.0, incumbent=100.0, mode="minimize", tolerance=0.02,
                pairs=_pairs_from_diffs([6.0, 5.0, 7.0, 4.0, 6.5, 5.5]),
            ),
            PAIRED_DIFFERENCE,
        )
        assert verdict.improved is True

    def test_the_rule_is_stateless_and_reusable(self):
        comparison = Comparison(
            candidate=94.0, incumbent=100.0, mode="minimize", tolerance=0.02,
            pairs=_pairs_from_diffs([6.0, 5.0, 7.0, 4.0, 6.5, 5.5]),
        )
        first = PAIRED_DIFFERENCE.decide(comparison)
        assert PAIRED_DIFFERENCE.decide(comparison) == first
        assert PairedDifference().decide(comparison) == first

    def test_alpha_is_configurable_without_touching_the_registry(self):
        strict = PairedDifference(alpha=0.01)
        pairs = _pairs_from_diffs([6.0, 5.0, 7.0, 4.0, 6.5, 5.5])   # p = 0.0156
        comparison = Comparison(
            candidate=94.0, incumbent=100.0, mode="minimize",
            tolerance=0.02, pairs=pairs,
        )
        assert PAIRED_DIFFERENCE.decide(comparison).improved is True
        assert strict.decide(comparison).improved is False
        # ... and says the design, not the candidate, was the limit.
        assert "cannot reach it" in strict.decide(comparison).reason

    def test_default_alpha_is_attainable_at_the_default_block_count(self):
        """A gate that can never open is a bug, not a strict gate."""
        from perflab.runners.paired import DEFAULT_BLOCKS, MIN_BLOCKS
        assert min_attainable_p(MIN_BLOCKS) < PairedDifference().alpha
        assert min_attainable_p(DEFAULT_BLOCKS) < PairedDifference().alpha
        assert PairedDifference().min_pairs == MIN_BLOCKS

    def test_exact_test_is_fast_enough_to_sit_in_the_accept_path(self):
        import time
        diffs = [float(i + 1) for i in range(50)]
        started = time.perf_counter()
        for _ in range(20):
            signed_rank_p_greater(diffs)
        assert time.perf_counter() - started < 1.0
        assert not math.isnan(signed_rank_p_greater(diffs)[0])
