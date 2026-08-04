"""Tests for perflab.runners.paired — block-interleaved A/B measurement.

The centrepiece is :class:`TestDriftCancellation`. Everything else in PerfLab's
statistics stack assumes the two arms of a comparison were measured under
comparable conditions; they are not, because the incumbent is benchmarked at
the top of an iteration and the candidate minutes later. These tests inject a
*known* linear drift into a simulated machine and show, arithmetically:

  * the status-quo sequential design turns pure drift into a fake improvement
    large enough to clear the accept gate;
  * naive ABAB alternation shrinks that bias but does not remove it;
  * counterbalanced ABBA removes it exactly.

Running the ordering logic against a synthetic machine (rather than real
subprocesses) is the point: with no measurement noise the expected bias of each
design is an exact number, so the assertions can be equalities rather than
hopeful inequalities.
"""
from __future__ import annotations

import math
import random
import statistics

import pytest

from perflab.analyzers.decision import (
    NON_OVERLAPPING_CI,
    PAIRED_DIFFERENCE,
    Comparison,
)
from perflab.runners import paired as paired_mod
from perflab.runners.paired import (
    CANDIDATE,
    DEFAULT_BLOCKS,
    INCUMBENT,
    MIN_BLOCKS,
    BlockPlan,
    PairedRun,
    SpawnMeasurement,
    first_block_penalty,
    interleave_order,
    plan_blocks,
    position_imbalance,
    run_interleaved,
    run_paired_benchmark,
)


class _DriftingMachine:
    """A machine whose measurements drift linearly with spawn position.

    ``truth`` is each arm's true value, ``drift`` the per-spawn change common
    to both arms (thermal wander, clock decay, a background job finishing).
    Optional Gaussian ``jitter`` is seeded, so every test here is deterministic.
    """

    def __init__(
        self,
        truth: dict[str, float],
        drift: float,
        jitter: float = 0.0,
        seed: int = 1234,
    ) -> None:
        self.truth = truth
        self.drift = drift
        self.jitter = jitter
        self._rng = random.Random(seed)
        self.position = 0
        self.calls: list[str] = []

    def __call__(self, arm: str) -> SpawnMeasurement:
        value = self.truth[arm] + self.drift * self.position
        if self.jitter:
            value += self._rng.gauss(0.0, self.jitter)
        self.position += 1
        self.calls.append(arm)
        return SpawnMeasurement(value=value, samples=[value], bench={"ok": True})

    def run_order(self, order: list[str]) -> dict[str, list[float]]:
        """Measure an arbitrary spawn order; returns per-arm value lists."""
        out: dict[str, list[float]] = {INCUMBENT: [], CANDIDATE: []}
        for arm in order:
            out[arm].append(self(arm).value)
        return out


def _sequential_order(n_pairs: int) -> list[str]:
    """The status quo: every A measurement, then every B measurement.

    This is what PerfLab does today -- ``phases/baseline.py`` measures the
    incumbent, then ``phases/evaluate.py`` measures candidates much later.
    """
    return [INCUMBENT] * n_pairs + [CANDIDATE] * n_pairs


class TestOrdering:
    def test_abba_flips_the_within_pair_order_every_other_pair(self):
        assert interleave_order(4) == list("ABBAABBA")

    def test_abab_is_available_for_contrast_only(self):
        assert interleave_order(4, counterbalanced=False) == list("ABABABAB")

    def test_empty_and_degenerate_counts(self):
        assert interleave_order(0) == []
        assert interleave_order(1) == list("AB")

    def test_position_imbalance_is_zero_for_even_abba(self):
        for n_pairs in (2, 4, 6, 8, 20):
            assert position_imbalance(interleave_order(n_pairs)) == 0.0

    def test_position_imbalance_is_exactly_one_slot_for_abab(self):
        """ABAB always measures B one slot later, at every pair count."""
        for n_pairs in (2, 4, 6, 7, 20):
            order = interleave_order(n_pairs, counterbalanced=False)
            assert position_imbalance(order) == 1.0

    def test_odd_pair_counts_leave_a_small_residual(self):
        """The last pair has no mirror, so 1/n_pairs of a slot survives.

        Still an order of magnitude better than ABAB -- and the reason
        DEFAULT_BLOCKS is even.
        """
        for n_pairs in (5, 7, 9):
            imbalance = position_imbalance(interleave_order(n_pairs))
            assert imbalance == pytest.approx(1.0 / n_pairs)
            assert 0 < imbalance < 0.25

    def test_sequential_imbalance_grows_with_the_run(self):
        """The design in use today: bias scales with how long the run is."""
        assert position_imbalance(_sequential_order(6)) == 6.0


class TestDriftCancellation:
    """A null candidate on a drifting machine must not look like a win."""

    # A machine that gets 0.6% faster per spawn -- a GPU settling after an
    # earlier stress, or a laptop whose background indexer finished. The
    # candidate is measured later, so drift flatters it. Latency, so lower is
    # better and a downward drift is an apparent improvement.
    TRUTH = {INCUMBENT: 100.0, CANDIDATE: 100.0}   # identical programs
    DRIFT = -0.6
    PAIRS = 6

    def _arm_means(self, order: list[str]) -> tuple[float, float]:
        machine = _DriftingMachine(self.TRUTH, self.DRIFT)
        measured = machine.run_order(order)
        return (
            statistics.fmean(measured[CANDIDATE]),
            statistics.fmean(measured[INCUMBENT]),
        )

    def test_sequential_design_manufactures_an_improvement(self):
        """The bias is drift x (mean position gap), and it is not small."""
        cand, inc = self._arm_means(_sequential_order(self.PAIRS))
        bias = cand - inc
        assert bias == pytest.approx(self.DRIFT * self.PAIRS)     # -3.6 ms
        apparent = (inc - cand) / inc                             # minimize mode
        assert apparent > 0.03
        # ... which is more than the project's default 2% accept gate.
        assert apparent > 0.02

    def test_abab_shrinks_the_bias_to_exactly_one_drift_step(self):
        cand, inc = self._arm_means(
            interleave_order(self.PAIRS, counterbalanced=False)
        )
        assert cand - inc == pytest.approx(self.DRIFT)            # -0.6 ms

    def test_abba_cancels_linear_drift_exactly(self):
        cand, inc = self._arm_means(interleave_order(self.PAIRS))
        assert cand - inc == pytest.approx(0.0, abs=1e-12)

    def test_abba_beats_abab_under_the_same_drift(self):
        """The whole reason counterbalancing was chosen over alternation."""
        abba = abs(self._arm_means(interleave_order(self.PAIRS))[0]
                   - self._arm_means(interleave_order(self.PAIRS))[1])
        abab_c, abab_i = self._arm_means(
            interleave_order(self.PAIRS, counterbalanced=False)
        )
        assert abba < abs(abab_c - abab_i)
        assert abba == pytest.approx(0.0, abs=1e-12)

    def test_abba_turns_the_bias_into_symmetric_spread_not_zero_residual(self):
        """The precise, non-overclaimed guarantee.

        Individual differences are NOT drift-free: an "AB" pair carries +c and
        the mirrored "BA" pair carries -c, because the two spawns in a pair are
        one slot apart either way. What ABBA guarantees is that those residuals
        arrive in equal numbers with opposite signs, so the distribution of d is
        centered on the true effect. ABAB puts +c on every pair instead, which
        shifts the whole distribution and no averaging removes it.
        """
        machine = _DriftingMachine(self.TRUTH, self.DRIFT)
        abba = run_interleaved(
            machine, plan_blocks(12, blocks=self.PAIRS, lead_in=False),
        ).diffs
        assert abba == pytest.approx([self.DRIFT, -self.DRIFT] * (self.PAIRS // 2))
        assert statistics.fmean(abba) == pytest.approx(0.0, abs=1e-12)
        assert statistics.median(abba) == pytest.approx(0.0, abs=1e-12)

        machine = _DriftingMachine(self.TRUTH, self.DRIFT)
        abab = run_interleaved(
            machine,
            plan_blocks(12, blocks=self.PAIRS, lead_in=False, counterbalanced=False),
        ).diffs
        # Every single pair biased the same way -- a shifted distribution.
        assert abab == pytest.approx([self.DRIFT] * self.PAIRS)
        assert statistics.fmean(abab) == pytest.approx(self.DRIFT)

    def test_the_symmetric_residual_cannot_manufacture_significance(self):
        """The residual spread costs power; it must never buy a false accept."""
        machine = _DriftingMachine(self.TRUTH, self.DRIFT)
        run = run_interleaved(machine, plan_blocks(12, blocks=self.PAIRS, lead_in=False))
        verdict = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=run.candidate_value, incumbent=run.incumbent_value,
            mode="minimize", tolerance=0.02, pairs=run.pairs,
        ))
        assert verdict.improved is False
        assert verdict.p_value is not None and verdict.p_value > 0.05

    def test_lead_in_spawns_do_not_change_the_residual_structure(self):
        """Priming shifts the intercept of the drift line, not its slope."""
        machine = _DriftingMachine(self.TRUTH, self.DRIFT)
        run = run_interleaved(machine, plan_blocks(12, blocks=self.PAIRS, lead_in=True))
        assert machine.position == 2 * self.PAIRS + 2
        assert statistics.fmean(run.diffs) == pytest.approx(0.0, abs=1e-12)

    def test_the_decision_gate_accepts_the_fake_win_unpaired_and_rejects_it_paired(self):
        """End to end: the same drifting null candidate, judged both ways.

        This is the finding in one assertion. Nothing about the candidate
        differs between the two branches -- only when its arm was measured.
        """
        # Unpaired, sequential: what PerfLab measures today.
        seq = _DriftingMachine(self.TRUTH, self.DRIFT, jitter=0.05, seed=7)
        measured = seq.run_order(_sequential_order(self.PAIRS))
        unpaired = Comparison(
            candidate=statistics.median(measured[CANDIDATE]),
            incumbent=statistics.median(measured[INCUMBENT]),
            mode="minimize", tolerance=0.02,
            candidate_samples=measured[CANDIDATE],
            incumbent_samples=measured[INCUMBENT],
        )
        unpaired_verdict = NON_OVERLAPPING_CI.decide(unpaired)
        assert unpaired_verdict.improved is True          # false positive
        assert unpaired_verdict.verified is True          # and confidently so

        # Paired, ABBA: same machine, same drift, same null candidate.
        machine = _DriftingMachine(self.TRUTH, self.DRIFT, jitter=0.05, seed=7)
        run = run_interleaved(machine, plan_blocks(12, blocks=self.PAIRS, lead_in=False))
        paired = Comparison(
            candidate=run.candidate_value, incumbent=run.incumbent_value,
            mode="minimize", tolerance=0.02,
            candidate_samples=run.candidate_samples,
            incumbent_samples=run.incumbent_samples,
            pairs=run.pairs,
        )
        paired_verdict = PAIRED_DIFFERENCE.decide(paired)
        assert paired_verdict.improved is False
        assert paired_verdict.paired is True
        assert paired_verdict.reason

    def test_a_real_win_still_survives_the_drift(self):
        """Pairing must remove the bias without removing the signal."""
        truth = {INCUMBENT: 100.0, CANDIDATE: 90.0}       # a genuine 10% win
        machine = _DriftingMachine(truth, self.DRIFT, jitter=0.05, seed=3)
        run = run_interleaved(machine, plan_blocks(12, blocks=self.PAIRS, lead_in=False))
        verdict = PAIRED_DIFFERENCE.decide(Comparison(
            candidate=run.candidate_value, incumbent=run.incumbent_value,
            mode="minimize", tolerance=0.02, pairs=run.pairs,
        ))
        assert verdict.improved is True
        assert verdict.p_value is not None and verdict.p_value < 0.05

    def test_pairing_reduces_the_dispersion_that_must_be_overcome(self):
        """The sensitivity claim, on a machine with correlated (common-mode) noise.

        Both arms share a slow wander; only a small part of the spread is
        arm-specific. Unpaired, the test fights the whole wander; paired, it
        fights what is left after the wander subtracts out.
        """
        rng = random.Random(99)
        n = 8
        common = [rng.gauss(0.0, 6.0) for _ in range(n)]   # shared wander, ~6%
        pairs: list[tuple[float, float]] = []
        for c in common:
            inc = 100.0 + c + rng.gauss(0.0, 0.5)
            cand = 100.0 + c + rng.gauss(0.0, 0.5)
            pairs.append((cand, inc))
        run = PairedRun(
            pairs=pairs,
            candidate_value=statistics.median([p[0] for p in pairs]),
            incumbent_value=statistics.median([p[1] for p in pairs]),
            candidate_blocks=[p[0] for p in pairs],
            incumbent_blocks=[p[1] for p in pairs],
            candidate_samples=[], incumbent_samples=[],
            candidate_bench={}, incumbent_bench={},
            order=interleave_order(n), spawns=[], lead_in=[],
            plan=plan_blocks(16, blocks=n, lead_in=False), wall_s=0.0,
        )
        arm_cv = run.arm_cv(INCUMBENT)
        paired_cv = run.paired_cv
        assert arm_cv is not None and paired_cv is not None
        assert arm_cv > 0.04                # the machine really is noisy
        assert paired_cv < arm_cv / 4       # and pairing removes most of it


class TestBlockPlan:
    def test_defaults_are_even_and_large_enough_for_the_test_to_reject(self):
        assert DEFAULT_BLOCKS % 2 == 0
        assert DEFAULT_BLOCKS >= MIN_BLOCKS
        assert 0.5 ** MIN_BLOCKS < 0.05

    def test_block_counts_the_test_could_never_reject_are_refused(self):
        with pytest.raises(ValueError, match="could never reject"):
            plan_blocks(20, blocks=4)

    def test_repeats_are_spread_across_blocks_when_the_task_asks_for_enough(self):
        plan = plan_blocks(20, blocks=5)
        assert plan.repeats_per_block == 4
        assert plan.measured_repeats_per_arm == 20
        # Same measured work per arm as the unpaired path, on two arms: 2x.
        assert plan.cost_ratio == pytest.approx(2.0)

    def test_a_block_floor_applies_when_the_task_asks_for_very_few_repeats(self):
        """matmul asks for 3 repeats; 6 blocks of 1 would be mostly warmup."""
        plan = plan_blocks(3, blocks=6)
        assert plan.repeats_per_block == 2
        assert plan.measured_repeats_per_arm == 12
        # And the plan says so rather than letting a caller assume 2x.
        assert plan.cost_ratio == pytest.approx(8.0)

    def test_spawn_count_includes_the_lead_in(self):
        assert plan_blocks(20, blocks=6, lead_in=True).spawns == 14
        assert plan_blocks(20, blocks=6, lead_in=False).spawns == 12

    def test_plan_reports_its_own_drift_imbalance(self):
        assert plan_blocks(20, blocks=6).imbalance == 0.0
        assert plan_blocks(20, blocks=6, counterbalanced=False).imbalance == 1.0

    def test_explicit_repeats_per_block_wins(self):
        assert plan_blocks(20, blocks=5, repeats_per_block=7).repeats_per_block == 7

    def test_zero_total_repeats_reports_infinite_cost_rather_than_dividing_by_zero(self):
        assert math.isinf(BlockPlan(6, 2, 0, 0, True).cost_ratio)


class TestRunInterleaved:
    def _machine(self, **kw) -> _DriftingMachine:
        return _DriftingMachine({INCUMBENT: 100.0, CANDIDATE: 95.0}, 0.0, **kw)

    def test_spawn_sequence_is_the_planned_order(self):
        machine = self._machine()
        run = run_interleaved(machine, plan_blocks(12, blocks=6, lead_in=False))
        assert machine.calls == list("ABBAABBAABBA")
        assert run.order == machine.calls

    def test_lead_in_primes_both_arms_and_is_excluded_from_the_pairs(self):
        machine = self._machine()
        run = run_interleaved(machine, plan_blocks(12, blocks=6, lead_in=True))
        assert machine.calls[:2] == [INCUMBENT, CANDIDATE]
        assert len(run.lead_in) == 2
        assert len(run.spawns) == 12
        assert len(run.pairs) == 6
        assert all(s.position < 0 for s in run.lead_in)

    def test_pairs_are_candidate_first(self):
        run = run_interleaved(self._machine(), plan_blocks(12, blocks=6, lead_in=False))
        assert all(cand == 95.0 and inc == 100.0 for cand, inc in run.pairs)


    def test_point_estimates_are_block_medians(self):
        run = run_interleaved(self._machine(), plan_blocks(12, blocks=6, lead_in=False))
        assert run.candidate_value == 95.0
        assert run.incumbent_value == 100.0

    def test_per_repeat_samples_are_flattened_in_spawn_order(self):
        run = run_interleaved(self._machine(), plan_blocks(12, blocks=6, lead_in=False))
        assert run.candidate_samples == [95.0] * 6
        assert run.incumbent_samples == [100.0] * 6

    def test_a_failed_spawn_propagates_rather_than_salvaging_a_broken_design(self):
        """A truncated ABBA sequence is no longer counterbalanced."""
        calls = {"n": 0}

        def flaky(arm: str) -> SpawnMeasurement:
            calls["n"] += 1
            if calls["n"] == 5:
                raise RuntimeError("benchmark exited with code 1")
            return SpawnMeasurement(value=1.0)

        with pytest.raises(RuntimeError, match="exited with code 1"):
            run_interleaved(flaky, plan_blocks(12, blocks=6, lead_in=False))

    def test_wall_clock_is_recorded_per_spawn_and_overall(self):
        run = run_interleaved(self._machine(), plan_blocks(12, blocks=6, lead_in=False))
        assert run.wall_s >= 0.0
        assert len(run.spawns) == 12
        assert all(s.wall_s >= 0.0 for s in run.spawns)

    def test_paired_cv_is_none_without_enough_pairs(self):
        run = run_interleaved(self._machine(), plan_blocks(12, blocks=6, lead_in=False))
        assert run.paired_cv == pytest.approx(0.0)
        assert run.arm_cv(CANDIDATE) == pytest.approx(0.0)


class TestFirstBlockPenalty:
    """The cold-start audit: is the lead-in earning its two spawns?"""

    def _run_with_cold_first(self, penalty: float) -> PairedRun:
        state = {"n": 0}

        def spawn(arm: str) -> SpawnMeasurement:
            # First pair (2 spawns) measures `penalty` higher on both arms.
            scale = 1.0 + penalty if state["n"] < 2 else 1.0
            state["n"] += 1
            base = 100.0 if arm == INCUMBENT else 95.0
            return SpawnMeasurement(value=base * scale)

        return run_interleaved(spawn, plan_blocks(12, blocks=6, lead_in=False))

    def test_detects_a_cold_first_block(self):
        run = self._run_with_cold_first(0.10)
        penalty = first_block_penalty(run)
        assert penalty is not None and penalty == pytest.approx(0.10)

    def test_reports_nothing_when_the_machine_is_already_warm(self):
        run = self._run_with_cold_first(0.0)
        assert first_block_penalty(run) == pytest.approx(0.0)


class TestRunPairedBenchmark:
    """The real-subprocess entry point, with run_benchmark stubbed out."""

    def _stub(self, monkeypatch, tmp_path):
        calls: list[dict] = []

        def fake_run_benchmark(cmd, cwd, **kwargs):
            calls.append({"cmd": cmd, "cwd": cwd, **kwargs})
            value = 100.0 if cwd.name == "a" else 95.0
            return None, {
                "ok": True,
                "latency_ms": {"p50": value, "raw_values": [value, value]},
            }

        monkeypatch.setattr(paired_mod, "run_benchmark", fake_run_benchmark)
        arm_a = tmp_path / "a"
        arm_b = tmp_path / "b"
        arm_a.mkdir()
        arm_b.mkdir()
        return calls, arm_a, arm_b

    def _run(self, monkeypatch, tmp_path, **kw):
        calls, arm_a, arm_b = self._stub(monkeypatch, tmp_path)
        run = run_paired_benchmark(
            "python bench.py", incumbent_cwd=arm_a, candidate_cwd=arm_b,
            metric_name="latency_ms.p50", total_repeats=12, warmup=2, **kw,
        )
        return calls, run

    def test_alternates_between_the_two_workspaces_in_abba_order(self, monkeypatch, tmp_path):
        calls, run = self._run(monkeypatch, tmp_path, lead_in=False)
        assert [c["cwd"].name for c in calls] == list("abbaabbaabba")
        assert run.candidate_value == 95.0
        assert run.incumbent_value == 100.0

    def test_block_size_is_forced_past_a_session_env_override(self, monkeypatch, tmp_path):
        """PERFLAB_BENCH_REPEATS outranks run_benchmark's `repeats` argument.

        That precedence is right for an ordinary run and catastrophic here: a
        stray env var would turn each block into a full-length benchmark and
        multiply the wall clock by the block count. The runner passes the block
        size through `env`, which run_benchmark seeds before every setdefault.
        """
        monkeypatch.setenv("PERFLAB_BENCH_REPEATS", "99")
        calls, _ = self._run(monkeypatch, tmp_path, lead_in=False)
        assert all(c["env"]["PERFLAB_BENCH_REPEATS"] == "2" for c in calls)

    def test_both_arms_get_identical_benchmark_settings(self, monkeypatch, tmp_path):
        """Any asymmetry here would be a confound the pairing cannot remove."""
        calls, _ = self._run(monkeypatch, tmp_path, lead_in=False)
        shape = {
            k: v for c in calls[:1] for k, v in c.items() if k != "cwd"
        }
        for c in calls:
            assert {k: v for k, v in c.items() if k != "cwd"} == shape

    def test_isolation_and_limits_are_forwarded_like_any_other_benchmark(
        self, monkeypatch, tmp_path,
    ):
        calls, _ = self._run(
            monkeypatch, tmp_path, lead_in=False,
            program_type="cuda", rlimit_as_gb=8.0,
            env_passthrough=["OMP_NUM_THREADS"],
        )
        assert calls[0]["program_type"] == "cuda"
        assert calls[0]["rlimit_as_gb"] == 8.0
        assert calls[0]["env_passthrough"] == ["OMP_NUM_THREADS"]
        assert calls[0]["fast_mode"] is False

    def test_on_spawn_reports_progress_for_every_measured_spawn(self, monkeypatch, tmp_path):
        seen: list[tuple[str, int, int]] = []
        calls, arm_a, arm_b = self._stub(monkeypatch, tmp_path)
        run_paired_benchmark(
            "python bench.py", incumbent_cwd=arm_a, candidate_cwd=arm_b,
            metric_name="latency_ms.p50", total_repeats=12, lead_in=False,
            on_spawn=lambda arm, block, pos: seen.append((arm, block, pos)),
        )
        assert len(seen) == 12
        assert [s[0] for s in seen] == list("ABBAABBAABBA")

    def test_bench_json_is_kept_for_contract_validation(self, monkeypatch, tmp_path):
        _, run = self._run(monkeypatch, tmp_path, lead_in=False)
        assert all(s.measurement.bench["ok"] is True for s in run.spawns)


class TestSignConventionEndToEnd:
    """The failure mode this class exists for is silent and total.

    ``Comparison.pairs`` is ``(candidate_i, incumbent_i)`` and
    ``Comparison.paired_diffs`` sign-normalises so positive always means the
    candidate is better -- in BOTH modes. If ``paired.py`` populated the tuples
    the other way round, every accept decision would invert: regressions would
    be promoted and wins discarded, and *every other test in this repo would
    still pass*, because the runner and the rule would agree with each other
    while both being backwards. So the check has to start from the thing that
    is unambiguous -- which workspace directory the measurement came from --
    and end at the accept/reject verdict.
    """

    BETTER_MINIMIZE = 80.0    # lower is better
    WORSE_MINIMIZE = 100.0
    BETTER_MAXIMIZE = 120.0   # higher is better
    WORSE_MAXIMIZE = 100.0

    def _run_by_directory(self, monkeypatch, tmp_path, *, value_by_dirname):
        """Interleave two real directories; each reports the value keyed by name."""
        def fake_run_benchmark(cmd, cwd, **kwargs):
            value = value_by_dirname[cwd.name]
            return None, {"ok": True, "latency_ms": {"p50": value}}

        monkeypatch.setattr(paired_mod, "run_benchmark", fake_run_benchmark)
        old = tmp_path / "old_code"
        new = tmp_path / "new_code"
        old.mkdir()
        new.mkdir()
        return run_paired_benchmark(
            "python bench.py",
            incumbent_cwd=old,          # the thing being defended
            candidate_cwd=new,          # the thing being proposed
            metric_name="latency_ms.p50", total_repeats=12, lead_in=False,
        )

    def test_tuple_order_matches_the_workspace_each_value_came_from(
        self, monkeypatch, tmp_path,
    ):
        """pairs[i] == (value measured in candidate_cwd, value in incumbent_cwd)."""
        run = self._run_by_directory(
            monkeypatch, tmp_path,
            value_by_dirname={"old_code": 100.0, "new_code": 80.0},
        )
        assert all(pair == (80.0, 100.0) for pair in run.pairs)
        assert run.candidate_value == 80.0     # came from candidate_cwd
        assert run.incumbent_value == 100.0    # came from incumbent_cwd

    @pytest.mark.parametrize("mode", ["minimize", "maximize"])
    def test_a_genuinely_better_candidate_is_accepted_in_both_modes(
        self, monkeypatch, tmp_path, mode,
    ):
        better, worse = (
            (self.BETTER_MINIMIZE, self.WORSE_MINIMIZE) if mode == "minimize"
            else (self.BETTER_MAXIMIZE, self.WORSE_MAXIMIZE)
        )
        run = self._run_by_directory(
            monkeypatch, tmp_path,
            value_by_dirname={"old_code": worse, "new_code": better},
        )
        # positive == better, in both modes, straight off Comparison.
        comparison = Comparison(
            candidate=run.candidate_value, incumbent=run.incumbent_value,
            mode=mode, tolerance=0.02, pairs=run.pairs,
        )
        assert all(d > 0 for d in comparison.paired_diffs)
        assert comparison.observed > 0
        verdict = PAIRED_DIFFERENCE.decide(comparison)
        assert verdict.improved is True
        assert verdict.p_value == pytest.approx(2 ** -6)

    @pytest.mark.parametrize("mode", ["minimize", "maximize"])
    def test_a_regression_is_rejected_in_both_modes(
        self, monkeypatch, tmp_path, mode,
    ):
        """The same two programs with the roles swapped must flip the verdict.

        This is the assertion a transposed tuple cannot survive: nothing about
        the numbers changed, only which directory is the incumbent.
        """
        better, worse = (
            (self.BETTER_MINIMIZE, self.WORSE_MINIMIZE) if mode == "minimize"
            else (self.BETTER_MAXIMIZE, self.WORSE_MAXIMIZE)
        )
        run = self._run_by_directory(
            monkeypatch, tmp_path,
            # The candidate is now the SLOWER program.
            value_by_dirname={"old_code": better, "new_code": worse},
        )
        comparison = Comparison(
            candidate=run.candidate_value, incumbent=run.incumbent_value,
            mode=mode, tolerance=0.02, pairs=run.pairs,
        )
        assert all(d < 0 for d in comparison.paired_diffs)
        assert comparison.observed < 0
        verdict = PAIRED_DIFFERENCE.decide(comparison)
        assert verdict.improved is False
        assert verdict.p_value == 1.0

    @pytest.mark.parametrize("mode", ["minimize", "maximize"])
    def test_the_raw_metric_direction_is_not_baked_into_the_runner(
        self, monkeypatch, tmp_path, mode,
    ):
        """``PairedRun.diffs`` stays in the metric's own units and sign.

        Mode-awareness belongs to Comparison.paired_diffs and nowhere else --
        if the runner ever "helpfully" normalised too, the two would compose
        into a double negation.
        """
        run = self._run_by_directory(
            monkeypatch, tmp_path,
            value_by_dirname={"old_code": 100.0, "new_code": 80.0},
        )
        # Raw: candidate minus incumbent, always, regardless of mode.
        assert all(d == pytest.approx(-20.0) for d in run.diffs)
        normalised = Comparison(
            candidate=80.0, incumbent=100.0, mode=mode, tolerance=0.02,
            pairs=run.pairs,
        ).paired_diffs
        expected = 20.0 if mode == "minimize" else -20.0
        assert all(d == pytest.approx(expected) for d in normalised)
