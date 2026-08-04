"""Tests for perflab.analyzers.decision — the single accept/reject module.

"Is B better than A?" used to be answered in three places (the knob search in
orchestrator.py, the agent beam search in optimizers/phases/evaluate.py, and
the regression check in ci.py) with three different behaviors. These tests pin
the shared module those call sites now depend on: the strategy protocol, the
two rules, rule selection from a task's constraints, and the fact that a paired
rule can be added later without reshaping anything.

The statistical content of NonOverlappingCI is covered by tests/test_noise_gate.py
(which exercises it through the metrics_rollup compatibility wrapper); this file
covers the abstraction around it.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perflab.analyzers.bench_stats import compute_bench_stats
from perflab.analyzers.decision import (
    DEFAULT_RULE,
    NON_OVERLAPPING_CI,
    SCREENING_RULE,
    TOLERANCE_ONLY,
    Comparison,
    ImprovementVerdict,
    NonOverlappingCI,
    ToleranceOnly,
    decide,
    get_rule,
    register_rule,
    rule_for_constraints,
    rule_names,
)
from perflab.task_spec import ContractSpec


def _spread(center: float, cv: float, n: int = 5) -> list[float]:
    """n samples centered on `center` with (approximately) the requested CV."""
    if n < 2:
        raise ValueError("need at least 2 samples")
    step = 2.0 / (n - 1)
    raw = [(-1.0 + i * step) for i in range(n)]
    stats = compute_bench_stats([center * (1 + cv * r) for r in raw])
    assert stats is not None
    scale = cv / stats.cv
    return [center * (1 + cv * scale * r) for r in raw]


def _within_noise() -> Comparison:
    """A 5% "win" measured on a machine with 8% run-to-run spread.

    Clears the 2% materiality floor; nowhere near resolvable. This is the
    comparison the two rules must disagree about — if they ever agree here the
    strategy abstraction has stopped meaning anything.
    """
    incumbent = _spread(100.0, 0.08)
    return Comparison(
        candidate=105.0, incumbent=100.0, mode="maximize", tolerance=0.02,
        candidate_samples=[v * 1.05 for v in incumbent], incumbent_samples=incumbent,
    )


class TestRulesDisagree:
    """The whole point of the abstraction: the rules are genuinely different."""

    def test_non_overlapping_ci_rejects_what_tolerance_only_accepts(self):
        comparison = _within_noise()
        assert NON_OVERLAPPING_CI.decide(comparison).improved is False
        assert TOLERANCE_ONLY.decide(comparison).improved is True

    def test_tolerance_only_never_claims_to_have_verified_anything(self):
        """It ignores the samples it was handed, and says so."""
        verdict = TOLERANCE_ONLY.decide(_within_noise())
        assert verdict.verified is False
        assert verdict.cv is None and verdict.n is None
        assert verdict.noise_required is None

    def test_both_rules_enforce_the_materiality_floor(self):
        """A rule may demand more than the tolerance, never less."""
        incumbent = _spread(100.0, 0.001, n=20)
        comparison = Comparison(
            candidate=100.5, incumbent=100.0, mode="maximize", tolerance=0.02,
            candidate_samples=[v * 1.005 for v in incumbent],
            incumbent_samples=incumbent,
        )
        for rule in (TOLERANCE_ONLY, NON_OVERLAPPING_CI):
            verdict = rule.decide(comparison)
            assert verdict.improved is False, rule.name
            assert verdict.beats_tolerance is False, rule.name
            assert "regression tolerance" in verdict.reason, rule.name

    def test_real_win_accepted_by_both(self):
        incumbent = _spread(100.0, 0.08)
        comparison = Comparison(
            candidate=1000.0, incumbent=100.0, mode="maximize", tolerance=0.02,
            candidate_samples=_spread(1000.0, 0.08), incumbent_samples=incumbent,
        )
        assert NON_OVERLAPPING_CI.decide(comparison).improved is True
        assert TOLERANCE_ONLY.decide(comparison).improved is True

    def test_verdict_records_which_rule_decided(self):
        comparison = _within_noise()
        assert NON_OVERLAPPING_CI.decide(comparison).rule == "non_overlapping_ci"
        assert TOLERANCE_ONLY.decide(comparison).rule == "tolerance_only"

    def test_missing_samples_degrade_visibly_not_silently(self):
        """No samples: the historical ratio, flagged unverified — by both rules."""
        comparison = Comparison(
            candidate=105.0, incumbent=100.0, mode="maximize", tolerance=0.02,
        )
        for rule in (TOLERANCE_ONLY, NON_OVERLAPPING_CI):
            verdict = rule.decide(comparison)
            assert verdict.improved is True, rule.name
            assert verdict.verified is False, rule.name
            assert verdict.rule == rule.name


class TestComparison:
    def test_observed_is_signed_relative_improvement(self):
        maximize = Comparison(110.0, 100.0, "maximize", 0.02)
        minimize = Comparison(90.0, 100.0, "minimize", 0.02)
        assert maximize.observed == pytest.approx(0.10)
        assert minimize.observed == pytest.approx(0.10)
        assert Comparison(90.0, 100.0, "maximize", 0.02).observed == pytest.approx(-0.10)

    def test_paired_diffs_are_sign_normalized_in_both_modes(self):
        """d_i > 0 always means "the candidate did better on that pair"."""
        pairs = [(9.0, 10.0), (8.0, 10.0)]
        assert Comparison(8.5, 10.0, "minimize", 0.02, pairs=pairs).paired_diffs == [1.0, 2.0]
        assert Comparison(8.5, 10.0, "maximize", 0.02, pairs=pairs).paired_diffs == [-1.0, -2.0]

    def test_is_paired_needs_at_least_two_pairs(self):
        assert Comparison(1.0, 2.0, "maximize", 0.02).is_paired is False
        assert Comparison(1.0, 2.0, "maximize", 0.02, pairs=[(1.0, 2.0)]).is_paired is False
        assert Comparison(
            1.0, 2.0, "maximize", 0.02, pairs=[(1.0, 2.0), (1.1, 2.0)],
        ).is_paired is True

    def test_existing_rules_ignore_pairs(self):
        """`pairs` is inert until a rule that consumes it lands."""
        without = Comparison(105.0, 100.0, "maximize", 0.02)
        with_pairs = Comparison(
            105.0, 100.0, "maximize", 0.02,
            pairs=[(105.0, 100.0), (105.0, 100.0)],
        )
        for rule in (TOLERANCE_ONLY, NON_OVERLAPPING_CI):
            assert rule.decide(without) == rule.decide(with_pairs)


class _SignTestPaired:
    """Stand-in for the planned PairedDifference rule.

    Deliberately NOT the real statistics — it is a sign test on ``d_i``, which
    is enough to prove that a paired rule fits the protocol as it stands: it
    reads ``Comparison.pairs``, reports ``paired=True`` and a ``p_value``, and
    needs no new field, no signature change, and no edit to any call site.
    """

    name = "sign_test_paired"

    def decide(self, comparison: Comparison) -> ImprovementVerdict:
        diffs = comparison.paired_diffs
        wins = sum(1 for d in diffs if d > 0)
        p_value = 0.5 ** len(diffs) if wins == len(diffs) and diffs else 1.0
        beats_tolerance = comparison.observed > comparison.tolerance
        improved = bool(diffs) and beats_tolerance and p_value < 0.05
        return ImprovementVerdict(
            improved=improved,
            beats_tolerance=beats_tolerance,
            observed=comparison.observed,
            required=comparison.tolerance,
            tolerance=comparison.tolerance,
            noise_required=None,
            cv=None,
            n=len(diffs) or None,
            verified=bool(diffs),
            reason="" if improved else f"paired sign test p={p_value:.3f}",
            rule=self.name,
            paired=True,
            p_value=p_value,
        )


class TestPairedRuleFits:
    """A paired rule must slot in without reshaping the protocol or verdict."""

    def test_paired_rule_decides_through_the_same_entry_point(self):
        comparison = Comparison(
            candidate=90.0, incumbent=100.0, mode="minimize", tolerance=0.02,
            pairs=[(91.0, 100.0), (89.0, 99.0), (90.0, 101.0),
                   (88.0, 98.0), (92.0, 102.0), (90.0, 100.0)],
        )
        verdict = decide(comparison, _SignTestPaired())
        assert verdict.improved is True
        assert verdict.paired is True
        assert verdict.p_value is not None and verdict.p_value < 0.05
        assert verdict.rule == "sign_test_paired"

    def test_paired_rule_is_registrable_and_selectable_by_name(self):
        rule = _SignTestPaired()
        register_rule(rule)
        try:
            assert get_rule("sign_test_paired") is rule
            assert "sign_test_paired" in rule_names()
            constraints = SimpleNamespace(noise_gate=True, decision_rule="sign_test_paired")
            assert rule_for_constraints(constraints) is rule
        finally:
            from perflab.analyzers import decision as decision_mod
            decision_mod._RULES.pop("sign_test_paired", None)

    def test_interval_rules_leave_the_paired_fields_alone(self):
        verdict = NON_OVERLAPPING_CI.decide(_within_noise())
        assert verdict.paired is False
        assert verdict.p_value is None


class TestRuleRegistry:
    def test_defaults(self):
        assert DEFAULT_RULE is NON_OVERLAPPING_CI
        assert SCREENING_RULE is TOLERANCE_ONLY
        assert rule_names() == [
            "non_overlapping_ci", "paired_difference", "tolerance_only",
        ]

    def test_get_rule_by_name(self):
        assert get_rule("non_overlapping_ci") is NON_OVERLAPPING_CI
        assert get_rule("tolerance_only") is TOLERANCE_ONLY

    def test_unknown_rule_names_the_valid_options(self):
        with pytest.raises(ValueError, match="unknown decision_rule 'wilcoxon'"):
            get_rule("wilcoxon")
        with pytest.raises(
            ValueError,
            match="non_overlapping_ci, paired_difference, tolerance_only",
        ):
            get_rule("wilcoxon")

    def test_decide_defaults_to_the_gated_rule(self):
        assert decide(_within_noise()).improved is False
        assert decide(_within_noise(), TOLERANCE_ONLY).improved is True

    def test_rules_are_stateless_and_reusable(self):
        """Singletons are shared across call sites; decisions must not drift."""
        comparison = _within_noise()
        first = NON_OVERLAPPING_CI.decide(comparison)
        assert NON_OVERLAPPING_CI.decide(comparison) == first
        assert NonOverlappingCI().decide(comparison) == first
        assert ToleranceOnly().decide(comparison) == TOLERANCE_ONLY.decide(comparison)


class TestRuleForConstraints:
    def test_default_when_unconfigured(self):
        assert rule_for_constraints(SimpleNamespace()) is DEFAULT_RULE

    def test_noise_gate_false_still_means_tolerance_only(self):
        constraints = SimpleNamespace(noise_gate=False, decision_rule=None)
        assert rule_for_constraints(constraints) is TOLERANCE_ONLY

    def test_named_rule_is_honored(self):
        constraints = SimpleNamespace(noise_gate=True, decision_rule="tolerance_only")
        assert rule_for_constraints(constraints) is TOLERANCE_ONLY

    def test_noise_gate_false_wins_over_decision_rule(self):
        """The explicit off switch is not quietly overridden by the other knob."""
        constraints = SimpleNamespace(noise_gate=False, decision_rule="non_overlapping_ci")
        assert rule_for_constraints(constraints) is TOLERANCE_ONLY

    def test_non_string_decision_rule_falls_back_to_default(self):
        """Duck-typed constraint objects (test doubles) must not explode here;
        real typos are caught at task.yaml load time instead."""
        assert rule_for_constraints(SimpleNamespace(decision_rule=object())) is DEFAULT_RULE
        assert rule_for_constraints(SimpleNamespace(decision_rule="")) is DEFAULT_RULE


class TestTaskSpecDecisionRule:
    _BASE = (
        "name: t\n"
        "program_type: python\n"
        "correctness:\n  cmd: 'python tests.py'\n"
        "benchmark:\n"
        "  cmd: 'python bench.py --json out/bench.json'\n"
        "  metric:\n    name: 'tflops.median'\n    mode: maximize\n"
        "edit_policy:\n  allowed_paths: ['a.py']\n"
    )

    def _load(self, tmp_path: Path, constraints: str):
        from perflab.task_spec import TaskSpec

        (tmp_path / "task.yaml").write_text(
            self._BASE + constraints, encoding="utf-8",
        )
        return TaskSpec.load(tmp_path / "task.yaml")

    def test_default_is_none(self, tmp_path: Path):
        task = self._load(tmp_path, "constraints:\n  max_iters: 1\n")
        assert task.constraints.decision_rule is None
        assert rule_for_constraints(task.constraints) is DEFAULT_RULE

    def test_parsed_and_resolved(self, tmp_path: Path):
        task = self._load(tmp_path, "constraints:\n  decision_rule: tolerance_only\n")
        assert task.constraints.decision_rule == "tolerance_only"
        assert rule_for_constraints(task.constraints) is TOLERANCE_ONLY

    def test_typo_fails_at_load_not_at_the_first_decision(self, tmp_path: Path):
        with pytest.raises(ValueError, match="unknown decision_rule"):
            self._load(tmp_path, "constraints:\n  decision_rule: non_overlaping_ci\n")


# --- The agent beam search honors the configured rule ------------------------


class _NoOpEventLog:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _make_ctx(tmp_path: Path, best_value: float, messages: list[str], **overrides):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    constraints = SimpleNamespace(
        regression_tolerance=overrides.get("tol", 0.02),
        noise_gate=overrides.get("noise_gate", True),
        decision_rule=overrides.get("decision_rule", None),
        cv_threshold=None,
        rlimit_as_gb=None,
        env_passthrough=[],
    )
    return SimpleNamespace(
        task=SimpleNamespace(
            benchmark=SimpleNamespace(
                metric=SimpleNamespace(name="throughput.median", mode="maximize"),
                cmd="python bench.py", warmup=1, repeats=5,
            ),
            build=None,
            program_type="python",
            contract=ContractSpec(),
            constraints=constraints,
            anti_gaming=SimpleNamespace(gaming_speedup_threshold=1000.0),
            out_dir=ws / "out",
        ),
        ws=ws,
        rp=SimpleNamespace(run_dir=run_dir),
        iteration=1,
        progress=SimpleNamespace(on_message=messages.append),
        event_log=_NoOpEventLog(),
        history=[],
        baseline_val=best_value,
        best_value=best_value,
        best_iter=0,
        accepted_patches=[],
        accepted_count=0,
        sec_metric=None,
        config=SimpleNamespace(isolation=None, top_k=3),
    )


def _accept(ctx, value: float, samples: list[float]) -> bool:
    from perflab.optimizers.phases import evaluate as evaluate_mod

    cand = evaluate_mod.BeamCandidate(
        iteration=1, index=0, blocks=[], description="candidate 1",
        value=value, samples=samples,
    )
    with patch.object(evaluate_mod, "snapshot_workspace", lambda *a, **k: None):
        accepted, _, _ = evaluate_mod.accept_best(ctx, [cand], use_fast=False)
    return accepted


class TestCIRegressionUsesTheSharedModule:
    """ci.py asks the same module, with the mode flipped."""

    def _assess(self, rule=None):
        from perflab.ci import _assess_regression

        baseline = _spread(100.0, 0.08)
        kwargs = {} if rule is None else {"rule": rule}
        # 5% DROP in a maximize metric, measured on an 8% machine.
        return _assess_regression(
            95.0, 100.0, "maximize", 0.02,
            current_samples=[v * 0.95 for v in baseline],
            baseline_samples=baseline,
            **kwargs,
        )

    def test_mode_flip_is_preserved(self):
        """A regression is an improvement measured the other way — the drop
        must be judged against the improving direction, not the metric's."""
        regression_pct, regressed, verdict = self._assess()
        assert regression_pct == pytest.approx(5.0)
        assert verdict is not None
        assert verdict.observed == pytest.approx(0.05)  # 5% "better" at being worse

    def test_default_rule_does_not_flag_a_drop_inside_the_noise(self):
        _, regressed, verdict = self._assess()
        assert regressed is False
        assert verdict is not None and verdict.beats_tolerance is True
        assert "within noise" in verdict.reason

    def test_tolerance_only_rule_flags_the_same_drop(self):
        _, regressed, verdict = self._assess(rule=TOLERANCE_ONLY)
        assert regressed is True
        assert verdict is not None and verdict.rule == "tolerance_only"

    def test_no_samples_is_the_historical_ratio_test(self):
        from perflab.ci import _check_regression

        assert _check_regression(0.90, 1.0, "maximize", 0.05) == (pytest.approx(10.0), True)
        assert _check_regression(0.97, 1.0, "maximize", 0.05)[1] is False


class TestAcceptBestUsesConfiguredRule:
    """accept_best must read the rule from constraints, not hardcode one."""

    def _run(self, tmp_path: Path, **overrides) -> bool:
        ctx = _make_ctx(tmp_path, best_value=100.0, messages=[], **overrides)
        incumbent = _spread(100.0, 0.08)
        (ctx.rp.run_dir / "bench.json").write_text(
            json.dumps({"ok": True, "throughput": {
                "median": 100.0, "raw_values": incumbent,
            }}),
            encoding="utf-8",
        )
        return _accept(ctx, 105.0, [v * 1.05 for v in incumbent])

    def test_default_rule_rejects_within_noise(self, tmp_path: Path):
        assert self._run(tmp_path) is False

    def test_tolerance_only_rule_accepts_within_noise(self, tmp_path: Path):
        assert self._run(tmp_path, decision_rule="tolerance_only") is True

    def test_noise_gate_false_still_accepts_within_noise(self, tmp_path: Path):
        assert self._run(tmp_path, noise_gate=False) is True
