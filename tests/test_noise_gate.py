"""Tests for the statistical accept gate.

Before this, `is_improvement` was a bare ratio (`new > best * (1 + tol)`) and
the variance statistics computed by `analyzers/bench_stats.py` were consumed
only by the report writer -- so on a noisy machine the optimizer would accept
run-to-run jitter as a win, over and over, and the beam search would chase it.
These tests pin the two directions that matter: a real win survives noisy
measurements, and a small "win" inside the noise does not.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from perflab.analyzers.bench_stats import (
    compute_bench_stats,
    cv_budget_for_gate,
    extract_repeated_values,
    relative_ci_margin,
    repeats_needed_for_gate,
)
from perflab.analyzers.metrics_rollup import assess_improvement, is_improvement
from perflab.task_spec import ContractSpec


def _spread(center: float, cv: float, n: int = 5) -> list[float]:
    """n samples centered on `center` with (approximately) the requested CV.

    Uses a symmetric linear spread; compute_bench_stats reports the exact CV,
    which the tests assert against rather than assuming.
    """
    if n < 2:
        raise ValueError("need at least 2 samples")
    step = 2.0 / (n - 1)
    raw = [(-1.0 + i * step) for i in range(n)]
    stats = compute_bench_stats([center * (1 + cv * r) for r in raw])
    assert stats is not None
    scale = cv / stats.cv
    return [center * (1 + cv * scale * r) for r in raw]


class TestBackwardCompatibility:
    """Old four-positional-argument calls must behave exactly as before."""

    def test_maximize_ratio_unchanged(self):
        assert is_improvement(1.1, 1.0, "maximize", 0.01) is True
        assert is_improvement(1.005, 1.0, "maximize", 0.01) is False

    def test_minimize_ratio_unchanged(self):
        assert is_improvement(0.9, 1.0, "minimize", 0.01) is True
        assert is_improvement(0.995, 1.0, "minimize", 0.01) is False

    def test_no_samples_marks_verdict_unverified(self):
        v = assess_improvement(1.1, 1.0, "maximize", 0.01)
        assert v.improved is True
        assert v.verified is False           # caller can log the unchecked path
        assert v.noise_required is None and v.cv is None

    def test_empty_sample_lists_are_the_no_samples_path(self):
        v = assess_improvement(1.1, 1.0, "maximize", 0.01,
                               new_samples=[], best_samples=[])
        assert v.improved is True and v.verified is False


class TestNoiseRejection:
    def test_one_percent_win_under_eight_percent_cv_is_rejected(self):
        """The headline case: 1% "win", 8% CV. Old code accepted this."""
        best = _spread(100.0, 0.08)
        cand = [v * 1.01 for v in best]
        v = assess_improvement(101.0, 100.0, "maximize", 0.005,
                               new_samples=cand, best_samples=best)
        assert v.improved is False
        assert v.beats_tolerance is True     # it DID clear the ratio gate
        assert v.verified is True
        assert "within noise" in v.reason
        assert "CV=8.0%" in v.reason
        assert v.noise_required is not None and v.noise_required > 0.1

    def test_five_percent_win_over_two_percent_tolerance_still_rejected(self):
        """Distinguishable from a tolerance failure: this clears 2% and dies
        on variance alone -- the exact scenario the old gate got wrong."""
        best = _spread(100.0, 0.08)
        cand = [v * 1.05 for v in best]
        v = assess_improvement(105.0, 100.0, "maximize", 0.02,
                               new_samples=cand, best_samples=best)
        assert v.improved is False and v.beats_tolerance is True
        assert "within noise" in v.reason

    def test_minimize_mode_noise_rejection(self):
        best = _spread(10.0, 0.08)
        cand = [v * 0.98 for v in best]
        v = assess_improvement(9.8, 10.0, "minimize", 0.01,
                               new_samples=cand, best_samples=best)
        assert v.improved is False and v.beats_tolerance is True
        assert "within noise" in v.reason

    def test_tolerance_failure_reason_is_distinct(self):
        best = _spread(100.0, 0.01)
        cand = [v * 1.001 for v in best]
        v = assess_improvement(100.1, 100.0, "maximize", 0.02,
                               new_samples=cand, best_samples=best)
        assert v.improved is False and v.beats_tolerance is False
        assert "regression tolerance" in v.reason
        assert "within noise" not in v.reason


class TestRealWinsSurvive:
    def test_ten_x_win_accepted_despite_very_noisy_samples(self):
        best = _spread(100.0, 0.30)
        cand = _spread(10.0, 0.30)
        v = assess_improvement(10.0, 100.0, "minimize", 0.02,
                               new_samples=cand, best_samples=best)
        assert v.improved is True and v.verified is True

    def test_ten_x_win_accepted_maximize(self):
        best = _spread(1.0, 0.30)
        cand = _spread(10.0, 0.30)
        v = assess_improvement(10.0, 1.0, "maximize", 0.02,
                               new_samples=cand, best_samples=best)
        assert v.improved is True

    def test_modest_win_accepted_when_environment_is_quiet(self):
        """More repeats / less jitter must buy the ability to resolve less."""
        best = _spread(100.0, 0.005, n=20)
        cand = [v * 1.03 for v in best]
        v = assess_improvement(103.0, 100.0, "maximize", 0.02,
                               new_samples=cand, best_samples=best)
        assert v.improved is True
        assert v.noise_required is not None and v.noise_required < 0.03

    def test_more_repeats_lower_the_resolvable_effect(self):
        few = assess_improvement(
            105.0, 100.0, "maximize", 0.02,
            new_samples=_spread(105.0, 0.05, n=3), best_samples=_spread(100.0, 0.05, n=3),
        )
        many = assess_improvement(
            105.0, 100.0, "maximize", 0.02,
            new_samples=_spread(105.0, 0.05, n=25), best_samples=_spread(100.0, 0.05, n=25),
        )
        assert few.noise_required is not None and many.noise_required is not None
        assert many.noise_required < few.noise_required
        assert few.improved is False and many.improved is True


class TestGateDegradation:
    def test_one_sided_samples_still_test(self):
        """Candidate samples only: incumbent is treated as a point estimate."""
        v = assess_improvement(101.0, 100.0, "maximize", 0.005,
                               new_samples=_spread(101.0, 0.08))
        assert v.verified is True and v.improved is False

    def test_noise_gate_off_falls_back_to_ratio(self):
        best = _spread(100.0, 0.08)
        cand = [v * 1.01 for v in best]
        v = assess_improvement(101.0, 100.0, "maximize", 0.005,
                               new_samples=cand, best_samples=best,
                               noise_gate=False)
        assert v.improved is True and v.verified is False

    def test_single_sample_is_not_usable(self):
        v = assess_improvement(101.0, 100.0, "maximize", 0.005,
                               new_samples=[101.0], best_samples=[100.0])
        assert v.improved is True and v.verified is False

    def test_zero_incumbent_does_not_crash(self):
        v = assess_improvement(1.0, 0.0, "maximize", 0.02,
                               new_samples=_spread(1.0, 0.05))
        assert v.improved is True and v.verified is False

    def test_zero_candidate_minimize_is_not_noise_tested(self):
        """A degenerate 0.0 latency has no meaningful relative interval; the
        zero-metric gaming warning in the evaluate phase owns that case."""
        v = assess_improvement(0.0, 10.0, "minimize", 0.02,
                               new_samples=[0.0, 0.0], best_samples=_spread(10.0, 0.05))
        assert v.verified is False


class TestExtractRepeatedValues:
    def test_metric_parent_is_primary(self):
        bench = {
            "throughput": {"median": 100.0, "raw_values": [95.0, 100.0, 105.0]},
            "times_ms": [1.0, 2.0, 3.0],
        }
        assert extract_repeated_values(bench, "throughput.median") == [95.0, 100.0, 105.0]

    def test_all_key_under_metric_parent(self):
        """The shape perflab's own task templates emit."""
        bench = {"latency_ms": {"median": 10.0, "all": [9.0, 10.0, 11.0]}}
        assert extract_repeated_values(bench, "latency_ms.median") == [9.0, 10.0, 11.0]

    def test_top_level_times_ms_fallback(self):
        bench = {"tflops": {"median": 5.0}, "times_ms": [10.0, 11.0, 9.0]}
        assert extract_repeated_values(bench, "tflops.median") == [10.0, 11.0, 9.0]

    def test_fallback_for_single_part_metric(self):
        bench = {"value": 42.0, "times_ms": [1.0, 1.1]}
        assert extract_repeated_values(bench, "value") == [1.0, 1.1]

    def test_empty_when_nothing_usable(self):
        assert extract_repeated_values({"throughput": {"median": 100.0}}, "throughput.median") == []
        assert extract_repeated_values({"value": 42.0}, "value") == []
        assert extract_repeated_values({"times_ms": [1.0]}, "x.y") == []  # too few
        assert extract_repeated_values({"times_ms": ["a", "b"]}, "x.y") == []
        assert extract_repeated_values({"times_ms": [1.0, float("nan")]}, "x.y") == []

    def test_non_dict_bench_is_safe(self):
        assert extract_repeated_values([], "a.b") == []  # type: ignore[arg-type]


class TestThresholdCoherence:
    def test_default_gate_needs_a_much_quieter_machine_than_ten_percent(self):
        """The old cv_threshold (10%) was ~5x looser than the 2% gate it was
        nominally protecting -- and was never consulted by the gate anyway."""
        budget = cv_budget_for_gate(0.02, 20)
        assert 0.02 < budget < 0.025
        assert budget < 0.10

    def test_budget_scales_with_repeats(self):
        assert cv_budget_for_gate(0.02, 100) > cv_budget_for_gate(0.02, 20)
        assert cv_budget_for_gate(0.05, 20) > cv_budget_for_gate(0.02, 20)

    def test_repeats_needed_inverts_the_budget(self):
        cv = cv_budget_for_gate(0.02, 20)
        assert repeats_needed_for_gate(cv, 0.02) in (20, 21)  # ceil rounding
        # 8% CV against a 2% gate is a wildly impractical sample count -- which
        # is the honest signal that the environment is the bottleneck.
        assert repeats_needed_for_gate(0.08, 0.02) > 250

    def test_budget_is_the_break_even_point_of_the_real_gate(self):
        """cv_budget_for_gate must agree with what assess_improvement does."""
        cv = cv_budget_for_gate(0.02, 20)
        best = _spread(100.0, cv, n=20)
        v = assess_improvement(102.0, 100.0, "maximize", 0.02,
                               new_samples=[x * 1.02 for x in best], best_samples=best)
        assert v.noise_required is not None
        assert abs(v.noise_required - 0.02) < 0.002

    def test_relative_ci_margin_matches_t_cv_over_sqrt_n(self):
        stats = compute_bench_stats(_spread(100.0, 0.10, n=5))
        assert stats is not None
        assert abs(relative_ci_margin(stats) - 2.0 * 0.10 / 5 ** 0.5) < 1e-9


# --- Integration with the real accept gate ----------------------------------


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
        cv_threshold=overrides.get("cv_threshold", None),
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


def _write_incumbent_bench(ctx, value: float, samples: list[float]) -> None:
    (ctx.rp.run_dir / "bench.json").write_text(
        json.dumps({"ok": True, "throughput": {"median": value, "raw_values": samples}}),
        encoding="utf-8",
    )


def _accept(ctx, value: float, samples: list[float]):
    from perflab.optimizers.phases import evaluate as evaluate_mod

    cand = evaluate_mod.BeamCandidate(
        iteration=1, index=0, blocks=[], description="candidate 1",
        value=value, samples=samples,
    )
    with patch.object(evaluate_mod, "snapshot_workspace", lambda *a, **k: None):
        accepted, _, _ = evaluate_mod.accept_best(ctx, [cand], use_fast=False)
    return accepted


class TestAcceptBestNoiseGate:
    def test_within_noise_candidate_is_rejected_and_explained(self, tmp_path):
        messages: list[str] = []
        ctx = _make_ctx(tmp_path, best_value=100.0, messages=messages)
        incumbent = _spread(100.0, 0.08)
        _write_incumbent_bench(ctx, 100.0, incumbent)

        accepted = _accept(ctx, 105.0, [v * 1.05 for v in incumbent])

        assert accepted is False
        joined = "\n".join(messages)
        assert "within noise" in joined
        assert "CV=8.0%" in joined
        # The actionable part: the machine, not the kernel, is the bottleneck.
        assert "raise benchmark.repeats" in joined
        assert "limiting factor" in joined
        # ...and it survives into the run history, not just the console.
        assert "within noise" in ctx.history[0]["description"]

    def test_real_win_accepted_with_noisy_samples(self, tmp_path):
        messages: list[str] = []
        ctx = _make_ctx(tmp_path, best_value=100.0, messages=messages)
        incumbent = _spread(100.0, 0.08)
        _write_incumbent_bench(ctx, 100.0, incumbent)

        accepted = _accept(ctx, 1000.0, _spread(1000.0, 0.08))

        assert accepted is True
        assert ctx.best_value == 1000.0

    def test_no_samples_keeps_old_behavior_but_says_so(self, tmp_path):
        messages: list[str] = []
        ctx = _make_ctx(tmp_path, best_value=100.0, messages=messages)

        accepted = _accept(ctx, 105.0, [])

        assert accepted is True  # exactly the pre-existing bare-ratio gate
        assert any("without variance verification" in m for m in messages)

    def test_noise_gate_disabled_accepts_within_noise(self, tmp_path):
        messages: list[str] = []
        ctx = _make_ctx(tmp_path, best_value=100.0, messages=messages, noise_gate=False)
        incumbent = _spread(100.0, 0.08)
        _write_incumbent_bench(ctx, 100.0, incumbent)

        assert _accept(ctx, 105.0, [v * 1.05 for v in incumbent]) is True

    def test_tolerance_rejection_does_not_claim_noise(self, tmp_path):
        messages: list[str] = []
        ctx = _make_ctx(tmp_path, best_value=100.0, messages=messages)
        incumbent = _spread(100.0, 0.001)
        _write_incumbent_bench(ctx, 100.0, incumbent)

        assert _accept(ctx, 100.5, [v * 1.005 for v in incumbent]) is False
        joined = "\n".join(messages)
        assert "within noise" not in joined
        assert "regression tolerance" in ctx.history[0]["description"]

    def test_stale_incumbent_bench_is_ignored(self, tmp_path):
        """bench.json whose metric no longer matches best_value (auto-tune moved
        it) must not be used as the incumbent's spread."""
        from perflab.optimizers.phases import evaluate as evaluate_mod

        messages: list[str] = []
        ctx = _make_ctx(tmp_path, best_value=100.0, messages=messages)
        _write_incumbent_bench(ctx, 77.0, _spread(77.0, 0.08))
        assert evaluate_mod._incumbent_samples(ctx) == []

        _write_incumbent_bench(ctx, 100.0, [99.0, 100.0, 101.0])
        assert evaluate_mod._incumbent_samples(ctx) == [99.0, 100.0, 101.0]

    def test_missing_or_corrupt_incumbent_bench_is_safe(self, tmp_path):
        from perflab.optimizers.phases import evaluate as evaluate_mod

        ctx = _make_ctx(tmp_path, best_value=100.0, messages=[])
        assert evaluate_mod._incumbent_samples(ctx) == []
        (ctx.rp.run_dir / "bench.json").write_text("{not json", encoding="utf-8")
        assert evaluate_mod._incumbent_samples(ctx) == []

    def test_fast_screen_does_not_apply_the_variance_test(self, tmp_path, monkeypatch):
        """A 2-repeat screen must not veto a candidate before it is measured
        properly; the full re-bench is the authoritative gate."""
        from perflab.optimizers.phases import evaluate as evaluate_mod
        from perflab.tools.shell import CmdResult

        messages: list[str] = []
        ctx = _make_ctx(tmp_path, best_value=100.0, messages=messages)
        incumbent = _spread(100.0, 0.02, n=20)
        _write_incumbent_bench(ctx, 100.0, incumbent)

        def fake_benchmark(cmd, cwd, **kwargs):
            return (
                CmdResult(cmd=[], returncode=0, stdout="", stderr="", duration_s=0.01),
                {"ok": True, "throughput": {
                    "median": 110.0, "raw_values": _spread(110.0, 0.02, n=20),
                }},
            )

        monkeypatch.setattr(evaluate_mod, "run_benchmark", fake_benchmark)

        # Screen value looks like a noisy 3% with only 2 (very jittery) samples.
        cand = evaluate_mod.BeamCandidate(
            iteration=1, index=0, blocks=[], description="candidate 1",
            value=103.0, samples=[80.0, 130.0],
        )
        with patch.object(evaluate_mod, "snapshot_workspace", lambda *a, **k: None):
            accepted, _, accepted_value = evaluate_mod.accept_best(
                ctx, [cand], use_fast=True,
            )

        assert accepted is True
        assert accepted_value == 110.0  # the full re-bench value, not the screen


class TestTaskSpecKnobs:
    def test_defaults(self, tmp_path):
        from perflab.task_spec import Constraints

        c = Constraints()
        assert c.noise_gate is True
        assert c.cv_threshold is None

    def test_parsed_from_yaml(self, tmp_path):
        from perflab.task_spec import TaskSpec

        (tmp_path / "task.yaml").write_text(
            "name: t\n"
            "program_type: python\n"
            "correctness:\n  cmd: 'python tests.py'\n"
            "benchmark:\n"
            "  cmd: 'python bench.py --json out/bench.json'\n"
            "  metric:\n    name: 'tflops.median'\n    mode: maximize\n"
            "constraints:\n"
            "  regression_tolerance: 0.05\n"
            "  noise_gate: false\n"
            "  cv_threshold: 0.03\n"
            "edit_policy:\n  allowed_paths: ['a.py']\n",
            encoding="utf-8",
        )
        task = TaskSpec.load(tmp_path / "task.yaml")
        assert task.constraints.noise_gate is False
        assert task.constraints.cv_threshold == 0.03
