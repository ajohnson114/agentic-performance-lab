"""Tests for perflab.ci — CI regression check and baseline saving."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from perflab.ci import (
    CICheckResult,
    MetricCheckResult,
    ProfilerRegression,
    _check_regression,
    _default_baseline_path,
    _detect_profiler_regressions,
    run_ci_check,
    save_baseline,
)

# ---------------------------------------------------------------------------
# CICheckResult
# ---------------------------------------------------------------------------

class TestCICheckResult:
    def test_to_dict(self):
        r = CICheckResult(
            passed=True,
            current_value=1.5,
            baseline_value=1.4,
            regression_pct=-7.1,
            tolerance_pct=5.0,
            metric_name="tflops.median",
            metric_mode="maximize",
        )
        d = r.to_dict()
        assert d["passed"] is True
        assert d["current_value"] == 1.5
        assert d["baseline_value"] == 1.4
        assert d["regression_pct"] == -7.1
        assert d["tolerance_pct"] == 5.0
        assert d["metric_name"] == "tflops.median"
        assert d["metric_mode"] == "maximize"
        assert "secondary" not in d

    def test_to_dict_no_baseline(self):
        r = CICheckResult(
            passed=True,
            current_value=1.0,
            baseline_value=None,
            regression_pct=None,
            tolerance_pct=5.0,
            metric_name="time.median",
            metric_mode="minimize",
        )
        d = r.to_dict()
        assert d["baseline_value"] is None
        assert d["regression_pct"] is None

    def test_to_dict_with_secondary(self):
        sec = MetricCheckResult(
            name="memory_mb",
            mode="minimize",
            current_value=120.0,
            baseline_value=100.0,
            regression_pct=20.0,
            tolerance_pct=5.0,
            regressed=True,
        )
        r = CICheckResult(
            passed=False,
            current_value=1.5,
            baseline_value=1.4,
            regression_pct=-7.1,
            tolerance_pct=5.0,
            metric_name="tflops.median",
            metric_mode="maximize",
            secondary=sec,
        )
        d = r.to_dict()
        assert "secondary" in d
        assert d["secondary"]["metric_name"] == "memory_mb"
        assert d["secondary"]["regressed"] is True
        assert d["secondary"]["regression_pct"] == 20.0

    def test_to_dict_with_profiler_regressions(self):
        r = CICheckResult(
            passed=True,
            current_value=1.0,
            baseline_value=1.0,
            regression_pct=0.0,
            tolerance_pct=5.0,
            metric_name="tflops.median",
            metric_mode="maximize",
            profiler_regressions=[
                ProfilerRegression(metric="sm_utilization_pct", current=40.0, baseline=60.0, direction="decreased"),
            ],
        )
        d = r.to_dict()
        assert "profiler_regressions" in d
        assert len(d["profiler_regressions"]) == 1
        assert d["profiler_regressions"][0]["metric"] == "sm_utilization_pct"
        assert d["profiler_regressions"][0]["direction"] == "decreased"

    def test_to_dict_with_bench_variance_warnings(self):
        r = CICheckResult(
            passed=True,
            current_value=1.0,
            baseline_value=1.0,
            regression_pct=0.0,
            tolerance_pct=5.0,
            metric_name="tflops.median",
            metric_mode="maximize",
            bench_variance_warnings=["Zero variance in latency_ms.all"],
        )
        d = r.to_dict()
        assert "bench_variance_warnings" in d
        assert len(d["bench_variance_warnings"]) == 1

    def test_to_dict_omits_empty_optional_fields(self):
        r = CICheckResult(
            passed=True,
            current_value=1.0,
            baseline_value=1.0,
            regression_pct=0.0,
            tolerance_pct=5.0,
            metric_name="tflops.median",
            metric_mode="maximize",
        )
        d = r.to_dict()
        assert "profiler_regressions" not in d
        assert "bench_variance_warnings" not in d
        assert "secondary" not in d


# ---------------------------------------------------------------------------
# _check_regression
# ---------------------------------------------------------------------------

class TestCheckRegression:
    def test_maximize_no_regression(self):
        pct, regressed = _check_regression(1.1, 1.0, "maximize", 0.05)
        assert regressed is False
        assert pct < 0  # improvement

    def test_maximize_regression(self):
        pct, regressed = _check_regression(0.90, 1.0, "maximize", 0.05)
        assert regressed is True
        assert pct == pytest.approx(10.0)

    def test_maximize_within_tolerance(self):
        pct, regressed = _check_regression(0.97, 1.0, "maximize", 0.05)
        assert regressed is False

    def test_minimize_no_regression(self):
        pct, regressed = _check_regression(0.9, 1.0, "minimize", 0.05)
        assert regressed is False
        assert pct < 0

    def test_minimize_regression(self):
        pct, regressed = _check_regression(1.10, 1.0, "minimize", 0.05)
        assert regressed is True
        assert pct == pytest.approx(10.0)

    def test_minimize_within_tolerance(self):
        pct, regressed = _check_regression(1.03, 1.0, "minimize", 0.05)
        assert regressed is False

    def test_baseline_zero(self):
        pct, regressed = _check_regression(1.0, 0.0, "maximize", 0.05)
        assert regressed is False
        assert pct == 0.0


# ---------------------------------------------------------------------------
# _detect_profiler_regressions
# ---------------------------------------------------------------------------

class TestDetectProfilerRegressions:
    def test_no_data_returns_empty(self):
        assert _detect_profiler_regressions({}, {}) == []

    def test_sm_util_decrease_detected(self):
        baseline = {"sm_utilization_pct": 80.0}
        current = {"sm_utilization_pct": 70.0}
        regs = _detect_profiler_regressions(current, baseline)
        assert len(regs) == 1
        assert regs[0].metric == "sm_utilization_pct"
        assert regs[0].direction == "decreased"

    def test_sm_util_small_decrease_not_flagged(self):
        baseline = {"sm_utilization_pct": 80.0}
        current = {"sm_utilization_pct": 77.0}  # within 5% threshold
        assert _detect_profiler_regressions(current, baseline) == []

    def test_bank_conflicts_increase_detected(self):
        baseline = {"bank_conflicts": 10.0}
        current = {"bank_conflicts": 100.0}
        regs = _detect_profiler_regressions(current, baseline)
        assert len(regs) == 1
        assert regs[0].metric == "bank_conflicts"
        assert regs[0].direction == "increased"

    def test_stall_pct_increase_detected(self):
        baseline = {"dominant_stall_pct": 20.0}
        current = {"dominant_stall_pct": 35.0}
        regs = _detect_profiler_regressions(current, baseline)
        assert len(regs) == 1
        assert regs[0].metric == "dominant_stall_pct"

    def test_sectors_per_request_increase_detected(self):
        baseline = {"sectors_per_request": 4.0}
        current = {"sectors_per_request": 6.0}
        regs = _detect_profiler_regressions(current, baseline)
        assert len(regs) == 1
        assert regs[0].metric == "sectors_per_request"

    def test_improvement_not_flagged(self):
        baseline = {"sm_utilization_pct": 60.0, "bank_conflicts": 100.0}
        current = {"sm_utilization_pct": 80.0, "bank_conflicts": 20.0}
        assert _detect_profiler_regressions(current, baseline) == []

    def test_missing_metrics_skipped(self):
        baseline = {"sm_utilization_pct": 80.0}
        current = {"achieved_occupancy_pct": 70.0}
        assert _detect_profiler_regressions(current, baseline) == []

    def test_multiple_regressions(self):
        baseline = {
            "sm_utilization_pct": 80.0,
            "tensor_core_utilization_pct": 50.0,
            "bank_conflicts": 10.0,
        }
        current = {
            "sm_utilization_pct": 60.0,
            "tensor_core_utilization_pct": 30.0,
            "bank_conflicts": 200.0,
        }
        regs = _detect_profiler_regressions(current, baseline)
        assert len(regs) == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(tmp_path, metric_name="tflops.median", metric_mode="maximize",
               regression_tolerance=0.05, secondary_name=None, secondary_mode="minimize"):
    """Create a minimal mock TaskSpec."""
    task = MagicMock()
    task.workspace = tmp_path
    task.name = "test_task"
    task.out_dir = tmp_path / "out"
    task.benchmark.metric.name = metric_name
    task.benchmark.metric.mode = metric_mode
    task.benchmark.cmd = "python bench.py"
    task.correctness.cmd = "python tests.py"
    task.correctness.expected_exit = 0
    task.build = None
    task.contract = MagicMock()
    task.contract.required_bench_fields = []
    task.contract.max_value = {}
    task.contract.min_value = {}
    task.constraints.regression_tolerance = regression_tolerance
    task.constraints.rlimit_as_gb = None
    task.program_type = "python"
    if secondary_name:
        task.benchmark.secondary_metric.name = secondary_name
        task.benchmark.secondary_metric.mode = secondary_mode
    else:
        task.benchmark.secondary_metric = None
    return task


def _write_baseline(path: Path, value: float, metric_name="tflops.median",
                    metric_mode="maximize", secondary_name=None,
                    secondary_mode="minimize", secondary_value=None,
                    ncu_summary=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metric_name": metric_name,
        "metric_mode": metric_mode,
        "value": value,
    }
    if secondary_name and secondary_value is not None:
        data["secondary_metric_name"] = secondary_name
        data["secondary_metric_mode"] = secondary_mode
        data["secondary_value"] = secondary_value
    if ncu_summary:
        data["ncu_summary"] = ncu_summary
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# _default_baseline_path
# ---------------------------------------------------------------------------

def test_default_baseline_path(tmp_path):
    task = _make_task(tmp_path)
    assert _default_baseline_path(task) == tmp_path / "baseline.json"


# ---------------------------------------------------------------------------
# save_baseline
# ---------------------------------------------------------------------------

class TestSaveBaseline:
    @patch("perflab.ci._find_latest_ncu_summary", return_value=None)
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 2.5}})
    def test_saves_primary_only(self, mock_bench, mock_ncu, tmp_path):
        task = _make_task(tmp_path)
        result_path = save_baseline(task)
        assert result_path == tmp_path / "baseline.json"
        data = json.loads(result_path.read_text())
        assert data["value"] == 2.5
        assert data["metric_name"] == "tflops.median"
        assert "secondary_value" not in data

    @patch("perflab.ci._find_latest_ncu_summary", return_value=None)
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 2.5}, "memory_mb": 100.0})
    def test_saves_with_secondary(self, mock_bench, mock_ncu, tmp_path):
        task = _make_task(tmp_path, secondary_name="memory_mb", secondary_mode="minimize")
        result_path = save_baseline(task)
        data = json.loads(result_path.read_text())
        assert data["value"] == 2.5
        assert data["secondary_value"] == 100.0
        assert data["secondary_metric_name"] == "memory_mb"
        assert data["secondary_metric_mode"] == "minimize"

    @patch("perflab.ci._find_latest_ncu_summary", return_value=None)
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 3.0}})
    def test_saves_to_custom_path(self, mock_bench, mock_ncu, tmp_path):
        task = _make_task(tmp_path)
        custom = tmp_path / "custom" / "bl.json"
        result_path = save_baseline(task, baseline_path=custom)
        assert result_path == custom
        assert custom.exists()

    @patch("perflab.ci._find_latest_ncu_summary", return_value=None)
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 2.5}})
    def test_secondary_missing_from_bench_skipped(self, mock_bench, mock_ncu, tmp_path):
        # secondary configured but not in bench output — should not crash
        task = _make_task(tmp_path, secondary_name="nonexistent_metric")
        result_path = save_baseline(task)
        data = json.loads(result_path.read_text())
        assert data["value"] == 2.5
        assert "secondary_value" not in data

    @patch("perflab.ci._find_latest_ncu_summary", return_value=None)
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 2.5}})
    def test_saves_ncu_summary_when_provided(self, mock_bench, mock_ncu, tmp_path):
        task = _make_task(tmp_path)
        ncu = {"sm_utilization_pct": 85.0, "achieved_occupancy_pct": 90.0}
        result_path = save_baseline(task, ncu_summary=ncu)
        data = json.loads(result_path.read_text())
        assert data["ncu_summary"] == ncu

    @patch("perflab.ci._find_latest_ncu_summary", return_value={"sm_utilization_pct": 75.0})
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 2.5}})
    def test_saves_ncu_from_run_store(self, mock_bench, mock_ncu, tmp_path):
        task = _make_task(tmp_path)
        result_path = save_baseline(task)
        data = json.loads(result_path.read_text())
        assert data["ncu_summary"]["sm_utilization_pct"] == 75.0

    @patch("perflab.ci._find_latest_ncu_summary", return_value=None)
    @patch("perflab.ci._run_bench_full", return_value={
        "tflops": {"median": 2.5},
        "latency_ms": {"all": [1.0, 1.0, 1.0, 1.0, 1.0]},
    })
    def test_saves_bench_variance_warnings(self, mock_bench, mock_ncu, tmp_path):
        task = _make_task(tmp_path)
        result_path = save_baseline(task)
        data = json.loads(result_path.read_text())
        assert "bench_variance_warnings" in data
        assert len(data["bench_variance_warnings"]) > 0


# ---------------------------------------------------------------------------
# run_ci_check — primary only (maximize)
# ---------------------------------------------------------------------------

class TestRunCICheckMaximize:
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_no_baseline_passes(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize")
        result = run_ci_check(task)
        assert result.passed is True
        assert result.baseline_value is None
        assert result.secondary is None

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.1}})
    def test_improvement_passes(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize", regression_tolerance=0.05)
        _write_baseline(tmp_path / "baseline.json", 1.0)
        result = run_ci_check(task)
        assert result.passed is True

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 0.97}})
    def test_small_regression_within_tolerance_passes(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize", regression_tolerance=0.05)
        _write_baseline(tmp_path / "baseline.json", 1.0)
        result = run_ci_check(task)
        assert result.passed is True

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 0.90}})
    def test_large_regression_fails(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize", regression_tolerance=0.05)
        _write_baseline(tmp_path / "baseline.json", 1.0)
        result = run_ci_check(task)
        assert result.passed is False
        assert result.regression_pct == pytest.approx(10.0)

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 0.95}})
    def test_exactly_at_tolerance_passes(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize", regression_tolerance=0.05)
        _write_baseline(tmp_path / "baseline.json", 1.0)
        result = run_ci_check(task)
        assert result.passed is True


# ---------------------------------------------------------------------------
# run_ci_check — primary only (minimize)
# ---------------------------------------------------------------------------

class TestRunCICheckMinimize:
    @patch("perflab.ci._run_bench_full", return_value={"time": {"median": 0.9}})
    def test_improvement_passes(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_name="time.median", metric_mode="minimize",
                          regression_tolerance=0.05)
        _write_baseline(tmp_path / "baseline.json", 1.0, metric_name="time.median",
                        metric_mode="minimize")
        result = run_ci_check(task)
        assert result.passed is True

    @patch("perflab.ci._run_bench_full", return_value={"time": {"median": 1.03}})
    def test_small_regression_within_tolerance_passes(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_name="time.median", metric_mode="minimize",
                          regression_tolerance=0.05)
        _write_baseline(tmp_path / "baseline.json", 1.0, metric_name="time.median",
                        metric_mode="minimize")
        result = run_ci_check(task)
        assert result.passed is True

    @patch("perflab.ci._run_bench_full", return_value={"time": {"median": 1.10}})
    def test_large_regression_fails(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_name="time.median", metric_mode="minimize",
                          regression_tolerance=0.05)
        _write_baseline(tmp_path / "baseline.json", 1.0, metric_name="time.median",
                        metric_mode="minimize")
        result = run_ci_check(task)
        assert result.passed is False
        assert result.regression_pct == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# run_ci_check — Pareto / secondary metric
# ---------------------------------------------------------------------------

class TestRunCICheckPareto:
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.1}, "memory_mb": 95.0})
    def test_both_improved_passes(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, secondary_name="memory_mb", secondary_mode="minimize")
        _write_baseline(tmp_path / "baseline.json", 1.0,
                        secondary_name="memory_mb", secondary_mode="minimize",
                        secondary_value=100.0)
        result = run_ci_check(task)
        assert result.passed is True
        assert result.secondary is not None
        assert result.secondary.regressed is False

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.1}, "memory_mb": 120.0})
    def test_primary_improved_secondary_regressed_fails(self, mock_bench, tmp_path):
        """Primary metric improved but secondary regressed beyond tolerance — should fail."""
        task = _make_task(tmp_path, regression_tolerance=0.05,
                          secondary_name="memory_mb", secondary_mode="minimize")
        _write_baseline(tmp_path / "baseline.json", 1.0,
                        secondary_name="memory_mb", secondary_mode="minimize",
                        secondary_value=100.0)
        result = run_ci_check(task)
        assert result.passed is False
        assert result.secondary is not None
        assert result.secondary.regressed is True
        assert result.secondary.regression_pct == pytest.approx(20.0)

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 0.90}, "memory_mb": 95.0})
    def test_primary_regressed_secondary_improved_fails(self, mock_bench, tmp_path):
        """Secondary improved but primary regressed — should fail."""
        task = _make_task(tmp_path, regression_tolerance=0.05,
                          secondary_name="memory_mb", secondary_mode="minimize")
        _write_baseline(tmp_path / "baseline.json", 1.0,
                        secondary_name="memory_mb", secondary_mode="minimize",
                        secondary_value=100.0)
        result = run_ci_check(task)
        assert result.passed is False
        assert result.secondary.regressed is False

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 0.90}, "memory_mb": 120.0})
    def test_both_regressed_fails(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, regression_tolerance=0.05,
                          secondary_name="memory_mb", secondary_mode="minimize")
        _write_baseline(tmp_path / "baseline.json", 1.0,
                        secondary_name="memory_mb", secondary_mode="minimize",
                        secondary_value=100.0)
        result = run_ci_check(task)
        assert result.passed is False
        assert result.secondary.regressed is True

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.1}, "memory_mb": 103.0})
    def test_secondary_within_tolerance_passes(self, mock_bench, tmp_path):
        """Secondary regressed slightly but within tolerance — should pass."""
        task = _make_task(tmp_path, regression_tolerance=0.05,
                          secondary_name="memory_mb", secondary_mode="minimize")
        _write_baseline(tmp_path / "baseline.json", 1.0,
                        secondary_name="memory_mb", secondary_mode="minimize",
                        secondary_value=100.0)
        result = run_ci_check(task)
        assert result.passed is True
        assert result.secondary.regressed is False

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}, "tokens_per_sec": 550.0})
    def test_secondary_maximize_mode(self, mock_bench, tmp_path):
        """Secondary metric in maximize mode (e.g. tokens/sec) — regression = decrease."""
        task = _make_task(tmp_path, regression_tolerance=0.05,
                          secondary_name="tokens_per_sec", secondary_mode="maximize")
        _write_baseline(tmp_path / "baseline.json", 1.0,
                        secondary_name="tokens_per_sec", secondary_mode="maximize",
                        secondary_value=500.0)
        result = run_ci_check(task)
        assert result.passed is True
        assert result.secondary.regressed is False
        assert result.secondary.regression_pct < 0  # improvement

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}, "tokens_per_sec": 400.0})
    def test_secondary_maximize_regression_fails(self, mock_bench, tmp_path):
        """Secondary maximize metric decreased beyond tolerance — should fail."""
        task = _make_task(tmp_path, regression_tolerance=0.05,
                          secondary_name="tokens_per_sec", secondary_mode="maximize")
        _write_baseline(tmp_path / "baseline.json", 1.0,
                        secondary_name="tokens_per_sec", secondary_mode="maximize",
                        secondary_value=500.0)
        result = run_ci_check(task)
        assert result.passed is False
        assert result.secondary.regressed is True

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_secondary_configured_but_not_in_bench_passes(self, mock_bench, tmp_path):
        """Secondary metric configured but not in bench output — skip gracefully."""
        task = _make_task(tmp_path, regression_tolerance=0.05,
                          secondary_name="memory_mb", secondary_mode="minimize")
        _write_baseline(tmp_path / "baseline.json", 1.0,
                        secondary_name="memory_mb", secondary_mode="minimize",
                        secondary_value=100.0)
        result = run_ci_check(task)
        assert result.passed is True
        assert result.secondary is None  # couldn't extract — skipped

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}, "memory_mb": 95.0})
    def test_secondary_not_in_baseline_ignored(self, mock_bench, tmp_path):
        """Secondary exists in bench but not in baseline — no comparison possible."""
        task = _make_task(tmp_path, regression_tolerance=0.05,
                          secondary_name="memory_mb", secondary_mode="minimize")
        # Baseline without secondary
        _write_baseline(tmp_path / "baseline.json", 1.0)
        result = run_ci_check(task)
        assert result.passed is True
        assert result.secondary is None


# ---------------------------------------------------------------------------
# run_ci_check — bench variance (anti-gaming)
# ---------------------------------------------------------------------------

class TestRunCICheckBenchVariance:
    @patch("perflab.ci._run_bench_full", return_value={
        "tflops": {"median": 1.0},
        "latency_ms": {"all": [5.0, 5.0, 5.0, 5.0, 5.0]},
    })
    def test_zero_variance_produces_warning(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize")
        _write_baseline(tmp_path / "baseline.json", 1.0)
        result = run_ci_check(task)
        assert len(result.bench_variance_warnings) > 0
        assert any("variance" in w.lower() for w in result.bench_variance_warnings)
        # Variance warnings are advisory — don't fail the check
        assert result.passed is True

    @patch("perflab.ci._run_bench_full", return_value={
        "tflops": {"median": 1.0},
        "latency_ms": {"all": [5.0, 5.1, 4.9, 5.2, 4.8]},
    })
    def test_normal_variance_no_warning(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize")
        _write_baseline(tmp_path / "baseline.json", 1.0)
        result = run_ci_check(task)
        assert len(result.bench_variance_warnings) == 0

    @patch("perflab.ci._run_bench_full", return_value={
        "tflops": {"median": 1.0},
        "latency_ms": {"all": [5.0, 5.0, 5.0, 5.0, 5.0]},
    })
    def test_variance_warning_in_to_dict(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize")
        result = run_ci_check(task)
        d = result.to_dict()
        assert "bench_variance_warnings" in d


# ---------------------------------------------------------------------------
# run_ci_check — profiler regressions
# ---------------------------------------------------------------------------

class TestRunCICheckProfilerRegressions:
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_profiler_regression_detected(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize")
        baseline_ncu = {"sm_utilization_pct": 80.0, "bank_conflicts": 10.0}
        current_ncu = {"sm_utilization_pct": 60.0, "bank_conflicts": 200.0}
        _write_baseline(tmp_path / "baseline.json", 1.0, ncu_summary=baseline_ncu)
        result = run_ci_check(task, ncu_summary=current_ncu)
        assert result.passed is True  # profiler regressions are advisory
        assert len(result.profiler_regressions) == 2

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_no_profiler_data_no_regressions(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize")
        _write_baseline(tmp_path / "baseline.json", 1.0)
        result = run_ci_check(task)
        assert result.profiler_regressions == []

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_only_baseline_ncu_no_comparison(self, mock_bench, tmp_path):
        """NCU data in baseline but not in current run — no comparison."""
        task = _make_task(tmp_path, metric_mode="maximize")
        _write_baseline(tmp_path / "baseline.json", 1.0,
                        ncu_summary={"sm_utilization_pct": 80.0})
        result = run_ci_check(task)
        assert result.profiler_regressions == []

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_profiler_improvement_not_flagged(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize")
        baseline_ncu = {"sm_utilization_pct": 60.0}
        current_ncu = {"sm_utilization_pct": 80.0}
        _write_baseline(tmp_path / "baseline.json", 1.0, ncu_summary=baseline_ncu)
        result = run_ci_check(task, ncu_summary=current_ncu)
        assert result.profiler_regressions == []

    @patch("perflab.ci._find_latest_ncu_summary", return_value={"sm_utilization_pct": 50.0})
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_ncu_from_run_store_used(self, mock_bench, mock_ncu, tmp_path):
        """When no explicit NCU summary, falls back to run store."""
        task = _make_task(tmp_path, metric_mode="maximize")
        baseline_ncu = {"sm_utilization_pct": 80.0}
        _write_baseline(tmp_path / "baseline.json", 1.0, ncu_summary=baseline_ncu)
        result = run_ci_check(task)
        assert len(result.profiler_regressions) == 1
        assert result.profiler_regressions[0].metric == "sm_utilization_pct"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRunCICheckEdgeCases:
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_baseline_value_zero(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize")
        _write_baseline(tmp_path / "baseline.json", 0.0)
        result = run_ci_check(task)
        assert result.passed is True
        assert result.regression_pct == 0.0

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_custom_baseline_path(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_mode="maximize")
        custom_bp = tmp_path / "alt" / "my_baseline.json"
        _write_baseline(custom_bp, 0.8)
        result = run_ci_check(task, baseline_path=custom_bp)
        assert result.passed is True
        assert result.baseline_value == 0.8

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_result_fields_populated(self, mock_bench, tmp_path):
        task = _make_task(tmp_path, metric_name="tflops.median", metric_mode="maximize",
                          regression_tolerance=0.10)
        _write_baseline(tmp_path / "baseline.json", 1.0)
        result = run_ci_check(task)
        assert result.metric_name == "tflops.median"
        assert result.metric_mode == "maximize"
        assert result.tolerance_pct == 10.0
        assert result.current_value == 1.0
        assert result.baseline_value == 1.0
