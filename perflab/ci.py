from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from perflab.analyzers.bottleneck_analyzer import AnalysisThresholds
from perflab.runners.benchmark import (
    metric_value,
    run_benchmark,
    validate_bench_variance,
    validate_contract,
)
from perflab.runners.correctness import run_correctness
from perflab.task_spec import TaskSpec

logger = logging.getLogger(__name__)


@dataclass
class MetricCheckResult:
    """Regression check result for a single metric."""
    name: str
    mode: str
    current_value: float
    baseline_value: float | None
    regression_pct: float | None
    tolerance_pct: float
    regressed: bool


@dataclass
class ProfilerRegression:
    """A profiler metric that regressed vs baseline."""
    metric: str
    current: float
    baseline: float
    direction: str  # "increased" or "decreased"


@dataclass
class CICheckResult:
    passed: bool
    current_value: float
    baseline_value: float | None
    regression_pct: float | None
    tolerance_pct: float
    metric_name: str
    metric_mode: str
    secondary: MetricCheckResult | None = None
    profiler_regressions: list[ProfilerRegression] = field(default_factory=list)
    bench_variance_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "passed": self.passed,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "regression_pct": self.regression_pct,
            "tolerance_pct": self.tolerance_pct,
            "metric_name": self.metric_name,
            "metric_mode": self.metric_mode,
        }
        if self.secondary is not None:
            d["secondary"] = {
                "metric_name": self.secondary.name,
                "metric_mode": self.secondary.mode,
                "current_value": self.secondary.current_value,
                "baseline_value": self.secondary.baseline_value,
                "regression_pct": self.secondary.regression_pct,
                "tolerance_pct": self.secondary.tolerance_pct,
                "regressed": self.secondary.regressed,
            }
        if self.profiler_regressions:
            d["profiler_regressions"] = [
                {"metric": r.metric, "current": r.current,
                 "baseline": r.baseline, "direction": r.direction}
                for r in self.profiler_regressions
            ]
        if self.bench_variance_warnings:
            d["bench_variance_warnings"] = self.bench_variance_warnings
        return d


def _default_baseline_path(task: TaskSpec) -> Path:
    return task.workspace / "baseline.json"


def _run_bench_full(task: TaskSpec) -> dict:
    """Run correctness + benchmark, return full bench dict."""
    ws = task.workspace
    (ws / "out").mkdir(parents=True, exist_ok=True)

    # Build
    if task.build is not None:
        import shlex

        from perflab.tools.shell import run_cmd
        bres = run_cmd(shlex.split(task.build.cmd), cwd=ws)
        if bres.returncode != task.build.expected_exit:
            raise RuntimeError(f"Build failed with code {bres.returncode}")

    # Correctness
    cres = run_correctness(task.correctness.cmd, cwd=ws, program_type=task.program_type, rlimit_as_gb=task.constraints.rlimit_as_gb, env_passthrough=task.constraints.env_passthrough, accuracy_tolerance=task.constraints.accuracy_tolerance)
    if cres.returncode != task.correctness.expected_exit:
        raise RuntimeError(f"Correctness failed with code {cres.returncode}")

    # Benchmark
    _, bench = run_benchmark(task.benchmark.cmd, cwd=ws, program_type=task.program_type, rlimit_as_gb=task.constraints.rlimit_as_gb, env_passthrough=task.constraints.env_passthrough, warmup=task.benchmark.warmup, repeats=task.benchmark.repeats)

    # Contract validation
    contract_errors = validate_contract(bench, task.contract)
    if contract_errors:
        raise RuntimeError(f"Contract violation: {contract_errors}")

    return bench


def _run_bench(task: TaskSpec) -> float:
    """Run correctness + benchmark, return primary metric value."""
    bench = _run_bench_full(task)
    return metric_value(bench, task.benchmark.metric.name)


def _check_regression(
    current: float, baseline: float, mode: str, tolerance: float,
) -> tuple[float, bool]:
    """Check if current value is a regression vs baseline.

    Returns (regression_pct, regressed).
    """
    if baseline == 0:
        return 0.0, False
    if mode == "maximize":
        regression_pct = (1.0 - current / baseline) * 100
        regressed = current < baseline * (1.0 - tolerance)
    else:
        regression_pct = (current / baseline - 1.0) * 100
        regressed = current > baseline * (1.0 + tolerance)
    return regression_pct, regressed


def _detect_profiler_regressions(
    current_ncu: dict, baseline_ncu: dict,
    thresholds: AnalysisThresholds | None = None,
) -> list[ProfilerRegression]:
    """Compare NCU profiler metrics between current and baseline.

    Flags metrics that moved in the wrong direction (e.g., TC util decreased,
    warp stalls increased, bank conflicts increased).
    Uses AnalysisThresholds for consistent thresholds with the rest of the framework.
    """
    regressions: list[ProfilerRegression] = []

    # Metrics where decrease is bad (threshold = min regression delta to flag)
    # 5pp: a 5-percentage-point drop in any of these indicates meaningful degradation
    _HIGHER_IS_BETTER = [
        ("sm_utilization_pct", 5.0),
        ("achieved_occupancy_pct", 5.0),
        ("tensor_core_utilization_pct", 5.0),
        ("branch_efficiency_pct", 5.0),
    ]
    for metric, delta_threshold in _HIGHER_IS_BETTER:
        curr = current_ncu.get(metric)
        base = baseline_ncu.get(metric)
        if curr is not None and base is not None and base - curr > delta_threshold:
            regressions.append(ProfilerRegression(
                metric=metric, current=curr, baseline=base, direction="decreased",
            ))

    # Metrics where increase is bad (delta thresholds for CI regression detection)
    # 10pp stall increase, 50 more bank conflicts, 1 more sector/request
    _LOWER_IS_BETTER = [
        ("dominant_stall_pct", 10.0),
        ("bank_conflicts", 50.0),
        ("sectors_per_request", 1.0),
    ]
    for metric, delta_threshold in _LOWER_IS_BETTER:
        curr = current_ncu.get(metric)
        base = baseline_ncu.get(metric)
        if curr is not None and base is not None and curr - base > delta_threshold:
            regressions.append(ProfilerRegression(
                metric=metric, current=curr, baseline=base, direction="increased",
            ))

    return regressions


def _find_latest_ncu_summary(task: TaskSpec) -> dict | None:
    """Look for NCU profiler summary from the most recent run in the run store."""
    try:
        from perflab.memory.run_store import RunStore
        store = RunStore(task.out_dir)
        runs = store.list_runs(task=task.name, limit=1)
        if runs:
            run_data = store.get_run(runs[0]["run_id"])
            summaries = run_data.get("profiler_summaries", {})
            return summaries.get("ncu") or summaries.get("ncu_profiler")
    except Exception:  # noqa: BLE001 -- best-effort lookup, missing/corrupt run store shouldn't break CI
        logger.debug("Failed to load NCU summary from prior run", exc_info=True)
    return None


def save_baseline(
    task: TaskSpec,
    baseline_path: Path | None = None,
    ncu_summary: dict | None = None,
) -> Path:
    """Run benchmark and save result as baseline.

    If ncu_summary is provided, it is saved in the baseline for future
    profiler regression detection. If not provided, attempts to find
    NCU data from the most recent run in the run store.
    """
    bp = baseline_path or _default_baseline_path(task)
    bench = _run_bench_full(task)
    primary_value = metric_value(bench, task.benchmark.metric.name)
    data: dict = {
        "metric_name": task.benchmark.metric.name,
        "metric_mode": task.benchmark.metric.mode,
        "value": primary_value,
    }
    sec = task.benchmark.secondary_metric
    if sec:
        try:
            data["secondary_metric_name"] = sec.name
            data["secondary_metric_mode"] = sec.mode
            data["secondary_value"] = metric_value(bench, sec.name)
        except (KeyError, TypeError):
            pass  # secondary metric not in bench output — skip

    # Save NCU profiler summary for future regression detection
    ncu = ncu_summary or _find_latest_ncu_summary(task)
    if ncu:
        data["ncu_summary"] = ncu

    # Save bench variance baseline for reference
    variance_warnings = validate_bench_variance(bench)
    if variance_warnings:
        data["bench_variance_warnings"] = variance_warnings

    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return bp


def run_ci_check(
    task: TaskSpec,
    baseline_path: Path | None = None,
    ncu_summary: dict | None = None,
    tolerance: float | None = None,
) -> CICheckResult:
    """Run benchmark and compare against baseline for regression.

    Checks primary metric, secondary metric (if configured), bench.json
    variance (anti-gaming), and NCU profiler regressions (when profiler
    data is available in both baseline and current run).

    The check fails if *either* metric regresses beyond the tolerance.
    Profiler regressions and variance warnings are reported but do not
    cause failure — they are advisory signals.

    If ncu_summary is provided, it is compared against the baseline's
    NCU data. If not provided, attempts to find NCU data from the most
    recent run in the run store.

    tolerance (a fraction, e.g. 0.15 = 15%) overrides the task's
    regression_tolerance — for noisy environments like shared CI runners
    where the task's locally-tuned tolerance would flake.
    """
    bp = baseline_path or _default_baseline_path(task)
    tol = tolerance if tolerance is not None else task.constraints.regression_tolerance
    metric_name = task.benchmark.metric.name
    metric_mode = task.benchmark.metric.mode
    sec = task.benchmark.secondary_metric

    bench = _run_bench_full(task)
    current_value = metric_value(bench, metric_name)

    # Anti-gaming: check bench variance
    variance_warnings = validate_bench_variance(bench)

    if not bp.exists():
        return CICheckResult(
            passed=True,
            current_value=current_value,
            baseline_value=None,
            regression_pct=None,
            tolerance_pct=tol * 100,
            metric_name=metric_name,
            metric_mode=metric_mode,
            bench_variance_warnings=variance_warnings,
        )

    baseline_data = json.loads(bp.read_text(encoding="utf-8"))
    baseline_value = baseline_data["value"]

    regression_pct, primary_regressed = _check_regression(
        current_value, baseline_value, metric_mode, tol,
    )

    # Secondary metric check
    secondary_result: MetricCheckResult | None = None
    secondary_regressed = False
    if sec and "secondary_value" in baseline_data:
        try:
            sec_current = metric_value(bench, sec.name)
            sec_baseline = baseline_data["secondary_value"]
            sec_regression_pct, sec_regressed = _check_regression(
                sec_current, sec_baseline, sec.mode, tol,
            )
            secondary_regressed = sec_regressed
            secondary_result = MetricCheckResult(
                name=sec.name,
                mode=sec.mode,
                current_value=sec_current,
                baseline_value=sec_baseline,
                regression_pct=sec_regression_pct,
                tolerance_pct=tol * 100,
                regressed=sec_regressed,
            )
        except (KeyError, TypeError):
            pass  # secondary metric not in bench output — skip

    # Profiler regression detection (advisory, does not affect pass/fail)
    profiler_regs: list[ProfilerRegression] = []
    baseline_ncu = baseline_data.get("ncu_summary")
    current_ncu = ncu_summary or _find_latest_ncu_summary(task)
    if baseline_ncu and current_ncu:
        profiler_regs = _detect_profiler_regressions(current_ncu, baseline_ncu)

    return CICheckResult(
        passed=not primary_regressed and not secondary_regressed,
        current_value=current_value,
        baseline_value=baseline_value,
        regression_pct=regression_pct,
        tolerance_pct=tol * 100,
        metric_name=metric_name,
        metric_mode=metric_mode,
        secondary=secondary_result,
        profiler_regressions=profiler_regs,
        bench_variance_warnings=variance_warnings,
    )
