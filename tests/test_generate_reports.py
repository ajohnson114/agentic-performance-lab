"""Tests for perflab.reporting.generate."""
from __future__ import annotations

import json
from pathlib import Path

from perflab.reporting.generate import ReportParams, generate_reports


def _make_run_dir(tmp_path: Path, *, summaries=None, baseline_summaries=None,
                  system_info=None, bench=None):
    """Create a minimal run directory structure."""
    run_dir = tmp_path / "run-001"
    run_dir.mkdir()
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()

    if summaries:
        for name, data in summaries.items():
            (artifacts / f"{name}_summary.json").write_text(
                json.dumps(data), encoding="utf-8"
            )

    if baseline_summaries:
        baseline_dir = run_dir / "artifacts_baseline"
        baseline_dir.mkdir()
        for name, data in baseline_summaries.items():
            (baseline_dir / f"{name}_summary.json").write_text(
                json.dumps(data), encoding="utf-8"
            )

    if system_info:
        (run_dir / "system_info.json").write_text(
            json.dumps(system_info), encoding="utf-8"
        )

    if bench:
        (run_dir / "bench.json").write_text(
            json.dumps(bench), encoding="utf-8"
        )

    return run_dir


def _minimal_history():
    return [
        {"iteration": 0, "value": 100.0, "accepted": True, "notes": "baseline"},
        {"iteration": 1, "value": 120.0, "accepted": True, "notes": "improved"},
    ]


def _minimal_params(run_dir: Path, **overrides) -> ReportParams:
    """Build a ReportParams with sensible defaults, overridable via kwargs."""
    defaults = dict(
        run_dir=run_dir,
        run_id="run-001",
        task_name="matmul",
        metric_name="gflops",
        metric_mode="maximize",
        program_type="cpp",
        history=_minimal_history(),
        baseline_val=100.0,
        best_value=120.0,
        best_iter=1,
    )
    defaults.update(overrides)
    return ReportParams(**defaults)


class TestGenerateReports:
    def test_minimal_run(self, tmp_path: Path):
        run_dir = _make_run_dir(tmp_path)
        result = generate_reports(_minimal_params(run_dir))
        assert (run_dir / "dashboard.html").exists()
        assert (run_dir / "report.json").exists()
        assert (run_dir / "report.md").exists()
        assert isinstance(result, dict)

    def test_report_json_structure(self, tmp_path: Path):
        run_dir = _make_run_dir(tmp_path)
        result = generate_reports(_minimal_params(run_dir))
        for key in ("task_name", "run_id", "metric_name", "metric_mode",
                     "best_value", "best_iter", "baseline_value", "rows",
                     "bottleneck_diagnoses", "run_summary", "latest_artifacts"):
            assert key in result

    def test_bottleneck_diagnoses_in_report(self, tmp_path: Path):
        run_dir = _make_run_dir(tmp_path, summaries={
            "ncu": {"sm_utilization_pct": 15},
        })
        result = generate_reports(_minimal_params(run_dir, program_type="cuda"))
        assert len(result["bottleneck_diagnoses"]) > 0
        assert result["bottleneck_diagnoses"][0]["rank"] == 1

    def test_build_flag_recs_passed_through(self, tmp_path: Path):
        run_dir = _make_run_dir(
            tmp_path,
            system_info={"cpu_isa": {"avx2": True, "max_simd_width_bits": 256}},
        )
        result = generate_reports(_minimal_params(
            run_dir,
            build_cmd="g++ -O2 -o matmul matmul.cpp",
        ))
        # Build flag recs go into the dashboard HTML, not report.json directly
        html = (run_dir / "dashboard.html").read_text()
        assert "-march=native" in html or "-O3" in html

    def test_profile_diff_with_baseline(self, tmp_path: Path):
        run_dir = _make_run_dir(
            tmp_path,
            summaries={"linux_perf": {"ipc": 1.5, "cache_miss_rate": 0.02}},
            baseline_summaries={"linux_perf": {"ipc": 0.8, "cache_miss_rate": 0.10}},
        )
        result = generate_reports(_minimal_params(run_dir))
        html = (run_dir / "dashboard.html").read_text()
        assert "Profile diff" in html or "profile diff" in html.lower() or "ipc" in html

    def test_no_artifacts_dir(self, tmp_path: Path):
        run_dir = tmp_path / "run-empty"
        run_dir.mkdir()
        # No artifacts/ directory at all
        result = generate_reports(ReportParams(
            run_dir=run_dir,
            run_id="run-empty",
            task_name="test",
            metric_name="time_s",
            metric_mode="minimize",
            program_type="cpp",
            history=[{"iteration": 0, "value": 1.0, "accepted": True}],
            baseline_val=1.0,
            best_value=1.0,
            best_iter=0,
        ))
        assert result["bottleneck_diagnoses"] == []
        assert (run_dir / "report.json").exists()
