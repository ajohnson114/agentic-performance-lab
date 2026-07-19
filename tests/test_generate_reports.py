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
        generate_reports(_minimal_params(
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
        generate_reports(_minimal_params(run_dir))
        html = (run_dir / "dashboard.html").read_text()
        assert "Profile diff" in html or "profile diff" in html.lower() or "ipc" in html

    def test_roofline_peaks_reach_report_md(self, tmp_path: Path):
        # generate.py computes roofline_peaks locally (for the HTML glance
        # object) but must also thread it into report_data so report.md's
        # Roofline section (which reads data["roofline_peaks"]) isn't dead.
        run_dir = _make_run_dir(tmp_path)
        result = generate_reports(_minimal_params(
            run_dir,
            roofline_peaks={
                "peak_tflops": 19.5,
                "peak_mem_bw_gbs": 900.0,
                "source": "known_gpu_spec",
                "device": "A100",
            },
        ))
        assert result.get("roofline_peaks", {}).get("peak_tflops") == 19.5
        text = (run_dir / "report.md").read_text(encoding="utf-8")
        assert "## Roofline" in text
        assert "19.500" in text
        assert "900.0" in text
        assert "known_gpu_spec" in text
        assert "A100" in text

    def test_hardware_mismatch_fallback_checks_all_gpus(self, tmp_path: Path):
        # Multi-GPU box where the target only matches the *second* detected
        # GPU. The old code compared against nvidia_gpus[0] only (via a
        # single detected_hardware string) and would wrongly show a mismatch
        # banner. hardware_mismatch isn't provided here (this ReportParams
        # site mirrors orchestrator.py's profile-only path, which has no
        # ctx.hardware_mismatch to pass through), so generate.py must fall
        # back to recomputing -- looping every GPU in system_info.
        run_dir = _make_run_dir(tmp_path, system_info={
            "nvidia_gpus": [
                {"name": "NVIDIA A100-SXM4-80GB"},
                {"name": "NVIDIA H100-PCIE-80GB"},
            ],
        })
        generate_reports(_minimal_params(run_dir, target_hardware="H100"))
        html = (run_dir / "dashboard.html").read_text(encoding="utf-8")
        assert '<div class="hardware-warning">' not in html

    def test_hardware_mismatch_fallback_flags_true_mismatch(self, tmp_path: Path):
        run_dir = _make_run_dir(tmp_path, system_info={
            "nvidia_gpus": [
                {"name": "NVIDIA A100-SXM4-80GB"},
                {"name": "NVIDIA V100-SXM2-16GB"},
            ],
        })
        generate_reports(_minimal_params(run_dir, target_hardware="H100"))
        html = (run_dir / "dashboard.html").read_text(encoding="utf-8")
        assert '<div class="hardware-warning">' in html
        assert "H100" in html

    def test_hardware_mismatch_prefers_explicit_ctx_value(self, tmp_path: Path):
        # When the caller (finalize.py) already resolved ctx.hardware_mismatch
        # (looping all GPUs itself), generate.py must use it as-is rather than
        # recomputing from system_info.
        run_dir = _make_run_dir(tmp_path, system_info={
            "nvidia_gpus": [{"name": "NVIDIA H100-PCIE-80GB"}],
        })
        generate_reports(_minimal_params(
            run_dir,
            target_hardware="H100",
            hardware_mismatch='Hardware mismatch: task targets "H100" but detected GPU is "A100"',
        ))
        html = (run_dir / "dashboard.html").read_text(encoding="utf-8")
        assert '<div class="hardware-warning">' in html
        assert "detected GPU is" in html

    def test_llm_estimated_cost_reaches_dashboard(self, tmp_path: Path):
        run_dir = _make_run_dir(tmp_path)
        generate_reports(_minimal_params(
            run_dir,
            llm_stats={
                "model": "claude-opus-4-8",
                "provider": "anthropic",
                "total_calls": 3,
                "total_input_tokens": 1_000_000,
                "total_output_tokens": 200_000,
                "total_llm_latency_s": 12.0,
                "estimated_cost_usd": 10.0,
            },
        ))
        html = (run_dir / "dashboard.html").read_text(encoding="utf-8")
        assert "Est. Cost" in html
        assert "$10.00" in html

    def test_llm_unknown_model_cost_shows_unknown_marker(self, tmp_path: Path):
        run_dir = _make_run_dir(tmp_path)
        generate_reports(_minimal_params(
            run_dir,
            llm_stats={
                "model": "some-unpriced-model",
                "provider": "openai",
                "total_calls": 1,
                "total_input_tokens": 100,
                "total_output_tokens": 100,
                "total_llm_latency_s": 1.0,
                "estimated_cost_usd": None,
            },
        ))
        html = (run_dir / "dashboard.html").read_text(encoding="utf-8")
        assert "n/a (unknown model pricing)" in html

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
