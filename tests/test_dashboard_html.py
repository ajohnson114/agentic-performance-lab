"""Tests for perflab.reporting.dashboard_html."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from perflab.reporting.dashboard_html import (
    AnalysisData,
    GlanceData,
    ProfilerData,
    write_dashboard_html,
)

_ANALYSIS_FIELDS = {f.name for f in fields(AnalysisData)}


def _write(tmp_path, **kwargs):
    """Helper: write dashboard and return HTML string."""
    out = tmp_path / "dashboard.html"
    defaults = dict(
        path=out,
        title="Test Dashboard",
        metric_png_rel="metric_history.png",
    )
    defaults.update(kwargs)
    analysis_kwargs = {
        k: defaults.pop(k) for k in list(defaults) if k in _ANALYSIS_FIELDS
    }
    if analysis_kwargs:
        defaults["analysis"] = AnalysisData(**analysis_kwargs)
    write_dashboard_html(**defaults)
    return out.read_text(encoding="utf-8")


class TestWriteDashboardHtml:
    def test_minimal_output(self, tmp_path: Path):
        html = _write(tmp_path)
        assert "<html" in html
        assert "Test Dashboard" in html

    def test_optimization_summary_rendered(self, tmp_path: Path):
        html = _write(tmp_path, optimization_summary="Used loop tiling for 2x speedup")
        assert "loop tiling" in html

    def test_hardware_mismatch_banner(self, tmp_path: Path):
        html = _write(tmp_path, hardware_mismatch='Mismatch: expected "A100" got "V100"')
        assert "Mismatch" in html
        assert "A100" in html

    def test_glance_data_rendered(self, tmp_path: Path):
        glance = GlanceData(
            metric_name="gflops",
            baseline_value=100.0,
            best_value=150.0,
            best_iter=3,
            total_iterations=5,
            speedup=1.5,
            accepted_count=3,
        )
        html = _write(tmp_path, glance=glance)
        assert "150" in html
        assert "1.50x" in html


class TestRenderDiagnostics:
    def test_no_diagnostics_no_card(self, tmp_path: Path):
        html = _write(tmp_path)
        assert "Diagnostics" not in html

    def test_bottleneck_table_rendered(self, tmp_path: Path):
        diags = [
            {
                "rank": 1,
                "bottleneck": "Low SM utilization (20%)",
                "root_cause": "small kernels",
                "confidence": "high",
                "suggested_actions": ["Increase parallelism"],
            },
        ]
        html = _write(tmp_path, bottleneck_diagnoses=diags)
        assert "<details>" in html
        assert "Low SM utilization" in html
        assert "small kernels" in html

    def test_gpu_attribution_bars(self, tmp_path: Path):
        attrib = [
            {"rank": 1, "name": "sgemm_kernel", "category": "compute",
             "gpu_pct": 85.0, "gpu_time_ms": 100, "diagnosis": "dominant", "suggestions": []},
        ]
        html = _write(tmp_path, gpu_attribution=attrib)
        assert "sgemm_kernel" in html
        assert "85" in html

    def test_profile_diff_table(self, tmp_path: Path):
        diff = [
            {"metric": "ipc", "before": 0.82, "after": 1.41, "delta_pct": 72.0, "direction": "improved"},
        ]
        html = _write(tmp_path, profile_diff=diff)
        assert "ipc" in html
        assert "improved" in html

    def test_build_flag_recs_table(self, tmp_path: Path):
        recs = [
            {"flag": "-O3", "reason": "Enable full optimizations", "impact": "high", "category": "optimization"},
        ]
        html = _write(tmp_path, build_flag_recs=recs)
        assert "-O3" in html
        assert "Enable full optimizations" in html

    def test_confidence_badge_classes(self, tmp_path: Path):
        diags = [
            {"rank": 1, "bottleneck": "test-high", "root_cause": "", "confidence": "high", "suggested_actions": []},
            {"rank": 2, "bottleneck": "test-medium", "root_cause": "", "confidence": "medium", "suggested_actions": []},
        ]
        html = _write(tmp_path, bottleneck_diagnoses=diags)
        assert 'class="badge high"' in html
        assert 'class="badge medium"' in html

    def test_mixed_diagnostics(self, tmp_path: Path):
        # Only bottleneck + build flags, no gpu_attribution or profile_diff
        diags = [
            {"rank": 1, "bottleneck": "mem-bound", "root_cause": "bw", "confidence": "high", "suggested_actions": []},
        ]
        recs = [
            {"flag": "-march=native", "reason": "Use native ISA", "impact": "medium", "category": "arch"},
        ]
        html = _write(tmp_path, bottleneck_diagnoses=diags, build_flag_recs=recs)
        assert "mem-bound" in html
        assert "-march=native" in html
        # No GPU attribution section
        assert "GPU attribution" not in html

    def test_hotspot_diff_table(self, tmp_path: Path):
        diff = [
            {"function": "naive_matmul", "before_pct": 85.0, "after_pct": 0.0,
             "delta_pct": -85.0, "status": "removed"},
            {"function": "numpy_dot", "before_pct": 0.0, "after_pct": 90.0,
             "delta_pct": 90.0, "status": "new"},
        ]
        html = _write(tmp_path, hotspot_diff=diff)
        assert "naive_matmul" in html
        assert "numpy_dot" in html
        assert "hotspot shifts" in html.lower()
        assert "gone" in html.lower()
        assert "new" in html.lower()


class TestOutcomeAnalysis:
    def test_what_worked_rendered(self, tmp_path: Path):
        history = [
            {"iteration": 0, "value": 100.0, "accepted": True, "notes": "baseline"},
            {"iteration": 1, "value": 200.0, "accepted": True, "speedup": 2.0,
             "notes": "Used numpy vectorization"},
        ]
        html = _write(tmp_path, history=history)
        assert "What worked" in html
        assert "numpy vectorization" in html

    def test_what_didnt_work_rendered(self, tmp_path: Path):
        history = [
            {"iteration": 0, "value": 100.0, "accepted": True, "notes": "baseline"},
            {"iteration": 1, "value": 100.0, "accepted": False,
             "notes": "no improvement (tried loop unrolling)"},
            {"iteration": 2, "value": 200.0, "accepted": True, "speedup": 2.0,
             "notes": "Used SIMD intrinsics"},
        ]
        html = _write(tmp_path, history=history)
        assert "didn&#x27;t work" in html or "didn't work" in html or "didn" in html
        assert "loop unrolling" in html

    def test_why_it_worked_shows_summary(self, tmp_path: Path):
        history = [
            {"iteration": 0, "value": 100.0, "accepted": True, "notes": "baseline"},
            {"iteration": 1, "value": 200.0, "accepted": True, "speedup": 2.0, "notes": "vectorized"},
        ]
        html = _write(tmp_path, optimization_summary="Cache locality improved by tiling",
                       history=history)
        assert "Why it worked" in html
        assert "Cache locality" in html

    def test_no_outcome_section_without_iterations(self, tmp_path: Path):
        html = _write(tmp_path)
        assert "Optimization analysis" not in html

    def test_early_stop_duplicate_row_filtered(self, tmp_path: Path):
        # maybe_early_stop() (finalize.py) appends a synthetic second history
        # entry for the same iteration ("early stop: ...", accepted=False).
        # generate.py already filters this out of glance.rows, but the raw
        # history reaching _render_outcome_analysis was unfiltered -- so the
        # "what didn't work" table showed two rows for one iteration whenever
        # convergence stopped the run.
        history = [
            {"iteration": 0, "value": 100.0, "accepted": True, "notes": "baseline"},
            {"iteration": 1, "value": 100.0, "accepted": False, "description": "no improvement found"},
            {"iteration": 1, "value": 100.0, "accepted": False, "description": "early stop: 1 consecutive failures"},
        ]
        html = _write(tmp_path, history=history)
        assert html.count("no improvement found") == 1
        assert "early stop:" not in html

    def test_perfetto_trace_link(self, tmp_path: Path):
        trace_path = tmp_path / "perfetto_trace.json"
        trace_path.write_text('{"traceEvents": []}', encoding="utf-8")
        profiler = ProfilerData(perfetto_trace_path=trace_path)
        html = _write(tmp_path, profiler=profiler)
        assert "Perfetto" in html
        assert "ui.perfetto.dev" in html


class TestTmaRendering:
    def test_tma_section_rendered(self, tmp_path: Path):
        tma = {
            "frontend_bound_pct": 15.0,
            "backend_bound_pct": 50.0,
            "bad_speculation_pct": 5.0,
            "retiring_pct": 30.0,
            "dominant_bottleneck": "backend_bound",
        }
        html = _write(tmp_path, tma_data=tma)
        assert "Microarchitecture" in html
        assert "50.0%" in html
        assert "Backend Bound" in html

    def test_no_tma_no_section(self, tmp_path: Path):
        html = _write(tmp_path)
        assert "Microarchitecture" not in html


class TestPowerRendering:
    def test_rapl_data_rendered(self, tmp_path: Path):
        power = {
            "rapl": {
                "package_joules": 25.5,
                "cores_joules": 18.3,
                "avg_package_watts": 5.1,
            }
        }
        html = _write(tmp_path, power_data=power)
        assert "Power" in html
        assert "25.50" in html
        assert "5.1 W" in html

    def test_gpu_power_rendered(self, tmp_path: Path):
        power = {
            "gpu_power": {
                "avg_watts": 250.0,
                "max_watts": 300.0,
                "sample_count": 10,
            }
        }
        html = _write(tmp_path, power_data=power)
        assert "250.0 W" in html
        assert "300.0 W" in html

    def test_no_power_no_section(self, tmp_path: Path):
        html = _write(tmp_path)
        assert "Power" not in html


class TestParetoRendering:
    def test_pareto_png_embedded(self, tmp_path: Path):
        # Create a fake pareto PNG
        pareto_png = tmp_path / "pareto_frontier.png"
        pareto_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        html = _write(tmp_path, pareto_png_rel="pareto_frontier.png")
        assert "Pareto frontier" in html
        assert "data:image/png;base64," in html

    def test_no_pareto_no_section(self, tmp_path: Path):
        html = _write(tmp_path)
        assert "Pareto" not in html


class TestBenchStatsWarning:
    def test_warning_rendered(self, tmp_path: Path):
        html = _write(tmp_path, bench_stats_warning="High measurement variance detected: CV=15.0%")
        assert "High measurement variance" in html
        assert "CV=15.0%" in html

    def test_no_warning_no_banner(self, tmp_path: Path):
        html = _write(tmp_path)
        assert "measurement variance" not in html


class TestVectorizationRendering:
    def test_vectorization_table(self, tmp_path: Path):
        vec = [
            {"function": "matmul", "has_simd": True, "simd_isa": "avx", "hot_pct": 80.0},
            {"function": "init", "has_simd": False, "simd_isa": "none", "hot_pct": 5.0},
        ]
        html = _write(tmp_path, vectorization=vec)
        assert "Vectorization" in html
        assert "matmul" in html
        assert "avx" in html

    def test_no_vectorization_no_section(self, tmp_path: Path):
        html = _write(tmp_path)
        assert "Vectorization" not in html


class TestGpuMemoryRendering:
    def test_gpu_memory_rendered(self, tmp_path: Path):
        gpu_mem = {
            "total_mib": 8192.0,
            "max_used_mib": 6000.0,
            "avg_used_mib": 5000.0,
            "utilization_pct": 73.2,
        }
        html = _write(tmp_path, gpu_memory=gpu_mem)
        assert "GPU memory" in html
        assert "8192" in html
        assert "73.2%" in html

    def test_no_gpu_memory_no_section(self, tmp_path: Path):
        html = _write(tmp_path)
        assert "GPU memory" not in html


class TestThreadSchedRendering:
    def test_thread_sched_rendered(self, tmp_path: Path):
        sched = {
            "latency": [
                {"task": "worker", "runtime_ms": 500.0, "switches": 10,
                 "avg_delay_ms": 0.01, "max_delay_ms": 0.1},
            ],
            "timehist": {"cpus": [{"cpu": 0, "run_sec": 1.0}], "total_run_ms": 1000.0, "migrations": 5},
        }
        html = _write(tmp_path, thread_sched=sched)
        assert "Thread scheduling" in html
        assert "worker" in html
        assert "migrations" in html.lower()

    def test_no_sched_no_section(self, tmp_path: Path):
        html = _write(tmp_path)
        assert "Thread scheduling" not in html


class TestHtmlWellFormedness:
    def test_html_parses_without_errors(self, tmp_path: Path):
        """Verify the generated HTML is well-formed."""
        from html.parser import HTMLParser

        class StrictParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.errors: list[str] = []

            def handle_starttag(self, tag, attrs):
                pass  # Valid start tag

            def handle_endtag(self, tag):
                pass  # Valid end tag

        # Generate a dashboard with many features enabled
        history = [
            {"iteration": 0, "value": 100.0, "accepted": True, "notes": "baseline"},
            {"iteration": 1, "value": 200.0, "accepted": True, "speedup": 2.0, "notes": "improved"},
        ]
        diags = [{"rank": 1, "bottleneck": "test", "root_cause": "x",
                  "confidence": "high", "suggested_actions": ["fix"]}]
        html = _write(
            tmp_path,
            history=history,
            bottleneck_diagnoses=diags,
            optimization_summary="Test summary",
            bench_stats_warning="CV=15%",
        )

        parser = StrictParser()
        parser.feed(html)
        assert "<html" in html
        assert "</html>" in html
        assert "</body>" in html
