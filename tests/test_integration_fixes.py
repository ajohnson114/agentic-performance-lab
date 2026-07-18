"""Regression tests for bug fixes and profiler key integrations."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Power samples bug regression test
# ---------------------------------------------------------------------------

class TestPowerSamplesBugFix:
    """Regression test: power_profiler must produce power_samples list
    that microarch.py and bottleneck_analyzer.py consume."""

    def test_compute_gpu_power_stats_includes_samples(self):
        from perflab.profilers.power_profiler import _compute_gpu_power_stats
        samples = [300.0, 290.0, 280.0, 310.0, 295.0]
        result = _compute_gpu_power_stats(samples)

        # REGRESSION: power_samples must be present
        assert "power_samples" in result, (
            "BUG REGRESSION: _compute_gpu_power_stats must include 'power_samples' list. "
            "microarch.py and bottleneck_analyzer.py depend on it for clock throttle detection."
        )
        assert isinstance(result["power_samples"], list)
        assert len(result["power_samples"]) == 5
        assert all("watts" in s for s in result["power_samples"])
        assert result["power_samples"][0]["watts"] == 300.0

    def test_compute_gpu_power_stats_still_has_aggregates(self):
        from perflab.profilers.power_profiler import _compute_gpu_power_stats
        samples = [300.0, 290.0, 310.0]
        result = _compute_gpu_power_stats(samples)

        assert "avg_watts" in result
        assert "min_watts" in result
        assert "max_watts" in result
        assert "sample_count" in result
        assert result["sample_count"] == 3

    def test_empty_samples(self):
        from perflab.profilers.power_profiler import _compute_gpu_power_stats
        assert _compute_gpu_power_stats([]) == {}

    def test_microarch_clock_throttle_with_power_samples(self):
        """Verify microarch can detect throttling from power_samples."""
        from perflab.analyzers.microarch import detect_clock_throttle
        power_data = {
            "gpu_power": {
                "power_samples": [
                    {"watts": 350}, {"watts": 340}, {"watts": 300},
                    {"watts": 270}, {"watts": 260}, {"watts": 280},
                ],
            }
        }
        result = detect_clock_throttle(power_data)
        assert result is not None
        assert result["throttle_detected"] is True

    def test_microarch_clock_throttle_fallback_to_aggregates(self):
        """Verify microarch falls back to aggregates when power_samples is missing."""
        from perflab.analyzers.microarch import detect_clock_throttle
        power_data = {
            "gpu_power": {
                "min_watts": 260.0,
                "max_watts": 350.0,
                "avg_watts": 300.0,
                # No power_samples list
            }
        }
        result = detect_clock_throttle(power_data)
        assert result is not None
        assert result["throttle_detected"] is True


# ---------------------------------------------------------------------------
# L1/L2 hit rates in kernel dossier
# ---------------------------------------------------------------------------

class TestL1L2InDossier:
    def test_low_l1_hit_rate_shown_in_dossier(self):
        from perflab.analyzers.gpu_attribution import KernelDossier
        from perflab.optimizers.prompt import PromptContext, build_prompt

        dossier = KernelDossier(
            name="sgemm",
            gpu_pct=85.0,
            gpu_time_ms=120.0,
            ncu_metrics={
                "sm_utilization_pct": 50.0,
                "memory_throughput_pct": 70.0,
                "compute_throughput_pct": 30.0,
                "l1_hit_rate": 45.0,
                "l2_hit_rate": 60.0,
            },
        )
        ctx = PromptContext(
            source_files={"k.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["k.cu"],
            program_type="cuda",
            kernel_dossiers=[dossier],
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        # L1 bottleneck detected (hit rate 45%)
        assert "L1" in full_text and "45" in full_text
        assert "bottleneck" in full_text.lower()

    def test_high_hit_rates_not_shown(self):
        from perflab.analyzers.gpu_attribution import KernelDossier
        from perflab.optimizers.prompt import PromptContext, build_prompt

        dossier = KernelDossier(
            name="sgemm",
            gpu_pct=85.0,
            gpu_time_ms=120.0,
            ncu_metrics={
                "sm_utilization_pct": 50.0,
                "l1_hit_rate": 95.0,
                "l2_hit_rate": 90.0,
            },
        )
        ctx = PromptContext(
            source_files={"k.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["k.cu"],
            program_type="cuda",
            kernel_dossiers=[dossier],
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        # High hit rates should NOT be flagged as issues
        assert "L1 hit" not in full_text
        assert "L2 hit" not in full_text


# ---------------------------------------------------------------------------
# PyTorch FLOPS in prompt
# ---------------------------------------------------------------------------

class TestPytorchFlopsInPrompt:
    def test_flops_shown_when_available(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"model.py": "code"},
            profiler_summaries={
                "torch_profiler": {
                    "total_tflops": 0.5432,
                    "top_ops_by_flops": [
                        {"name": "aten::mm", "flops": 5e11, "pct": 80.0},
                        {"name": "aten::add", "flops": 1e11, "pct": 16.0},
                    ],
                },
            },
            bench_results={"tokens_per_sec": {"median": 1000}},
            roofline=None,
            history=[],
            allowed_paths=["model.py"],
            program_type="pytorch",
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "0.5432 TFLOPS" in full_text
        assert "aten::mm" in full_text
        assert "80%" in full_text

    def test_no_flops_no_section(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"model.py": "code"},
            profiler_summaries={"torch_profiler": {}},
            bench_results={"tokens_per_sec": {"median": 1000}},
            roofline=None,
            history=[],
            allowed_paths=["model.py"],
            program_type="pytorch",
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "operator FLOPS" not in full_text


# ---------------------------------------------------------------------------
# CI profiler regression detection
# ---------------------------------------------------------------------------

class TestCIProfilerRegression:
    def test_detect_tc_util_regression(self):
        from perflab.ci import _detect_profiler_regressions
        current = {"tensor_core_utilization_pct": 20.0}
        baseline = {"tensor_core_utilization_pct": 50.0}
        regressions = _detect_profiler_regressions(current, baseline)
        assert len(regressions) >= 1
        assert regressions[0].metric == "tensor_core_utilization_pct"
        assert regressions[0].direction == "decreased"

    def test_detect_stall_increase(self):
        from perflab.ci import _detect_profiler_regressions
        current = {"dominant_stall_pct": 50.0}
        baseline = {"dominant_stall_pct": 20.0}
        regressions = _detect_profiler_regressions(current, baseline)
        assert len(regressions) >= 1
        assert regressions[0].direction == "increased"

    def test_no_regression_when_improved(self):
        from perflab.ci import _detect_profiler_regressions
        current = {"sm_utilization_pct": 80.0, "tensor_core_utilization_pct": 60.0}
        baseline = {"sm_utilization_pct": 40.0, "tensor_core_utilization_pct": 10.0}
        regressions = _detect_profiler_regressions(current, baseline)
        assert len(regressions) == 0

    def test_empty_data(self):
        from perflab.ci import _detect_profiler_regressions
        assert _detect_profiler_regressions({}, {}) == []
