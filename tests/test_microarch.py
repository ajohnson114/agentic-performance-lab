"""Tests for micro-architecture analysis: SASS instruction efficiency,
kernel ceiling, benchmark stability, clock throttle, pipeline heatmap."""
from __future__ import annotations

import pytest

from perflab.analyzers.microarch import (
    build_microarch_summary,
    compute_benchmark_stability,
    detect_clock_throttle,
    format_pipeline_heatmap,
)
from perflab.profilers.ncu_profiler import classify_sass_instructions
from perflab.reporting.roofline import compute_kernel_ceiling

# ---------------------------------------------------------------------------
# SASS instruction classification
# ---------------------------------------------------------------------------

class TestSassInstructionClassification:
    def test_basic_classification(self):
        sass = """\
\t/*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
\t/*0010*/                   S2R R0, SR_CTAID.X ;
\t/*0020*/                   FFMA R2, R5, R6, R2 ;
\t/*0030*/                   FFMA R2, R5, R6, R2 ;
\t/*0040*/                   FFMA R2, R5, R6, R2 ;
\t/*0050*/                   LDG.E R4, [R2.64] ;
\t/*0060*/                   STG.E [R6.64], R4 ;
\t/*0070*/                   BAR.SYNC 0x0 ;
\t/*0080*/                   EXIT ;
"""
        result = classify_sass_instructions(sass)
        assert result["total_instructions"] == 9
        assert result["categories"]["compute"] == 3  # 3 FFMA
        assert result["categories"]["memory_global"] == 2  # LDG + STG
        assert result["categories"]["address_math"] >= 1  # IMAD
        assert result["categories"]["sync"] == 1  # BAR
        assert result["efficiency_pct"] > 0

    def test_tensor_core_kernel(self):
        sass = """\
\t/*0000*/                   HMMA.16816.F32 R4, R0, R2, R4 ;
\t/*0010*/                   HMMA.16816.F32 R8, R0, R6, R8 ;
\t/*0020*/                   LDS R0, [R10] ;
\t/*0030*/                   STS [R12], R4 ;
"""
        result = classify_sass_instructions(sass)
        assert result["categories"]["tensor_core"] == 2
        assert result["categories"]["memory_shared"] == 2
        assert result["efficiency_pct"] == pytest.approx(50.0)  # 2 TC out of 4

    def test_high_overhead_kernel(self):
        sass = """\
\t/*0000*/                   IADD3 R0, R1, R2, R3 ;
\t/*0010*/                   IMAD R4, R0, R5, R6 ;
\t/*0020*/                   ISETP.GT.AND P0, PT, R0, R7, PT ;
\t/*0030*/                   @P0 BRA 0x100 ;
\t/*0040*/                   FFMA R2, R5, R6, R2 ;
"""
        result = classify_sass_instructions(sass)
        assert result["overhead_pct"] > 50  # 3 addr + 1 ctrl out of 5
        assert result["efficiency_pct"] == pytest.approx(20.0)  # 1 compute out of 5

    def test_empty_sass(self):
        assert classify_sass_instructions("") == {}

    def test_predicated_instructions(self):
        sass = """\
\t/*0000*/                   @P0 FFMA R2, R5, R6, R2 ;
\t/*0010*/                   @!P0 MOV R2, RZ ;
"""
        result = classify_sass_instructions(sass)
        assert result["total_instructions"] == 2

    def test_category_pcts_sum_to_100(self):
        sass = """\
\t/*0000*/                   FFMA R2, R5, R6, R2 ;
\t/*0010*/                   LDG.E R4, [R2.64] ;
\t/*0020*/                   IMAD R0, R1, R2, R3 ;
\t/*0030*/                   BAR.SYNC 0x0 ;
\t/*0040*/                   EXIT ;
"""
        result = classify_sass_instructions(sass)
        total_pct = sum(result["category_pcts"].values())
        assert total_pct == pytest.approx(100.0, abs=0.5)


# ---------------------------------------------------------------------------
# Kernel theoretical ceiling
# ---------------------------------------------------------------------------

class TestKernelCeiling:
    def test_basic_ceiling(self):
        result = compute_kernel_ceiling(50.0, 300.0, achieved_tflops=5.0)
        assert result["kernel_ceiling_tflops"] == 150.0
        assert result["pct_of_ceiling"] == pytest.approx(3.3, abs=0.1)
        assert result["pct_of_peak"] == pytest.approx(1.67, abs=0.1)

    def test_low_occupancy_flagged(self):
        result = compute_kernel_ceiling(25.0, 300.0, achieved_tflops=10.0)
        assert result["occupancy_limited"] is True

    def test_high_occupancy_not_flagged(self):
        result = compute_kernel_ceiling(75.0, 300.0, achieved_tflops=100.0)
        assert result["occupancy_limited"] is False

    def test_no_achieved(self):
        result = compute_kernel_ceiling(50.0, 300.0)
        assert result["kernel_ceiling_tflops"] == 150.0
        assert "pct_of_ceiling" not in result


# ---------------------------------------------------------------------------
# Benchmark stability
# ---------------------------------------------------------------------------

class TestBenchmarkStability:
    def test_stable_benchmark(self):
        times = [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0]
        result = compute_benchmark_stability({"times_ms": times})
        assert result is not None
        assert result["is_stable"] is True
        assert result["cv_pct"] < 5.0
        assert "stable" in result["assessment"].lower()

    def test_noisy_benchmark(self):
        times = [10.0, 15.0, 8.0, 20.0, 5.0, 12.0, 18.0, 7.0, 14.0, 9.0]
        result = compute_benchmark_stability({"times_ms": times})
        assert result is not None
        assert result["is_stable"] is False
        assert result["cv_pct"] > 10.0

    def test_insufficient_samples(self):
        assert compute_benchmark_stability({"times_ms": [10.0]}) is None
        assert compute_benchmark_stability({"times_ms": []}) is None
        assert compute_benchmark_stability({}) is None

    def test_min_meaningful_improvement(self):
        times = [10.0, 10.5, 9.5, 10.0, 10.3]
        result = compute_benchmark_stability({"times_ms": times})
        assert result is not None
        assert result["min_meaningful_improvement_pct"] > 0


# ---------------------------------------------------------------------------
# Clock throttle detection
# ---------------------------------------------------------------------------

class TestClockThrottle:
    def test_throttling_detected(self):
        power_data = {
            "gpu_power": {
                "power_samples": [
                    {"watts": 350}, {"watts": 340}, {"watts": 300},
                    {"watts": 280}, {"watts": 270}, {"watts": 290},
                ],
            }
        }
        result = detect_clock_throttle(power_data)
        assert result is not None
        assert result["throttle_detected"] is True
        assert result["power_drop_pct"] > 10

    def test_no_throttling(self):
        power_data = {
            "gpu_power": {
                "power_samples": [
                    {"watts": 300}, {"watts": 298}, {"watts": 302},
                    {"watts": 299}, {"watts": 301},
                ],
            }
        }
        result = detect_clock_throttle(power_data)
        assert result is not None
        assert result["throttle_detected"] is False

    def test_insufficient_samples(self):
        assert detect_clock_throttle({"gpu_power": {"power_samples": []}}) is None
        assert detect_clock_throttle({}) is None


# ---------------------------------------------------------------------------
# Pipeline heatmap
# ---------------------------------------------------------------------------

class TestPipelineHeatmap:
    def test_basic_heatmap(self):
        metrics = {
            "instruction_mix": {
                "fp32_fma": 60.0,
                "int_alu": 20.0,
                "sfu": 5.0,
            },
            "tensor_core_utilization_pct": 0.0,
        }
        result = format_pipeline_heatmap(metrics)
        assert result is not None
        assert "FP32/FMA" in result
        assert "Tensor Core" in result
        assert "MED" in result   # FP32 at 60% is MED (30-70 range)
        assert "LOW" in result   # TC at 0%

    def test_no_data_returns_none(self):
        assert format_pipeline_heatmap({}) is None

    def test_all_pipes_shown(self):
        metrics = {
            "instruction_mix": {
                "fp32_fma": 40.0,
                "fp64": 10.0,
                "int_alu": 15.0,
                "sfu": 5.0,
            },
            "tensor_core_utilization_pct": 30.0,
        }
        result = format_pipeline_heatmap(metrics)
        assert "FP64" in result
        assert "INT/ALU" in result
        assert "SFU" in result


# ---------------------------------------------------------------------------
# Full microarch summary
# ---------------------------------------------------------------------------

class TestMicroarchSummary:
    def test_builds_with_all_data(self):
        bench = {"times_ms": [10.0, 10.1, 9.9], "tflops": {"median": 5.0}}
        summaries = {
            "ncu": {
                "dominant_kernel": {
                    "name": "kern",
                    "achieved_occupancy_pct": 35.0,
                    "instruction_mix": {"fp32_fma": 50.0},
                    "tensor_core_utilization_pct": 0.0,
                },
            },
            "power": {
                "gpu_power": {
                    "power_samples": [{"watts": 300}, {"watts": 295}, {"watts": 290}],
                },
            },
        }
        result = build_microarch_summary(bench, summaries, peak_tflops=300.0)
        assert result is not None
        assert "benchmark_stability" in result
        assert "kernel_ceiling" in result
        assert result["kernel_ceiling"]["occupancy_limited"] is True
        assert "pipeline_heatmap" in result

    def test_builds_with_minimal_data(self):
        bench = {"times_ms": [10.0, 10.1, 9.9]}
        result = build_microarch_summary(bench, {})
        assert result is not None
        assert "benchmark_stability" in result

    def test_returns_none_with_no_data(self):
        result = build_microarch_summary({}, {})
        assert result is None


# ---------------------------------------------------------------------------
# Prompt integration
# ---------------------------------------------------------------------------

class TestMicroarchInPrompt:
    def test_microarch_renders_in_prompt(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"kern.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 5.0}},
            roofline=None,
            history=[],
            allowed_paths=["kern.cu"],
            program_type="cuda",
            microarch_summary={
                "kernel_ceiling": {
                    "occupancy_pct": 35.0,
                    "kernel_ceiling_tflops": 105.0,
                    "peak_tflops": 300.0,
                    "achieved_tflops": 5.0,
                    "pct_of_ceiling": 4.8,
                    "pct_of_peak": 1.7,
                    "occupancy_limited": True,
                },
                "benchmark_stability": {
                    "cv_pct": 2.1,
                    "is_stable": True,
                    "assessment": "Very stable (CV=2.1%) — improvements >1% are real",
                },
                "pipeline_heatmap": "Pipeline utilization:\n  FP32/FMA  ██████░░░░  60.0% [MED]",
            },
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "Micro-architecture" in full_text
        assert "105.0 TFLOPS" in full_text
        assert "Occupancy is the primary limiter" in full_text
        assert "Very stable" in full_text
        assert "Pipeline utilization" in full_text
