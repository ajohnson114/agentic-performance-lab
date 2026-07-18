"""Tests for MPS cross-profiler CPU/GPU join in bottleneck_analyzer."""
from __future__ import annotations

from perflab.analyzers.bottleneck_analyzer import (
    AnalysisThresholds,
    _analyze_cross_profiler_cpu_gpu,
    diagnose_bottlenecks,
)


def _make_torch_summary(top_ops_us: list[float]) -> dict:
    """Build a minimal torch_profiler summary with given op times in microseconds."""
    return {
        "top_ops": [{"name": f"op_{i}", "total_us": us} for i, us in enumerate(top_ops_us)],
    }


def _make_metal_summary(gpu_time_total_ms: float, duration_s: float) -> dict:
    return {"gpu_time_total_ms": gpu_time_total_ms, "duration_s": duration_s}


class TestCrossProfilerCpuGpu:
    """Direct tests for _analyze_cross_profiler_cpu_gpu."""

    def test_cpu_bound_low_ratio(self):
        """Low GPU/CPU ratio triggers CPU-dispatch bottleneck finding."""
        # CPU: 10000 us total, GPU: 1 ms = 1000 us -> ratio = 0.1 (< 0.5)
        torch_s = _make_torch_summary([5000, 5000])
        metal_s = _make_metal_summary(gpu_time_total_ms=1, duration_s=1)
        thresholds = AnalysisThresholds()

        findings = _analyze_cross_profiler_cpu_gpu(torch_s, metal_s, thresholds)

        cpu_bound = [f for f in findings if "CPU dispatch" in f.bottleneck or "GPU/CPU ratio" in f.bottleneck]
        assert len(cpu_bound) >= 1
        assert cpu_bound[0].confidence == "high"

    def test_gpu_underutilization(self):
        """Low gpu_util triggers MPS GPU active finding."""
        # GPU: 100 ms over 10s duration -> gpu_util = 0.01 (< 0.3)
        torch_s = _make_torch_summary([100])
        metal_s = _make_metal_summary(gpu_time_total_ms=100, duration_s=10)
        thresholds = AnalysisThresholds()

        findings = _analyze_cross_profiler_cpu_gpu(torch_s, metal_s, thresholds)

        gpu_findings = [f for f in findings if "MPS GPU active" in f.bottleneck]
        assert len(gpu_findings) >= 1
        assert gpu_findings[0].confidence == "medium"

    def test_writeback_cpu_vs_gpu(self):
        """cpu_vs_gpu dict is written back into torch_summary."""
        torch_s = _make_torch_summary([2000, 3000])
        metal_s = _make_metal_summary(gpu_time_total_ms=10, duration_s=1)
        thresholds = AnalysisThresholds()

        _analyze_cross_profiler_cpu_gpu(torch_s, metal_s, thresholds)

        assert "cpu_vs_gpu" in torch_s
        info = torch_s["cpu_vs_gpu"]
        assert info["total_cpu_op_us"] == 5000.0
        assert info["total_gpu_kernel_us"] == 10000.0  # 10 ms * 1000
        assert info["source"] == "cross_profiler_mps"
        assert isinstance(info["ratio"], float)

    def test_no_findings_both_zero(self):
        """No findings when both CPU and GPU time are zero."""
        torch_s = _make_torch_summary([0, 0])
        metal_s = _make_metal_summary(gpu_time_total_ms=0, duration_s=1)
        thresholds = AnalysisThresholds()

        findings = _analyze_cross_profiler_cpu_gpu(torch_s, metal_s, thresholds)

        assert findings == []
        # cpu_vs_gpu should NOT be written back when early-returning
        assert "cpu_vs_gpu" not in torch_s

    def test_no_findings_empty_ops(self):
        """No findings when top_ops is empty (total_cpu_us = 0) and gpu is also 0."""
        torch_s = {"top_ops": []}
        metal_s = _make_metal_summary(gpu_time_total_ms=0, duration_s=1)
        thresholds = AnalysisThresholds()

        findings = _analyze_cross_profiler_cpu_gpu(torch_s, metal_s, thresholds)
        assert findings == []

    def test_no_findings_healthy_ratio(self):
        """No findings when GPU/CPU ratio is above threshold and utilization is good."""
        # CPU: 1000 us, GPU: 5 ms = 5000 us -> ratio = 5.0 (>> 0.5)
        # gpu_util = 5 ms / 10 ms = 0.5 (> 0.3)
        torch_s = _make_torch_summary([1000])
        metal_s = _make_metal_summary(gpu_time_total_ms=5, duration_s=0.01)
        thresholds = AnalysisThresholds()

        findings = _analyze_cross_profiler_cpu_gpu(torch_s, metal_s, thresholds)

        assert findings == []

    def test_both_findings_triggered(self):
        """Both CPU-bound and GPU underutilization can fire together."""
        # CPU: 100000 us, GPU: 5 ms = 5000 us -> ratio = 0.05 (< 0.5)
        # gpu_util = 5 ms / 1000 ms = 0.005 (< 0.3)
        torch_s = _make_torch_summary([100000])
        metal_s = _make_metal_summary(gpu_time_total_ms=5, duration_s=1)
        thresholds = AnalysisThresholds()

        findings = _analyze_cross_profiler_cpu_gpu(torch_s, metal_s, thresholds)

        assert len(findings) == 2

    def test_custom_thresholds(self):
        """Custom thresholds change what triggers findings."""
        # ratio = 0.4, default threshold 0.5 would trigger; custom 0.3 does not
        # gpu_util = 0.4 ms / 1 s = 0.0004, default threshold 0.3 would trigger; custom 0.0001 does not
        torch_s = _make_torch_summary([1000])
        metal_s = _make_metal_summary(gpu_time_total_ms=0.4, duration_s=1)
        thresholds = AnalysisThresholds(cross_gpu_cpu_ratio_low=0.3, cross_gpu_util_low=0.0001)

        findings = _analyze_cross_profiler_cpu_gpu(torch_s, metal_s, thresholds)

        assert findings == []


class TestCrossProfilerIntegration:
    """Integration: diagnose_bottlenecks calls cross-profiler when both profilers present."""

    def test_diagnose_bottlenecks_calls_cross_profiler(self):
        """When both torch_profiler and metal_trace are present (and torch lacks GPU data),
        cross-profiler findings appear in output."""
        summaries = {
            "torch_profiler": _make_torch_summary([10000, 10000]),
            "metal_trace": _make_metal_summary(gpu_time_total_ms=1, duration_s=1),
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")

        cross = [d for d in diags if "cross_profiler" in d.root_cause.lower() or "GPU/CPU ratio" in d.bottleneck]
        assert len(cross) >= 1

    def test_diagnose_bottlenecks_skips_when_torch_has_gpu_data(self):
        """Cross-profiler is skipped when torch_profiler already has GPU kernel data."""
        summaries = {
            "torch_profiler": {
                "top_ops": [{"name": "op_0", "total_us": 10000}],
                "cpu_vs_gpu": {"total_gpu_kernel_us": 5000},
            },
            "metal_trace": _make_metal_summary(gpu_time_total_ms=1, duration_s=1),
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")

        cross = [d for d in diags if "cross_profiler" in (d.root_cause or "").lower()]
        assert len(cross) == 0

    def test_diagnose_bottlenecks_skips_without_metal(self):
        """Cross-profiler is not called when metal_trace is absent."""
        summaries = {
            "torch_profiler": _make_torch_summary([10000]),
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")

        cross = [d for d in diags if "cross_profiler" in (d.root_cause or "").lower()]
        assert len(cross) == 0
