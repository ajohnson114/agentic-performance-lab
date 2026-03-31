"""Tests for perflab.analyzers.bottleneck_analyzer."""
from __future__ import annotations

from perflab.analyzers.bottleneck_analyzer import (
    AnalysisThresholds,
    BottleneckDiagnosis,
    compute_source_hints,
    diagnose_bottlenecks,
)


class TestDiagnoseBottlenecks:
    def test_empty_summaries_returns_empty(self):
        assert diagnose_bottlenecks({}, "cpp") == []

    def test_ncu_low_sm_util(self):
        summaries = {"ncu": {"sm_utilization_pct": 20}}
        diags = diagnose_bottlenecks(summaries, "cuda")
        assert any("SM utilization" in d.bottleneck for d in diags)

    def test_ncu_memory_bound(self):
        summaries = {
            "ncu": {
                "dominant_kernel": {
                    "name": "matmul_kern",
                    "memory_throughput_pct": 85,
                    "compute_throughput_pct": 20,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        assert any("memory-bound" in d.bottleneck.lower() for d in diags)

    def test_nsys_cpu_bound(self):
        summaries = {"nsys": {"cuda_kernel_time_ms": 100, "duration_s": 10}}
        diags = diagnose_bottlenecks(summaries, "cuda")
        assert any("CPU-bound" in d.bottleneck for d in diags)

    def test_nsys_kernel_gap(self):
        summaries = {"nsys": {"avg_kernel_gap_us": 300}}
        diags = diagnose_bottlenecks(summaries, "cuda")
        assert any("launch overhead" in d.bottleneck.lower() or "gap" in d.bottleneck.lower() for d in diags)

    def test_perf_low_ipc(self):
        summaries = {"linux_perf": {"ipc": 0.4}}
        diags = diagnose_bottlenecks(summaries, "cpp")
        assert any("IPC" in d.bottleneck or "ipc" in d.bottleneck.lower() for d in diags)

    def test_perf_high_cache_miss(self):
        summaries = {"linux_perf": {"cache_miss_rate": 0.15}}
        diags = diagnose_bottlenecks(summaries, "cpp")
        assert any("cache" in d.bottleneck.lower() for d in diags)

    def test_metal_gpu_underutilized(self):
        summaries = {"metal_trace": {"gpu_time_total_ms": 50, "duration_s": 10}}
        diags = diagnose_bottlenecks(summaries, "pytorch")
        assert any("underutilized" in d.bottleneck.lower() or "GPU" in d.bottleneck for d in diags)

    def test_jax_recompilation(self):
        summaries = {"jax": {"xla_recompilations": 3}}
        diags = diagnose_bottlenecks(summaries, "jax")
        assert any("recompilation" in d.bottleneck.lower() for d in diags)

    def test_top_n_limits_results(self):
        # Feed multiple profiler sources to generate many findings
        summaries = {
            "ncu": {"sm_utilization_pct": 10, "memory_throughput_pct": 95},
            "nsys": {"cuda_kernel_time_ms": 50, "duration_s": 10, "avg_kernel_gap_us": 300},
            "linux_perf": {"ipc": 0.3, "cache_miss_rate": 0.2},
        }
        diags = diagnose_bottlenecks(summaries, "cuda", top_n=2)
        assert len(diags) <= 2

    def test_custom_thresholds(self):
        # With default thresholds, sm_util=45 is below ncu_sm_util_low=50 → triggers
        summaries = {"ncu": {"sm_utilization_pct": 45}}
        diags_default = diagnose_bottlenecks(summaries, "cuda")
        assert any("SM utilization" in d.bottleneck for d in diags_default)

        # With relaxed threshold, 45 is above 40 → should NOT trigger
        relaxed = AnalysisThresholds(ncu_sm_util_low=40.0, ncu_sm_util_critical=20.0)
        diags_relaxed = diagnose_bottlenecks(summaries, "cuda", thresholds=relaxed)
        assert not any("SM utilization" in d.bottleneck for d in diags_relaxed)

    def test_ranks_are_sequential(self):
        summaries = {
            "ncu": {"sm_utilization_pct": 10, "memory_throughput_pct": 95},
            "nsys": {"cuda_kernel_time_ms": 50, "duration_s": 10},
        }
        diags = diagnose_bottlenecks(summaries, "cuda", top_n=5)
        for i, d in enumerate(diags):
            assert d.rank == i + 1


class TestComputeSourceHints:
    def test_cpp_with_simd(self):
        sources = {"main.cpp": '#include <immintrin.h>\nint main() {}'}
        hints = compute_source_hints(sources, "cpp")
        assert hints.get("has_simd") is True

    def test_cpp_with_openmp(self):
        sources = {"main.cpp": '#pragma omp parallel for\nfor (int i=0; i<n; i++) {}'}
        hints = compute_source_hints(sources, "cpp")
        assert hints.get("has_openmp") is True

    def test_plain_cpp(self):
        sources = {"main.cpp": "int main() { return 0; }"}
        hints = compute_source_hints(sources, "cpp")
        assert hints.get("has_simd") is not True
        assert hints.get("has_openmp") is not True
        assert hints.get("has_threading") is not True

    def test_non_cpp_returns_empty(self):
        sources = {"main.py": '#include <immintrin.h>'}
        hints = compute_source_hints(sources, "pytorch")
        assert hints == {}
