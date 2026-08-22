"""Tests for perflab.analyzers.bottleneck_analyzer."""
from __future__ import annotations

from perflab.analyzers.bottleneck_analyzer import (
    AnalysisThresholds,
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

    def test_nsys_gpu_load_imbalance(self):
        summaries = {"nsys": {"gpu_active_pct_by_device": {0: 95.0, 1: 20.0}}}
        diags = diagnose_bottlenecks(summaries, "cuda")
        assert any("imbalance" in d.bottleneck.lower() for d in diags)

    def test_nsys_gpu_balanced_devices_no_finding(self):
        summaries = {"nsys": {"gpu_active_pct_by_device": {0: 90.0, 1: 85.0}}}
        diags = diagnose_bottlenecks(summaries, "cuda")
        assert not any("imbalance" in d.bottleneck.lower() for d in diags)

    def test_nsys_per_device_kernel_divergence(self):
        summaries = {"nsys": {
            "top_kernels": [{"name": "sgemm", "pct": 60.0, "total_ms": 100}],
            "top_kernels_by_device": {
                0: [{"name": "sgemm", "pct": 90.0, "total_ms": 90}],
                1: [{"name": "conv2d", "pct": 95.0, "total_ms": 95}],
            },
        }}
        diags = diagnose_bottlenecks(summaries, "cuda")
        assert any("different kernel" in d.bottleneck.lower() for d in diags)

    def test_nsys_no_divergence_when_same_kernel_dominates_every_device(self):
        summaries = {"nsys": {
            "top_kernels": [{"name": "sgemm", "pct": 60.0, "total_ms": 100}],
            "top_kernels_by_device": {
                0: [{"name": "sgemm", "pct": 90.0, "total_ms": 90}],
                1: [{"name": "sgemm", "pct": 92.0, "total_ms": 92}],
            },
        }}
        diags = diagnose_bottlenecks(summaries, "cuda")
        assert not any("different kernel" in d.bottleneck.lower() for d in diags)

    def test_nsys_high_nccl_time(self):
        summaries = {"nsys": {"nccl_pct": 35.0}}
        diags = diagnose_bottlenecks(summaries, "cuda")
        assert any("nccl" in d.bottleneck.lower() for d in diags)

    def test_nsys_low_nccl_time_no_finding(self):
        summaries = {"nsys": {"nccl_pct": 5.0}}
        diags = diagnose_bottlenecks(summaries, "cuda")
        assert not any("nccl" in d.bottleneck.lower() for d in diags)

    def test_perf_low_ipc(self):
        summaries = {"linux_perf": {"ipc": 0.4}}
        diags = diagnose_bottlenecks(summaries, "cpp")
        assert any("IPC" in d.bottleneck or "ipc" in d.bottleneck.lower() for d in diags)

    def test_perf_high_cache_miss(self):
        summaries = {"linux_perf": {"cache_miss_rate": 0.15}}
        diags = diagnose_bottlenecks(summaries, "cpp")
        assert any("cache" in d.bottleneck.lower() for d in diags)

    def test_perf_multiprocess_cpu_imbalance(self):
        summaries = {"linux_perf": {"cpu_pct_by_pid": {100: 90.0, 200: 10.0}}}
        diags = diagnose_bottlenecks(summaries, "python")
        assert any("imbalance" in d.bottleneck.lower() for d in diags)
        imbalance = next(d for d in diags if "imbalance" in d.bottleneck.lower())
        assert "Worker" in imbalance.bottleneck
        assert "CPU share" in imbalance.bottleneck

    def test_perf_multiprocess_balanced_no_imbalance_finding(self):
        summaries = {"linux_perf": {"cpu_pct_by_pid": {100: 55.0, 200: 45.0}}}
        diags = diagnose_bottlenecks(summaries, "python")
        assert not any("imbalance" in d.bottleneck.lower() for d in diags)

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
