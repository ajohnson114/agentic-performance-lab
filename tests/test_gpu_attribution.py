"""Tests for perflab.analyzers.gpu_attribution."""
from __future__ import annotations

from perflab.analyzers.gpu_attribution import (
    AttributionEntry,
    CpuGpuEdge,
    build_cpu_gpu_call_graph,
    compute_attribution_ranking,
    detect_pipeline_stalls,
    enrich_with_framework_context,
)


class TestBuildCpuGpuCallGraph:
    def test_aggregation(self):
        correlations = [
            {"api_name": "cudaLaunchKernel", "kernel_name": "sgemm", "stream_id": 1,
             "gpu_duration_ns": 5_000_000, "launch_overhead_ns": 10_000},
            {"api_name": "cudaLaunchKernel", "kernel_name": "sgemm", "stream_id": 1,
             "gpu_duration_ns": 4_000_000, "launch_overhead_ns": 12_000},
        ]
        edges = build_cpu_gpu_call_graph(correlations)
        assert len(edges) == 1
        assert edges[0].count == 2
        assert edges[0].total_gpu_ms > 0
        assert edges[0].pct_of_total_gpu > 0

    def test_multiple_kernels(self):
        correlations = [
            {"api_name": "cudaLaunchKernel", "kernel_name": "sgemm", "stream_id": 1,
             "gpu_duration_ns": 5_000_000, "launch_overhead_ns": 10_000},
            {"api_name": "cudaLaunchKernel", "kernel_name": "relu", "stream_id": 1,
             "gpu_duration_ns": 1_000_000, "launch_overhead_ns": 5_000},
        ]
        edges = build_cpu_gpu_call_graph(correlations)
        assert len(edges) == 2
        assert edges[0].kernel_name == "sgemm"  # sorted by total_gpu_ms desc
        total_pct = sum(e.pct_of_total_gpu for e in edges)
        assert abs(total_pct - 100.0) < 0.1

    def test_empty_correlations(self):
        assert build_cpu_gpu_call_graph([]) == []


class TestAttributionRanking:
    def test_top_kernel_gets_rank_1(self):
        nsys = {
            "top_kernels": [
                {"name": "sgemm", "pct": 85.0, "total_ms": 100},
                {"name": "relu", "pct": 15.0, "total_ms": 18},
            ],
            "cpu_gpu_correlations": [
                {"api_name": "cudaLaunchKernel", "kernel_name": "sgemm", "stream_id": 1,
                 "gpu_duration_ns": 100_000_000, "launch_overhead_ns": 10_000}
            ],
        }
        ranking = compute_attribution_ranking(nsys)
        assert len(ranking) > 0
        assert ranking[0].rank == 1
        assert ranking[0].gpu_pct > 0

    def test_launch_overhead_penalty(self):
        nsys = {
            "top_kernels": [
                {"name": "small_kern", "pct": 50.0, "total_ms": 10},
            ],
            "cpu_gpu_correlations": [
                {"api_name": "cudaLaunchKernel", "kernel_name": "small_kern", "stream_id": 1,
                 "gpu_duration_ns": 10_000_000, "launch_overhead_ns": 100_000}  # 100 us overhead
            ],
        }
        ranking = compute_attribution_ranking(nsys)
        assert any(e.launch_overhead_us and e.launch_overhead_us > 50 for e in ranking)
        # Should suggest CUDA graphs
        assert any("CUDA graphs" in s for e in ranking for s in e.suggestions)

    def test_empty_nsys(self):
        ranking = compute_attribution_ranking({})
        assert ranking == []


class TestPipelineStalls:
    def test_idle_stream_detected(self):
        per_stream_gaps = {1: {"avg_gap_us": 200, "max_gap_us": 500, "num_gaps": 10}}
        stream_util = {1: {"active_pct": 30, "kernel_count": 20, "total_kernel_ms": 5}}
        entries = detect_pipeline_stalls(per_stream_gaps, stream_util)
        assert len(entries) >= 1
        assert entries[0].category == "pipeline-stall"
        assert "idle" in entries[0].diagnosis.lower()

    def test_multi_stream_serialization(self):
        per_stream_gaps = {}
        stream_util = {
            1: {"active_pct": 90, "kernel_count": 100, "total_kernel_ms": 50},
            2: {"active_pct": 10, "kernel_count": 5, "total_kernel_ms": 2},
        }
        entries = detect_pipeline_stalls(per_stream_gaps, stream_util)
        assert any("imbalance" in e.diagnosis.lower() for e in entries)

    def test_empty_data(self):
        assert detect_pipeline_stalls({}, {}) == []


class TestCorrelationExtraction:
    """Test that the correlation data structure is handled correctly."""

    def test_call_graph_from_correlations(self):
        correlations = [
            {"api_name": "cudaLaunchKernel", "kernel_name": "volta_sgemm_128x128_nn",
             "stream_id": 7, "gpu_duration_ns": 50_000_000, "launch_overhead_ns": 8_000},
            {"api_name": "cudaLaunchKernel", "kernel_name": "volta_sgemm_128x128_nn",
             "stream_id": 7, "gpu_duration_ns": 48_000_000, "launch_overhead_ns": 9_000},
            {"api_name": "cudaLaunchKernel", "kernel_name": "relu_kernel",
             "stream_id": 7, "gpu_duration_ns": 2_000_000, "launch_overhead_ns": 5_000},
        ]
        edges = build_cpu_gpu_call_graph(correlations)
        assert len(edges) == 2
        # sgemm should be first (higher total GPU time)
        assert edges[0].kernel_name == "volta_sgemm_128x128_nn"
        assert edges[0].count == 2
        assert edges[1].kernel_name == "relu_kernel"
        assert edges[1].count == 1


class TestFrameworkEnrichment:
    def test_pytorch_gemm_detection(self):
        edges = [CpuGpuEdge("cudaLaunchKernel", "volta_sgemm_128x128", 1, 10, 100.0, 10.0, 80.0)]
        enriched = enrich_with_framework_context(edges, program_type="pytorch")
        assert enriched[0].framework_op == "aten::mm"

    def test_triton_kernel_name(self):
        edges = [CpuGpuEdge("cudaLaunchKernel", "triton_poi_fused_relu_0", 1, 5, 20.0, 8.0, 15.0)]
        enriched = enrich_with_framework_context(edges, program_type="pytorch")
        assert enriched[0].framework_op is not None
        assert "triton" in enriched[0].framework_op

    def test_fallback_without_nvtx(self):
        edges = [CpuGpuEdge("cudaLaunchKernel", "my_custom_kernel", 1, 5, 20.0, 8.0, 15.0)]
        enriched = enrich_with_framework_context(edges, nvtx_ranges=None, program_type="cuda")
        # Raw CUDA: no framework op inferred for unknown kernels
        # This is fine — framework_op is optional

    def test_empty_edges(self):
        enriched = enrich_with_framework_context([], program_type="cuda")
        assert enriched == []


class TestEmptyData:
    def test_empty_correlations(self):
        assert build_cpu_gpu_call_graph([]) == []

    def test_empty_ranking(self):
        assert compute_attribution_ranking({}) == []

    def test_empty_stalls(self):
        assert detect_pipeline_stalls({}, {}) == []
