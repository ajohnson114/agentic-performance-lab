"""Tests for hierarchical roofline (L2 bandwidth ceiling) and profiler FLOPS integration."""
from __future__ import annotations

import pytest

from perflab.reporting.roofline import RooflinePoint, compute_roofline_point
from perflab.roofline_peaks import _lookup_l2_bw

# ---------------------------------------------------------------------------
# L2 bandwidth lookup
# ---------------------------------------------------------------------------

class TestL2BandwidthLookup:
    def test_a100_l2_bw(self):
        assert _lookup_l2_bw("NVIDIA A100-SXM4-80GB") == 6000.0

    def test_h100_l2_bw(self):
        assert _lookup_l2_bw("NVIDIA H100 SXM") == 12000.0

    def test_rtx_4090_l2_bw(self):
        assert _lookup_l2_bw("NVIDIA GeForce RTX 4090") == 3200.0

    def test_v100_l2_bw(self):
        assert _lookup_l2_bw("Tesla V100-SXM2") == 3100.0

    def test_unknown_gpu_returns_none(self):
        assert _lookup_l2_bw("NVIDIA RTX 5090") is None

    def test_empty_string(self):
        assert _lookup_l2_bw("") is None


# ---------------------------------------------------------------------------
# Profiler FLOPS integration into roofline
# ---------------------------------------------------------------------------

class TestProfilerFlopsRoofline:
    def test_profiler_flops_creates_roofline_point(self):
        """When meta has no M/N/K or flops, profiler_flops + measured DRAM bytes
        should create a valid roofline point."""
        bench = {
            "latency_ms": {"p50": 10.0},
            "meta": {},
        }
        point = compute_roofline_point(
            bench,
            measured_dram_bytes=1e9,  # 1 GB
            profiler_flops=2e12,      # 2 TFLOPS worth of work
        )
        assert point is not None
        assert point.tflops == pytest.approx(200.0, rel=0.01)  # 2e12 / 0.01s / 1e12
        assert point.ai == pytest.approx(2000.0, rel=0.01)    # 2e12 / 1e9

    def test_profiler_flops_no_bytes_returns_none(self):
        """Without bytes_moved or measured DRAM, can't compute AI."""
        bench = {
            "latency_ms": {"p50": 10.0},
            "meta": {},
        }
        point = compute_roofline_point(
            bench,
            profiler_flops=2e12,
            # No measured_dram_bytes
        )
        assert point is None

    def test_meta_flops_takes_priority_over_profiler(self):
        """If meta has explicit flops+bytes_moved, they should be used, not profiler_flops."""
        bench = {
            "latency_ms": {"p50": 10.0},
            "meta": {"flops": 1e12, "bytes_moved": 1e9},
        }
        point = compute_roofline_point(
            bench,
            profiler_flops=999e12,  # Should be ignored
        )
        assert point is not None
        assert point.tflops == pytest.approx(100.0, rel=0.01)  # 1e12 / 0.01s / 1e12

    def test_matmul_meta_takes_priority_over_profiler(self):
        """If meta has M/N/K, matmul mode should be used, not profiler_flops."""
        bench = {
            "latency_ms": {"p50": 10.0},
            "meta": {"M": 1024, "N": 1024, "K": 1024},
        }
        point = compute_roofline_point(
            bench,
            profiler_flops=999e12,
        )
        assert point is not None
        # Matmul flops = 2*1024^3 = ~2.15e9, not 999e12
        assert point.tflops < 1.0

    def test_profiler_flops_zero_ignored(self):
        bench = {
            "latency_ms": {"p50": 10.0},
            "meta": {},
        }
        point = compute_roofline_point(
            bench,
            profiler_flops=0,
            measured_dram_bytes=1e9,
        )
        assert point is None


# ---------------------------------------------------------------------------
# Hierarchical roofline in classify_bound
# ---------------------------------------------------------------------------

class TestHierarchicalClassifyBound:
    def test_dram_bottleneck_identified(self):
        from perflab.optimizers.prompt import PromptContext, _classify_bound

        ctx = PromptContext(
            source_files={},
            profiler_summaries={},
            bench_results={"tflops": {"median": 5.0}},
            roofline={
                "peak_tflops": 300.0,
                "peak_mem_bw_gbs": 2000.0,
                "peak_l2_bw_gbs": 6000.0,
                "achieved_bw_gbs": 1500.0,  # 75% of DRAM peak — saturating
                "computed_ai": 10.0,
            },
            history=[],
            allowed_paths=[],
            program_type="cuda",
        )
        result = _classify_bound(ctx)
        assert result is not None
        assert result["bound"] == "memory-bound"
        assert result["bw_bottleneck_level"] == "DRAM"
        assert result["peak_l2_bw_gbs"] == 6000.0

    def test_l2_bottleneck_identified(self):
        from perflab.optimizers.prompt import PromptContext, _classify_bound

        ctx = PromptContext(
            source_files={},
            profiler_summaries={},
            bench_results={"tflops": {"median": 5.0}},
            roofline={
                "peak_tflops": 300.0,
                "peak_mem_bw_gbs": 2000.0,
                "peak_l2_bw_gbs": 6000.0,
                "achieved_bw_gbs": 800.0,  # 40% of DRAM peak — not saturating
                "computed_ai": 10.0,
            },
            history=[],
            allowed_paths=[],
            program_type="cuda",
        )
        result = _classify_bound(ctx)
        assert result is not None
        assert result["bw_bottleneck_level"] == "L2-or-below"

    def test_no_l2_data_no_level(self):
        from perflab.optimizers.prompt import PromptContext, _classify_bound

        ctx = PromptContext(
            source_files={},
            profiler_summaries={},
            bench_results={"tflops": {"median": 5.0}},
            roofline={
                "peak_tflops": 300.0,
                "peak_mem_bw_gbs": 2000.0,
                "computed_ai": 10.0,
            },
            history=[],
            allowed_paths=[],
            program_type="cuda",
        )
        result = _classify_bound(ctx)
        assert result is not None
        assert result.get("bw_bottleneck_level") is None

    def test_compute_bound_no_bw_level(self):
        from perflab.optimizers.prompt import PromptContext, _classify_bound

        ctx = PromptContext(
            source_files={},
            profiler_summaries={},
            bench_results={"tflops": {"median": 200.0}},
            roofline={
                "peak_tflops": 300.0,
                "peak_mem_bw_gbs": 2000.0,
                "peak_l2_bw_gbs": 6000.0,
                "computed_ai": 500.0,  # Above knee — compute-bound
            },
            history=[],
            allowed_paths=[],
            program_type="cuda",
        )
        result = _classify_bound(ctx)
        assert result is not None
        assert result["bound"] == "compute-bound"
        assert result.get("bw_bottleneck_level") is None


# ---------------------------------------------------------------------------
# Roofline PNG with L2 ceiling (smoke test)
# ---------------------------------------------------------------------------

class TestRooflinePngL2:
    def test_write_roofline_with_l2(self, tmp_path):
        from perflab.reporting.roofline import write_roofline_png

        png_path = tmp_path / "roofline.png"
        point = RooflinePoint(ai=50.0, tflops=5.0, gbs=100.0)
        write_roofline_png(
            png_path,
            point=point,
            peak_tflops=300.0,
            peak_mem_bw_gbs=2000.0,
            title="Test Hierarchical Roofline",
            l2_bw_gbs=6000.0,
        )
        assert png_path.exists()
        assert png_path.stat().st_size > 1000  # Not empty

    def test_write_roofline_without_l2(self, tmp_path):
        from perflab.reporting.roofline import write_roofline_png

        png_path = tmp_path / "roofline.png"
        point = RooflinePoint(ai=50.0, tflops=5.0, gbs=100.0)
        write_roofline_png(
            png_path,
            point=point,
            peak_tflops=300.0,
            peak_mem_bw_gbs=2000.0,
            title="Test No L2",
        )
        assert png_path.exists()
