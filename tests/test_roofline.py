"""Tests for perflab.reporting.roofline."""
from __future__ import annotations

from perflab.reporting.roofline import (
    RooflinePoint,
    compute_roofline_point,
    select_peak_tflops,
)

# Access private helper for unit testing dtype mapping
from perflab.reporting.roofline import _dtype_bytes


class TestDtypeBytes:
    def test_fp16(self):
        assert _dtype_bytes("fp16") == 2
        assert _dtype_bytes("float16") == 2
        assert _dtype_bytes("half") == 2

    def test_fp32(self):
        assert _dtype_bytes("fp32") == 4
        assert _dtype_bytes("float32") == 4

    def test_fp64(self):
        assert _dtype_bytes("fp64") == 8
        assert _dtype_bytes("double") == 8

    def test_unknown_defaults_to_4(self):
        assert _dtype_bytes("int8") == 4
        assert _dtype_bytes("") == 4


class TestComputeRooflinePoint:
    def test_matmul_mode(self):
        bench = {
            "meta": {"M": 1024, "N": 1024, "K": 1024, "dtype": "fp32"},
            "latency_ms": {"p50": 1.0},
        }
        pt = compute_roofline_point(bench)
        assert pt is not None
        assert pt.ai > 0
        assert pt.tflops > 0
        assert pt.gbs > 0

    def test_generic_mode(self):
        bench = {
            "meta": {"flops": 1e12, "bytes_moved": 1e9},
            "latency_ms": {"p50": 10.0},
        }
        pt = compute_roofline_point(bench)
        assert pt is not None
        # AI = 1e12 / 1e9 = 1000
        assert abs(pt.ai - 1000.0) < 1.0
        # tflops = 1e12 / 0.01 / 1e12 = 100
        assert abs(pt.tflops - 100.0) < 0.1

    def test_no_meta_returns_none(self):
        assert compute_roofline_point({}) is None
        assert compute_roofline_point({"meta": {}}) is None

    def test_zero_flops_returns_none(self):
        bench = {
            "meta": {"flops": 0, "bytes_moved": 1e9},
            "latency_ms": {"p50": 1.0},
        }
        assert compute_roofline_point(bench) is None

    def test_measured_dram_overrides(self):
        bench = {
            "meta": {"flops": 1e12, "bytes_moved": 1e9},
            "latency_ms": {"p50": 10.0},
        }
        measured = 2e9  # twice the theoretical bytes
        pt = compute_roofline_point(bench, measured_dram_bytes=measured)
        assert pt is not None
        # AI should use measured: 1e12 / 2e9 = 500
        assert abs(pt.ai - 500.0) < 1.0
        assert pt.achieved_bw_gbs is not None

    def test_matmul_correct_ai(self):
        # M=N=K=1024, batch=1, fp32 (4 bytes)
        # flops = 2*1024^3 = 2,147,483,648
        # bytes = 4*(1024*1024 + 1024*1024 + 2*1024*1024) = 4*4*1024^2 = 16,777,216
        # AI = 2,147,483,648 / 16,777,216 = 128
        bench = {
            "meta": {"M": 1024, "N": 1024, "K": 1024, "dtype": "fp32"},
            "latency_ms": {"p50": 1.0},
        }
        pt = compute_roofline_point(bench)
        assert pt is not None
        assert abs(pt.ai - 128.0) < 0.1

    def test_timing_from_tflops_median(self):
        # No latency_ms, but tflops.median provided
        bench = {
            "meta": {"M": 1024, "N": 1024, "K": 1024, "dtype": "fp32"},
            "tflops": {"median": 2.0},
        }
        pt = compute_roofline_point(bench)
        assert pt is not None
        assert abs(pt.tflops - 2.0) < 0.01


class TestSelectPeakTflops:
    def test_no_dtype_peaks(self):
        assert select_peak_tflops(100.0) == 100.0
        assert select_peak_tflops(100.0, dtype="fp16") == 100.0
        assert select_peak_tflops(100.0, dtype_peaks={}) == 100.0

    def test_fp16_peak(self):
        peaks = {"peak_tflops_fp16": 312.0}
        assert select_peak_tflops(100.0, dtype="fp16", dtype_peaks=peaks) == 312.0

    def test_bf16_peak(self):
        peaks = {"peak_tflops_bf16": 250.0}
        assert select_peak_tflops(100.0, dtype="bf16", dtype_peaks=peaks) == 250.0

    def test_fp32_uses_tf32_first(self):
        peaks = {"peak_tflops_tf32": 156.0, "peak_tflops_fp32": 78.0}
        assert select_peak_tflops(100.0, dtype="fp32", dtype_peaks=peaks) == 156.0

    def test_fp64_peak(self):
        peaks = {"peak_tflops_fp64": 39.0}
        assert select_peak_tflops(100.0, dtype="fp64", dtype_peaks=peaks) == 39.0

    def test_unknown_dtype_falls_back(self):
        peaks = {"peak_tflops_fp16": 312.0}
        assert select_peak_tflops(100.0, dtype="int4", dtype_peaks=peaks) == 100.0
