"""Tests for perflab.analyzers.build_flags."""
from __future__ import annotations

from perflab.analyzers.build_flags import (
    FlagRecommendation,
    recommend_build_flags,
)


class TestRecommendBuildFlags:
    def test_missing_march_native(self):
        recs = recommend_build_flags(
            "g++ -O2 -o matmul matmul.cpp",
            {"avx2": True, "max_simd_width_bits": 256, "fma": True},
            "cpp",
        )
        assert any(r.flag == "-march=native" for r in recs)

    def test_o2_to_o3_suggestion(self):
        recs = recommend_build_flags(
            "g++ -O2 -o matmul matmul.cpp",
            {"avx2": True, "max_simd_width_bits": 256},
            "cpp",
        )
        assert any(r.flag == "-O3" for r in recs)

    def test_g_to_gline_tables(self):
        recs = recommend_build_flags(
            "g++ -O3 -g -o matmul matmul.cpp",
            {"avx2": True, "max_simd_width_bits": 256},
            "cpp",
        )
        assert any(r.flag == "-gline-tables-only" for r in recs)

    def test_sanitizer_warning(self):
        recs = recommend_build_flags(
            "g++ -O3 -fsanitize=address -o matmul matmul.cpp",
            {"avx2": True, "max_simd_width_bits": 256},
            "cpp",
        )
        assert any("sanitize" in r.flag.lower() or "sanitize" in r.reason.lower() for r in recs)

    def test_fma_recommendation(self):
        recs = recommend_build_flags(
            "g++ -O3 -o matmul matmul.cpp",
            {"avx2": True, "fma": True, "max_simd_width_bits": 256},
            "cpp",
        )
        assert any(r.flag in ("-mfma", "-march=native") for r in recs)


class TestNoRecommendations:
    def test_already_optimal(self):
        recs = recommend_build_flags(
            "g++ -O3 -march=native -flto -gline-tables-only -o matmul matmul.cpp",
            {"avx2": True, "fma": True, "max_simd_width_bits": 256},
            "cpp",
        )
        # Should have no ISA recs since -march=native is present
        isa_recs = [r for r in recs if r.category == "isa"]
        assert len(isa_recs) == 0
        # No -O3 rec since already present
        assert not any(r.flag == "-O3" for r in recs)

    def test_python_returns_empty(self):
        recs = recommend_build_flags(
            "python bench.py",
            {"avx2": True, "max_simd_width_bits": 256},
            "python",
        )
        assert recs == []


class TestCudaFlags:
    def test_missing_arch(self):
        recs = recommend_build_flags(
            "nvcc -O3 -o matmul matmul.cu",
            {"avx2": True, "max_simd_width_bits": 256},
            "cuda",
        )
        assert any("-arch=" in r.flag for r in recs)

    def test_arch_present(self):
        recs = recommend_build_flags(
            "nvcc -O3 -arch=sm_80 -o matmul matmul.cu",
            {"avx2": True, "max_simd_width_bits": 256},
            "cuda",
        )
        assert not any("-arch=" in r.flag for r in recs)
