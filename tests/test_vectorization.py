"""Tests for perflab.analyzers.vectorization."""
from __future__ import annotations

from perflab.analyzers.vectorization import (
    VectorizationReport,
    VectorizationSummary,
    analyze_vectorization,
    check_vectorization_from_perf_annotate,
    format_vectorization_for_prompt,
)


class TestAnalyzeVectorization:
    def test_empty_input(self):
        summary = analyze_vectorization([])
        assert summary.vectorized_count == 0
        assert summary.not_vectorized_count == 0

    def test_function_without_raw_asm(self):
        entries = [{"function": "matmul", "hot_lines": [{"pct": 50.0}]}]
        summary = analyze_vectorization(entries)
        assert len(summary.functions) == 1
        assert not summary.functions[0].has_simd
        assert summary.not_vectorized_count == 1

    def test_function_with_avx_asm(self):
        entries = [{
            "function": "matmul",
            "hot_lines": [{"pct": 80.0}],
            "raw_asm": "vmovaps %ymm0, %ymm1\nvaddps %ymm2, %ymm3, %ymm4\nmov %rax, %rbx\n",
        }]
        summary = analyze_vectorization(entries)
        assert summary.functions[0].has_simd
        assert summary.functions[0].simd_isa == "avx"
        assert summary.vectorized_count == 1

    def test_warning_for_unvectorized(self):
        entries = [{"function": "slow_func", "hot_lines": [{"pct": 60.0}]}]
        summary = analyze_vectorization(entries)
        assert "SIMD" in summary.warning
        assert "slow_func" in summary.warning


class TestCheckFromAnnotate:
    def test_avx_detected(self):
        text = (
            "Percent |  Source code & Disassembly of matmul_bin\n"
            "  45.20 :  14: vmovaps %ymm0, (%rax)\n"
            "  12.00 :  15: vaddps %ymm1, %ymm2, %ymm3\n"
            "   3.00 :  16: mov %rax, %rbx\n"
        )
        summary = check_vectorization_from_perf_annotate(text)
        assert len(summary.functions) == 1
        assert summary.functions[0].has_simd
        assert summary.functions[0].simd_isa == "avx"

    def test_no_simd_detected(self):
        text = (
            "Percent |  Source code & Disassembly of matmul_bin\n"
            "  45.20 :  14: mov %rax, %rbx\n"
            "  12.00 :  15: add %rcx, %rdx\n"
        )
        summary = check_vectorization_from_perf_annotate(text)
        assert len(summary.functions) == 1
        assert not summary.functions[0].has_simd
        assert "SIMD" in summary.warning

    def test_neon_detected(self):
        text = (
            "Percent |  Source code & Disassembly of matmul_bin\n"
            "  30.00 :  14: fmla v0.4s, v1.4s, v2.4s\n"
            "  10.00 :  15: ld1 {v3.4s}, [x0]\n"
        )
        summary = check_vectorization_from_perf_annotate(text)
        assert summary.functions[0].has_simd
        assert summary.functions[0].simd_isa == "neon"

    def test_multiple_functions(self):
        text = (
            "Percent |  Source code & Disassembly of func_a\n"
            "  40.00 :  10: vmulps %ymm0, %ymm1, %ymm2\n"
            "Percent |  Source code & Disassembly of func_b\n"
            "  20.00 :  20: mov %rax, %rbx\n"
        )
        summary = check_vectorization_from_perf_annotate(text)
        assert len(summary.functions) == 2
        assert summary.vectorized_count == 1
        assert summary.not_vectorized_count == 1

    def test_avx512_detected(self):
        text = (
            "Percent |  Source code & Disassembly of kernel\n"
            "  50.00 :  10: vmovaps zmm0, [rax]\n"
            "  30.00 :  11: vaddps zmm1, zmm2, zmm3\n"
        )
        summary = check_vectorization_from_perf_annotate(text)
        assert summary.functions[0].simd_isa == "avx512"


class TestFormat:
    def test_empty_summary(self):
        summary = VectorizationSummary()
        assert format_vectorization_for_prompt(summary) == ""

    def test_with_functions(self):
        summary = VectorizationSummary(
            functions=[
                VectorizationReport("matmul", True, "avx", hot_pct=80.0),
                VectorizationReport("init", False, "none", hot_pct=5.0),
            ],
            vectorized_count=1,
            not_vectorized_count=1,
            warning="1 hot function(s) lack SIMD",
        )
        text = format_vectorization_for_prompt(summary)
        assert "matmul" in text
        assert "SIMD (avx)" in text
        assert "NO SIMD" in text
        assert "WARNING" in text
