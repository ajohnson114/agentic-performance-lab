"""Tests for perflab.analyzers.compiler_diagnostics."""
from __future__ import annotations

from perflab.analyzers.compiler_diagnostics import (
    CompilerDiagnostics,
    OptimizationRemark,
    _fuzzy_kernel_match,
    _infer_element_bits,
    _parse_clang_remarks,
    _parse_gcc_diagnostics,
    _parse_gcc_remarks,
    _parse_jax_diagnostics,
    _parse_nvcc_diagnostics,
    _parse_nvcc_remarks,
    _parse_pytorch_diagnostics,
    _parse_triton_diagnostics,
    _summarize_findings,
    cross_reference_diagnostics,
    detect_compiler,
    get_diagnostic_build_flags,
    get_diagnostic_env_vars,
    parse_compiler_output,
)

# ---------------------------------------------------------------------------
# get_diagnostic_build_flags
# ---------------------------------------------------------------------------

class TestGetDiagnosticBuildFlags:
    def test_cpp_gcc(self):
        flags = get_diagnostic_build_flags("cpp", compiler="gcc")
        assert "-fopt-info-all-optall" in flags
        assert "-gline-tables-only" in flags

    def test_cpp_default(self):
        flags = get_diagnostic_build_flags("cpp")
        assert "-fopt-info-all-optall" in flags

    def test_cpp_clang(self):
        flags = get_diagnostic_build_flags("cpp", compiler="clang")
        assert any("-Rpass=" in f for f in flags)
        assert "-gline-tables-only" in flags

    def test_cuda(self):
        flags = get_diagnostic_build_flags("cuda")
        assert flags == ["--ptxas-options=-v", "--generate-line-info"]

    def test_pytorch_returns_empty(self):
        assert get_diagnostic_build_flags("pytorch") == []

    def test_jax_returns_empty(self):
        assert get_diagnostic_build_flags("jax") == []

    def test_triton_returns_empty(self):
        assert get_diagnostic_build_flags("triton") == []

    def test_python_returns_empty(self):
        assert get_diagnostic_build_flags("python") == []


# ---------------------------------------------------------------------------
# get_diagnostic_env_vars
# ---------------------------------------------------------------------------

class TestGetDiagnosticEnvVars:
    def test_pytorch(self):
        env = get_diagnostic_env_vars("pytorch")
        assert "TORCH_LOGS" in env
        assert "+dynamo" in env["TORCH_LOGS"]

    def test_jax(self):
        env = get_diagnostic_env_vars("jax")
        assert env["JAX_LOG_COMPILES"] == "1"

    def test_triton(self):
        env = get_diagnostic_env_vars("triton")
        assert env["TRITON_DEBUG"] == "1"

    def test_cpp_gcc(self):
        env = get_diagnostic_env_vars("cpp")
        assert "PERFLAB_CXXFLAGS" in env
        assert "-fopt-info-all-optall" in env["PERFLAB_CXXFLAGS"]

    def test_cpp_clang(self):
        env = get_diagnostic_env_vars("cpp", compiler="clang")
        assert "PERFLAB_CXXFLAGS" in env
        assert "-Rpass=" in env["PERFLAB_CXXFLAGS"]

    def test_cuda(self):
        env = get_diagnostic_env_vars("cuda")
        assert "PERFLAB_NVCCFLAGS" in env
        assert "--ptxas-options=-v" in env["PERFLAB_NVCCFLAGS"]

    def test_python_returns_empty(self):
        assert get_diagnostic_env_vars("python") == {}


# ---------------------------------------------------------------------------
# GCC parser (legacy flat)
# ---------------------------------------------------------------------------

class TestParseGccDiagnostics:
    def test_missed_vectorization(self):
        stderr = """\
matmul.cpp:42:3: missed: not vectorized: complicated access pattern
matmul.cpp:55:7: missed: not vectorized: unsupported data-ref
"""
        findings = _parse_gcc_diagnostics(stderr)
        assert any("Missed vectorizations: 2" in f for f in findings)

    def test_successful_vectorization(self):
        stderr = """\
matmul.cpp:30:5: optimized: loop vectorized using 32 byte vectors
matmul.cpp:35:5: optimized: loop vectorized using 16 byte vectors
"""
        findings = _parse_gcc_diagnostics(stderr)
        assert any("Successful vectorizations: 2" in f for f in findings)

    def test_missed_inline(self):
        stderr = """\
matmul.cpp:10:5: missed: not inlinable: call to helper_fn
"""
        findings = _parse_gcc_diagnostics(stderr)
        assert any("Missed inlines: 1" in f for f in findings)

    def test_empty_stderr(self):
        assert _parse_gcc_diagnostics("") == []

    def test_fallback_keyword_counting(self):
        """Falls back to keyword counting for non-structured output."""
        stderr = "note: loop vectorized using 256 bit vectors\nmissed: not vectorized: bad access"
        findings = _parse_gcc_diagnostics(stderr)
        assert len(findings) >= 1


# ---------------------------------------------------------------------------
# Structured GCC remark parser
# ---------------------------------------------------------------------------

class TestParseGccRemarks:
    def test_vectorized_with_width(self):
        stderr = "matmul.cpp:14:25: optimized: loop vectorized using 32 byte vectors"
        remarks = _parse_gcc_remarks(stderr)
        assert len(remarks) == 1
        assert remarks[0].width == 256
        assert remarks[0].category == "vectorize"
        assert remarks[0].status == "applied"
        assert remarks[0].file == "matmul.cpp"
        assert remarks[0].line == 14

    def test_missed_vectorization(self):
        stderr = "matmul.cpp:17:5: missed: couldn't vectorize loop: complicated access pattern"
        remarks = _parse_gcc_remarks(stderr)
        assert len(remarks) == 1
        assert remarks[0].status == "missed"
        assert remarks[0].category == "vectorize"

    def test_inline_remark(self):
        stderr = "matmul.cpp:10:5: optimized: Inlined helper_fn/42 into main/50"
        remarks = _parse_gcc_remarks(stderr)
        assert len(remarks) == 1
        assert remarks[0].category == "inline"
        assert remarks[0].status == "applied"

    def test_unroll_remark(self):
        stderr = "matmul.cpp:14:9: optimized: loop with 4 iterations completely unrolled"
        remarks = _parse_gcc_remarks(stderr)
        assert len(remarks) == 1
        assert remarks[0].category == "unroll"

    def test_multiple_remarks(self):
        stderr = """\
matmul.cpp:14:25: optimized: loop vectorized using 32 byte vectors
matmul.cpp:17:5: missed: couldn't vectorize loop: complicated access pattern
matmul.cpp:10:5: optimized: Inlined helper_fn/42 into main/50
"""
        remarks = _parse_gcc_remarks(stderr)
        assert len(remarks) == 3

    def test_empty(self):
        assert _parse_gcc_remarks("") == []

    def test_16_byte_vectors(self):
        stderr = "matmul.cpp:14:25: optimized: loop vectorized using 16 byte vectors"
        remarks = _parse_gcc_remarks(stderr)
        assert remarks[0].width == 128


# ---------------------------------------------------------------------------
# Structured Clang remark parser
# ---------------------------------------------------------------------------

class TestParseClangRemarks:
    def test_vectorized_loop(self):
        stderr = "matmul.cpp:14:9: remark: vectorized loop (vectorization width: 8) [-Rpass=loop-vectorize]"
        remarks = _parse_clang_remarks(stderr)
        assert len(remarks) == 1
        assert remarks[0].category == "vectorize"
        assert remarks[0].status == "applied"
        assert remarks[0].width == 256  # 8 elements * 32 bits

    def test_missed_vectorization(self):
        stderr = "matmul.cpp:16:17: remark: loop not vectorized: aliasing [-Rpass-missed=loop-vectorize]"
        remarks = _parse_clang_remarks(stderr)
        assert len(remarks) == 1
        assert remarks[0].status == "missed"
        assert remarks[0].category == "vectorize"

    def test_inline_remark(self):
        stderr = "matmul.cpp:10:5: remark: inlined function helper_fn [-Rpass=inline]"
        remarks = _parse_clang_remarks(stderr)
        assert len(remarks) == 1
        assert remarks[0].category == "inline"
        assert remarks[0].status == "applied"

    def test_analysis_remark(self):
        stderr = "matmul.cpp:20:3: remark: analysis for vectorization [-Rpass-analysis=loop-vectorize]"
        remarks = _parse_clang_remarks(stderr)
        assert len(remarks) == 1
        assert remarks[0].status == "analysis"

    def test_empty(self):
        assert _parse_clang_remarks("") == []


# ---------------------------------------------------------------------------
# Cross-reference diagnostics
# ---------------------------------------------------------------------------

class TestCrossReferenceDiagnostics:
    def test_vectorization_gap(self):
        remarks = [OptimizationRemark("matmul.cpp", 14, 25, "vectorize", "applied", "using 16 byte vectors", width=128)]
        insights = cross_reference_diagnostics(remarks, {}, cpu_isa={"max_simd_width_bits": 256})
        assert any(i.category == "vectorization-gap" for i in insights)

    def test_missed_vec_at_hotspot(self):
        remarks = [OptimizationRemark("matmul.cpp", 14, 5, "vectorize", "missed", "complicated access")]
        perf_summary = {
            "annotated_hotspots": [{"function": "matmul", "hot_lines": [{"file": "matmul.cpp", "line": 14, "pct": 45.0}]}]
        }
        insights = cross_reference_diagnostics(remarks, perf_summary, cpu_isa={"max_simd_width_bits": 256})
        assert any(i.category == "missed-vec-in-hotspot" for i in insights)
        assert any(i.priority == "high" for i in insights)

    def test_alias_blocking(self):
        remarks = [OptimizationRemark("matmul.cpp", 14, 5, "alias", "missed", "possible aliasing")]
        insights = cross_reference_diagnostics(remarks, {}, cpu_isa={"max_simd_width_bits": 256})
        assert any(i.category == "alias-blocking" for i in insights)

    def test_missed_fma(self):
        remarks = [OptimizationRemark("matmul.cpp", 14, 5, "fma", "missed", "multiply-add not fused")]
        insights = cross_reference_diagnostics(remarks, {}, cpu_isa={"max_simd_width_bits": 256})
        assert any(i.category == "missed-fma" for i in insights)

    def test_no_issues_no_insights(self):
        insights = cross_reference_diagnostics([], {}, cpu_isa={"max_simd_width_bits": 256})
        assert insights == []

    def test_no_isa_no_width_gap(self):
        remarks = [OptimizationRemark("matmul.cpp", 14, 25, "vectorize", "applied", "using 16 byte vectors", width=128)]
        insights = cross_reference_diagnostics(remarks, {}, cpu_isa=None)
        assert not any(i.category == "vectorization-gap" for i in insights)

    def test_width_already_max(self):
        remarks = [OptimizationRemark("matmul.cpp", 14, 25, "vectorize", "applied", "using 32 byte vectors", width=256)]
        insights = cross_reference_diagnostics(remarks, {}, cpu_isa={"max_simd_width_bits": 256})
        assert not any(i.category == "vectorization-gap" for i in insights)


# ---------------------------------------------------------------------------
# Compiler detection
# ---------------------------------------------------------------------------

class TestDetectCompiler:
    def test_gcc(self):
        # detect_compiler uses subprocess, so we test the string parsing part
        assert detect_compiler("nvcc -o out main.cu") == "nvcc"

    def test_clang_explicit(self):
        assert detect_compiler("clang++ -O3 -o out main.cpp") == "clang"

    def test_unknown(self):
        assert detect_compiler("weird_compiler main.cpp") == "unknown"

    def test_empty(self):
        assert detect_compiler("") == "unknown"


# ---------------------------------------------------------------------------
# NVCC parser
# ---------------------------------------------------------------------------

class TestParseNvccDiagnostics:
    def test_register_usage(self):
        stderr = "ptxas info    : Used 32 registers, 8192 bytes smem, 384 bytes cmem[0]"
        findings = _parse_nvcc_diagnostics(stderr)
        assert any("Registers/thread: 32" in f for f in findings)
        assert any("Shared memory: 8192" in f for f in findings)

    def test_high_register_warning(self):
        stderr = "ptxas info    : Used 128 registers, 0 bytes smem"
        findings = _parse_nvcc_diagnostics(stderr)
        assert any("WARNING" in f and "128" in f for f in findings)

    def test_spill_detection(self):
        stderr = """\
ptxas info    : Used 64 registers, 0 bytes smem
ptxas info    : 256 bytes spill stores, 256 bytes spill loads
"""
        findings = _parse_nvcc_diagnostics(stderr)
        assert any("Spill stores" in f for f in findings)
        assert any("Spill loads" in f for f in findings)

    def test_empty_stderr(self):
        assert _parse_nvcc_diagnostics("") == []


# ---------------------------------------------------------------------------
# PyTorch parser
# ---------------------------------------------------------------------------

class TestParsePytorchDiagnostics:
    def test_graph_breaks(self):
        stderr = """\
[2024-01-01] GRAPH BREAK: unsupported op torch._C.TensorBase.item
[2024-01-01] GRAPH BREAK: data-dependent control flow
"""
        findings = _parse_pytorch_diagnostics(stderr)
        assert any("Graph breaks: 2" in f for f in findings)

    def test_eager_fallback(self):
        stderr = "[WARNING] eager fallback: aten.special_op not supported"
        findings = _parse_pytorch_diagnostics(stderr)
        assert any("Eager fallbacks: 1" in f for f in findings)

    def test_recompilation(self):
        stderr = "[dynamo] Recompiling function forward because of guard failure"
        findings = _parse_pytorch_diagnostics(stderr)
        assert any("Recompilations: 1" in f for f in findings)

    def test_fusion_events(self):
        stderr = """\
[inductor] fusing pointwise ops: add, mul
[inductor] fusion: 3 ops fused into 1 kernel
"""
        findings = _parse_pytorch_diagnostics(stderr)
        assert any("Fusion events: 2" in f for f in findings)

    def test_empty_stderr(self):
        assert _parse_pytorch_diagnostics("") == []


# ---------------------------------------------------------------------------
# JAX parser
# ---------------------------------------------------------------------------

class TestParseJaxDiagnostics:
    def test_compilation_events(self):
        stderr = """\
Finished XLA compilation of jit(train_step) in 2345 ms
Finished XLA compilation of jit(predict) in 120 ms
"""
        findings = _parse_jax_diagnostics(stderr)
        assert any("XLA compilations: 2" in f for f in findings)

    def test_recompilation(self):
        stderr = "Recompiling jit(forward) due to shape change"
        findings = _parse_jax_diagnostics(stderr)
        assert any("Recompilations: 1" in f for f in findings)

    def test_empty_stderr(self):
        assert _parse_jax_diagnostics("") == []


# ---------------------------------------------------------------------------
# Triton parser
# ---------------------------------------------------------------------------

class TestParseTritonDiagnostics:
    def test_shared_memory(self):
        stderr = "shared memory: 16384 bytes"
        findings = _parse_triton_diagnostics(stderr)
        assert any("Shared memory: 16384" in f for f in findings)

    def test_registers(self):
        stderr = "registers: 40"
        findings = _parse_triton_diagnostics(stderr)
        assert any("Registers: 40" in f for f in findings)

    def test_num_warps(self):
        stderr = "num_warps: 8"
        findings = _parse_triton_diagnostics(stderr)
        assert any("num_warps: 8" in f for f in findings)

    def test_compilation_events(self):
        stderr = """\
compiling kernel matmul_kernel
compiling kernel softmax_kernel
"""
        findings = _parse_triton_diagnostics(stderr)
        assert any("Compilation events: 2" in f for f in findings)

    def test_empty_stderr(self):
        assert _parse_triton_diagnostics("") == []


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

class TestSummarizeFindings:
    def test_warnings_first(self):
        findings = [
            "Registers/thread: 32",
            "WARNING: Spill stores: 256 bytes",
            "Shared memory: 8192 bytes",
        ]
        summary = _summarize_findings(findings)
        lines = summary.strip().splitlines()
        assert "WARNING" in lines[0]

    def test_truncation(self):
        findings = [f"Finding {i}: " + "x" * 100 for i in range(100)]
        summary = _summarize_findings(findings)
        assert len(summary) <= 2048
        assert "truncated" in summary

    def test_empty(self):
        assert _summarize_findings([]) == ""


# ---------------------------------------------------------------------------
# parse_compiler_output dispatcher
# ---------------------------------------------------------------------------

class TestParseCompilerOutput:
    def test_aot_uses_build_stderr(self):
        build_stderr = "ptxas info    : Used 32 registers, 8192 bytes smem"
        result = parse_compiler_output("cuda", build_stderr=build_stderr, bench_stderr="")
        assert any("Registers/thread: 32" in f for f in result.findings)
        assert result.program_type == "cuda"

    def test_jit_uses_bench_stderr(self):
        bench_stderr = "[dynamo] GRAPH BREAK: unsupported op"
        result = parse_compiler_output("pytorch", build_stderr="", bench_stderr=bench_stderr)
        assert any("Graph breaks: 1" in f for f in result.findings)

    def test_jit_ignores_build_stderr(self):
        build_stderr = "[dynamo] GRAPH BREAK: should not be parsed"
        result = parse_compiler_output("pytorch", build_stderr=build_stderr, bench_stderr="")
        assert result.findings == []

    def test_unknown_program_type(self):
        result = parse_compiler_output("unknown_type", build_stderr="stuff", bench_stderr="more stuff")
        assert result.findings == []
        assert result.summary == ""

    def test_python_returns_empty(self):
        result = parse_compiler_output("python", build_stderr="", bench_stderr="")
        assert result.findings == []

    def test_deduplication(self):
        # Same finding from both build and bench stderr for AOT type
        stderr = "ptxas info    : Used 32 registers, 8192 bytes smem"
        result = parse_compiler_output("cuda", build_stderr=stderr, bench_stderr=stderr)
        reg_findings = [f for f in result.findings if "Registers/thread: 32" in f]
        assert len(reg_findings) == 1

    def test_cpp_produces_remarks(self):
        stderr = "matmul.cpp:14:25: optimized: loop vectorized using 32 byte vectors"
        result = parse_compiler_output("cpp", build_stderr=stderr, compiler="gcc")
        assert len(result.remarks) == 1
        assert result.remarks[0].width == 256

    def test_cpp_clang_produces_remarks(self):
        stderr = "matmul.cpp:14:9: remark: vectorized loop (vectorization width: 8) [-Rpass=loop-vectorize]"
        result = parse_compiler_output("cpp", build_stderr=stderr, compiler="clang")
        assert len(result.remarks) == 1
        assert result.remarks[0].category == "vectorize"

    def test_cuda_produces_remarks(self):
        stderr = """\
ptxas info    : Compiling entry function '_Z12sgemm_naivePfS_S_iii' for 'sm_80'
ptxas info    : Function properties for _Z12sgemm_naivePfS_S_iii
                    0 bytes stack frame, 16 bytes spill stores, 16 bytes spill loads
ptxas info    : Used 72 registers, 8192 bytes smem, 360 bytes cmem[0]
"""
        result = parse_compiler_output("cuda", build_stderr=stderr)
        assert len(result.remarks) >= 3  # regs (high), spill stores, spill loads
        cats = [r.category for r in result.remarks]
        assert "register-pressure" in cats
        # High register usage should be "missed"
        reg_remark = [r for r in result.remarks if "72 registers" in r.detail][0]
        assert reg_remark.status == "missed"
        assert reg_remark.file == "_Z12sgemm_naivePfS_S_iii"


# ---------------------------------------------------------------------------
# NVCC structured remarks
# ---------------------------------------------------------------------------

class TestParseNvccRemarks:
    def test_register_usage_normal(self):
        stderr = """\
ptxas info    : Function properties for _Z6kernelPf
ptxas info    : Used 32 registers, 4096 bytes smem
"""
        remarks = _parse_nvcc_remarks(stderr)
        reg_remarks = [r for r in remarks if "32 registers" in r.detail]
        assert len(reg_remarks) == 1
        assert reg_remarks[0].status == "analysis"  # <= 64, not high
        assert reg_remarks[0].file == "_Z6kernelPf"
        assert reg_remarks[0].line == 0

    def test_register_usage_high(self):
        stderr = """\
ptxas info    : Compiling entry function '_Z6kernelPf' for 'sm_80'
ptxas info    : Used 96 registers, 0 bytes smem
"""
        remarks = _parse_nvcc_remarks(stderr)
        reg_remarks = [r for r in remarks if "96 registers" in r.detail]
        assert len(reg_remarks) == 1
        assert reg_remarks[0].status == "missed"
        assert "limit occupancy" in reg_remarks[0].detail

    def test_spill_detection(self):
        stderr = """\
ptxas info    : Function properties for _Z6kernelPf
                    0 bytes stack frame, 256 bytes spill stores, 128 bytes spill loads
ptxas info    : Used 40 registers
"""
        remarks = _parse_nvcc_remarks(stderr)
        spill_remarks = [r for r in remarks if "spill" in r.detail.lower()]
        assert len(spill_remarks) == 2
        assert all(r.status == "missed" for r in spill_remarks)
        assert all(r.category == "register-pressure" for r in spill_remarks)

    def test_shared_memory(self):
        stderr = """\
ptxas info    : Function properties for _Z6kernelPf
ptxas info    : Used 32 registers, 16384 bytes smem
"""
        remarks = _parse_nvcc_remarks(stderr)
        smem_remarks = [r for r in remarks if r.category == "shared-memory"]
        assert len(smem_remarks) == 1
        assert "16384" in smem_remarks[0].detail

    def test_multiple_kernels(self):
        stderr = """\
ptxas info    : Function properties for _Z6kernelAPf
ptxas info    : Used 32 registers
ptxas info    : Function properties for _Z6kernelBPf
ptxas info    : Used 96 registers
"""
        remarks = _parse_nvcc_remarks(stderr)
        a_remarks = [r for r in remarks if r.file == "_Z6kernelAPf"]
        b_remarks = [r for r in remarks if r.file == "_Z6kernelBPf"]
        assert len(a_remarks) >= 1
        assert len(b_remarks) >= 1
        assert a_remarks[0].status == "analysis"  # 32 regs, fine
        assert b_remarks[0].status == "missed"    # 96 regs, high

    def test_empty_stderr(self):
        assert _parse_nvcc_remarks("") == []

    def test_zero_spill_not_reported(self):
        stderr = """\
ptxas info    : Function properties for _Z6kernelPf
                    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers
"""
        remarks = _parse_nvcc_remarks(stderr)
        spill_remarks = [r for r in remarks if "spill" in r.detail.lower()]
        assert len(spill_remarks) == 0


# ---------------------------------------------------------------------------
# Clang type-aware width extraction
# ---------------------------------------------------------------------------

class TestInferElementBits:
    def test_double(self):
        assert _infer_element_bits("vectorized loop over double values") == 64

    def test_float_default(self):
        assert _infer_element_bits("vectorized loop") == 32

    def test_i8(self):
        assert _infer_element_bits("loop processes i8 data") == 8

    def test_half(self):
        assert _infer_element_bits("half precision computation") == 16

    def test_i64(self):
        assert _infer_element_bits("processing i64 integers") == 64

    def test_bfloat(self):
        assert _infer_element_bits("bfloat16 matmul") == 16


class TestClangWidthExtraction:
    def test_clang_width_double(self):
        """Clang remark with double type should produce 64-bit elements."""
        stderr = "test.cpp:5:3: remark: vectorized loop (vectorization width: 4, double) [-Rpass=loop-vectorize]"
        remarks = _parse_clang_remarks(stderr)
        assert len(remarks) == 1
        assert remarks[0].width == 256  # 4 * 64 bits

    def test_clang_width_float_default(self):
        """Clang remark without type info defaults to 32-bit elements."""
        stderr = "test.cpp:5:3: remark: vectorized loop (vectorization width: 8) [-Rpass=loop-vectorize]"
        remarks = _parse_clang_remarks(stderr)
        assert len(remarks) == 1
        assert remarks[0].width == 256  # 8 * 32 bits

    def test_clang_width_i8(self):
        """Clang remark with i8 type should produce 8-bit elements."""
        stderr = "test.cpp:5:3: remark: vectorized loop (vectorization width: 32, i8 elements) [-Rpass=loop-vectorize]"
        remarks = _parse_clang_remarks(stderr)
        assert len(remarks) == 1
        assert remarks[0].width == 256  # 32 * 8 bits


# ---------------------------------------------------------------------------
# CUDA cross-referencing
# ---------------------------------------------------------------------------

class TestCudaCrossReferencing:
    def test_register_pressure_in_hot_kernel(self):
        """High register usage in a kernel that dominates GPU time."""
        remarks = [
            OptimizationRemark(
                file="_Z12sgemm_naivePfS_S_iii", line=0, col=None,
                category="register-pressure", status="missed",
                detail="Used 96 registers/thread — high register pressure may limit occupancy",
            ),
        ]
        gpu_attrib = [
            {"name": "sgemm_naive", "gpu_pct": 85.0, "gpu_time_ms": 100.0},
        ]
        insights = cross_reference_diagnostics(
            remarks, perf_summary={}, gpu_attribution=gpu_attrib,
        )
        assert len(insights) == 1
        assert insights[0].category == "cuda-register-pressure"
        assert insights[0].priority == "high"
        assert insights[0].perf_pct == 85.0

    def test_register_pressure_cold_kernel(self):
        """High register usage in a kernel with low GPU time → medium priority."""
        remarks = [
            OptimizationRemark(
                file="_Z6initPf", line=0, col=None,
                category="register-pressure", status="missed",
                detail="Used 80 registers/thread — high register pressure",
            ),
        ]
        gpu_attrib = [
            {"name": "init", "gpu_pct": 5.0, "gpu_time_ms": 5.0},
        ]
        insights = cross_reference_diagnostics(
            remarks, perf_summary={}, gpu_attribution=gpu_attrib,
        )
        assert len(insights) == 1
        assert insights[0].priority == "medium"

    def test_no_gpu_attrib_skips_cuda_rules(self):
        """CUDA remarks without GPU attribution data produce no insights."""
        remarks = [
            OptimizationRemark(
                file="_Z6kernelPf", line=0, col=None,
                category="register-pressure", status="missed",
                detail="Used 96 registers/thread",
            ),
        ]
        insights = cross_reference_diagnostics(remarks, perf_summary={})
        assert len(insights) == 0


class TestFuzzyKernelMatch:
    def test_mangled_vs_demangled(self):
        assert _fuzzy_kernel_match("_Z12sgemm_naivePfS_S_iii", "sgemm_naive") is True

    def test_substring(self):
        assert _fuzzy_kernel_match("volta_sgemm_128x128", "sgemm") is True

    def test_no_match(self):
        assert _fuzzy_kernel_match("_Z6kernelAPf", "completely_different") is False

    def test_exact_match(self):
        assert _fuzzy_kernel_match("my_kernel", "my_kernel") is True


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestDiagnosticsSerialization:
    def test_remark_round_trip(self):
        remark = OptimizationRemark(
            file="matmul.cpp", line=14, col=25,
            category="vectorize", status="applied",
            detail="loop vectorized using 32 byte vectors", width=256,
        )
        d = remark.to_dict()
        restored = OptimizationRemark.from_dict(d)
        assert restored.file == "matmul.cpp"
        assert restored.line == 14
        assert restored.col == 25
        assert restored.width == 256
        assert restored.category == "vectorize"

    def test_remark_none_fields(self):
        remark = OptimizationRemark(
            file="_Z6kernelPf", line=0, col=None,
            category="register-pressure", status="missed",
            detail="Used 96 registers",
        )
        d = remark.to_dict()
        restored = OptimizationRemark.from_dict(d)
        assert restored.col is None
        assert restored.width is None

    def test_diagnostics_round_trip(self):
        diag = CompilerDiagnostics(
            program_type="cpp",
            findings=["Missed vectorizations: 2", "Successful vectorizations: 1"],
            summary="- Missed vectorizations: 2\n- Successful vectorizations: 1",
            remarks=[
                OptimizationRemark(
                    file="matmul.cpp", line=14, col=25,
                    category="vectorize", status="applied",
                    detail="loop vectorized", width=256,
                ),
                OptimizationRemark(
                    file="matmul.cpp", line=17, col=5,
                    category="vectorize", status="missed",
                    detail="couldn't vectorize loop",
                ),
            ],
        )
        d = diag.to_dict()
        restored = CompilerDiagnostics.from_dict(d)
        assert restored.program_type == "cpp"
        assert len(restored.findings) == 2
        assert len(restored.remarks) == 2
        assert restored.remarks[0].width == 256
        assert restored.remarks[1].width is None
        assert restored.summary == diag.summary

    def test_empty_diagnostics_round_trip(self):
        diag = CompilerDiagnostics(program_type="python")
        d = diag.to_dict()
        restored = CompilerDiagnostics.from_dict(d)
        assert restored.program_type == "python"
        assert restored.findings == []
        assert restored.remarks == []
