"""Tests for hot loop assembly extraction and prompt integration."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

from perflab.profilers.linux_perf import _demangle, _parse_perf_annotate, extract_hot_assembly

# ---------------------------------------------------------------------------
# Sample perf annotate output
# ---------------------------------------------------------------------------

SAMPLE_ANNOTATE = dedent("""\
 Percent |  Source code & Disassembly of matmul_bin
         :  static void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
         :      for (int i = 0; i < M; i++) {
   0.10  :  4005a0:  mov    $0x0,%ecx
         :          for (int k = 0; k < K; k++) {
   0.30  :  4005a8:  mov    $0x0,%edx
         :              float a = A[i*K+k];
   2.50  :  4005b0:  movss  (%rax,%rdx,4),%xmm0
         :              for (int j = 0; j < N; j++) {
   1.00  :  4005b8:  mov    $0x0,%r8d
         :                  C[i*N+j] += a * B[k*N+j];
  45.20  :  4005c0:  vmulss (%rbx,%r8,4),%xmm0,%xmm1
  30.10  :  4005c8:  vaddss (%rcx,%r8,4),%xmm1,%xmm1
  12.50  :  4005d0:  vmovss %xmm1,(%rcx,%r8,4)
   0.50  :  4005d8:  inc    %r8d
         :  4005dc:  cmp    %r9d,%r8d
         :  4005e0:  jl     4005c0
         :          }
         :      }
         :  }
   0.05  :  4005f0:  retq

 Percent |  Source code & Disassembly of helper_func
         :  void helper_func() {
   3.00  :  400700:  push   %rbp
   1.00  :  400708:  pop    %rbp
         :  40070c:  retq
""")

SAMPLE_ANNOTATE_SIMD = dedent("""\
 Percent |  Source code & Disassembly of matmul_avx
         :  static void matmul_avx(...) {
   8.20  :  4005c0:  vmovaps (%rbx,%r8,4),%ymm0
  35.50  :  4005c8:  vfmadd231ps (%rax,%r8,4),%ymm0,%ymm1
  20.10  :  4005d0:  vmovaps %ymm1,(%rcx,%r8,4)
   5.00  :  4005d8:  add    $0x8,%r8d
   0.30  :  4005dc:  cmp    %r9d,%r8d
         :  4005e0:  jl     4005c0
""")


SAMPLE_ANNOTATE_MANGLED = dedent("""\
 Percent |  Source code & Disassembly of _Z6matmulPKfS0_Pfiii
         :  static void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  45.20  :  4005c0:  vmulss (%rbx,%r8,4),%xmm0,%xmm1
  30.10  :  4005c8:  vaddss (%rcx,%r8,4),%xmm1,%xmm1
""")


class TestDemangle:
    """Tests for _demangle() helper."""

    def setup_method(self):
        # Clear caches between tests so mocks take effect
        _demangle.cache_clear()
        from perflab.profilers.linux_perf import _cxxfilt_available
        _cxxfilt_available.cache_clear()

    def teardown_method(self):
        _demangle.cache_clear()
        from perflab.profilers.linux_perf import _cxxfilt_available
        _cxxfilt_available.cache_clear()

    def test_returns_original_when_cxxfilt_unavailable(self):
        with patch("perflab.profilers.linux_perf.shutil.which", return_value=None):
            assert _demangle("_Z6matmulPKfS0_Pfiii") == "_Z6matmulPKfS0_Pfiii"

    def test_returns_original_for_non_mangled_name(self):
        with patch("perflab.profilers.linux_perf.shutil.which", return_value="/usr/bin/c++filt"):
            assert _demangle("matmul") == "matmul"

    def test_demangles_mangled_name(self):
        mock_result = MagicMock()
        mock_result.stdout = "matmul(float const*, float const*, float*, int, int, int)\n"
        with patch("perflab.profilers.linux_perf.shutil.which", return_value="/usr/bin/c++filt"), \
             patch("perflab.profilers.linux_perf.subprocess.run", return_value=mock_result) as mock_run:
            result = _demangle("_Z6matmulPKfS0_Pfiii")
            assert result == "matmul(float const*, float const*, float*, int, int, int)"
            mock_run.assert_called_once_with(
                ["c++filt", "_Z6matmulPKfS0_Pfiii"],
                capture_output=True,
                text=True,
                timeout=5,
            )

    def test_demangled_names_in_extract_hot_assembly(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.stdout = "matmul(float const*, float const*, float*, int, int, int)\n"
        p = tmp_path / "perf_annotate.txt"
        p.write_text(SAMPLE_ANNOTATE_MANGLED)
        with patch("perflab.profilers.linux_perf.shutil.which", return_value="/usr/bin/c++filt"), \
             patch("perflab.profilers.linux_perf.subprocess.run", return_value=mock_result):
            results = extract_hot_assembly(p)
            assert len(results) == 1
            assert results[0]["function"] == "matmul(float const*, float const*, float*, int, int, int)"


class TestExtractHotAssembly:
    """Tests for extract_hot_assembly()."""

    def test_basic_extraction(self, tmp_path: Path):
        p = tmp_path / "perf_annotate.txt"
        p.write_text(SAMPLE_ANNOTATE)
        results = extract_hot_assembly(p)
        assert len(results) >= 1
        top = results[0]
        assert top["function"] == "matmul_bin"
        assert top["hot_pct"] == 45.2
        assert "vmulss" in top["snippet"]

    def test_snippet_contains_surrounding_context(self, tmp_path: Path):
        p = tmp_path / "perf_annotate.txt"
        p.write_text(SAMPLE_ANNOTATE)
        results = extract_hot_assembly(p)
        top = results[0]
        # Should include lines around the hottest instruction
        assert "vaddss" in top["snippet"]
        assert "vmovss" in top["snippet"]

    def test_sorted_by_hottest(self, tmp_path: Path):
        p = tmp_path / "perf_annotate.txt"
        p.write_text(SAMPLE_ANNOTATE)
        results = extract_hot_assembly(p, min_pct=1.0)
        if len(results) > 1:
            assert results[0]["hot_pct"] >= results[1]["hot_pct"]

    def test_min_pct_filters(self, tmp_path: Path):
        p = tmp_path / "perf_annotate.txt"
        p.write_text(SAMPLE_ANNOTATE)
        # helper_func has max 3.0% — should be excluded at min_pct=5.0
        results = extract_hot_assembly(p, min_pct=5.0)
        funcs = [r["function"] for r in results]
        assert "helper_func" not in funcs

    def test_min_pct_includes_low_threshold(self, tmp_path: Path):
        p = tmp_path / "perf_annotate.txt"
        p.write_text(SAMPLE_ANNOTATE)
        results = extract_hot_assembly(p, min_pct=1.0)
        funcs = [r["function"] for r in results]
        assert "helper_func" in funcs

    def test_max_functions_limit(self, tmp_path: Path):
        p = tmp_path / "perf_annotate.txt"
        p.write_text(SAMPLE_ANNOTATE)
        results = extract_hot_assembly(p, max_functions=1, min_pct=1.0)
        assert len(results) == 1
        assert results[0]["function"] == "matmul_bin"

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "perf_annotate.txt"
        p.write_text("")
        assert extract_hot_assembly(p) == []

    def test_missing_file(self, tmp_path: Path):
        p = tmp_path / "does_not_exist.txt"
        assert extract_hot_assembly(p) == []

    def test_context_lines_parameter(self, tmp_path: Path):
        p = tmp_path / "perf_annotate.txt"
        p.write_text(SAMPLE_ANNOTATE)
        # Very small context
        results = extract_hot_assembly(p, context_lines=1)
        top = results[0]
        lines = top["snippet"].splitlines()
        # Should be a small window (hottest line + 1 before + 1 after = ~3)
        assert len(lines) <= 5

    def test_simd_instructions_visible(self, tmp_path: Path):
        p = tmp_path / "perf_annotate.txt"
        p.write_text(SAMPLE_ANNOTATE_SIMD)
        results = extract_hot_assembly(p)
        assert len(results) == 1
        top = results[0]
        assert "vfmadd231ps" in top["snippet"]
        assert "vmovaps" in top["snippet"]


class TestPromptIntegration:
    """Tests for hot_loop_assembly in PromptContext and build_prompt()."""

    def test_prompt_context_field(self):
        from perflab.optimizers.prompt import PromptContext
        ctx = PromptContext(
            source_files={},
            profiler_summaries={},
            bench_results={},
            hot_loop_assembly=[{"function": "foo", "hot_pct": 45.0, "snippet": "vmulss ..."}],
        )
        assert ctx.hot_loop_assembly is not None
        assert len(ctx.hot_loop_assembly) == 1

    def test_prompt_context_default_none(self):
        from perflab.optimizers.prompt import PromptContext
        ctx = PromptContext(
            source_files={},
            profiler_summaries={},
            bench_results={},
        )
        assert ctx.hot_loop_assembly is None

    def test_build_prompt_includes_assembly(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt
        ctx = PromptContext(
            source_files={"matmul.cpp": "void matmul() {}"},
            profiler_summaries={},
            bench_results={"throughput": {"median": 0.5}},
            allowed_paths=["matmul.cpp"],
            hot_loop_assembly=[
                {
                    "function": "matmul_kernel",
                    "hot_pct": 45.2,
                    "snippet": "  45.20 :  4005c0:  vmulss ...\n  30.10 :  4005c8:  vaddss ...",
                },
            ],
        )
        messages = build_prompt(ctx)
        # Find the assembly section in the prompt content
        full_text = "\n".join(m.content for m in messages)
        assert "Hot loop assembly" in full_text
        assert "matmul_kernel" in full_text
        assert "45.2% CPU" in full_text
        assert "vmulss" in full_text
        assert "SIMD instructions" in full_text

    def test_build_prompt_excludes_when_none(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt
        ctx = PromptContext(
            source_files={"matmul.cpp": "void matmul() {}"},
            profiler_summaries={},
            bench_results={"throughput": {"median": 0.5}},
            allowed_paths=["matmul.cpp"],
            hot_loop_assembly=None,
        )
        messages = build_prompt(ctx)
        full_text = "\n".join(m.content for m in messages)
        assert "Hot loop assembly" not in full_text

    def test_build_prompt_multiple_functions(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt
        ctx = PromptContext(
            source_files={"matmul.cpp": "void matmul() {}"},
            profiler_summaries={},
            bench_results={"throughput": {"median": 0.5}},
            allowed_paths=["matmul.cpp"],
            hot_loop_assembly=[
                {"function": "hot_func_a", "hot_pct": 50.0, "snippet": "vaddps ..."},
                {"function": "hot_func_b", "hot_pct": 20.0, "snippet": "addss ..."},
            ],
        )
        messages = build_prompt(ctx)
        full_text = "\n".join(m.content for m in messages)
        assert "hot_func_a" in full_text
        assert "hot_func_b" in full_text
        assert "50.0% CPU" in full_text
        assert "20.0% CPU" in full_text


class TestParseAnnotateExisting:
    """Ensure the existing _parse_perf_annotate still works correctly."""

    def test_parse_source_hotspots(self, tmp_path: Path):
        """_parse_perf_annotate extracts source lines (file:line: format)."""
        source_annotate = dedent("""\
         Percent |  Source code & Disassembly of matmul_bin
                 :  static void matmul(...) {
          45.20  :  matmul.cpp:14:     sum += A[i*K+k] * B[k*N+j];
          12.10  :  matmul.cpp:15:     // next line
        """)
        p = tmp_path / "perf_annotate.txt"
        p.write_text(source_annotate)
        results = _parse_perf_annotate(p)
        assert len(results) >= 1
        top = results[0]
        assert top["function"] == "matmul_bin"
        assert len(top["hot_lines"]) > 0
        pcts = [h["pct"] for h in top["hot_lines"]]
        assert 45.2 in pcts
