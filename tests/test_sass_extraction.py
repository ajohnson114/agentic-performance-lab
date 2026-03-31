"""Tests for CUDA SASS disassembly extraction via cuobjdump."""
from __future__ import annotations

from pathlib import Path

import pytest

from perflab.profilers.ncu_profiler import _parse_sass_dump, _demangle_kernel_name


# ---------------------------------------------------------------------------
# SASS dump parsing
# ---------------------------------------------------------------------------

SAMPLE_SASS = """\
code for sm_80
\tFunction : _Z18sgemm_naive_kerneliiiiPKfS0_Pf
\t.headerflags    @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM80 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM80)"
\t/*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
\t/*0010*/                   S2R R0, SR_CTAID.X ;
\t/*0020*/                   S2R R3, SR_TID.X ;
\t/*0030*/                   IMAD R0, R0, c[0x0][0x0], R3 ;
\t/*0040*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x168], PT ;
\t/*0050*/                   @P0 EXIT ;
\t/*0060*/                   S2R R3, SR_CTAID.Y ;
\t/*0070*/                   S2R R7, SR_TID.Y ;
\t/*0080*/                   IMAD R3, R3, c[0x0][0x4], R7 ;
\t/*0090*/                   ISETP.GE.AND P0, PT, R3, c[0x0][0x160], PT ;
\t/*00a0*/                   @P0 EXIT ;
\t/*00b0*/                   MOV R2, RZ ;
\t/*00c0*/                   FFMA R2, R5, R6, R2 ;
\t/*00d0*/                   FFMA R2, R5, R6, R2 ;
\t/*00e0*/                   FFMA R2, R5, R6, R2 ;
\t/*00f0*/                   STG.E [R4.64], R2 ;
\tFunction : _Z12hgemm_wmma_niiPKDhS0_Pf
\t.headerflags    @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM80"
\t/*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
\t/*0010*/                   HMMA.16816.F32 R4, R0, R2, R4 ;
\t/*0020*/                   HMMA.16816.F32 R8, R0, R6, R8 ;
\t/*0030*/                   STG.E [R10.64], R4 ;
"""


class TestParseSassDump:
    def test_basic_parsing(self):
        results = _parse_sass_dump(SAMPLE_SASS)
        assert len(results) == 2

    def test_kernel_names_extracted(self):
        results = _parse_sass_dump(SAMPLE_SASS)
        names = [r["kernel"] for r in results]
        # Should be demangled or at least have the base name
        assert any("sgemm" in n.lower() for n in names)
        assert any("hgemm" in n.lower() or "wmma" in n.lower() for n in names)

    def test_instruction_count(self):
        results = _parse_sass_dump(SAMPLE_SASS)
        # First kernel (sgemm) has more instructions, should be sorted first
        assert results[0]["instruction_count"] > results[1]["instruction_count"]
        assert results[0]["instruction_count"] == 16  # 16 instructions in sgemm
        assert results[1]["instruction_count"] == 4   # 4 instructions in hgemm

    def test_snippet_contains_instructions(self):
        results = _parse_sass_dump(SAMPLE_SASS)
        for r in results:
            assert "/*" in r["snippet"]  # SASS address markers
            assert len(r["snippet"]) > 0

    def test_max_kernels_limit(self):
        results = _parse_sass_dump(SAMPLE_SASS, max_kernels=1)
        assert len(results) == 1

    def test_snippet_truncation(self):
        """Large kernels should be truncated with head + tail."""
        # Create a kernel with 50 instructions
        lines = ["code for sm_80", "\tFunction : _Z4bigfv"]
        for i in range(50):
            lines.append(f"\t/*{i:04x}*/                   FFMA R2, R5, R6, R2 ;")
        text = "\n".join(lines)

        results = _parse_sass_dump(text, context_lines=5)
        assert len(results) == 1
        assert "omitted" in results[0]["snippet"]
        # Should have 5 head + 1 omission line + 5 tail = 11 lines
        snippet_lines = results[0]["snippet"].splitlines()
        assert len(snippet_lines) == 11

    def test_empty_input(self):
        assert _parse_sass_dump("") == []

    def test_no_function_header(self):
        assert _parse_sass_dump("code for sm_80\nrandom text\n") == []

    def test_sass_contains_hmma(self):
        """Verify we can see Tensor Core instructions in SASS."""
        results = _parse_sass_dump(SAMPLE_SASS)
        hgemm = [r for r in results if "hgemm" in r["kernel"].lower() or "wmma" in r["kernel"].lower()]
        assert len(hgemm) >= 1
        assert "HMMA" in hgemm[0]["snippet"]


# ---------------------------------------------------------------------------
# Kernel name demangling
# ---------------------------------------------------------------------------

class TestDemangleKernelName:
    def test_mangled_name(self):
        result = _demangle_kernel_name("_Z18sgemm_naive_kerneliiiiPKfS0_Pf")
        assert "sgemm_naive_kernel" in result

    def test_simple_mangled(self):
        result = _demangle_kernel_name("_Z6myFuncPfi")
        assert "myFunc" in result

    def test_non_mangled_passthrough(self):
        result = _demangle_kernel_name("plain_name")
        assert result == "plain_name"


# ---------------------------------------------------------------------------
# Integration: SASS in prompt context
# ---------------------------------------------------------------------------

class TestSassInPrompt:
    def test_prompt_renders_sass(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"kern.cu": "__global__ void kern() {}"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["kern.cu"],
            program_type="cuda",
            cuda_sass=[{
                "kernel": "sgemm_naive",
                "snippet": "/*0000*/ FFMA R2, R5, R6, R2 ;",
                "instruction_count": 50,
            }],
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages if isinstance(m.content, str))
        assert "SASS" in full_text
        assert "FFMA" in full_text
        assert "sgemm_naive" in full_text

    def test_prompt_without_sass(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"kern.cu": "__global__ void kern() {}"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["kern.cu"],
            program_type="cuda",
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages if isinstance(m.content, str))
        assert "SASS disassembly" not in full_text
