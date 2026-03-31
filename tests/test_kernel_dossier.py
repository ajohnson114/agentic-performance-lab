"""Tests for unified kernel dossier: attribution + NCU + SASS matching and rendering."""
from __future__ import annotations

import pytest

from perflab.analyzers.gpu_attribution import (
    KernelDossier,
    _match_kernel,
    build_kernel_dossiers,
)


# ---------------------------------------------------------------------------
# Fuzzy kernel name matching
# ---------------------------------------------------------------------------

class TestMatchKernel:
    def test_exact_match(self):
        candidates = [{"name": "sgemm_naive"}, {"name": "other_kernel"}]
        result = _match_kernel("sgemm_naive", candidates)
        assert result is not None
        assert result["name"] == "sgemm_naive"

    def test_substring_match(self):
        """NSys might use 'volta_sgemm_128x128_nn', NCU uses 'sgemm_naive'."""
        candidates = [{"name": "sgemm_naive"}, {"name": "relu_kernel"}]
        result = _match_kernel("volta_sgemm_128x128_nn", candidates)
        assert result is not None
        assert result["name"] == "sgemm_naive"

    def test_base_name_match(self):
        """Mangled name matches demangled base."""
        candidates = [
            {"name": "sgemm_naive(int, int, int, float const*, float*, float*)"},
            {"name": "other_kernel"},
        ]
        result = _match_kernel("sgemm_naive", candidates)
        assert result is not None
        assert "sgemm_naive" in result["name"]

    def test_demangled_vs_mangled(self):
        """Match demangled SASS name against NCU name."""
        candidates = [
            {"name": "hgemm_wmma_naive"},
            {"name": "sgemm_naive"},
        ]
        result = _match_kernel("hgemm_wmma_naive(int, int, half const*)", candidates)
        assert result is not None
        assert "hgemm" in result["name"]

    def test_token_overlap_match(self):
        """Partial token matching for loosely related names."""
        candidates = [
            {"name": "cutlass_80_tensorop_s1688gemm_256x128"},
            {"name": "elementwise_kernel"},
        ]
        result = _match_kernel("gemm_kernel", candidates)
        assert result is not None
        assert "gemm" in result["name"].lower()

    def test_no_match_returns_none(self):
        candidates = [{"name": "relu_kernel"}, {"name": "softmax_kernel"}]
        result = _match_kernel("sgemm_naive", candidates)
        assert result is None

    def test_empty_candidates(self):
        assert _match_kernel("sgemm", []) is None

    def test_empty_target(self):
        assert _match_kernel("", [{"name": "kern"}]) is None

    def test_custom_key(self):
        candidates = [{"kernel": "sgemm_naive"}, {"kernel": "other"}]
        result = _match_kernel("sgemm_naive", candidates, key="kernel")
        assert result is not None
        assert result["kernel"] == "sgemm_naive"


# ---------------------------------------------------------------------------
# Building dossiers
# ---------------------------------------------------------------------------

class TestBuildKernelDossiers:
    def test_basic_join(self):
        attrib = [
            {"name": "sgemm_naive", "gpu_pct": 85.0, "gpu_time_ms": 120.0,
             "diagnosis": "Kernel dominates", "suggestions": ["Optimize this"]},
        ]
        ncu = {
            "kernels": [
                {"name": "sgemm_naive", "invocations": 10,
                 "sm_utilization_pct": 45.0, "memory_throughput_pct": 82.0,
                 "compute_throughput_pct": 23.0, "tensor_core_utilization_pct": 0.0,
                 "achieved_occupancy_pct": 35.0, "dominant_stall_reason": "long_scoreboard",
                 "dominant_stall_pct": 42.0, "occupancy_limit_registers_pct": 38.0},
            ],
        }
        sass = [
            {"kernel": "sgemm_naive", "snippet": "/*0000*/ FFMA R2, R5, R6, R2 ;",
             "instruction_count": 50},
        ]

        dossiers = build_kernel_dossiers(attrib, ncu, sass)
        assert len(dossiers) == 1
        d = dossiers[0]
        assert d.name == "sgemm_naive"
        assert d.gpu_pct == 85.0
        assert d.ncu_metrics is not None
        assert d.ncu_metrics["sm_utilization_pct"] == 45.0
        assert d.ncu_metrics["dominant_stall_reason"] == "long_scoreboard"
        assert d.sass_snippet is not None
        assert "FFMA" in d.sass_snippet
        assert d.sass_instruction_count == 50

    def test_multiple_kernels_ranked(self):
        attrib = [
            {"name": "sgemm_naive", "gpu_pct": 70.0, "gpu_time_ms": 100.0},
            {"name": "relu_kernel", "gpu_pct": 20.0, "gpu_time_ms": 30.0},
        ]
        ncu = {"kernels": [
            {"name": "sgemm_naive", "invocations": 10, "sm_utilization_pct": 50.0},
            {"name": "relu_kernel", "invocations": 5, "sm_utilization_pct": 80.0},
        ]}
        sass = [
            {"kernel": "relu_kernel", "snippet": "/*0000*/ FADD R2, R3, R4 ;", "instruction_count": 20},
            {"kernel": "sgemm_naive", "snippet": "/*0000*/ FFMA R2, R5, R6, R2 ;", "instruction_count": 50},
        ]

        dossiers = build_kernel_dossiers(attrib, ncu, sass)
        assert len(dossiers) == 2
        # Order follows attribution ranking (by GPU time), not SASS instruction count
        assert dossiers[0].name == "sgemm_naive"
        assert dossiers[1].name == "relu_kernel"

    def test_partial_data(self):
        """Dossier still works with missing NCU or SASS."""
        attrib = [
            {"name": "my_kernel", "gpu_pct": 90.0, "gpu_time_ms": 200.0},
        ]
        dossiers = build_kernel_dossiers(attrib, None, None)
        assert len(dossiers) == 1
        assert dossiers[0].ncu_metrics is None
        assert dossiers[0].sass_snippet is None

    def test_fuzzy_match_across_sources(self):
        """Names differ across NSys, NCU, and SASS."""
        attrib = [
            {"name": "volta_sgemm_128x128_nn", "gpu_pct": 80.0, "gpu_time_ms": 100.0},
        ]
        ncu = {"kernels": [
            {"name": "sgemm_naive(int, int, int, float const*, float*, float*)",
             "invocations": 10, "sm_utilization_pct": 40.0},
        ]}
        sass = [
            {"kernel": "sgemm_naive_kernel", "snippet": "FFMA R2, R5, R6, R2 ;",
             "instruction_count": 30},
        ]

        dossiers = build_kernel_dossiers(attrib, ncu, sass)
        assert len(dossiers) == 1
        d = dossiers[0]
        # NCU should match via "sgemm" base name overlap
        assert d.ncu_metrics is not None
        assert d.ncu_metrics["sm_utilization_pct"] == 40.0
        # SASS should match via "sgemm" base name overlap
        assert d.sass_snippet is not None

    def test_no_attribution_returns_empty(self):
        assert build_kernel_dossiers(None, None, None) == []
        assert build_kernel_dossiers([], None, None) == []

    def test_max_kernels_respected(self):
        attrib = [
            {"name": f"kernel_{i}", "gpu_pct": 10.0, "gpu_time_ms": 5.0}
            for i in range(10)
        ]
        dossiers = build_kernel_dossiers(attrib, None, None, max_kernels=2)
        assert len(dossiers) == 2


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

class TestDossierPromptRendering:
    def test_dossier_renders_in_prompt(self):
        from perflab.analyzers.gpu_attribution import KernelDossier
        from perflab.optimizers.prompt import PromptContext, build_prompt

        dossier = KernelDossier(
            name="sgemm_naive",
            gpu_pct=85.0,
            gpu_time_ms=120.0,
            ncu_metrics={
                "sm_utilization_pct": 45.0,
                "memory_throughput_pct": 82.0,
                "compute_throughput_pct": 23.0,
                "tensor_core_utilization_pct": 0.0,
                "achieved_occupancy_pct": 35.0,
                "dominant_stall_reason": "long_scoreboard",
                "dominant_stall_pct": 42.0,
            },
            sass_snippet="/*0000*/ FFMA R2, R5, R6, R2 ;",
            sass_instruction_count=50,
            suggestions=["Use shared memory tiling"],
        )

        ctx = PromptContext(
            source_files={"kern.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["kern.cu"],
            program_type="cuda",
            kernel_dossiers=[dossier],
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        # Should contain dossier-style rendering
        assert "sgemm_naive" in full_text
        assert "85%" in full_text
        assert "SM util 45%" in full_text
        assert "Memory-bound" in full_text
        assert "TC util 0%" in full_text
        assert "long_scoreboard" in full_text
        assert "FFMA" in full_text
        assert "shared memory tiling" in full_text

    def test_dossier_suppresses_standalone_sass(self):
        """When dossiers are present, standalone SASS section should not render."""
        from perflab.analyzers.gpu_attribution import KernelDossier
        from perflab.optimizers.prompt import PromptContext, build_prompt

        dossier = KernelDossier(
            name="my_kernel", gpu_pct=90.0, gpu_time_ms=100.0,
            sass_snippet="FFMA R2, R5, R6, R2 ;", sass_instruction_count=10,
        )

        ctx = PromptContext(
            source_files={"kern.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["kern.cu"],
            program_type="cuda",
            kernel_dossiers=[dossier],
            cuda_sass=[{"kernel": "my_kernel", "snippet": "FFMA", "instruction_count": 10}],
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        # Should NOT have the standalone SASS section header
        assert "CUDA SASS disassembly (GPU assembly)" not in full_text
        # Should have dossier-style rendering
        assert "GPU kernel analysis" in full_text

    def test_fallback_to_separate_sections(self):
        """Without dossiers, GPU attribution and SASS render separately."""
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"kern.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["kern.cu"],
            program_type="cuda",
            gpu_attribution=[{
                "rank": 1, "name": "sgemm", "gpu_pct": 80.0,
                "diagnosis": "Kernel dominates", "suggestions": [],
            }],
            cuda_sass=[{"kernel": "sgemm", "snippet": "FFMA", "instruction_count": 10}],
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        # Should have both separate sections
        assert "GPU attribution" in full_text
        assert "CUDA SASS disassembly" in full_text
