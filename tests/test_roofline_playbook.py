"""Tests for roofline-aware optimization playbook."""
from __future__ import annotations

import pytest
from perflab.optimizers.prompt import (
    PromptContext,
    _classify_bound,
    _build_optimization_playbook,
    _BOUND_ACTIONS,
)


def _make_ctx(
    peak_tflops=10.0,
    peak_mem_bw_gbs=900.0,
    computed_ai=None,
    achieved_bw_gbs=None,
    achieved_tflops=None,
    program_type="cuda",
    meta_ai=None,
) -> PromptContext:
    roofline = {
        "peak_tflops": peak_tflops,
        "peak_mem_bw_gbs": peak_mem_bw_gbs,
    }
    if computed_ai is not None:
        roofline["computed_ai"] = computed_ai
    if achieved_bw_gbs is not None:
        roofline["achieved_bw_gbs"] = achieved_bw_gbs
    if achieved_tflops is not None:
        roofline["computed_achieved_tflops"] = achieved_tflops

    bench = {"tflops": {"median": achieved_tflops or 1.0}}
    if meta_ai is not None:
        bench["meta"] = {"arithmetic_intensity": meta_ai}

    return PromptContext(
        source_files={},
        profiler_summaries={},
        bench_results=bench,
        roofline=roofline,
        history=[],
        allowed_paths=[],
        n_candidates=1,
        program_type=program_type,
    )


class TestClassifyBound:
    def test_memory_bound(self):
        # knee = 1000 * 10 / 900 ≈ 11.1, AI=2 → memory-bound
        ctx = _make_ctx(computed_ai=2.0)
        result = _classify_bound(ctx)
        assert result is not None
        assert result["bound"] == "memory-bound"
        assert result["ai"] == 2.0
        assert result["knee_ai"] == pytest.approx(1000.0 * 10.0 / 900.0, rel=0.01)

    def test_compute_bound(self):
        # knee ≈ 11.1, AI=200 → compute-bound
        ctx = _make_ctx(computed_ai=200.0)
        result = _classify_bound(ctx)
        assert result is not None
        assert result["bound"] == "compute-bound"

    def test_at_knee_is_memory_bound(self):
        # AI exactly at knee → not greater, so memory-bound
        knee = 1000.0 * 10.0 / 900.0
        ctx = _make_ctx(computed_ai=knee)
        result = _classify_bound(ctx)
        assert result is not None
        assert result["bound"] == "memory-bound"

    def test_no_roofline_returns_none(self):
        ctx = PromptContext(
            source_files={}, profiler_summaries={}, bench_results={},
            roofline=None, history=[], allowed_paths=[], n_candidates=1,
            program_type="cuda",
        )
        assert _classify_bound(ctx) is None

    def test_no_ai_returns_none(self):
        ctx = _make_ctx(computed_ai=None, meta_ai=None)
        assert _classify_bound(ctx) is None

    def test_meta_ai_fallback(self):
        # computed_ai absent, meta.arithmetic_intensity present
        ctx = _make_ctx(computed_ai=None, meta_ai=2.0)
        result = _classify_bound(ctx)
        assert result is not None
        assert result["bound"] == "memory-bound"

    def test_bw_pct_computed(self):
        ctx = _make_ctx(computed_ai=2.0, achieved_bw_gbs=450.0)
        result = _classify_bound(ctx)
        assert result["bw_pct"] == pytest.approx(50.0, rel=0.01)

    def test_compute_pct_computed(self):
        ctx = _make_ctx(computed_ai=200.0, achieved_tflops=5.0)
        result = _classify_bound(ctx)
        assert result["compute_pct"] == pytest.approx(50.0, rel=0.01)

    def test_no_bw_data(self):
        ctx = _make_ctx(computed_ai=2.0, achieved_bw_gbs=None)
        result = _classify_bound(ctx)
        assert result["bw_pct"] is None

    def test_zero_peak_returns_none(self):
        ctx = _make_ctx(peak_tflops=0.0, computed_ai=2.0)
        assert _classify_bound(ctx) is None


class TestBoundActionsExist:
    """Verify _BOUND_ACTIONS covers all expected program types."""

    @pytest.mark.parametrize("program_type", ["pytorch", "cuda", "triton", "jax", "cpp", "python"])
    def test_memory_bound_actions_exist(self, program_type):
        assert ("memory-bound", program_type) in _BOUND_ACTIONS
        assert len(_BOUND_ACTIONS[("memory-bound", program_type)]) >= 3

    @pytest.mark.parametrize("program_type", ["pytorch", "cuda", "triton", "jax", "cpp", "python"])
    def test_compute_bound_actions_exist(self, program_type):
        assert ("compute-bound", program_type) in _BOUND_ACTIONS
        assert len(_BOUND_ACTIONS[("compute-bound", program_type)]) >= 3


class TestPlaybookIntegration:
    def test_memory_bound_playbook_contains_bound_actions(self):
        ctx = _make_ctx(computed_ai=2.0, program_type="cuda")
        playbook = _build_optimization_playbook(ctx, primary_bottleneck=None)
        assert "memory-bound" in playbook
        assert "Roofline analysis" in playbook
        assert "coalesced" in playbook.lower() or "shared memory" in playbook.lower()

    def test_compute_bound_playbook_contains_bound_actions(self):
        ctx = _make_ctx(computed_ai=200.0, program_type="cuda")
        playbook = _build_optimization_playbook(ctx, primary_bottleneck=None)
        assert "compute-bound" in playbook
        assert "Tensor Core" in playbook

    def test_no_roofline_falls_back_to_tiers(self):
        ctx = PromptContext(
            source_files={}, profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None, history=[], allowed_paths=[], n_candidates=1,
            program_type="cuda",
        )
        playbook = _build_optimization_playbook(ctx, primary_bottleneck=None)
        assert "Roofline analysis" not in playbook
        # Should still have tier actions
        assert "Tier-appropriate" in playbook or "Optimization" in playbook

    def test_knee_mentioned_in_playbook(self):
        ctx = _make_ctx(computed_ai=2.0, program_type="cuda")
        playbook = _build_optimization_playbook(ctx, primary_bottleneck=None)
        assert "knee" in playbook.lower()

    def test_bw_reasoning_in_memory_bound(self):
        ctx = _make_ctx(computed_ai=2.0, achieved_bw_gbs=100.0, program_type="cuda")
        playbook = _build_optimization_playbook(ctx, primary_bottleneck=None)
        # 100/900 ≈ 11% → "very low"
        assert "very low" in playbook.lower() or "bandwidth" in playbook.lower()

    def test_compute_reasoning_in_compute_bound(self):
        ctx = _make_ctx(computed_ai=200.0, achieved_tflops=2.0, program_type="cuda")
        playbook = _build_optimization_playbook(ctx, primary_bottleneck=None)
        # 2/10 = 20% → "very low"
        assert "very low" in playbook.lower() or "compute" in playbook.lower()

    def test_all_program_types_produce_playbook(self):
        for pt in ["pytorch", "cuda", "triton", "jax", "cpp", "python"]:
            ctx = _make_ctx(computed_ai=2.0, program_type=pt)
            playbook = _build_optimization_playbook(ctx, primary_bottleneck=None)
            assert "memory-bound" in playbook
