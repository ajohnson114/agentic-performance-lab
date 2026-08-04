"""Tests for the modernized CPU/C++ optimization guidance in the playbook.

Covers the three things the C++ guidance has to get right:
  1. compute-bound CPU work leads with FMA-latency hiding (independent
     accumulators), register blocking / microkernel, and panel packing;
  2. the dated advice (_mm_prefetch, non-temporal stores) survives only in
     qualified form — stating when it helps AND when it hurts;
  3. CPU tasks get CPU-flavored utilization reasoning, not GPU wording
     (SM occupancy / warp divergence / Tensor Cores).
"""
from __future__ import annotations

import pytest

from perflab.optimizers.prompt import (
    _BOUND_ACTIONS,
    _BW_REASONING,
    _BW_REASONING_CPU,
    _COMPUTE_REASONING,
    _COMPUTE_REASONING_CPU,
    _TIER_ACTIONS,
    PromptContext,
    _add_perf_vs_peak,
    _build_optimization_playbook,
    _reasoning_tiers,
)


def _cpp_ctx(*, ai: float, achieved_tflops: float, achieved_bw: float,
             program_type: str = "cpp") -> PromptContext:
    """Context resembling the matmul/cpp demo task (CPU peaks, knee = 5 FLOP/byte)."""
    return PromptContext(
        source_files={},
        profiler_summaries={},
        bench_results={"tflops": {"median": achieved_tflops}},
        roofline={
            "peak_tflops": 0.5,
            "peak_mem_bw_gbs": 100.0,
            "computed_ai": ai,
            "computed_achieved_tflops": achieved_tflops,
            "achieved_bw_gbs": achieved_bw,
        },
        history=[],
        allowed_paths=["matmul.cpp"],
        n_candidates=4,
        program_type=program_type,
        target_hardware="CPU",
    )


def _compute_bound_cpp() -> str:
    return _build_optimization_playbook(
        _cpp_ctx(ai=60.0, achieved_tflops=0.04, achieved_bw=8.0), primary_bottleneck=None
    )


def _memory_bound_cpp() -> str:
    return _build_optimization_playbook(
        _cpp_ctx(ai=0.5, achieved_tflops=0.02, achieved_bw=25.0), primary_bottleneck=None
    )


def _joined(bound: str, program_type: str) -> str:
    return "\n".join(_BOUND_ACTIONS[(bound, program_type)]).lower()


class TestComputeBoundCppActions:
    """The levers that separate 5% of peak from 80% on a CPU matmul."""

    @pytest.mark.parametrize(
        "needle",
        [
            "independent accumulator",  # FMA dependency chain — the dominant lever
            "fma latency is ~4 cycles",  # the reason: latency x ports
            "2 fma ports",
            "register-block",  # outer-product microkernel
            "outer product",
            "blis",  # the structure being described
            "register file",  # tile chosen from the register budget
            "spill",  # and the failure mode when it's overshot
            "pack a and b panels",  # contiguous, cache/TLB-friendly panels
            "tlb",
            "unroll the k loop",  # ILP, but only after accumulators are independent
            "64-byte aligned",  # alignment
            "openmp",  # parallel scaling (matmul/cpp_parallel)
            "__restrict__",
        ],
    )
    def test_action_present(self, needle):
        assert needle in _joined("compute-bound", "cpp"), f"missing compute-bound cpp guidance: {needle}"

    def test_accumulators_listed_first(self):
        """Highest-value lever leads — the agent works top-down through this list."""
        first = _BOUND_ACTIONS[("compute-bound", "cpp")][0].lower()
        assert "independent accumulator" in first

    def test_output_kept_out_of_inner_loop(self):
        joined = _joined("compute-bound", "cpp")
        assert "innermost loop" in joined
        assert "store-to-load" in joined or "serializes" in joined

    def test_no_unqualified_prefetch_or_streaming_store_advice(self):
        """Dated advice must not appear at all in the compute-bound list."""
        joined = _joined("compute-bound", "cpp")
        assert "_mm_prefetch" not in joined
        assert "_mm_stream" not in joined

    def test_reaches_playbook_for_compute_bound_cpp(self):
        playbook = _compute_bound_cpp()
        assert "compute-bound" in playbook
        assert "independent accumulator" in playbook.lower()
        assert "microkernel" in playbook.lower()


class TestMemoryBoundCppActions:
    @pytest.mark.parametrize(
        "needle",
        [
            "stride-1",  # loop order — the naive i,j,k baseline's core defect
            "i,k,j",
            "l2",  # blocking per cache level, not one tile size
            "l3",
            "pack each reused tile",  # packing shows up on the memory side too
            "first-touch",  # NUMA placement for the OpenMP variant
            "omp_proc_bind",
            "__restrict__",
        ],
    )
    def test_action_present(self, needle):
        assert needle in _joined("memory-bound", "cpp"), f"missing memory-bound cpp guidance: {needle}"

    def test_loop_order_listed_first(self):
        first = _BOUND_ACTIONS[("memory-bound", "cpp")][0].lower()
        assert "stride-1" in first

    def test_reaches_playbook_for_memory_bound_cpp(self):
        playbook = _memory_bound_cpp().lower()
        assert "memory-bound" in playbook
        assert "stride-1" in playbook


class TestDatedAdviceIsQualifiedNotDeleted:
    """_mm_prefetch / non-temporal stores stay, but only with their conditions."""

    def _entry(self, needle: str) -> str:
        matches = [a for a in _BOUND_ACTIONS[("memory-bound", "cpp")] if needle in a]
        assert len(matches) == 1, f"expected exactly one entry mentioning {needle}"
        return matches[0].lower()

    def test_non_temporal_stores_state_both_conditions(self):
        entry = self._entry("_mm256_stream_ps")
        # When it helps: write-once output, RFO avoided
        assert "written once" in entry
        assert "read-for-ownership" in entry
        # When it hurts: the data is re-read / accumulated into
        assert "re-read" in entry
        assert "accumulated into" in entry
        # And the correctness caveat
        assert "_mm_sfence" in entry

    def test_software_prefetch_states_both_conditions(self):
        entry = self._entry("_mm_prefetch")
        # The common case: hardware prefetcher already wins on linear access
        assert "hardware prefetcher" in entry
        assert "linear access" in entry
        # The case where it is right: irregular / indirect addressing
        assert "indirect" in entry or "irregular" in entry
        assert "pointer chase" in entry or "gather" in entry
        # And the instruction to verify rather than assume
        assert "measure" in entry

    def test_micro_tier_prefetch_entry_is_qualified(self):
        micro = "\n".join(_TIER_ACTIONS[("micro", "cpp")]).lower()
        assert "hardware prefetcher" in micro
        assert "irregular" in micro or "indirect" in micro

    def test_micro_tier_non_temporal_entry_is_qualified(self):
        micro = "\n".join(_TIER_ACTIONS[("micro", "cpp")]).lower()
        assert "non-temporal" in micro
        assert "write-once" in micro
        assert "never for accumulators" in micro


class TestCppTierActionsModernized:
    def test_kernel_tier_has_microkernel_guidance(self):
        kernel = "\n".join(_TIER_ACTIONS[("kernel", "cpp")]).lower()
        assert "independent accumulators" in kernel
        assert "register-block" in kernel
        assert "pack" in kernel

    def test_kernel_tier_dropped_bare_prefetch_advice(self):
        kernel = "\n".join(_TIER_ACTIONS[("kernel", "cpp")]).lower()
        assert "use software prefetching for streaming access" not in kernel

    def test_standard_tier_has_loop_order_and_fma(self):
        standard = "\n".join(_TIER_ACTIONS[("standard", "cpp")]).lower()
        assert "stride-1" in standard
        assert "fmadd" in standard or "fma" in standard

    def test_fine_tune_tier_tunes_microkernel_and_blocks(self):
        fine = "\n".join(_TIER_ACTIONS[("fine_tune", "cpp")]).lower()
        assert "mr×nr" in fine or "mrxnr" in fine
        assert "spill" in fine


class TestCpuReasoningTiers:
    def test_reasoning_tiers_selects_cpu_tables_for_cpp(self):
        assert _reasoning_tiers("memory-bound", "cpp") is _BW_REASONING_CPU
        assert _reasoning_tiers("compute-bound", "cpp") is _COMPUTE_REASONING_CPU

    @pytest.mark.parametrize("program_type", ["cuda", "pytorch", "triton", "jax", "python"])
    def test_reasoning_tiers_unchanged_for_non_cpp(self, program_type):
        assert _reasoning_tiers("memory-bound", program_type) is _BW_REASONING
        assert _reasoning_tiers("compute-bound", program_type) is _COMPUTE_REASONING

    def test_cpu_compute_reasoning_has_no_gpu_vocabulary(self):
        playbook = _compute_bound_cpp().lower()
        # Reasoning paragraph is the line right after the roofline header
        reasoning = playbook.split("### roofline analysis")[1].split("**targeted actions")[0]
        for gpu_word in ("sm occupancy", "warp divergence", "tensor core"):
            assert gpu_word not in reasoning, f"GPU wording leaked into cpp reasoning: {gpu_word}"
        assert "accumulator" in reasoning
        assert "vectorized" in reasoning

    def test_cpu_bandwidth_reasoning_has_no_gpu_vocabulary(self):
        playbook = _memory_bound_cpp().lower()
        reasoning = playbook.split("### roofline analysis")[1].split("**targeted actions")[0]
        for gpu_word in ("uncoalesced", "kernel launches", "cpu-gpu synchronization"):
            assert gpu_word not in reasoning, f"GPU wording leaked into cpp reasoning: {gpu_word}"
        assert "peak dram bandwidth" in reasoning
        assert "64-byte line" in reasoning or "cache" in reasoning

    def test_cuda_reasoning_still_gpu_flavored(self):
        """Regression guard: the GPU tables are untouched."""
        ctx = _cpp_ctx(ai=60.0, achieved_tflops=0.04, achieved_bw=8.0, program_type="cuda")
        playbook = _build_optimization_playbook(ctx, primary_bottleneck=None).lower()
        assert "sm occupancy" in playbook

    @pytest.mark.parametrize("tiers", [_BW_REASONING_CPU, _COMPUTE_REASONING_CPU])
    def test_cpu_tier_tables_are_well_formed(self, tiers):
        assert [t for t, _ in tiers] == [30.0, 70.0, 100.1]
        key = "bw_pct" if tiers is _BW_REASONING_CPU else "compute_pct"
        for _, template in tiers:
            assert "{" + key in template  # formats without KeyError
            template.format(**{key: 42.0})


class TestPerfVsPeakCppGuidance:
    def test_compute_bound_cpp_priority_is_fma_not_tensor_cores(self):
        parts: list[str] = []
        _add_perf_vs_peak(parts, _cpp_ctx(ai=60.0, achieved_tflops=0.04, achieved_bw=8.0))
        text = "\n".join(parts).lower()
        assert "compute-bound" in text
        assert "independent" in text and "accumulator" in text
        assert "register-block" in text
        assert "tensor core" not in text

    def test_compute_bound_cuda_still_says_tensor_cores(self):
        parts: list[str] = []
        _add_perf_vs_peak(
            parts, _cpp_ctx(ai=60.0, achieved_tflops=0.04, achieved_bw=8.0, program_type="cuda")
        )
        text = "\n".join(parts).lower()
        assert "tensor cores" in text

    def test_memory_bound_cpp_unchanged_branch(self):
        parts: list[str] = []
        _add_perf_vs_peak(parts, _cpp_ctx(ai=0.5, achieved_tflops=0.02, achieved_bw=25.0))
        text = "\n".join(parts).lower()
        assert "memory-bound" in text
        assert "reduce bytes moved" in text
