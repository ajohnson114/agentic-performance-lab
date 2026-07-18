"""Tests for perflab.analyzers.profile_diff."""
from __future__ import annotations

from perflab.analyzers.profile_diff import (
    HotspotShift,
    ProfileDelta,
    compute_hotspot_diff,
    compute_profile_diff,
    format_hotspot_diff,
    format_profile_diff,
)


class TestComputeProfileDiff:
    def test_ipc_improvement(self):
        prev = {"linux_perf": {"ipc": 0.82, "cache_miss_rate": 0.12}}
        curr = {"linux_perf": {"ipc": 1.41, "cache_miss_rate": 0.03}}
        deltas = compute_profile_diff(prev, curr, "maximize")
        ipc_delta = next(d for d in deltas if d.metric == "linux_perf.ipc")
        assert ipc_delta.direction == "improved"
        assert ipc_delta.delta > 0

    def test_cache_miss_improvement(self):
        prev = {"linux_perf": {"ipc": 0.82, "cache_miss_rate": 0.12}}
        curr = {"linux_perf": {"ipc": 1.41, "cache_miss_rate": 0.03}}
        deltas = compute_profile_diff(prev, curr, "maximize")
        cache_delta = next(d for d in deltas if d.metric == "linux_perf.cache_miss_rate")
        assert cache_delta.direction == "improved"
        assert cache_delta.delta < 0

    def test_regression_detected(self):
        prev = {"linux_perf": {"ipc": 1.5}}
        curr = {"linux_perf": {"ipc": 0.8}}
        deltas = compute_profile_diff(prev, curr, "maximize")
        ipc_delta = next(d for d in deltas if d.metric == "linux_perf.ipc")
        assert ipc_delta.direction == "regressed"

    def test_unchanged_metric(self):
        prev = {"linux_perf": {"ipc": 1.0}}
        curr = {"linux_perf": {"ipc": 1.005}}
        deltas = compute_profile_diff(prev, curr, "maximize")
        ipc_delta = next(d for d in deltas if d.metric == "linux_perf.ipc")
        assert ipc_delta.direction == "unchanged"

    def test_nsys_metrics(self):
        prev = {"nsys": {"gpu_active_pct": 45.0, "avg_kernel_gap_us": 100.0}}
        curr = {"nsys": {"gpu_active_pct": 72.0, "avg_kernel_gap_us": 30.0}}
        deltas = compute_profile_diff(prev, curr, "maximize")
        gpu_delta = next(d for d in deltas if d.metric == "nsys.gpu_active_pct")
        assert gpu_delta.direction == "improved"
        gap_delta = next(d for d in deltas if d.metric == "nsys.avg_kernel_gap_us")
        assert gap_delta.direction == "improved"

    def test_kernel_time_reduction_is_improvement(self):
        prev = {"nsys": {"cuda_kernel_time_ms": 120.0}}
        curr = {"nsys": {"cuda_kernel_time_ms": 60.0}}
        deltas = compute_profile_diff(prev, curr, "maximize")
        kt_delta = next(d for d in deltas if d.metric == "nsys.cuda_kernel_time_ms")
        assert kt_delta.direction == "improved"

    def test_flop_count_reduction_is_improvement(self):
        prev = {"torch_profiler": {"total_tflops": 4.0}}
        curr = {"torch_profiler": {"total_tflops": 2.0}}
        deltas = compute_profile_diff(prev, curr, "maximize")
        fl_delta = next(d for d in deltas if d.metric == "torch_profiler.total_tflops")
        assert fl_delta.direction == "improved"


class TestSignificance:
    def test_high_significance(self):
        prev = {"linux_perf": {"ipc": 0.5}}
        curr = {"linux_perf": {"ipc": 1.5}}  # 200% increase
        deltas = compute_profile_diff(prev, curr, "maximize")
        assert deltas[0].significance == "high"

    def test_medium_significance(self):
        prev = {"linux_perf": {"ipc": 1.0}}
        curr = {"linux_perf": {"ipc": 1.1}}  # 10% increase
        deltas = compute_profile_diff(prev, curr, "maximize")
        assert deltas[0].significance == "medium"

    def test_low_significance(self):
        prev = {"linux_perf": {"ipc": 1.0}}
        curr = {"linux_perf": {"ipc": 1.03}}  # 3% increase
        deltas = compute_profile_diff(prev, curr, "maximize")
        assert deltas[0].significance == "low"


class TestMissingKeys:
    def test_different_profilers(self):
        prev = {"linux_perf": {"ipc": 1.0}}
        curr = {"nsys": {"gpu_active_pct": 50.0}}
        deltas = compute_profile_diff(prev, curr, "maximize")
        assert deltas == []

    def test_partial_overlap(self):
        prev = {"linux_perf": {"ipc": 1.0, "cache_miss_rate": 0.05}}
        curr = {"linux_perf": {"ipc": 1.2}}  # missing cache_miss_rate
        deltas = compute_profile_diff(prev, curr, "maximize")
        assert len(deltas) == 1
        assert deltas[0].metric == "linux_perf.ipc"

    def test_empty_summaries(self):
        assert compute_profile_diff({}, {}, "maximize") == []


class TestFormatProfileDiff:
    def test_format_output(self):
        deltas = [
            ProfileDelta("ipc", 0.82, 1.41, 0.59, 72.0, "improved", "high"),
            ProfileDelta("cache_miss_rate", 0.12, 0.03, -0.09, -75.0, "improved", "high"),
        ]
        text = format_profile_diff(deltas)
        assert "ipc" in text
        assert "cache_miss_rate" in text
        assert "improved" in text

    def test_empty_deltas(self):
        assert format_profile_diff([]) == ""


class TestComputeHotspotDiff:
    def test_function_removed(self):
        prev = {"pyspy": {"hotspots": [{"function": "naive_matmul", "pct": 85.0}]}}
        curr = {"pyspy": {"hotspots": [{"function": "numpy_dot", "pct": 90.0}]}}
        shifts = compute_hotspot_diff(prev, curr)
        removed = [s for s in shifts if s.status == "removed"]
        assert len(removed) == 1
        assert removed[0].function == "naive_matmul"

    def test_function_new(self):
        prev = {"pyspy": {"hotspots": [{"function": "naive_matmul", "pct": 85.0}]}}
        curr = {"pyspy": {"hotspots": [{"function": "numpy_dot", "pct": 90.0}]}}
        shifts = compute_hotspot_diff(prev, curr)
        new = [s for s in shifts if s.status == "new"]
        assert len(new) == 1
        assert new[0].function == "numpy_dot"

    def test_function_decreased(self):
        prev = {"pyspy": {"hotspots": [{"function": "compute", "pct": 60.0}]}}
        curr = {"pyspy": {"hotspots": [{"function": "compute", "pct": 20.0}]}}
        shifts = compute_hotspot_diff(prev, curr)
        assert len(shifts) == 1
        assert shifts[0].status == "decreased"
        assert shifts[0].delta_pct == -40.0

    def test_function_increased(self):
        prev = {"pyspy": {"hotspots": [{"function": "io_wait", "pct": 5.0}]}}
        curr = {"pyspy": {"hotspots": [{"function": "io_wait", "pct": 35.0}]}}
        shifts = compute_hotspot_diff(prev, curr)
        assert shifts[0].status == "increased"

    def test_unchanged_filtered_out(self):
        prev = {"pyspy": {"hotspots": [{"function": "f", "pct": 50.0}]}}
        curr = {"pyspy": {"hotspots": [{"function": "f", "pct": 50.5}]}}
        shifts = compute_hotspot_diff(prev, curr)
        assert len(shifts) == 0

    def test_merges_pyspy_and_perf(self):
        prev = {
            "pyspy": {"hotspots": [{"function": "hot_py", "pct": 40.0}]},
            "linux_perf": {"hotspots": [{"function": "hot_c", "pct": 30.0}]},
        }
        curr = {
            "pyspy": {"hotspots": []},
            "linux_perf": {"hotspots": [{"function": "hot_c", "pct": 10.0}]},
        }
        shifts = compute_hotspot_diff(prev, curr)
        funcs = {s.function for s in shifts}
        assert "hot_py" in funcs  # removed
        assert "hot_c" in funcs   # decreased

    def test_empty_summaries(self):
        assert compute_hotspot_diff({}, {}) == []

    def test_top_n_limit(self):
        prev = {"pyspy": {"hotspots": [{"function": f"f{i}", "pct": float(50 - i)} for i in range(20)]}}
        curr = {"pyspy": {"hotspots": []}}
        shifts = compute_hotspot_diff(prev, curr, top_n=5)
        assert len(shifts) == 5


class TestFormatHotspotDiff:
    def test_format_output(self):
        shifts = [
            HotspotShift("matmul", 85.0, 0.0, -85.0, "removed"),
            HotspotShift("numpy_dot", 0.0, 90.0, 90.0, "new"),
        ]
        text = format_hotspot_diff(shifts)
        assert "matmul" in text
        assert "GONE" in text
        assert "numpy_dot" in text
        assert "NEW" in text

    def test_empty_shifts(self):
        assert format_hotspot_diff([]) == ""

    def test_delta_format(self):
        shifts = [HotspotShift("f", 60.0, 20.0, -40.0, "decreased")]
        text = format_hotspot_diff(shifts)
        assert "60.0%" in text
        assert "20.0%" in text
        assert "-40.0pp" in text


# ---------------------------------------------------------------------------
# New metric tracking tests (NCU, TMA, torch, JAX, dotted paths)
# ---------------------------------------------------------------------------

class TestNcuMetricDiff:
    def test_tc_util_improvement(self):
        prev = {"ncu": {"tensor_core_utilization_pct": 0.0, "sm_utilization_pct": 30.0}}
        curr = {"ncu": {"tensor_core_utilization_pct": 60.0, "sm_utilization_pct": 75.0}}
        deltas = compute_profile_diff(prev, curr)
        tc = [d for d in deltas if d.metric == "ncu.tensor_core_utilization_pct"]
        assert len(tc) == 1
        assert tc[0].after == 60.0
        assert tc[0].direction == "improved"

    def test_l1_hit_rate_improvement(self):
        prev = {"ncu": {"l1_hit_rate": 30.0, "l2_hit_rate": 40.0}}
        curr = {"ncu": {"l1_hit_rate": 85.0, "l2_hit_rate": 75.0}}
        deltas = compute_profile_diff(prev, curr)
        l1 = [d for d in deltas if d.metric == "ncu.l1_hit_rate"]
        assert len(l1) == 1
        assert l1[0].direction == "improved"

    def test_bank_conflicts_decrease(self):
        prev = {"ncu": {"bank_conflicts": 500.0}}
        curr = {"ncu": {"bank_conflicts": 10.0}}
        deltas = compute_profile_diff(prev, curr)
        bc = [d for d in deltas if d.metric == "ncu.bank_conflicts"]
        assert len(bc) == 1
        assert bc[0].direction == "improved"

    def test_stall_pct_decrease(self):
        prev = {"ncu": {"dominant_stall_pct": 45.0}}
        curr = {"ncu": {"dominant_stall_pct": 15.0}}
        deltas = compute_profile_diff(prev, curr)
        s = [d for d in deltas if d.metric == "ncu.dominant_stall_pct"]
        assert len(s) == 1
        assert s[0].direction == "improved"


class TestTmaMetricDiff:
    def test_tma_level1_diff(self):
        prev = {"linux_perf": {"tma": {"backend_bound_pct": 55.0, "retiring_pct": 20.0}}}
        curr = {"linux_perf": {"tma": {"backend_bound_pct": 30.0, "retiring_pct": 50.0}}}
        deltas = compute_profile_diff(prev, curr)
        backend = [d for d in deltas if d.metric == "linux_perf.tma.backend_bound_pct"]
        retiring = [d for d in deltas if d.metric == "linux_perf.tma.retiring_pct"]
        assert len(backend) == 1
        assert backend[0].before == 55.0
        assert backend[0].after == 30.0
        assert len(retiring) == 1
        assert retiring[0].after == 50.0

    def test_tma_level2_diff(self):
        prev = {"linux_perf": {"tma_level2": {"memory_bound_pct": 40.0, "l2_bound_pct": 25.0}}}
        curr = {"linux_perf": {"tma_level2": {"memory_bound_pct": 15.0, "l2_bound_pct": 5.0}}}
        deltas = compute_profile_diff(prev, curr)
        mem = [d for d in deltas if d.metric == "linux_perf.tma_level2.memory_bound_pct"]
        l2 = [d for d in deltas if d.metric == "linux_perf.tma_level2.l2_bound_pct"]
        assert len(mem) == 1
        assert mem[0].before == 40.0
        assert mem[0].after == 15.0
        assert len(l2) == 1


class TestTorchProfilerDiff:
    def test_sync_count_diff(self):
        prev = {"torch_profiler": {"sync_count": 20, "total_sync_time_us": 5000.0}}
        curr = {"torch_profiler": {"sync_count": 3, "total_sync_time_us": 500.0}}
        deltas = compute_profile_diff(prev, curr)
        sync = [d for d in deltas if d.metric == "torch_profiler.sync_count"]
        assert len(sync) == 1
        assert sync[0].before == 20
        assert sync[0].after == 3

    def test_dotted_path_memory(self):
        prev = {"torch_profiler": {"memory": {"peak_memory_mb": 4096.0, "total_allocations": 1000}}}
        curr = {"torch_profiler": {"memory": {"peak_memory_mb": 2048.0, "total_allocations": 200}}}
        deltas = compute_profile_diff(prev, curr)
        peak = [d for d in deltas if d.metric == "torch_profiler.memory.peak_memory_mb"]
        alloc = [d for d in deltas if d.metric == "torch_profiler.memory.total_allocations"]
        assert len(peak) == 1
        assert peak[0].before == 4096.0
        assert peak[0].after == 2048.0
        assert len(alloc) == 1

    def test_dotted_path_gpu_ratio(self):
        prev = {"torch_profiler": {"cpu_vs_gpu": {"ratio": 0.3}}}
        curr = {"torch_profiler": {"cpu_vs_gpu": {"ratio": 1.5}}}
        deltas = compute_profile_diff(prev, curr)
        ratio = [d for d in deltas if d.metric == "torch_profiler.cpu_vs_gpu.ratio"]
        assert len(ratio) == 1
        assert ratio[0].after == 1.5

    def test_flops_diff(self):
        # total_tflops is an op count for a fixed workload, not a rate:
        # executing 4x the FLOPs for the same task is extra work, not a win
        prev = {"torch_profiler": {"total_tflops": 0.5}}
        curr = {"torch_profiler": {"total_tflops": 2.0}}
        deltas = compute_profile_diff(prev, curr)
        flops = [d for d in deltas if d.metric == "torch_profiler.total_tflops"]
        assert len(flops) == 1
        assert flops[0].direction == "regressed"


class TestJaxMetricDiff:
    def test_mxu_improvement(self):
        prev = {"jax": {"mxu_utilization_pct": 30.0, "device_fraction": 0.5}}
        curr = {"jax": {"mxu_utilization_pct": 85.0, "device_fraction": 0.9}}
        deltas = compute_profile_diff(prev, curr)
        mxu = [d for d in deltas if d.metric == "jax.mxu_utilization_pct"]
        assert len(mxu) == 1
        assert mxu[0].direction == "improved"

    def test_recompilation_decrease(self):
        prev = {"jax": {"xla_recompilations": 5}}
        curr = {"jax": {"xla_recompilations": 0}}
        deltas = compute_profile_diff(prev, curr)
        recomp = [d for d in deltas if d.metric == "jax.xla_recompilations"]
        assert len(recomp) == 1
        assert recomp[0].direction == "improved"


class TestDottedPathResolution:
    def test_simple_key(self):
        from perflab.analyzers.profile_diff import _resolve_dotted
        assert _resolve_dotted({"a": 1}, "a") == 1

    def test_nested_key(self):
        from perflab.analyzers.profile_diff import _resolve_dotted
        assert _resolve_dotted({"a": {"b": 42}}, "a.b") == 42

    def test_deep_nested(self):
        from perflab.analyzers.profile_diff import _resolve_dotted
        assert _resolve_dotted({"a": {"b": {"c": 99}}}, "a.b.c") == 99

    def test_missing_key(self):
        from perflab.analyzers.profile_diff import _resolve_dotted
        assert _resolve_dotted({"a": 1}, "b") is None

    def test_missing_nested(self):
        from perflab.analyzers.profile_diff import _resolve_dotted
        assert _resolve_dotted({"a": {"b": 1}}, "a.c") is None

    def test_non_dict_intermediate(self):
        from perflab.analyzers.profile_diff import _resolve_dotted
        assert _resolve_dotted({"a": 42}, "a.b") is None
