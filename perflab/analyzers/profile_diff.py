"""Differential profiling between agent iterations.

Compares profiler summaries from consecutive iterations to surface what
changed — improved, regressed, or unchanged — so the LLM prompt includes
clear before/after context.  Includes hotspot-level diffing for py-spy
and perf to show which functions gained or lost CPU share.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProfileDelta:
    """A delta between two profiling measurements."""
    metric: str         # "ipc", "cache_miss_rate", "gpu_active_pct", etc.
    before: float
    after: float
    delta: float        # after - before
    delta_pct: float    # (after - before) / before * 100
    direction: str      # "improved", "regressed", "unchanged"
    significance: str   # "high", "medium", "low"


@dataclass
class HotspotShift:
    """A change in a function's CPU share between two profiles."""
    function: str
    before_pct: float
    after_pct: float
    delta_pct: float    # after - before
    status: str         # "new", "removed", "increased", "decreased", "unchanged"


# Metrics where higher is better (maximize)
_HIGHER_IS_BETTER = {
    # linux_perf
    "ipc", "cpus_utilized",
    "tma.retiring_pct",
    # nsys
    "gpu_active_pct", "cuda_kernel_time_ms",
    # ncu
    "sm_utilization_pct", "achieved_occupancy_pct", "achieved_bw_gbs",
    "memory_throughput_pct", "compute_throughput_pct",
    "tensor_core_utilization_pct", "branch_efficiency_pct",
    "l1_hit_rate", "l2_hit_rate",
    # torch_profiler
    "total_flops", "total_tflops", "cpu_vs_gpu.ratio",
    # jax
    "mxu_utilization_pct", "device_fraction", "hlo_cost_tflops",
}

# Metrics where lower is better (minimize)
_LOWER_IS_BETTER = {
    # linux_perf
    "cache_miss_rate", "branch_miss_rate", "task_clock_ms",
    "l1_dcache_misses", "llc_misses",
    "tma.frontend_bound_pct", "tma.backend_bound_pct", "tma.bad_speculation_pct",
    "tma_level2.memory_bound_pct", "tma_level2.core_bound_pct",
    "tma_level2.l1_bound_pct", "tma_level2.l2_bound_pct",
    "tma_level2.l3_bound_pct", "tma_level2.dram_bound_pct",
    # nsys
    "avg_kernel_gap_us", "api_overhead_ms", "memcpy_time_ms", "total_sync_ms",
    # ncu
    "bank_conflicts", "sectors_per_request", "dominant_stall_pct",
    # torch_profiler
    "sync_count", "total_sync_time_us",
    "memory.total_allocations", "memory.peak_memory_mb",
    # memray
    "peak_memory_mb", "total_allocated_mb", "total_allocations",
    # jax
    "xla_compilations", "xla_compilation_time_ms", "xla_recompilations",
    "hlo_module_count", "infeed_stall_pct",
}

# Profiler → tracked metrics
_TRACKED_METRICS: dict[str, list[str]] = {
    "linux_perf": [
        "ipc", "cache_miss_rate", "branch_miss_rate", "cpus_utilized",
        "task_clock_ms", "l1_dcache_misses", "llc_misses",
        "tma.frontend_bound_pct", "tma.backend_bound_pct",
        "tma.bad_speculation_pct", "tma.retiring_pct",
        "tma_level2.memory_bound_pct", "tma_level2.core_bound_pct",
        "tma_level2.l1_bound_pct", "tma_level2.l2_bound_pct",
        "tma_level2.l3_bound_pct", "tma_level2.dram_bound_pct",
    ],
    "nsys": ["gpu_active_pct", "cuda_kernel_time_ms", "avg_kernel_gap_us", "api_overhead_ms", "memcpy_time_ms", "total_sync_ms"],
    "ncu": [
        "sm_utilization_pct", "achieved_occupancy_pct", "achieved_bw_gbs",
        "memory_throughput_pct", "compute_throughput_pct",
        "tensor_core_utilization_pct", "branch_efficiency_pct",
        "bank_conflicts", "sectors_per_request",
        "l1_hit_rate", "l2_hit_rate",
        "dominant_stall_pct",
    ],
    "torch_profiler": [
        "sync_count", "total_sync_time_us",
        "total_flops", "total_tflops",
        "memory.total_allocations", "memory.peak_memory_mb",
        "cpu_vs_gpu.ratio",
    ],
    "memray": ["peak_memory_mb", "total_allocated_mb", "total_allocations"],
    "jax": [
        "xla_compilations", "xla_compilation_time_ms", "xla_recompilations",
        "hlo_module_count", "hlo_cost_tflops",
        "mxu_utilization_pct", "infeed_stall_pct", "device_fraction",
    ],
}


def _resolve_dotted(d: dict, path: str):
    """Resolve a dotted path like 'memory.peak_memory_mb' into nested dicts."""
    parts = path.split(".")
    val = d
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return None
    return val


def compute_profile_diff(
    prev_summaries: dict[str, dict],
    curr_summaries: dict[str, dict],
    metric_mode: str = "maximize",
) -> list[ProfileDelta]:
    """Compare profiler summaries from two iterations.

    Returns a list of ProfileDelta for each matched metric that changed.
    metric_mode is the task's benchmark metric mode, used as fallback
    direction for unknown metrics.
    """
    deltas: list[ProfileDelta] = []

    for profiler_name, metrics in _TRACKED_METRICS.items():
        prev = prev_summaries.get(profiler_name, {})
        curr = curr_summaries.get(profiler_name, {})

        if not prev or not curr:
            continue

        for metric in metrics:
            prev_val = _resolve_dotted(prev, metric)
            curr_val = _resolve_dotted(curr, metric)

            if prev_val is None or curr_val is None:
                continue

            try:
                prev_f = float(prev_val)
                curr_f = float(curr_val)
            except (TypeError, ValueError):
                continue

            if prev_f == 0:
                # Avoid division by zero
                if curr_f == 0:
                    continue
                delta_pct = 100.0  # went from 0 to something
            else:
                delta_pct = (curr_f - prev_f) / abs(prev_f) * 100.0

            delta = curr_f - prev_f

            # Determine direction
            direction = _classify_direction(metric, delta, metric_mode)

            # Determine significance
            abs_pct = abs(delta_pct)
            if abs_pct > 20:
                significance = "high"
            elif abs_pct > 5:
                significance = "medium"
            else:
                significance = "low"

            # Skip unchanged
            if abs_pct < 1.0:
                direction = "unchanged"

            deltas.append(ProfileDelta(
                metric=f"{profiler_name}.{metric}",
                before=prev_f,
                after=curr_f,
                delta=delta,
                delta_pct=delta_pct,
                direction=direction,
                significance=significance,
            ))

    return deltas


def _classify_direction(metric: str, delta: float, metric_mode: str) -> str:
    """Classify whether a delta is an improvement or regression."""
    if abs(delta) < 1e-10:
        return "unchanged"

    # Strip profiler prefix for lookup (e.g., "ncu.sm_utilization_pct" → "sm_utilization_pct")
    # But keep dotted metric paths (e.g., "linux_perf.tma.backend_bound_pct" → "tma.backend_bound_pct")
    _PROFILER_PREFIXES = ("linux_perf.", "nsys.", "ncu.", "torch_profiler.", "memray.", "jax.")
    bare = metric
    for prefix in _PROFILER_PREFIXES:
        if metric.startswith(prefix):
            bare = metric[len(prefix):]
            break

    if metric in _HIGHER_IS_BETTER or bare in _HIGHER_IS_BETTER:
        return "improved" if delta > 0 else "regressed"
    if metric in _LOWER_IS_BETTER or bare in _LOWER_IS_BETTER:
        return "improved" if delta < 0 else "regressed"

    # Fallback: use task metric mode
    if metric_mode == "maximize":
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def format_profile_diff(deltas: list[ProfileDelta]) -> str:
    """Render profile diff as a compact string for the prompt."""
    if not deltas:
        return ""

    lines = ["Profile changes from previous iteration:"]
    for d in deltas:
        arrow = "\u2191" if d.delta > 0 else "\u2193" if d.delta < 0 else "\u2194"
        sign = "+" if d.delta_pct >= 0 else ""
        lines.append(
            f"  {d.metric}: {_fmt_val(d.metric, d.before)} \u2192 {_fmt_val(d.metric, d.after)} "
            f"({sign}{d.delta_pct:.0f}% {arrow} {d.direction})"
        )

    return "\n".join(lines)


def _fmt_val(metric: str, value: float) -> str:
    """Format a metric value for display."""
    if "rate" in metric or "pct" in metric:
        if value < 1:
            return f"{value * 100:.1f}%"
        return f"{value:.1f}%"
    if "ms" in metric:
        return f"{value:.1f}ms"
    if "us" in metric:
        return f"{value:.1f}us"
    if "gbs" in metric:
        return f"{value:.1f}GB/s"
    return f"{value:.2f}"


def compute_hotspot_diff(
    prev_summaries: dict[str, dict],
    curr_summaries: dict[str, dict],
    top_n: int = 8,
) -> list[HotspotShift]:
    """Compare CPU hotspot functions between two profiles.

    Merges py-spy and perf hotspots to show which functions gained or
    lost CPU share.
    """
    prev_hotspots = _extract_hotspots(prev_summaries)
    curr_hotspots = _extract_hotspots(curr_summaries)

    if not prev_hotspots and not curr_hotspots:
        return []

    all_funcs = set(prev_hotspots.keys()) | set(curr_hotspots.keys())

    shifts: list[HotspotShift] = []
    for func in all_funcs:
        before = prev_hotspots.get(func, 0.0)
        after = curr_hotspots.get(func, 0.0)
        delta = after - before

        if before == 0 and after > 0:
            status = "new"
        elif after == 0 and before > 0:
            status = "removed"
        elif abs(delta) < 1.0:
            status = "unchanged"
        elif delta > 0:
            status = "increased"
        else:
            status = "decreased"

        if status == "unchanged":
            continue

        shifts.append(HotspotShift(
            function=func,
            before_pct=before,
            after_pct=after,
            delta_pct=delta,
            status=status,
        ))

    # Sort by absolute delta descending
    shifts.sort(key=lambda s: abs(s.delta_pct), reverse=True)
    return shifts[:top_n]


def _extract_hotspots(summaries: dict[str, dict]) -> dict[str, float]:
    """Extract function → CPU% map from pyspy and perf summaries."""
    funcs: dict[str, float] = {}

    # py-spy hotspots
    pyspy = summaries.get("pyspy", {})
    for h in pyspy.get("hotspots", []):
        func = h.get("function", "")
        if func:
            funcs[func] = max(funcs.get(func, 0), h.get("pct", 0))

    # perf hotspots
    perf = summaries.get("linux_perf", {})
    for h in perf.get("hotspots", []):
        func = h.get("function", "")
        if func:
            funcs[func] = max(funcs.get(func, 0), h.get("pct", 0))

    return funcs


def format_hotspot_diff(shifts: list[HotspotShift]) -> str:
    """Render hotspot shifts as a compact string for the LLM prompt."""
    if not shifts:
        return ""

    lines = ["CPU hotspot changes (baseline vs optimized):"]
    for s in shifts:
        if s.status == "new":
            lines.append(f"  {s.function}: NEW at {s.after_pct:.1f}%")
        elif s.status == "removed":
            lines.append(f"  {s.function}: GONE (was {s.before_pct:.1f}%)")
        else:
            sign = "+" if s.delta_pct > 0 else ""
            arrow = "\u2191" if s.delta_pct > 0 else "\u2193"
            lines.append(
                f"  {s.function}: {s.before_pct:.1f}% \u2192 {s.after_pct:.1f}% "
                f"({sign}{s.delta_pct:.1f}pp {arrow})"
            )

    return "\n".join(lines)
