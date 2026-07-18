"""Profiler insights card: has-data predicates, dispatcher, and CPU/framework sections."""
from __future__ import annotations

from .accelerator_sections import (
    _render_jax_section,
    _render_metal_section,
    _render_ncu_section,
    _render_nsys_section,
    _render_tpu_section,
)
from .data import ProfilerData
from .widgets import (
    _metric_pill,
    _render_bar_chart,
    _render_speedscope_link,
    _summary_ok,
)

# ---------------------------------------------------------------------------
# Has-data checks
# ---------------------------------------------------------------------------

def _has_torch_data(prof: ProfilerData) -> bool:
    for s in (prof.torch_summary, prof.baseline_torch_summary):
        if s and s.get("returncode") == 0:
            return True
    return False


def _has_pyspy_data(prof: ProfilerData) -> bool:
    for s in (prof.pyspy_summary, prof.baseline_pyspy_summary):
        if s and s.get("returncode") == 0 and s.get("hotspots"):
            return True
    return False


def _has_metal_data(prof: ProfilerData) -> bool:
    for s in (prof.metal_summary, prof.baseline_metal_summary):
        if s and s.get("returncode", 0) == 0:
            return True
    return False


def _has_nsys_data(prof: ProfilerData) -> bool:
    for s in (prof.nsys_summary, prof.baseline_nsys_summary):
        if s and s.get("returncode", 0) == 0:
            return True
    return False


def _has_ncu_data(prof: ProfilerData) -> bool:
    for s in (prof.ncu_summary, prof.baseline_ncu_summary):
        if s and s.get("returncode", 0) == 0:
            return True
    return False


def _has_jax_data(prof: ProfilerData) -> bool:
    for s in (prof.jax_summary, prof.baseline_jax_summary):
        if s and s.get("returncode", 0) == 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _has_memray_data(prof: ProfilerData) -> bool:
    s = prof.memray_summary
    return s is not None and s.get("returncode", 0) == 0


def _render_profiler(parts: list[str], prof: ProfilerData, esc) -> None:
    """Render profiler data as embedded visualizations."""
    any_data = (
        _has_torch_data(prof) or _has_pyspy_data(prof) or
        _has_metal_data(prof) or _has_nsys_data(prof) or _has_ncu_data(prof) or
        _has_jax_data(prof) or _has_memray_data(prof)
    )
    if not any_data:
        return

    parts.append('<div class="card profiler-section">')
    parts.append('<h2>Profiler insights</h2>')

    if _has_torch_data(prof):
        _render_torch_section(parts, prof, esc)
    if _has_pyspy_data(prof):
        _render_pyspy_section(parts, prof, esc)
    if _has_metal_data(prof):
        _render_metal_section(parts, prof, esc)
    if _has_nsys_data(prof):
        _render_nsys_section(parts, prof, esc)
    if _has_ncu_data(prof):
        _render_ncu_section(parts, prof, esc)
    if _has_jax_data(prof):
        _render_jax_section(parts, prof, esc)
        _render_tpu_section(parts, prof, esc)
    if _has_memray_data(prof):
        _render_memray_section(parts, prof, esc)

    parts.append('</div>')  # .card


# ---------------------------------------------------------------------------
# Torch section
# ---------------------------------------------------------------------------

def _filter_torch_ops(summary: dict | None) -> list[dict]:
    """Extract filtered top ops from a torch summary."""
    if not summary:
        return []
    top_ops = summary.get("top_ops", [])
    return [
        {
            "name": op["name"],
            "pct": op.get("pct", 0),
            "total_ms": op.get("total_us", 0) / 1000,
            "count": op.get("count", 0),
        }
        for op in top_ops
        if not op["name"].startswith("PyTorch Profiler")
        and not op["name"].startswith("bench.py")
        and op.get("pct", 0) > 0.5
    ][:10]


def _render_torch_section(parts: list[str], prof: ProfilerData, esc) -> None:
    ts = prof.torch_summary
    bts = prof.baseline_torch_summary
    has_opt = _summary_ok(ts)
    has_base = _summary_ok(bts)
    # Reads below are guarded by has_opt; rebind non-Optional for the type checker.
    ts = ts or {}

    parts.append('<h3>Torch profiler</h3>')

    # --- Top ops bar chart (optimized) ---
    if has_opt:
        ops = _filter_torch_ops(ts)
        if ops:
            parts.append('<h4>Top operations</h4>')
            _render_bar_chart(parts, ops, esc, color="#3b82f6")

        # CPU vs GPU breakdown
        cpu_gpu = ts.get("cpu_vs_gpu", {})
        cpu_us = cpu_gpu.get("total_cpu_op_us", 0)
        gpu_us = cpu_gpu.get("total_gpu_kernel_us", 0)
        total_us = cpu_us + gpu_us
        if total_us > 0:
            cpu_pct = cpu_us / total_us * 100
            gpu_pct = gpu_us / total_us * 100
            parts.append('<h4>CPU vs GPU time</h4>')
            parts.append('<div class="split-bar">')
            if cpu_pct > 0:
                parts.append(f'<div class="cpu" style="width: {cpu_pct:.1f}%"></div>')
            if gpu_pct > 0:
                parts.append(f'<div class="gpu" style="width: {gpu_pct:.1f}%"></div>')
            parts.append('</div>')
            parts.append(
                f'<div class="split-legend">'
                f'<span><span class="dot" style="background:#60a5fa"></span>CPU: {cpu_us/1000:.1f}ms ({cpu_pct:.0f}%)</span>'
                f'<span><span class="dot" style="background:#34d399"></span>GPU: {gpu_us/1000:.1f}ms ({gpu_pct:.0f}%)</span>'
                f'</div>'
            )
        elif cpu_us > 0:
            parts.append(f'<p style="font-size:0.9em; color:#666">CPU only: {cpu_us/1000:.1f}ms total (no GPU kernels recorded)</p>')

        # Memory stats
        mem = ts.get("memory", {})
        allocs = mem.get("total_allocations", 0)
        alloc_time = mem.get("total_allocation_time_us", 0)
        if allocs > 0:
            parts.append(
                f'<p style="font-size:0.85em; color:#666; margin-top:8px">'
                f'Memory: {allocs:,} allocations ({alloc_time/1000:.1f}ms)</p>'
            )

    # --- Comparison dropdown ---
    if has_opt and has_base:
        parts.append('<details><summary>Compare with baseline (torch)</summary>')
        parts.append('<div class="compare-row">')
        parts.append('<div class="compare-col baseline"><h4>Baseline</h4>')
        baseline_ops = _filter_torch_ops(bts)
        if baseline_ops:
            _render_bar_chart(parts, baseline_ops, esc, color="#9ca3af")
        else:
            parts.append('<p style="color:#888">No ops data</p>')
        parts.append('</div>')
        parts.append('<div class="compare-col optimized"><h4>Optimized</h4>')
        opt_ops = _filter_torch_ops(ts)
        if opt_ops:
            _render_bar_chart(parts, opt_ops, esc, color="#3b82f6")
        else:
            parts.append('<p style="color:#888">No ops data</p>')
        parts.append('</div>')
        parts.append('</div>')
        parts.append('</details>')

    # --- Torch trace link ---
    if prof.torch_trace_path and prof.torch_trace_path.exists():
        trace_size_mb = prof.torch_trace_path.stat().st_size / 1024 / 1024
        parts.append(
            f'<p style="font-size:0.85em; margin-top:12px; color:#666">'
            f'Torch trace ({trace_size_mb:.1f}MB): open '
            f'<a href="https://ui.perfetto.dev/" target="_blank">Perfetto UI</a> '
            f'and load <code>{esc(str(prof.torch_trace_path))}</code></p>'
        )


# ---------------------------------------------------------------------------
# Py-spy section
# ---------------------------------------------------------------------------

def _render_pyspy_hotspots(parts: list[str], summary: dict | None, esc, color: str = "#f59e0b") -> None:
    """Render py-spy hotspot bars from a summary."""
    if not summary:
        return
    hotspots = summary.get("hotspots", [])[:8]
    if not hotspots:
        return
    items = []
    for h in hotspots:
        func = h.get("function", "?")
        loc = h.get("location", "")
        label = f"{func}" + (f"  ({loc})" if loc else "")
        if len(label) > 50:
            label = label[:47] + "..."
        items.append({"name": label, "pct": h.get("pct", 0)})
    _render_bar_chart(parts, items, esc, color=color)


def _render_pyspy_section(parts: list[str], prof: ProfilerData, esc) -> None:
    ps = prof.pyspy_summary
    bps = prof.baseline_pyspy_summary
    has_opt = ps and ps.get("returncode") == 0 and ps.get("hotspots")
    has_base = bps and bps.get("returncode") == 0 and bps.get("hotspots")
    scope_path = prof.speedscope_json_path
    if not (scope_path and scope_path.exists()):
        scope_path = None
    base_scope_path = prof.baseline_speedscope_json_path
    if not (base_scope_path and base_scope_path.exists()):
        base_scope_path = None

    parts.append('<h3>CPU profiler (py-spy)</h3>')

    # --- Hotspots (optimized) ---
    if has_opt:
        parts.append('<h4>CPU hotspots</h4>')
        _render_pyspy_hotspots(parts, ps, esc)

    # --- Interactive flame graph via speedscope ---
    if scope_path:
        parts.append('<h4>Interactive flame graph</h4>')
        _render_speedscope_link(parts, scope_path, esc)

    # --- Comparison dropdown: hotspots ---
    if has_opt and has_base:
        parts.append('<details><summary>Compare hotspots with baseline</summary>')
        parts.append('<div class="compare-row">')
        parts.append('<div class="compare-col baseline"><h4>Baseline</h4>')
        _render_pyspy_hotspots(parts, bps, esc, color="#9ca3af")
        parts.append('</div>')
        parts.append('<div class="compare-col optimized"><h4>Optimized</h4>')
        _render_pyspy_hotspots(parts, ps, esc)
        parts.append('</div>')
        parts.append('</div>')
        parts.append('</details>')

    # --- Comparison dropdown: speedscope links ---
    if scope_path and base_scope_path:
        parts.append('<details><summary>Compare flame graphs with baseline</summary>')
        parts.append('<div class="compare-row">')
        parts.append('<div class="compare-col baseline"><h4>Baseline</h4>')
        _render_speedscope_link(parts, base_scope_path, esc, label="Open baseline in Speedscope")
        parts.append('</div>')
        parts.append('<div class="compare-col optimized"><h4>Optimized</h4>')
        _render_speedscope_link(parts, scope_path, esc, label="Open optimized in Speedscope")
        parts.append('</div>')
        parts.append('</div>')
        parts.append('</details>')

    # --- Differential flame graph ---
    if prof.diff_flame_svg_path and prof.diff_flame_svg_path.exists():
        parts.append('<details><summary>Differential flame graph (red=hotter, blue=cooler)</summary>')
        parts.append('<div class="flame-container">')
        parts.append(prof.diff_flame_svg_path.read_text(encoding="utf-8"))
        parts.append('</div>')
        parts.append('</details>')


# ---------------------------------------------------------------------------
# Memray section
# ---------------------------------------------------------------------------

def _render_memray_section(parts: list[str], prof: ProfilerData, esc) -> None:
    ms = prof.memray_summary
    if not ms:
        return

    parts.append('<h3>Memory profiler (memray)</h3>')

    # Metric pills
    parts.append('<div class="metric-pills">')
    peak_mb = ms.get("peak_memory_mb")
    if peak_mb is not None:
        _metric_pill(parts, "Peak Memory", f"{peak_mb:.1f}MB", None, peak_mb, lower_is_better=True)
    total_alloc_mb = ms.get("total_allocated_mb")
    if total_alloc_mb is not None:
        _metric_pill(parts, "Total Allocated", f"{total_alloc_mb:.1f}MB", None, total_alloc_mb, lower_is_better=True)
    total_allocs = ms.get("total_allocations")
    if total_allocs is not None:
        _metric_pill(parts, "Allocations", f"{total_allocs:,}", None, float(total_allocs), lower_is_better=True)
    parts.append('</div>')

    # Top allocators bar chart
    top_allocs = ms.get("top_allocators", [])
    if top_allocs:
        max_size = max(a.get("size_mb", 0) for a in top_allocs) or 1
        items = [
            {
                "name": f'{a.get("function", "?")} ({a.get("location", "")})',
                "pct": a.get("size_mb", 0) / max_size * 100,
                "total_ms": a.get("size_mb", 0),  # repurpose for display
            }
            for a in top_allocs[:8]
        ]
        parts.append('<h4>Top allocators (by size)</h4>')
        _render_bar_chart(parts, items, esc, color="#e879f9")
