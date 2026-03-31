from __future__ import annotations
import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
import html as _html


@dataclass
class GlanceData:
    """Data for the at-a-glance section of the dashboard."""
    metric_name: str = ""
    baseline_value: float = 0.0
    best_value: float = 0.0
    best_iter: int = 0
    total_iterations: int = 0
    speedup: float = 1.0
    accepted_count: int = 0
    early_stop_reason: str | None = None
    rows: list[dict] = field(default_factory=list)
    accepted_patches: list[dict] = field(default_factory=list)
    achieved_tflops: float | None = None
    peak_tflops: float | None = None
    achieved_bw_gbs: float | None = None
    peak_mem_bw_gbs: float | None = None
    roofline_source: str | None = None  # e.g. "cpu-spec", "nvidia-smi", "mps-heuristic"
    roofline_device: str | None = None  # e.g. "Apple M4", "NVIDIA A100"
    llm_model: str = ""
    llm_provider: str = ""
    llm_total_calls: int = 0
    llm_total_input_tokens: int = 0
    llm_total_output_tokens: int = 0
    llm_total_latency_s: float = 0.0


@dataclass
class ProfilerData:
    """Profiler artifacts to embed in the dashboard."""
    torch_summary: dict | None = None
    pyspy_summary: dict | None = None
    speedscope_json_path: Path | None = None  # pyspy_speedscope.json for interactive viewer
    torch_trace_path: Path | None = None  # torch_trace.json path
    metal_summary: dict | None = None
    nsys_summary: dict | None = None
    ncu_summary: dict | None = None
    roofline_png_path: Path | None = None
    jax_summary: dict | None = None
    memray_summary: dict | None = None
    perfetto_trace_path: Path | None = None
    # Baseline (for comparison)
    baseline_torch_summary: dict | None = None
    baseline_pyspy_summary: dict | None = None
    baseline_speedscope_json_path: Path | None = None
    baseline_metal_summary: dict | None = None
    baseline_nsys_summary: dict | None = None
    baseline_ncu_summary: dict | None = None
    baseline_jax_summary: dict | None = None
    diff_flame_svg_path: Path | None = None


def write_dashboard_html(
    path: Path,
    title: str,
    metric_png_rel: str | None,
    optimization_summary: str | None = None,
    glance: GlanceData | None = None,
    profiler: ProfilerData | None = None,
    system_info: dict | None = None,
    hardware_mismatch: str | None = None,
    bottleneck_diagnoses: list[dict] | None = None,
    gpu_attribution: list[dict] | None = None,
    profile_diff: list[dict] | None = None,
    build_flag_recs: list[dict] | None = None,
    hotspot_diff: list[dict] | None = None,
    history: list[dict] | None = None,
    tma_data: dict | None = None,
    tma_level2_data: dict | None = None,
    power_data: dict | None = None,
    pareto_png_rel: str | None = None,
    bench_stats_warning: str = "",
    vectorization: list[dict] | None = None,
    gpu_memory: dict | None = None,
    thread_sched: dict | None = None,
    ebpf_data: dict | None = None,
    lock_contention_data: dict | None = None,
    hlo_attribution: list[dict] | None = None,
    user_actions: list[dict] | None = None,
    microarch_summary: dict | None = None,
    torch_flops: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    esc = _html.escape

    parts = [f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{esc(title)}</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #1a1a1a; }}
    .card {{ border: 1px solid #ddd; border-radius: 14px; padding: 16px; margin: 12px 0; }}
    code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 6px; font-size: 0.9em; }}
    a {{ text-decoration: none; color: #0066cc; }}
    a:hover {{ text-decoration: underline; }}
    .explanation {{ background: #f9f9f9; padding: 12px; border-radius: 8px; line-height: 1.5; }}

    /* At-a-glance */
    .glance {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }}
    .stat {{
      flex: 1; min-width: 130px;
      background: #f7f7f7; border-radius: 12px; padding: 14px 16px; text-align: center;
    }}
    .stat .label {{ font-size: 0.8em; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }}
    .stat .value {{ font-size: 1.6em; font-weight: 700; margin-top: 4px; }}
    .stat .value.good {{ color: #16a34a; }}
    .stat .value.neutral {{ color: #1a1a1a; }}
    .stop-reason {{ font-size: 0.85em; color: #888; margin-top: 8px; }}

    /* Iteration table */
    .iter-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; margin-top: 8px; }}
    .iter-table th {{ background: #f0f0f0; padding: 6px 10px; text-align: left; font-weight: 600; }}
    .iter-table td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
    .iter-table tr.accepted {{ background: #f0fdf4; }}
    .badge {{
      display: inline-block; padding: 1px 7px; border-radius: 8px; font-size: 0.8em; font-weight: 600;
    }}
    .badge.yes {{ background: #dcfce7; color: #16a34a; }}
    .badge.no {{ background: #fee2e2; color: #b91c1c; }}
    .badge.high {{ background: #dcfce7; color: #16a34a; }}
    .badge.medium {{ background: #fef9c3; color: #a16207; }}
    .badge.low {{ background: #f3f4f6; color: #6b7280; }}

    /* Profiler bars */
    .op-bar-container {{ margin: 4px 0; display: flex; align-items: center; gap: 8px; }}
    .op-bar-label {{ min-width: 260px; font-size: 0.85em; font-family: monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .op-bar-track {{ flex: 1; background: #eee; border-radius: 4px; height: 18px; position: relative; }}
    .op-bar-fill {{ height: 100%; border-radius: 4px; background: #3b82f6; min-width: 2px; }}
    .op-bar-pct {{ min-width: 50px; font-size: 0.8em; color: #666; text-align: right; }}
    .op-bar-meta {{ font-size: 0.75em; color: #999; min-width: 80px; text-align: right; }}

    /* CPU vs GPU */
    .split-bar {{ display: flex; height: 28px; border-radius: 6px; overflow: hidden; margin: 8px 0; }}
    .split-bar .cpu {{ background: #60a5fa; }}
    .split-bar .gpu {{ background: #34d399; }}
    .split-legend {{ display: flex; gap: 16px; font-size: 0.85em; margin-top: 4px; }}
    .split-legend .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}

    /* Flame graph */
    .flame-container {{ width: 100%; overflow-x: auto; }}
    .flame-container svg {{ width: 100%; height: auto; }}

    /* What Changed - patch diffs */
    .patch-card {{ border: 1px solid #ddd; border-radius: 14px; padding: 16px; margin: 12px 0; }}
    .patch-card details {{ margin: 8px 0; }}
    .patch-card summary {{ cursor: pointer; font-weight: 600; font-size: 0.95em; padding: 4px 0; }}
    .diff-block {{ font-family: monospace; font-size: 0.85em; border-radius: 6px; overflow-x: auto; margin: 6px 0; }}
    .diff-file {{ font-weight: 600; font-size: 0.85em; color: #555; margin: 8px 0 2px; }}
    .diff-del {{ background: #fee2e2; padding: 4px 8px; white-space: pre-wrap; word-break: break-all; }}
    .diff-add {{ background: #dcfce7; padding: 4px 8px; white-space: pre-wrap; word-break: break-all; }}

    /* Comparison layout */
    .compare-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .compare-col {{ flex: 1; min-width: 300px; }}
    .compare-col.baseline h4 {{ color: #888; }}
    .compare-col.optimized h4 {{ color: #16a34a; }}

    /* Metric pills */
    .metric-pills {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }}
    .metric-pill {{
      display: inline-flex; align-items: center; gap: 6px;
      background: #f7f7f7; border-radius: 20px; padding: 6px 14px; font-size: 0.85em;
    }}
    .metric-pill .pill-label {{ color: #666; }}
    .metric-pill .pill-value {{ font-weight: 700; }}
    .metric-pill .delta {{ font-size: 0.85em; font-weight: 600; }}
    .metric-pill .delta.good {{ color: #16a34a; }}
    .metric-pill .delta.bad {{ color: #b91c1c; }}

    /* Profiler section details */
    .profiler-section details summary {{ cursor: pointer; font-weight: 600; padding: 4px 0; }}

    /* Hardware mismatch warning */
    .hardware-warning {{
      background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px;
      padding: 12px 16px; margin: 12px 0; font-size: 0.95em; color: #92400e;
    }}
  </style>
</head>
<body>
<h1>{esc(title)}</h1>
"""]

    # --- Hardware mismatch warning ---
    if hardware_mismatch:
        parts.append(f'<div class="hardware-warning">\u26a0 {esc(hardware_mismatch)}</div>')

    # --- Benchmark noise warning ---
    if bench_stats_warning:
        parts.append(f'<div class="hardware-warning">\u26a0 {esc(bench_stats_warning)}</div>')

    # --- At a Glance ---
    if glance:
        speedup_str = f"{glance.speedup:.2f}x"
        parts.append('<div class="card">')
        parts.append('<h2>At a glance</h2>')
        parts.append('<div class="glance">')
        delta = glance.best_value - glance.baseline_value
        delta_str = f"+{delta:.4g}" if delta >= 0 else f"{delta:.4g}"
        parts.append(f'<div class="stat"><div class="label">Baseline</div><div class="value neutral">{glance.baseline_value:.4g}</div></div>')
        parts.append(f'<div class="stat"><div class="label">Best</div><div class="value good">{glance.best_value:.4g}</div></div>')
        parts.append(f'<div class="stat"><div class="label">Delta</div><div class="value good">{delta_str}</div></div>')
        parts.append(f'<div class="stat"><div class="label">Speedup</div><div class="value good">{speedup_str}</div></div>')
        parts.append(f'<div class="stat"><div class="label">Accepted</div><div class="value neutral">{glance.accepted_count} / {glance.total_iterations}</div></div>')
        parts.append(f'<div class="stat"><div class="label">Best Iter</div><div class="value neutral">{glance.best_iter}</div></div>')
        if glance.achieved_tflops is not None:
            parts.append(f'<div class="stat"><div class="label">TFLOPS</div><div class="value good">{glance.achieved_tflops:.2f}</div></div>')
        if glance.achieved_tflops is not None and glance.peak_tflops is not None and glance.peak_tflops > 0:
            pct_of_peak = glance.achieved_tflops / glance.peak_tflops * 100
            parts.append(f'<div class="stat"><div class="label">% of Peak</div><div class="value good">{pct_of_peak:.1f}%</div></div>')
        if glance.achieved_bw_gbs is not None:
            parts.append(f'<div class="stat"><div class="label">BW (GB/s)</div><div class="value neutral">{glance.achieved_bw_gbs:.1f}</div></div>')
        if glance.achieved_bw_gbs is not None and glance.peak_mem_bw_gbs is not None and glance.peak_mem_bw_gbs > 0:
            bw_pct = glance.achieved_bw_gbs / glance.peak_mem_bw_gbs * 100
            parts.append(f'<div class="stat"><div class="label">% BW Peak</div><div class="value good">{bw_pct:.1f}%</div></div>')
        if glance.peak_tflops is not None and glance.roofline_source:
            src_label = esc(glance.roofline_device or glance.roofline_source)
            parts.append(f'<div class="stat"><div class="label">Roofline</div><div class="value neutral" style="font-size:0.9em">{glance.peak_tflops:.3f} TFLOPS<br><small>{src_label}</small></div></div>')
        if glance.llm_model:
            parts.append(f'<div class="stat"><div class="value neutral" style="font-size:1.1em">{esc(glance.llm_model)}</div><div class="label">LLM Model</div></div>')
        parts.append('</div>')
        if glance.early_stop_reason:
            parts.append(f'<p class="stop-reason">{esc(glance.early_stop_reason)}</p>')
        parts.append('</div>')

    # --- Environment info ---
    if system_info:
        _render_environment(parts, system_info, esc)

    # --- LLM Usage ---
    if glance and glance.llm_total_calls > 0:
        total_tokens = glance.llm_total_input_tokens + glance.llm_total_output_tokens
        parts.append('<details class="card" style="cursor:pointer">')
        parts.append('<summary><h2 style="display:inline">LLM Usage</h2></summary>')
        parts.append('<table class="iter-table" style="margin-top:8px">')
        llm_rows = [
            ("Provider", esc(glance.llm_provider) if glance.llm_provider else "—"),
            ("Model", esc(glance.llm_model) if glance.llm_model else "—"),
            ("Total Calls", str(glance.llm_total_calls)),
            ("Input Tokens", f"{glance.llm_total_input_tokens:,}"),
            ("Output Tokens", f"{glance.llm_total_output_tokens:,}"),
            ("Total Tokens", f"{total_tokens:,}"),
            ("Total Latency", f"{glance.llm_total_latency_s:.1f}s"),
        ]
        for label, value in llm_rows:
            parts.append(
                f'<tr><td style="font-weight:600;width:140px">{label}</td>'
                f'<td>{value}</td></tr>'
            )
        parts.append('</table>')
        parts.append('</details>')

    # --- Iteration table ---
    if glance:
        if glance.rows:
            parts.append('<div class="card">')
            parts.append('<h2>Iterations</h2>')
            parts.append('<table class="iter-table">')
            parts.append(f'<tr><th>Iter</th><th>{esc(glance.metric_name)}</th><th>Speedup</th><th>Status</th><th>Notes</th></tr>')
            for row in glance.rows:
                it = row.get("iter", "?")
                val = row.get("value", 0)
                spd = row.get("speedup", 1.0)
                accepted = row.get("accepted", False)
                notes = row.get("notes", "")
                tr_class = ' class="accepted"' if accepted else ""
                badge = '<span class="badge yes">accepted</span>' if accepted else '<span class="badge no">rejected</span>'
                if it == 0:
                    badge = '<span class="badge yes">baseline</span>'
                parts.append(
                    f'<tr{tr_class}>'
                    f'<td>{esc(str(it))}</td>'
                    f'<td>{val:.4g}</td>'
                    f'<td>{spd:.2f}x</td>'
                    f'<td>{badge}</td>'
                    f'<td>{esc(str(notes))}</td>'
                    f'</tr>'
                )
            parts.append('</table>')
            parts.append('</div>')

    # --- What Changed ---
    if glance and glance.accepted_patches:
        parts.append('<div class="patch-card">')
        parts.append('<h2>What changed</h2>')
        for patch in glance.accepted_patches:
            p_iter = patch.get("iteration", "?")
            p_desc = patch.get("description", "")
            p_val = patch.get("value")
            val_str = f" ({p_val:.4g})" if p_val is not None else ""
            parts.append(f'<details><summary>Iter {p_iter} — {esc(p_desc)}{val_str}</summary>')
            for block in patch.get("blocks", []):
                fp = block.get("file_path", "")
                search = block.get("search", "")
                replace = block.get("replace", "")
                parts.append(f'<div class="diff-file">{esc(fp)}</div>')
                parts.append('<div class="diff-block">')
                if search:
                    parts.append(f'<pre class="diff-del">{esc(search)}</pre>')
                if replace:
                    parts.append(f'<pre class="diff-add">{esc(replace)}</pre>')
                parts.append('</div>')
            parts.append('</details>')
        parts.append('</div>')

    # --- Metric history chart ---
    parts.append('<div class="card"><h2>Metric history</h2>')
    if metric_png_rel:
        png_path = path.parent / metric_png_rel
        if png_path.exists():
            b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
            parts.append(f'<img src="data:image/png;base64,{b64}" style="max-width: 100%; height: auto;" />')
        else:
            parts.append(f'<img src="{esc(metric_png_rel)}" style="max-width: 100%; height: auto;" />')
    else:
        parts.append("<p>(No plot)</p>")
    parts.append("</div>")

    # --- Pareto frontier chart (optional) ---
    if pareto_png_rel:
        pareto_path = path.parent / pareto_png_rel
        if pareto_path.exists():
            parts.append('<div class="card"><h2>Pareto frontier</h2>')
            b64 = base64.b64encode(pareto_path.read_bytes()).decode("ascii")
            parts.append(f'<img src="data:image/png;base64,{b64}" style="max-width: 100%; height: auto;" />')
            parts.append("</div>")

    # --- Roofline chart ---
    if profiler and profiler.roofline_png_path and profiler.roofline_png_path.exists():
        parts.append('<div class="card"><h2>Roofline</h2>')
        b64 = base64.b64encode(profiler.roofline_png_path.read_bytes()).decode("ascii")
        parts.append(f'<img src="data:image/png;base64,{b64}" style="max-width: 100%; height: auto;" />')
        parts.append("</div>")

    # --- What worked / What didn't ---
    _render_outcome_analysis(parts, esc, optimization_summary=optimization_summary,
                             history=history, glance=glance)

    # --- User Action Required ---
    if user_actions:
        _render_user_actions(parts, esc, user_actions)

    # --- Diagnostics ---
    _render_diagnostics(parts, esc,
                        bottleneck_diagnoses=bottleneck_diagnoses,
                        gpu_attribution=gpu_attribution,
                        profile_diff=profile_diff,
                        build_flag_recs=build_flag_recs,
                        hotspot_diff=hotspot_diff,
                        tma_data=tma_data,
                        tma_level2_data=tma_level2_data,
                        power_data=power_data,
                        vectorization=vectorization,
                        gpu_memory=gpu_memory,
                        thread_sched=thread_sched,
                        ebpf_data=ebpf_data,
                        lock_contention_data=lock_contention_data,
                        hlo_attribution=hlo_attribution)

    # --- Micro-architecture analysis ---
    if microarch_summary:
        parts.append('<div class="card"><h2>Micro-Architecture Analysis</h2>')

        ceiling = microarch_summary.get("kernel_ceiling")
        if ceiling:
            occ = ceiling.get("occupancy_pct", 0)
            ceil_tf = ceiling.get("kernel_ceiling_tflops", 0)
            ach = ceiling.get("achieved_tflops")
            parts.append(f'<h3>Kernel Performance Ceiling</h3>')
            parts.append(f'<p>Occupancy: <strong>{occ:.0f}%</strong> → '
                         f'theoretical max: <strong>{ceil_tf:.1f} TFLOPS</strong></p>')
            if ach is not None:
                pct_ceil = ceiling.get("pct_of_ceiling", 0)
                parts.append(f'<p>Currently achieving: <strong>{ach:.3f} TFLOPS</strong> '
                             f'({pct_ceil:.0f}% of kernel ceiling)</p>')
            if ceiling.get("occupancy_limited"):
                parts.append('<p style="color:#e74c3c;font-weight:bold;">'
                             '→ Occupancy is the primary limiter. Fix occupancy BEFORE optimizing compute.</p>')

        heatmap = microarch_summary.get("pipeline_heatmap")
        if heatmap:
            parts.append(f'<h3>Pipeline Utilization</h3><pre style="font-size:0.85em;">{esc(heatmap)}</pre>')

        stability = microarch_summary.get("benchmark_stability")
        if stability:
            color = "#27ae60" if stability.get("is_stable") else "#e74c3c"
            parts.append(f'<h3>Benchmark Stability</h3>')
            parts.append(f'<p style="color:{color}">{esc(stability.get("assessment", ""))}</p>')

        throttle = microarch_summary.get("clock_throttle")
        if throttle and throttle.get("throttle_detected"):
            parts.append(f'<h3>GPU Thermal Status</h3>')
            parts.append(f'<p style="color:#e67e22">{esc(throttle.get("assessment", ""))}</p>')

        parts.append('</div>')

    # --- PyTorch FLOPS ---
    if torch_flops and torch_flops.get("total_tflops"):
        parts.append('<div class="card"><h2>PyTorch Operator FLOPS</h2>')
        parts.append(f'<p>Total: <strong>{torch_flops["total_tflops"]:.4f} TFLOPS</strong></p>')
        top_ops = torch_flops.get("top_ops_by_flops", [])
        if top_ops:
            parts.append('<table class="iter-table"><tr><th>Operator</th><th>% of FLOPS</th></tr>')
            for op in top_ops[:5]:
                parts.append(f'<tr><td><code>{esc(op["name"])}</code></td><td>{op["pct"]:.1f}%</td></tr>')
            parts.append('</table>')
        parts.append('</div>')

    # --- Profiler insights ---
    if profiler:
        _render_profiler(parts, profiler, esc)

    # --- Perfetto trace link ---
    if profiler and profiler.perfetto_trace_path and profiler.perfetto_trace_path.exists():
        trace_size_kb = profiler.perfetto_trace_path.stat().st_size / 1024
        parts.append(
            '<div class="card">'
            '<h2>Perfetto trace</h2>'
            f'<p>Open <a href="https://ui.perfetto.dev/" target="_blank">Perfetto UI</a> '
            f'and load <code>{esc(str(profiler.perfetto_trace_path))}</code> '
            f'({trace_size_kb:.0f} KB) for interactive timeline visualization of '
            f'CPU hotspots and hardware counters.</p>'
            '</div>'
        )

    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_user_actions(parts: list[str], esc, user_actions: list[dict]) -> None:
    """Render a prominent 'User Action Required' card for build suggestions."""
    parts.append(
        '<div class="card" style="background:#fff8e1; border-left:4px solid #f57f17;">'
        '<h2 style="color:#e65100;">&#9888; User Action Required</h2>'
        '<p>The optimizer suggested build/compilation changes that it could not apply '
        'automatically (task.yaml is a protected file). Update your '
        '<code>build.cmd</code> in <code>task.yaml</code> and re-run to unlock '
        'further improvements.</p>'
        '<table class="iter-table">'
        '<tr><th>Flag</th><th>Suggestion</th><th>Iteration</th></tr>'
    )
    for a in user_actions:
        parts.append(
            f'<tr><td><code>{esc(a["flag"])}</code></td>'
            f'<td>{esc(a["suggestion"])}</td>'
            f'<td>{esc(str(a.get("iteration", "—")))}</td></tr>'
        )
    parts.append('</table></div>')


def _render_diagnostics(
    parts: list[str],
    esc,
    *,
    bottleneck_diagnoses: list[dict] | None = None,
    gpu_attribution: list[dict] | None = None,
    profile_diff: list[dict] | None = None,
    build_flag_recs: list[dict] | None = None,
    hotspot_diff: list[dict] | None = None,
    tma_data: dict | None = None,
    tma_level2_data: dict | None = None,
    power_data: dict | None = None,
    vectorization: list[dict] | None = None,
    gpu_memory: dict | None = None,
    thread_sched: dict | None = None,
    ebpf_data: dict | None = None,
    lock_contention_data: dict | None = None,
    hlo_attribution: list[dict] | None = None,
) -> None:
    """Render the Diagnostics card with collapsible sections."""
    if not any([bottleneck_diagnoses, gpu_attribution, profile_diff, build_flag_recs,
                hotspot_diff, tma_data, tma_level2_data, power_data, vectorization, gpu_memory,
                thread_sched, ebpf_data, lock_contention_data, hlo_attribution]):
        return

    parts.append('<div class="card"><h2>Diagnostics</h2>')

    # (a) Bottleneck diagnosis
    if bottleneck_diagnoses:
        parts.append('<details><summary>Bottleneck diagnosis</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Rank</th><th>Bottleneck</th><th>Root Cause</th>'
                     '<th>Confidence</th><th>Suggested Actions</th></tr>')
        for d in bottleneck_diagnoses:
            conf = d.get("confidence", "")
            badge_cls = conf if conf in ("high", "medium", "low") else ""
            actions = d.get("suggested_actions", [])
            actions_str = "; ".join(actions) if isinstance(actions, list) else str(actions)
            parts.append(
                f'<tr>'
                f'<td>{esc(str(d.get("rank", "")))}</td>'
                f'<td>{esc(str(d.get("bottleneck", "")))}</td>'
                f'<td>{esc(str(d.get("root_cause", "")))}</td>'
                f'<td><span class="badge {badge_cls}">{esc(conf)}</span></td>'
                f'<td>{esc(actions_str)}</td>'
                f'</tr>'
            )
        parts.append('</table></details>')

    # (b) GPU attribution
    if gpu_attribution:
        parts.append('<details><summary>GPU attribution</summary>')
        bar_items = [
            {"name": a["name"], "pct": a.get("gpu_pct", 0), "total_ms": a.get("gpu_time_ms", 0)}
            for a in gpu_attribution
        ]
        _render_bar_chart(parts, bar_items, esc, color="#34d399")
        # Per-kernel diagnosis text
        for a in gpu_attribution:
            diag = a.get("diagnosis", "")
            sugg = a.get("suggestions", [])
            if diag or sugg:
                parts.append(f'<p style="font-size:0.85em;margin:2px 0 6px 4px;">'
                             f'<strong>{esc(a["name"][:50])}</strong>: {esc(diag)}')
                if sugg:
                    parts.append(f'<br/><em>{esc("; ".join(sugg))}</em>')
                parts.append('</p>')
        parts.append('</details>')

    # (b2) HLO attribution (JAX/TPU)
    if hlo_attribution:
        parts.append('<details><summary>XLA/HLO op attribution</summary>')
        bar_items = [
            {
                "name": f'{a["op"]} ({a.get("category", "?")})',
                "pct": a.get("estimated_device_pct", 0),
                "total_ms": a.get("count", 0),
            }
            for a in hlo_attribution[:10]
        ]
        _render_bar_chart(parts, bar_items, esc, color="#f59e0b")
        for a in hlo_attribution[:10]:
            diag = a.get("diagnosis", "")
            sugg = a.get("suggestions", [])
            if diag or sugg:
                parts.append(
                    f'<p style="font-size:0.85em;margin:2px 0 6px 4px;">'
                    f'<strong>{esc(str(a["op"]))}</strong> '
                    f'({a.get("count", 0)} ops, ~{a.get("estimated_device_pct", 0):.0f}% device time): '
                    f'{esc(diag)}'
                )
                if sugg:
                    parts.append(f'<br/><em>{esc("; ".join(sugg))}</em>')
                parts.append('</p>')
        parts.append('</details>')

    # (c) Profile diff
    if profile_diff:
        parts.append('<details><summary>Profile diff: baseline vs optimized</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Metric</th><th>Before</th><th>After</th>'
                     '<th>Change%</th><th>Direction</th></tr>')
        for d in profile_diff:
            direction = d.get("direction", "unchanged")
            if direction == "improved":
                dir_badge = '<span class="badge yes">improved</span>'
            elif direction == "regressed":
                dir_badge = '<span class="badge no">regressed</span>'
            else:
                dir_badge = f'<span class="badge">{esc(direction)}</span>'
            delta_pct = d.get("delta_pct", 0)
            sign = "+" if delta_pct > 0 else ""
            parts.append(
                f'<tr>'
                f'<td><code>{esc(str(d.get("metric", "")))}</code></td>'
                f'<td>{d.get("before", 0):.4g}</td>'
                f'<td>{d.get("after", 0):.4g}</td>'
                f'<td>{sign}{delta_pct:.1f}%</td>'
                f'<td>{dir_badge}</td>'
                f'</tr>'
            )
        parts.append('</table></details>')

    # (d) Hotspot diff
    if hotspot_diff:
        parts.append('<details><summary>CPU hotspot shifts: baseline vs optimized</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Function</th><th>Before</th><th>After</th>'
                     '<th>Change</th><th>Status</th></tr>')
        for s in hotspot_diff:
            status = s.get("status", "unchanged")
            before = s.get("before_pct", 0)
            after = s.get("after_pct", 0)
            delta = s.get("delta_pct", 0)
            func = s.get("function", "?")
            if len(func) > 50:
                func = func[:47] + "..."
            if status == "new":
                status_badge = '<span class="badge no">new</span>'
            elif status == "removed":
                status_badge = '<span class="badge yes">gone</span>'
            elif status == "decreased":
                status_badge = '<span class="badge yes">decreased</span>'
            elif status == "increased":
                status_badge = '<span class="badge no">increased</span>'
            else:
                status_badge = f'<span class="badge">{esc(status)}</span>'
            sign = "+" if delta > 0 else ""
            parts.append(
                f'<tr>'
                f'<td><code>{esc(func)}</code></td>'
                f'<td>{before:.1f}%</td>'
                f'<td>{after:.1f}%</td>'
                f'<td>{sign}{delta:.1f}pp</td>'
                f'<td>{status_badge}</td>'
                f'</tr>'
            )
        parts.append('</table></details>')

    # (e) Build flag recommendations
    if build_flag_recs:
        parts.append('<details open><summary>Recommended compile flags</summary>')
        parts.append('<p style="color:#999;font-size:0.85em;">Based on ISA detection + profiler analysis. '
                     'Apply these to your build command for better performance.</p>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Flag</th><th>Reason</th><th>Impact</th><th>Category</th></tr>')
        for r in build_flag_recs:
            impact = r.get("impact", "")
            impact_cls = impact if impact in ("high", "medium", "low") else ""
            parts.append(
                f'<tr>'
                f'<td><code>{esc(str(r.get("flag", "")))}</code></td>'
                f'<td>{esc(str(r.get("reason", "")))}</td>'
                f'<td><span class="badge {impact_cls}">{esc(impact)}</span></td>'
                f'<td>{esc(str(r.get("category", "")))}</td>'
                f'</tr>'
            )
        parts.append('</table>')
        parts.append('<p style="color:#999;font-size:0.85em;margin-top:8px;">'
                     '<strong>Production build:</strong> After optimization converges, compile with: '
                     '<code>-O3 -march=native -mtune=native -flto -DNDEBUG</code>. '
                     'For +10-20%, use PGO: <code>-fprofile-generate</code> → run → '
                     '<code>-fprofile-use -flto</code>.</p>')
        parts.append('</details>')

    # (f) Top-Down Microarchitecture Analysis
    if tma_data:
        parts.append('<details><summary>Top-Down Microarchitecture Analysis (TMA)</summary>')
        parts.append('<div style="display:flex;gap:16px;flex-wrap:wrap;margin:8px 0;">')
        for label, key, color in [
            ("Frontend Bound", "frontend_bound_pct", "#e74c3c"),
            ("Backend Bound", "backend_bound_pct", "#e67e22"),
            ("Bad Speculation", "bad_speculation_pct", "#9b59b6"),
            ("Retiring", "retiring_pct", "#27ae60"),
        ]:
            val = tma_data.get(key, 0)
            parts.append(
                f'<div style="text-align:center;min-width:120px;">'
                f'<div style="font-size:24px;font-weight:bold;color:{color}">{val:.1f}%</div>'
                f'<div style="font-size:12px;color:#666">{label}</div>'
                f'</div>'
            )
        parts.append('</div>')
        dominant = tma_data.get("dominant_bottleneck", "").replace("_", " ").title()
        if dominant:
            parts.append(f'<p>Dominant bottleneck: <strong>{esc(dominant)}</strong></p>')

        # TMA Level 2/3 (from toplev or AMD perf events)
        l2 = tma_level2_data
        if l2:
            parts.append('<h4 style="margin-top:12px;">Level 2/3 Breakdown</h4>')
            parts.append('<div style="display:flex;gap:16px;flex-wrap:wrap;margin:8px 0;">')
            for label, key, color in [
                ("Memory Bound", "memory_bound_pct", "#e67e22"),
                ("Core Bound", "core_bound_pct", "#3498db"),
                ("Fetch Latency", "fetch_latency_pct", "#e74c3c"),
                ("Fetch BW", "fetch_bandwidth_pct", "#c0392b"),
            ]:
                val = l2.get(key)
                if val is not None:
                    parts.append(
                        f'<div style="text-align:center;min-width:110px;">'
                        f'<div style="font-size:20px;font-weight:bold;color:{color}">{val:.1f}%</div>'
                        f'<div style="font-size:11px;color:#666">{label}</div>'
                        f'</div>'
                    )
            parts.append('</div>')

            # Memory hierarchy breakdown
            mem_levels = [
                ("L1", "l1_bound_pct"), ("L2", "l2_bound_pct"),
                ("L3", "l3_bound_pct"), ("DRAM", "dram_bound_pct"),
                ("Store", "store_bound_pct"),
            ]
            active = [(n, l2[k]) for n, k in mem_levels if l2.get(k) is not None]
            if active:
                parts.append('<p style="margin-top:4px;">Memory hierarchy: ')
                parts.append(' → '.join(f'<strong>{n}</strong> {v:.1f}%' for n, v in active))
                dom = l2.get("dominant_memory_level")
                if dom:
                    parts.append(f' (bottleneck: <strong>{esc(dom)}</strong>)')
                parts.append('</p>')

            source = l2.get("source", "")
            if source:
                parts.append(f'<p style="color:#999;font-size:0.8em;">Source: {esc(source)}</p>')

        parts.append('</details>')

    # (g) Power/Energy profiling
    if power_data:
        parts.append('<details><summary>Power &amp; energy profiling</summary>')
        rapl = power_data.get("rapl", {})
        gpu_power = power_data.get("gpu_power", {})
        if rapl:
            parts.append('<h4>CPU (RAPL)</h4><table class="iter-table">')
            if "package_joules" in rapl:
                parts.append(f'<tr><td>Package energy</td><td>{rapl["package_joules"]:.2f} J</td></tr>')
            if "cores_joules" in rapl:
                parts.append(f'<tr><td>Cores energy</td><td>{rapl["cores_joules"]:.2f} J</td></tr>')
            if "avg_package_watts" in rapl:
                parts.append(f'<tr><td>Avg package power</td><td>{rapl["avg_package_watts"]:.1f} W</td></tr>')
            parts.append('</table>')
        if gpu_power:
            parts.append('<h4>GPU (nvidia-smi)</h4><table class="iter-table">')
            if "avg_watts" in gpu_power:
                parts.append(f'<tr><td>Avg power draw</td><td>{gpu_power["avg_watts"]:.1f} W</td></tr>')
            if "max_watts" in gpu_power:
                parts.append(f'<tr><td>Peak power draw</td><td>{gpu_power["max_watts"]:.1f} W</td></tr>')
            if "sample_count" in gpu_power:
                parts.append(f'<tr><td>Samples</td><td>{gpu_power["sample_count"]}</td></tr>')
            parts.append('</table>')
        parts.append('</details>')

    # (h) Vectorization analysis
    if vectorization:
        parts.append('<details><summary>Vectorization analysis</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Function</th><th>SIMD</th><th>ISA</th><th>CPU %</th></tr>')
        for v in vectorization:
            has = v.get("has_simd", False)
            badge = '<span class="badge yes">yes</span>' if has else '<span class="badge no">no</span>'
            parts.append(
                f'<tr>'
                f'<td><code>{esc(str(v.get("function", "")))}</code></td>'
                f'<td>{badge}</td>'
                f'<td>{esc(str(v.get("simd_isa", "none")))}</td>'
                f'<td>{v.get("hot_pct", 0):.1f}%</td>'
                f'</tr>'
            )
        parts.append('</table></details>')

    # (i) GPU memory utilization
    if gpu_memory:
        parts.append('<details><summary>GPU memory utilization</summary>')
        parts.append('<table class="iter-table">')
        if "total_mib" in gpu_memory:
            parts.append(f'<tr><td>Total VRAM</td><td>{gpu_memory["total_mib"]:.0f} MiB</td></tr>')
        if "max_used_mib" in gpu_memory:
            parts.append(f'<tr><td>Peak used</td><td>{gpu_memory["max_used_mib"]:.0f} MiB</td></tr>')
        if "avg_used_mib" in gpu_memory:
            parts.append(f'<tr><td>Avg used</td><td>{gpu_memory["avg_used_mib"]:.0f} MiB</td></tr>')
        if "utilization_pct" in gpu_memory:
            pct = gpu_memory["utilization_pct"]
            color = "#b91c1c" if pct > 90 else "#16a34a"
            parts.append(f'<tr><td>Utilization</td><td style="color:{color};font-weight:700">{pct:.1f}%</td></tr>')
        parts.append('</table></details>')

    # (j) Thread scheduling
    if thread_sched:
        latency = thread_sched.get("latency", [])
        timehist = thread_sched.get("timehist", {})
        if latency or timehist:
            parts.append('<details><summary>Thread scheduling (perf sched)</summary>')
            if latency:
                parts.append('<table class="iter-table">')
                parts.append('<tr><th>Thread</th><th>Runtime</th><th>Switches</th>'
                             '<th>Avg Delay</th><th>Max Delay</th></tr>')
                for entry in latency[:10]:
                    parts.append(
                        f'<tr>'
                        f'<td><code>{esc(str(entry.get("task", "")))}</code></td>'
                        f'<td>{entry.get("runtime_ms", 0):.1f}ms</td>'
                        f'<td>{entry.get("switches", 0)}</td>'
                        f'<td>{entry.get("avg_delay_ms", 0):.3f}ms</td>'
                        f'<td>{entry.get("max_delay_ms", 0):.3f}ms</td>'
                        f'</tr>'
                    )
                parts.append('</table>')
            if timehist.get("migrations"):
                parts.append(f'<p>Thread migrations: <strong>{timehist["migrations"]}</strong></p>')
            parts.append('</details>')

    # (k) eBPF I/O tracing
    if ebpf_data:
        has_io_data = ebpf_data.get("read_syscalls") or ebpf_data.get("write_syscalls")
        if has_io_data:
            parts.append('<details><summary>I/O syscall tracing (eBPF)</summary>')
            parts.append('<table class="iter-table">')
            if ebpf_data.get("read_syscalls") is not None:
                parts.append(f'<tr><td>Read syscalls</td><td>{ebpf_data["read_syscalls"]:,}</td></tr>')
            if ebpf_data.get("write_syscalls") is not None:
                parts.append(f'<tr><td>Write syscalls</td><td>{ebpf_data["write_syscalls"]:,}</td></tr>')
            if ebpf_data.get("read_bytes") is not None:
                mb = ebpf_data["read_bytes"] / (1024 * 1024)
                parts.append(f'<tr><td>Read bytes</td><td>{mb:.1f} MB</td></tr>')
            if ebpf_data.get("write_bytes") is not None:
                mb = ebpf_data["write_bytes"] / (1024 * 1024)
                parts.append(f'<tr><td>Write bytes</td><td>{mb:.1f} MB</td></tr>')
            read_lat = ebpf_data.get("read_latency", {})
            if read_lat.get("p50_ns") is not None:
                parts.append(f'<tr><td>Read latency p50</td><td>{read_lat["p50_ns"] / 1000:.1f} µs</td></tr>')
            if read_lat.get("p99_ns") is not None:
                p99_us = read_lat["p99_ns"] / 1000
                color = "#b91c1c" if p99_us > 10000 else "#16a34a"
                parts.append(f'<tr><td>Read latency p99</td><td style="color:{color};font-weight:700">{p99_us:.0f} µs</td></tr>')
            write_lat = ebpf_data.get("write_latency", {})
            if write_lat.get("p50_ns") is not None:
                parts.append(f'<tr><td>Write latency p50</td><td>{write_lat["p50_ns"] / 1000:.1f} µs</td></tr>')
            if write_lat.get("p99_ns") is not None:
                p99_us = write_lat["p99_ns"] / 1000
                color = "#b91c1c" if p99_us > 10000 else "#16a34a"
                parts.append(f'<tr><td>Write latency p99</td><td style="color:{color};font-weight:700">{p99_us:.0f} µs</td></tr>')
            parts.append('</table></details>')

    # (l) Lock contention
    if lock_contention_data:
        lock_stats = lock_contention_data.get("lock_stats", {})
        c2c_stats = lock_contention_data.get("c2c_stats", {})
        if lock_stats.get("locks") or c2c_stats.get("total_hitm"):
            parts.append('<details><summary>Lock contention (perf lock / c2c)</summary>')
            locks = lock_stats.get("locks", [])
            if locks:
                parts.append('<h4>Lock statistics</h4>')
                parts.append('<table class="iter-table">')
                parts.append('<tr><th>Lock</th><th>Acquired</th><th>Contended</th><th>Contention %</th><th>Total Wait</th></tr>')
                for lock in locks[:10]:
                    acq = lock.get("acquired", 0)
                    cont = lock.get("contended", 0)
                    cont_pct = (cont / acq * 100) if acq > 0 else 0
                    wait_ms = lock.get("total_wait_ns", 0) / 1e6
                    color = "#b91c1c" if cont_pct > 10 else "#16a34a"
                    parts.append(
                        f'<tr>'
                        f'<td><code>{esc(str(lock.get("name", "?")))}</code></td>'
                        f'<td>{acq:,}</td>'
                        f'<td>{cont:,}</td>'
                        f'<td style="color:{color};font-weight:700">{cont_pct:.1f}%</td>'
                        f'<td>{wait_ms:.1f} ms</td>'
                        f'</tr>'
                    )
                parts.append('</table>')
            if c2c_stats.get("total_hitm", 0) > 0:
                parts.append('<h4>False sharing (perf c2c)</h4>')
                parts.append('<table class="iter-table">')
                parts.append(f'<tr><td>Total HITM events</td><td>{c2c_stats["total_hitm"]:,}</td></tr>')
                if c2c_stats.get("total_store"):
                    parts.append(f'<tr><td>Total stores</td><td>{c2c_stats["total_store"]:,}</td></tr>')
                parts.append('</table>')
                sharing_lines = c2c_stats.get("false_sharing_lines", [])
                if sharing_lines:
                    parts.append('<p>Top cache-line conflicts:</p>')
                    parts.append('<table class="iter-table">')
                    parts.append('<tr><th>Address</th><th>HITM</th><th>Stores</th></tr>')
                    for cl in sharing_lines[:5]:
                        parts.append(
                            f'<tr><td><code>{esc(str(cl.get("address", "")))}</code></td>'
                            f'<td>{cl.get("hitm", 0):,}</td>'
                            f'<td>{cl.get("store", 0):,}</td></tr>'
                        )
                    parts.append('</table>')
            parts.append('</details>')

    parts.append('</div>')


def _render_outcome_analysis(
    parts: list[str],
    esc,
    *,
    optimization_summary: str | None = None,
    history: list[dict] | None = None,
    glance: GlanceData | None = None,
) -> None:
    """Render the What Worked / What Didn't analysis card."""
    rows = history or (glance.rows if glance else None) or []
    accepted_rows = [r for r in rows if r.get("accepted") and r.get("iter", r.get("iteration", 0)) > 0]
    rejected_rows = [r for r in rows if not r.get("accepted") and r.get("iter", r.get("iteration", 0)) > 0]

    has_content = optimization_summary or accepted_rows or rejected_rows
    if not has_content:
        return

    parts.append('<div class="card">')
    parts.append('<h2>Optimization analysis</h2>')

    # --- What worked ---
    if accepted_rows:
        parts.append('<details open><summary style="color:#16a34a;font-weight:700">What worked</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Iter</th><th>Speedup</th><th>Description</th></tr>')
        for r in accepted_rows:
            it = r.get("iter", r.get("iteration", "?"))
            spd = r.get("speedup", 1.0)
            notes = r.get("notes", r.get("description", ""))
            parts.append(
                f'<tr class="accepted">'
                f'<td>{esc(str(it))}</td>'
                f'<td>{spd:.2f}x</td>'
                f'<td>{esc(str(notes))}</td>'
                f'</tr>'
            )
        parts.append('</table>')
        parts.append('</details>')

    # --- What didn't work ---
    if rejected_rows:
        parts.append('<details><summary style="color:#b91c1c;font-weight:700">What didn\'t work</summary>')
        parts.append('<table class="iter-table">')
        parts.append('<tr><th>Iter</th><th>Description</th></tr>')
        for r in rejected_rows:
            it = r.get("iter", r.get("iteration", "?"))
            notes = r.get("notes", r.get("description", ""))
            parts.append(
                f'<tr>'
                f'<td>{esc(str(it))}</td>'
                f'<td>{esc(str(notes))}</td>'
                f'</tr>'
            )
        parts.append('</table>')
        parts.append('</details>')

    # --- LLM-generated explanation ---
    if optimization_summary:
        parts.append('<details open><summary style="font-weight:700">Why it worked</summary>')
        parts.append(f'<p class="explanation">{esc(optimization_summary)}</p>')
        parts.append('</details>')

    parts.append('</div>')


def _render_environment(parts: list[str], system_info: dict, esc) -> None:
    """Render an environment details/summary dropdown from system_info dict."""
    # Map system_info keys to display labels (ordered)
    field_map = [
        ("nvidia_gpus", "GPU"),
        ("driver_version", "Driver"),
        ("cuda_version", "CUDA"),
        ("torch_version", "PyTorch"),
        ("torch_cuda_version", "PyTorch CUDA"),
        ("tpu_chip", "TPU"),
        ("tpu_count", "TPU Chips"),
        ("jax_version", "JAX"),
        ("triton_version", "Triton"),
        ("cpp_compiler", "C++ Compiler"),
        ("openmp_version", "OpenMP"),
        ("cpu_model", "CPU"),
        ("cpu_count", "CPU Count"),
        ("python_version", "Python"),
        ("platform", "Platform"),
    ]
    # Build rows, skipping missing keys
    rows: list[tuple[str, str]] = []
    for key, label in field_map:
        val = system_info.get(key)
        if val is None:
            continue
        if key == "nvidia_gpus" and isinstance(val, list):
            for i, gpu in enumerate(val):
                name = gpu.get("name", "?")
                mem = gpu.get("memory_mib", "?")
                drv = gpu.get("driver_version", "")
                suffix = f" ({mem} MiB)" if mem != "?" else ""
                prefix = f"GPU {i}" if len(val) > 1 else "GPU"
                rows.append((prefix, f"{name}{suffix}"))
                if drv and not any(r[0] == "Driver" for r in rows):
                    rows.append(("Driver", drv))
        elif key == "driver_version":
            # Skip if already added from nvidia_gpus
            if not any(r[0] == "Driver" for r in rows):
                rows.append((label, str(val)))
        else:
            rows.append((label, str(val)))

    if not rows:
        return

    parts.append('<details class="card" style="cursor:pointer">')
    parts.append('<summary><h2 style="display:inline">Environment</h2></summary>')
    parts.append('<table class="iter-table" style="margin-top:8px">')
    for label, value in rows:
        parts.append(
            f'<tr><td style="font-weight:600;width:140px">{esc(label)}</td>'
            f'<td>{esc(value)}</td></tr>'
        )
    parts.append('</table>')
    parts.append('</details>')


def _render_bar_chart(
    parts: list[str],
    items: list[dict],
    esc,
    color: str = "#3b82f6",
) -> None:
    """Render a bar chart from items: [{name, pct, total_ms?, count?}]."""
    if not items:
        return
    max_pct = max(it.get("pct", 0) for it in items) or 1
    for it in items:
        pct = it.get("pct", 0)
        bar_width = pct / max_pct * 100
        name = it["name"]
        if len(name) > 45:
            name = "..." + name[-42:]
        meta_parts = []
        if "total_ms" in it:
            meta_parts.append(f"{it['total_ms']:.1f}ms")
        if "count" in it:
            meta_parts.append(f"x{it['count']}")
        meta = " ".join(meta_parts)
        parts.append(
            f'<div class="op-bar-container">'
            f'<span class="op-bar-label" title="{esc(it["name"])}">{esc(name)}</span>'
            f'<div class="op-bar-track"><div class="op-bar-fill" style="width: {bar_width:.1f}%; background: {color}"></div></div>'
            f'<span class="op-bar-pct">{pct:.1f}%</span>'
            f'<span class="op-bar-meta">{esc(meta)}</span>'
            f'</div>'
        )


def _metric_pill(
    parts: list[str],
    label: str,
    val_str: str,
    baseline_val: float | None,
    current_val: float | None,
    lower_is_better: bool = True,
) -> None:
    """Render a metric pill badge with optional delta from baseline."""
    delta_html = ""
    if baseline_val is not None and current_val is not None and baseline_val != 0:
        diff = current_val - baseline_val
        pct = diff / abs(baseline_val) * 100
        if abs(pct) >= 0.5:
            sign = "+" if pct > 0 else ""
            if lower_is_better:
                css_class = "good" if pct < 0 else "bad"
            else:
                css_class = "good" if pct > 0 else "bad"
            delta_html = f' <span class="delta {css_class}">{sign}{pct:.0f}%</span>'
    parts.append(
        f'<span class="metric-pill">'
        f'<span class="pill-label">{_html.escape(str(label))}</span>'
        f'<span class="pill-value">{_html.escape(str(val_str))}</span>'
        f'{delta_html}'
        f'</span>'
    )


def _fmt_ms(val: float | None) -> str:
    if val is None:
        return "n/a"
    return f"{val:.1f}ms"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "n/a"
    return f"{val:.1f}%"


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


def _summary_ok(s: dict | None) -> bool:
    """Check if a summary dict is present and successful."""
    return s is not None and s.get("returncode", 0) == 0


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


def _render_speedscope_link(parts: list[str], json_path: Path, esc, label: str = "Open in Speedscope") -> None:
    """Render a button linking to speedscope.dev with the local file path shown."""
    abs_path = esc(str(json_path.resolve()))
    parts.append(
        f'<div style="margin:8px 0;">'
        f'<a href="https://www.speedscope.app" target="_blank" rel="noopener" '
        f'style="display:inline-block;padding:8px 16px;background:#2563eb;color:#fff;'
        f'border-radius:6px;text-decoration:none;font-weight:600;font-size:0.9em;">'
        f'{esc(label)} &#x2197;</a>'
        f'<span style="margin-left:12px;font-size:0.82em;color:#666;">'
        f'Drag &amp; drop <code>{abs_path}</code> into the page</span>'
        f'</div>'
    )


def _render_pyspy_section(parts: list[str], prof: ProfilerData, esc) -> None:
    ps = prof.pyspy_summary
    bps = prof.baseline_pyspy_summary
    has_opt = ps and ps.get("returncode") == 0 and ps.get("hotspots")
    has_base = bps and bps.get("returncode") == 0 and bps.get("hotspots")
    has_speedscope = prof.speedscope_json_path and prof.speedscope_json_path.exists()
    has_base_speedscope = prof.baseline_speedscope_json_path and prof.baseline_speedscope_json_path.exists()

    parts.append('<h3>CPU profiler (py-spy)</h3>')

    # --- Hotspots (optimized) ---
    if has_opt:
        parts.append('<h4>CPU hotspots</h4>')
        _render_pyspy_hotspots(parts, ps, esc)

    # --- Interactive flame graph via speedscope ---
    if has_speedscope:
        parts.append('<h4>Interactive flame graph</h4>')
        _render_speedscope_link(parts, prof.speedscope_json_path, esc)

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
    if has_speedscope and has_base_speedscope:
        parts.append('<details><summary>Compare flame graphs with baseline</summary>')
        parts.append('<div class="compare-row">')
        parts.append('<div class="compare-col baseline"><h4>Baseline</h4>')
        _render_speedscope_link(parts, prof.baseline_speedscope_json_path, esc, label="Open baseline in Speedscope")
        parts.append('</div>')
        parts.append('<div class="compare-col optimized"><h4>Optimized</h4>')
        _render_speedscope_link(parts, prof.speedscope_json_path, esc, label="Open optimized in Speedscope")
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
# Metal section
# ---------------------------------------------------------------------------

def _render_metal_section(parts: list[str], prof: ProfilerData, esc) -> None:
    ms = prof.metal_summary
    bms = prof.baseline_metal_summary
    has_opt = _summary_ok(ms)
    has_base = _summary_ok(bms)

    parts.append('<h3>Metal GPU profiler</h3>')

    if has_opt:
        # Metric pills
        parts.append('<div class="metric-pills">')
        gpu_time = ms.get("gpu_time_total_ms")
        base_gpu_time = bms.get("gpu_time_total_ms") if has_base else None
        _metric_pill(parts, "GPU Time", _fmt_ms(gpu_time), base_gpu_time, gpu_time, lower_is_better=True)

        submissions = ms.get("gpu_submissions")
        base_submissions = bms.get("gpu_submissions") if has_base else None
        sub_str = str(submissions) if submissions is not None else "n/a"
        _metric_pill(parts, "Submissions", sub_str,
                     float(base_submissions) if base_submissions is not None else None,
                     float(submissions) if submissions is not None else None,
                     lower_is_better=True)

        gpu_idle = ms.get("gpu_idle_pct")
        base_idle = bms.get("gpu_idle_pct") if has_base else None
        _metric_pill(parts, "GPU Idle", _fmt_pct(gpu_idle), base_idle, gpu_idle, lower_is_better=True)
        parts.append('</div>')

        # Top submissions bar chart
        top_subs = ms.get("top_submissions", [])
        if top_subs:
            items = [
                {
                    "name": s.get("label", s.get("encoder_type", "?")),
                    "pct": s.get("gpu_time_ms", 0) / (gpu_time or 1) * 100 if gpu_time else 0,
                    "total_ms": s.get("gpu_time_ms", 0),
                }
                for s in top_subs[:10]
            ]
            parts.append('<h4>Top submissions</h4>')
            _render_bar_chart(parts, items, esc, color="#a78bfa")

        # GPU counters
        counters = ms.get("gpu_counters", {})
        if counters:
            parts.append('<div class="metric-pills" style="margin-top:8px">')
            base_counters = bms.get("gpu_counters", {}) if has_base else {}
            for key, label in [("alu_utilization", "ALU Util"), ("memory_bandwidth", "Mem BW"),
                               ("occupancy", "Occupancy"), ("gpu_active", "GPU Active")]:
                val = counters.get(key)
                bval = base_counters.get(key)
                if val is not None:
                    _metric_pill(parts, label, _fmt_pct(val), bval, val, lower_is_better=False)
            parts.append('</div>')

    # --- Comparison dropdown ---
    if has_opt and has_base:
        parts.append('<details><summary>Compare with baseline (Metal)</summary>')
        parts.append('<div class="compare-row">')
        parts.append('<div class="compare-col baseline"><h4>Baseline</h4>')
        base_subs = bms.get("top_submissions", [])
        base_gpu_time_val = bms.get("gpu_time_total_ms") or 1
        if base_subs:
            items = [
                {
                    "name": s.get("label", s.get("encoder_type", "?")),
                    "pct": s.get("gpu_time_ms", 0) / base_gpu_time_val * 100,
                    "total_ms": s.get("gpu_time_ms", 0),
                }
                for s in base_subs[:10]
            ]
            _render_bar_chart(parts, items, esc, color="#9ca3af")
        else:
            parts.append('<p style="color:#888">No submissions data</p>')
        parts.append('</div>')
        parts.append('<div class="compare-col optimized"><h4>Optimized</h4>')
        opt_subs = ms.get("top_submissions", [])
        opt_gpu_time_val = ms.get("gpu_time_total_ms") or 1
        if opt_subs:
            items = [
                {
                    "name": s.get("label", s.get("encoder_type", "?")),
                    "pct": s.get("gpu_time_ms", 0) / opt_gpu_time_val * 100,
                    "total_ms": s.get("gpu_time_ms", 0),
                }
                for s in opt_subs[:10]
            ]
            _render_bar_chart(parts, items, esc, color="#a78bfa")
        else:
            parts.append('<p style="color:#888">No submissions data</p>')
        parts.append('</div>')
        parts.append('</div>')
        parts.append('</details>')


# ---------------------------------------------------------------------------
# Nsys section
# ---------------------------------------------------------------------------

def _render_nsys_section(parts: list[str], prof: ProfilerData, esc) -> None:
    ns = prof.nsys_summary
    bns = prof.baseline_nsys_summary
    has_opt = _summary_ok(ns)
    has_base = _summary_ok(bns)

    parts.append('<h3>Nsight Systems (nsys)</h3>')

    if has_opt:
        # Metric pills
        parts.append('<div class="metric-pills">')
        kernel_time = ns.get("cuda_kernel_time_ms")
        base_kernel_time = bns.get("cuda_kernel_time_ms") if has_base else None
        _metric_pill(parts, "GPU Kernel Time", _fmt_ms(kernel_time), base_kernel_time, kernel_time, lower_is_better=True)

        memcpy_time = ns.get("memcpy_time_ms")
        base_memcpy_time = bns.get("memcpy_time_ms") if has_base else None
        _metric_pill(parts, "Memcpy Time", _fmt_ms(memcpy_time), base_memcpy_time, memcpy_time, lower_is_better=True)

        gpu_active = ns.get("gpu_active_pct")
        base_gpu_active = bns.get("gpu_active_pct") if has_base else None
        _metric_pill(parts, "GPU Active", _fmt_pct(gpu_active), base_gpu_active, gpu_active, lower_is_better=False)
        parts.append('</div>')

        # Top GPU kernels bar chart
        top_kernels = ns.get("top_kernels", [])
        if top_kernels:
            items = [
                {
                    "name": k.get("name", "?"),
                    "pct": k.get("pct", 0),
                    "total_ms": k.get("total_ms", 0),
                    "count": k.get("count", 0),
                }
                for k in top_kernels[:10]
            ]
            parts.append('<h4>Top GPU kernels</h4>')
            _render_bar_chart(parts, items, esc, color="#34d399")

        # Memory transfers
        memcpy_list = ns.get("memcpy", [])
        if memcpy_list:
            parts.append('<h4>Memory transfers</h4>')
            for mc in memcpy_list:
                direction = mc.get("direction", "?")
                count = mc.get("count", 0)
                total_bytes = mc.get("total_bytes", 0)
                total_ms = mc.get("total_ms", 0)
                size_str = f"{total_bytes / 1024 / 1024:.1f}MB" if total_bytes > 1024 * 1024 else f"{total_bytes / 1024:.1f}KB"
                parts.append(
                    f'<p style="font-size:0.85em; color:#666">'
                    f'{esc(direction)}: {count} transfers, {size_str}, {total_ms:.1f}ms</p>'
                )

    # --- Comparison dropdown ---
    if has_opt and has_base:
        parts.append('<details><summary>Compare with baseline (nsys)</summary>')
        parts.append('<div class="compare-row">')
        parts.append('<div class="compare-col baseline"><h4>Baseline</h4>')
        base_kernels = bns.get("top_kernels", [])
        if base_kernels:
            items = [
                {"name": k.get("name", "?"), "pct": k.get("pct", 0), "total_ms": k.get("total_ms", 0), "count": k.get("count", 0)}
                for k in base_kernels[:10]
            ]
            _render_bar_chart(parts, items, esc, color="#9ca3af")
        else:
            parts.append('<p style="color:#888">No kernel data</p>')
        parts.append('</div>')
        parts.append('<div class="compare-col optimized"><h4>Optimized</h4>')
        opt_kernels = ns.get("top_kernels", [])
        if opt_kernels:
            items = [
                {"name": k.get("name", "?"), "pct": k.get("pct", 0), "total_ms": k.get("total_ms", 0), "count": k.get("count", 0)}
                for k in opt_kernels[:10]
            ]
            _render_bar_chart(parts, items, esc, color="#34d399")
        else:
            parts.append('<p style="color:#888">No kernel data</p>')
        parts.append('</div>')
        parts.append('</div>')
        parts.append('</details>')


# ---------------------------------------------------------------------------
# Ncu section
# ---------------------------------------------------------------------------

def _render_ncu_section(parts: list[str], prof: ProfilerData, esc) -> None:
    nc = prof.ncu_summary
    bnc = prof.baseline_ncu_summary
    has_opt = _summary_ok(nc)
    has_base = _summary_ok(bnc)

    parts.append('<h3>Nsight Compute (ncu)</h3>')

    if has_opt:
        # Metric pills
        parts.append('<div class="metric-pills">')
        sm_util = nc.get("sm_utilization_pct")
        base_sm = bnc.get("sm_utilization_pct") if has_base else None
        _metric_pill(parts, "SM Util", _fmt_pct(sm_util), base_sm, sm_util, lower_is_better=False)

        occupancy = nc.get("achieved_occupancy_pct")
        base_occ = bnc.get("achieved_occupancy_pct") if has_base else None
        _metric_pill(parts, "Occupancy", _fmt_pct(occupancy), base_occ, occupancy, lower_is_better=False)

        mem_tp = nc.get("memory_throughput_pct")
        base_mem = bnc.get("memory_throughput_pct") if has_base else None
        _metric_pill(parts, "Mem Throughput", _fmt_pct(mem_tp), base_mem, mem_tp, lower_is_better=False)

        achieved_bw = nc.get("achieved_bw_gbs")
        base_bw = bnc.get("achieved_bw_gbs") if has_base else None
        if achieved_bw is not None:
            _metric_pill(parts, "Achieved BW", f"{achieved_bw:.1f} GB/s", base_bw, achieved_bw, lower_is_better=False)

        branch_eff = nc.get("branch_efficiency_pct")
        base_branch_eff = bnc.get("branch_efficiency_pct") if has_base else None
        if branch_eff is not None:
            _metric_pill(parts, "Branch Eff", _fmt_pct(branch_eff), base_branch_eff, branch_eff, lower_is_better=False)

        tc_util = nc.get("tensor_core_utilization_pct")
        base_tc = bnc.get("tensor_core_utilization_pct") if has_base else None
        if tc_util is not None:
            _metric_pill(parts, "TC Util", _fmt_pct(tc_util), base_tc, tc_util, lower_is_better=False)

        l1_hr = nc.get("l1_hit_rate")
        base_l1 = bnc.get("l1_hit_rate") if has_base else None
        if l1_hr is not None:
            _metric_pill(parts, "L1 Hit Rate", _fmt_pct(l1_hr), base_l1, l1_hr, lower_is_better=False)

        l2_hr = nc.get("l2_hit_rate")
        base_l2 = bnc.get("l2_hit_rate") if has_base else None
        if l2_hr is not None:
            _metric_pill(parts, "L2 Hit Rate", _fmt_pct(l2_hr), base_l2, l2_hr, lower_is_better=False)

        parts.append('</div>')

        # GPU cache hierarchy diagnosis
        if l1_hr is not None and l2_hr is not None:
            mem_tp_val = nc.get("memory_throughput_pct", 0)
            if l1_hr < 50:
                cache_diag = f'<span style="color:#e74c3c;font-weight:bold">L1 bottleneck</span> (L1 hit {l1_hr:.0f}% → tiles too large for shared mem)'
            elif l2_hr < 50 and mem_tp_val > 40:
                cache_diag = f'<span style="color:#e67e22;font-weight:bold">L2 bottleneck</span> (L1 {l1_hr:.0f}% OK, L2 hit {l2_hr:.0f}% → working set exceeds L2)'
            elif mem_tp_val > 70:
                cache_diag = f'<span style="color:#f39c12;font-weight:bold">DRAM saturated</span> (caches OK, bandwidth wall)'
            else:
                cache_diag = f'<span style="color:#27ae60">Healthy</span> (L1 {l1_hr:.0f}%, L2 {l2_hr:.0f}%)'
            parts.append(f'<p style="margin-top:8px;">GPU cache hierarchy: {cache_diag}</p>')

        # DRAM traffic breakdown
        dram_read = nc.get("dram_bytes_read_total")
        dram_write = nc.get("dram_bytes_written_total")
        if dram_read is not None or dram_write is not None:
            read_mb = (dram_read or 0) / 1024 / 1024
            write_mb = (dram_write or 0) / 1024 / 1024
            parts.append(
                f'<p style="font-size:0.85em; color:#666; margin-top:4px">'
                f'DRAM traffic: {read_mb:.1f} MB read, {write_mb:.1f} MB written</p>'
            )

        # Per-kernel table (top 8)
        kernels = nc.get("kernels", [])[:8]
        if kernels:
            parts.append('<h4>Top kernels</h4>')
            has_bw_col = any(k.get("achieved_bw_gbs") is not None for k in kernels)
            parts.append('<table class="iter-table">')
            bw_th = "<th>BW (GB/s)</th>" if has_bw_col else ""
            parts.append(f'<tr><th>Kernel</th><th>Invocations</th><th>SM%</th><th>Occupancy%</th><th>Mem%</th>{bw_th}</tr>')
            for k in kernels:
                name = k.get("name", "?")
                if len(name) > 40:
                    name = "..." + name[-37:]
                bw_td = f"<td>{k['achieved_bw_gbs']:.1f}</td>" if has_bw_col and k.get("achieved_bw_gbs") is not None else ("<td>-</td>" if has_bw_col else "")
                parts.append(
                    f'<tr>'
                    f'<td title="{esc(k.get("name", ""))}" style="font-family:monospace;font-size:0.85em">{esc(name)}</td>'
                    f'<td>{k.get("invocations", 0)}</td>'
                    f'<td>{k.get("sm_utilization_pct", 0):.1f}%</td>'
                    f'<td>{k.get("achieved_occupancy_pct", 0):.1f}%</td>'
                    f'<td>{k.get("memory_throughput_pct", 0):.1f}%</td>'
                    f'{bw_td}'
                    f'</tr>'
                )
            parts.append('</table>')

    # --- Comparison dropdown ---
    if has_opt and has_base:
        parts.append('<details><summary>Compare with baseline (ncu)</summary>')
        parts.append('<div class="compare-row">')

        # Baseline kernel table
        parts.append('<div class="compare-col baseline"><h4>Baseline</h4>')
        base_kernels = bnc.get("kernels", [])[:8]
        if base_kernels:
            parts.append('<table class="iter-table">')
            parts.append('<tr><th>Kernel</th><th>Inv</th><th>SM%</th><th>Occ%</th><th>Mem%</th></tr>')
            for k in base_kernels:
                name = k.get("name", "?")
                if len(name) > 30:
                    name = "..." + name[-27:]
                parts.append(
                    f'<tr>'
                    f'<td style="font-family:monospace;font-size:0.85em">{esc(name)}</td>'
                    f'<td>{k.get("invocations", 0)}</td>'
                    f'<td>{k.get("sm_utilization_pct", 0):.1f}%</td>'
                    f'<td>{k.get("achieved_occupancy_pct", 0):.1f}%</td>'
                    f'<td>{k.get("memory_throughput_pct", 0):.1f}%</td>'
                    f'</tr>'
                )
            parts.append('</table>')
        else:
            parts.append('<p style="color:#888">No kernel data</p>')
        parts.append('</div>')

        # Optimized kernel table
        parts.append('<div class="compare-col optimized"><h4>Optimized</h4>')
        opt_kernels = nc.get("kernels", [])[:8]
        if opt_kernels:
            parts.append('<table class="iter-table">')
            parts.append('<tr><th>Kernel</th><th>Inv</th><th>SM%</th><th>Occ%</th><th>Mem%</th></tr>')
            for k in opt_kernels:
                name = k.get("name", "?")
                if len(name) > 30:
                    name = "..." + name[-27:]
                parts.append(
                    f'<tr>'
                    f'<td style="font-family:monospace;font-size:0.85em">{esc(name)}</td>'
                    f'<td>{k.get("invocations", 0)}</td>'
                    f'<td>{k.get("sm_utilization_pct", 0):.1f}%</td>'
                    f'<td>{k.get("achieved_occupancy_pct", 0):.1f}%</td>'
                    f'<td>{k.get("memory_throughput_pct", 0):.1f}%</td>'
                    f'</tr>'
                )
            parts.append('</table>')
        else:
            parts.append('<p style="color:#888">No kernel data</p>')
        parts.append('</div>')

        parts.append('</div>')
        parts.append('</details>')


# ---------------------------------------------------------------------------
# JAX section
# ---------------------------------------------------------------------------

def _render_jax_section(parts: list[str], prof: ProfilerData, esc) -> None:
    js = prof.jax_summary
    bjs = prof.baseline_jax_summary
    has_opt = _summary_ok(js)
    has_base = _summary_ok(bjs)

    parts.append('<h3>JAX / XLA profiler</h3>')

    if has_opt:
        # Metric pills
        parts.append('<div class="metric-pills">')
        compilations = js.get("xla_compilations")
        base_compilations = bjs.get("xla_compilations") if has_base else None
        comp_str = str(compilations) if compilations is not None else "n/a"
        _metric_pill(parts, "XLA Compilations", comp_str,
                     float(base_compilations) if base_compilations is not None else None,
                     float(compilations) if compilations is not None else None,
                     lower_is_better=True)

        compile_time = js.get("xla_compilation_time_ms")
        base_compile_time = bjs.get("xla_compilation_time_ms") if has_base else None
        _metric_pill(parts, "Compile Time", _fmt_ms(compile_time), base_compile_time, compile_time, lower_is_better=True)

        recomps = js.get("xla_recompilations")
        base_recomps = bjs.get("xla_recompilations") if has_base else None
        recomp_str = str(recomps) if recomps is not None else "n/a"
        _metric_pill(parts, "Recompilations", recomp_str,
                     float(base_recomps) if base_recomps is not None else None,
                     float(recomps) if recomps is not None else None,
                     lower_is_better=True)
        parts.append('</div>')

        # HLO module count
        hlo_count = js.get("hlo_module_count")
        if hlo_count is not None:
            parts.append(
                f'<p style="font-size:0.85em; color:#666; margin-top:4px">'
                f'HLO modules: {hlo_count}</p>'
            )

        # HLO ops table
        hlo_ops = js.get("hlo_ops", [])
        if hlo_ops:
            parts.append('<h4>Top HLO operations</h4>')
            parts.append('<table class="iter-table">')
            parts.append('<tr><th>Operation</th><th>Count</th></tr>')
            for op in hlo_ops[:10]:
                parts.append(
                    f'<tr>'
                    f'<td style="font-family:monospace;font-size:0.85em">{esc(str(op.get("op", "?")))}</td>'
                    f'<td>{op.get("count", 0)}</td>'
                    f'</tr>'
                )
            parts.append('</table>')


# ---------------------------------------------------------------------------
# TPU section (rendered within JAX when TPU data is present)
# ---------------------------------------------------------------------------

def _render_tpu_section(parts: list[str], prof: ProfilerData, esc) -> None:
    js = prof.jax_summary
    if not _summary_ok(js):
        return
    # Only render if TPU-specific data is present
    tpu_chip = js.get("tpu_chip")
    has_tpu = (
        tpu_chip is not None
        or js.get("mxu_utilization_pct") is not None
        or js.get("device_time_us") is not None
    )
    if not has_tpu:
        return

    parts.append('<h3>TPU device metrics</h3>')

    # Device info header
    tpu_count = js.get("tpu_count", 1)
    if tpu_chip:
        parts.append(
            f'<p style="font-size:0.85em; color:#666;">'
            f'{esc(str(tpu_chip))} &mdash; {tpu_count} chip{"s" if tpu_count > 1 else ""}'
            f'</p>'
        )

    # Metric pills
    parts.append('<div class="metric-pills">')

    mxu_util = js.get("mxu_utilization_pct")
    if mxu_util is not None:
        _metric_pill(parts, "MXU Utilization", f"{mxu_util:.1f}%",
                     None, mxu_util, lower_is_better=False)

    device_frac = js.get("device_fraction")
    if device_frac is not None:
        _metric_pill(parts, "Device Active", f"{device_frac * 100:.1f}%",
                     None, device_frac * 100, lower_is_better=False)

    infeed = js.get("infeed_stall_pct")
    if infeed is not None:
        _metric_pill(parts, "Infeed Stall", f"{infeed:.1f}%",
                     None, infeed, lower_is_better=True)

    parts.append('</div>')

    # Host-device time breakdown
    host_us = js.get("host_time_us")
    dev_us = js.get("device_time_us")
    if host_us is not None and dev_us is not None and (host_us + dev_us) > 0:
        total = host_us + dev_us
        h_pct = host_us / total * 100
        d_pct = dev_us / total * 100
        parts.append('<h4>Host vs Device time</h4>')
        parts.append(
            '<div style="display:flex;height:24px;border-radius:6px;overflow:hidden;'
            'margin:4px 0 8px 0;">'
        )
        if h_pct > 0:
            parts.append(
                f'<div style="width:{h_pct:.1f}%;background:#60a5fa;" '
                f'title="Host: {host_us/1e3:.1f}ms ({h_pct:.1f}%)"></div>'
            )
        if d_pct > 0:
            parts.append(
                f'<div style="width:{d_pct:.1f}%;background:#34d399;" '
                f'title="Device: {dev_us/1e3:.1f}ms ({d_pct:.1f}%)"></div>'
            )
        parts.append('</div>')
        parts.append(
            '<p style="font-size:0.8em;color:#888;">'
            f'<span style="color:#60a5fa">&#9632;</span> Host {h_pct:.1f}% '
            f'<span style="color:#34d399">&#9632;</span> Device {d_pct:.1f}%'
            '</p>'
        )

    # HLO cost metrics (FLOPS and bytes from XLA cost annotations)
    hlo_tflops = js.get("hlo_cost_tflops")
    hlo_bytes = js.get("hlo_cost_bytes_accessed")
    if hlo_tflops is not None or hlo_bytes is not None:
        parts.append('<h4>XLA Cost Estimate</h4>')
        cost_parts = []
        if hlo_tflops is not None:
            cost_parts.append(f'<strong>{hlo_tflops:.4f} TFLOPS</strong> (estimated from HLO)')
        if hlo_bytes is not None:
            bytes_mb = hlo_bytes / (1024 * 1024)
            cost_parts.append(f'{bytes_mb:.1f} MB bytes accessed')
        if hlo_tflops is not None and hlo_bytes is not None and hlo_bytes > 0:
            hlo_flops = js.get("hlo_cost_flops", hlo_tflops * 1e12)
            ai = hlo_flops / hlo_bytes
            cost_parts.append(f'AI = {ai:.1f} FLOP/byte')
        parts.append(f'<p>{" | ".join(cost_parts)}</p>')


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
