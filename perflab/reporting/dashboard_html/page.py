"""Top-level dashboard entrypoint: HTML skeleton, CSS, at-a-glance, orchestration."""
from __future__ import annotations

import base64
import html as _html
from pathlib import Path

from perflab.llm.pricing import format_cost_usd

from .data import AnalysisData, GlanceData, ProfilerData
from .diagnostics import (
    _render_diagnostics,
    _render_environment,
    _render_outcome_analysis,
    _render_user_actions,
)
from .profiler_sections import _render_profiler


def write_dashboard_html(
    path: Path,
    title: str,
    metric_png_rel: str | None,
    optimization_summary: str | None = None,
    glance: GlanceData | None = None,
    profiler: ProfilerData | None = None,
    analysis: AnalysisData | None = None,
    system_info: dict | None = None,
    hardware_mismatch: str | None = None,
    pareto_png_rel: str | None = None,
    bench_stats_warning: str = "",
) -> None:
    # Unpack the analyzer/diagnostic payloads (rendering logic below is unchanged).
    a = analysis if analysis is not None else AnalysisData()
    bottleneck_diagnoses = a.bottleneck_diagnoses
    gpu_attribution = a.gpu_attribution
    profile_diff = a.profile_diff
    build_flag_recs = a.build_flag_recs
    hotspot_diff = a.hotspot_diff
    history = a.history
    tma_data = a.tma_data
    tma_level2_data = a.tma_level2_data
    power_data = a.power_data
    vectorization = a.vectorization
    gpu_memory = a.gpu_memory
    thread_sched = a.thread_sched
    ebpf_data = a.ebpf_data
    lock_contention_data = a.lock_contention_data
    hlo_attribution = a.hlo_attribution
    user_actions = a.user_actions
    microarch_summary = a.microarch_summary
    torch_flops = a.torch_flops

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
            # Show the tier explicitly (e.g. "peaks: table/computed/measured") so a
            # measured/estimated roofline is never mistaken for a spec-sheet one.
            tier_label = f"peaks: {esc(glance.roofline_source)}"
            device_label = esc(glance.roofline_device) if glance.roofline_device else ""
            detail = f"{device_label}<br>{tier_label}" if device_label else tier_label
            parts.append(f'<div class="stat"><div class="label">Roofline</div><div class="value neutral" style="font-size:0.9em">{glance.peak_tflops:.3f} TFLOPS<br><small>{detail}</small></div></div>')
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
            ("Est. Cost", format_cost_usd(glance.llm_estimated_cost_usd)),
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
            parts.append('<h3>Kernel Performance Ceiling</h3>')
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
            parts.append('<h3>Benchmark Stability</h3>')
            parts.append(f'<p style="color:{color}">{esc(stability.get("assessment", ""))}</p>')

        throttle = microarch_summary.get("clock_throttle")
        if throttle and throttle.get("throttle_detected"):
            parts.append('<h3>GPU Thermal Status</h3>')
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
