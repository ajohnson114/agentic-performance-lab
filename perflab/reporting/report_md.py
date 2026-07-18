from __future__ import annotations

from pathlib import Path


def write_report_md(report_path: Path, data: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render(data), encoding="utf-8")


def _cell(value: object) -> str:
    """Escape a value for a Markdown table cell.

    notes/bottleneck/root_cause carry LLM- and candidate-derived text; a
    literal '|' or newline would break the table row it lands in.
    """
    return str(value).replace("|", "\\|").replace("\n", " ")

def _render(data: dict) -> str:
    lines = []
    lines.append(f"# PerfLab Report — {data.get('task_name','')}")
    lines.append("")
    lines.append(f"Run ID: `{data.get('run_id','')}`")
    lines.append("")

    # Best metric
    lines.append("## Best metric")
    lines.append(f"- Metric: `{data.get('metric_name')}` ({data.get('metric_mode')})")
    lines.append(f"- Best value: **{data.get('best_value')}** at iteration **{data.get('best_iter')}**")
    baseline = data.get("baseline_value")
    if baseline is not None:
        best = data.get("best_value")
        if baseline != 0 and best is not None:
            speedup = best / baseline
            lines.append(f"- Baseline -> Best: {baseline:.6g} -> {best:.6g} ({speedup:.2f}x)")
    lines.append("")

    # Profiling overhead (from baseline row if available)
    for row in data.get("rows", []):
        overhead = row.get("profiling_overhead_pct")
        if overhead is not None:
            lines.append(f"Profiling overhead: ~{overhead:.1f}% of benchmark wall time")
            lines.append("")
            break

    # Roofline peaks (source tier: table / computed / measured / etc.)
    roofline = data.get("roofline_peaks")
    if roofline and roofline.get("peak_tflops") is not None:
        lines.append("## Roofline")
        lines.append("")
        bw = roofline.get("peak_mem_bw_gbs")
        bw_str = f", {bw:.1f} GB/s" if bw is not None else ""
        lines.append(f"- Peak: {roofline['peak_tflops']:.3f} TFLOPS{bw_str}")
        source = roofline.get("source")
        if source:
            device = roofline.get("device")
            device_str = f" ({device})" if device else ""
            lines.append(f"- Source: {source}{device_str}")
        lines.append("")

    # Bottleneck diagnosis
    diags = data.get("bottleneck_diagnoses", [])
    if diags:
        lines.append("## Bottleneck diagnosis")
        lines.append("")
        lines.append("| Rank | Bottleneck | Root cause | Confidence |")
        lines.append("|---:|---|---|:---:|")
        for d in diags:
            lines.append(f"| {d.get('rank', '?')} | {_cell(d.get('bottleneck', ''))} | {_cell(d.get('root_cause', ''))} | {_cell(d.get('confidence', ''))} |")
        lines.append("")

    # Iterations table with before/after columns
    lines.append("## Iterations")
    lines.append("")
    has_speedup = any(row.get("speedup") is not None for row in data.get("rows", []))
    if has_speedup:
        lines.append("| iter | value | delta | speedup | accepted | notes |")
        lines.append("|---:|---:|---:|---:|:---:|---|")
        for row in data.get("rows", []):
            delta = row.get("delta")
            speedup = row.get("speedup")
            delta_str = f"{delta:+.6g}" if delta is not None else ""
            speedup_str = f"{speedup:.2f}x" if speedup is not None else ""
            accepted_str = "yes" if row["accepted"] else ""
            lines.append(f"| {row['iter']} | {row['value']:.6g} | {delta_str} | {speedup_str} | {accepted_str} | {_cell(row.get('notes',''))} |")
    else:
        lines.append("| iter | value | accepted | notes |")
        lines.append("|---:|---:|:---:|---|")
        for row in data.get("rows", []):
            accepted_str = "yes" if row["accepted"] else ""
            lines.append(f"| {row['iter']} | {row['value']:.6g} | {accepted_str} | {_cell(row.get('notes',''))} |")
    lines.append("")

    # Run summary
    summary = data.get("run_summary")
    if summary:
        lines.append("## Run summary")
        lines.append("")
        lines.append(f"- Baseline: {summary.get('baseline_value', '?'):.6g}")
        lines.append(f"- Best: {summary.get('best_value', '?'):.6g}")
        lines.append(f"- Median speedup: {summary.get('median_speedup', 1.0):.2f}x")
        lines.append(f"- P90 speedup: {summary.get('p90_speedup', 1.0):.2f}x")
        ttfi = summary.get("time_to_first_improvement")
        lines.append(f"- Time to first improvement: {'iter ' + str(ttfi) if ttfi is not None else 'N/A'}")
        lines.append(f"- Success rate: {summary.get('success_rate', 0.0):.0%}")
        lines.append(f"- Total iterations: {summary.get('total_iterations', 0)}")
        lines.append("")

    # Early stop reason
    early_stop = data.get("early_stop_reason")
    if early_stop:
        lines.append("## Early stop")
        lines.append(f"{early_stop}")
        lines.append("")

    # Optimization explanation
    opt_summary = data.get("optimization_summary")
    if opt_summary:
        lines.append("## Optimization explanation")
        lines.append("")
        lines.append(opt_summary)
        lines.append("")

    # Artifacts
    lines.append("## Artifacts (latest)")
    for k, v in data.get("latest_artifacts", {}).items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    return "\n".join(lines)
