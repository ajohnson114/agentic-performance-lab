from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from perflab.analyzers.bench_stats import compute_bench_stats, extract_repeated_values
from perflab.analyzers.bottleneck_analyzer import AnalysisThresholds, diagnose_bottlenecks
from perflab.analyzers.diff_flamegraph import compute_diff_stacks, generate_diff_svg
from perflab.analyzers.gpu_attribution import compute_attribution_ranking
from perflab.analyzers.hlo_attribution import compute_hlo_attribution
from perflab.analyzers.metrics_rollup import compute_run_summary
from perflab.analyzers.profile_diff import compute_hotspot_diff, compute_profile_diff
from perflab.analyzers.vectorization import check_vectorization_from_perf_annotate
from perflab.reporting.dashboard_html import (
    AnalysisData,
    GlanceData,
    ProfilerData,
    write_dashboard_html,
)
from perflab.reporting.plots import (
    compute_pareto_frontier,
    plot_metric_history,
    plot_pareto_frontier,
)
from perflab.reporting.report_md import write_report_md

logger = logging.getLogger(__name__)


@dataclass
class ReportParams:
    """Parameter object for generate_reports().

    Replaces the 19+ keyword arguments with a single structured object.
    Used by both the agent loop and the knob-search orchestrator.
    """

    # --- Required fields ---
    run_dir: Path
    run_id: str
    task_name: str
    metric_name: str
    metric_mode: str
    program_type: str
    history: list[dict]
    baseline_val: float
    best_value: float
    best_iter: int

    # --- Optional fields ---
    early_stop_reason: str | None = None
    optimization_summary_text: str | None = None
    analysis_thresholds: AnalysisThresholds | None = None
    accepted_patches: list[dict] | None = None
    roofline_peaks: dict | None = None
    llm_stats: dict | None = None
    target_hardware: str | None = None
    detected_hardware: str | None = None
    hardware_mismatch: str | None = None
    build_cmd: str | None = None
    secondary_metric_name: str | None = None
    secondary_metric_mode: str | None = None
    top_n: int = 3
    user_actions: list[dict] | None = None


def _build_microarch_for_dashboard(
    final_summaries: dict, bench_data: dict | None,
) -> dict | None:
    """Build micro-architecture summary for dashboard display."""
    try:
        from perflab.analyzers.microarch import build_microarch_summary
        return build_microarch_summary(bench_data or {}, final_summaries)
    except Exception:  # noqa: BLE001 -- best-effort dashboard section, must not abort report generation
        logger.warning("Microarch summary build failed", exc_info=True)
        return None


def _compute_hardware_mismatch_fallback(
    target_hardware: str | None,
    detected_hardware: str | None,
    system_info: dict | None,
) -> str | None:
    """Fallback hardware-mismatch recompute for ReportParams call sites that
    don't already provide a resolved ``hardware_mismatch`` (see
    ``optimizers/phases/baseline.py._check_hardware_mismatch``, which is the
    preferred source -- it loops every GPU and is threaded through by
    finalize.py).

    Checks every GPU in *system_info* (not just the first one) before
    declaring a mismatch; falls back to the single *detected_hardware* name
    only when system_info has no GPU list at all.
    """
    if not target_hardware:
        return None
    t_low = target_hardware.lower()
    gpus = (system_info or {}).get("nvidia_gpus") or []
    names = [g.get("name", "") for g in gpus if g.get("name")]
    if not names and detected_hardware:
        names = [detected_hardware]
    if not names:
        return None
    for name in names:
        d_low = name.lower()
        if t_low in d_low or d_low in t_low:
            return None
    return (
        f'Hardware mismatch: configured for "{target_hardware}" '
        f'but running on "{names[0]}"'
    )


def generate_reports(p: ReportParams) -> dict:
    """Generate all reports (markdown, HTML dashboard, JSON) for a completed run.

    Returns the report_data dict that was written to report.json.
    """
    run_dir = p.run_dir
    artifacts_dir = run_dir / "artifacts"

    # --- Unpack frequently used fields ---
    run_id = p.run_id
    task_name = p.task_name
    metric_name = p.metric_name
    metric_mode = p.metric_mode
    program_type = p.program_type
    history = p.history
    baseline_val = p.baseline_val
    best_value = p.best_value
    best_iter = p.best_iter
    early_stop_reason = p.early_stop_reason
    optimization_summary_text = p.optimization_summary_text
    analysis_thresholds = p.analysis_thresholds
    accepted_patches = p.accepted_patches
    roofline_peaks = p.roofline_peaks
    llm_stats = p.llm_stats
    target_hardware = p.target_hardware
    detected_hardware = p.detected_hardware
    hardware_mismatch_hint = p.hardware_mismatch
    build_cmd = p.build_cmd
    secondary_metric_name = p.secondary_metric_name
    secondary_metric_mode = p.secondary_metric_mode
    top_n = p.top_n
    user_actions = p.user_actions

    # --- Metric history plot ---
    metric_hist = run_dir / "metric_history.png"
    iters_list = [h["iteration"] for h in history]
    vals_list = [h["value"] for h in history]
    plot_metric_history(metric_hist, iters_list, vals_list, metric_name, baseline_val=baseline_val)

    # --- Pareto frontier plot (optional, when secondary metric available) ---
    pareto_png_rel: str | None = None
    if secondary_metric_name:
        pareto_points = []
        for h in history:
            sec_val = h.get("secondary_value")
            if sec_val is not None:
                pareto_points.append({
                    "primary": h["value"],
                    "secondary": sec_val,
                    "label": f"iter {h['iteration']}",
                })
        if len(pareto_points) >= 2:
            frontier = compute_pareto_frontier(
                pareto_points,
                primary_mode=metric_mode,
                secondary_mode=secondary_metric_mode or "minimize",
            )
            pareto_png = run_dir / "pareto_frontier.png"
            plot_pareto_frontier(
                pareto_png, pareto_points, frontier,
                primary_name=metric_name,
                secondary_name=secondary_metric_name,
                primary_mode=metric_mode,
                secondary_mode=secondary_metric_mode or "minimize",
            )
            if pareto_png.exists():
                pareto_png_rel = str(pareto_png.relative_to(run_dir))

    # --- Load final artifacts ---
    artifacts: dict[str, str] = {}
    if artifacts_dir.exists():
        for f in artifacts_dir.iterdir():
            if f.is_file():
                artifacts[f.stem] = str(f)

    # --- Compute run summary ---
    run_summary = compute_run_summary(history, baseline_val, metric_mode)

    # --- Bottleneck diagnoses from latest profiler summaries ---
    final_summaries: dict[str, dict] = {}
    if artifacts_dir.exists():
        for sf in artifacts_dir.glob("*_summary.json"):
            try:
                final_summaries[sf.stem.replace("_summary", "")] = json.loads(
                    sf.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load profiler summary %s", sf, exc_info=True)
    # Try to read device from bench.json for MPS-aware diagnosis
    report_device: str | None = None
    _bench_data_for_microarch: dict | None = None
    bench_json_path = run_dir / "bench.json"
    if bench_json_path.exists():
        try:
            _bench_data_for_microarch = json.loads(bench_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load bench.json for microarch", exc_info=True)
    if bench_json_path.exists():
        try:
            report_device = json.loads(
                bench_json_path.read_text(encoding="utf-8")
            ).get("meta", {}).get("device")
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read device from bench.json", exc_info=True)
    # Load system_info.json for CPU count etc.
    report_system_info: dict | None = None
    system_info_path = run_dir / "system_info.json"
    if system_info_path.exists():
        try:
            report_system_info = json.loads(
                system_info_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load system_info.json", exc_info=True)
    final_diags = diagnose_bottlenecks(
        final_summaries, program_type, device=report_device,
        top_n=top_n, thresholds=analysis_thresholds,
        system_info=report_system_info,
    ) if final_summaries else []

    # --- Assemble report data ---
    # Normalize history entries: support both "notes" and "description" keys
    report_data: dict = {
        "task_name": task_name,
        "run_id": run_id,
        "metric_name": metric_name,
        "metric_mode": metric_mode,
        "best_value": best_value,
        "best_iter": best_iter,
        "baseline_value": baseline_val,
        "rows": [
            {
                "iter": h["iteration"],
                "value": h["value"],
                "accepted": h["accepted"],
                "notes": h.get("notes", h.get("description", "")),
                "delta": h.get("delta"),
                "speedup": h.get("speedup"),
                "bench_wall_time_s": h.get("bench_wall_time_s"),
                "profiling_overhead_pct": h.get("profiling_overhead_pct"),
            }
            for h in history
        ],
        "bottleneck_diagnoses": [
            {
                "rank": d.rank,
                "bottleneck": d.bottleneck,
                "root_cause": d.root_cause,
                "confidence": d.confidence,
                "suggested_actions": d.suggested_actions,
            }
            for d in final_diags
        ],
        "run_summary": {
            "baseline_value": run_summary.baseline_value,
            "best_value": run_summary.best_value,
            "median_speedup": run_summary.median_speedup,
            "p90_speedup": run_summary.p90_speedup,
            "time_to_first_improvement": run_summary.time_to_first_improvement,
            "success_rate": run_summary.success_rate,
            "total_iterations": run_summary.total_iterations,
        },
        "latest_artifacts": artifacts,
    }

    if early_stop_reason:
        report_data["early_stop_reason"] = early_stop_reason
    if optimization_summary_text:
        report_data["optimization_summary"] = optimization_summary_text
    if roofline_peaks:
        report_data["roofline_peaks"] = roofline_peaks

    # --- Machine fingerprint ---
    if report_system_info:
        report_data["hardware"] = {
            k: report_system_info[k]
            for k in ("cpu_model", "cpu_count", "machine", "platform",
                       "nvidia_gpus", "cuda_version", "cpp_compiler",
                       "tpu_chip", "tpu_count", "tpu_devices")
            if k in report_system_info
        }

    # --- Benchmark noise analysis ---
    bench_stats_data: dict | None = None
    bench_stats_warning: str = ""
    if bench_json_path.exists():
        try:
            bench_blob = json.loads(bench_json_path.read_text(encoding="utf-8"))
            raw_vals = extract_repeated_values(bench_blob, metric_name)
            if raw_vals:
                stats = compute_bench_stats(raw_vals)
                if stats:
                    bench_stats_data = {
                        "n": stats.n,
                        "mean": stats.mean,
                        "std": stats.std,
                        "cv": stats.cv,
                        "ci_95": [stats.ci_95_low, stats.ci_95_high],
                        "is_noisy": stats.is_noisy,
                    }
                    report_data["bench_stats"] = bench_stats_data
                    if stats.is_noisy:
                        bench_stats_warning = stats.warning
        except (json.JSONDecodeError, OSError, ValueError):
            logger.warning("Bench stats computation failed", exc_info=True)

    # --- Build at-a-glance data ---
    accepted_count = sum(1 for h in history if h.get("accepted") and h.get("iteration", 0) > 0)
    if metric_mode == "minimize":
        speedup = baseline_val / best_value if best_value != 0 else 1.0
    else:
        speedup = best_value / baseline_val if baseline_val != 0 else 1.0
    glance = GlanceData(
        metric_name=metric_name,
        baseline_value=baseline_val,
        best_value=best_value,
        best_iter=best_iter,
        total_iterations=run_summary.total_iterations,
        speedup=speedup,
        accepted_count=accepted_count,
        early_stop_reason=early_stop_reason,
        rows=[
            {
                "iter": h["iteration"],
                "value": h["value"],
                "speedup": h.get("speedup", 1.0),
                "accepted": h.get("accepted", False),
                "notes": h.get("notes", h.get("description", "")),
            }
            for h in history
            # Skip duplicate early-stop entries
            if not str(h.get("description", "")).startswith("early stop:")
        ],
        accepted_patches=accepted_patches or [],
        llm_model=llm_stats.get("model", "") if llm_stats else "",
        llm_provider=llm_stats.get("provider", "") if llm_stats else "",
        llm_total_calls=llm_stats.get("total_calls", 0) if llm_stats else 0,
        llm_total_input_tokens=llm_stats.get("total_input_tokens", 0) if llm_stats else 0,
        llm_total_output_tokens=llm_stats.get("total_output_tokens", 0) if llm_stats else 0,
        llm_total_latency_s=llm_stats.get("total_llm_latency_s", 0.0) if llm_stats else 0.0,
    )

    # --- Wire TFLOPS + % of peak into at-a-glance ---
    if bench_json_path.exists():
        try:
            bench_data = json.loads(bench_json_path.read_text(encoding="utf-8"))
            tflops_data = bench_data.get("tflops", {})
            if isinstance(tflops_data, dict) and tflops_data.get("median") is not None:
                glance.achieved_tflops = float(tflops_data["median"])
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass
    if roofline_peaks and roofline_peaks.get("peak_tflops"):
        glance.peak_tflops = float(roofline_peaks["peak_tflops"])
    if roofline_peaks and roofline_peaks.get("peak_mem_bw_gbs"):
        glance.peak_mem_bw_gbs = float(roofline_peaks["peak_mem_bw_gbs"])
    if roofline_peaks and roofline_peaks.get("source"):
        glance.roofline_source = roofline_peaks["source"]
    if roofline_peaks and roofline_peaks.get("device"):
        glance.roofline_device = roofline_peaks["device"]

    # Wire achieved bandwidth from ncu summary
    ncu_final = final_summaries.get("ncu", {})
    if ncu_final.get("achieved_bw_gbs") is not None:
        glance.achieved_bw_gbs = float(ncu_final["achieved_bw_gbs"])

    # --- Load baseline profiler summaries ---
    baseline_artifacts_dir = run_dir / "artifacts_baseline"
    baseline_summaries: dict[str, dict] = {}
    if baseline_artifacts_dir.exists():
        for sf in baseline_artifacts_dir.glob("*_summary.json"):
            try:
                baseline_summaries[sf.stem.replace("_summary", "")] = json.loads(
                    sf.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load baseline summary %s", sf, exc_info=True)

    # --- GPU attribution ---
    gpu_attrib = None
    nsys_final = final_summaries.get("nsys", {})
    if nsys_final.get("cpu_gpu_correlations"):
        try:
            gpu_attrib = compute_attribution_ranking(nsys_final)
        except Exception:  # noqa: BLE001 -- best-effort report section, must not abort the whole report
            logger.warning("GPU attribution computation failed", exc_info=True)

    # --- HLO attribution (JAX/TPU) ---
    hlo_attrib = None
    jax_final = final_summaries.get("jax", {})
    if jax_final.get("hlo_ops"):
        try:
            hlo_attrib = compute_hlo_attribution(jax_final, jax_final)
        except Exception:  # noqa: BLE001 -- best-effort report section, must not abort the whole report
            logger.warning("HLO attribution computation failed", exc_info=True)

    # --- Profile diff (baseline vs final) ---
    profile_diff = None
    hotspot_diff = None
    if baseline_summaries and final_summaries:
        try:
            profile_diff = compute_profile_diff(baseline_summaries, final_summaries, metric_mode)
        except Exception:  # noqa: BLE001 -- best-effort report section, must not abort the whole report
            logger.warning("Profile diff computation failed", exc_info=True)
        try:
            hotspot_diff = compute_hotspot_diff(baseline_summaries, final_summaries)
        except Exception:  # noqa: BLE001 -- best-effort report section, must not abort the whole report
            logger.warning("Hotspot diff computation failed", exc_info=True)

    # --- Differential flame graph ---
    diff_flame_svg_path = None
    if baseline_summaries and final_summaries:
        try:
            diffs = compute_diff_stacks(baseline_summaries, final_summaries)
            if diffs:
                diff_svg = artifacts_dir / "diff_flamegraph.svg"
                diff_flame_svg_path = generate_diff_svg(
                    diffs, diff_svg,
                    title=f"Differential Flame Graph — {task_name}",
                )
        except Exception:  # noqa: BLE001 -- best-effort report section, must not abort the whole report
            logger.warning("Differential flame graph generation failed", exc_info=True)

    # --- Vectorization analysis ---
    vectorization_data: list[dict] | None = None
    perf_annotate_path = artifacts_dir / "perf_annotate.txt"
    if perf_annotate_path.exists():
        try:
            annotate_text = perf_annotate_path.read_text(encoding="utf-8", errors="replace")
            vec_summary = check_vectorization_from_perf_annotate(annotate_text)
            if vec_summary.functions:
                vectorization_data = [
                    {
                        "function": f.function,
                        "has_simd": f.has_simd,
                        "simd_isa": f.simd_isa,
                        "hot_pct": f.hot_pct,
                    }
                    for f in vec_summary.functions
                ]
                if vec_summary.warning:
                    report_data["vectorization_warning"] = vec_summary.warning
        except (OSError, ValueError):
            logger.warning("Vectorization analysis failed", exc_info=True)

    # --- Build flag recommendations (static ISA + dynamic profiler-driven) ---
    build_flag_recs = None
    if build_cmd:
        try:
            from perflab.analyzers.build_flags import (
                recommend_build_flags,
                recommend_flags_from_profiling,
            )
            all_recs = []
            cpu_isa = (report_system_info or {}).get("cpu_isa")
            if cpu_isa:
                all_recs.extend(recommend_build_flags(build_cmd, cpu_isa, program_type))
            # Dynamic: recommendations based on profiler output
            prof_recs = recommend_flags_from_profiling(
                build_cmd, final_summaries, program_type, cpu_isa=cpu_isa,
            )
            all_recs.extend(prof_recs)
            # Deduplicate by flag
            seen: set[str] = set()
            deduped = []
            for r in all_recs:
                if r.flag not in seen:
                    seen.add(r.flag)
                    deduped.append(r)
            if deduped:
                build_flag_recs = deduped
        except Exception:  # noqa: BLE001 -- best-effort report section, must not abort the whole report
            logger.warning("Build flag recommendation failed", exc_info=True)

    # --- Build profiler data for dashboard embedding ---
    profiler_data = ProfilerData(
        torch_summary=final_summaries.get("torch_profiler"),
        pyspy_summary=final_summaries.get("pyspy"),
        metal_summary=final_summaries.get("metal_trace"),
        nsys_summary=final_summaries.get("nsys"),
        ncu_summary=final_summaries.get("ncu"),
        jax_summary=final_summaries.get("jax"),
        memray_summary=final_summaries.get("memray"),
        baseline_torch_summary=baseline_summaries.get("torch_profiler"),
        baseline_pyspy_summary=baseline_summaries.get("pyspy"),
        baseline_metal_summary=baseline_summaries.get("metal_trace"),
        baseline_nsys_summary=baseline_summaries.get("nsys"),
        baseline_ncu_summary=baseline_summaries.get("ncu"),
        baseline_jax_summary=baseline_summaries.get("jax"),
    )
    roofline_png = artifacts_dir / "roofline.png"
    if roofline_png.exists():
        profiler_data.roofline_png_path = roofline_png
    if diff_flame_svg_path and diff_flame_svg_path.exists():
        profiler_data.diff_flame_svg_path = diff_flame_svg_path
    speedscope_json = artifacts_dir / "pyspy_speedscope.json"
    if speedscope_json.exists() and speedscope_json.stat().st_size > 0:
        profiler_data.speedscope_json_path = speedscope_json
    baseline_speedscope_json = baseline_artifacts_dir / "pyspy_speedscope.json"
    if baseline_speedscope_json.exists() and baseline_speedscope_json.stat().st_size > 0:
        profiler_data.baseline_speedscope_json_path = baseline_speedscope_json
    torch_trace = artifacts_dir / "torch_trace.json"
    if torch_trace.exists():
        profiler_data.torch_trace_path = torch_trace

    # --- Perfetto trace export ---
    try:
        from perflab.profilers.perfetto_export import export_perfetto_trace
        perfetto_path = export_perfetto_trace(
            run_dir / "perfetto_trace.json",
            pyspy_summary=final_summaries.get("pyspy"),
            perf_summary=final_summaries.get("linux_perf"),
            memray_summary=final_summaries.get("memray"),
            metadata={"task_name": task_name, "run_id": run_id},
        )
        if perfetto_path:
            profiler_data.perfetto_trace_path = perfetto_path
    except Exception:  # noqa: BLE001 -- best-effort report section, must not abort the whole report
        logger.warning("Perfetto trace export failed", exc_info=True)

    # --- Hardware mismatch detection ---
    # Prefer the caller's already-resolved value (e.g. finalize.py passes
    # ctx.hardware_mismatch, computed by baseline.py._check_hardware_mismatch
    # looping every detected GPU) when provided. Otherwise fall back to a
    # local recompute for callers that don't have one (e.g. orchestrator.py's
    # profile-only / knob-search paths) -- also looping every GPU in
    # system_info, not just index 0.
    hw_mismatch = hardware_mismatch_hint
    if hw_mismatch is None and target_hardware:
        hw_mismatch = _compute_hardware_mismatch_fallback(
            target_hardware, detected_hardware, report_system_info,
        )

    # --- Write reports ---
    write_report_md(run_dir / "report.md", report_data)
    write_dashboard_html(
        run_dir / "dashboard.html",
        title=f"PerfLab — {task_name} ({run_id})",
        metric_png_rel=str(metric_hist.relative_to(run_dir)),
        optimization_summary=optimization_summary_text,
        glance=glance,
        profiler=profiler_data,
        analysis=AnalysisData(
            bottleneck_diagnoses=[
                {
                    "rank": d.rank,
                    "bottleneck": d.bottleneck,
                    "root_cause": d.root_cause,
                    "confidence": d.confidence,
                    "suggested_actions": d.suggested_actions,
                }
                for d in final_diags
            ] if final_diags else None,
            gpu_attribution=[
                {
                    "rank": a.rank,
                    "name": a.name,
                    "category": a.category,
                    "gpu_pct": a.gpu_pct,
                    "gpu_time_ms": a.gpu_time_ms,
                    "diagnosis": a.diagnosis,
                    "suggestions": a.suggestions,
                }
                for a in gpu_attrib
            ] if gpu_attrib else None,
            profile_diff=[
                {
                    "metric": d.metric,
                    "before": d.before,
                    "after": d.after,
                    "delta_pct": d.delta_pct,
                    "direction": d.direction,
                }
                for d in profile_diff
            ] if profile_diff else None,
            build_flag_recs=[
                {
                    "flag": r.flag,
                    "reason": r.reason,
                    "impact": r.impact,
                    "category": r.category,
                }
                for r in build_flag_recs
            ] if build_flag_recs else None,
            hotspot_diff=[
                {
                    "function": s.function,
                    "before_pct": s.before_pct,
                    "after_pct": s.after_pct,
                    "delta_pct": s.delta_pct,
                    "status": s.status,
                }
                for s in hotspot_diff
            ] if hotspot_diff else None,
            history=history,
            tma_data=final_summaries.get("linux_perf", {}).get("tma"),
            tma_level2_data=final_summaries.get("linux_perf", {}).get("tma_level2"),
            power_data=final_summaries.get("power"),
            vectorization=vectorization_data,
            gpu_memory=final_summaries.get("power", {}).get("gpu_memory"),
            thread_sched=final_summaries.get("thread_sched"),
            ebpf_data=final_summaries.get("ebpf"),
            lock_contention_data=final_summaries.get("lock_contention"),
            hlo_attribution=[
                {
                    "op": e.op,
                    "count": e.count,
                    "pct_of_ops": e.pct_of_ops,
                    "category": e.category,
                    "estimated_device_pct": e.estimated_device_pct,
                    "diagnosis": e.diagnosis,
                    "suggestions": e.suggestions,
                }
                for e in hlo_attrib.entries
            ] if hlo_attrib else None,
            user_actions=user_actions,
            microarch_summary=_build_microarch_for_dashboard(final_summaries, _bench_data_for_microarch),
            torch_flops=final_summaries.get("torch_profiler", {}),
        ),
        system_info=report_system_info,
        hardware_mismatch=hw_mismatch,
        pareto_png_rel=pareto_png_rel,
        bench_stats_warning=bench_stats_warning,
    )
    (run_dir / "report.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )

    return report_data
