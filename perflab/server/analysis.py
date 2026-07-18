"""MCP tools for on-demand analysis of stored run data."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from perflab.server.core import _guard_output_size, _to_dicts, mcp

# ===========================================================================
# Analysis — on-demand from stored run data
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": True})
def get_bottlenecks(run_id: str, out_dir: str = "out") -> list[dict]:
    """Load profiler summaries for a run and diagnose bottlenecks."""
    from perflab.analyzers.bottleneck_analyzer import diagnose_bottlenecks
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    run_data = store.get_run(run_id)
    program_type = run_data.get("meta", {}).get("program_type", "python")
    summaries = run_data.get("profiler_summaries", {})

    if not summaries:
        return []

    device = run_data.get("meta", {}).get("device")
    # Load system_info for CPU count etc.
    run_dir = Path(out_dir) / "runs" / run_id
    mcp_system_info: dict | None = None
    system_info_path = run_dir / "system_info.json"
    if system_info_path.exists():
        try:
            mcp_system_info = json.loads(
                system_info_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            pass
    diags = diagnose_bottlenecks(summaries, program_type, device=device, system_info=mcp_system_info)
    return [
        {
            "rank": d.rank,
            "bottleneck": d.bottleneck,
            "root_cause": d.root_cause,
            "confidence": d.confidence,
            "suggested_actions": d.suggested_actions,
        }
        for d in diags
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def get_gpu_attribution(run_id: str, out_dir: str = "out") -> dict:
    """Compute GPU attribution ranking for a run: which kernels consume the most GPU time, CPU→GPU call graph, and pipeline stalls."""
    from perflab.analyzers.gpu_attribution import compute_attribution_ranking
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    run_data = store.get_run(run_id)
    summaries = run_data.get("profiler_summaries", {})

    nsys_summary = summaries.get("nsys") or summaries.get("nsys_profiler")
    if not nsys_summary:
        return {"error": "No NSys profiler data found for this run. GPU attribution requires CUDA workloads profiled with Nsight Systems."}

    perf_summary = summaries.get("linux_perf") or summaries.get("perf")
    entries = compute_attribution_ranking(nsys_summary, perf_summary)

    return _guard_output_size({
        "run_id": run_id,
        "attribution": _to_dicts(entries),
    })


@mcp.tool(annotations={"readOnlyHint": True})
def get_profile_diff(run_a: str, run_b: str, out_dir: str = "out") -> dict:
    """Compare profiler metrics between two runs: IPC, cache misses, GPU utilization, and function-level hotspot shifts."""
    from perflab.analyzers.profile_diff import compute_hotspot_diff, compute_profile_diff
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    data_a = store.get_run(run_a)
    data_b = store.get_run(run_b)

    summaries_a = data_a.get("profiler_summaries", {})
    summaries_b = data_b.get("profiler_summaries", {})

    if not summaries_a or not summaries_b:
        return {"error": "Both runs must have profiler summaries for comparison."}

    metric_mode = data_a.get("meta", {}).get("metric_mode", "maximize")
    deltas = compute_profile_diff(summaries_a, summaries_b, metric_mode=metric_mode)
    hotspots = compute_hotspot_diff(summaries_a, summaries_b)

    return _guard_output_size({
        "run_a": run_a,
        "run_b": run_b,
        "deltas": _to_dicts(deltas),
        "hotspot_shifts": _to_dicts(hotspots),
    })


@mcp.tool(annotations={"readOnlyHint": True})
def get_hlo_attribution(run_id: str, out_dir: str = "out") -> dict:
    """Compute HLO operation attribution for a JAX/TPU run: op rankings, cost estimates, dtype distribution, and optimization suggestions."""
    from perflab.analyzers.hlo_attribution import compute_hlo_attribution
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    run_data = store.get_run(run_id)
    summaries = run_data.get("profiler_summaries", {})

    jax_summary = summaries.get("jax") or summaries.get("jax_profiler")
    if not jax_summary:
        return {"error": "No JAX profiler data found for this run. HLO attribution requires JAX workloads."}

    result = compute_hlo_attribution(jax_summary)
    if result is None:
        return {"error": "Could not compute HLO attribution (no HLO operation data in JAX summary)."}

    return _guard_output_size(dataclasses.asdict(result))


@mcp.tool(annotations={"readOnlyHint": True})
def get_build_recommendations(task_yaml: str, run_id: str | None = None, out_dir: str = "out") -> dict:
    """Get build flag recommendations based on ISA detection and profiler data.

    Without run_id: static ISA-based recommendations from the build command.
    With run_id: also includes profiler-driven recommendations (cache miss rates, TMA data, etc.).
    """
    from perflab.analyzers.build_flags import recommend_build_flags, recommend_flags_from_profiling
    from perflab.task_spec import TaskSpec

    task = TaskSpec.load(Path(task_yaml))
    build_cmd = task.build.cmd if task.build else ""
    if not build_cmd:
        return {"error": "Task has no build command — build recommendations only apply to compiled languages (C++, CUDA)."}

    cpu_isa: dict = {}
    profiler_summaries: dict = {}

    if run_id:
        from perflab.memory.run_store import validate_run_id
        try:
            validate_run_id(run_id)
        except ValueError as exc:
            return {"error": str(exc)}
        run_dir = Path(out_dir) / "runs" / run_id
        sys_path = run_dir / "system_info.json"
        if sys_path.exists():
            try:
                sys_info = json.loads(sys_path.read_text(encoding="utf-8"))
                cpu_isa = sys_info.get("cpu_isa", {})
            except (json.JSONDecodeError, OSError):
                pass

        from perflab.memory.run_store import RunStore
        store = RunStore(Path(out_dir))
        run_data = store.get_run(run_id)
        profiler_summaries = run_data.get("profiler_summaries", {})

    # Static ISA-based recommendations
    static_recs = recommend_build_flags(build_cmd, cpu_isa, task.program_type)

    # Profiler-driven recommendations (if run data available)
    dynamic_recs = []
    if profiler_summaries:
        dynamic_recs = recommend_flags_from_profiling(
            build_cmd, profiler_summaries, task.program_type, cpu_isa=cpu_isa,
        )

    return {
        "build_cmd": build_cmd,
        "program_type": task.program_type,
        "isa_recommendations": _to_dicts(static_recs),
        "profiler_recommendations": _to_dicts(dynamic_recs),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def get_roofline_analysis(run_id: str, out_dir: str = "out") -> dict:
    """Compute roofline analysis for a run: arithmetic intensity, achieved TFLOPS, peak utilization %, and memory bandwidth."""
    from perflab.memory.run_store import RunStore
    from perflab.reporting.roofline import compute_roofline_point
    from perflab.roofline_peaks import infer_peaks

    store = RunStore(Path(out_dir))
    run_data = store.get_run(run_id)

    bench = run_data.get("bench", {})
    if not bench:
        return {"error": "No benchmark data found for this run."}

    summaries = run_data.get("profiler_summaries", {})

    # Try to get profiler FLOPS
    profiler_flops = None
    for key in ("torch", "pytorch_profiler", "torch_profiler"):
        s = summaries.get(key, {})
        if s.get("total_flops"):
            profiler_flops = s["total_flops"]
            break

    # Try to get measured DRAM bytes from NCU
    measured_dram = None
    for key in ("ncu", "ncu_profiler"):
        s = summaries.get(key, {})
        if s.get("dram_bytes"):
            measured_dram = s["dram_bytes"]
            break

    point = compute_roofline_point(bench, measured_dram_bytes=measured_dram, profiler_flops=profiler_flops)
    if point is None:
        return {"error": "Could not compute roofline point (bench.json lacks flops/bytes data or meta.M/N/K)."}

    result: dict = {
        "run_id": run_id,
        "roofline_point": dataclasses.asdict(point),
    }

    peaks = infer_peaks("auto")
    if peaks:
        result["peaks"] = {
            "device": peaks.device,
            "source": peaks.source,
            "peak_tflops": peaks.peak_tflops,
            "peak_mem_bw_gbs": peaks.peak_mem_bw_gbs,
        }
        if peaks.dtype_peaks:
            result["peaks"]["dtype_peaks"] = peaks.dtype_peaks
        if peaks.peak_tflops > 0 and point.tflops > 0:
            result["pct_of_peak"] = round(point.tflops / peaks.peak_tflops * 100, 1)

    return result


@mcp.tool(annotations={"readOnlyHint": True})
def get_thresholds(task_yaml: str | None = None) -> dict:
    """List analysis thresholds used for bottleneck diagnosis. Shows defaults and task-specific overrides."""
    from perflab.analyzers.bottleneck_analyzer import AnalysisThresholds

    defaults = AnalysisThresholds()
    effective = defaults

    if task_yaml:
        from perflab.task_spec import TaskSpec
        task = TaskSpec.load(Path(task_yaml))
        effective = task.analysis_thresholds

    fields_data = {}
    for f in dataclasses.fields(AnalysisThresholds):
        val = getattr(effective, f.name)
        def_val = getattr(defaults, f.name)
        entry: dict = {"value": val}
        if val != def_val:
            entry["default"] = def_val
            entry["overridden"] = True
        fields_data[f.name] = entry

    overridden = sum(1 for v in fields_data.values() if v.get("overridden"))
    return {
        "thresholds": fields_data,
        "total_fields": len(fields_data),
        "overridden_count": overridden,
    }
