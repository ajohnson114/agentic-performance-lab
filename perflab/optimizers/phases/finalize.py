from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from perflab.analyzers.metrics_rollup import calc_speedup
from perflab.memory.run_store import RunStore, load_profiler_summaries
from perflab.optimizers.history import make_history_entry
from perflab.optimizers.progress import fmt_elapsed
from perflab.reporting.generate import ReportParams, generate_reports
from perflab.task_spec import TaskSpec

if TYPE_CHECKING:
    from perflab.optimizers.agent import AgentContext

logger = logging.getLogger(__name__)


def _resolve_roofline_peaks(task: TaskSpec) -> dict | None:
    """Resolve roofline peaks: use task.yaml config if set, else auto-detect."""
    from perflab.roofline_peaks import resolve_roofline
    return resolve_roofline(task)


def _regenerate_roofline_with_history(ctx: AgentContext) -> None:
    """Regenerate roofline.png with the full optimization trail from ctx.history.

    Only runs if the task has a roofline spec and at least one history entry
    has roofline data. Overwrites the last per-iteration PNG so the dashboard
    picks up the trail automatically.
    """
    task = ctx.task
    if task.roofline is None:
        return

    trail = [e for e in ctx.history if "roofline_ai" in e and "roofline_tflops" in e]
    if not trail:
        return

    # Gold star = best accepted iteration, not just last profiled entry.
    # Fall back to trail[-1] if best_iter has no roofline data (e.g. NCU skipped that iter).
    best_entries = [e for e in trail if e.get("iteration") == ctx.best_iter]
    final = best_entries[0] if best_entries else trail[-1]
    roof_png = ctx.rp.artifacts_dir / "roofline.png"

    try:
        from perflab.reporting.roofline import RooflinePoint, write_roofline_png
        from perflab.roofline_peaks import _lookup_dtype_peaks, _lookup_l2_bw

        pt = RooflinePoint(
            ai=final["roofline_ai"],
            tflops=final["roofline_tflops"],
            gbs=final["roofline_tflops"] * 1000.0 / max(final["roofline_ai"], 1e-9),
        )
        write_roofline_png(
            roof_png,
            point=pt,
            peak_tflops=float(task.roofline.peak_tflops),
            peak_mem_bw_gbs=float(task.roofline.peak_mem_bw_gbs),
            title=task.roofline.title or f"Roofline — {task.name}",
            dtype_peaks=_lookup_dtype_peaks(task.target_hardware or ""),
            l2_bw_gbs=_lookup_l2_bw(task.target_hardware or ""),
            history_points=trail,  # full trail; gold star renders on top of final dot
        )
    except Exception:  # noqa: BLE001 -- best-effort report artifact, must not block report generation
        logger.warning("Failed to regenerate roofline with history trail", exc_info=True)


def generate_optimization_summary(
    ctx: AgentContext,
    device: str = "unknown",
    profiler_summaries: dict[str, dict] | None = None,
) -> tuple[str | None, dict, float]:
    """Call LLM to generate a concise explanation of what was optimized and why.

    Returns (summary_text, usage_dict, latency_seconds).
    """
    from perflab.llm.base import Message

    provider = ctx.provider
    task = ctx.task
    baseline_val = ctx.baseline_val
    best_value = ctx.best_value
    accepted_patches = ctx.accepted_patches

    speedup = calc_speedup(best_value, baseline_val)
    patches_desc = []
    for p in accepted_patches:
        patches_desc.append(f"- Iter {p['iteration']}: {p['description']}")
        if p.get("reasoning"):
            patches_desc.append(f"  Reasoning: {p['reasoning'][:300]}")

    # Build profiler context so the LLM doesn't hallucinate hardware claims
    profiler_context = f"Device: {device}\n"
    if profiler_summaries:
        torch_summary = profiler_summaries.get("torch_profiler", {})
        cpu_gpu = torch_summary.get("cpu_vs_gpu", {})
        cpu_us = cpu_gpu.get("total_cpu_op_us", 0)
        gpu_us = cpu_gpu.get("total_gpu_kernel_us", 0)
        if cpu_us > 0 or gpu_us > 0:
            profiler_context += f"CPU time: {cpu_us/1000:.1f}ms, GPU kernel time: {gpu_us/1000:.1f}ms\n"
        if device == "mps":
            profiler_context += (
                "Note: This runs on Apple Silicon (MPS). The torch profiler shows 0 GPU "
                "kernel time because MPS Metal operations are not visible to the profiler. "
                "Do NOT claim GPU utilization improvements — say 'device' or 'MPS' instead.\n"
            )

    user_content = (
        "You previously optimized this code. Here is a summary of what was changed:\n\n"
        "Accepted patches:\n" + "\n".join(patches_desc) + "\n\n"
        f"{profiler_context}\n"
        f"Before: {baseline_val:.6g} {task.benchmark.metric.name}\n"
        f"After: {best_value:.6g} {task.benchmark.metric.name}\n"
        f"Speedup: {speedup:.2f}x\n\n"
        f"Write a concise but comprehensive technical explanation structured as:\n"
        f"1. WHAT WORKED: For each accepted patch, explain what optimization was applied "
        f"and the technical reason it improved performance (e.g., reduced cache misses, "
        f"enabled vectorization, eliminated redundant computation).\n"
        f"2. WHY IT WORKED: Explain the underlying performance principle — what hardware "
        f"or algorithmic bottleneck was addressed.\n\n"
        f"Reference specific code changes. "
        f"Be accurate about the hardware — only mention GPU if profiler data confirms GPU activity."
    )
    messages = [
        Message(role="system", content="You are a performance engineering expert. Write concise technical summaries."),
        Message(role="user", content=user_content),
    ]
    t0 = time.monotonic()
    result = provider.complete(messages, temperature=0.3, max_tokens=512)
    latency = time.monotonic() - t0
    text = result.content.strip() if result.content else None
    return text, result.usage, latency


def maybe_early_stop(ctx: AgentContext, it: int) -> bool:
    """Check the convergence detector; if it says to stop, record the reason and return True.

    The caller (agent.py's iteration loop) is responsible for breaking out of
    the loop when this returns True -- only the loop driver controls loop flow.
    """
    if not ctx.convergence:
        return False
    should_stop, reason = ctx.convergence.should_stop()
    if not should_stop:
        return False
    ctx.early_stop_reason = reason
    ctx.progress.on_message(f"[agent] {reason}")
    ctx.event_log.early_stop(it, reason)
    ctx.history.append(make_history_entry(
        it, f"early stop: {reason}", ctx.best_value, ctx.baseline_val,
        accepted=False,
    ))
    return True


def run(ctx: AgentContext) -> None:
    """Finalize phase: optimization summary, reports, and run-completion bookkeeping.

    Runs after the iteration loop exits. Mutates ctx.total_llm_calls/latency/
    tokens (from the optimization-summary LLM call).
    """
    task = ctx.task
    rp = ctx.rp
    progress = ctx.progress
    event_log = ctx.event_log
    llm_config = ctx.llm_config

    # --- Post-optimization explanation ---
    optimization_summary_text: str | None = None
    bench_data_final = {}
    bench_json = rp.run_dir / "bench.json"
    if bench_json.exists():
        bench_data_final = json.loads(bench_json.read_text(encoding="utf-8"))
    profiler_summaries_final = load_profiler_summaries(rp.artifacts_dir)

    if ctx.best_value != ctx.baseline_val and ctx.accepted_patches:
        try:
            optimization_summary_text, summary_usage, summary_latency = generate_optimization_summary(
                ctx,
                device=bench_data_final.get("meta", {}).get("device", "unknown"),
                profiler_summaries=profiler_summaries_final,
            )
            ctx.total_llm_calls += 1
            ctx.total_llm_latency += summary_latency
            ctx.total_input_tokens += summary_usage.get("input_tokens") or summary_usage.get("prompt_tokens", 0)
            ctx.total_output_tokens += summary_usage.get("output_tokens") or summary_usage.get("completion_tokens", 0)
            if optimization_summary_text:
                (rp.run_dir / "optimization_summary.md").write_text(
                    optimization_summary_text, encoding="utf-8",
                )
        except Exception as exc:  # noqa: BLE001 -- best-effort LLM summary, must not block report generation
            progress.on_message(f"[agent] Failed to generate optimization summary: {exc}")

    event_log.run_complete(ctx.best_value, ctx.best_iter, ctx.baseline_val, ctx.config.max_iters, ctx.total_llm_calls)

    # --- Generate reports ---
    progress.on_message("\n[agent] Generating reports...")
    llm_stats = {
        "model": llm_config.model,
        "provider": llm_config.provider,
        "total_calls": ctx.total_llm_calls,
        "total_input_tokens": ctx.total_input_tokens,
        "total_output_tokens": ctx.total_output_tokens,
        "total_llm_latency_s": ctx.total_llm_latency,
    }
    # Determine detected GPU name for hardware mismatch reporting
    detected_hw: str | None = None
    if ctx.sysinfo.get("nvidia_gpus"):
        detected_hw = ctx.sysinfo["nvidia_gpus"][0].get("name")

    # Regenerate the roofline PNG with the full optimization trail before reporting
    _regenerate_roofline_with_history(ctx)

    generate_reports(ReportParams(
        run_dir=rp.run_dir,
        run_id=rp.run_id,
        task_name=task.name,
        metric_name=task.benchmark.metric.name,
        metric_mode=task.benchmark.metric.mode,
        program_type=task.program_type,
        history=ctx.history,
        baseline_val=ctx.baseline_val,
        best_value=ctx.best_value,
        best_iter=ctx.best_iter,
        early_stop_reason=ctx.early_stop_reason,
        optimization_summary_text=optimization_summary_text,
        analysis_thresholds=task.analysis_thresholds,
        accepted_patches=ctx.accepted_patches,
        roofline_peaks=_resolve_roofline_peaks(task),
        llm_stats=llm_stats,
        target_hardware=task.target_hardware,
        detected_hardware=detected_hw,
        build_cmd=task.build.cmd if task.build else None,
        secondary_metric_name=task.benchmark.secondary_metric.name if task.benchmark.secondary_metric else None,
        secondary_metric_mode=task.benchmark.secondary_metric.mode if task.benchmark.secondary_metric else None,
        top_n=task.constraints.top_n,
        user_actions=ctx.user_actions or None,
    ))

    # Update meta with completion status
    RunStore(task.out_dir).update_meta(rp.run_id, {
        "status": "completed",
        "best_value": ctx.best_value,
        "baseline_value": ctx.baseline_val,
        "completed_at": time.strftime("%Y%m%d-%H%M%S"),
        "program_type": task.program_type,
    })

    # --- Surface user actions (build suggestions the LLM couldn't apply) ---
    if ctx.user_actions:
        (rp.run_dir / "user_actions.json").write_text(
            json.dumps(ctx.user_actions, indent=2), encoding="utf-8",
        )
        progress.on_message("\n[agent] === USER ACTION REQUIRED ===")
        progress.on_message("[agent] The optimizer suggested build/compilation changes that require manual task.yaml updates:")
        for a in ctx.user_actions:
            progress.on_message(f"[agent]   • {a['suggestion']} (iter {a['iteration']})")
        progress.on_message(f"[agent] Details saved to {rp.run_dir / 'user_actions.json'}")

    total_wall = time.monotonic() - ctx.wall_start
    progress.on_message(f"[agent] Done in {fmt_elapsed(total_wall)}. Best {task.benchmark.metric.name} = {ctx.best_value:.6g} at iter {ctx.best_iter}")
    progress.on_message(f"[agent] Reports: {rp.run_dir}")
