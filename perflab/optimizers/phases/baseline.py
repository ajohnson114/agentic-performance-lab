from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING

from perflab.memory.run_store import load_profiler_summaries, snapshot_workspace
from perflab.optimizers.history import make_history_entry
from perflab.optimizers.progress import fmt_elapsed
from perflab.runners.benchmark import metric_value
from perflab.runners.pipeline import run_pipeline_for_ctx

if TYPE_CHECKING:
    from perflab.optimizers.agent import AgentContext

logger = logging.getLogger(__name__)


def _check_hardware_mismatch(
    target_hardware: str | None,
    system_info: dict,
) -> str | None:
    """Check if target_hardware matches any detected GPU.

    Returns a mismatch message string if there's a mismatch, or None if OK.
    """
    if not target_hardware:
        return None
    gpus = system_info.get("nvidia_gpus") or []
    if not gpus:
        return None
    target_lower = target_hardware.lower()
    for gpu in gpus:
        gpu_name = (gpu.get("name") or "").lower()
        if target_lower in gpu_name or gpu_name in target_lower:
            return None
    detected_name = gpus[0].get("name", "unknown GPU")
    return (
        f'Hardware mismatch: task targets "{target_hardware}" '
        f'but detected GPU is "{detected_name}"'
    )


def run(ctx: AgentContext) -> None:
    """Baseline phase: capture system info, run the baseline profile + benchmark,
    snapshot the workspace, and load any prior-run context.

    Mutates ctx.sysinfo, ctx.hardware_mismatch, ctx.baseline_val, ctx.best_value,
    ctx.latest_diagnostics, ctx.sec_metric, ctx.baseline_sec_val, ctx.history,
    and ctx.prior_run_context.
    """
    task = ctx.task
    progress = ctx.progress
    rp = ctx.rp
    event_log = ctx.event_log

    # System info capture
    sysinfo: dict = {}
    try:
        from perflab.tools.sysinfo import capture_system_info, warn_if_noisy
        sysinfo = capture_system_info(rp.run_dir)
        for warning in warn_if_noisy():
            progress.on_message(f"[agent] WARNING: {warning}")
    except Exception as exc:  # noqa: BLE001 -- best-effort system info capture, must not block the run
        progress.on_message(f"[agent] Failed to collect system info: {exc}")
    ctx.sysinfo = sysinfo

    # Hardware mismatch check
    if task.target_hardware and sysinfo:
        ctx.hardware_mismatch = _check_hardware_mismatch(task.target_hardware, sysinfo)
        if ctx.hardware_mismatch:
            progress.on_message(f"[agent] ⚠ {ctx.hardware_mismatch}")
            progress.on_message("[agent]   Roofline analysis and optimization hints may be inaccurate.")
            progress.on_message("[agent]   Set target_hardware to null for auto-detection, or update it to match your hardware.")

    # Save resolved configuration for run reproducibility
    try:
        from perflab.config import load_config
        resolved_cfg = load_config()
        resolved_cfg.save(rp.run_dir / "resolved_config.json")
    except Exception:  # noqa: BLE001 -- best-effort reproducibility artifact, must not block the run
        logger.debug("Failed to save resolved config", exc_info=True)

    # --- Baseline profile + benchmark ---
    progress.on_message("[agent] Baseline run...")
    bench_base, bench_wall, prof_wall, latest_diagnostics = run_pipeline_for_ctx(
        ctx, do_profiles=True, capture_diagnostics=True,
    )
    ctx.baseline_val = metric_value(bench_base, task.benchmark.metric.name)
    ctx.best_value = ctx.baseline_val
    ctx.latest_diagnostics = latest_diagnostics

    # Secondary metric tracking (optional, for Pareto optimization)
    ctx.sec_metric = task.benchmark.secondary_metric
    if ctx.sec_metric:
        try:
            ctx.baseline_sec_val = metric_value(bench_base, ctx.sec_metric.name)
        except (KeyError, TypeError):
            ctx.sec_metric = None  # Not available in bench.json, disable

    # value == baseline here, so the computed delta/speedup are exactly 0.0/1.0
    # (calc_speedup returns 1.0 for a zero baseline).
    ctx.history.append(make_history_entry(
        0, "baseline", ctx.baseline_val, ctx.baseline_val,
        accepted=True,
        secondary_value=ctx.baseline_sec_val,
        bench_wall_time_s=bench_wall,
        profiling_overhead_pct=(
            (prof_wall - bench_wall) / bench_wall * 100
            if bench_wall is not None and prof_wall is not None and bench_wall > 0
            else None
        ),
    ))
    progress.on_message(f"[agent] Baseline {task.benchmark.metric.name} = {ctx.baseline_val:.6g}")
    _bench_s = f"bench={fmt_elapsed(bench_wall)}" if bench_wall is not None else ""
    _prof_s = f", profiling={fmt_elapsed(prof_wall)}" if prof_wall is not None else ""
    if _bench_s:
        progress.on_message(f"[agent] Baseline timing: {_bench_s}{_prof_s}")

    # Snapshot baseline code
    snapshot_workspace(task, rp.run_dir, "baseline")

    profiler_summaries = load_profiler_summaries(rp.artifacts_dir)

    # Preserve baseline artifacts before they get overwritten by re-profiling
    baseline_artifacts_dir = rp.run_dir / "artifacts_baseline"
    if rp.artifacts_dir.exists() and not baseline_artifacts_dir.exists():
        shutil.copytree(rp.artifacts_dir, baseline_artifacts_dir)

    event_log.baseline_complete(
        ctx.baseline_val, bench_base, list(profiler_summaries.keys()),
    )

    # --- Cross-run learning: load prior run context ---
    try:
        from perflab.optimizers.cross_run import load_prior_run_context
        ctx.prior_run_context = load_prior_run_context(task.out_dir, current_run_id=rp.run_id)
        if ctx.prior_run_context:
            progress.on_message("[agent] Loaded context from prior runs")
    except Exception:  # noqa: BLE001 -- best-effort cross-run context, must not block the run
        logger.warning("Failed to load prior run context", exc_info=True)
