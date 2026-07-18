from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from perflab.optimizers.history import make_history_entry

if TYPE_CHECKING:
    from perflab.optimizers.agent import AgentContext

logger = logging.getLogger(__name__)


def _auto_tune_sweep(
    ctx: AgentContext,
    max_trials: int = 15,
) -> float | None:
    """Run a parameter sweep on tuning.yaml if it has a sweep section.

    Returns the new best value if the sweep found something better,
    or None if no sweep was run or no improvement was found.
    """
    task = ctx.task
    progress = ctx.progress
    event_log = ctx.event_log
    iteration = ctx.iteration
    current_best = ctx.best_value

    knobs_path = task.workspace / "tuning.yaml"
    if not knobs_path.exists():
        return None

    original_knobs: dict | None = None
    try:
        from perflab.optimizers.propose_params import (
            generate_sweep_candidates,
            load_knobs,
            sample_candidates,
            save_knobs,
        )
        from perflab.runners.benchmark import run_benchmark
        from perflab.runners.correctness import run_correctness

        knobs = load_knobs(knobs_path)
        if not knobs.get("sweep"):
            return None

        candidates = generate_sweep_candidates(knobs)
        if not candidates:
            return None

        # Limit trials
        if len(candidates) > max_trials:
            candidates = sample_candidates(candidates, max_trials)

        progress.on_message(f"[agent] Auto-tuning: sweeping {len(candidates)} parameter combinations...")

        best_sweep_value = current_best
        best_sweep_knobs: dict | None = None
        original_knobs = dict(knobs)  # captured for restore on error/no-improvement

        for i, cand in enumerate(candidates):
            # Write candidate knobs
            save_knobs(knobs_path, cand.new_knobs)

            # Build if needed
            if task.build:
                import shlex

                from perflab.tools.shell import run_cmd
                build_res = run_cmd(shlex.split(task.build.cmd), cwd=task.workspace)
                if build_res.returncode != 0:
                    continue

            # Correctness check
            cres = run_correctness(
                task.correctness.cmd, cwd=task.workspace,
                program_type=task.program_type,
                rlimit_as_gb=task.constraints.rlimit_as_gb,
                isolation=ctx.config.isolation,
            )
            if cres.returncode != task.correctness.expected_exit:
                continue

            # Benchmark
            _, bench_data = run_benchmark(
                task.benchmark.cmd, cwd=task.workspace,
                program_type=task.program_type,
                rlimit_as_gb=task.constraints.rlimit_as_gb,
                isolation=ctx.config.isolation,
            )

            from perflab.runners.benchmark import metric_value
            value = metric_value(bench_data, task.benchmark.metric.name)

            is_better = (
                (task.benchmark.metric.mode == "maximize" and value > best_sweep_value) or
                (task.benchmark.metric.mode == "minimize" and value < best_sweep_value)
            )
            if is_better:
                best_sweep_value = value
                best_sweep_knobs = dict(cand.new_knobs)
                progress.on_message(
                    f"[agent]   Sweep {i+1}/{len(candidates)}: {cand.description} → {value:.6g} (NEW BEST)"
                )

        # Restore best configuration or original. Either way, keep the sweep
        # section: candidates have it stripped (see generate_sweep_candidates),
        # and dropping it here would permanently disable tuning for the rest
        # of the run. The workspace is rebuilt with the final knobs by the
        # reprofile that follows this phase.
        if best_sweep_knobs is not None:
            save_knobs(knobs_path, {**best_sweep_knobs, "sweep": original_knobs["sweep"]})
            event_log._write("auto_tune_sweep", iteration, {
                "candidates_tried": len(candidates),
                "best_value": best_sweep_value,
                "best_knobs": best_sweep_knobs,
                "improvement": best_sweep_value - current_best,
            })
            progress.on_message(
                f"[agent] Auto-tune: found better params ({best_sweep_value:.6g} vs {current_best:.6g})"
            )
            return best_sweep_value
        else:
            save_knobs(knobs_path, original_knobs)
            return None

    except Exception as exc:  # noqa: BLE001 -- best-effort sweep over untrusted build/bench subprocess calls, must not abort the run
        progress.on_message(f"[agent] Auto-tune sweep failed: {exc}")
        # Restore the pre-sweep knobs (the sweep may have left a losing
        # candidate's values in tuning.yaml). Keep the sweep section: the
        # sweep didn't complete, so it should be retryable next time.
        if original_knobs is not None:
            try:
                save_knobs(knobs_path, original_knobs)
            except Exception:  # noqa: BLE001 -- best-effort cleanup, nothing more to do if this fails
                logger.warning("Failed to restore knobs after sweep error", exc_info=True)
        return None


def run(ctx: AgentContext, max_trials: int = 15) -> None:
    """Auto-tune phase: sweep tuning.yaml (if present) and fold in any improvement.

    If the sweep finds a better parameter combination, updates ctx.best_value
    and appends a history entry -- mirroring the manual accept-and-record flow
    used for LLM-proposed patches.
    """
    sweep_result = _auto_tune_sweep(ctx, max_trials=max_trials)
    if sweep_result is not None:
        ctx.best_value = sweep_result
        ctx.history.append(make_history_entry(
            ctx.iteration, "Auto-tune sweep found better parameters",
            ctx.best_value, ctx.baseline_val,
            accepted=True,
        ))
