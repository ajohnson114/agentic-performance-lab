from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from perflab.analyzers.metrics_rollup import calc_speedup, is_improvement
from perflab.memory.run_store import snapshot_workspace
from perflab.optimizers.history import make_history_entry
from perflab.optimizers.patch import (
    SearchReplaceBlock,
    apply_patch,
    backup_files,
    restore_files,
    validate_patch,
)
from perflab.runners.benchmark import metric_value, run_benchmark, validate_contract
from perflab.runners.correctness import run_correctness
from perflab.runners.pipeline import run_pipeline_for_ctx

if TYPE_CHECKING:
    from perflab.optimizers.agent import AgentContext

logger = logging.getLogger(__name__)


@dataclass
class BeamCandidate:
    iteration: int
    index: int
    blocks: list[SearchReplaceBlock]
    description: str
    reasoning: str = ""
    value: float | None = None
    accepted: bool = False


def evaluate_single_candidate(
    ctx: AgentContext,
    ci: int,
    blocks: list[SearchReplaceBlock],
    reasoning: str,
    use_fast: bool,
) -> tuple[BeamCandidate, list[dict]]:
    """Evaluate a single candidate: validate, apply, correctness, benchmark.

    Returns (candidate, errors) where errors is a list of error dicts for feedback.
    """
    task = ctx.task
    ws = ctx.ws
    rp = ctx.rp
    it = ctx.iteration
    progress = ctx.progress
    event_log = ctx.event_log

    errors: list[dict] = []
    desc = f"candidate {ci + 1}: {len(blocks)} blocks"
    screen_label = " (fast screen)" if use_fast else ""
    progress.on_message(f"[agent]   Evaluating {desc}{screen_label}...")

    # Log patch content
    event_log.candidate_patch(it, ci, [
        {"file_path": b.file_path, "search": b.search, "replace": b.replace}
        for b in blocks
    ])

    # Validate
    patch_notices: list[str] = []
    validation_errors = validate_patch(
        blocks, task.edit_policy.allowed_paths, ws, notices=patch_notices
    )
    event_log.candidate_validation(it, ci, not validation_errors, validation_errors)
    if patch_notices:
        for note in patch_notices:
            progress.on_message(f"[agent]   Patch note: {note}")
        event_log.patch_fuzzy_correction(it, ci, patch_notices)

    if validation_errors:
        progress.on_message(f"[agent]   Validation errors: {validation_errors}")
        return BeamCandidate(
            iteration=it, index=ci, blocks=blocks,
            description=f"{desc} (INVALID: {validation_errors[0]})",
            reasoning=reasoning,
        ), errors

    # Backup -> apply -> correctness -> benchmark -> restore
    backup_dir = rp.run_dir / "backups" / f"iter{it}"
    backed_up = backup_files(blocks, ws, backup_dir)

    def _warn_if_rlimit_failed(rlimits_applied: bool | None, stage: str) -> None:
        if rlimits_applied is False:
            event_log.rlimit_warning(
                it, f"rlimit failed for candidate {ci + 1} during {stage}",
                candidate_index=ci,
            )

    try:
        apply_patch(blocks, ws)

        # Correctness check
        cres = run_correctness(
            task.correctness.cmd, cwd=ws, program_type=task.program_type,
            rlimit_as_gb=task.constraints.rlimit_as_gb,
            env_passthrough=task.constraints.env_passthrough,
            isolation=ctx.config.isolation,
        )
        _warn_if_rlimit_failed(cres.rlimits_applied, "correctness")
        event_log.candidate_correctness(
            it, ci, cres.returncode == task.correctness.expected_exit,
            cres.returncode, cres.stderr,
        )

        if cres.returncode != task.correctness.expected_exit:
            progress.on_message(f"[agent]   Correctness FAILED (rc={cres.returncode})")
            errors.append({
                "type": "correctness",
                "description": f"candidate {ci + 1} failed correctness (exit code {cres.returncode})",
                "output": (cres.stderr or "")[:3000],
            })
            return BeamCandidate(
                iteration=it, index=ci, blocks=blocks,
                description=f"{desc} (correctness failed)",
                reasoning=reasoning,
            ), errors

        # Benchmark (fast screen or full depending on mode)
        try:
            bres, bench = run_benchmark(
                task.benchmark.cmd, cwd=ws, fast_mode=use_fast, program_type=task.program_type,
                rlimit_as_gb=task.constraints.rlimit_as_gb,
                env_passthrough=task.constraints.env_passthrough,
                isolation=ctx.config.isolation,
            )
            _warn_if_rlimit_failed(bres.rlimits_applied, "benchmark")
        except Exception as exc:  # noqa: BLE001 -- untrusted candidate's benchmark subprocess can fail in arbitrary ways; must feed back as candidate error, not crash the run
            # Keep candidate-controlled text (exception messages embed the
            # subprocess's stdout/stderr) out of "description" -- descriptions
            # flow into prompt headings and failure_memory unsanitized; only
            # "output" is sanitized at prompt-render time.
            errors.append({
                "type": "benchmark",
                "description": f"candidate {ci + 1} benchmark failed ({type(exc).__name__})",
                "output": str(exc)[:3000],
            })
            progress.on_message(f"[agent]   Benchmark FAILED: {exc}")
            return BeamCandidate(
                iteration=it, index=ci, blocks=blocks,
                description=f"{desc} (benchmark failed)",
                reasoning=reasoning,
            ), errors

        # Contract validation
        contract_errors = validate_contract(bench, task.contract)
        if contract_errors:
            progress.on_message(f"[agent]   Contract violation: {contract_errors}")
            errors.append({
                "type": "contract_violation",
                "description": f"candidate {ci + 1}: {contract_errors[0]}",
                "output": "",
            })
            return BeamCandidate(
                iteration=it, index=ci, blocks=blocks,
                description=f"{desc} (contract violation)",
                reasoning=reasoning,
            ), errors

        val = metric_value(bench, task.benchmark.metric.name)
        progress.on_message(f"[agent]   {task.benchmark.metric.name} = {val:.6g}{screen_label}")

        event_log.candidate_benchmark(it, ci, val, task.benchmark.metric.name)

        return BeamCandidate(
            iteration=it, index=ci, blocks=blocks,
            description=desc, reasoning=reasoning, value=val,
        ), errors
    finally:
        restore_files(backed_up, ws)
        shutil.rmtree(backup_dir, ignore_errors=True)


def accept_best(
    ctx: AgentContext,
    candidates: list[BeamCandidate],
    backup_dir: Path,
    use_fast: bool,
) -> tuple[bool, float | None, float | None]:
    """Find the best improving candidate and accept it (apply permanently, record history).

    Does not run the auto-tune sweep or re-profile -- the caller (agent.py's
    iteration loop) is responsible for invoking the autotune phase and
    reprofile_after_accept() afterward, so this module never has to import
    another phase.

    Mutates ctx.history, ctx.accepted_patches, ctx.accepted_count, ctx.best_value,
    ctx.best_iter in place.
    Returns (accepted, rel_improvement, accepted_value) -- True + relative
    improvement + the accepted candidate's value for success, or
    (False, None, None) for no improvement.
    """
    task = ctx.task
    ws = ctx.ws
    rp = ctx.rp
    it = ctx.iteration
    progress = ctx.progress
    event_log = ctx.event_log

    scored = [c for c in candidates if c.value is not None]

    def _cand_value(c: BeamCandidate) -> float:
        assert c.value is not None  # `scored` is filtered to candidates with a value
        return c.value

    scored.sort(key=_cand_value, reverse=(task.benchmark.metric.mode == "maximize"))

    for cand in scored:
        assert cand.value is not None  # guaranteed by the `scored` filter above
        if not is_improvement(
            cand.value, ctx.best_value,
            task.benchmark.metric.mode,
            task.constraints.regression_tolerance,
        ):
            continue

        # If we used fast screening, re-benchmark with full precision
        if use_fast:
            progress.on_message("[agent]   Re-benchmarking top candidate with full precision...")
            backed_up = backup_files(cand.blocks, ws, backup_dir)
            try:
                apply_patch(cand.blocks, ws)
                _, bench_full = run_benchmark(task.benchmark.cmd, cwd=ws, fast_mode=False, program_type=task.program_type, rlimit_as_gb=task.constraints.rlimit_as_gb, isolation=ctx.config.isolation)
                contract_errors = validate_contract(bench_full, task.contract)
                if contract_errors:
                    progress.on_message(f"[agent]   Contract violation on full re-bench: {contract_errors}")
                    continue
                full_val = metric_value(bench_full, task.benchmark.metric.name)
                progress.on_message(f"[agent]   Full benchmark: {task.benchmark.metric.name} = {full_val:.6g}")
                cand.value = full_val
            finally:
                restore_files(backed_up, ws)

            # Re-check improvement with full benchmark value
            if not is_improvement(
                cand.value, ctx.best_value,
                task.benchmark.metric.mode,
                task.constraints.regression_tolerance,
            ):
                progress.on_message("[agent]   Full benchmark did not confirm improvement, skipping")
                continue

        # Compute relative improvement BEFORE updating best_value
        old_best = ctx.best_value

        # Re-apply permanently
        progress.on_message(f"[agent]   ACCEPTING {cand.description} (value={cand.value:.6g})")
        apply_patch(cand.blocks, ws)
        delta = cand.value - ctx.baseline_val
        speedup = calc_speedup(cand.value, ctx.baseline_val)
        ctx.best_value = cand.value
        ctx.best_iter = it
        cand.accepted = True

        event_log.candidate_accepted(it, cand.index, cand.value, delta, speedup, cand.description)

        # Gaming detector: warn if suspiciously large speedup on first iteration
        if it == 1 and speedup > 10.0:
            progress.on_message(f"[agent] WARNING: suspiciously large speedup ({speedup:.1f}x) on first iteration — possible gaming")

        # Track secondary metric if available
        sec_val = None
        if ctx.sec_metric:
            try:
                bench_json_path = rp.run_dir / "bench.json"
                if bench_json_path.exists():
                    bench_for_sec = json.loads(bench_json_path.read_text(encoding="utf-8"))
                    sec_val = metric_value(bench_for_sec, ctx.sec_metric.name)
            except (KeyError, TypeError):
                logger.debug("Secondary metric extraction failed for iteration %d", it, exc_info=True)
        ctx.history.append(make_history_entry(
            it, cand.description, cand.value, ctx.baseline_val,
            accepted=True,
            reasoning=cand.reasoning or None,
            secondary_value=sec_val,
        ))

        # Track for post-optimization summary
        ctx.accepted_patches.append({
            "iteration": it,
            "description": cand.description,
            "reasoning": cand.reasoning,
            "blocks": [{"file_path": b.file_path, "search": b.search[:500], "replace": b.replace[:500]} for b in cand.blocks],
            "value": cand.value,
        })

        ctx.accepted_count += 1

        # Snapshot accepted code
        snapshot_workspace(task, rp.run_dir, f"iter{it}")

        accepted_value = cand.value
        rel_improvement = abs(cand.value - old_best) / abs(old_best) if old_best != 0 else 0
        shutil.rmtree(backup_dir, ignore_errors=True)
        return (True, rel_improvement, accepted_value)

    # No candidate improved
    shutil.rmtree(backup_dir, ignore_errors=True)
    progress.on_message("[agent]   No improving candidate this iteration")
    best_desc = scored[0].description if scored else "no valid candidates"
    best_val = scored[0].value if scored else ctx.best_value
    reject_val = best_val or ctx.best_value
    ctx.history.append(make_history_entry(
        it, f"no improvement ({best_desc})", reject_val, ctx.baseline_val,
        accepted=False,
    ))
    return (False, None, None)


def reprofile_after_accept(ctx: AgentContext, accepted_value: float) -> None:
    """Re-profile after an accepted patch, and run a periodic drift check.

    Mutates ctx.latest_diagnostics. `accepted_value` is the just-accepted
    candidate's value (captured before any auto-tune sweep runs), used as the
    drift-check baseline -- intentionally distinct from ctx.best_value, which
    the auto-tune phase may have already moved past this candidate's value.
    """
    task = ctx.task
    ws = ctx.ws
    it = ctx.iteration
    progress = ctx.progress
    event_log = ctx.event_log

    _, _, _, diag = run_pipeline_for_ctx(ctx, do_profiles=True, capture_diagnostics=True)
    if diag is not None:
        ctx.latest_diagnostics = diag

    # Drift detection: every 3 accepted patches, re-run benchmark clean
    if ctx.accepted_count % 3 == 0:
        try:
            progress.on_message(f"[agent]   Drift check (accepted #{ctx.accepted_count})...")
            _, drift_bench = run_benchmark(task.benchmark.cmd, cwd=ws, fast_mode=False, program_type=task.program_type, rlimit_as_gb=task.constraints.rlimit_as_gb, isolation=ctx.config.isolation)
            drift_val = metric_value(drift_bench, task.benchmark.metric.name)
            drift_pct = abs(drift_val - accepted_value) / abs(accepted_value) * 100 if accepted_value else 0
            event_log.drift_check(it, drift_val, accepted_value, drift_pct)
            if drift_pct > 5:
                progress.on_message(f"[agent]   WARNING: drift of {drift_pct:.1f}% detected")
        except Exception as exc:  # noqa: BLE001 -- best-effort periodic sanity check, must not abort the run
            progress.on_message(f"[agent]   Drift check failed: {exc}")
