from __future__ import annotations

import contextlib
import json
import logging
import math
import shlex
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from perflab.analyzers.metrics_rollup import improvement_factor, is_improvement
from perflab.memory.run_store import snapshot_workspace
from perflab.optimizers.history import make_history_entry
from perflab.optimizers.patch import (
    SearchReplaceBlock,
    apply_patch,
    validate_patch,
    workspace_copy_ignore,
)
from perflab.runners.benchmark import (
    metric_value,
    run_benchmark,
    validate_bench_variance,
    validate_contract,
)
from perflab.runners.correctness import run_correctness, run_correctness_twice
from perflab.runners.pipeline import run_pipeline_for_ctx
from perflab.task_spec import DEFAULT_BUILD_TIMEOUT_S
from perflab.tools.shell import run_cmd

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


@contextlib.contextmanager
def _patched_workspace_copy(
    ws: Path, blocks: list[SearchReplaceBlock], prefix: str, out_dir: Path,
) -> Iterator[Path]:
    """Yield a temporary copy of the workspace with the patch applied.

    Candidate code runs with the workspace as cwd and can write arbitrary
    files at runtime -- not just the ones the patch touched -- so correctness
    and benchmark subprocesses must never execute in the real workspace.
    A candidate that rewrites tests.py mid-benchmark poisons only its own
    discarded copy, not the checks applied to later candidates.

    out_dir's contents are excluded from the copy (out/runs grows every
    iteration); the empty directory is kept for bench.json writes.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        temp_ws = temp_dir / "ws"
        shutil.copytree(
            ws, temp_ws, dirs_exist_ok=True,
            ignore=workspace_copy_ignore(ws, out_dir),
        )
        apply_patch(blocks, temp_ws)
        yield temp_ws
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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

    def _warn_if_rlimit_failed(rlimits_applied: bool | None, stage: str) -> None:
        if rlimits_applied is False:
            event_log.rlimit_warning(
                it, f"rlimit failed for candidate {ci + 1} during {stage}",
                candidate_index=ci,
            )

    def _reject(suffix: str, error: dict) -> tuple[BeamCandidate, list[dict]]:
        errors.append(error)
        return BeamCandidate(
            iteration=it, index=ci, blocks=blocks,
            description=f"{desc} ({suffix})", reasoning=reasoning,
        ), errors

    # Apply -> build -> correctness -> benchmark, all inside a temporary
    # workspace copy so nothing the candidate's processes write survives.
    with _patched_workspace_copy(ws, blocks, f"perflab_eval_{ci}_", task.out_dir) as temp_ws:
        # Build the patched copy so compiled tasks benchmark the patched
        # binary (prescreen built only its own, already-discarded copy).
        if task.build is not None:
            build_res = run_cmd(
                shlex.split(task.build.cmd), cwd=temp_ws,
                timeout_s=task.build.timeout_s or DEFAULT_BUILD_TIMEOUT_S,
            )
            if build_res.returncode != task.build.expected_exit:
                progress.on_message(f"[agent]   Build FAILED (rc={build_res.returncode})")
                return _reject("build failed", {
                    "type": "build",
                    "description": f"candidate {ci + 1} failed build (exit code {build_res.returncode})",
                    "output": (build_res.stderr or "")[:3000],
                })

        # Correctness check (re-run with a different seed when the
        # anti-gaming determinism check is enabled)
        det_warnings: list[str] = []
        if task.anti_gaming.determinism_rerun:
            cres, det_warnings = run_correctness_twice(
                task.correctness.cmd, cwd=temp_ws, program_type=task.program_type,
                rlimit_as_gb=task.constraints.rlimit_as_gb,
                expected_exit=task.correctness.expected_exit,
                env_passthrough=task.constraints.env_passthrough,
                isolation=ctx.config.isolation,
                accuracy_tolerance=task.constraints.accuracy_tolerance,
            )
        else:
            cres = run_correctness(
                task.correctness.cmd, cwd=temp_ws, program_type=task.program_type,
                rlimit_as_gb=task.constraints.rlimit_as_gb,
                env_passthrough=task.constraints.env_passthrough,
                isolation=ctx.config.isolation,
                accuracy_tolerance=task.constraints.accuracy_tolerance,
            )
        _warn_if_rlimit_failed(cres.rlimits_applied, "correctness")
        event_log.candidate_correctness(
            it, ci, cres.returncode == task.correctness.expected_exit,
            cres.returncode, cres.stderr,
        )

        if cres.returncode != task.correctness.expected_exit:
            progress.on_message(f"[agent]   Correctness FAILED (rc={cres.returncode})")
            return _reject("correctness failed", {
                "type": "correctness",
                "description": f"candidate {ci + 1} failed correctness (exit code {cres.returncode})",
                "output": (cres.stderr or "")[:3000],
            })

        if det_warnings:
            for warning in det_warnings:
                event_log.anti_gaming_warning(
                    it, "determinism_rerun", warning, candidate_index=ci,
                )
            progress.on_message("[agent]   Determinism re-run FAILED — rejecting candidate")
            return _reject("determinism re-run failed", {
                "type": "anti_gaming",
                "description": (
                    f"candidate {ci + 1} passed correctness once but failed the "
                    f"re-run with a different seed (possible caching/gaming)"
                ),
                "output": det_warnings[0][:3000],
            })

        # Benchmark (fast screen or full depending on mode)
        try:
            bres, bench = run_benchmark(
                task.benchmark.cmd, cwd=temp_ws, fast_mode=use_fast, program_type=task.program_type,
                rlimit_as_gb=task.constraints.rlimit_as_gb,
                env_passthrough=task.constraints.env_passthrough,
                isolation=ctx.config.isolation,
                warmup=task.benchmark.warmup, repeats=task.benchmark.repeats,
            )
            _warn_if_rlimit_failed(bres.rlimits_applied, "benchmark")
        except Exception as exc:  # noqa: BLE001 -- untrusted candidate's benchmark subprocess can fail in arbitrary ways; must feed back as candidate error, not crash the run
            # Keep candidate-controlled text (exception messages embed the
            # subprocess's stdout/stderr) out of "description" -- descriptions
            # flow into prompt headings and failure_memory unsanitized; only
            # "output" is sanitized at prompt-render time.
            progress.on_message(f"[agent]   Benchmark FAILED: {exc}")
            return _reject("benchmark failed", {
                "type": "benchmark",
                "description": f"candidate {ci + 1} benchmark failed ({type(exc).__name__})",
                "output": str(exc)[:3000],
            })

        # Anti-gaming: zero-variance timing arrays suggest memoization/caching
        # (advisory — coarse timers can legitimately produce identical values)
        if task.anti_gaming.bench_variance_check:
            for warning in validate_bench_variance(bench):
                event_log.anti_gaming_warning(
                    it, "bench_variance", warning, candidate_index=ci,
                )
                progress.on_message(f"[agent]   WARNING (anti-gaming): {warning}")

        # Anti-gaming: reject candidates that spin up background threads
        # (opt-in; requires the bench harness to report thread_delta)
        if task.anti_gaming.thread_count_check:
            # bench.json is candidate-controlled: "meta" may be null/non-dict,
            # and thread_delta may be non-numeric or non-finite ("3.0", NaN,
            # Infinity, "lots"). Any of these must reject the candidate, never
            # crash the run.
            meta = bench.get("meta")
            thread_delta = bench.get("thread_delta")
            if thread_delta is None and isinstance(meta, dict):
                thread_delta = meta.get("thread_delta")
            if thread_delta is None:
                event_log.anti_gaming_warning(
                    it, "thread_count",
                    "thread_count_check enabled but bench.json reports no thread_delta field",
                    candidate_index=ci,
                )
            else:
                try:
                    parsed = float(thread_delta)
                    if not math.isfinite(parsed):
                        raise ValueError("non-finite thread_delta")
                    thread_delta_int = int(parsed)
                except (TypeError, ValueError, OverflowError):
                    event_log.anti_gaming_warning(
                        it, "thread_count",
                        f"unparseable thread_delta value: {thread_delta!r}",
                        candidate_index=ci,
                    )
                    progress.on_message(
                        "[agent]   Thread check FAILED (unparseable thread_delta) — rejecting candidate"
                    )
                    # Keep the raw candidate-controlled value in "output" (see
                    # the prompt-injection note near the benchmark-failure path
                    # above), not in "description".
                    return _reject("thread check failed", {
                        "type": "anti_gaming",
                        "description": (
                            f"candidate {ci + 1} reported an unparseable thread_delta value"
                        ),
                        "output": str(thread_delta)[:3000],
                    })
                if thread_delta_int > task.anti_gaming.max_thread_delta:
                    event_log.anti_gaming_warning(
                        it, "thread_count",
                        f"thread_delta={thread_delta_int} exceeds max_thread_delta={task.anti_gaming.max_thread_delta}",
                        candidate_index=ci,
                    )
                    progress.on_message(
                        f"[agent]   Thread check FAILED (delta={thread_delta_int}) — rejecting candidate"
                    )
                    return _reject("thread check failed", {
                        "type": "anti_gaming",
                        "description": (
                            f"candidate {ci + 1} created {thread_delta_int} new thread(s) during "
                            f"benchmarking (max allowed: {task.anti_gaming.max_thread_delta})"
                        ),
                        "output": "",
                    })

        # Contract validation (fast screens intentionally run warmup=0/repeats=2,
        # so min-sampling enforcement applies only to full benchmarks)
        contract_errors = validate_contract(
            bench, task.contract, enforce_min_sampling=not use_fast,
        )
        if contract_errors:
            progress.on_message(f"[agent]   Contract violation: {contract_errors}")
            return _reject("contract violation", {
                "type": "contract_violation",
                "description": f"candidate {ci + 1}: {contract_errors[0]}",
                "output": "",
            })

        val = metric_value(bench, task.benchmark.metric.name)
        progress.on_message(f"[agent]   {task.benchmark.metric.name} = {val:.6g}{screen_label}")

        event_log.candidate_benchmark(it, ci, val, task.benchmark.metric.name)

        return BeamCandidate(
            iteration=it, index=ci, blocks=blocks,
            description=desc, reasoning=reasoning, value=val,
        ), errors


def accept_best(
    ctx: AgentContext,
    candidates: list[BeamCandidate],
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

    # top_k is a per-iteration budget of *full re-benchmarks*, not a hard cap
    # on candidates examined. The old scored[:top_k] slice discarded
    # lower-ranked candidates before the loop, so a genuinely-improving
    # candidate was never examined when the higher-ranked ones failed their
    # full re-bench. Instead: non-improving candidates are skipped for free (the
    # `continue` below), and the budget is charged only when a fast-screened
    # candidate actually enters the full re-bench. top_k <= 0 means unlimited.
    # Non-fast mode never re-benches, so it ignores the budget and accepts the
    # first improving candidate as before.
    rebench_budget = ctx.config.top_k

    for idx, cand in enumerate(scored):
        assert cand.value is not None  # guaranteed by the `scored` filter above
        if not is_improvement(
            cand.value, ctx.best_value,
            task.benchmark.metric.mode,
            task.constraints.regression_tolerance,
        ):
            continue

        # If we used fast screening, re-benchmark with full precision --
        # in a temp workspace copy, same as the initial evaluation, so the
        # re-bench can't leak candidate writes into the real workspace.
        if use_fast:
            if ctx.config.top_k > 0 and rebench_budget <= 0:
                remaining = sum(
                    1 for c in scored[idx:]
                    if is_improvement(
                        _cand_value(c), ctx.best_value,
                        task.benchmark.metric.mode,
                        task.constraints.regression_tolerance,
                    )
                )
                progress.on_message(
                    f"[agent]   Re-bench budget (top_k={ctx.config.top_k}) exhausted; "
                    f"{remaining} improving candidate(s) left unexamined"
                )
                break
            rebench_budget -= 1
            progress.on_message("[agent]   Re-benchmarking top candidate with full precision...")
            with _patched_workspace_copy(ws, cand.blocks, "perflab_rebench_", task.out_dir) as temp_ws:
                if task.build is not None:
                    build_res = run_cmd(
                        shlex.split(task.build.cmd), cwd=temp_ws,
                        timeout_s=task.build.timeout_s or DEFAULT_BUILD_TIMEOUT_S,
                    )
                    if build_res.returncode != task.build.expected_exit:
                        progress.on_message(f"[agent]   Build failed on full re-bench (rc={build_res.returncode})")
                        continue
                _, bench_full = run_benchmark(
                    task.benchmark.cmd, cwd=temp_ws, fast_mode=False,
                    program_type=task.program_type,
                    rlimit_as_gb=task.constraints.rlimit_as_gb,
                    env_passthrough=task.constraints.env_passthrough,
                    isolation=ctx.config.isolation,
                    warmup=task.benchmark.warmup, repeats=task.benchmark.repeats,
                )
                contract_errors = validate_contract(bench_full, task.contract)
                if contract_errors:
                    progress.on_message(f"[agent]   Contract violation on full re-bench: {contract_errors}")
                    continue
                full_val = metric_value(bench_full, task.benchmark.metric.name)
                progress.on_message(f"[agent]   Full benchmark: {task.benchmark.metric.name} = {full_val:.6g}")
                cand.value = full_val

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
        speedup = improvement_factor(cand.value, ctx.baseline_val, task.benchmark.metric.mode)
        ctx.best_value = cand.value
        ctx.best_iter = it
        cand.accepted = True

        event_log.candidate_accepted(it, cand.index, cand.value, delta, speedup, cand.description)

        # Gaming detector: a single-iteration jump beyond the configured
        # threshold is suspicious. improvement_factor is mode-aware (>1 always
        # means better), unlike calc_speedup, which for minimize-mode metrics
        # shrinks toward 0 on improvement and could never cross a threshold.
        gain = improvement_factor(cand.value, old_best, task.benchmark.metric.mode)
        threshold = task.anti_gaming.gaming_speedup_threshold
        if gain > threshold:
            gaming_msg = (
                f"suspiciously large improvement ({gain:.1f}x > {threshold:g}x "
                f"threshold) in a single iteration — possible benchmark gaming"
            )
            progress.on_message(f"[agent] WARNING: {gaming_msg}")
            event_log.anti_gaming_warning(
                it, "speedup_threshold", gaming_msg, candidate_index=cand.index,
            )

        # Zero-metric gaming: improvement_factor returns a neutral 1.0 when a
        # value is 0, so the gain>threshold check above never fires for a
        # minimize-mode candidate reporting exactly 0.0 -- a stubbed/no-op
        # kernel, the most extreme gaming case. Flag it explicitly.
        if task.benchmark.metric.mode == "minimize" and cand.value == 0.0:
            zero_msg = (
                f"candidate reports zero {task.benchmark.metric.name} — degenerate "
                f"value, almost certainly benchmark gaming"
            )
            progress.on_message(f"[agent] WARNING: {zero_msg}")
            event_log.anti_gaming_warning(
                it, "zero_metric", zero_msg, candidate_index=cand.index,
            )

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
            accepted=True, mode=task.benchmark.metric.mode,
            reasoning=cand.reasoning or None,
            secondary_value=sec_val,
            # getattr guards test doubles that build ctx as a duck-typed
            # SimpleNamespace without this field -- real AgentContext always
            # has it (dataclass default None).
            estimated_cost_usd=getattr(ctx, "total_estimated_cost_usd", None),
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
        return (True, rel_improvement, accepted_value)

    # No candidate improved
    progress.on_message("[agent]   No improving candidate this iteration")
    best_desc = scored[0].description if scored else "no valid candidates"
    # Explicit None check: a genuine 0.0 metric value must be recorded, not
    # silently replaced by ctx.best_value.
    best_val = scored[0].value if scored else None
    reject_val = best_val if best_val is not None else ctx.best_value
    ctx.history.append(make_history_entry(
        it, f"no improvement ({best_desc})", reject_val, ctx.baseline_val,
        accepted=False, mode=task.benchmark.metric.mode,
        estimated_cost_usd=getattr(ctx, "total_estimated_cost_usd", None),
    ))
    return (False, None, None)


def reprofile_after_accept(ctx: AgentContext, accepted_value: float) -> None:
    """Re-profile after an accepted patch, and run a periodic drift check.

    Mutates ctx.latest_diagnostics. `accepted_value` is the just-accepted
    candidate's value (captured before any auto-tune sweep runs), used as the
    drift-check baseline -- intentionally distinct from ctx.best_value, which
    the auto-tune phase may have already moved past this candidate's value.

    When drift triggers a baseline re-measure, the drift benchmark's own
    measurement of the current workspace (drift_val) is passed through as
    current_value so remeasure_baseline can re-anchor ctx.best_value under the
    same conditions -- keeping both sides of the final speedup consistent (see
    remeasure_baseline's both-sides-same-conditions invariant) with no extra run.
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
            _, drift_bench = run_benchmark(task.benchmark.cmd, cwd=ws, fast_mode=False, program_type=task.program_type, rlimit_as_gb=task.constraints.rlimit_as_gb, env_passthrough=task.constraints.env_passthrough, isolation=ctx.config.isolation, warmup=task.benchmark.warmup, repeats=task.benchmark.repeats)
            drift_val = metric_value(drift_bench, task.benchmark.metric.name)
            drift_pct = abs(drift_val - accepted_value) / abs(accepted_value) * 100 if accepted_value else 0
            event_log.drift_check(it, drift_val, accepted_value, drift_pct)
            if drift_pct > 5:
                progress.on_message(f"[agent]   WARNING: drift of {drift_pct:.1f}% detected — re-measuring baseline")
                remeasure_baseline(ctx, current_value=drift_val)
        except Exception as exc:  # noqa: BLE001 -- best-effort periodic sanity check, must not abort the run
            progress.on_message(f"[agent]   Drift check failed: {exc}")


def remeasure_baseline(ctx: AgentContext, current_value: float | None = None) -> None:
    """Re-benchmark the baseline program under current machine conditions.

    Machine drift (thermal throttling, background load) makes the run-start
    baseline stale: speedups computed against it compare measurements taken
    under different conditions. Accepted patches can only touch
    edit_policy.allowed_paths, and snapshots/baseline.zip holds exactly those
    files as of the baseline run, so extracting the zip over a temp copy of
    the current workspace reconstructs the baseline program. Updates
    ctx.baseline_val (used by subsequent history entries and the final
    report); earlier history entries keep the speedups they were recorded
    with. Best-effort: keeps the original baseline on any failure.

    Invariant: final speedup = baseline/best must compare both sides under the
    SAME machine conditions. Re-measuring only the baseline would leave
    ctx.best_value at its old-conditions value, so the ratio could show <1x for
    a genuinely good run (or be inflated the other way). When ``current_value``
    is given -- the drift-check measurement of the current workspace taken
    moments ago under these same conditions -- ctx.best_value is re-anchored to
    it, keeping both sides consistent without any extra benchmark run.
    """
    task = ctx.task
    progress = ctx.progress
    baseline_zip = ctx.rp.run_dir / "snapshots" / "baseline.zip"
    if not baseline_zip.exists():
        progress.on_message("[agent]   No baseline snapshot found; keeping original baseline")
        return
    temp_dir = Path(tempfile.mkdtemp(prefix="perflab_baseline_rebench_"))
    try:
        temp_ws = temp_dir / "ws"
        shutil.copytree(
            ctx.ws, temp_ws, dirs_exist_ok=True,
            ignore=workspace_copy_ignore(ctx.ws, task.out_dir),
        )
        with zipfile.ZipFile(baseline_zip) as zf:
            zf.extractall(temp_ws)
        if task.build is not None:
            build_res = run_cmd(
                shlex.split(task.build.cmd), cwd=temp_ws,
                timeout_s=task.build.timeout_s or DEFAULT_BUILD_TIMEOUT_S,
            )
            if build_res.returncode != task.build.expected_exit:
                progress.on_message(
                    f"[agent]   Baseline re-measure build failed (rc={build_res.returncode}); keeping original baseline"
                )
                return
        _, bench = run_benchmark(
            task.benchmark.cmd, cwd=temp_ws, fast_mode=False,
            program_type=task.program_type,
            rlimit_as_gb=task.constraints.rlimit_as_gb,
            env_passthrough=task.constraints.env_passthrough,
            isolation=ctx.config.isolation,
            warmup=task.benchmark.warmup, repeats=task.benchmark.repeats,
        )
        new_baseline = metric_value(bench, task.benchmark.metric.name)
    except Exception as exc:  # noqa: BLE001 -- best-effort recalibration, must not abort the run
        progress.on_message(f"[agent]   Baseline re-measure failed: {exc}; keeping original baseline")
        return
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    old_baseline = ctx.baseline_val
    ctx.baseline_val = new_baseline
    if current_value is not None:
        best_old = ctx.best_value
        ctx.best_value = current_value
        ctx.event_log.baseline_remeasured(
            ctx.iteration, old_baseline, new_baseline,
            best_old=best_old, best_new=current_value,
        )
        progress.on_message(
            f"[agent]   Baseline re-measured: {old_baseline:.6g} -> {new_baseline:.6g}; "
            f"best re-anchored under same conditions: {best_old:.6g} -> {current_value:.6g}"
        )
    else:
        ctx.event_log.baseline_remeasured(ctx.iteration, old_baseline, new_baseline)
        progress.on_message(
            f"[agent]   Baseline re-measured: {old_baseline:.6g} -> {new_baseline:.6g}"
        )
