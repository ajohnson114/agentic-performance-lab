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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from perflab.analyzers.bench_stats import (
    cv_budget_for_gate,
    extract_repeated_values,
    repeats_needed_for_gate,
)
from perflab.analyzers.decision import (
    SCREENING_RULE,
    Comparison,
    ImprovementVerdict,
    is_paired_rule,
    rule_for_constraints,
)
from perflab.analyzers.metrics_rollup import improvement_factor
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
from perflab.runners.paired import PairedRun, run_paired_benchmark
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
    # Per-repeat measurements behind `value`, when the harness published them
    # (see bench_stats.extract_repeated_values). Feeds the statistical accept
    # gate; empty means "no variance information", which degrades the gate to
    # the bare ratio test rather than blocking acceptance.
    samples: list[float] = field(default_factory=list)


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
            samples=extract_repeated_values(bench, task.benchmark.metric.name),
        ), errors


@contextlib.contextmanager
def _paired_workspaces(
    ws: Path, blocks: list[SearchReplaceBlock], out_dir: Path,
) -> Iterator[tuple[Path, Path]]:
    """Two *equivalent* disposable copies: incumbent (unpatched) and candidate.

    Both come from the same routine, the same source tree and the same
    tempfile root, and differ only in whether the patch was applied. That
    symmetry is load-bearing: benchmarking the incumbent in the real workspace
    while the candidate ran from a temp copy would put a systematic
    between-arm difference (different page-cache state, possibly a different
    filesystem) straight back into the comparison the interleaving exists to
    clean up -- on top of the usual reason candidate code never runs in the
    real workspace.
    """
    with _patched_workspace_copy(ws, [], "perflab_paired_a_", out_dir) as incumbent_ws, \
         _patched_workspace_copy(ws, blocks, "perflab_paired_b_", out_dir) as candidate_ws:
        yield incumbent_ws, candidate_ws


def _build_arm(ctx: AgentContext, cwd: Path) -> str:
    """Build one arm's workspace. Returns "" on success, else an error string."""
    build = ctx.task.build
    if build is None:
        return ""
    res = run_cmd(
        shlex.split(build.cmd), cwd=cwd,
        timeout_s=build.timeout_s or DEFAULT_BUILD_TIMEOUT_S,
    )
    if res.returncode != build.expected_exit:
        return f"build failed (rc={res.returncode})"
    return ""


def _validate_paired_contract(ctx: AgentContext, run: PairedRun) -> list[str]:
    """Contract-check every spawn of an interleaved run.

    ``enforce_min_sampling=False`` per spawn, for the same reason the fast
    screen is exempted: the framework -- not the candidate -- chose the block
    size, and a block is deliberately shorter than a full run. The floor is not
    dropped, it is moved to the aggregate below, where it belongs: what
    ``contract.min_repeats`` is defending against is a candidate quietly
    reducing how much evidence backs its number, and the interleaved design
    collects at least as many repeats in total, on *both* arms.

    Everything else in the contract (fixed_params, required fields) is checked
    on every single spawn, so a candidate that changes problem size partway
    through a sequence is still caught.
    """
    errors: list[str] = []
    for spawn in run.spawns:
        for err in validate_contract(
            spawn.measurement.bench, ctx.task.contract, enforce_min_sampling=False,
        ):
            errors.append(f"{spawn.arm} block {spawn.pair}: {err}")
    measured = run.plan.measured_repeats_per_arm
    minimum = ctx.task.contract.min_repeats
    if measured < minimum:
        errors.append(
            f"interleaved run measured {measured} repeats per arm "
            f"({run.plan.blocks} blocks x {run.plan.repeats_per_block}), "
            f"below contract min_repeats={minimum}"
        )
    return errors[:5]


def _remeasure_paired(
    ctx: AgentContext, cand: BeamCandidate,
) -> tuple[PairedRun | None, str]:
    """Authoritative measurement by block-interleaved A/B (see runners.paired).

    On success, mutates ``cand.value``/``cand.samples`` to the interleaved
    candidate measurement and returns ``(run, "")``. On any failure returns
    ``(None, reason)``; the caller then falls back to the ordinary full
    re-benchmark, and the paired decision rule -- seeing no pairs -- falls back
    to the unpaired gate with ``verified=False``. Nothing about that path is
    silent, and none of it accepts a candidate the unpaired gate would reject.
    """
    task = ctx.task
    progress = ctx.progress
    try:
        with _paired_workspaces(ctx.ws, cand.blocks, task.out_dir) as (arm_a, arm_b):
            for label, cwd in (("incumbent", arm_a), ("candidate", arm_b)):
                err = _build_arm(ctx, cwd)
                if err:
                    return None, f"{label} arm {err}"

            def _on_spawn(arm: str, block: int, position: int) -> None:
                progress.on_message(
                    f"[agent]     block {position + 1}: arm {arm}"
                )

            run = run_paired_benchmark(
                task.benchmark.cmd,
                incumbent_cwd=arm_a, candidate_cwd=arm_b,
                metric_name=task.benchmark.metric.name,
                total_repeats=task.benchmark.repeats,
                warmup=task.benchmark.warmup,
                program_type=task.program_type,
                rlimit_as_gb=task.constraints.rlimit_as_gb,
                env_passthrough=task.constraints.env_passthrough,
                isolation=ctx.config.isolation,
                on_spawn=_on_spawn,
            )
    except Exception as exc:  # noqa: BLE001 -- an untrusted candidate's benchmark can fail in arbitrary ways; degrade to the unpaired path, never crash the run
        logger.debug("Interleaved measurement failed", exc_info=True)
        return None, f"interleaved measurement failed ({type(exc).__name__}: {exc})"

    contract_errors = _validate_paired_contract(ctx, run)
    if contract_errors:
        return None, f"contract violation in interleaved run: {contract_errors[0]}"

    cand.value = run.candidate_value
    cand.samples = run.candidate_samples

    metric = task.benchmark.metric.name
    paired_cv = run.paired_cv
    arm_cv = run.arm_cv("A")
    progress.on_message(
        f"[agent]   Interleaved A/B: {run.plan.blocks} pairs x "
        f"{run.plan.repeats_per_block} repeats, order "
        f"{''.join(run.order)} (drift imbalance {run.plan.imbalance:+.2f} slots), "
        f"{run.wall_s:.1f}s"
    )
    progress.on_message(
        f"[agent]   {metric}: candidate {run.candidate_value:.6g} vs "
        f"incumbent {run.incumbent_value:.6g}"
        + (f"; per-arm CV {arm_cv:.1%}" if arm_cv is not None else "")
        + (f", paired-difference CV {paired_cv:.1%}" if paired_cv is not None else "")
    )
    # The incumbent was last measured under different conditions; saying how
    # far it moved is the honest version of the periodic drift check, and it
    # costs nothing here because both numbers were just measured.
    if ctx.best_value:
        drift = abs(run.incumbent_value - ctx.best_value) / abs(ctx.best_value)
        if drift > 0.02:
            progress.on_message(
                f"[agent]   Incumbent re-measured at {run.incumbent_value:.6g} vs "
                f"tracked best {ctx.best_value:.6g} ({drift:.1%} machine drift since "
                f"it was last benchmarked) — the paired comparison uses the fresh "
                f"measurement, so this drift does not reach the decision"
            )
    return run, ""


def _remeasure_full(ctx: AgentContext, cand: BeamCandidate) -> str:
    """Ordinary (unpaired) full-precision re-benchmark of one candidate.

    Returns "" on success, else a rejection reason. Mutates
    ``cand.value``/``cand.samples``.
    """
    task = ctx.task
    with _patched_workspace_copy(
        ctx.ws, cand.blocks, "perflab_rebench_", task.out_dir,
    ) as temp_ws:
        err = _build_arm(ctx, temp_ws)
        if err:
            ctx.progress.on_message(f"[agent]   Build failed on full re-bench ({err})")
            return "build failed on full re-bench"
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
            ctx.progress.on_message(
                f"[agent]   Contract violation on full re-bench: {contract_errors}"
            )
            return "contract violation on full re-bench"
        full_val = metric_value(bench_full, task.benchmark.metric.name)
        ctx.progress.on_message(
            f"[agent]   Full benchmark: {task.benchmark.metric.name} = {full_val:.6g}"
        )
        cand.value = full_val
        cand.samples = extract_repeated_values(bench_full, task.benchmark.metric.name)
    return ""


def _incumbent_samples(ctx: AgentContext) -> list[float]:
    """Per-repeat samples for the *current best* program, or [] if unavailable.

    run_dir/bench.json is written by run_pipeline_for_ctx, which runs on the
    real workspace at baseline and again after every accepted patch -- i.e. it
    always describes the incumbent. It is only trusted here when its metric
    still equals ctx.best_value: the auto-tune phase can move best_value
    without rewriting bench.json, and stale samples would misstate the
    incumbent's spread. On any mismatch (or any I/O or parse failure) we return
    [] and the gate degrades to a one-sided interval test against the incumbent
    as a point estimate -- still stricter than the bare ratio it replaced.
    """
    try:
        path = ctx.rp.run_dir / "bench.json"
        if not path.exists():
            return []
        blob = json.loads(path.read_text(encoding="utf-8"))
        val = metric_value(blob, ctx.task.benchmark.metric.name)
    except Exception:  # noqa: BLE001 -- best-effort variance lookup; a missing/garbled bench.json must not abort the run
        logger.debug("Incumbent sample lookup failed", exc_info=True)
        return []
    if not math.isclose(val, ctx.best_value, rel_tol=1e-9, abs_tol=0.0):
        return []
    return extract_repeated_values(blob, ctx.task.benchmark.metric.name)


def _report_noise_rejection(ctx: AgentContext, verdict: ImprovementVerdict) -> None:
    """Explain a candidate rejected for being inside the measurement noise.

    This is the message the whole statistical gate exists to produce: the
    candidate DID beat the tolerance, and was still thrown away, because this
    machine cannot tell the difference. Say so, and say what would fix it --
    the environment is the bottleneck here, not the kernel.
    """
    task = ctx.task
    progress = ctx.progress
    progress.on_message(f"[agent]   Rejected: {verdict.reason}")
    if verdict.cv is None:
        return
    if verdict.paired:
        # The advice below is derived from the UNPAIRED resolution formula
        # (resolvable ~ 2*t*CV/sqrt(n) across two independent arms), and a
        # paired design does not obey it: its dispersion is the spread of the
        # differences and its n is the pair count. Quoting a repeats number
        # from the wrong model would be worse than saying nothing, so this
        # says the thing that is actually actionable for a paired run.
        progress.on_message(
            f"[agent]   Paired-difference CV is {verdict.cv:.1%} across "
            f"{verdict.n} interleaved pair(s) — the candidate's effect is not "
            f"consistently signed across blocks. More blocks (or a quieter "
            f"machine) would resolve it; more repeats inside a block would not, "
            f"since the spread here is between blocks, not within them."
        )
        return
    configured = getattr(task.constraints, "cv_threshold", None)
    tol = task.constraints.regression_tolerance
    repeats = task.benchmark.repeats
    budget = configured if configured is not None else cv_budget_for_gate(tol, repeats)
    if verdict.cv <= budget:
        return
    needed = repeats_needed_for_gate(verdict.cv, tol)
    progress.on_message(
        f"[agent]   Measurement noise (CV={verdict.cv:.1%}) exceeds the {budget:.1%} "
        f"this machine would need to resolve a {tol:.1%} gate at repeats={repeats}: "
        f"raise benchmark.repeats to ~{needed}, quiet the machine (pin clocks, "
        f"stop background load), or widen constraints.regression_tolerance. "
        f"Until then the measurement environment -- not the kernel -- is the "
        f"limiting factor."
    )


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

    # Two strategies, deliberately (perflab.analyzers.decision owns both):
    #
    #   accept_rule  -- what the task configured (default: non-overlapping 95%
    #                   CIs on top of the tolerance floor). Applied to the
    #                   *authoritative* measurement, and to it only.
    #   SCREENING_RULE -- tolerance only. The fast screen benchmarks at
    #                   repeats=2 purely to rank candidates; running the
    #                   variance test on 2 samples would veto genuine wins
    #                   before they are ever measured properly. The full
    #                   re-bench below is the gate that decides.
    #
    # Non-fast mode never re-benches, so its single check is authoritative and
    # uses accept_rule with the samples.
    mode = task.benchmark.metric.mode
    tol = task.constraints.regression_tolerance
    accept_rule = rule_for_constraints(task.constraints)
    incumbent_samples = _incumbent_samples(ctx)
    reject_reasons: dict[int, str] = {}

    # A paired rule needs a block-interleaved measurement, so selecting it also
    # selects how the authoritative benchmark is taken (see runners.paired for
    # why that cannot be bolted on afterwards -- pairing is a property of the
    # experiment, not of the arithmetic applied to it). The fast screen is
    # deliberately untouched: it is a cheap ranking pass, and doubling its cost
    # to make it rigorous would defeat the reason it exists.
    paired_enabled = is_paired_rule(accept_rule)
    # Both modes now route through the same "screen, then measure
    # authoritatively" shape when pairing is on; without it, behavior is
    # exactly as before.
    needs_remeasure = use_fast or paired_enabled

    def _comparison(
        c: BeamCandidate, paired_run: PairedRun | None = None,
    ) -> Comparison:
        """The measured facts. Which of them matter is the rule's business."""
        if paired_run is not None:
            # The incumbent value comes from the interleaved run, not from
            # ctx.best_value: the entire point is that both sides were measured
            # under the same conditions, moments apart. Using the tracked best
            # here would smuggle the stale-baseline bias back into the one
            # comparison that was built to exclude it.
            return Comparison(
                candidate=paired_run.candidate_value,
                incumbent=paired_run.incumbent_value,
                mode=mode,
                tolerance=tol,
                candidate_samples=paired_run.candidate_samples or None,
                incumbent_samples=paired_run.incumbent_samples or None,
                pairs=paired_run.pairs,
            )
        assert c.value is not None
        return Comparison(
            candidate=c.value,
            incumbent=ctx.best_value,
            mode=mode,
            tolerance=tol,
            candidate_samples=c.samples or None,
            incumbent_samples=incumbent_samples or None,
        )

    def _assess(
        c: BeamCandidate, paired_run: PairedRun | None = None,
    ) -> ImprovementVerdict:
        return accept_rule.decide(_comparison(c, paired_run))

    def _screen(c: BeamCandidate) -> bool:
        return SCREENING_RULE.decide(_comparison(c)).improved

    for idx, cand in enumerate(scored):
        assert cand.value is not None  # guaranteed by the `scored` filter above
        paired_run: PairedRun | None = None
        if needs_remeasure:
            # Cheap directional screen (SCREENING_RULE) -- see the note above.
            if not _screen(cand):
                screen_name = "fast screen" if use_fast else "screen"
                reject_reasons[idx] = (
                    f"{screen_name} did not beat the incumbent by {tol:.1%}"
                )
                continue
            verdict = None
        else:
            verdict = _assess(cand)
            if not verdict.improved:
                reject_reasons[idx] = verdict.reason
                if verdict.beats_tolerance:
                    _report_noise_rejection(ctx, verdict)
                continue

        # Authoritative re-measurement, in temp workspace copies (same as the
        # initial evaluation, so it can't leak candidate writes into the real
        # workspace). Either an ordinary full-precision benchmark or, when a
        # paired rule is selected, a block-interleaved A/B against the
        # incumbent.
        if needs_remeasure:
            if ctx.config.top_k > 0 and rebench_budget <= 0:
                # Counted with the screening rule, since that is the bar these
                # candidates cleared to get here -- none of them has an
                # authoritative measurement yet.
                remaining = sum(1 for c in scored[idx:] if _screen(c))
                progress.on_message(
                    f"[agent]   Re-bench budget (top_k={ctx.config.top_k}) exhausted; "
                    f"{remaining} improving candidate(s) left unexamined"
                )
                break
            rebench_budget -= 1

            paired_error = ""
            if paired_enabled:
                progress.on_message(
                    "[agent]   Re-benchmarking top candidate, block-interleaved "
                    "against the incumbent..."
                )
                paired_run, paired_error = _remeasure_paired(ctx, cand)
                if paired_run is None:
                    # Fail closed, and loudly: fall back to the ordinary
                    # re-bench, which leaves the comparison with no pairs, so
                    # PairedDifference degrades to the unpaired gate and
                    # reports verified=False.
                    progress.on_message(
                        f"[agent]   Paired measurement unavailable ({paired_error}); "
                        f"falling back to an unpaired full benchmark"
                    )
            if paired_run is None:
                if not paired_enabled:
                    progress.on_message(
                        "[agent]   Re-benchmarking top candidate with full precision..."
                    )
                rebench_error = _remeasure_full(ctx, cand)
                if rebench_error:
                    reject_reasons[idx] = rebench_error
                    continue

            # Re-check improvement with the authoritative measurement. This is
            # the gate that decides, so it carries the samples (and the pairs).
            verdict = _assess(cand, paired_run)
            if not verdict.improved:
                reject_reasons[idx] = verdict.reason
                if verdict.beats_tolerance:
                    progress.on_message(
                        "[agent]   Full benchmark improved the metric but not beyond noise"
                    )
                    _report_noise_rejection(ctx, verdict)
                else:
                    progress.on_message("[agent]   Full benchmark did not confirm improvement, skipping")
                continue

        assert verdict is not None  # set by whichever branch reached here
        if verdict.rule == accept_rule.name and paired_enabled and not verdict.paired:
            # Never silent, and distinct from the missing-samples note below:
            # the interleaved measurement did not happen, so this accept rests
            # on the unpaired gate and carries the drift bias pairing removes.
            progress.on_message(
                "[agent]   NOTE: accepted WITHOUT the paired test — the interleaved "
                "measurement produced no usable pairs, so the decision fell back to "
                "the unpaired interval gate (still stricter than the bare ratio, but "
                "the incumbent and candidate were not measured under matched conditions)"
            )
        elif not verdict.verified:
            # Never silent: the accept happened on a bare ratio because the
            # harness published no per-repeat samples for this metric (see
            # bench_stats.extract_repeated_values for the shapes we accept).
            progress.on_message(
                "[agent]   NOTE: accepted without variance verification — bench.json "
                f"exposed no per-repeat samples for {task.benchmark.metric.name}; "
                "emit a 'raw_values'/'samples' array beside the metric (or a "
                "top-level 'times_ms') to enable the statistical gate"
            )

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
    # Carry the top candidate's rejection reason into the history/report so
    # "did not beat baseline" and "beat it but within measurement noise" stay
    # distinguishable after the run.
    top_reason = reject_reasons.get(0, "")
    if top_reason:
        best_desc = f"{best_desc}: {top_reason}"
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
