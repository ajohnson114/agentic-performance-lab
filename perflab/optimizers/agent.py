from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from perflab.analyzers.compiler_diagnostics import CompilerDiagnostics
from perflab.analyzers.metrics_rollup import improvement_factor
from perflab.llm.base import LLMProvider
from perflab.llm.config import LLMConfig, create_provider
from perflab.llm.pricing import estimate_cost_usd
from perflab.memory.run_store import RunPaths, RunStore
from perflab.optimizers.convergence import ConvergenceDetector
from perflab.optimizers.event_log import AgentEventLog
from perflab.optimizers.history import make_history_entry
from perflab.optimizers.patch import snapshot_protected_files, verify_protected_files
from perflab.optimizers.phases import autotune, baseline, evaluate, finalize, generate, prescreen
from perflab.optimizers.progress import AgentProgress, PrintProgress, fmt_elapsed
from perflab.task_spec import SecondaryMetricSpec, TaskSpec
from perflab.tools.isolation import IsolationPolicy

# Suppress JAX fork() warnings — harmless in agent context where fork is controlled
warnings.filterwarnings("ignore", message=".*os.fork.*multithreaded.*", category=RuntimeWarning)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    n_candidates: int = 6
    top_k: int = 2
    max_iters: int = 12
    early_stop: bool = True
    fast_screen: bool = True
    max_wall_time_s: int = 3600
    # OS-level sandboxing for candidate subprocess execution (see
    # perflab.tools.isolation). None means rlimits/env-allowlist only.
    # Applied uniformly to baseline AND candidate benchmark/correctness runs
    # so sandbox overhead cancels out of speedup comparisons.
    isolation: IsolationPolicy | None = None
    # Estimated-cost budget in USD (None = unlimited). Checked each iteration
    # against AgentContext.total_estimated_cost_usd; the run stops gracefully
    # (normal finalize/report) once the estimate reaches this limit. cli.py
    # fails closed at startup if this is set but the configured model's
    # pricing is unknown, rather than running un-metered.
    max_cost_usd: float | None = None


@dataclass
class AgentContext:
    """All state for an optimization run, threaded through every phase.

    The Context pattern replaces the 30+ local variables and 18-parameter
    function signatures in the agent loop with a single structured object.
    Each handler reads what it needs and writes what it produces.

    Fields are organized by lifecycle:
      - Immutable inputs: set at construction, never changed
      - Baseline state: set once during the baseline phase
      - Evolving state: mutated across iterations by handlers
      - Per-iteration transient: reset at the start of each iteration
      - Accumulating metadata: counters and logs that grow monotonically
    """

    # --- Immutable inputs (set at construction) ---
    task: TaskSpec
    config: AgentConfig
    llm_config: LLMConfig
    provider: LLMProvider
    progress: AgentProgress
    ws: Path                                              # task.workspace shorthand
    rp: RunPaths                                          # RunPaths from run_store.new_run()
    event_log: AgentEventLog
    expert_suggestion: str | None = None

    # --- Baseline state (set once after baseline phase) ---
    baseline_val: float = 0.0
    sec_metric: SecondaryMetricSpec | None = None
    baseline_sec_val: float | None = None
    sysinfo: dict = field(default_factory=dict)
    hardware_mismatch: str | None = None
    prior_run_context: str | None = None

    # --- Evolving optimization state (mutated across iterations) ---
    iteration: int = 0
    best_value: float = 0.0
    best_iter: int = 0
    accepted_count: int = 0
    history: list[dict] = field(default_factory=list)
    accepted_patches: list[dict] = field(default_factory=list)
    failure_memory: list[dict] = field(default_factory=list)
    latest_diagnostics: CompilerDiagnostics | None = None
    prev_summaries: dict | None = None
    profiler_summaries: dict = field(default_factory=dict)

    # --- Per-iteration transient (reset each iteration) ---
    last_errors: list[dict] = field(default_factory=list)
    promising_alternatives: list[dict] = field(default_factory=list)

    # --- Accumulating metadata ---
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_latency: float = 0.0
    # Recomputed from total_input_tokens/total_output_tokens whenever they
    # change (the price table is fixed for the run, so this is equivalent to
    # summing per-call deltas). None when the configured model's pricing is
    # unknown -- never a fabricated dollar figure.
    total_estimated_cost_usd: float | None = None
    user_actions: list[dict] = field(default_factory=list)
    early_stop_reason: str | None = None
    convergence: ConvergenceDetector | None = None
    wall_start: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        """Serialize optimization state for the per-iteration state.json run
        artifact (excludes non-serializable objects)."""
        d = {
            "iteration": self.iteration,
            "best_value": self.best_value,
            "best_iter": self.best_iter,
            "baseline_val": self.baseline_val,
            "accepted_count": self.accepted_count,
            "history": self.history,
            "accepted_patches": self.accepted_patches,
            "failure_memory": self.failure_memory,
            "last_errors": self.last_errors,
            "promising_alternatives": self.promising_alternatives,
            "total_llm_calls": self.total_llm_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_llm_latency": self.total_llm_latency,
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
            "user_actions": self.user_actions,
            "early_stop_reason": self.early_stop_reason,
        }
        if self.latest_diagnostics is not None:
            d["latest_diagnostics"] = self.latest_diagnostics.to_dict()
        return d


@dataclass
class AgentResult:
    best_value: float
    best_iter: int
    baseline_value: float
    history: list[dict]
    run_dir: Path


def _verify_and_warn_protected_files(
    ctx: AgentContext, protected_dir: Path, protected_hashes: dict[str, str],
) -> list[str]:
    """Verify protected files against their snapshot, logging + restoring any tamper.

    Shared by the mid-iteration check (right after an accept, before autotune's
    sweep trials run against the live workspace) and the end-of-iteration
    check, so the warning message is constructed in exactly one place.
    """
    tampered = verify_protected_files(ctx.ws, protected_dir, protected_hashes)
    if tampered:
        ctx.progress.on_message(
            f"[agent] WARNING: protected file(s) modified during candidate "
            f"execution — restored from snapshot: {', '.join(tampered)}"
        )
        ctx.event_log.anti_gaming_warning(
            ctx.iteration, "protected_file_tampering",
            f"modified at runtime and restored: {', '.join(tampered)}",
        )
    return tampered


def _collect_promising_alternatives(
    candidates: list[evaluate.BeamCandidate],
    baseline_val: float,
    mode: Literal["maximize", "minimize"],
) -> list[dict]:
    """Good-but-not-best candidates from this iteration, best first (top 3).

    Mode-aware via improvement_factor (>1.0 always means better), so
    minimize-mode metrics don't report regressions as promising.
    """
    alts: list[dict] = []
    for cand in candidates:
        if cand.value is None or cand.accepted:
            continue
        gain = improvement_factor(cand.value, baseline_val, mode)
        if gain <= 1.0:
            continue
        alts.append({
            "description": cand.description,
            "reasoning": cand.reasoning,
            "value": cand.value,
            "improvement": round(gain, 2),
        })
    alts.sort(key=lambda x: x["improvement"], reverse=True)
    return alts[:3]


def _update_and_check_cost_limit(ctx: AgentContext, it: int) -> bool:
    """Recompute cumulative estimated LLM cost and enforce --max-cost.

    Mutates ctx.total_estimated_cost_usd from the running token totals --
    the price table is fixed for the run, so recomputing from cumulative
    totals is equivalent to summing per-call deltas. Returns True when
    config.max_cost_usd is set and reached, after logging an event, a
    progress message, and an early-stop history entry; the caller breaks the
    iteration loop so finalize.run still produces a normal report.
    """
    ctx.total_estimated_cost_usd = estimate_cost_usd(
        ctx.llm_config.model, ctx.total_input_tokens, ctx.total_output_tokens,
        overrides=ctx.llm_config.pricing,
    )
    max_cost = ctx.config.max_cost_usd
    if max_cost is None or ctx.total_estimated_cost_usd is None:
        return False
    if ctx.total_estimated_cost_usd < max_cost:
        return False

    reason = f"cost limit reached (est. ${ctx.total_estimated_cost_usd:.2f} >= ${max_cost:.2f})"
    ctx.early_stop_reason = reason
    ctx.progress.on_message(f"[agent] {reason}")
    ctx.event_log.cost_limit_reached(it, ctx.total_estimated_cost_usd, max_cost)
    ctx.history.append(make_history_entry(
        it, f"early stop: {reason}", ctx.best_value, ctx.baseline_val,
        accepted=False, mode=ctx.task.benchmark.metric.mode,
    ))
    return True


def _write_state_snapshot(ctx: AgentContext, state_path: Path) -> None:
    """Persist per-iteration optimization state as a run artifact.

    Best-effort: write failures are logged, not raised, so a snapshot
    problem never aborts the run. Called from every loop exit path (not just
    the bottom of a fully-processed iteration) so state.json is never
    skipped for exactly the iterations where the LLM misbehaved worst (e.g.
    no candidates parsed).
    """
    try:
        state_path.write_text(
            json.dumps(ctx.to_dict(), indent=2), encoding="utf-8",
        )
    except (OSError, TypeError):
        logger.warning("Failed to save iteration state", exc_info=True)


def run_agent(
    task: TaskSpec,
    task_file: Path,
    config: AgentConfig,
    llm_config: LLMConfig,
    expert_suggestion: str | None = None,
    progress: AgentProgress | None = None,
    provider: LLMProvider | None = None,
) -> AgentResult:
    """Run the agentic beam-search optimizer.

    1. Baseline profile + benchmark
    2. Per iteration: build prompt -> LLM -> parse candidates -> evaluate each
    3. Accept best improving candidate, re-apply permanently, re-profile
    4. Generate reports
    """
    # Validate contract structure before spending time on benchmarks
    contract_errors = task.contract.validate()
    if contract_errors:
        raise ValueError(f"Invalid contract in task.yaml: {'; '.join(contract_errors)}")

    wall_start = time.monotonic()

    if progress is None:
        progress = PrintProgress()

    if provider is None:
        provider = create_provider(llm_config)
    ws = task.workspace
    run_store = RunStore(task.out_dir)
    rp = run_store.new_run(task.name, program_type=task.program_type)

    # Event logging
    event_log = AgentEventLog(rp.run_dir)

    # Construct AgentContext
    ctx = AgentContext(
        task=task,
        config=config,
        llm_config=llm_config,
        provider=provider,
        progress=progress,
        ws=ws,
        rp=rp,
        event_log=event_log,
        expert_suggestion=expert_suggestion,
        convergence=ConvergenceDetector() if config.early_stop else None,
        wall_start=wall_start,
    )

    # Tamper guard: candidate evaluation runs in disposable workspace copies,
    # but accepted code still executes in the real workspace (autotune,
    # re-profiling, drift checks) and could rewrite tests.py/bench.py/task.yaml
    # at runtime. Snapshot them up front; verify + restore after each iteration.
    protected_dir = rp.run_dir / "protected_snapshot"
    protected_hashes = snapshot_protected_files(ws, protected_dir)

    baseline.run(ctx)

    # A crash (or Ctrl-C) hours into a run must not lose the reports and
    # finalize output for the iterations that did complete: finalize with
    # partial state on the failure path, then re-raise for the caller.
    try:
        _run_iteration_loop(ctx, rp.run_dir / "state.json", protected_dir, protected_hashes)
    except BaseException as exc:  # noqa: BLE001 -- re-raised below after best-effort finalize
        ctx.early_stop_reason = ctx.early_stop_reason or f"crashed: {type(exc).__name__}: {exc}"
        try:
            finalize.run(ctx, status="failed")
        except Exception:  # noqa: BLE001 -- crash-path reporting is best-effort; the original error must surface
            logger.warning("Report generation after crash failed", exc_info=True)
        raise

    finalize.run(ctx)

    return AgentResult(
        best_value=ctx.best_value,
        best_iter=ctx.best_iter,
        baseline_value=ctx.baseline_val,
        history=ctx.history,
        run_dir=rp.run_dir,
    )


def _run_iteration_loop(
    ctx: AgentContext,
    state_path: Path,
    protected_dir: Path,
    protected_hashes: dict[str, str],
) -> None:
    """The optimizer's iteration loop, extracted from run_agent so the caller
    can wrap it in crash-safe finalize handling.

    Mutates ctx throughout; returns when the iteration budget is exhausted or
    an early-stop condition breaks the loop.
    """
    task = ctx.task
    config = ctx.config
    progress = ctx.progress
    event_log = ctx.event_log

    for it in range(1, config.max_iters + 1):
        ctx.iteration = it
        iteration_errors: list[dict] = []
        # Wall-clock budget check
        elapsed = time.monotonic() - ctx.wall_start
        if elapsed > config.max_wall_time_s:
            reason = f"Wall-clock budget exceeded ({elapsed:.0f}s > {config.max_wall_time_s}s)"
            progress.on_message(f"[agent] {reason}")
            event_log.early_stop(it, reason)
            ctx.history.append(make_history_entry(it, f"early stop: {reason}", ctx.best_value, ctx.baseline_val, accepted=False, mode=task.benchmark.metric.mode))
            break

        iter_start = time.monotonic()
        remaining = max(0, config.max_wall_time_s - elapsed)
        progress.on_message(f"\n[agent] === Iteration {it}/{config.max_iters} === [elapsed {fmt_elapsed(elapsed)}, {fmt_elapsed(remaining)} remaining]")

        if ctx.last_errors:
            progress.on_message(f"[agent] Feeding {len(ctx.last_errors)} error(s) from previous iteration back to LLM")
            event_log.error_feedback(it, ctx.last_errors)
        if task.constraints.prompt_token_budget > 0:
            progress.on_message(f"[agent] Prompt token budget: {task.constraints.prompt_token_budget}")

        # Build prompt, call LLM, and parse candidates
        gen_result = generate.run(ctx)

        # Cost guard: recompute cumulative estimated cost from this iteration's
        # (now-updated) token totals and stop gracefully if --max-cost is reached.
        if _update_and_check_cost_limit(ctx, it):
            break

        if gen_result.llm_failed:
            event_log.iteration_complete(it, ctx.best_value, False)
            _write_state_snapshot(ctx, state_path)
            continue
        candidate_blocks = gen_result.candidate_blocks
        candidate_reasoning = gen_result.candidate_reasoning
        # Truncated-output / dropped-block errors feed back to the LLM next iteration
        iteration_errors.extend(gen_result.generation_errors)

        if not candidate_blocks:
            progress.on_message("[agent] No valid candidates parsed from LLM response")
            # Preserve generation errors (e.g. truncated output) for the next
            # prompt even though this iteration ends early.
            ctx.last_errors = iteration_errors
            ctx.history.append(make_history_entry(
                it, "no candidates parsed", ctx.best_value, ctx.baseline_val,
                accepted=False, mode=task.benchmark.metric.mode,
            ))
            if ctx.convergence:
                ctx.convergence.record_failure()
            event_log.iteration_complete(it, ctx.best_value, False)
            # Snapshot state here too -- this branch used to `continue` before
            # reaching the snapshot write at the bottom of the loop, so
            # state.json was never written for exactly the iterations where
            # the LLM misbehaved worst (no parseable candidates).
            _write_state_snapshot(ctx, state_path)
            if finalize.maybe_early_stop(ctx, it):
                break
            continue

        progress.on_message(f"[agent] Parsed {len(candidate_blocks)} candidates")

        # Phase 1: Parallel prescreen (build + correctness, no benchmark)
        prescreen_results = prescreen.run(
            ctx, candidate_blocks, candidate_reasoning,
        )

        # Collect prescreen errors into failure tracking
        for pr in prescreen_results:
            if not pr.get("passed") and pr.get("error"):
                iteration_errors.append(pr["error"])

        # Phase 2: Sequential benchmark (GPU-bound, only for survivors)
        candidates: list[evaluate.BeamCandidate] = []
        use_fast = config.fast_screen and len(candidate_blocks) > 1

        survivors = [pr for pr in prescreen_results if pr.get("passed")]
        failed = [pr for pr in prescreen_results if not pr.get("passed")]

        # Create BeamCandidates for failed prescreens
        for pr in failed:
            ci = pr["ci"]
            err_desc = pr.get("error", {}).get("description", "prescreen failed")
            candidates.append(evaluate.BeamCandidate(
                iteration=it, index=ci, blocks=pr["blocks"],
                description=f"candidate {ci + 1}: {len(pr['blocks'])} blocks ({err_desc})",
                reasoning=pr.get("reasoning", ""),
            ))
            event_log.candidate_validation(it, ci, False, [err_desc])

        # Benchmark survivors sequentially (GPU contention)
        for pr in survivors:
            ci = pr["ci"]
            blocks = pr["blocks"]
            reasoning = pr.get("reasoning", "")
            candidate, eval_errors = evaluate.evaluate_single_candidate(
                ctx, ci, blocks, reasoning, use_fast,
            )
            candidates.append(candidate)
            iteration_errors.extend(eval_errors)

        # Find best improving candidate and accept it
        accepted_any, rel_improvement, accepted_value = evaluate.accept_best(
            ctx, candidates, use_fast,
        )
        if accepted_any:
            assert accepted_value is not None  # accept_best guarantees a value when accepted_any is True
            # Tamper guard: verify + restore now, before autotune's up-to-15
            # correctness+bench trials execute the accepted code against the
            # live workspace -- otherwise a candidate that rewrites tests.py
            # at runtime poisons every sweep decision until the (previously
            # only) end-of-iteration check below caught it.
            _verify_and_warn_protected_files(ctx, protected_dir, protected_hashes)
            # Auto-tune: if tuning.yaml has sweep parameters, run a quick sweep
            # to find optimal parameter values for the new code
            autotune.run(ctx)
            # Re-profile with accepted changes (+ periodic drift check)
            evaluate.reprofile_after_accept(ctx, accepted_value)

        # End-of-iteration check: also catches tampering during autotune,
        # reprofiling, or the drift-check re-benchmark above.
        _verify_and_warn_protected_files(ctx, protected_dir, protected_hashes)

        if accepted_any and ctx.convergence and rel_improvement is not None:
            ctx.convergence.record_improvement(rel_improvement)
        elif not accepted_any and ctx.convergence:
            ctx.convergence.record_failure()

        ctx.last_errors = iteration_errors

        # Collect promising alternatives (improved but not best)
        ctx.promising_alternatives = (
            _collect_promising_alternatives(
                candidates, ctx.baseline_val, task.benchmark.metric.mode,
            )
            if accepted_any else []
        )

        # Accumulate structured failure memory from this iteration's errors
        for ci, cand in enumerate(candidates):
            if cand.value is None:  # Failed candidate
                reasoning_text = cand.reasoning or cand.description
                for err in iteration_errors:
                    if f"candidate {ci + 1}" in err.get("description", ""):
                        ctx.failure_memory.append({
                            "iteration": it,
                            "strategy": reasoning_text[:200],
                            "failure_type": err.get("type", "unknown"),
                            "reason": err.get("description", "")[:300],
                            "profiler_context": err.get("output", "")[:200] if err.get("output") else None,
                        })
                        break

        iter_elapsed = time.monotonic() - iter_start
        event_log.iteration_complete(it, ctx.best_value, accepted_any)
        progress.on_message(f"[agent] Iteration {it} completed in {fmt_elapsed(iter_elapsed)}")

        # Persist per-iteration optimization state as a run artifact for
        # debugging and post-hoc inspection.
        _write_state_snapshot(ctx, state_path)

        if finalize.maybe_early_stop(ctx, it):
            break
