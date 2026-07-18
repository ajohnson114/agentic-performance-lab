from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from perflab.analyzers.compiler_diagnostics import CompilerDiagnostics
from perflab.llm.base import LLMProvider
from perflab.llm.config import LLMConfig, create_provider
from perflab.memory.run_store import RunPaths, RunStore
from perflab.optimizers.convergence import ConvergenceDetector
from perflab.optimizers.event_log import AgentEventLog
from perflab.optimizers.history import make_history_entry
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
    diversity_penalty: float = 0.1
    early_stop: bool = True
    fast_screen: bool = True
    max_wall_time_s: int = 3600
    # OS-level sandboxing for candidate subprocess execution (see
    # perflab.tools.isolation). None means rlimits/env-allowlist only.
    # Applied uniformly to baseline AND candidate benchmark/correctness runs
    # so sandbox overhead cancels out of speedup comparisons.
    isolation: IsolationPolicy | None = None


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

    baseline.run(ctx)
    state_path = rp.run_dir / "state.json"

    # --- Iterate ---
    for it in range(1, config.max_iters + 1):
        ctx.iteration = it
        iteration_errors: list[dict] = []
        # Wall-clock budget check
        elapsed = time.monotonic() - ctx.wall_start
        if elapsed > config.max_wall_time_s:
            reason = f"Wall-clock budget exceeded ({elapsed:.0f}s > {config.max_wall_time_s}s)"
            progress.on_message(f"[agent] {reason}")
            event_log.early_stop(it, reason)
            ctx.history.append(make_history_entry(it, f"early stop: {reason}", ctx.best_value, ctx.baseline_val, accepted=False))
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
        if gen_result.llm_failed:
            event_log.iteration_complete(it, ctx.best_value, False)
            continue
        candidate_blocks = gen_result.candidate_blocks
        candidate_reasoning = gen_result.candidate_reasoning

        if not candidate_blocks:
            progress.on_message("[agent] No valid candidates parsed from LLM response")
            ctx.history.append(make_history_entry(
                it, "no candidates parsed", ctx.best_value, ctx.baseline_val,
                accepted=False,
            ))
            if ctx.convergence:
                ctx.convergence.record_failure()
            event_log.iteration_complete(it, ctx.best_value, False)
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
        backup_dir = rp.run_dir / "backups" / f"iter{it}"
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
            ctx, candidates, backup_dir, use_fast,
        )
        if accepted_any:
            assert accepted_value is not None  # accept_best guarantees a value when accepted_any is True
            # Auto-tune: if tuning.yaml has sweep parameters, run a quick sweep
            # to find optimal parameter values for the new code
            autotune.run(ctx)
            # Re-profile with accepted changes (+ periodic drift check)
            evaluate.reprofile_after_accept(ctx, accepted_value)

        if accepted_any and ctx.convergence and rel_improvement is not None:
            ctx.convergence.record_improvement(rel_improvement)
        elif not accepted_any and ctx.convergence:
            ctx.convergence.record_failure()

        ctx.last_errors = iteration_errors

        # Collect promising alternatives (improved but not best)
        ctx.promising_alternatives = []
        if accepted_any:
            for cand in candidates:
                if cand.value is not None and cand.value > ctx.baseline_val and not cand.accepted:
                    ctx.promising_alternatives.append({
                        "description": cand.description,
                        "reasoning": cand.reasoning,
                        "value": cand.value,
                        "improvement": round(cand.value / ctx.baseline_val, 2) if ctx.baseline_val > 0 else 0,
                    })
            # Sort by value descending
            ctx.promising_alternatives.sort(key=lambda x: x.get("value", 0), reverse=True)
            ctx.promising_alternatives = ctx.promising_alternatives[:3]  # Top 3

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
        try:
            state_path.write_text(
                json.dumps(ctx.to_dict(), indent=2), encoding="utf-8",
            )
        except (OSError, TypeError):
            logger.warning("Failed to save iteration state", exc_info=True)

        if finalize.maybe_early_stop(ctx, it):
            break

    finalize.run(ctx)

    return AgentResult(
        best_value=ctx.best_value,
        best_iter=ctx.best_iter,
        baseline_value=ctx.baseline_val,
        history=ctx.history,
        run_dir=rp.run_dir,
    )
