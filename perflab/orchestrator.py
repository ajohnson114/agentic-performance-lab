from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from perflab.analyzers.bench_stats import extract_repeated_values
from perflab.analyzers.decision import (
    Comparison,
    DecisionRule,
    ImprovementVerdict,
    Mode,
    rule_for_constraints,
)
from perflab.memory.run_store import RunStore
from perflab.optimizers.propose_params import (
    load_knobs,
    propose_knob_sweep,
    sample_candidates,
    save_knobs,
)
from perflab.reporting.generate import ReportParams, generate_reports
from perflab.roofline_peaks import resolve_roofline
from perflab.runners.benchmark import metric_value
from perflab.runners.pipeline import run_pipeline
from perflab.task_spec import TaskSpec

logger = logging.getLogger(__name__)


@dataclass
class IterationRow:
    iter: int
    value: float
    accepted: bool
    notes: str = ""


def _judge(
    rule: DecisionRule,
    *,
    candidate: float,
    incumbent: float,
    mode: Mode,
    tolerance: float,
    candidate_samples: list[float],
    incumbent_samples: list[float],
    label: str,
) -> ImprovementVerdict:
    """Decide whether a knob configuration beats the incumbent, and say why.

    The knob search used to call the bare ratio test directly, so it accepted
    any configuration whose measured value happened to land ``tolerance`` above
    the incumbent -- including pure run-to-run jitter, which it then wrote back
    into tuning.yaml as the user's new default. It now asks the same module as
    the agent beam search and the CI regression check
    (``perflab.analyzers.decision``), so a "win" this machine cannot resolve is
    rejected.

    Every decision is logged, in all three outcomes: accepted, accepted
    unverified (no per-repeat samples were published, so the historical ratio
    is all that ran), and rejected with the bar that was missed. A sweep that
    silently stops picking winners is a support ticket; a sweep that says
    "improvement within noise (CV=8.0% at n=5 ...)" is an answer.
    """
    verdict = rule.decide(Comparison(
        candidate=candidate,
        incumbent=incumbent,
        mode=mode,
        tolerance=tolerance,
        candidate_samples=candidate_samples or None,
        incumbent_samples=incumbent_samples or None,
    ))
    if not verdict.improved:
        # A trial that never cleared the tolerance is the ordinary, expected
        # outcome of a sweep -- INFO. A trial that DID clear it and was rejected
        # anyway is the surprising one: the knob looked like a win and the
        # machine could not resolve it. That is a behavior change from the old
        # bare-ratio search, and the CLI runs at WARNING by default, so at INFO
        # the user would watch the sweep reject everything and be told nothing.
        # The reason names the fix (more repeats, a quieter host, or a wider
        # tolerance), which is only useful if it is actually shown.
        level = logging.WARNING if verdict.beats_tolerance else logging.INFO
        logger.log(level, "%s: rejected — %s", label, verdict.reason)
    elif verdict.verified:
        logger.info(
            "%s: accepted (%+.2f%% vs incumbent, needed %+.2f%%)",
            label, verdict.observed * 100, verdict.required * 100,
        )
    else:
        logger.info(
            "%s: accepted (%+.2f%%) without variance verification — the "
            "benchmark published no per-repeat samples, so only the %.1f%% "
            "tolerance was checked",
            label, verdict.observed * 100, verdict.tolerance * 100,
        )
    return verdict


def _rejection_note(base: str, verdict: ImprovementVerdict) -> str:
    """Row note for a rejected trial, carrying a noise rejection's explanation.

    Only annotates the case the note cannot otherwise convey: the trial DID
    beat the tolerance and was thrown away anyway because the measurement could
    not resolve it. A plain "did not beat the tolerance" rejection is already
    obvious from the row's value, and is left with its bare description.
    """
    if verdict.beats_tolerance and not verdict.improved and verdict.reason:
        return f"{base}: {verdict.reason}"
    return base


def profile_only(task: TaskSpec) -> Path:
    contract_errors = task.contract.validate()
    if contract_errors:
        raise ValueError(f"Invalid contract in task.yaml: {'; '.join(contract_errors)}")

    run_store = RunStore(task.out_dir)
    rp = run_store.new_run(task.name, program_type=task.program_type)
    try:
        from perflab.tools.sysinfo import capture_system_info
        capture_system_info(rp.run_dir)
    except Exception:  # noqa: BLE001 -- best-effort system info capture, must not abort the profiling run
        logger.warning("Failed to collect system info", exc_info=True)

    result = run_pipeline(
        task, rp.run_dir, rp.run_dir / "artifacts",
        do_profiles=True, capture_diagnostics=True,
        save_logs=True, validate_contract_spec=True,
    )

    # Generate reports (dashboard + markdown) so the printed links work
    val = metric_value(result.bench, task.benchmark.metric.name)
    history = [
        {"iteration": 0, "value": val, "accepted": True,
         "description": "profile", "delta": 0.0, "speedup": 1.0},
    ]
    generate_reports(ReportParams(
        run_dir=rp.run_dir,
        run_id=rp.run_id,
        task_name=task.name,
        metric_name=task.benchmark.metric.name,
        metric_mode=task.benchmark.metric.mode,
        program_type=task.program_type,
        history=history,
        baseline_val=val,
        best_value=val,
        best_iter=0,
        roofline_peaks=resolve_roofline(task),
        target_hardware=task.target_hardware,
        build_cmd=task.build.cmd if task.build else None,
        top_n=task.constraints.top_n,
    ))
    run_store.update_meta(rp.run_id, {
        "status": "profiled",
        "best_value": val,
        "completed_at": time.strftime("%Y%m%d-%H%M%S"),
    })
    return rp.run_dir

def optimize(task: TaskSpec, iters: int | None = None, max_trials: int | None = None) -> Path:
    # Validate contract structure before spending time on benchmarks
    contract_errors = task.contract.validate()
    if contract_errors:
        raise ValueError(f"Invalid contract in task.yaml: {'; '.join(contract_errors)}")

    max_iters = iters or task.constraints.max_iters
    run_store = RunStore(task.out_dir)
    rp = run_store.new_run(task.name, program_type=task.program_type)
    artifacts_dir = rp.run_dir / "artifacts"

    try:
        from perflab.tools.sysinfo import capture_system_info
        capture_system_info(rp.run_dir)
    except Exception:  # noqa: BLE001 -- best-effort system info capture, must not abort the optimize run
        logger.warning("Failed to collect system info", exc_info=True)

    rows: list[IterationRow] = []
    best_value = None
    best_iter = 0

    # One accept/reject rule for the whole sweep, resolved from the task the
    # same way the agent and ci-check resolve theirs (perflab.analyzers.decision).
    metric_name = task.benchmark.metric.name
    mode = task.benchmark.metric.mode
    tol = task.constraints.regression_tolerance
    rule = rule_for_constraints(task.constraints)

    # Work in the task workspace; write outputs under workspace/out (task harness convention)
    ws = task.workspace
    knobs_path = ws / "tuning.yaml"

    # Baseline
    baseline_result = run_pipeline(
        task, rp.run_dir, artifacts_dir,
        do_profiles=True, capture_diagnostics=True,
        save_logs=True, validate_contract_spec=True,
    )
    v = metric_value(baseline_result.bench, metric_name)
    best_value = v
    baseline_value = v
    # Per-repeat samples behind each point estimate, when the harness published
    # them; [] degrades the decision to the historical ratio test (and says so).
    baseline_samples = extract_repeated_values(baseline_result.bench, metric_name)
    best_samples = baseline_samples
    rows.append(IterationRow(iter=0, value=v, accepted=True, notes="baseline"))
    best_iter = 0

    # Save baseline knobs snapshot
    if knobs_path.exists():
        shutil.copy2(knobs_path, rp.run_dir / "knobs_iter0.yaml")

    if not knobs_path.exists():
        rows.append(IterationRow(iter=1, value=v, accepted=False, notes="no tuning.yaml; stopping"))
    else:
        current_knobs = load_knobs(knobs_path)
        candidates = propose_knob_sweep(current_knobs)

        if max_trials:
            candidates = sample_candidates(candidates, max_trials)

        if not candidates:
            rows.append(IterationRow(iter=1, value=v, accepted=False, notes="no candidates; stopping"))
        elif current_knobs.get("sweep"):
            # Grid search mode: evaluate all candidates, keep the best
            best_knobs = {k: v for k, v in current_knobs.items() if k != "sweep"}
            for trial, cand in enumerate(candidates, 1):
                save_knobs(knobs_path, cand.new_knobs)
                try:
                    trial_result = run_pipeline(
                        task, rp.run_dir, artifacts_dir,
                        save_logs=True, validate_contract_spec=True,
                    )
                    vi = metric_value(trial_result.bench, metric_name)
                    vi_samples = extract_repeated_values(trial_result.bench, metric_name)
                except Exception as exc:  # noqa: BLE001 -- a single bad trial must not abort the whole sweep
                    rows.append(IterationRow(iter=trial, value=rows[-1].value, accepted=False, notes=f"{cand.description} (error: {exc})"))
                    save_knobs(knobs_path, {k: v for k, v in current_knobs.items() if k != "sweep"})
                    continue

                verdict = _judge(
                    rule, candidate=vi, incumbent=best_value, mode=mode, tolerance=tol,
                    candidate_samples=vi_samples, incumbent_samples=best_samples,
                    label=f"trial {trial} ({cand.description})",
                )
                rows.append(IterationRow(
                    iter=trial, value=vi, accepted=verdict.improved,
                    notes=_rejection_note(cand.description, verdict),
                ))
                if verdict.improved:
                    best_value = vi
                    best_samples = vi_samples
                    best_iter = trial
                    best_knobs = dict(cand.new_knobs)
                    shutil.copy2(knobs_path, rp.run_dir / f"knobs_trial{trial}.yaml")

                # Revert for next trial
                save_knobs(knobs_path, {k: v for k, v in current_knobs.items() if k != "sweep"})

            # Write the winning knobs and confirm with a full re-benchmark.
            # Keep the sweep: section — optimize must not permanently rewrite
            # the user's tuning.yaml into legacy (no-sweep) mode.
            save_knobs(knobs_path, {**best_knobs, "sweep": current_knobs["sweep"]})
            if best_iter > 0:
                try:
                    confirm_result = run_pipeline(
                        task, rp.run_dir, artifacts_dir,
                        save_logs=True, validate_contract_spec=True,
                    )
                    confirmed_val = metric_value(confirm_result.bench, metric_name)
                    confirmed_samples = extract_repeated_values(confirm_result.bench, metric_name)
                    # Against the *baseline*, not the sweep's best: this asks
                    # "does the winner still beat where we started", which is
                    # what the user is being handed.
                    confirm_verdict = _judge(
                        rule, candidate=confirmed_val, incumbent=baseline_value,
                        mode=mode, tolerance=tol,
                        candidate_samples=confirmed_samples,
                        incumbent_samples=baseline_samples,
                        label=f"confirmation re-benchmark (trial {best_iter})",
                    )
                    if confirm_verdict.improved:
                        best_value = confirmed_val
                        best_samples = confirmed_samples
                        rows.append(IterationRow(iter=best_iter, value=confirmed_val, accepted=True, notes="confirmed re-benchmark"))
                    else:
                        # Put the user's original knobs back. The winning
                        # configuration was written above, BEFORE the
                        # confirmation ran -- so without this the sweep leaves
                        # behind a tuning.yaml whose improvement it just
                        # rejected, which is the opposite of what the gate is
                        # for. (The window was always here; it was near-
                        # unreachable while confirmation was a bare ratio,
                        # because anything that won a sweep also cleared a 2%
                        # threshold. A gate that can say "not distinguishable
                        # from noise" makes this the common path.)
                        save_knobs(knobs_path, current_knobs)
                        best_knobs = {
                            k: v for k, v in current_knobs.items() if k != "sweep"
                        }
                        best_value = baseline_value
                        best_samples = baseline_samples
                        best_iter = 0
                        rows.append(IterationRow(
                            iter=best_iter, value=confirmed_val, accepted=False,
                            notes=_rejection_note(
                                "confirmation re-benchmark did not hold; "
                                "reverted tuning.yaml to its pre-sweep values",
                                confirm_verdict,
                            ),
                        ))
                except Exception:  # noqa: BLE001 -- best-effort confirmation re-benchmark, keep the sweep's winner if it fails
                    logger.warning("Confirmation re-benchmark failed", exc_info=True)
        else:
            # Legacy mode: iterate until no improvement
            for it in range(1, max_iters + 1):
                accepted_any = False
                # Rejected candidates get no row of their own here (only the
                # per-iteration summary below), so a noise rejection would
                # otherwise vanish from the report. Keep the first one to say
                # why the loop stopped.
                noise_reason = ""
                for cand in candidates:
                    save_knobs(knobs_path, cand.new_knobs)
                    try:
                        iter_result = run_pipeline(
                            task, rp.run_dir, artifacts_dir,
                            save_logs=True, validate_contract_spec=True,
                        )
                        vi = metric_value(iter_result.bench, metric_name)
                        vi_samples = extract_repeated_values(iter_result.bench, metric_name)
                    except Exception as exc:  # noqa: BLE001 -- a single bad candidate must not abort the run (mirrors the sweep path above)
                        rows.append(IterationRow(iter=it, value=rows[-1].value, accepted=False, notes=f"{cand.description} (error: {exc})"))
                        save_knobs(knobs_path, current_knobs)
                        continue

                    verdict = _judge(
                        rule, candidate=vi, incumbent=best_value, mode=mode, tolerance=tol,
                        candidate_samples=vi_samples, incumbent_samples=best_samples,
                        label=f"iteration {it} ({cand.description})",
                    )
                    if verdict.improved:
                        best_value = vi
                        best_samples = vi_samples
                        best_iter = it
                        accepted_any = True
                        rows.append(IterationRow(iter=it, value=vi, accepted=True, notes=cand.description))
                        shutil.copy2(knobs_path, rp.run_dir / f"knobs_iter{it}.yaml")
                        break
                    else:
                        if not noise_reason and verdict.beats_tolerance and verdict.reason:
                            noise_reason = f"{cand.description}: {verdict.reason}"
                        save_knobs(knobs_path, current_knobs)

                if not accepted_any:
                    notes = "no improvement"
                    if noise_reason:
                        notes = f"{notes} ({noise_reason})"
                    rows.append(IterationRow(iter=it, value=rows[-1].value, accepted=False, notes=notes))
                    break

                # Reload knobs for next iteration (may have changed)
                current_knobs = load_knobs(knobs_path)
                candidates = propose_knob_sweep(current_knobs)
                if not candidates:
                    break

    # Profile the final (current) state for artifacts
    run_pipeline(
        task, rp.run_dir, artifacts_dir,
        do_profiles=True, capture_diagnostics=True,
        save_logs=True, validate_contract_spec=True,
    )

    # Convert rows to history dicts for generate_reports
    history = [
        {"iteration": r.iter, "value": r.value, "accepted": r.accepted,
         "notes": r.notes, "delta": r.value - baseline_value,
         "speedup": r.value / baseline_value if baseline_value != 0 else 1.0}
        for r in rows
    ]
    generate_reports(ReportParams(
        run_dir=rp.run_dir,
        run_id=rp.run_id,
        task_name=task.name,
        metric_name=task.benchmark.metric.name,
        metric_mode=task.benchmark.metric.mode,
        program_type=task.program_type,
        history=history,
        baseline_val=baseline_value,
        best_value=best_value,
        best_iter=best_iter,
        optimization_summary_text=None,
        roofline_peaks=resolve_roofline(task),
        top_n=task.constraints.top_n,
    ))
    return rp.run_dir
