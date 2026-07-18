from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from perflab.analyzers.metrics_rollup import is_improvement
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

def profile_only(task: TaskSpec) -> Path:
    contract_errors = task.contract.validate()
    if contract_errors:
        raise ValueError(f"Invalid contract in task.yaml: {'; '.join(contract_errors)}")

    run_store = RunStore(task.out_dir)
    rp = run_store.new_run(task.name, program_type=task.program_type)
    # Capture system info
    try:
        from perflab.tools.sysinfo import collect_system_info
        sysinfo = collect_system_info()
        (rp.run_dir / "system_info.json").write_text(
            json.dumps(sysinfo, indent=2), encoding="utf-8"
        )
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

    # Capture system info
    try:
        from perflab.tools.sysinfo import collect_system_info
        sysinfo = collect_system_info()
        (rp.run_dir / "system_info.json").write_text(
            json.dumps(sysinfo, indent=2), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 -- best-effort system info capture, must not abort the optimize run
        logger.warning("Failed to collect system info", exc_info=True)

    rows: list[IterationRow] = []
    best_value = None
    best_iter = 0

    # Work in the task workspace; write outputs under workspace/out (task harness convention)
    ws = task.workspace
    knobs_path = ws / "tuning.yaml"

    # Baseline
    baseline_result = run_pipeline(
        task, rp.run_dir, artifacts_dir,
        do_profiles=True, capture_diagnostics=True,
        save_logs=True, validate_contract_spec=True,
    )
    v = metric_value(baseline_result.bench, task.benchmark.metric.name)
    best_value = v
    baseline_value = v
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
                    vi = metric_value(trial_result.bench, task.benchmark.metric.name)
                except Exception as exc:  # noqa: BLE001 -- a single bad trial must not abort the whole sweep
                    rows.append(IterationRow(iter=trial, value=rows[-1].value, accepted=False, notes=f"{cand.description} (error: {exc})"))
                    save_knobs(knobs_path, {k: v for k, v in current_knobs.items() if k != "sweep"})
                    continue

                improved = is_improvement(vi, best_value, task.benchmark.metric.mode, task.constraints.regression_tolerance)
                rows.append(IterationRow(iter=trial, value=vi, accepted=improved, notes=cand.description))
                if improved:
                    best_value = vi
                    best_iter = trial
                    best_knobs = dict(cand.new_knobs)
                    shutil.copy2(knobs_path, rp.run_dir / f"knobs_trial{trial}.yaml")

                # Revert for next trial
                save_knobs(knobs_path, {k: v for k, v in current_knobs.items() if k != "sweep"})

            # Write the winning knobs and confirm with a full re-benchmark
            save_knobs(knobs_path, best_knobs)
            if best_iter > 0:
                try:
                    confirm_result = run_pipeline(
                        task, rp.run_dir, artifacts_dir,
                        save_logs=True, validate_contract_spec=True,
                    )
                    confirmed_val = metric_value(confirm_result.bench, task.benchmark.metric.name)
                    if is_improvement(confirmed_val, baseline_value, task.benchmark.metric.mode, task.constraints.regression_tolerance):
                        best_value = confirmed_val
                        rows.append(IterationRow(iter=best_iter, value=confirmed_val, accepted=True, notes="confirmed re-benchmark"))
                    else:
                        rows.append(IterationRow(iter=best_iter, value=confirmed_val, accepted=False, notes="confirmation re-benchmark did not hold"))
                except Exception:  # noqa: BLE001 -- best-effort confirmation re-benchmark, keep the sweep's winner if it fails
                    logger.warning("Confirmation re-benchmark failed", exc_info=True)
        else:
            # Legacy mode: iterate until no improvement
            for it in range(1, max_iters + 1):
                accepted_any = False
                for cand in candidates:
                    save_knobs(knobs_path, cand.new_knobs)
                    iter_result = run_pipeline(
                        task, rp.run_dir, artifacts_dir,
                        save_logs=True, validate_contract_spec=True,
                    )
                    vi = metric_value(iter_result.bench, task.benchmark.metric.name)

                    if is_improvement(vi, best_value, task.benchmark.metric.mode, task.constraints.regression_tolerance):
                        best_value = vi
                        best_iter = it
                        accepted_any = True
                        rows.append(IterationRow(iter=it, value=vi, accepted=True, notes=cand.description))
                        shutil.copy2(knobs_path, rp.run_dir / f"knobs_iter{it}.yaml")
                        break
                    else:
                        save_knobs(knobs_path, current_knobs)

                if not accepted_any:
                    rows.append(IterationRow(iter=it, value=rows[-1].value, accepted=False, notes="no improvement"))
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
