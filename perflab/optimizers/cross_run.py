"""Cross-run learning: mine knowledge from prior optimization runs.

When the agent is invoked on a task that has prior runs, this module
loads summaries of what worked/didn't from those runs so the LLM
can avoid repeating failed approaches and build on successful ones.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_prior_run_context(out_dir: Path, current_run_id: str | None = None) -> str | None:
    """Scan prior runs in out_dir and build a context summary.

    Returns a formatted string suitable for inclusion in the LLM prompt,
    or None if no prior runs exist.
    """
    runs_dir = out_dir / "runs"
    if not runs_dir.exists():
        return None

    prior_runs: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        # Skip the current run
        if current_run_id and run_dir.name == current_run_id:
            continue

        report_json = run_dir / "report.json"
        if not report_json.exists():
            continue

        try:
            data = json.loads(report_json.read_text(encoding="utf-8"))
            prior_runs.append(_summarize_run(data, run_dir))
        except (json.JSONDecodeError, OSError, KeyError):
            logger.warning("Failed to load prior run %s", run_dir.name, exc_info=True)
            continue

    if not prior_runs:
        return None

    # Build context string
    sections = [
        "## Prior optimization runs on this task\n",
        "Note: The current source code may differ from what these runs started with. "
        "Use these results as strategy guidance, not as assumptions about current code state.\n",
    ]
    for run in prior_runs[-3:]:  # Last 3 runs to keep context manageable
        sections.append(_format_run_summary(run))

    return "\n".join(sections)


def load_prior_event_insights(out_dir: Path, current_run_id: str | None = None) -> list[dict]:
    """Extract key insights from prior agent_events.jsonl logs.

    Returns a list of insight dicts with keys: type, description, iteration, value.
    """
    runs_dir = out_dir / "runs"
    if not runs_dir.exists():
        return []

    insights: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if current_run_id and run_dir.name == current_run_id:
            continue

        events_file = run_dir / "agent_events.jsonl"
        if not events_file.exists():
            continue

        try:
            run_insights = _extract_event_insights(events_file)
            insights.extend(run_insights)
        except (json.JSONDecodeError, OSError, KeyError):
            logger.warning("Failed to extract event insights from %s", events_file, exc_info=True)
            continue

    return insights[-20:]  # Cap at 20 most recent insights


def _summarize_run(data: dict, run_dir: Path) -> dict:
    """Extract key info from a report.json."""
    rows = data.get("rows", [])
    accepted = [r for r in rows if r.get("accepted") and r.get("iter", 0) > 0]
    rejected = [r for r in rows if not r.get("accepted") and r.get("iter", 0) > 0]

    # Load optimization summary if available
    summary_text = None
    summary_path = run_dir / "optimization_summary.md"
    if summary_path.exists():
        try:
            summary_text = summary_path.read_text(encoding="utf-8")[:500]
        except OSError:
            pass

    return {
        "run_id": data.get("run_id", run_dir.name),
        "task_name": data.get("task_name", ""),
        "baseline_value": data.get("baseline_value"),
        "best_value": data.get("best_value"),
        "best_iter": data.get("best_iter"),
        "metric_name": data.get("metric_name", ""),
        "accepted_patches": [
            {"iter": r.get("iter"), "notes": r.get("notes", "")[:200]}
            for r in accepted
        ],
        "rejected_attempts": [
            {"iter": r.get("iter"), "notes": r.get("notes", "")[:200]}
            for r in rejected[:5]  # Limit rejected to 5
        ],
        "optimization_summary": summary_text,
        "bottleneck_diagnoses": data.get("bottleneck_diagnoses", [])[:3],
    }


def _format_run_summary(run: dict) -> str:
    """Format a single prior run summary for LLM consumption."""
    lines = [f"### Run {run['run_id']}"]

    baseline = run.get("baseline_value")
    best = run.get("best_value")
    metric = run.get("metric_name", "metric")
    if baseline is not None and best is not None:
        speedup = best / baseline if baseline != 0 else 1.0
        lines.append(f"Baseline: {baseline:.6g} → Best: {best:.6g} ({speedup:.2f}x) [{metric}]")

    if run.get("accepted_patches"):
        lines.append("\nWhat worked:")
        for p in run["accepted_patches"]:
            lines.append(f"  - Iter {p['iter']}: {p['notes']}")

    if run.get("rejected_attempts"):
        lines.append("\nWhat didn't work:")
        for r in run["rejected_attempts"]:
            lines.append(f"  - Iter {r['iter']}: {r['notes']}")

    if run.get("optimization_summary"):
        lines.append(f"\nSummary: {run['optimization_summary']}")

    lines.append("")
    return "\n".join(lines)


def _extract_event_insights(events_file: Path) -> list[dict]:
    """Extract actionable insights from agent_events.jsonl."""
    insights: list[dict] = []

    for line in events_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("event")

        # Track accepted candidates
        if event_type == "candidate_accepted":
            insights.append({
                "type": "success",
                "description": event.get("description", ""),
                "iteration": event.get("iteration"),
                "value": event.get("value"),
            })

        # Track early stops
        elif event_type == "early_stop":
            insights.append({
                "type": "early_stop",
                "description": event.get("reason", ""),
                "iteration": event.get("iteration"),
                "value": None,
            })

    return insights
