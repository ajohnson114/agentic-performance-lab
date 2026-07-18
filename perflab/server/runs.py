"""MCP tools for browsing, comparing, and replaying stored runs."""
from __future__ import annotations

import json
from pathlib import Path

from perflab.server.core import _guard_output_size, mcp

# ===========================================================================
# Run management
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": True})
def list_runs(task: str | None = None, limit: int = 20, out_dir: str = "out") -> list[dict]:
    """List stored optimization runs (newest first)."""
    from perflab.memory.run_store import RunStore
    store = RunStore(Path(out_dir))
    return store.list_runs(task=task, limit=limit)


@mcp.tool(annotations={"readOnlyHint": True})
def get_run(run_id: str, out_dir: str = "out") -> dict:
    """Get full details for a single run (meta, report, bench, profiler summaries).

    For large runs this may be truncated — use get_run_section for specific data.
    """
    from perflab.memory.run_store import RunStore
    store = RunStore(Path(out_dir))
    return _guard_output_size(store.get_run(run_id))


@mcp.tool(annotations={"readOnlyHint": True})
def get_run_section(run_id: str, section: str, out_dir: str = "out") -> dict:
    """Get a specific section of a run to avoid output size limits.

    Sections: meta, report, bench, profiler_summaries, system_info, event_log,
    or a specific profiler name (torch, nsys, ncu, jax, metal, pyspy, perf,
    memray, power, ebpf, lock_contention, thread_sched).
    """
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    run_data = store.get_run(run_id)
    run_dir = Path(out_dir) / "runs" / run_id

    if section in ("meta", "report", "bench"):
        data = run_data.get(section)
        if data is None:
            return {"error": f"Section '{section}' not found in run {run_id}"}
        return _guard_output_size(data)

    if section == "system_info":
        path = run_dir / "system_info.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"error": "system_info.json not found"}

    if section == "event_log":
        path = run_dir / "agent_events.jsonl"
        if path.exists():
            events = []
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            return _guard_output_size({"events": events})
        return {"error": "agent_events.jsonl not found"}

    if section == "profiler_summaries":
        return _guard_output_size(run_data.get("profiler_summaries", {}))

    # Try as a specific profiler name
    summaries = run_data.get("profiler_summaries", {})
    for key, data in summaries.items():
        if section in key:
            return _guard_output_size(data)

    available = ["meta", "report", "bench", "system_info", "event_log", "profiler_summaries"] + list(summaries.keys())
    return {"error": f"Unknown section: '{section}'", "available_sections": available}


@mcp.tool(annotations={"readOnlyHint": True})
def compare_runs(run_a: str, run_b: str, out_dir: str = "out") -> dict:
    """Compare two runs: values, delta, speedup, bottleneck diff."""
    from perflab.memory.run_store import RunStore
    store = RunStore(Path(out_dir))
    return store.compare_runs(run_a, run_b)


@mcp.tool(annotations={"readOnlyHint": True})
def replay_run(run_id: str, out_dir: str = "out") -> dict:
    """Replay and summarize an agent run from its event log."""
    from perflab.memory.run_store import validate_run_id
    from perflab.optimizers.event_log import replay_events

    try:
        validate_run_id(run_id)
    except ValueError as exc:
        return {"error": str(exc)}
    run_dir = Path(out_dir) / "runs" / run_id
    if not run_dir.exists():
        return {"error": f"Run directory not found: {run_dir}"}

    summary = replay_events(run_dir)
    return {"run_id": run_id, "summary": summary}
