"""Export profiler data to Chromium Trace Event JSON for Perfetto UI.

Converts py-spy hotspots and perf hardware counters into a trace file
that can be opened in https://ui.perfetto.dev/ for interactive timeline
visualization.  No instrumentation SDK required -- this re-uses data
already collected by the sampling profilers.

Format reference: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
"""
from __future__ import annotations

import json
from pathlib import Path


def export_perfetto_trace(
    output_path: Path,
    pyspy_summary: dict | None = None,
    perf_summary: dict | None = None,
    memray_summary: dict | None = None,
    metadata: dict | None = None,
) -> Path | None:
    """Write a Chromium Trace Event JSON file from profiler summaries.

    Returns the output path if events were written, None otherwise.
    """
    events: list[dict] = []
    pid = 1

    # --- Metadata ---
    if metadata:
        events.append({
            "name": "process_name",
            "ph": "M",
            "pid": pid,
            "tid": 0,
            "args": {"name": metadata.get("task_name", "PerfLab")},
        })

    # --- py-spy hotspots → synthetic flame chart ---
    if pyspy_summary and pyspy_summary.get("hotspots"):
        tid_cpu = 1
        events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": tid_cpu,
            "args": {"name": "CPU Hotspots (py-spy)"},
        })
        total_samples = pyspy_summary.get("total_samples", 100)
        # Scale: 1 sample = 1 microsecond in the synthetic timeline
        ts = 0
        for hotspot in pyspy_summary["hotspots"]:
            pct = hotspot.get("pct", 0)
            func = hotspot.get("function", "?")
            loc = hotspot.get("location", "")
            dur = int(pct / 100.0 * total_samples)
            if dur < 1:
                dur = 1
            event = {
                "name": func,
                "cat": "cpu",
                "ph": "X",
                "ts": ts,
                "dur": dur,
                "pid": pid,
                "tid": tid_cpu,
            }
            if loc:
                event["args"] = {"location": loc, "pct": f"{pct:.1f}%"}
            else:
                event["args"] = {"pct": f"{pct:.1f}%"}
            events.append(event)
            ts += dur + 1

    # --- perf counters → counter tracks ---
    if perf_summary:
        tid_perf = 2
        events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": tid_perf,
            "args": {"name": "Hardware Counters (perf)"},
        })

        counter_metrics = [
            ("ipc", "IPC"),
            ("cache_miss_rate", "Cache Miss Rate"),
            ("branch_miss_rate", "Branch Miss Rate"),
            ("cpus_utilized", "CPUs Utilized"),
        ]
        for key, display_name in counter_metrics:
            val = perf_summary.get(key)
            if val is not None:
                events.append({
                    "name": display_name,
                    "ph": "C",
                    "ts": 0,
                    "pid": pid,
                    "tid": tid_perf,
                    "args": {display_name: float(val)},
                })

        # perf hotspots as flame chart
        hotspots = perf_summary.get("hotspots", [])
        if hotspots:
            tid_perf_hot = 3
            events.append({
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid_perf_hot,
                "args": {"name": "CPU Hotspots (perf)"},
            })
            ts = 0
            for h in hotspots:
                func = h.get("function", "?")
                pct = h.get("pct", 0)
                module = h.get("module", "")
                dur = max(1, int(pct * 10))
                event = {
                    "name": func,
                    "cat": "cpu",
                    "ph": "X",
                    "ts": ts,
                    "dur": dur,
                    "pid": pid,
                    "tid": tid_perf_hot,
                    "args": {"pct": f"{pct:.1f}%"},
                }
                if module:
                    event["args"]["module"] = module
                events.append(event)
                ts += dur + 1

    # --- memray allocation hotspots ---
    if memray_summary and memray_summary.get("top_allocators"):
        tid_mem = 4
        events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": tid_mem,
            "args": {"name": "Memory Allocations (memray)"},
        })
        ts = 0
        for alloc in memray_summary["top_allocators"]:
            func = alloc.get("function", "?")
            size_mb = alloc.get("size_mb", 0)
            dur = max(1, int(size_mb * 10))
            events.append({
                "name": func,
                "cat": "memory",
                "ph": "X",
                "ts": ts,
                "dur": dur,
                "pid": pid,
                "tid": tid_mem,
                "args": {
                    "size_mb": f"{size_mb:.2f}",
                    "location": alloc.get("location", ""),
                },
            })
            ts += dur + 1

    if not events:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"traceEvents": events, "displayTimeUnit": "us"}, indent=1),
        encoding="utf-8",
    )
    return output_path
