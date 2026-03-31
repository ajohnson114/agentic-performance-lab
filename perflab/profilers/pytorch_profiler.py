from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from pathlib import Path
import json
import shlex
from perflab.profilers.base import ProfileResult
from perflab.tools.shell import run_cmd

logger = logging.getLogger(__name__)


def _is_gpu_event(cat_lower: str) -> bool:
    """Return True if the event category indicates GPU kernel execution."""
    return (
        "kernel" in cat_lower
        or cat_lower in ("mps", "gpu", "gpu_op")
        or ("metal" in cat_lower and "kernel" in cat_lower)
    )


def _parse_torch_trace(trace_path: Path, top_n: int = 10) -> dict:
    """Extract structured profiling data from a PyTorch Chrome trace JSON.

    Returns a dict with top_ops, top_gpu_kernels, cpu_vs_gpu, memory,
    sync info, and operator shapes.
    """
    result: dict = {}
    if not trace_path.exists():
        return result

    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to parse torch trace %s", trace_path, exc_info=True)
        return result

    # Chrome traces can be a list of events or {"traceEvents": [...]}
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    if not events:
        return result

    # Categorize events
    op_stats: dict[str, dict] = {}       # all ph=X events by name
    gpu_kernel_stats: dict[str, dict] = {}  # cat=kernel events
    cpu_op_stats: dict[str, dict] = {}      # cat=cpu_op/operator events
    raw_cpu_ops: list[dict] = []             # timestamped CPU ops for temporal cross-ref
    sync_count = 0
    total_sync_us = 0.0
    memory_alloc_count = 0
    memory_alloc_time_us = 0.0
    peak_memory_bytes: int | None = None

    # Track top operator args for shape extraction
    op_events_by_name: dict[str, list[dict]] = {}

    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        dur = ev.get("dur", 0)
        cat = ev.get("cat", "")
        args = ev.get("args", {})

        if not name or dur <= 0:
            continue
        # Skip meta events
        if name.startswith("Profiler") or name.startswith("process_"):
            continue

        # All operators aggregate
        if name not in op_stats:
            op_stats[name] = {"total_us": 0.0, "count": 0}
        op_stats[name]["total_us"] += float(dur)
        op_stats[name]["count"] += 1

        # Store events for shape extraction
        if args:
            op_events_by_name.setdefault(name, []).append(ev)

        # GPU kernels (CUDA kernels + MPS/Metal GPU events)
        cat_lower = cat.lower() if cat else ""
        if _is_gpu_event(cat_lower):
            if name not in gpu_kernel_stats:
                gpu_kernel_stats[name] = {"total_us": 0.0, "count": 0}
            gpu_kernel_stats[name]["total_us"] += float(dur)
            gpu_kernel_stats[name]["count"] += 1

        # CPU operators
        if cat_lower in ("cpu_op", "operator", "cpu_op|operator"):
            if name not in cpu_op_stats:
                cpu_op_stats[name] = {"total_us": 0.0, "count": 0}
            cpu_op_stats[name]["total_us"] += float(dur)
            cpu_op_stats[name]["count"] += 1
            # Collect timestamped events for temporal cross-reference with nsys
            ts = ev.get("ts", 0)
            if ts > 0 and dur > 100:  # only ops > 100us to limit volume
                raw_cpu_ops.append({"name": name, "ts": ts, "dur": dur})

        # Sync detection
        if name in ("cudaDeviceSynchronize", "cudaStreamSynchronize"):
            sync_count += 1
            total_sync_us += float(dur)

        # Memory allocation detection
        is_memory = (
            "memory" in cat_lower
            or "cudaMalloc" in name
            or "aten::empty" in name
            or "[memory]" in name
        )
        if is_memory:
            memory_alloc_count += 1
            memory_alloc_time_us += float(dur)

        # Track peak memory from args
        if isinstance(args, dict):
            for key in ("bytes", "Bytes", "Total Allocated"):
                if key in args:
                    try:
                        mem_bytes = int(args[key])
                        if peak_memory_bytes is None or mem_bytes > peak_memory_bytes:
                            peak_memory_bytes = mem_bytes
                    except (ValueError, TypeError):
                        pass

    if not op_stats:
        return result

    total_us = sum(s["total_us"] for s in op_stats.values())
    if total_us <= 0:
        return result

    # Top operators (all events)
    sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]["total_us"], reverse=True)
    top_ops = []
    for name, stats in sorted_ops[:top_n]:
        pct = (stats["total_us"] / total_us * 100.0) if total_us > 0 else 0.0
        entry: dict = {
            "name": name,
            "total_us": round(stats["total_us"], 1),
            "count": stats["count"],
            "pct": round(pct, 1),
        }
        top_ops.append(entry)
    result["top_ops"] = top_ops

    # Extract shapes for top-5 operators
    try:
        for entry in top_ops[:5]:
            op_name = entry["name"]
            ev_list = op_events_by_name.get(op_name, [])
            if ev_list:
                args = ev_list[0].get("args", {})
                for key in ("Input Dims", "input_shapes", "Input type"):
                    if key in args:
                        entry["shapes"] = str(args[key])
                        break
    except (KeyError, TypeError):
        pass

    # Extract per-operator FLOPS from trace args (requires with_flops=True)
    total_flops: float = 0.0
    op_flops: dict[str, float] = {}
    for name, ev_list in op_events_by_name.items():
        for ev in ev_list:
            args = ev.get("args", {})
            if isinstance(args, dict):
                flops_val = args.get("flops") or args.get("FLOPs") or args.get("FLOPS")
                if flops_val is not None:
                    try:
                        f_val = float(flops_val)
                        total_flops += f_val
                        op_flops[name] = op_flops.get(name, 0.0) + f_val
                    except (ValueError, TypeError):
                        pass
    if total_flops > 0:
        result["total_flops"] = total_flops
        result["total_tflops"] = round(total_flops / 1e12, 6)
        # Top ops by FLOPS
        sorted_flops = sorted(op_flops.items(), key=lambda x: x[1], reverse=True)
        result["top_ops_by_flops"] = [
            {"name": n, "flops": f, "pct": round(f / total_flops * 100, 1)}
            for n, f in sorted_flops[:5]
        ]

    # Top GPU kernels
    if gpu_kernel_stats:
        try:
            total_gpu_us = sum(s["total_us"] for s in gpu_kernel_stats.values())
            sorted_gpu = sorted(gpu_kernel_stats.items(), key=lambda x: x[1]["total_us"], reverse=True)
            top_gpu = []
            for name, stats in sorted_gpu[:top_n]:
                pct = (stats["total_us"] / total_gpu_us * 100.0) if total_gpu_us > 0 else 0.0
                top_gpu.append({
                    "name": name,
                    "total_us": round(stats["total_us"], 1),
                    "count": stats["count"],
                    "pct": round(pct, 1),
                })
            result["top_gpu_kernels"] = top_gpu
        except (KeyError, TypeError, ZeroDivisionError):
            logger.warning("Failed to compute top GPU kernels", exc_info=True)

    # CPU vs GPU breakdown
    try:
        total_cpu_op_us = sum(s["total_us"] for s in cpu_op_stats.values())
        total_gpu_kernel_us = sum(s["total_us"] for s in gpu_kernel_stats.values())
        if total_cpu_op_us > 0 or total_gpu_kernel_us > 0:
            ratio = (total_gpu_kernel_us / total_cpu_op_us) if total_cpu_op_us > 0 else float("inf")
            result["cpu_vs_gpu"] = {
                "total_cpu_op_us": round(total_cpu_op_us, 1),
                "total_gpu_kernel_us": round(total_gpu_kernel_us, 1),
                "ratio": round(ratio, 3),
            }
    except (KeyError, TypeError, ZeroDivisionError):
        pass

    # Memory summary
    try:
        if memory_alloc_count > 0:
            mem_info: dict = {
                "total_allocations": memory_alloc_count,
                "total_allocation_time_us": round(memory_alloc_time_us, 1),
            }
            if peak_memory_bytes is not None:
                mem_info["peak_memory_mb"] = round(peak_memory_bytes / (1024 * 1024), 1)
            result["memory"] = mem_info
    except (KeyError, TypeError):
        pass

    # Sync points
    if sync_count > 0:
        result["sync_count"] = sync_count
        result["total_sync_time_us"] = round(total_sync_us, 1)

    # ── Phase extraction from record_function markers ──
    phase_re = re.compile(r"^##\s*(\w+)\s*##$")
    phase_map: dict[str, dict] = {}  # name -> {total_us, count, ts_ranges}

    # Pass 1: find phase marker events
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        m = phase_re.match(ev.get("name", ""))
        if not m:
            continue
        phase_name = m.group(1)
        dur = ev.get("dur", 0)
        ts = ev.get("ts", 0)
        if dur <= 0:
            continue
        if phase_name not in phase_map:
            phase_map[phase_name] = {"total_us": 0.0, "count": 0, "ts_ranges": [], "gpu_us": 0.0}
        phase_map[phase_name]["total_us"] += float(dur)
        phase_map[phase_name]["count"] += 1
        phase_map[phase_name]["ts_ranges"].append((float(ts), float(ts) + float(dur)))

    # Pass 2: attribute GPU events to phases
    if phase_map:
        for ev in events:
            if not isinstance(ev, dict) or ev.get("ph") != "X":
                continue
            ev_cat = (ev.get("cat", "") or "").lower()
            if not _is_gpu_event(ev_cat):
                continue
            ev_ts = float(ev.get("ts", 0))
            ev_dur = float(ev.get("dur", 0))
            if ev_dur <= 0:
                continue
            for pinfo in phase_map.values():
                for start, end in pinfo["ts_ranges"]:
                    if start <= ev_ts < end:
                        pinfo["gpu_us"] += ev_dur
                        break

        total_phase_us = sum(p["total_us"] for p in phase_map.values())
        phases_list = []
        for name, pinfo in phase_map.items():
            pct = (pinfo["total_us"] / total_phase_us * 100.0) if total_phase_us > 0 else 0.0
            cpu_us = pinfo["total_us"] - pinfo["gpu_us"]
            phases_list.append({
                "name": name,
                "total_us": round(pinfo["total_us"], 1),
                "gpu_us": round(pinfo["gpu_us"], 1),
                "cpu_us": round(max(cpu_us, 0.0), 1),
                "count": pinfo["count"],
                "pct": round(pct, 1),
            })
        phases_list.sort(key=lambda p: p["total_us"], reverse=True)
        result["phases"] = phases_list

    # Store timestamped CPU ops for temporal cross-reference with nsys GPU data.
    # Keyed with underscore prefix to indicate internal use (not displayed in prompt).
    if raw_cpu_ops:
        # Keep top 200 by duration to limit summary size
        raw_cpu_ops.sort(key=lambda x: x["dur"], reverse=True)
        result["_raw_cpu_ops"] = raw_cpu_ops[:200]

    return result


@dataclass
class TorchProfiler:
    name: str = "torch_profiler"

    def is_available(self) -> bool:
        # available if torch is importable in the current environment
        try:
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        # Resolve to absolute so the subprocess (which runs in cwd) and
        # this process agree on where the trace file lives.
        artifacts_dir = artifacts_dir.resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        trace_path = artifacts_dir / "torch_trace.json"
        # Convention: benchmark harness reads PERFLAB_TORCH_PROFILE=1 and PERFLAB_TORCH_TRACE_PATH.
        env = {
            "PERFLAB_TORCH_PROFILE": "1",
            "PERFLAB_TORCH_TRACE_PATH": str(trace_path),
            "PERFLAB_TORCH_WITH_FLOPS": "1",
        }
        res = run_cmd(shlex.split(bench_cmd), cwd=cwd, env=env)
        summary: dict = {"returncode": res.returncode, "duration_s": res.duration_s}
        artifacts = {}
        if trace_path.exists():
            artifacts["torch_trace_json"] = str(trace_path)
            # Parse structured data from the Chrome trace
            trace_data = _parse_torch_trace(trace_path)
            summary.update(trace_data)
        return ProfileResult(name=self.name, artifacts=artifacts, summary=summary)
