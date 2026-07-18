from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from perflab.profilers.base import ProfileResult, run_bench_with_sudo_fallback

logger = logging.getLogger(__name__)


def _parse_speedscope_json(json_path: Path) -> list[dict]:
    """Extract timestamped samples from py-spy speedscope JSON output.

    Speedscope format has shared.frames[] and profiles[].samples[]/weights[].
    Each sample is a list of frame indices (call stack), and weights are durations.
    We extract the leaf (most specific) frame with its timestamp for temporal
    cross-referencing with GPU profiler data.
    """
    if not json_path.exists():
        return []

    try:
        import json as _json
        data = _json.loads(json_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        logger.warning("Failed to parse speedscope JSON %s", json_path, exc_info=True)
        return []

    frames = data.get("shared", {}).get("frames", [])
    if not frames:
        return []

    timed_samples: list[dict] = []

    for profile in data.get("profiles", []):
        profile_type = profile.get("type", "")
        samples = profile.get("samples", [])
        weights = profile.get("weights", [])
        start_value = profile.get("startValue", 0)

        if profile_type == "evented":
            # Evented profile: events[] with type/at/frame
            events = profile.get("events", [])
            # Track open frames via a stack
            open_frames: dict[int, int] = {}  # frame_idx -> open_at
            for ev in events:
                ev_type = ev.get("type", "")
                at = ev.get("at", 0)
                frame_idx = ev.get("frame", 0)
                if ev_type == "O":  # open
                    open_frames[frame_idx] = at
                elif ev_type == "C":  # close
                    open_at = open_frames.pop(frame_idx, None)
                    if open_at is not None and frame_idx < len(frames):
                        frame = frames[frame_idx]
                        func = frame.get("name", "")
                        if func:
                            timed_samples.append({
                                "function": func,
                                "file": frame.get("file", ""),
                                "ts_ns": int(open_at * 1_000_000),  # ms → ns
                                "dur_ns": int((at - open_at) * 1_000_000),
                            })
        elif samples:
            # Sampled profile: samples[] are frame index lists, weights[] are durations
            cumulative_ns = int(start_value)
            for i, sample in enumerate(samples):
                weight = weights[i] if i < len(weights) else 0
                weight_ns = int(weight)
                # Leaf frame is the last in the sample stack
                if sample:
                    leaf_idx = sample[-1]
                    if leaf_idx < len(frames):
                        frame = frames[leaf_idx]
                        func = frame.get("name", "")
                        if func:
                            timed_samples.append({
                                "function": func,
                                "file": frame.get("file", ""),
                                "ts_ns": cumulative_ns,
                                "dur_ns": weight_ns if weight_ns > 0 else 10_000_000,
                            })
                cumulative_ns += weight_ns

    return timed_samples


def _extract_hotspots_from_speedscope(
    timed_samples: list[dict], top_n: int = 10,
) -> list[dict]:
    """Derive hotspot list from speedscope timed samples.

    Aggregates sample durations by leaf function to produce the same hotspot
    format as the legacy SVG parser (function, location, pct).
    """
    if not timed_samples:
        return []

    func_info: dict[str, dict] = {}  # function -> {dur_ns, file}
    total_dur = 0
    for s in timed_samples:
        func = s.get("function", "")
        dur = s.get("dur_ns", 0)
        if not func or dur <= 0:
            continue
        total_dur += dur
        if func not in func_info:
            func_info[func] = {"dur_ns": 0, "file": s.get("file", "")}
        func_info[func]["dur_ns"] += dur

    if not func_info or total_dur <= 0:
        return []

    sorted_funcs = sorted(func_info.items(), key=lambda x: x[1]["dur_ns"], reverse=True)
    hotspots = []
    for func, info in sorted_funcs[:top_n]:
        pct = info["dur_ns"] / total_dur * 100.0
        hotspots.append({
            "function": func,
            "location": info["file"],
            "pct": round(pct, 1),
        })
    return hotspots


@dataclass
class PySpyProfiler:
    name: str = "pyspy"

    def is_available(self) -> bool:
        return shutil.which("py-spy") is not None

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        artifacts_dir = artifacts_dir.resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        out_speedscope = artifacts_dir / "pyspy_speedscope.json"

        # Speedscope JSON gives structured data + timestamps for temporal
        # GPU cross-referencing in a single run (no second benchmark execution).
        # Each attempt escalates to sudo when the non-sudo run failed and
        # produced no artifact (see run_bench_with_sudo_fallback).
        res, _ = run_bench_with_sudo_fallback(
            ["py-spy", "record", "--native", "--format", "speedscope",
             "-o", str(out_speedscope), "--"],
            bench_cmd, cwd, expect_artifact=out_speedscope,
        )
        native_mode = True

        # Fall back to without --native (some platforms don't support it)
        if res.returncode != 0 and not out_speedscope.exists():
            res, _ = run_bench_with_sudo_fallback(
                ["py-spy", "record", "--format", "speedscope",
                 "-o", str(out_speedscope), "--"],
                bench_cmd, cwd, expect_artifact=out_speedscope,
            )
            native_mode = False

        summary: dict = {
            "returncode": res.returncode,
            "duration_s": res.duration_s,
            "native_mode": native_mode,
        }

        if out_speedscope.exists():
            timed_samples = _parse_speedscope_json(out_speedscope)
            if timed_samples:
                summary["timed_samples"] = timed_samples
                hotspots = _extract_hotspots_from_speedscope(timed_samples)
                if hotspots:
                    summary["hotspots"] = hotspots
                summary["total_samples"] = len(timed_samples)

        artifacts: dict[str, str] = {}
        if out_speedscope.exists():
            artifacts["pyspy_speedscope"] = str(out_speedscope)

        return ProfileResult(
            name=self.name,
            artifacts=artifacts,
            summary=summary,
        )
