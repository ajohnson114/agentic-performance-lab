"""Memory profiler using memray (Python heap profiling).

Captures peak memory, allocation counts, and top allocators so the LLM
can identify memory-bound bottlenecks that sampling profilers miss.

Requires: pip install memray
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from perflab.profilers.base import ProfileResult


@dataclass
class MemrayProfiler:
    name: str = "memray"

    def is_available(self) -> bool:
        return shutil.which("memray") is not None

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        from perflab.tools.shell import run_cmd

        artifacts_dir = artifacts_dir.resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        bin_path = artifacts_dir / "memray_output.bin"
        stats_path = artifacts_dir / "memray_stats.txt"
        flamegraph_path = artifacts_dir / "memray_flamegraph.html"

        bench_parts = shlex.split(bench_cmd)

        # Run memray to capture allocations
        record_cmd = [
            "memray", "run", "--output", str(bin_path),
            "--force",  # overwrite existing
        ] + bench_parts
        res = run_cmd(record_cmd, cwd=cwd)

        summary: dict = {"returncode": res.returncode}

        if res.returncode != 0 or not bin_path.exists():
            return ProfileResult(
                name=self.name,
                artifacts={},
                summary=summary,
            )

        # Generate text stats
        stats_cmd = ["memray", "stats", str(bin_path)]
        stats_res = run_cmd(stats_cmd, cwd=cwd)
        if stats_res.returncode == 0 and stats_res.stdout.strip():
            stats_path.write_text(stats_res.stdout, encoding="utf-8")
            parsed = _parse_memray_stats(stats_res.stdout)
            summary.update(parsed)

        # Generate flamegraph HTML
        flame_cmd = ["memray", "flamegraph", str(bin_path), "-o", str(flamegraph_path), "--force"]
        run_cmd(flame_cmd, cwd=cwd)

        artifacts: dict[str, str] = {}
        if stats_path.exists():
            artifacts["memray_stats_txt"] = str(stats_path)
        if flamegraph_path.exists():
            artifacts["memray_flamegraph_html"] = str(flamegraph_path)

        # Clean up the potentially large binary
        try:
            bin_path.unlink()
        except OSError:
            pass

        return ProfileResult(
            name=self.name,
            artifacts=artifacts,
            summary=summary,
        )


def _parse_memray_stats(text: str) -> dict:
    """Parse memray stats output to extract key metrics."""
    result: dict = {}

    # Total allocations
    m = re.search(r"Total allocations:\s*(\d[\d,]*)", text)
    if m:
        result["total_allocations"] = int(m.group(1).replace(",", ""))

    # Total memory allocated
    m = re.search(r"Total memory allocated:\s*([\d.]+)\s*(\w+)", text)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        if unit == "GB":
            result["total_allocated_mb"] = val * 1024
        elif unit == "MB":
            result["total_allocated_mb"] = val
        elif unit == "KB":
            result["total_allocated_mb"] = val / 1024
        else:
            result["total_allocated_mb"] = val

    # Peak memory
    m = re.search(r"Peak memory.*?:\s*([\d.]+)\s*(\w+)", text)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        if unit == "GB":
            result["peak_memory_mb"] = val * 1024
        elif unit == "MB":
            result["peak_memory_mb"] = val
        elif unit == "KB":
            result["peak_memory_mb"] = val / 1024
        else:
            result["peak_memory_mb"] = val

    # Top allocators (function-level)
    top_allocators: list[dict] = []
    # Pattern: lines with allocator info, varies by memray version
    # Try to find table rows with function, file:line, size
    alloc_re = re.compile(
        r"^\s*(\d+)\.\s+"           # rank
        r"([\d.]+)\s*(\w+)\s+"      # size + unit
        r"(\d+)\s+"                 # count
        r"(.+?)\s+"                 # function
        r"(\S+:\d+)\s*$",           # location
        re.MULTILINE,
    )
    for m in alloc_re.finditer(text):
        size_val = float(m.group(2))
        size_unit = m.group(3).upper()
        if size_unit == "GB":
            size_mb = size_val * 1024
        elif size_unit == "MB":
            size_mb = size_val
        elif size_unit == "KB":
            size_mb = size_val / 1024
        else:
            size_mb = size_val / (1024 * 1024)
        top_allocators.append({
            "function": m.group(5).strip(),
            "location": m.group(6).strip(),
            "size_mb": round(size_mb, 2),
            "count": int(m.group(4)),
        })

    # Fallback: simpler parsing for allocator lines
    if not top_allocators:
        lines = text.splitlines()
        in_allocators = False
        for line in lines:
            if "top allocat" in line.lower() or "biggest allocat" in line.lower():
                in_allocators = True
                continue
            if in_allocators and line.strip():
                # Try to extract function and size from the line
                parts = line.strip().split()
                if len(parts) >= 3:
                    # Look for a size pattern
                    for i, p in enumerate(parts):
                        size_m = re.match(r"([\d.]+)(GB|MB|KB|B)", p, re.IGNORECASE)
                        if size_m:
                            size_val = float(size_m.group(1))
                            unit = size_m.group(2).upper()
                            if unit == "GB":
                                size_mb = size_val * 1024
                            elif unit == "MB":
                                size_mb = size_val
                            elif unit == "KB":
                                size_mb = size_val / 1024
                            else:
                                size_mb = size_val / (1024 * 1024)
                            func = " ".join(parts[:i]) or parts[-1]
                            top_allocators.append({
                                "function": func,
                                "location": "",
                                "size_mb": round(size_mb, 2),
                                "count": 0,
                            })
                            break
            if in_allocators and not line.strip():
                break

    if top_allocators:
        result["top_allocators"] = top_allocators[:10]

    return result
