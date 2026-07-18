from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from perflab.profilers.base import ProfileResult, run_bench_under
from perflab.tools.shell import run_cmd

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cxxfilt_available() -> bool:
    """Return True if c++filt is on PATH."""
    return shutil.which("c++filt") is not None


@lru_cache(maxsize=512)
def _demangle(name: str) -> str:
    """Demangle a C++ symbol using c++filt if available.

    Returns the original name unchanged when c++filt is not installed
    or the symbol is not a mangled C++ name (``_Z`` prefix).
    Results are cached so repeated calls for the same symbol avoid
    spawning a subprocess.
    """
    if not name.startswith("_Z") or not _cxxfilt_available():
        return name
    try:
        result = subprocess.run(
            ["c++filt", name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        demangled = result.stdout.strip()
        return demangled if demangled else name
    except (OSError, subprocess.TimeoutExpired):
        return name


@dataclass
class LinuxPerfProfiler:
    name: str = "linux_perf"

    def is_available(self) -> bool:
        return shutil.which("perf") is not None

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        stat_path = artifacts_dir / "perf_stat.txt"
        perf_data = artifacts_dir / "perf.data"
        script_path = artifacts_dir / "perf_script.txt"

        # perf stat: high-level counters with specific events
        stat_res = run_bench_under([
            "perf", "stat",
            "-e", "cycles,instructions,cache-references,cache-misses,"
                  "branch-instructions,branch-misses,"
                  "L1-dcache-load-misses,LLC-load-misses,"
                  "task-clock",
            "-o", str(stat_path), "--",
        ], bench_cmd, cwd=cwd)

        # perf record: call graph sampling
        record_res = run_bench_under(
            ["perf", "record", "-g", "-o", str(perf_data), "--"], bench_cmd, cwd=cwd,
        )

        # perf script: export call stacks for analysis
        if perf_data.exists():
            script_cmd = ["perf", "script", "-i", str(perf_data)]
            script_res = run_cmd(script_cmd, cwd=cwd)
            script_path.write_text(script_res.stdout, encoding="utf-8")

        # perf annotate: source-level hotspot mapping
        annotate_path = artifacts_dir / "perf_annotate.txt"
        if perf_data.exists():
            annotate_cmd = ["perf", "annotate", "--stdio", "-i", str(perf_data)]
            annotate_res = run_cmd(annotate_cmd, cwd=cwd)
            if annotate_res.returncode == 0 and annotate_res.stdout.strip():
                annotate_path.write_text(annotate_res.stdout, encoding="utf-8")

        # TMA (Top-Down Microarchitecture Analysis)
        tma_result = None
        tma_level2 = None
        try:
            from perflab.analyzers.tma import collect_tma, collect_tma_level2
            tma_path = artifacts_dir / "tma_stat.txt"
            tma_result = collect_tma(bench_cmd, cwd, tma_path)
            # Level 2/3 TMA (toplev on Intel, perf events on AMD)
            tma_l2_path = artifacts_dir / "tma_level2.txt"
            tma_level2 = collect_tma_level2(bench_cmd, cwd, tma_l2_path)
        except (ImportError, OSError, ValueError):
            logger.warning("TMA collection failed", exc_info=True)

        summary = _parse_perf_stat(stat_path)
        summary["stat_returncode"] = stat_res.returncode
        summary["record_returncode"] = record_res.returncode

        if tma_result is not None:
            summary["tma"] = tma_result.to_dict()
        if tma_level2 is not None:
            summary["tma_level2"] = tma_level2.to_dict()

        # Parse hotspots from perf script output
        if script_path.exists():
            hotspots = _parse_perf_script(script_path)
            if hotspots:
                summary["hotspots"] = hotspots

        # Parse annotated hotspots from perf annotate output
        if annotate_path.exists():
            annotated = _parse_perf_annotate(annotate_path)
            if annotated:
                summary["annotated_hotspots"] = annotated

        artifacts: dict[str, str] = {}
        if stat_path.exists():
            artifacts["perf_stat_txt"] = str(stat_path)
        if perf_data.exists():
            artifacts["perf_data"] = str(perf_data)
        if script_path.exists():
            artifacts["perf_script_txt"] = str(script_path)
        if annotate_path.exists():
            artifacts["perf_annotate_txt"] = str(annotate_path)

        return ProfileResult(name=self.name, artifacts=artifacts, summary=summary)


def _parse_perf_stat(path: Path) -> dict:
    """Extract cycles, instructions, IPC, cache misses, branch misses from perf stat output."""
    result: dict = {}
    if not path.exists():
        return result

    text = path.read_text(encoding="utf-8", errors="replace")

    # Pattern: task-clock line with CPUs utilized
    # e.g. "4,002.12 msec task-clock  #  3.998 CPUs utilized"
    for line in text.splitlines():
        tc_m = re.match(
            r"^\s*([\d,\.]+)\s+msec\s+task-clock\s+#\s+([\d,\.]+)\s+CPUs\s+utilized",
            line,
        )
        if tc_m:
            try:
                result["task_clock_ms"] = float(tc_m.group(1).replace(",", ""))
            except ValueError:
                pass
            try:
                result["cpus_utilized"] = float(tc_m.group(2).replace(",", ""))
            except ValueError:
                pass
            break

    # Pattern: <count> <event-name>  e.g. "1,234,567 cycles"
    for line in text.splitlines():
        line = line.strip()
        # Match lines like "1,234,567      cycles" or "1234567 instructions"
        m = re.match(r"^([\d,\.]+)\s+([\w:-]+)", line)
        if not m:
            continue
        count_str = m.group(1).replace(",", "")
        event = m.group(2)

        count: int | float
        try:
            count = int(count_str)
        except ValueError:
            try:
                count = float(count_str)
            except ValueError:
                continue

        if event == "cycles":
            result["cycles"] = count
        elif event == "instructions":
            result["instructions"] = count
        elif "cache-misses" == event:
            result["cache_misses"] = count
        elif "cache-references" == event:
            result["cache_references"] = count
        elif "branch-misses" == event:
            result["branch_misses"] = count
        elif "branch-instructions" == event:
            result["branch_instructions"] = count
        elif "L1-dcache-load-misses" == event:
            result["l1_dcache_misses"] = count
        elif "LLC-load-misses" == event:
            result["llc_misses"] = count

    # Compute IPC if we have both
    if "cycles" in result and "instructions" in result and result["cycles"] > 0:
        result["ipc"] = result["instructions"] / result["cycles"]

    # Compute cache miss rate
    if "cache_misses" in result and "cache_references" in result and result["cache_references"] > 0:
        result["cache_miss_rate"] = result["cache_misses"] / result["cache_references"]

    # Compute branch miss rate
    if "branch_misses" in result and "branch_instructions" in result and result["branch_instructions"] > 0:
        result["branch_miss_rate"] = result["branch_misses"] / result["branch_instructions"]

    return result


def _parse_perf_script(script_path: Path, top_n: int = 10) -> list[dict]:
    """Extract top-N CPU hotspot functions from perf script output.

    perf script format is blocks like:

        python 12345 1234.567: cycles:
            ffffaaaa func_a+0x10 (/path/to/lib.so)
            ffffbbbb func_b+0x20 (/path/to/lib.so)

    The first indented frame in each block is the "self" sample — the function
    that was actually executing when the sample was taken.
    """
    if not script_path.exists():
        return []

    text = script_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []

    # Parse: each sample block starts with a non-whitespace header line,
    # followed by indented stack frames.  The first frame is "self".
    frame_re = re.compile(
        r"^\s+[0-9a-fA-F]+\s+"        # hex address
        r"(.+?)"                        # function+offset
        r"\s+\((.+?)\)\s*$"            # (module)
    )

    func_counts: dict[str, dict] = {}  # func_name -> {"module": str, "count": int}
    total_samples = 0
    expect_self = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            expect_self = False
            continue

        # Non-indented line = header of a new sample block
        if not line[0].isspace():
            expect_self = True
            continue

        # First indented line after header = self frame
        if expect_self:
            expect_self = False
            total_samples += 1
            m = frame_re.match(line)
            if m:
                raw_func = m.group(1).strip()
                module = m.group(2).strip()
                # Strip the +0xNN offset
                func_name = re.sub(r"\+0x[0-9a-fA-F]+$", "", raw_func)
                key = func_name
                if key not in func_counts:
                    func_counts[key] = {"module": module, "count": 0}
                func_counts[key]["count"] += 1

    if not func_counts or total_samples == 0:
        return []

    sorted_funcs = sorted(func_counts.items(), key=lambda x: x[1]["count"], reverse=True)
    hotspots = []
    for func_name, info in sorted_funcs[:top_n]:
        pct = info["count"] / total_samples * 100.0
        hotspots.append({
            "function": func_name,
            "module": info["module"],
            "count": info["count"],
            "pct": round(pct, 1),
        })

    return hotspots


def _parse_perf_annotate(annotate_path: Path, min_pct: float = 1.0) -> list[dict]:
    """Parse perf annotate --stdio output to extract source-line hotspots.

    perf annotate --stdio output format:
        Percent |  Source code & Disassembly of matmul_bin
                 :  static void matmul(...) {
          45.20  :  14:     sum += A[i*K+k] * B[k*N+j];
          12.10  :  15:     // next line...

    Returns a list of {function, hot_lines: [{file, line, pct}]} dicts.
    """
    if not annotate_path.exists():
        return []

    text = annotate_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []

    # Function header: "Percent |  Source code & Disassembly of <name>"
    func_header_re = re.compile(
        r"Source code & Disassembly of\s+(\S+)"
    )
    # Source line with percentage: "  45.20  :  14:     code..."
    # Also matches: "  45.20  :  filename:14:     code..."
    source_line_re = re.compile(
        r"^\s*([\d.]+)\s+:\s+(?:([^:]+):)?(\d+):\s*(.*)$"
    )

    results: list[dict] = []
    current_func: str | None = None
    hot_lines: list[dict] = []

    for line in text.splitlines():
        # Check for function header
        m = func_header_re.search(line)
        if m:
            # Save previous function
            if current_func and hot_lines:
                results.append({"function": current_func, "hot_lines": hot_lines})
            current_func = m.group(1)
            hot_lines = []
            continue

        # Check for annotated source line
        m = source_line_re.match(line)
        if m and current_func:
            try:
                pct = float(m.group(1))
            except ValueError:
                continue
            if pct >= min_pct:
                file_name = m.group(2) or current_func
                line_no = int(m.group(3))
                hot_lines.append({
                    "file": file_name,
                    "line": line_no,
                    "pct": round(pct, 1),
                })

    # Save last function
    if current_func and hot_lines:
        results.append({"function": current_func, "hot_lines": hot_lines})

    return results


def extract_hot_assembly(
    annotate_path: Path,
    *,
    max_functions: int = 3,
    context_lines: int = 8,
    min_pct: float = 5.0,
) -> list[dict]:
    """Extract small assembly snippets around the hottest instructions.

    Parses ``perf annotate --stdio`` output and returns the hottest
    disassembly windows so the LLM can see whether SIMD, branch, or
    memory-access patterns are present.

    Returns a list of dicts:
        {function, hot_pct, snippet}
    where *snippet* is a short block of annotated assembly lines centred
    on the hottest instruction in each function.
    """
    if not annotate_path.exists():
        return []

    text = annotate_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []

    func_header_re = re.compile(r"Source code & Disassembly of\s+(\S+)")
    # Matches annotated lines (assembly or source) with a percentage
    pct_line_re = re.compile(r"^\s*([\d.]+)\s+:")

    # Split into per-function blocks
    blocks: list[tuple[str, list[str]]] = []
    current_func: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        m = func_header_re.search(line)
        if m:
            if current_func is not None:
                blocks.append((current_func, current_lines))
            current_func = m.group(1)
            current_lines = []
            continue
        if current_func is not None:
            current_lines.append(line)

    if current_func is not None:
        blocks.append((current_func, current_lines))

    # For each function, find the hottest line and extract a window
    results: list[dict] = []
    for func_name, lines in blocks:
        # Find the line with the highest percentage
        best_pct = 0.0
        best_idx = -1
        for i, line in enumerate(lines):
            m = pct_line_re.match(line)
            if m:
                try:
                    pct = float(m.group(1))
                except ValueError:
                    continue
                if pct > best_pct:
                    best_pct = pct
                    best_idx = i

        if best_pct < min_pct or best_idx < 0:
            continue

        # Extract a window around the hottest line
        start = max(0, best_idx - context_lines)
        end = min(len(lines), best_idx + context_lines + 1)
        snippet_lines = lines[start:end]

        # Trim trailing blank lines
        while snippet_lines and not snippet_lines[-1].strip():
            snippet_lines.pop()

        results.append({
            "function": _demangle(func_name),
            "hot_pct": round(best_pct, 1),
            "snippet": "\n".join(snippet_lines),
        })

    # Sort by hottest first, limit count
    results.sort(key=lambda r: r["hot_pct"], reverse=True)
    return results[:max_functions]
