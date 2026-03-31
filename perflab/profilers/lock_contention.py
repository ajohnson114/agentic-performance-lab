"""Lock contention profiler using perf lock and perf c2c.

Detects mutex/spinlock contention and false sharing in multi-threaded code.
Linux only — requires perf with lock/c2c support.
"""
from __future__ import annotations

import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from perflab.profilers.base import ProfileResult
from perflab.tools.shell import run_cmd


@dataclass
class LockContentionProfiler:
    name: str = "lock_contention"

    def is_available(self) -> bool:
        if shutil.which("perf") is None:
            return False
        # Check if perf lock subcommand is available
        res = run_cmd(["perf", "lock", "--help"], timeout=5)
        return res.returncode == 0

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cmd_parts = shlex.split(bench_cmd)
        summary: dict = {}
        artifacts: dict[str, str] = {}

        # --- perf lock record + report ---
        lock_data = artifacts_dir / "perf_lock.data"
        lock_report_path = artifacts_dir / "perf_lock_report.txt"

        record_cmd = ["perf", "lock", "record", "-o", str(lock_data), "--"] + cmd_parts
        run_cmd(record_cmd, cwd=cwd)

        if lock_data.exists():
            report_cmd = ["perf", "lock", "report", "-i", str(lock_data)]
            report_res = run_cmd(report_cmd, cwd=cwd)
            if report_res.returncode == 0 and report_res.stdout.strip():
                lock_report_path.write_text(report_res.stdout, encoding="utf-8")
                artifacts["perf_lock_report"] = str(lock_report_path)
                lock_stats = _parse_perf_lock_report(report_res.stdout)
                summary["lock_stats"] = lock_stats

        # --- perf c2c (false sharing detection) ---
        c2c_data = artifacts_dir / "perf_c2c.data"
        c2c_report_path = artifacts_dir / "perf_c2c_report.txt"

        c2c_cmd = ["perf", "c2c", "record", "-o", str(c2c_data), "--"] + cmd_parts
        c2c_res = run_cmd(c2c_cmd, cwd=cwd)

        if c2c_data.exists() and c2c_res.returncode == 0:
            c2c_report_cmd = ["perf", "c2c", "report", "-i", str(c2c_data), "--stdio"]
            c2c_report_res = run_cmd(c2c_report_cmd, cwd=cwd)
            if c2c_report_res.returncode == 0 and c2c_report_res.stdout.strip():
                c2c_report_path.write_text(c2c_report_res.stdout, encoding="utf-8")
                artifacts["perf_c2c_report"] = str(c2c_report_path)
                c2c_stats = _parse_perf_c2c_report(c2c_report_res.stdout)
                summary["c2c_stats"] = c2c_stats

        return ProfileResult(name=self.name, artifacts=artifacts, summary=summary)


def _parse_perf_lock_report(text: str) -> dict:
    """Parse perf lock report output for contention statistics.

    Example output lines:
        Name   acquired  contended   avg wait (ns)   total wait (ns)   max wait (ns)
        mutex_a      500         42           1234           51828           5000
    """
    result: dict = {"locks": [], "total_contended": 0, "total_wait_ns": 0}

    # Find lines with lock statistics
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Detect table header
        if "acquired" in stripped and "contended" in stripped:
            in_table = True
            continue

        if not in_table:
            continue

        # Skip separator lines
        if stripped.startswith("-") or stripped.startswith("="):
            continue

        parts = stripped.split()
        if len(parts) >= 4:
            try:
                lock_entry = {
                    "name": parts[0],
                    "acquired": int(parts[1]),
                    "contended": int(parts[2]),
                }
                if len(parts) >= 4:
                    lock_entry["avg_wait_ns"] = _parse_number(parts[3])
                if len(parts) >= 5:
                    lock_entry["total_wait_ns"] = _parse_number(parts[4])
                if len(parts) >= 6:
                    lock_entry["max_wait_ns"] = _parse_number(parts[5])

                result["locks"].append(lock_entry)
                result["total_contended"] += lock_entry["contended"]
                result["total_wait_ns"] += lock_entry.get("total_wait_ns", 0)
            except (ValueError, IndexError):
                continue

    return result


def _parse_perf_c2c_report(text: str) -> dict:
    """Parse perf c2c report for false sharing indicators.

    Looks for cache line contention: HITM (Hit In Modified) events
    which indicate true/false sharing between cores.
    """
    result: dict = {
        "total_hitm": 0,
        "total_store": 0,
        "false_sharing_lines": [],
    }

    # Parse summary line: "Total records: 12345"
    total_m = re.search(r"Total records\s*:\s*(\d+)", text)
    if total_m:
        result["total_records"] = int(total_m.group(1))

    # Parse HITM totals
    hitm_m = re.search(r"Total\s+HITM\s*:\s*(\d+)", text, re.IGNORECASE)
    if hitm_m:
        result["total_hitm"] = int(hitm_m.group(1))

    store_m = re.search(r"Total\s+Store\s*:\s*(\d+)", text, re.IGNORECASE)
    if store_m:
        result["total_store"] = int(store_m.group(1))

    # Parse cache line entries with high HITM counts
    # Lines like: "0x7fff... | 42 | 10 | symbol_name"
    cacheline_re = re.compile(
        r"(0x[0-9a-fA-F]+)\s*\|\s*(\d+)\s*\|\s*(\d+)"
    )
    for m in cacheline_re.finditer(text):
        hitm_count = int(m.group(2))
        if hitm_count > 0:
            result["false_sharing_lines"].append({
                "address": m.group(1),
                "hitm": hitm_count,
                "store": int(m.group(3)),
            })

    # Limit to top entries
    result["false_sharing_lines"] = sorted(
        result["false_sharing_lines"],
        key=lambda x: x["hitm"],
        reverse=True,
    )[:10]

    return result


def _parse_number(s: str) -> int:
    """Parse a number that may contain commas."""
    return int(s.replace(",", ""))
