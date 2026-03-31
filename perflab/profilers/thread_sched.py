"""Thread scheduling profiler using perf sched.

Captures per-thread scheduling statistics for multi-threaded workloads:
runtime, wait time, migrations, and scheduling latency.
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
class ThreadSchedProfiler:
    name: str = "thread_sched"

    def is_available(self) -> bool:
        if shutil.which("perf") is None:
            return False
        # perf sched requires root or perf_event_paranoid <= 1
        res = run_cmd(["perf", "sched", "--help"], timeout=5)
        return res.returncode == 0

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cmd_parts = shlex.split(bench_cmd)
        summary: dict = {}
        artifacts: dict[str, str] = {}
        perf_data = artifacts_dir / "perf_sched.data"

        # perf sched record
        record_cmd = [
            "perf", "sched", "record", "-o", str(perf_data), "--",
        ] + cmd_parts
        record_res = run_cmd(record_cmd, cwd=cwd, timeout=300)

        if record_res.returncode != 0 or not perf_data.exists():
            summary["error"] = f"perf sched record failed (rc={record_res.returncode})"
            return ProfileResult(name=self.name, artifacts=artifacts, summary=summary)

        # perf sched latency
        latency_path = artifacts_dir / "sched_latency.txt"
        latency_cmd = ["perf", "sched", "latency", "-i", str(perf_data)]
        latency_res = run_cmd(latency_cmd, cwd=cwd, timeout=60)
        if latency_res.returncode == 0 and latency_res.stdout.strip():
            latency_path.write_text(latency_res.stdout, encoding="utf-8")
            artifacts["sched_latency"] = str(latency_path)
            latency_data = _parse_sched_latency(latency_res.stdout)
            if latency_data:
                summary["latency"] = latency_data

        # perf sched timehist --summary
        timehist_path = artifacts_dir / "sched_timehist.txt"
        timehist_cmd = [
            "perf", "sched", "timehist", "--summary", "-i", str(perf_data),
        ]
        timehist_res = run_cmd(timehist_cmd, cwd=cwd, timeout=60)
        if timehist_res.returncode == 0 and timehist_res.stdout.strip():
            timehist_path.write_text(timehist_res.stdout, encoding="utf-8")
            artifacts["sched_timehist"] = str(timehist_path)
            timehist_data = _parse_sched_timehist(timehist_res.stdout)
            if timehist_data:
                summary["timehist"] = timehist_data

        return ProfileResult(name=self.name, artifacts=artifacts, summary=summary)


def _parse_sched_latency(text: str) -> list[dict]:
    """Parse perf sched latency output.

    Example format:
    -----------------------------------------------------------------------------------------------------------------
     Task                  |   Runtime ms  | Switches | Avg delay ms    | Max delay ms    | Max delay start  |
    -----------------------------------------------------------------------------------------------------------------
     matmul_bin:12345      |   1234.567 ms |      42  | avg:    0.012 ms | max:    0.456 ms | max start: ...  |
    """
    results: list[dict] = []

    # Match lines with task stats
    line_re = re.compile(
        r"^\s*(\S+):(\d+)\s*\|\s*([\d.]+)\s*ms\s*\|\s*(\d+)\s*\|"
        r"\s*avg:\s*([\d.]+)\s*ms\s*\|\s*max:\s*([\d.]+)\s*ms"
    )

    for line in text.splitlines():
        m = line_re.match(line)
        if m:
            results.append({
                "task": m.group(1),
                "pid": int(m.group(2)),
                "runtime_ms": float(m.group(3)),
                "switches": int(m.group(4)),
                "avg_delay_ms": float(m.group(5)),
                "max_delay_ms": float(m.group(6)),
            })

    return results


def _parse_sched_timehist(text: str) -> dict:
    """Parse perf sched timehist --summary output.

    Extracts per-CPU summary lines like:
        CPU 0:   1.234 sec  total run time, ...
    """
    result: dict = {"cpus": [], "total_run_ms": 0.0, "total_wait_ms": 0.0}

    # Per-CPU lines
    cpu_re = re.compile(
        r"CPU\s+(\d+).*?([\d.]+)\s*sec\s*total run"
    )
    # Total migrations
    migration_re = re.compile(r"Total\s+number\s+of\s+context\s+switches.*?:\s*(\d+)", re.I)
    migrations_re = re.compile(r"migrations.*?:\s*(\d+)", re.I)

    for line in text.splitlines():
        m = cpu_re.search(line)
        if m:
            cpu_id = int(m.group(1))
            run_sec = float(m.group(2))
            result["cpus"].append({"cpu": cpu_id, "run_sec": run_sec})
            result["total_run_ms"] += run_sec * 1000

        m = migrations_re.search(line)
        if m:
            result["migrations"] = int(m.group(1))

        m = migration_re.search(line)
        if m:
            result["context_switches"] = int(m.group(1))

    return result


def format_sched_summary(summary: dict) -> str:
    """Format thread scheduling summary for the LLM prompt."""
    parts: list[str] = []

    latency = summary.get("latency", [])
    if latency:
        parts.append("Thread scheduling (perf sched latency):")
        for entry in latency[:5]:
            parts.append(
                f"  {entry['task']}:{entry['pid']} — "
                f"runtime={entry['runtime_ms']:.1f}ms, "
                f"switches={entry['switches']}, "
                f"avg_delay={entry['avg_delay_ms']:.3f}ms, "
                f"max_delay={entry['max_delay_ms']:.3f}ms"
            )

    timehist = summary.get("timehist", {})
    if timehist.get("cpus"):
        parts.append(f"  Total run time: {timehist['total_run_ms']:.1f}ms across {len(timehist['cpus'])} CPUs")
        if "migrations" in timehist:
            parts.append(f"  Thread migrations: {timehist['migrations']}")

    return "\n".join(parts)
