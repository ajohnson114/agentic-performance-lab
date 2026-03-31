"""eBPF-based syscall and I/O tracer using bpftrace (Linux only).

Captures syscall latency distributions and I/O patterns that sampling
profilers miss.  Useful for identifying dataloader bottlenecks, file I/O
stalls, and network latency in training pipelines.

Requires: bpftrace (Linux kernel 4.15+)
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from perflab.profilers.base import ProfileResult


# One-liner bpftrace scripts for common patterns
_SYSCALL_LATENCY_SCRIPT = r"""
tracepoint:raw_syscalls:sys_enter { @start[tid] = nsecs; }
tracepoint:raw_syscalls:sys_exit /@start[tid]/ {
    @syscall_ns[args->id] = hist(nsecs - @start[tid]);
    @total_calls[args->id] = count();
    delete(@start[tid]);
}
interval:s:1 { print(@total_calls); }
END { print(@syscall_ns); print(@total_calls); }
"""

_IO_LATENCY_SCRIPT = r"""
tracepoint:syscalls:sys_enter_read { @read_start[tid] = nsecs; }
tracepoint:syscalls:sys_exit_read /@read_start[tid]/ {
    @read_ns = hist(nsecs - @read_start[tid]);
    @read_bytes = sum(args->ret > 0 ? args->ret : 0);
    @read_count = count();
    delete(@read_start[tid]);
}
tracepoint:syscalls:sys_enter_write { @write_start[tid] = nsecs; }
tracepoint:syscalls:sys_exit_write /@write_start[tid]/ {
    @write_ns = hist(nsecs - @write_start[tid]);
    @write_bytes = sum(args->ret > 0 ? args->ret : 0);
    @write_count = count();
    delete(@write_start[tid]);
}
END { print(@read_ns); print(@write_ns); }
"""


@dataclass
class EbpfProfiler:
    name: str = "ebpf"

    def is_available(self) -> bool:
        # Only available on Linux with bpftrace
        if os.uname().sysname != "Linux":
            return False
        return shutil.which("bpftrace") is not None

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        from perflab.tools.shell import run_cmd

        artifacts_dir = artifacts_dir.resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        io_output_path = artifacts_dir / "ebpf_io_trace.txt"

        script_path = artifacts_dir / "io_trace.bt"
        script_path.write_text(_IO_LATENCY_SCRIPT, encoding="utf-8")

        # Start bpftrace as a background process via Popen, then run the
        # benchmark synchronously, then terminate bpftrace and collect output.
        # We use Popen instead of subprocess.run because we need the two
        # processes to run concurrently — bpftrace must be tracing syscalls
        # while the benchmark executes.  The previous approach used a bash
        # wrapper ("cmd & BPF_PID=$!; bench; kill $BPF_PID") which lost
        # separate exit codes and made error handling fragile.
        bpf_proc = None
        bench_returncode = -1
        try:
            with open(io_output_path, "w") as bpf_out:
                bpf_proc = subprocess.Popen(
                    ["sudo", "-n", "bpftrace", str(script_path)],
                    stdout=bpf_out,
                    stderr=subprocess.STDOUT,
                    cwd=str(cwd),
                )

                # Give bpftrace a moment to attach probes before starting
                # the benchmark, otherwise early syscalls may be missed.
                time.sleep(0.5)

                # Run the benchmark synchronously
                res = run_cmd(shlex.split(bench_cmd), cwd=cwd)
                bench_returncode = res.returncode
        finally:
            if bpf_proc is not None:
                bpf_proc.terminate()
                try:
                    bpf_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    bpf_proc.kill()
                    bpf_proc.wait()

        summary: dict = {"returncode": bench_returncode}

        # Parse bpftrace output
        if io_output_path.exists():
            trace_text = io_output_path.read_text(encoding="utf-8", errors="replace")
            parsed = _parse_bpftrace_output(trace_text)
            summary.update(parsed)

        artifacts: dict[str, str] = {}
        if io_output_path.exists():
            artifacts["ebpf_io_trace_txt"] = str(io_output_path)

        return ProfileResult(
            name=self.name,
            artifacts=artifacts,
            summary=summary,
        )


def _parse_bpftrace_output(text: str) -> dict:
    """Parse bpftrace histogram and counter output."""
    result: dict = {}

    # Parse @read_count and @write_count
    m = re.search(r"@read_count:\s*(\d+)", text)
    if m:
        result["read_syscalls"] = int(m.group(1))
    m = re.search(r"@write_count:\s*(\d+)", text)
    if m:
        result["write_syscalls"] = int(m.group(1))

    # Parse @read_bytes and @write_bytes
    m = re.search(r"@read_bytes:\s*(\d+)", text)
    if m:
        result["read_bytes"] = int(m.group(1))
    m = re.search(r"@write_bytes:\s*(\d+)", text)
    if m:
        result["write_bytes"] = int(m.group(1))

    # Parse histogram buckets for latency distribution
    for prefix, key in [("@read_ns", "read_latency"), ("@write_ns", "write_latency")]:
        hist = _parse_histogram(text, prefix)
        if hist:
            result[key] = hist

    return result


def _parse_histogram(text: str, var_name: str) -> dict | None:
    """Parse a bpftrace hist() output into percentile estimates."""
    # Find the histogram section
    pattern = re.escape(var_name) + r":\s*\n((?:\[.*\n)*)"
    m = re.search(pattern, text)
    if not m:
        return None

    buckets: list[tuple[int, int, int]] = []
    total = 0
    for line in m.group(1).splitlines():
        # Format: [low, high)  count |@@@@|
        bm = re.match(r"\s*\[(\d+)(?:K|M|G)?,\s*(\d+)(?:K|M|G)?\)\s+(\d+)", line)
        if bm:
            low = int(bm.group(1))
            high = int(bm.group(2))
            count = int(bm.group(3))
            buckets.append((low, high, count))
            total += count

    if not buckets or total == 0:
        return None

    # Estimate percentiles
    result = {"total_count": total}
    cumulative = 0
    for low, high, count in buckets:
        cumulative += count
        mid = (low + high) // 2
        pct = cumulative / total
        if pct >= 0.5 and "p50_ns" not in result:
            result["p50_ns"] = mid
        if pct >= 0.9 and "p90_ns" not in result:
            result["p90_ns"] = mid
        if pct >= 0.99 and "p99_ns" not in result:
            result["p99_ns"] = mid

    return result
