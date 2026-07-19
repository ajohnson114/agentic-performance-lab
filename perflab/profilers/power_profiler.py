"""Power and energy profiler using RAPL (Linux) and nvidia-smi.

Measures CPU package energy via perf stat RAPL events and GPU power draw
via nvidia-smi polling during benchmark execution.
"""
from __future__ import annotations

import re
import shutil
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from perflab.profilers.base import ProfileResult, run_bench_under
from perflab.tools.shell import run_cmd

# RAPL energy counters read by the perf stat wrapper. Shared by the wrapper
# built in run() and the _rapl_usable() probe so both open the exact same
# event set.
_RAPL_EVENTS = "power/energy-pkg/,power/energy-cores/,power/energy-ram/"

# Recorded in the summary when the RAPL PMU is listed by `perf list` but perf
# can't actually open the events (so CPU energy is missing but GPU power is
# still measured from a real run).
_RAPL_UNAVAILABLE_REASON = (
    "perf cannot open RAPL energy events (check "
    "/proc/sys/kernel/perf_event_paranoid or CAP_PERFMON permissions)"
)


@dataclass
class PowerProfiler:
    name: str = "power"

    def is_available(self) -> bool:
        # Available if either RAPL or nvidia-smi is present
        return _has_rapl() or shutil.which("nvidia-smi") is not None

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        summary: dict = {}
        artifacts: dict[str, str] = {}

        has_rapl = _has_rapl()
        # _has_rapl() only proves `perf list` exposes the PMU; perf may still
        # be unable to open the events (perf_event_paranoid / CAP_PERFMON). If
        # it can't, wrapping the run makes perf exit before exec'ing the
        # benchmark, so nvidia-smi would sample an idle GPU -- run bare instead.
        rapl_wrapped = has_rapl and _rapl_usable()
        has_smi = shutil.which("nvidia-smi") is not None

        # RAPL PMU is listed but perf can't open it: still run the benchmark
        # (bare, below) so GPU polling gets a real window, and record why CPU
        # energy is missing.
        if has_rapl and not rapl_wrapped:
            summary["rapl_unavailable"] = _RAPL_UNAVAILABLE_REASON

        if not rapl_wrapped and not has_smi:
            # Nothing measurable: no usable RAPL counters and no GPU to poll.
            return ProfileResult(name=self.name, artifacts=artifacts, summary=summary)

        # Single benchmark run: RAPL counting (perf stat wrapper, when the
        # events are usable) with nvidia-smi polling alongside. Counting RAPL
        # events doesn't perturb GPU sampling, so CPU and GPU power come from
        # the same execution window -- previously these were two separate bench
        # runs, which doubled the cost and made CPU-vs-GPU comparisons mix
        # measurements from different runs.
        rapl_path = artifacts_dir / "rapl_stat.txt"
        wrapper: list[str] = []
        if rapl_wrapped:
            wrapper = [
                "perf", "stat",
                "-e", _RAPL_EVENTS,
                "-o", str(rapl_path), "--",
            ]

        gpu_power_path = artifacts_dir / "gpu_power_log.txt"
        samples: list[float] = []
        mem_samples: list[dict] = []
        power_unavailable_reason: str | None = None
        stop_event = threading.Event()
        poller: threading.Thread | None = None

        def poll_gpu_power():
            nonlocal power_unavailable_reason
            while not stop_event.is_set():
                try:
                    res = run_cmd(
                        ["nvidia-smi",
                         "--query-gpu=power.draw,memory.used,memory.total",
                         "--format=csv,noheader,nounits"],
                        timeout_s=5,
                    )
                    if res.returncode == 0:
                        for line in res.stdout.strip().splitlines():
                            parts = [p.strip() for p in line.split(",")]
                            # power.draw returns "[N/A]" in containers or MIG mode
                            try:
                                samples.append(float(parts[0]))
                            except (ValueError, IndexError):
                                if parts and power_unavailable_reason is None:
                                    power_unavailable_reason = parts[0]
                            if len(parts) >= 3:
                                try:
                                    mem_samples.append({
                                        "used_mib": float(parts[1]),
                                        "total_mib": float(parts[2]),
                                    })
                                except (ValueError, IndexError):
                                    pass
                except Exception:  # noqa: BLE001 -- best-effort polling loop, a single failed sample must not kill the thread
                    pass
                stop_event.wait(0.5)

        if has_smi:
            poller = threading.Thread(target=poll_gpu_power, daemon=True)
            poller.start()

        run_bench_under(wrapper, bench_cmd, cwd=cwd)

        if poller is not None:
            stop_event.set()
            poller.join(timeout=5)

        if rapl_wrapped and rapl_path.exists():
            rapl_data = _parse_rapl_output(rapl_path)
            if rapl_data:
                summary["rapl"] = rapl_data
                artifacts["rapl_stat"] = str(rapl_path)

        if has_smi:
            if samples:
                gpu_data = _compute_gpu_power_stats(samples)
                summary["gpu_power"] = gpu_data
                gpu_power_path.write_text(
                    "\n".join(f"{s:.1f}" for s in samples),
                    encoding="utf-8",
                )
                artifacts["gpu_power_log"] = str(gpu_power_path)
            elif power_unavailable_reason is not None:
                # nvidia-smi responded but power.draw is unavailable (e.g.
                # "[N/A]" in containers, MIG mode, or insufficient privileges).
                summary["gpu_power_unavailable"] = power_unavailable_reason

            if mem_samples:
                gpu_mem = _compute_gpu_memory_stats(mem_samples)
                summary["gpu_memory"] = gpu_mem

        return ProfileResult(name=self.name, artifacts=artifacts, summary=summary)


@lru_cache(maxsize=1)
def _has_rapl() -> bool:
    """Check if RAPL energy events are available via perf.

    Cached: called from both is_available() and run(), and ``perf list``
    spawns a subprocess.
    """
    if shutil.which("perf") is None:
        return False
    res = run_cmd(["perf", "list"], timeout_s=5)
    return "power/energy-pkg/" in res.stdout


@lru_cache(maxsize=1)
def _rapl_usable() -> bool:
    """Whether perf can actually open the RAPL events, not just list them.

    _has_rapl() checks `perf list` (PMU is exposed); this opens the events for
    a trivial ``-- true`` command. When perf_event_paranoid (or a missing
    CAP_PERFMON) blocks unprivileged access, the counting run exits nonzero
    before running anything, so run() must fall back to a bare benchmark run.
    Cached: probed once per process, and each probe spawns perf.
    """
    res = run_cmd(["perf", "stat", "-e", _RAPL_EVENTS, "--", "true"], timeout_s=10)
    return res.returncode == 0


def _parse_rapl_output(path: Path) -> dict:
    """Parse perf stat RAPL output for energy consumption.

    Example lines:
        12.34 Joules power/energy-pkg/
         8.56 Joules power/energy-cores/
         2.10 Joules power/energy-ram/
    """
    result: dict = {}
    if not path.exists():
        return result

    text = path.read_text(encoding="utf-8", errors="replace")

    for line in text.splitlines():
        line = line.strip()
        # Match: <number> Joules <event-name>
        m = re.match(r"^([\d,\.]+)\s+Joules\s+([\w/\-]+)", line)
        if m:
            try:
                joules = float(m.group(1).replace(",", ""))
                event = m.group(2)
                if "energy-pkg" in event:
                    result["package_joules"] = joules
                elif "energy-cores" in event:
                    result["cores_joules"] = joules
                elif "energy-ram" in event:
                    result["ram_joules"] = joules
            except ValueError:
                continue

    # Parse elapsed time for average power calculation
    elapsed_m = re.search(r"([\d,\.]+)\s+seconds\s+time\s+elapsed", text)
    if elapsed_m:
        try:
            elapsed = float(elapsed_m.group(1).replace(",", ""))
            result["elapsed_seconds"] = elapsed
            if "package_joules" in result and elapsed > 0:
                result["avg_package_watts"] = result["package_joules"] / elapsed
        except ValueError:
            pass

    return result


def _compute_gpu_power_stats(samples: list[float]) -> dict:
    """Compute statistics from GPU power samples (in watts).

    Includes both aggregated statistics and the raw power_samples list
    (as dicts with 'watts' key) for clock throttle detection.
    """
    if not samples:
        return {}

    sorted_s = sorted(samples)
    n = len(sorted_s)
    return {
        "sample_count": n,
        "avg_watts": sum(samples) / n,
        "min_watts": sorted_s[0],
        "max_watts": sorted_s[-1],
        "p50_watts": sorted_s[n // 2],
        "p95_watts": sorted_s[int(0.95 * (n - 1))],
        # Raw samples for clock throttle detection (microarch.py)
        "power_samples": [{"watts": w} for w in samples],
    }


def _compute_gpu_memory_stats(mem_samples: list[dict]) -> dict:
    """Compute statistics from GPU memory samples (in MiB)."""
    if not mem_samples:
        return {}

    used_vals = [s["used_mib"] for s in mem_samples]
    total_mib = mem_samples[0]["total_mib"]
    n = len(used_vals)
    avg_used = sum(used_vals) / n
    max_used = max(used_vals)

    return {
        "sample_count": n,
        "total_mib": total_mib,
        "avg_used_mib": avg_used,
        "max_used_mib": max_used,
        "utilization_pct": max_used / total_mib * 100 if total_mib > 0 else 0.0,
    }
