"""Top-Down Microarchitecture Analysis (TMA).

Classifies CPU cycles into four high-level buckets:
  - Frontend Bound: instruction fetch/decode stalls
  - Backend Bound: execution unit / memory stalls
  - Bad Speculation: mispredicted branches, cancelled work
  - Retiring: useful work done

Uses perf stat with Intel's TopdownL1 metric group, or estimates
from raw counters when metric groups aren't available.
"""
from __future__ import annotations

import logging
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from perflab.tools.shell import run_cmd

logger = logging.getLogger(__name__)


@dataclass
class TMAResult:
    frontend_bound_pct: float
    backend_bound_pct: float
    bad_speculation_pct: float
    retiring_pct: float
    raw_text: str = ""

    @property
    def dominant_bottleneck(self) -> str:
        vals = {
            "frontend_bound": self.frontend_bound_pct,
            "backend_bound": self.backend_bound_pct,
            "bad_speculation": self.bad_speculation_pct,
            "retiring": self.retiring_pct,
        }
        return max(vals, key=lambda k: vals[k])

    def to_dict(self) -> dict:
        return {
            "frontend_bound_pct": round(self.frontend_bound_pct, 1),
            "backend_bound_pct": round(self.backend_bound_pct, 1),
            "bad_speculation_pct": round(self.bad_speculation_pct, 1),
            "retiring_pct": round(self.retiring_pct, 1),
            "dominant_bottleneck": self.dominant_bottleneck,
        }


def is_tma_available() -> bool:
    """Check if TMA metrics are available on this system."""
    if shutil.which("perf") is None:
        return False
    # Check for TopdownL1 metric group
    res = run_cmd(["perf", "list", "metric"], timeout_s=5)
    return "TopdownL1" in res.stdout or "topdown" in res.stdout.lower()


def collect_tma(bench_cmd: str, cwd: Path, output_path: Path) -> TMAResult | None:
    """Run perf stat with TMA metrics and parse results.

    Tries TopdownL1 metric group first, falls back to raw counter estimation.
    """
    cmd_parts = shlex.split(bench_cmd)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try metric group first
    stat_cmd = [
        "perf", "stat", "-M", "TopdownL1",
        "-o", str(output_path), "--",
    ] + cmd_parts
    run_cmd(stat_cmd, cwd=cwd)

    if output_path.exists():
        text = output_path.read_text(encoding="utf-8", errors="replace")
        result = _parse_tma_output(text)
        if result is not None:
            return result

    # Fallback: use topdown-* events (perf >= 5.10)
    fallback_events = (
        "topdown-fetch-bubbles,topdown-recovery-bubbles,"
        "topdown-slots-issued,topdown-slots-retired,topdown-total-slots"
    )
    fallback_path = output_path.with_suffix(".fallback.txt")
    fallback_cmd = [
        "perf", "stat", "-e", fallback_events,
        "-o", str(fallback_path), "--",
    ] + cmd_parts
    run_cmd(fallback_cmd, cwd=cwd)

    if fallback_path.exists():
        text = fallback_path.read_text(encoding="utf-8", errors="replace")
        result = _parse_tma_fallback(text)
        if result is not None:
            return result

    return None


def _parse_tma_output(text: str) -> TMAResult | None:
    """Parse perf stat -M TopdownL1 output.

    Expected format (varies by perf version):
        23.4%  frontend bound
        45.2%  backend bound
         8.1%  bad speculation
        23.3%  retiring
    Or:
        retiring  23.3%
        ...
    """
    # Match per-line to avoid cross-line false matches.
    # Forward: "23.4%  frontend bound"
    # Reversed: "frontend bound:  23.4%"
    line_patterns = [
        ("frontend", re.compile(r"([\d.]+)\s*%?\s*frontend[\s_-]*bound", re.IGNORECASE)),
        ("frontend", re.compile(r"frontend[\s_-]*bound\s*[:\s]+\s*([\d.]+)\s*%?", re.IGNORECASE)),
        ("backend", re.compile(r"([\d.]+)\s*%?\s*backend[\s_-]*bound", re.IGNORECASE)),
        ("backend", re.compile(r"backend[\s_-]*bound\s*[:\s]+\s*([\d.]+)\s*%?", re.IGNORECASE)),
        ("speculation", re.compile(r"([\d.]+)\s*%?\s*bad[\s_-]*speculation", re.IGNORECASE)),
        ("speculation", re.compile(r"bad[\s_-]*speculation\s*[:\s]+\s*([\d.]+)\s*%?", re.IGNORECASE)),
        ("retiring", re.compile(r"([\d.]+)\s*%?\s*retiring", re.IGNORECASE)),
        ("retiring", re.compile(r"retiring\s*[:\s]+\s*([\d.]+)\s*%?", re.IGNORECASE)),
    ]

    values: dict[str, float] = {}
    for line in text.splitlines():
        for key, pattern in line_patterns:
            if key in values:
                continue
            m = pattern.search(line)
            if m:
                values[key] = float(m.group(1))

    if len(values) < 4:
        return None

    return TMAResult(
        frontend_bound_pct=values["frontend"],
        backend_bound_pct=values["backend"],
        bad_speculation_pct=values["speculation"],
        retiring_pct=values["retiring"],
        raw_text=text,
    )


def _parse_tma_fallback(text: str) -> TMAResult | None:
    """Estimate TMA from raw topdown-* counters.

    Formula (simplified Intel Top-Down):
      Frontend Bound = fetch_bubbles / total_slots
      Bad Speculation = (issued - retired + recovery_bubbles) / total_slots
      Retiring = retired / total_slots
      Backend Bound = 1 - Frontend - BadSpec - Retiring
    """
    counters: dict[str, int] = {}
    counter_names = [
        "topdown-fetch-bubbles",
        "topdown-recovery-bubbles",
        "topdown-slots-issued",
        "topdown-slots-retired",
        "topdown-total-slots",
    ]

    for line in text.splitlines():
        for name in counter_names:
            if name in line:
                m = re.match(r"^\s*([\d,]+)\s+", line.strip())
                if m:
                    counters[name] = int(m.group(1).replace(",", ""))

    total = counters.get("topdown-total-slots", 0)
    if total == 0:
        return None

    fetch_bubbles = counters.get("topdown-fetch-bubbles", 0)
    recovery = counters.get("topdown-recovery-bubbles", 0)
    issued = counters.get("topdown-slots-issued", 0)
    retired = counters.get("topdown-slots-retired", 0)

    frontend = fetch_bubbles / total * 100
    bad_spec = (issued - retired + recovery) / total * 100
    retiring = retired / total * 100
    backend = max(0, 100 - frontend - bad_spec - retiring)

    return TMAResult(
        frontend_bound_pct=frontend,
        backend_bound_pct=backend,
        bad_speculation_pct=bad_spec,
        retiring_pct=retiring,
        raw_text=text,
    )


@dataclass
class TMALevel2Result:
    """Level 2+ TMA breakdown (from toplev or AMD equivalent)."""
    # Level 2 — Backend Bound split
    memory_bound_pct: float | None = None
    core_bound_pct: float | None = None
    # Level 2 — Frontend Bound split
    fetch_latency_pct: float | None = None
    fetch_bandwidth_pct: float | None = None
    # Level 3 — Memory Bound split
    l1_bound_pct: float | None = None
    l2_bound_pct: float | None = None
    l3_bound_pct: float | None = None
    dram_bound_pct: float | None = None
    store_bound_pct: float | None = None
    # Level 3 — Core Bound split
    divider_pct: float | None = None
    port_utilization_pct: float | None = None
    # Metadata
    source: str = ""  # "toplev", "amd-perf", "perf-topdown"
    raw_text: str = ""

    def to_dict(self) -> dict:
        d: dict = {}
        for key in (
            "memory_bound_pct", "core_bound_pct",
            "fetch_latency_pct", "fetch_bandwidth_pct",
            "l1_bound_pct", "l2_bound_pct", "l3_bound_pct",
            "dram_bound_pct", "store_bound_pct",
            "divider_pct", "port_utilization_pct",
        ):
            val = getattr(self, key)
            if val is not None:
                d[key] = round(val, 1)
        if self.source:
            d["source"] = self.source
        # Identify dominant memory bottleneck level
        mem_levels = {
            "L1": self.l1_bound_pct, "L2": self.l2_bound_pct,
            "L3": self.l3_bound_pct, "DRAM": self.dram_bound_pct,
            "Store": self.store_bound_pct,
        }
        valid_levels = {k: v for k, v in mem_levels.items() if v is not None and v > 0}
        if valid_levels:
            d["dominant_memory_level"] = max(valid_levels, key=valid_levels.get)  # type: ignore[arg-type]
        return d


def _detect_cpu_vendor() -> str:
    """Detect CPU vendor: 'intel', 'amd', or 'unknown'."""
    import platform
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "vendor_id" in line:
                        lower = line.lower()
                        if "genuineintel" in lower:
                            return "intel"
                        if "authenticamd" in lower:
                            return "amd"
                        break
        elif platform.system() == "Darwin":
            import subprocess
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.vendor"],
                capture_output=True, text=True, timeout=5,
            )
            if "intel" in res.stdout.lower():
                return "intel"
    except (OSError, ValueError):
        pass
    return "unknown"


def collect_tma_level2(
    bench_cmd: str, cwd: Path, output_path: Path,
) -> TMALevel2Result | None:
    """Collect Level 2/3 TMA metrics using toplev (Intel) or perf events (AMD).

    Requires:
    - Intel: toplev from pmu-tools (pip install pmu-tools or git clone)
    - AMD: perf with Zen 3+ topdown event support
    """
    vendor = _detect_cpu_vendor()

    if vendor == "intel" and shutil.which("toplev"):
        return _collect_toplev(bench_cmd, cwd, output_path)

    if vendor == "intel" and shutil.which("toplev.py"):
        return _collect_toplev(bench_cmd, cwd, output_path, binary="toplev.py")

    # AMD fallback: use perf stat with Zen topdown-like events
    if vendor == "amd":
        return _collect_amd_tma(bench_cmd, cwd, output_path)

    return None


def _collect_toplev(
    bench_cmd: str, cwd: Path, output_path: Path, binary: str = "toplev",
) -> TMALevel2Result | None:
    """Run toplev --level 3 for Intel Level 2/3 TMA."""
    cmd_parts = shlex.split(bench_cmd)
    out_file = output_path.with_suffix(".toplev.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    toplev_cmd = [
        binary, "--level", "3", "--single-thread",
        "--csv", str(out_file), "--no-desc",
        "--", *cmd_parts,
    ]
    run_cmd(toplev_cmd, cwd=cwd, timeout_s=120)

    if not out_file.exists():
        return None

    text = out_file.read_text(encoding="utf-8", errors="replace")
    return _parse_toplev_output(text)


def _parse_toplev_output(text: str) -> TMALevel2Result | None:
    """Parse toplev CSV output for Level 2/3 metrics.

    toplev CSV format:
        Area,Value,Unit,Description
        Frontend_Bound,15.2,%,...
        Frontend_Bound.Fetch_Latency,10.1,%,...
        Backend_Bound,52.3,%,...
        Backend_Bound.Memory_Bound,38.5,%,...
        Backend_Bound.Memory_Bound.L1_Bound,5.2,%,...
    """
    if not text.strip():
        return None

    metrics: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        area = parts[0].strip()
        try:
            value = float(parts[1].strip().rstrip("%"))
        except (ValueError, IndexError):
            continue
        metrics[area.lower()] = value

    # Also handle non-CSV toplev output (space-separated)
    if not metrics:
        for line in text.splitlines():
            m = re.match(r"^\s*([\w.]+)\s+([\d.]+)\s*%", line)
            if m:
                metrics[m.group(1).lower()] = float(m.group(2))

    if not metrics:
        return None

    def _get(key: str) -> float | None:
        # Try multiple naming conventions
        for variant in [key, key.replace(".", "_"), key.replace("_", ".")]:
            if variant in metrics:
                return metrics[variant]
        return None

    result = TMALevel2Result(
        memory_bound_pct=_get("backend_bound.memory_bound"),
        core_bound_pct=_get("backend_bound.core_bound"),
        fetch_latency_pct=_get("frontend_bound.fetch_latency"),
        fetch_bandwidth_pct=_get("frontend_bound.fetch_bandwidth"),
        l1_bound_pct=_get("backend_bound.memory_bound.l1_bound"),
        l2_bound_pct=_get("backend_bound.memory_bound.l2_bound"),
        l3_bound_pct=_get("backend_bound.memory_bound.l3_bound"),
        dram_bound_pct=_get("backend_bound.memory_bound.dram_bound"),
        store_bound_pct=_get("backend_bound.memory_bound.store_bound"),
        divider_pct=_get("backend_bound.core_bound.divider"),
        port_utilization_pct=_get("backend_bound.core_bound.ports_utilization"),
        source="toplev",
        raw_text=text,
    )

    # Check if we got any useful data
    if result.memory_bound_pct is None and result.core_bound_pct is None:
        return None

    return result


def _collect_amd_tma(
    bench_cmd: str, cwd: Path, output_path: Path,
) -> TMALevel2Result | None:
    """AMD TMA approximation using Zen perf events.

    AMD Zen 3+ supports limited topdown-like classification via:
    - ex_ret_brn_misp (branch mispredictions → Bad Speculation)
    - dc_access / l2_cache_req_stat (cache hierarchy → Memory Bound level)
    """
    cmd_parts = shlex.split(bench_cmd)
    out_file = output_path.with_suffix(".amd_tma.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Zen 3/4/5 cache hierarchy events
    events = (
        "cycles,instructions,"
        "L1-dcache-load-misses,L1-dcache-loads,"
        "l2_cache_req_stat.ls_rd_blk_c,"  # L2 misses to L3
        "LLC-load-misses,LLC-loads"  # L3 misses to DRAM
    )
    stat_cmd = [
        "perf", "stat", "-e", events,
        "-o", str(out_file), "--",
    ] + cmd_parts
    run_cmd(stat_cmd, cwd=cwd)

    if not out_file.exists():
        return None

    text = out_file.read_text(encoding="utf-8", errors="replace")
    return _parse_amd_tma(text)


def _parse_amd_tma(text: str) -> TMALevel2Result | None:
    """Parse AMD perf stat output to estimate memory hierarchy bottleneck."""
    counters: dict[str, int] = {}
    patterns = {
        "l1_misses": re.compile(r"([\d,]+)\s+L1-dcache-load-misses"),
        "l1_loads": re.compile(r"([\d,]+)\s+L1-dcache-loads"),
        "llc_misses": re.compile(r"([\d,]+)\s+LLC-load-misses"),
        "llc_loads": re.compile(r"([\d,]+)\s+LLC-loads"),
    }

    for line in text.splitlines():
        for name, pat in patterns.items():
            m = pat.search(line)
            if m:
                counters[name] = int(m.group(1).replace(",", ""))

    l1_loads = counters.get("l1_loads", 0)
    l1_misses = counters.get("l1_misses", 0)
    llc_loads = counters.get("llc_loads", 0)
    llc_misses = counters.get("llc_misses", 0)

    if l1_loads == 0:
        return None

    # Estimate hierarchy: what fraction of loads hit at each level
    l1_miss_rate = l1_misses / l1_loads if l1_loads > 0 else 0
    l3_miss_rate = llc_misses / llc_loads if llc_loads > 0 else 0

    # Rough heuristic: distribute "memory bound" across levels
    # High L1 miss rate → L1 bound; high L3 miss rate → DRAM bound
    l1_bound = min(100.0, l1_miss_rate * 100) if l1_miss_rate > 0.05 else 0
    dram_bound = min(100.0, l3_miss_rate * 100) if l3_miss_rate > 0.05 else 0
    l2_l3_bound = max(0, l1_bound - dram_bound) if l1_bound > dram_bound else 0

    if l1_bound == 0 and dram_bound == 0:
        return None

    return TMALevel2Result(
        memory_bound_pct=l1_bound + dram_bound,
        l1_bound_pct=l1_bound if l1_bound > 5 else None,
        dram_bound_pct=dram_bound if dram_bound > 5 else None,
        l2_bound_pct=l2_l3_bound if l2_l3_bound > 5 else None,
        source="amd-perf",
        raw_text=text,
    )


def format_tma_summary(tma: TMAResult, level2: TMALevel2Result | None = None) -> str:
    """Format TMA result as human-readable text for LLM prompt."""
    dominant = tma.dominant_bottleneck.replace("_", " ").title()
    lines = [
        "Top-Down Microarchitecture Analysis:",
        f"  Frontend Bound: {tma.frontend_bound_pct:.1f}%",
        f"  Backend Bound:  {tma.backend_bound_pct:.1f}%",
        f"  Bad Speculation: {tma.bad_speculation_pct:.1f}%",
        f"  Retiring:       {tma.retiring_pct:.1f}%",
        f"  Dominant bottleneck: {dominant}",
    ]

    if level2 is not None:
        lines.append("")
        if level2.memory_bound_pct is not None or level2.core_bound_pct is not None:
            lines.append("  Level 2 breakdown:")
            if level2.memory_bound_pct is not None:
                lines.append(f"    Memory Bound: {level2.memory_bound_pct:.1f}%")
            if level2.core_bound_pct is not None:
                lines.append(f"    Core Bound:   {level2.core_bound_pct:.1f}%")
            if level2.fetch_latency_pct is not None:
                lines.append(f"    Fetch Latency: {level2.fetch_latency_pct:.1f}%")
            if level2.fetch_bandwidth_pct is not None:
                lines.append(f"    Fetch Bandwidth: {level2.fetch_bandwidth_pct:.1f}%")

        mem_levels = [
            ("L1", level2.l1_bound_pct), ("L2", level2.l2_bound_pct),
            ("L3", level2.l3_bound_pct), ("DRAM", level2.dram_bound_pct),
            ("Store", level2.store_bound_pct),
        ]
        active_levels = [(n, v) for n, v in mem_levels if v is not None]
        if active_levels:
            lines.append("  Level 3 memory hierarchy:")
            for name, val in active_levels:
                lines.append(f"    {name} Bound: {val:.1f}%")

    return "\n".join(lines)
