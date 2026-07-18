from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from perflab.profilers.base import ProfileResult, run_bench_under
from perflab.tools.shell import run_cmd

logger = logging.getLogger(__name__)


@dataclass
class MetalTraceProfiler:
    name: str = "metal_trace"

    def is_available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        if shutil.which("xctrace") is None:
            return False
        # xctrace exists but may fail if only Command Line Tools are installed
        # (full Xcode with Instruments is required). Smoke-test with a cheap
        # sub-command to verify it actually works.
        try:
            r = subprocess.run(
                ["xctrace", "list", "templates"],
                capture_output=True, timeout=10,
            )
            return r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        trace_path = artifacts_dir / "metal_trace.trace"
        export_xml = artifacts_dir / "metal_trace_export.xml"
        counters_xml = artifacts_dir / "metal_counters_export.xml"

        # Record Metal System Trace
        record_res = run_bench_under([
            "xctrace", "record",
            "--template", "Metal System Trace",
            "--output", str(trace_path),
            "--launch", "--",
        ], bench_cmd, cwd=cwd)

        # Export GPU submission data
        if trace_path.exists():
            export_cmd = [
                "xctrace", "export",
                "--input", str(trace_path),
                "--xpath",
                '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-submission"]',
            ]
            export_res = run_cmd(export_cmd, cwd=cwd)
            if export_res.stdout:
                export_xml.write_text(export_res.stdout, encoding="utf-8")

            # Export GPU performance counters (optional — not all traces have this)
            try:
                counters_cmd = [
                    "xctrace", "export",
                    "--input", str(trace_path),
                    "--xpath",
                    '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-counter"]',
                ]
                counters_res = run_cmd(counters_cmd, cwd=cwd)
                if counters_res.stdout and counters_res.returncode == 0:
                    counters_xml.write_text(counters_res.stdout, encoding="utf-8")
            except (OSError, subprocess.SubprocessError):
                logger.warning("Failed to export Metal GPU counters", exc_info=True)

        summary: dict = {}
        if export_xml.exists():
            summary = _parse_xctrace_export(export_xml)

        # Parse GPU counters if available
        if counters_xml.exists():
            try:
                gpu_counters = _parse_gpu_counters(counters_xml)
                if gpu_counters:
                    summary["gpu_counters"] = gpu_counters
            except (ElementTree.ParseError, OSError):
                logger.warning("Failed to parse Metal GPU counters", exc_info=True)

        summary["record_returncode"] = record_res.returncode
        summary["duration_s"] = record_res.duration_s

        artifacts: dict[str, str] = {}
        if trace_path.exists():
            artifacts["metal_trace_bundle"] = str(trace_path)
        if export_xml.exists():
            artifacts["metal_trace_xml"] = str(export_xml)
        if counters_xml.exists():
            artifacts["metal_counters_xml"] = str(counters_xml)

        return ProfileResult(name=self.name, artifacts=artifacts, summary=summary)


def _parse_xctrace_export(xml_path: Path) -> dict:
    """Extract GPU utilization and per-submission data from xctrace export XML."""
    result: dict = {}
    try:
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()
    except (ElementTree.ParseError, OSError):
        logger.warning("Failed to parse xctrace export XML %s", xml_path, exc_info=True)
        return result

    # Detect column layout from the first row's attributes
    # Each <row> contains child elements whose tag or "name" attribute indicates the column
    submissions: list[dict] = []

    for row in root.iter("row"):
        entry: dict = {}
        for col in row:
            attr_name = (col.get("name", "") or col.tag).lower()
            text = col.text or ""

            # Extract numeric value
            m = re.search(r"([\d.]+)", text)
            val = float(m.group(1)) if m else None

            if "gpu" in attr_name and ("time" in attr_name or "duration" in attr_name):
                if val is not None:
                    entry["gpu_time_ns"] = val
            elif "start" in attr_name and "time" in attr_name:
                if val is not None:
                    entry["start_ns"] = val
            elif "label" in attr_name or ("name" in attr_name and "attr" not in attr_name):
                entry["label"] = text.strip()
            elif "encoder" in attr_name and "type" in attr_name:
                entry["encoder_type"] = text.strip().lower()
            elif "shader" in attr_name and ("time" in attr_name or "duration" in attr_name):
                if val is not None:
                    entry["shader_time_ns"] = val

        if entry.get("gpu_time_ns") is not None:
            submissions.append(entry)

    if not submissions:
        # Fall back to simpler extraction
        gpu_times: list[float] = []
        for row in root.iter("row"):
            for col in row:
                attr_name = (col.get("name", "") or "").lower()
                text = col.text or ""
                m = re.search(r"([\d.]+)", text)
                if m and "gpu" in attr_name and ("time" in attr_name or "duration" in attr_name):
                    gpu_times.append(float(m.group(1)))
        if gpu_times:
            result["gpu_time_total_ms"] = sum(gpu_times)
            result["gpu_time_avg_ms"] = sum(gpu_times) / len(gpu_times)
            result["gpu_submissions"] = len(gpu_times)
        return result

    # Aggregate totals
    total_gpu_ns = sum(s.get("gpu_time_ns", 0) for s in submissions)
    result["gpu_time_total_ms"] = total_gpu_ns / 1e6
    result["gpu_time_avg_ms"] = (total_gpu_ns / len(submissions)) / 1e6
    result["gpu_submissions"] = len(submissions)

    # Group by encoder type
    type_groups: dict[str, dict] = {}
    for s in submissions:
        etype = s.get("encoder_type", "unknown")
        if etype not in type_groups:
            type_groups[etype] = {"count": 0, "total_ms": 0.0}
        type_groups[etype]["count"] += 1
        type_groups[etype]["total_ms"] += s.get("gpu_time_ns", 0) / 1e6
    result["submissions_by_type"] = type_groups

    # Top 5 submissions by GPU time
    sorted_subs = sorted(submissions, key=lambda s: s.get("gpu_time_ns", 0), reverse=True)
    top_subs = []
    for s in sorted_subs[:5]:
        top_subs.append({
            "label": s.get("label", "(unnamed)"),
            "encoder_type": s.get("encoder_type", "unknown"),
            "gpu_time_ms": s.get("gpu_time_ns", 0) / 1e6,
        })
    result["top_submissions"] = top_subs

    # GPU idle percentage (gaps between submissions)
    if len(submissions) >= 2:
        # Sort by start time
        timed = [s for s in submissions if s.get("start_ns") is not None]
        if len(timed) >= 2:
            timed.sort(key=lambda s: s["start_ns"])
            first_start = timed[0]["start_ns"]
            last_end = max(s["start_ns"] + s.get("gpu_time_ns", 0) for s in timed)
            total_span = last_end - first_start
            if total_span > 0:
                idle_ns = total_span - total_gpu_ns
                result["gpu_idle_pct"] = round(max(0, idle_ns / total_span * 100), 1)

    return result


def _parse_gpu_counters(xml_path: Path) -> dict[str, float]:
    """Extract GPU performance counters from metal-gpu-counter export.

    Counter names vary by GPU generation (M1/M2/M3). We look for common
    patterns and return whatever we find.
    """
    counters: dict[str, float] = {}
    try:
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()
    except (ElementTree.ParseError, OSError):
        logger.warning("Failed to parse GPU counter XML %s", xml_path, exc_info=True)
        return counters

    # Known counter patterns to look for
    patterns = {
        "gpu_active": ["gpu active", "gpu-active", "gpu_active"],
        "alu_utilization": ["alu active", "alu-active", "alu_utilization", "alu utilization"],
        "memory_bandwidth": ["memory bandwidth", "memory-bandwidth", "mem_bandwidth"],
        "occupancy": ["occupancy"],
    }

    for row in root.iter("row"):
        for col in row:
            attr_name = (col.get("name", "") or col.tag).lower()
            text = col.text or ""
            m = re.search(r"([\d.]+)", text)
            if m is None:
                continue
            val = float(m.group(1))

            for key, names in patterns.items():
                if any(name in attr_name for name in names):
                    counters[key] = val
                    break

    return counters
