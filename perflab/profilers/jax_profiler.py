from __future__ import annotations

import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from perflab.profilers.base import ProfileResult, run_bench_under
from perflab.profilers.interval_union import union_duration

logger = logging.getLogger(__name__)


@dataclass
class JaxProfiler:
    name: str = "jax"

    def is_available(self) -> bool:
        try:
            import jax  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        hlo_dir = artifacts_dir / "xla_hlo_dump"
        hlo_dir.mkdir(parents=True, exist_ok=True)
        trace_dir = artifacts_dir / "jax_trace"
        trace_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["JAX_LOG_COMPILES"] = "1"
        existing_flags = env.get("XLA_FLAGS", "")
        xla_flags = f"{existing_flags} --xla_dump_hlo_as_text --xla_dump_to={hlo_dir}"
        # Enable TPU profiling via TensorBoard trace if on TPU
        env["XLA_FLAGS"] = xla_flags.strip()
        # JAX profiler trace: writes TensorBoard-compatible trace files
        env["PERFLAB_JAX_TRACE_DIR"] = str(trace_dir)

        t0 = time.perf_counter()
        res = run_bench_under([], bench_cmd, cwd=cwd, env=env)
        duration_s = time.perf_counter() - t0

        summary: dict = {
            "returncode": res.returncode,
            "duration_s": round(duration_s, 3),
        }

        # Parse compilation metrics from stderr
        comp_metrics = _parse_compilation_metrics(res.stderr)
        summary.update(comp_metrics)

        # Parse HLO dump if produced
        hlo_metrics = _parse_hlo_dump(hlo_dir)
        summary.update(hlo_metrics)

        # Detect TPU and add device info
        tpu_info = _detect_tpu_device()
        if tpu_info:
            summary.update(tpu_info)

        # Try programmatic jax.profiler.trace for TPU/GPU metrics
        trace_metrics = _collect_jax_trace_metrics(trace_dir)
        if trace_metrics:
            summary.update(trace_metrics)

        artifacts: dict[str, str] = {}
        if hlo_dir.exists() and any(hlo_dir.iterdir()):
            artifacts["xla_hlo_dump"] = str(hlo_dir)
        if trace_dir.exists() and any(trace_dir.iterdir()):
            artifacts["jax_trace"] = str(trace_dir)

        return ProfileResult(name=self.name, artifacts=artifacts, summary=summary)


def _parse_compilation_metrics(stderr: str) -> dict:
    """Extract XLA compilation metrics from JAX log output."""
    result: dict = {}
    if not stderr:
        return result

    # Count compilation log lines (JAX_LOG_COMPILES=1 emits lines like:
    #   "Compiling <func> for args..." or "Compilation of <func> took Nms")
    compile_lines = re.findall(
        r"(?:Compil(?:ing|ation)\b)", stderr, re.IGNORECASE,
    )
    result["xla_compilations"] = len(compile_lines) // 2 or len(compile_lines)

    # Sum compilation times (regex on "compilation...Nms" or "took N ms")
    time_matches = re.findall(
        r"(?:took|compilation[^0-9]*?)(\d+(?:\.\d+)?)\s*ms", stderr, re.IGNORECASE,
    )
    if time_matches:
        result["xla_compilation_time_ms"] = round(
            sum(float(t) for t in time_matches), 2,
        )

    # Count recompilation warnings
    recomp_matches = re.findall(r"recompil", stderr, re.IGNORECASE)
    result["xla_recompilations"] = len(recomp_matches)

    return result


def _parse_hlo_dump(hlo_dir: Path) -> dict:
    """Parse XLA HLO text dump files for module and operation statistics."""
    result: dict = {}
    if not hlo_dir.exists():
        return result

    hlo_files = list(hlo_dir.glob("*.txt")) + list(hlo_dir.glob("*.hlo"))
    if not hlo_files:
        return result

    result["hlo_module_count"] = len(hlo_files)

    # Count HLO operation types across all files
    op_counter: Counter = Counter()
    # HLO ops appear as: ROOT name = <op_type>[shape](operands)
    # or: name = <op_type>(operands)
    op_pattern = re.compile(r"=\s*(\w+)[\[\(]")

    for f in hlo_files[:20]:  # Cap to avoid scanning huge dumps
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            for match in op_pattern.finditer(text):
                op_name = match.group(1)
                # Filter out metadata / non-op tokens
                if op_name not in ("param", "ROOT", "HloModule", "ENTRY"):
                    op_counter[op_name] += 1
        except (OSError, UnicodeDecodeError):
            continue

    if op_counter:
        result["hlo_ops"] = [
            {"op": op, "count": count}
            for op, count in op_counter.most_common(10)
        ]

    # Extract FLOP estimates from HLO metadata comments
    # XLA HLO dumps may contain cost annotations like:
    #   "// cost: flops=N, bytes_accessed=M"
    #   or "metadata={op_name="..." cost_estimate={flops=N}}"
    total_flops: float = 0.0
    total_bytes_accessed: float = 0.0
    flop_pattern = re.compile(r"flops[=:]\s*(\d+(?:\.\d+)?(?:e\+?\d+)?)", re.IGNORECASE)
    bytes_pattern = re.compile(r"bytes_accessed[=:]\s*(\d+(?:\.\d+)?(?:e\+?\d+)?)", re.IGNORECASE)

    for f in hlo_files[:20]:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            for m in flop_pattern.finditer(text):
                total_flops += float(m.group(1))
            for m in bytes_pattern.finditer(text):
                total_bytes_accessed += float(m.group(1))
        except (OSError, UnicodeDecodeError, ValueError):
            continue

    if total_flops > 0:
        result["hlo_cost_flops"] = total_flops
        result["hlo_cost_tflops"] = round(total_flops / 1e12, 6)
    if total_bytes_accessed > 0:
        result["hlo_cost_bytes_accessed"] = total_bytes_accessed

    return result


def _detect_tpu_device() -> dict:
    """Detect TPU device info via jax.devices()."""
    try:
        import jax
        devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == "tpu"]
        if not tpu_devices:
            return {}
        d0 = tpu_devices[0]
        return {
            "tpu_chip": str(d0.device_kind),
            "tpu_count": len(tpu_devices),
            "tpu_device_ids": [d.id for d in tpu_devices],
        }
    except ImportError:
        return {}
    except Exception:  # noqa: BLE001 -- best-effort TPU detection, must not abort profiling
        logger.warning("TPU device detection failed", exc_info=True)
        return {}


def _collect_jax_trace_metrics(trace_dir: Path) -> dict:
    """Parse JAX profiler trace output for MXU utilization and host-device timing.

    JAX traces produce TensorBoard-compatible protobuf or JSON trace files.
    We parse what we can to extract TPU/GPU performance metrics.
    """
    result: dict = {}
    if not trace_dir.exists():
        return result

    trace_files = (
        list(trace_dir.glob("*.trace.json"))
        + list(trace_dir.glob("*.json"))
        + list(trace_dir.glob("*.pb"))
        + list(trace_dir.glob("*.xplane.pb"))
    )
    if not trace_files:
        return result

    # Parse JSON trace files (Chrome trace format from jax.profiler.trace)
    host_time_us: float = 0.0
    device_time_us: float = 0.0
    mxu_events: list[float] = []
    infeed_time_us: float = 0.0
    total_step_time_us: float = 0.0

    for tf in trace_files[:5]:  # Cap to avoid processing huge trace dirs
        if not tf.suffix == ".json":
            continue
        try:
            import json
            data = json.loads(tf.read_text(encoding="utf-8", errors="replace"))
            events = data if isinstance(data, list) else data.get("traceEvents", [])
            # Per-file busy intervals: events overlap within a trace
            # (concurrent device streams, nested host spans), so wall-clock
            # time is the union of intervals, not the sum of durations.
            # Events without a timestamp fall back to plain sums.
            host_iv: list[tuple[float, float]] = []
            device_iv: list[tuple[float, float]] = []
            infeed_iv: list[tuple[float, float]] = []
            all_iv: list[tuple[float, float]] = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                cat = ev.get("cat", "")
                dur = ev.get("dur", 0)
                name = ev.get("name", "")
                ts = ev.get("ts")
                interval = None
                if isinstance(ts, (int, float)) and dur > 0:
                    interval = (float(ts), float(ts) + float(dur))

                # Categorize events by host vs device
                if "host" in cat.lower() or cat in ("python", "cpu_op"):
                    if interval:
                        host_iv.append(interval)
                    else:
                        host_time_us += dur
                elif "device" in cat.lower() or "tpu" in cat.lower() or "gpu" in cat.lower():
                    if interval:
                        device_iv.append(interval)
                    else:
                        device_time_us += dur

                # MXU utilization events (XProf format)
                args = ev.get("args", {})
                if "mxu_utilization" in args:
                    try:
                        mxu_events.append(float(args["mxu_utilization"]))
                    except (ValueError, TypeError):
                        pass

                # Infeed stall tracking
                if "infeed" in name.lower() and dur > 0:
                    if interval:
                        infeed_iv.append(interval)
                    else:
                        infeed_time_us += dur

                if interval:
                    all_iv.append(interval)
                else:
                    total_step_time_us += dur

            host_time_us += union_duration(host_iv)
            device_time_us += union_duration(device_iv)
            infeed_time_us += union_duration(infeed_iv)
            total_step_time_us += union_duration(all_iv)
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            continue

    if host_time_us > 0 or device_time_us > 0:
        result["host_time_us"] = round(host_time_us, 1)
        result["device_time_us"] = round(device_time_us, 1)
        total = host_time_us + device_time_us
        if total > 0:
            result["device_fraction"] = round(device_time_us / total, 3)

    if mxu_events:
        avg_mxu = sum(mxu_events) / len(mxu_events)
        result["mxu_utilization_pct"] = round(avg_mxu, 1)

    if infeed_time_us > 0 and total_step_time_us > 0:
        result["infeed_stall_pct"] = round(
            (infeed_time_us / total_step_time_us) * 100, 1,
        )

    return result
