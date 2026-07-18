"""Profiler selection and shared utilities."""
from __future__ import annotations

import logging
import shlex
from pathlib import Path

from perflab.profilers.base import Profiler
from perflab.profilers.ebpf_profiler import EbpfProfiler
from perflab.profilers.jax_profiler import JaxProfiler
from perflab.profilers.linux_perf import LinuxPerfProfiler
from perflab.profilers.lock_contention import LockContentionProfiler
from perflab.profilers.memray_profiler import MemrayProfiler
from perflab.profilers.metal_trace import MetalTraceProfiler
from perflab.profilers.ncu_profiler import NcuProfiler
from perflab.profilers.nsys_profiler import NsysProfiler
from perflab.profilers.power_profiler import PowerProfiler
from perflab.profilers.python_pyspy import PySpyProfiler
from perflab.profilers.pytorch_profiler import TorchProfiler
from perflab.profilers.thread_sched import ThreadSchedProfiler

logger = logging.getLogger(__name__)


def select_profilers(task):
    """Select appropriate profilers based on task program_type.

    Returns a list of profiler instances suitable for the given task.
    """
    profs: list[Profiler] = []
    # Always attempt py-spy for python-ish tasks
    if task.program_type in {"python", "pytorch", "jax", "triton"}:
        profs.append(PySpyProfiler())
    if task.program_type == "pytorch":
        profs.append(TorchProfiler())

    # Linux perf for all program types
    profs.append(LinuxPerfProfiler())

    # JAX-specific profiler
    if task.program_type == "jax":
        profs.append(JaxProfiler())

    # CUDA profilers for GPU-accelerated types
    cuda_types = {"pytorch", "jax", "triton", "cuda", "cpp"}
    if task.program_type in cuda_types:
        profs.append(NsysProfiler())
        profs.append(NcuProfiler())

    # Metal trace for MPS-capable types on macOS
    mps_types = {"pytorch", "jax", "triton"}
    if task.program_type in mps_types:
        profs.append(MetalTraceProfiler())

    # Memory profiler for Python-based tasks
    if task.program_type in {"python", "pytorch", "jax", "triton"}:
        profs.append(MemrayProfiler())

    # eBPF syscall/IO tracer (Linux only, auto-skipped elsewhere)
    profs.append(EbpfProfiler())

    # Lock contention profiler for multi-threaded C/C++ tasks
    if task.program_type in {"cpp", "cuda"}:
        profs.append(LockContentionProfiler())

    # Thread scheduling profiler for multi-threaded tasks
    if task.program_type in {"cpp", "cuda"}:
        profs.append(ThreadSchedProfiler())

    # Power/energy profiler (RAPL on Linux, nvidia-smi for GPU)
    profs.append(PowerProfiler())

    return profs


def extract_sass_from_build(
    build_cmd: str,
    workspace: Path,
    artifacts_dir: Path,
    *,
    max_kernels: int = 3,
    context_lines: int = 10,
) -> list[dict] | None:
    """Extract CUDA SASS disassembly from a compiled binary.

    Parses the build command to find the -o output binary, then runs
    cuobjdump to extract SASS. Returns a list of SASS snippet dicts
    or None if extraction fails.
    """
    try:
        from perflab.profilers.ncu_profiler import extract_cuda_sass
        build_parts = shlex.split(build_cmd)
        binary_name = None
        for i, p in enumerate(build_parts):
            if p == "-o" and i + 1 < len(build_parts):
                binary_name = build_parts[i + 1]
                break
        if not binary_name:
            return None
        binary_path = workspace / binary_name
        return extract_cuda_sass(
            binary_path,
            max_kernels=max_kernels,
            context_lines=context_lines,
            artifacts_dir=artifacts_dir,
        )
    except FileNotFoundError:
        logger.warning("cuobjdump not found — SASS extraction unavailable")
        return None
    except Exception:  # noqa: BLE001 -- best-effort optional SASS extraction, must not abort profiling
        logger.warning("SASS extraction failed", exc_info=True)
        return None
