from __future__ import annotations

import logging
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def _detect_cpu_vendor() -> str:
    """Detect CPU vendor: 'intel', 'amd', or 'unknown'."""
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
        logger.debug("CPU vendor detection failed", exc_info=True)
    return "unknown"


@dataclass
class CheckResult:
    name: str
    status: Literal["pass", "fail", "warn"]
    message: str


def check_python_version() -> CheckResult:
    ver = sys.version_info
    version_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    if ver >= (3, 10):
        return CheckResult("python_version", "pass", f"Python {version_str}")
    return CheckResult("python_version", "fail", f"Python {version_str} (need >=3.10)")


def check_required_packages() -> list[CheckResult]:
    results: list[CheckResult] = []
    for pkg in ("typer", "yaml", "matplotlib"):
        import_name = pkg
        try:
            __import__(import_name)
            results.append(CheckResult(f"pkg:{pkg}", "pass", "installed"))
        except ImportError:
            results.append(CheckResult(f"pkg:{pkg}", "fail", "not installed"))
    return results


def check_security() -> list[CheckResult]:
    """Check for security issues (config file permissions, etc.)."""
    results: list[CheckResult] = []

    config_path = Path.home() / ".config" / "perflab" / "config.yaml"
    if config_path.exists():
        try:
            import stat
            mode = config_path.stat().st_mode
            if mode & stat.S_IROTH:
                results.append(CheckResult("security:config_perms", "fail",
                    f"Config file {config_path} is WORLD-READABLE (contains API keys). "
                    f"Fix with: chmod 600 {config_path}"))
            elif mode & stat.S_IRGRP:
                results.append(CheckResult("security:config_perms", "warn",
                    f"Config file {config_path} is group-readable. "
                    f"Tighten with: chmod 600 {config_path}"))
            else:
                results.append(CheckResult("security:config_perms", "pass",
                    "Config file permissions OK (owner-only)"))
        except OSError:
            results.append(CheckResult("security:config_perms", "pass", "Could not check permissions"))
    else:
        results.append(CheckResult("security:config_perms", "pass", "No config file yet (run perflab init)"))

    return results


def check_optional_packages() -> list[CheckResult]:
    results: list[CheckResult] = []
    for pkg, import_name in [("torch", "torch"), ("openai", "openai"), ("anthropic", "anthropic"), ("fastmcp", "fastmcp")]:
        try:
            __import__(import_name)
            results.append(CheckResult(f"pkg:{pkg}", "pass", "installed"))
        except ImportError:
            results.append(CheckResult(f"pkg:{pkg}", "warn", "not installed (optional)"))
    return results


def check_profiler_tools() -> list[CheckResult]:
    results: list[CheckResult] = []
    is_linux = platform.system() == "Linux"
    is_macos = platform.system() == "Darwin"

    # py-spy (cross-platform) — CPU sampling profiler for Python, generates flame graphs
    if shutil.which("py-spy"):
        results.append(CheckResult("tool:py-spy", "pass", "found (CPU flame graph profiler for Python)"))
    else:
        results.append(CheckResult("tool:py-spy", "warn",
            "not found (CPU flame graph profiler for Python) — install with: pip install py-spy"))

    # memray (cross-platform) — Python memory profiler, tracks allocations and peak usage
    if shutil.which("memray"):
        results.append(CheckResult("tool:memray", "pass", "found (Python memory allocation profiler)"))
    else:
        results.append(CheckResult("tool:memray", "warn",
            "not found (Python memory allocation profiler) — install with: pip install memray"))

    # perf (Linux only) — hardware counter profiler: IPC, cache misses, branch mispredictions
    if is_linux:
        if shutil.which("perf"):
            results.append(CheckResult("tool:perf", "pass", "found (hardware counter profiler: IPC, cache misses)"))
        else:
            results.append(CheckResult("tool:perf", "warn",
                "not found (optional: hardware counter profiler) — not available on all systems"))
    else:
        results.append(CheckResult("tool:perf", "pass", "N/A (Linux only)"))

    # bpftrace (Linux only) — eBPF-based I/O and syscall tracer
    if is_linux:
        if shutil.which("bpftrace"):
            results.append(CheckResult("tool:bpftrace", "pass", "found (eBPF I/O and syscall tracer)"))
        else:
            results.append(CheckResult("tool:bpftrace", "warn",
                "not found (optional: eBPF I/O tracer) — may not be available in all cloud environments"))
    else:
        results.append(CheckResult("tool:bpftrace", "pass", "N/A (Linux only)"))

    # nsys (NVIDIA) — GPU timeline profiler: kernel launches, memory transfers, API overhead
    if shutil.which("nsys"):
        results.append(CheckResult("tool:nsys", "pass", "found (NVIDIA GPU timeline profiler)"))
    else:
        results.append(CheckResult("tool:nsys", "warn",
            "not found (optional: NVIDIA GPU timeline profiler) — usually pre-installed on cloud GPU providers"))

    # ncu (NVIDIA) — GPU kernel profiler: SM utilization, memory throughput, occupancy
    if shutil.which("ncu"):
        results.append(CheckResult("tool:ncu", "pass", "found (NVIDIA GPU kernel profiler)"))
    else:
        results.append(CheckResult("tool:ncu", "warn",
            "not found (optional: NVIDIA GPU kernel profiler) — usually pre-installed on cloud GPU providers"))

    # TMA Level 2/3 — toplev for Intel, perf cache events for AMD
    if is_linux:
        cpu_vendor = _detect_cpu_vendor()
        if cpu_vendor == "intel":
            has_toplev = shutil.which("toplev") or shutil.which("toplev.py")
            has_pmu = False
            if not has_toplev:
                # Try importing pmu-tools (multiple possible module names)
                for mod in ("pmu", "toplev"):
                    try:
                        __import__(mod)
                        has_pmu = True
                        break
                    except ImportError:
                        pass

            if has_toplev or has_pmu:
                results.append(CheckResult("tool:tma-l2", "pass",
                    "Intel TMA Level 2/3 available (L1/L2/L3/DRAM Bound analysis)"))
            else:
                results.append(CheckResult("tool:tma-l2", "warn",
                    "Intel CPU detected but toplev not found (TMA Level 2/3 analysis) "
                    "— install with: pip install pmu-tools"))
        elif cpu_vendor == "amd":
            # AMD uses perf stat with cache hierarchy events — check perf is available
            if shutil.which("perf"):
                results.append(CheckResult("tool:tma-l2", "pass",
                    "AMD CPU detected — perf available for cache hierarchy analysis "
                    "(L1/LLC miss rates for memory bottleneck identification)"))
            else:
                results.append(CheckResult("tool:tma-l2", "warn",
                    "AMD CPU detected but perf not found — install linux-tools for "
                    "cache hierarchy analysis (L1-dcache-load-misses, LLC-load-misses)"))
        else:
            results.append(CheckResult("tool:tma-l2", "warn",
                "CPU vendor not detected — TMA Level 2/3 requires Intel (toplev) or AMD (perf)"))
    else:
        results.append(CheckResult("tool:tma-l2", "pass", "N/A (Linux only)"))

    # cuobjdump (NVIDIA) — CUDA SASS disassembly for GPU kernel instruction analysis
    if shutil.which("cuobjdump"):
        results.append(CheckResult("tool:cuobjdump", "pass",
            "found (CUDA SASS disassembly for GPU kernel analysis)"))
    else:
        if shutil.which("nvcc"):
            results.append(CheckResult("tool:cuobjdump", "warn",
                "not found but nvcc is present — cuobjdump should be in CUDA toolkit bin/"))
        else:
            results.append(CheckResult("tool:cuobjdump", "warn",
                "not found (CUDA SASS disassembly) — bundled with CUDA toolkit"))

    # c++filt — C++ name demangling for kernel name resolution
    if shutil.which("c++filt"):
        results.append(CheckResult("tool:c++filt", "pass",
            "found (C++ name demangling for kernel names)"))
    else:
        results.append(CheckResult("tool:c++filt", "warn",
            "not found (C++ name demangling) — install with: sudo apt install binutils (Linux) or brew install binutils (macOS)"))

    # xctrace (macOS only) — Metal GPU profiler for Apple Silicon
    if is_macos:
        if shutil.which("xctrace"):
            try:
                import subprocess
                r = subprocess.run(
                    ["xctrace", "list", "templates"],
                    capture_output=True, timeout=10,
                )
                if r.returncode == 0:
                    results.append(CheckResult("tool:xctrace", "pass", "found (Metal GPU profiler for Apple Silicon)"))
                else:
                    results.append(CheckResult("tool:xctrace", "warn",
                        "found but not functional (Metal GPU profiler — full Xcode required, not just Command Line Tools)"))
            except (OSError, subprocess.SubprocessError):
                results.append(CheckResult("tool:xctrace", "warn", "found but not functional"))
        else:
            results.append(CheckResult("tool:xctrace", "warn",
                "not found (Metal GPU profiler for Apple Silicon) "
                "— install Xcode from the App Store (Command Line Tools alone are not sufficient)"))
    else:
        results.append(CheckResult("tool:xctrace", "pass", "N/A (macOS only)"))

    return results


def check_hardware() -> list[CheckResult]:
    results: list[CheckResult] = []
    is_macos = platform.system() == "Darwin"

    # CUDA
    if shutil.which("nvidia-smi"):
        results.append(CheckResult("hw:cuda", "pass", "nvidia-smi found"))
    else:
        results.append(CheckResult("hw:cuda", "warn",
            "nvidia-smi not found — install NVIDIA drivers or CUDA toolkit"))

    # MPS/Metal
    if is_macos:
        results.append(CheckResult("hw:metal", "pass", "macOS detected (Metal likely available)"))
    else:
        results.append(CheckResult("hw:metal", "pass", "N/A (macOS only)"))

    # JAX device detection (GPU/Metal, TPU)
    try:
        import jax
        devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == "tpu"]
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        cpu_only = all(d.platform == "cpu" for d in devices)

        if tpu_devices:
            chip = str(tpu_devices[0].device_kind)
            results.append(CheckResult("hw:tpu", "pass",
                f"detected {len(tpu_devices)} {chip} chip(s) via JAX"))
        else:
            results.append(CheckResult("hw:tpu", "pass", "no TPU detected (JAX sees GPU/CPU only)"))

        # JAX GPU (Metal on Mac, CUDA on Linux)
        if gpu_devices:
            kinds = {str(d.device_kind) for d in gpu_devices}
            results.append(CheckResult("hw:jax_gpu", "pass",
                f"JAX sees {len(gpu_devices)} GPU device(s): {', '.join(kinds)}"))
        elif is_macos and cpu_only:
            results.append(CheckResult("hw:jax_gpu", "warn",
                "JAX installed but sees CPU only on macOS — install jax-metal for GPU acceleration: "
                "pip install jax-metal (or pip install -e \".[tasks-jax-metal]\")"))
        else:
            results.append(CheckResult("hw:jax_gpu", "pass",
                "JAX sees CPU only (expected on CPU-only systems)"))
    except ImportError:
        results.append(CheckResult("hw:tpu", "pass", "JAX not installed (TPU requires JAX)"))
        results.append(CheckResult("hw:jax_gpu", "pass", "JAX not installed"))
    except Exception:
        logger.warning("Could not query JAX devices", exc_info=True)
        results.append(CheckResult("hw:tpu", "pass", "could not query JAX devices"))
        results.append(CheckResult("hw:jax_gpu", "pass", "could not query JAX devices"))

    return results


def check_llm_provider() -> CheckResult:
    """Check if any LLM provider is configured. Uses lazy import to avoid circular deps."""
    try:
        from perflab.llm.config import LLMConfig, create_provider
        cfg = LLMConfig.load()
        provider = create_provider(cfg)
        if provider.is_available():
            return CheckResult("llm_provider", "pass", f"{cfg.provider}:{cfg.model}")
        return CheckResult("llm_provider", "warn", f"{cfg.provider} configured but not available")
    except Exception as exc:
        return CheckResult("llm_provider", "warn", f"not configured ({exc})")


def run_doctor(
    check_profilers: bool = True,
    check_llm: bool = True,
    check_all: bool = False,
) -> list[CheckResult]:
    """Run environment checks and return results."""
    if check_all:
        check_profilers = True
        check_llm = True

    results: list[CheckResult] = []

    # Always check basics
    results.append(check_python_version())
    results.extend(check_security())
    results.extend(check_required_packages())
    results.extend(check_optional_packages())
    results.extend(check_hardware())

    if check_profilers:
        results.extend(check_profiler_tools())

    if check_llm:
        results.append(check_llm_provider())

    return results
