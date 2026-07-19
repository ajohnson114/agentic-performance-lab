"""System information collection for reproducible benchmarks."""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def capture_system_info(run_dir: Path) -> dict[str, Any]:
    """Collect system info and persist it as <run_dir>/system_info.json.

    Callers wrap this in their own try/except — capture is best-effort and
    must never abort a run, but how failures are surfaced (logger vs. agent
    progress messages) differs per entrypoint.
    """
    info = collect_system_info()
    (run_dir / "system_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def collect_system_info() -> dict[str, Any]:
    """Collect system information. Each field is wrapped in try/except for portability."""
    info: dict[str, Any] = {}

    # Platform basics
    try:
        info["platform"] = platform.platform()
        info["python_version"] = platform.python_version()
        info["machine"] = platform.machine()
        info["system"] = platform.system()
    except OSError:
        logger.warning("Failed to collect platform basics", exc_info=True)

    # CPU model
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info["cpu_model"] = result.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
    except (OSError, subprocess.SubprocessError):
        pass

    # CPU count
    try:
        info["cpu_count"] = os.cpu_count()
    except OSError:
        pass

    # CPU governor (Linux only)
    try:
        if platform.system() == "Linux":
            gov_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
            if os.path.exists(gov_path):
                with open(gov_path) as f:
                    info["cpu_governor"] = f.read().strip()
    except (OSError, subprocess.SubprocessError):
        pass

    # GPU info via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,persistence_mode,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpu_entry: dict[str, Any] = {
                        "name": parts[0],
                        "memory_mib": parts[1],
                        "driver_version": parts[2],
                    }
                    if len(parts) >= 4:
                        gpu_entry["persistence_mode"] = parts[3]
                    if len(parts) >= 5:
                        gpu_entry["throttle_reasons"] = parts[4]
                    gpus.append(gpu_entry)
            if gpus:
                info["nvidia_gpus"] = gpus
    except FileNotFoundError:
        pass
    except (OSError, subprocess.SubprocessError):
        logger.warning("nvidia-smi query failed", exc_info=True)

    # CUDA version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "release" in line.lower():
                    info["cuda_version"] = line.strip()
                    break
    except FileNotFoundError:
        pass
    except (OSError, subprocess.SubprocessError):
        logger.warning("nvcc version check failed", exc_info=True)

    # PyTorch version
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda or ""
    except ImportError:
        pass

    # JAX version and TPU detection
    try:
        # Suppress JAX fork() warning — JAX is multithreaded and internally uses subprocess
        # which calls os.fork(), triggering a harmless but noisy warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*os.fork.*multithreaded.*")
            import jax
            info["jax_version"] = jax.__version__
            devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == "tpu"]
        if tpu_devices:
            d0 = tpu_devices[0]
            info["tpu_devices"] = [{
                "name": str(d.device_kind),
                "id": d.id,
                "platform": d.platform,
            } for d in tpu_devices]
            info["tpu_chip"] = str(d0.device_kind)
            info["tpu_count"] = len(tpu_devices)
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 -- best-effort hardware probe, must not abort sysinfo collection
        logger.warning("JAX device detection failed", exc_info=True)

    # Triton version
    try:
        import triton
        info["triton_version"] = triton.__version__
    except ImportError:
        pass

    # C++ compiler
    cpp_compiler = detect_cpp_compiler()
    if cpp_compiler:
        info["cpp_compiler"] = cpp_compiler

    # OpenMP version
    try:
        result = subprocess.run(
            ["g++", "-fopenmp", "-dM", "-E", "-x", "c++", "-"],
            input="", capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "_OPENMP" in line:
                    info["openmp_version"] = line.strip()
                    break
    except FileNotFoundError:
        pass
    except (OSError, subprocess.SubprocessError):
        logger.warning("OpenMP version detection failed", exc_info=True)

    # Load average
    try:
        load = os.getloadavg()
        info["load_average"] = {"1min": load[0], "5min": load[1], "15min": load[2]}
    except (OSError, AttributeError):
        pass

    # CPU ISA features
    try:
        info["cpu_isa"] = detect_cpu_isa_features()
    except Exception:  # noqa: BLE001 -- best-effort hardware probe, must not abort sysinfo collection
        logger.warning("CPU ISA feature detection failed", exc_info=True)

    return info


def detect_cpp_compiler() -> str | None:
    """Detect the available C++ compiler, preferring g++ and falling back to c++.

    Each candidate gets its own try/except. They used to share one try block
    with the c++ fallback nested inside the g++ branch's ``else``, so a
    missing g++ raised FileNotFoundError straight out of the block (skipping
    the c++ probe entirely) -- the fallback only ever ran when g++ was
    present but exited nonzero. Returns None if neither is found/usable.
    """
    for compiler in ("g++", "c++"):
        try:
            result = subprocess.run(
                [compiler, "--version"],
                capture_output=True, text=True, timeout=5,
            )
        except FileNotFoundError:
            continue
        except (OSError, subprocess.SubprocessError):
            logger.warning("%s version check failed", compiler, exc_info=True)
            continue
        if result.returncode == 0:
            return result.stdout.splitlines()[0].strip()
    return None


def detect_cpu_isa_features() -> dict:
    """Detect CPU ISA features (AVX, AVX2, AVX-512, FMA, NEON, SSE).

    Returns a dict with boolean flags and max_simd_width_bits.
    Linux: parses /proc/cpuinfo flags.
    macOS: uses sysctl for x86, infers NEON for ARM.
    """
    features = {
        "sse": False,
        "sse2": False,
        "sse4_1": False,
        "sse4_2": False,
        "avx": False,
        "avx2": False,
        "avx512f": False,
        "fma": False,
        "neon": False,
        "max_simd_width_bits": 0,
    }

    system = platform.system()
    machine = platform.machine()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("flags"):
                        flags_str = line.split(":", 1)[1].lower()
                        flags_set = set(flags_str.split())
                        features["sse"] = "sse" in flags_set
                        features["sse2"] = "sse2" in flags_set
                        features["sse4_1"] = "sse4_1" in flags_set
                        features["sse4_2"] = "sse4_2" in flags_set
                        features["avx"] = "avx" in flags_set
                        features["avx2"] = "avx2" in flags_set
                        features["avx512f"] = "avx512f" in flags_set
                        features["fma"] = "fma" in flags_set
                        break
                    # ARM: check for neon
                    if line.startswith("Features"):
                        flags_str = line.split(":", 1)[1].lower()
                        if "neon" in flags_str or "asimd" in flags_str:
                            features["neon"] = True
                        break
        except OSError:
            logger.warning("Failed to read /proc/cpuinfo for ISA features", exc_info=True)

    elif system == "Darwin":
        if machine in ("arm64", "aarch64"):
            # Apple Silicon: always has NEON (128-bit SIMD)
            features["neon"] = True
        else:
            # Intel Mac: use sysctl
            _sysctl_flags = {
                "avx": "hw.optional.avx1_0",
                "avx2": "hw.optional.avx2_0",
                "avx512f": "hw.optional.avx512f",
                "fma": "hw.optional.fma",
                "sse": "hw.optional.sse",
                "sse2": "hw.optional.sse2",
                "sse4_1": "hw.optional.sse4_1",
                "sse4_2": "hw.optional.sse4_2",
            }
            for feat, sysctl_key in _sysctl_flags.items():
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", sysctl_key],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip() == "1":
                        features[feat] = True
                except (OSError, subprocess.SubprocessError):
                    pass

    # Compute max SIMD width
    if features["avx512f"]:
        features["max_simd_width_bits"] = 512
    elif features["avx2"] or features["avx"]:
        features["max_simd_width_bits"] = 256
    elif features["sse"] or features["sse2"] or features["neon"]:
        features["max_simd_width_bits"] = 128

    return features


def warn_if_noisy() -> list[str]:
    """Return warnings if system conditions may cause noisy benchmarks."""
    warnings: list[str] = []

    # Non-performance CPU governor
    try:
        if platform.system() == "Linux":
            gov_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
            if os.path.exists(gov_path):
                with open(gov_path) as f:
                    governor = f.read().strip()
                if governor != "performance":
                    warnings.append(
                        f"CPU governor is '{governor}', not 'performance'. "
                        f"Benchmark results may be noisy. "
                        f"Run: sudo cpupower frequency-set -g performance"
                    )
    except OSError:
        pass

    # High load average
    try:
        load_1min = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        if load_1min > cpu_count * 0.7:
            warnings.append(
                f"System load ({load_1min:.1f}) is high relative to CPU count ({cpu_count}). "
                f"Other processes may affect benchmark results."
            )
    except (OSError, AttributeError):
        pass

    # GPU persistence mode
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=persistence_mode", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().splitlines()):
                if line.strip().lower() == "disabled":
                    warnings.append(
                        f"GPU {i} persistence mode is Disabled. "
                        f"This adds latency to the first CUDA call. "
                        f"Run: sudo nvidia-smi -pm 1"
                    )
    except FileNotFoundError:
        pass
    except (OSError, subprocess.SubprocessError):
        logger.warning("GPU persistence mode check failed", exc_info=True)

    # Transparent hugepages (Linux only)
    try:
        if platform.system() == "Linux":
            thp_path = "/sys/kernel/mm/transparent_hugepage/enabled"
            if os.path.exists(thp_path):
                with open(thp_path) as f:
                    content = f.read().strip()
                # The active setting is enclosed in brackets, e.g. "[never] always madvise"
                if "[never]" in content:
                    warnings.append(
                        "Transparent hugepages are set to 'never'. "
                        "This may reduce performance for memory-intensive workloads. "
                        "Run: echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"
                    )
    except OSError:
        pass

    # GPU throttling
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks_throttle_reasons.active",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().splitlines()):
                val = line.strip()
                # nvidia-smi reports throttle reasons as a hex bitmask; 0x0 means no throttling
                if val and val != "0x0000000000000000" and val.lower() not in ("0x0", "0", "not supported"):
                    warnings.append(
                        f"GPU {i} is being throttled (reason: {val}). "
                        f"This may indicate thermal, power, or other throttling. "
                        f"Check GPU cooling and power limits."
                    )
    except FileNotFoundError:
        pass
    except (OSError, subprocess.SubprocessError):
        logger.warning("GPU throttle check failed", exc_info=True)

    # GPU clock locking
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.current.sm,clocks.max.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().splitlines()):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        current = int(parts[0])
                        maximum = int(parts[1])
                        if maximum > 0 and current < maximum * 0.95:
                            warnings.append(
                                f"GPU {i} SM clock ({current} MHz) is below max ({maximum} MHz). "
                                f"Clocks are not locked — benchmark variance may be ~{(maximum - current) * 100 // maximum}%. "
                                f"Run: sudo nvidia-smi -lgc {maximum},{maximum}"
                            )
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    except (OSError, subprocess.SubprocessError):
        logger.warning("GPU clock check failed", exc_info=True)

    # GPU temperature — warn if hot before benchmarking
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().splitlines()):
                try:
                    temp_c = int(line.strip())
                    if temp_c > 80:
                        warnings.append(
                            f"GPU {i} temperature is {temp_c}°C — thermal throttling likely. "
                            f"Let GPU cool before benchmarking for stable results."
                        )
                    elif temp_c > 70:
                        warnings.append(
                            f"GPU {i} temperature is {temp_c}°C — approaching thermal throttle zone. "
                            f"Results may degrade over long runs."
                        )
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    except (OSError, subprocess.SubprocessError):
        logger.warning("GPU temperature check failed", exc_info=True)

    # Multi-GPU without CUDA_VISIBLE_DEVICES
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpu_count = len(result.stdout.strip().splitlines())
            if gpu_count > 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
                warnings.append(
                    f"Multi-GPU node detected ({gpu_count} GPUs) but CUDA_VISIBLE_DEVICES is not set. "
                    f"Other GPU processes may interfere with benchmarks. "
                    f"Run: export CUDA_VISIBLE_DEVICES=0"
                )
    except FileNotFoundError:
        pass
    except (OSError, subprocess.SubprocessError):
        logger.warning("Multi-GPU detection failed", exc_info=True)

    return warnings
