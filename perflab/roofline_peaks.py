from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class Peaks:
    peak_tflops: float
    peak_mem_bw_gbs: float
    source: str
    device: str
    dtype_peaks: dict[str, float] | None = None  # per-dtype peaks if available


# Known per-dtype peaks (TFLOPS) for common GPUs.
# Values are theoretical peaks for tensor core / CUDA core operations.
_KNOWN_GPU_DTYPE_PEAKS: dict[str, dict[str, float]] = {
    "A100": {
        "peak_tflops_fp32": 19.5,
        "peak_tflops_tf32": 156.0,
        "peak_tflops_fp16": 312.0,
        "peak_tflops_bf16": 312.0,
    },
    "A100-SXM": {
        "peak_tflops_fp32": 19.5,
        "peak_tflops_tf32": 156.0,
        "peak_tflops_fp16": 312.0,
        "peak_tflops_bf16": 312.0,
    },
    "H100": {
        "peak_tflops_fp32": 67.0,
        "peak_tflops_tf32": 989.0,
        "peak_tflops_fp16": 1979.0,
        "peak_tflops_bf16": 1979.0,
    },
    "H100-SXM": {
        "peak_tflops_fp32": 67.0,
        "peak_tflops_tf32": 989.0,
        "peak_tflops_fp16": 1979.0,
        "peak_tflops_bf16": 1979.0,
    },
    "RTX 4090": {
        "peak_tflops_fp32": 82.6,
        "peak_tflops_tf32": 82.6,
        "peak_tflops_fp16": 165.2,
        "peak_tflops_bf16": 165.2,
    },
    "RTX 4080": {
        "peak_tflops_fp32": 48.7,
        "peak_tflops_tf32": 48.7,
        "peak_tflops_fp16": 97.5,
        "peak_tflops_bf16": 97.5,
    },
    "RTX 3090": {
        "peak_tflops_fp32": 35.6,
        "peak_tflops_tf32": 71.0,
        "peak_tflops_fp16": 142.0,
        "peak_tflops_bf16": 142.0,
    },
    "V100": {
        "peak_tflops_fp32": 15.7,
        "peak_tflops_fp16": 125.0,
    },
}


# Known L2 cache bandwidth (GB/s) for common GPUs.
# L2 is the practical upper bound for data-reuse-heavy kernels (tiled matmuls).
# Values are approximate peak aggregate read bandwidth across all SMs.
_KNOWN_GPU_L2_BW: dict[str, float] = {
    "A100": 6000.0,       # 40 MB L2, ~6 TB/s aggregate read
    "A100-SXM": 6000.0,
    "H100": 12000.0,      # 50 MB L2, ~12 TB/s aggregate read
    "H100-SXM": 12000.0,
    "RTX 4090": 3200.0,   # 72 MB L2, ~3.2 TB/s
    "RTX 4080": 2400.0,   # 64 MB L2, ~2.4 TB/s
    "RTX 3090": 2400.0,   # 6 MB L2, ~2.4 TB/s
    "V100": 3100.0,       # 6 MB L2, ~3.1 TB/s
}


def _lookup_l2_bw(device_name: str) -> float | None:
    """Try to match a device name against known GPU L2 bandwidth tables."""
    name_upper = (device_name or "").upper()
    for key, bw in _KNOWN_GPU_L2_BW.items():
        if key.upper() in name_upper:
            return bw
    return None


# Known GPU SM resource limits for hardware context in LLM prompts.
# max_regs_per_thread: always 255 for CUDA (hardware limit)
# max_smem_per_sm_kb: configurable shared memory per SM
# num_sms: number of streaming multiprocessors
_KNOWN_GPU_SM_SPECS: dict[str, dict[str, int | float]] = {
    "V100":     {"max_smem_per_sm_kb": 96,  "num_sms": 80,  "max_regs_per_sm": 65536},
    "A100":     {"max_smem_per_sm_kb": 164, "num_sms": 108, "max_regs_per_sm": 65536},
    "A100-SXM": {"max_smem_per_sm_kb": 164, "num_sms": 108, "max_regs_per_sm": 65536},
    "RTX 3090": {"max_smem_per_sm_kb": 100, "num_sms": 82,  "max_regs_per_sm": 65536},
    "RTX 4090": {"max_smem_per_sm_kb": 100, "num_sms": 128, "max_regs_per_sm": 65536},
    "RTX 4080": {"max_smem_per_sm_kb": 100, "num_sms": 76,  "max_regs_per_sm": 65536},
    "H100":     {"max_smem_per_sm_kb": 228, "num_sms": 132, "max_regs_per_sm": 65536},
    "H100-SXM": {"max_smem_per_sm_kb": 228, "num_sms": 132, "max_regs_per_sm": 65536},
}


def _lookup_sm_specs(device_name: str) -> dict[str, int | float] | None:
    """Look up SM resource limits for a GPU."""
    name_upper = (device_name or "").upper()
    for key, specs in _KNOWN_GPU_SM_SPECS.items():
        if key.upper() in name_upper:
            return dict(specs)
    return None


def _lookup_dtype_peaks(device_name: str) -> dict[str, float] | None:
    """Try to match a device name against known GPU dtype peak tables."""
    name_upper = (device_name or "").upper()
    for key, peaks in _KNOWN_GPU_DTYPE_PEAKS.items():
        if key.upper() in name_upper:
            return dict(peaks)
    return None

_CACHE_PATH = Path(os.environ.get("PERFLAB_PEAKS_CACHE", str(Path.home() / ".cache" / "perflab" / "peaks.json")))

def cache_path() -> Path:
    return _CACHE_PATH

def _use_cache() -> bool:
    return os.environ.get("PERFLAB_PEAKS_NO_CACHE", "").strip() == ""

def _load_cache() -> dict[str, Any]:
    if not _use_cache():
        return {}
    try:
        if _CACHE_PATH.exists():
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load roofline peaks cache", exc_info=True)
    return {}

def _save_cache(cache: dict[str, Any]) -> None:
    if not _use_cache():
        return
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("Failed to save roofline peaks cache", exc_info=True)

def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except (OSError, subprocess.SubprocessError):
        return None

# ---------------- CUDA (multi-GPU) ----------------

def _nvidia_smi_query(fields: list[str]) -> list[dict[str,str]] | None:
    q = ",".join(fields)
    out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    res = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        d = {}
        for k, v in zip(fields, parts):
            d[k] = v
        res.append(d)
    return res

def _visible_cuda_indices() -> list[int]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return []
    if any(c.isalpha() for c in cvd):
        return []  # UUIDs; ignore
    idx = []
    for tok in cvd.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idx.append(int(tok))
        except ValueError:
            continue
    return idx

def _pick_gpu(gpus: list[dict[str,str]], preferred_index: int | None) -> tuple[int, dict[str,str]]:
    if not gpus:
        raise RuntimeError("no gpus")
    if preferred_index is not None and 0 <= preferred_index < len(gpus):
        return preferred_index, gpus[preferred_index]
    vis = _visible_cuda_indices()
    if vis:
        i = min(max(vis[0], 0), len(gpus)-1)
        return i, gpus[i]
    return 0, gpus[0]

def _cores_per_sm(cc: str) -> int | None:
    try:
        major = int(str(cc).split(".")[0])
        minor = int(str(cc).split(".")[1]) if "." in str(cc) else 0
    except (ValueError, IndexError):
        return None
    if major == 7:
        return 64
    if major == 8:
        return 128 if minor >= 6 else 64
    if major >= 9:
        return 128
    if major == 6:
        return 128 if minor >= 1 else 64
    return None

def infer_cuda_peaks(preferred_index: int | None = None) -> Peaks | None:
    fields = ["name", "compute_cap", "clocks.max.sm", "memory.clock", "memory.bus_width", "multiprocessor_count"]
    gpus = _nvidia_smi_query(fields)
    if not gpus:
        fields2 = [f for f in fields if f != "multiprocessor_count"]
        gpus = _nvidia_smi_query(fields2)
        if not gpus:
            return None

    idx, g = _pick_gpu(gpus, preferred_index)
    name = g.get("name", f"cuda:{idx}")
    cc = g.get("compute_cap", "")
    device_id = f"{name} cc{cc} idx{idx}"

    bw = None
    try:
        bus = float(g.get("memory.bus_width", "0"))
        mem_mhz = float(g.get("memory.clock", "0"))
        if bus > 0 and mem_mhz > 0:
            bw = (bus / 8.0) * (mem_mhz * 1e6) * 2.0 / 1e9
    except (ValueError, TypeError, ZeroDivisionError):
        bw = None

    tflops = None
    try:
        sm_clock_ghz = float(g.get("clocks.max.sm", "0")) / 1000.0
        sms = None
        if g.get("multiprocessor_count"):
            try:
                sms = int(float(g["multiprocessor_count"]))
            except (ValueError, TypeError):
                sms = None
        csm = _cores_per_sm(cc)
        if sms is not None and csm is not None and sm_clock_ghz > 0:
            tflops = (sms * csm * 2.0 * sm_clock_ghz) / 1000.0
    except (ValueError, TypeError, ZeroDivisionError):
        tflops = None

    src = "nvidia-smi"
    if bw is None or tflops is None:
        calib = infer_torch_calibration(device=f"cuda:{idx}")
        if calib:
            if bw is None:
                bw = calib.peak_mem_bw_gbs
            if tflops is None:
                tflops = calib.peak_tflops
            src = "nvidia-smi+torch-calib"

    if bw is None or tflops is None:
        return None

    dtype_peaks = _lookup_dtype_peaks(name)
    return Peaks(float(tflops), float(bw), src, device_id, dtype_peaks=dtype_peaks)

# ---------------- MPS / Metal (multi device awareness) ----------------

def _apple_chip_name() -> str | None:
    if platform.system() != "Darwin":
        return None
    s = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return s or None

def _metal_devices() -> list[dict[str, str]]:
    if platform.system() != "Darwin":
        return []
    out = _run(["system_profiler", "SPDisplaysDataType", "-json"])
    if out:
        try:
            j = json.loads(out)
            items = j.get("SPDisplaysDataType", []) or []
            devs = []
            for it in items:
                name = it.get("sppci_model", it.get("_name", "GPU"))
                vendor = it.get("spdisplays_vendor", "") or it.get("spdisplays_vendor-id", "")
                dtype = "integrated" if it.get("spdisplays_integrated", False) else "discrete"
                if it.get("spdisplays_external", False):
                    dtype = "external"
                devs.append({"name": str(name), "vendor": str(vendor), "type": str(dtype)})
            return devs
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    out2 = _run(["system_profiler", "SPDisplaysDataType"])
    if not out2:
        return []
    devs = []
    for line in out2.splitlines():
        if "Chipset Model:" in line:
            name = line.split("Chipset Model:", 1)[1].strip()
            devs.append({"name": name, "vendor": "", "type": ""})
    return devs

def _select_mps_device(devs: list[dict[str,str]]) -> dict[str,str] | None:
    if not devs:
        return None
    m = os.environ.get("PERFLAB_MPS_DEVICE_MATCH", "").strip()
    if m:
        for d in devs:
            if m.lower() in d.get("name","").lower():
                return d
    idx_s = os.environ.get("PERFLAB_MPS_DEVICE_INDEX", "").strip()
    if idx_s:
        try:
            i = int(idx_s)
            if 0 <= i < len(devs):
                return devs[i]
        except (ValueError, IndexError):
            pass
    return devs[0]

_MPS_HEURISTICS = {
    "M1":  (2.6, 68.0),
    "M1 Pro": (5.2, 200.0),
    "M1 Max": (10.4, 400.0),
    "M2":  (3.6, 100.0),
    "M2 Pro": (6.8, 200.0),
    "M2 Max": (13.6, 400.0),
    "M3": (4.1, 100.0),
    "M3 Pro": (6.2, 150.0),
    "M3 Max": (14.2, 300.0),
}

def infer_mps_peaks() -> Peaks | None:
    chip = _apple_chip_name() or "Apple Silicon"
    devs = _metal_devices()
    chosen = _select_mps_device(devs)
    dev_note = ""
    if devs and len(devs) > 1:
        names = ", ".join(d.get("name","GPU") for d in devs[:6])
        dev_note = f" (multiple GPUs detected: {names})"
    if chosen is not None:
        dev_note = f" (selected: {chosen.get('name','GPU')})" + (dev_note if len(devs) > 1 else "")

    best = None
    for k, (tf, bw) in _MPS_HEURISTICS.items():
        if k in chip:
            best = (k, tf, bw)
            break
    if best is not None:
        k, tf, bw = best
        return Peaks(float(tf), float(bw), "mps-heuristic", f"{chip} ({k}){dev_note}")

    calib = infer_torch_calibration(device="mps")
    if calib:
        calib.device = calib.device + dev_note
        return Peaks(calib.peak_tflops, calib.peak_mem_bw_gbs, "torch-calib", calib.device)

    return None

# ---------------- CPU ----------------

def _estimate_cpu_peaks() -> Peaks | None:
    """Estimate CPU peaks from hardware specs (clock × cores × SIMD width)."""
    import multiprocessing

    sys_name = platform.system()
    cores: int | None = None
    clock_ghz: float | None = None
    simd_width: int = 1  # FP32 ops per SIMD instruction
    cpu_name: str = "CPU"

    if sys_name == "Darwin":
        # macOS: use sysctl
        cpu_name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple CPU"

        # Get performance core count (prefer P-cores for peak)
        perf_cores = _run(["sysctl", "-n", "hw.perflevel0.logicalcpu"])
        if perf_cores:
            try:
                cores = int(perf_cores)
            except ValueError:
                pass
        if cores is None:
            phys = _run(["sysctl", "-n", "hw.physicalcpu"])
            if phys:
                try:
                    cores = int(phys)
                except ValueError:
                    pass

        # Clock speed
        freq = _run(["sysctl", "-n", "hw.cpufrequency_max"])
        if freq:
            try:
                clock_ghz = float(freq) / 1e9
            except ValueError:
                pass

        # Apple Silicon NEON: 128-bit = 4 FP32 ops, 2 FMA units = 8 FLOP/cycle
        if "Apple" in cpu_name:
            simd_width = 8  # 2 NEON FMA units × 4 FP32
            # Apple Silicon doesn't report frequency via sysctl reliably
            # Use known frequencies for common chips
            for chip, freq_ghz in [("M1", 3.2), ("M2", 3.5), ("M3", 4.05), ("M4", 4.4)]:
                if chip in cpu_name:
                    clock_ghz = freq_ghz
                    break
            if clock_ghz is None:
                clock_ghz = 3.2  # conservative default for Apple Silicon

            # Memory bandwidth from known specs
            bw_gbs: float | None = None
            for chip, bw in [
                ("M1 Ultra", 800.0), ("M1 Max", 400.0), ("M1 Pro", 200.0), ("M1", 68.0),
                ("M2 Ultra", 800.0), ("M2 Max", 400.0), ("M2 Pro", 200.0), ("M2", 100.0),
                ("M3 Ultra", 800.0), ("M3 Max", 400.0), ("M3 Pro", 150.0), ("M3", 100.0),
                ("M4 Max", 546.0), ("M4 Pro", 273.0), ("M4", 120.0),
            ]:
                if chip in cpu_name:
                    bw_gbs = bw
                    break

            if cores and clock_ghz:
                peak_tflops = (cores * simd_width * clock_ghz) / 1000.0
                if bw_gbs is None:
                    bw_gbs = 68.0  # conservative
                return Peaks(peak_tflops, bw_gbs, "cpu-spec", cpu_name)

    elif sys_name == "Linux":
        # Linux: parse /proc/cpuinfo and lscpu
        cpu_name = _run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]) or "CPU"
        cpu_name = cpu_name.strip()

        # Core count
        try:
            cores = multiprocessing.cpu_count()
        except (OSError, NotImplementedError):
            pass

        # Max clock
        freq_str = _run(["bash", "-c", "lscpu | grep 'CPU max MHz' | awk '{print $NF}'"])
        if freq_str:
            try:
                clock_ghz = float(freq_str) / 1000.0
            except ValueError:
                pass
        if clock_ghz is None:
            freq_str = _run(["bash", "-c", "grep -m1 'cpu MHz' /proc/cpuinfo | awk '{print $NF}'"])
            if freq_str:
                try:
                    clock_ghz = float(freq_str) / 1000.0
                except ValueError:
                    pass

        # SIMD width detection from flags
        flags = _run(["bash", "-c", "grep -m1 'flags' /proc/cpuinfo | tr ' ' '\\n'"]) or ""
        if "avx512f" in flags:
            simd_width = 32  # 512-bit / 32-bit × 2 (FMA) = 32 FLOP/cycle
        elif "avx2" in flags or "avx" in flags or "fma" in flags:
            simd_width = 16  # 256-bit / 32-bit × 2 (FMA) = 16 FLOP/cycle
        elif "sse" in flags:
            simd_width = 8   # 128-bit / 32-bit × 2 = 8 FLOP/cycle
        else:
            simd_width = 2   # scalar FMA

    if cores and clock_ghz and cores > 0 and clock_ghz > 0:
        peak_tflops = (cores * simd_width * clock_ghz) / 1000.0
        # Estimate bandwidth: default heuristic for DDR
        # Try to get from dmidecode or lshw, fall back to conservative estimate
        bw_gbs = None
        if sys_name == "Linux":
            # Try lsmem / dmidecode for memory bandwidth
            mem_info = _run(["bash", "-c", "sudo dmidecode -t memory 2>/dev/null | grep -i 'speed:' | head -1 | awk '{print $2}'"])
            channels_str = _run(["bash", "-c", "sudo dmidecode -t memory 2>/dev/null | grep -ic 'size:.*[0-9]'"])
            if mem_info and channels_str:
                try:
                    mem_mhz = float(mem_info)
                    channels = int(channels_str)
                    bw_gbs = (mem_mhz * 8 * channels) / 1000.0  # DDR = 8 bytes per transfer
                except (ValueError, ZeroDivisionError):
                    pass

        if bw_gbs is None:
            # Conservative fallback: ~50 GB/s for a modern desktop
            bw_gbs = 50.0

        return Peaks(peak_tflops, bw_gbs, "cpu-spec", cpu_name)

    return None


def infer_cpu_peaks() -> Peaks | None:
    """Infer CPU peaks from hardware specs first, then torch calibration fallback."""
    spec = _estimate_cpu_peaks()
    if spec:
        return spec
    calib = infer_torch_calibration(device="cpu")
    if calib:
        return Peaks(calib.peak_tflops, calib.peak_mem_bw_gbs, "torch-calib", calib.device)
    return None

# ---------------- Torch calibration (fallback) ----------------

def infer_torch_calibration(device: str) -> Peaks | None:
    try:
        import torch
    except ImportError:
        return None

    dev = torch.device(device)
    desc = device
    try:
        if dev.type == "cuda":
            desc = torch.cuda.get_device_name(dev.index or 0)
        elif dev.type == "mps":
            desc = _apple_chip_name() or "Apple MPS"
        else:
            desc = platform.processor() or platform.machine() or "CPU"
    except (RuntimeError, ValueError):
        pass

    key = f"torchcalib::{device}::{desc}"
    cache = _load_cache()
    if key in cache:
        c = cache[key]
        return Peaks(float(c["peak_tflops"]), float(c["peak_mem_bw_gbs"]), c.get("source","torch-calib-cache"), c.get("device", desc))

    # Matmul calibration
    try:
        import torch
        torch.manual_seed(0)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        M = N = K = 2048 if dev.type != "cpu" else 1024
        a = torch.randn((M, K), device=dev, dtype=torch.float32)
        b = torch.randn((K, N), device=dev, dtype=torch.float32)
        for _ in range(3):
            _ = a @ b
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        iters = 10 if dev.type != "cpu" else 5
        for _ in range(iters):
            _ = a @ b
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t1 = time.perf_counter()
        seconds = (t1 - t0) / float(iters)
        flops = 2.0 * M * N * K
        tflops = (flops / seconds) / 1e12 if seconds > 0 else 0.0
    except (RuntimeError, ValueError, TypeError):
        tflops = None

    # Bandwidth via copy
    try:
        import torch
        n = 256 * 1024 * 1024 // 4  # 256MB float32
        x = torch.empty((n,), device=dev, dtype=torch.float32)
        y = torch.empty((n,), device=dev, dtype=torch.float32)
        for _ in range(3):
            y.copy_(x)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        iters = 20 if dev.type != "cpu" else 10
        for _ in range(iters):
            y.copy_(x)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t1 = time.perf_counter()
        seconds = (t1 - t0) / float(iters)
        bytes_moved = x.numel() * x.element_size()
        gbs = (bytes_moved / seconds) / 1e9 if seconds > 0 else 0.0
    except (RuntimeError, ValueError, TypeError):
        gbs = None

    if not tflops or not gbs or tflops <= 0 or gbs <= 0:
        return None

    peaks = Peaks(float(tflops), float(gbs), "torch-calib", desc)
    cache[key] = {"peak_tflops": peaks.peak_tflops, "peak_mem_bw_gbs": peaks.peak_mem_bw_gbs, "source": peaks.source, "device": peaks.device}
    _save_cache(cache)
    return peaks


def list_cuda_gpus() -> list[dict[str, str]]:
    """Best-effort list of CUDA GPUs via nvidia-smi."""
    fields = ["index", "name", "uuid", "compute_cap", "memory.total"]
    gpus = _nvidia_smi_query(fields)
    if not gpus:
        out = _run(["nvidia-smi", "-L"])
        if not out:
            return []
        res = []
        for line in out.splitlines():
            line = line.strip()
            if line:
                res.append({"raw": line})
        return res
    res = []
    for g in gpus:
        res.append({
            "index": str(g.get("index","")),
            "name": str(g.get("name","")),
            "uuid": str(g.get("uuid","")),
            "compute_cap": str(g.get("compute_cap","")),
            "memory_total_mib": str(g.get("memory.total","")),
        })
    return res

def list_metal_gpus() -> list[dict[str, str]]:
    """Best-effort list of macOS GPU devices (Metal/Displays)."""
    return _metal_devices()

def selection_hints() -> dict[str, str]:
    return {
        "cuda": "Use CUDA_VISIBLE_DEVICES or `perflab peaks --cuda-index N` to choose which GPU peaks are inferred for.",
        "mps": "Use PERFLAB_MPS_DEVICE_INDEX or PERFLAB_MPS_DEVICE_MATCH to influence which Metal device is referenced in reporting.",
        "cache": "Set PERFLAB_PEAKS_NO_CACHE=1 to bypass caching; set PERFLAB_PEAKS_CACHE to override cache path.",
    }


# ---------------- TPU ----------------

# Known per-chip specs for Google TPU generations.
# BF16 TFLOPS and HBM bandwidth (GB/s) per chip.
_KNOWN_TPU_SPECS: dict[str, tuple[float, float, int]] = {
    # (peak_bf16_tflops, hbm_bw_gbs, hbm_gb)
    "TPU v4":  (275.0,  1200.0, 32),
    "TPU v5e": (197.0,  819.0,  16),
    "TPU v5p": (459.0,  2765.0, 95),
    "TPU v6e": (918.0,  1600.0, 32),
}


def infer_tpu_peaks() -> Peaks | None:
    """Detect TPU via jax.devices() and return known peak specs."""
    try:
        import jax
        devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == "tpu"]
        if not tpu_devices:
            return None

        chip_kind = str(tpu_devices[0].device_kind)  # e.g. "TPU v4"
        n_chips = len(tpu_devices)

        # Match against known specs
        for name, (bf16_tflops, hbm_bw, hbm_gb) in _KNOWN_TPU_SPECS.items():
            if name.lower() in chip_kind.lower() or chip_kind.lower() in name.lower():
                return Peaks(
                    peak_tflops=bf16_tflops,
                    peak_mem_bw_gbs=hbm_bw,
                    source="tpu-spec",
                    device=f"{chip_kind} ({n_chips} chip{'s' if n_chips > 1 else ''})",
                    dtype_peaks={
                        "peak_tflops_bf16": bf16_tflops,
                        "peak_tflops_fp32": bf16_tflops / 2.0,  # MXU runs bf16 natively
                    },
                )

        # Unknown TPU generation — return None rather than guess
        return None
    except ImportError:
        return None
    except Exception:
        logger.warning("TPU peak detection failed", exc_info=True)
        return None


# ---------------- Public API ----------------

def infer_peaks(target: str, preferred_cuda_index: int | None = None) -> Peaks | None:
    t = (target or "auto").lower()
    if t in ("auto", "tpu"):
        p = infer_tpu_peaks()
        if p:
            return p
        if t == "tpu":
            return None
    if t in ("auto", "cuda"):
        p = infer_cuda_peaks(preferred_index=preferred_cuda_index)
        if p:
            return p
        if t == "cuda":
            return None
    if t in ("auto", "mps"):
        p = infer_mps_peaks()
        if p:
            return p
        if t == "mps":
            return None
    if t in ("auto", "cpu"):
        return infer_cpu_peaks()
    return None


def resolve_roofline(task) -> dict | None:
    """Resolve roofline peaks for a task: use explicit config if set, else auto-detect.

    Shared implementation used by both orchestrator and agent modules.
    """
    if task.roofline:
        return {
            "peak_tflops": task.roofline.peak_tflops,
            "peak_mem_bw_gbs": task.roofline.peak_mem_bw_gbs,
        }
    try:
        peaks = infer_peaks(task.target_hardware or "auto")
        if peaks:
            return {
                "peak_tflops": peaks.peak_tflops,
                "peak_mem_bw_gbs": peaks.peak_mem_bw_gbs,
                "source": peaks.source,
                "device": peaks.device,
            }
    except Exception:
        logger.warning("Roofline auto-detect failed", exc_info=True)
    return None
