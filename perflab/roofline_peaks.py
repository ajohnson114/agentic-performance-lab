from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
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


@dataclass(frozen=True)
class GpuSpec:
    """One consolidated spec-sheet entry per known GPU.

    Adding a GPU means adding a single entry to _KNOWN_GPU_SPECS below.

    dtype_peaks: theoretical per-dtype peaks (TFLOPS) for tensor core /
        CUDA core operations.
    mem_bw_gbs: spec-sheet HBM/GDDR bandwidth (GB/s). Kept as a separate
        field rather than folded into dtype_peaks, since downstream
        consumers of dtype_peaks treat it as TFLOPS-only.
    l2_bw_gbs: approximate peak aggregate L2 read bandwidth across all SMs
        (GB/s). L2 is the practical upper bound for data-reuse-heavy
        kernels (tiled matmuls).
    smem_per_sm_kb / num_sms / max_regs_per_sm: SM resource limits for
        hardware context in LLM prompts (configurable shared memory per SM,
        number of streaming multiprocessors, registers per SM;
        max_regs_per_thread is always 255 for CUDA -- a hardware limit).
    """

    dtype_peaks: dict[str, float]
    mem_bw_gbs: float | None = None
    l2_bw_gbs: float | None = None
    smem_per_sm_kb: int | None = None
    num_sms: int | None = None
    max_regs_per_sm: int | None = None


# Known specs for common GPUs (per-dtype peaks, memory/L2 bandwidth, SM
# resource limits -- see GpuSpec).
# Keys are matched two ways against a real `nvidia-smi --query-gpu=name`
# string: an exact match against a full marketing name (tier 1, source=
# "table"), or -- logged as an unverified "assumed" match, never claiming
# source="table" -- a substring match against a short legacy key. The short
# keys stay ambiguous on purpose: e.g. "H100" alone can't tell PCIe/SXM/NVL
# apart even though they differ ~1.6x on bandwidth, so callers that need a
# real nvidia-smi name resolved should prefer an exact hit or fall through to
# the computed/measured tiers instead of trusting a substring guess.
# NOTE: insertion order matters -- the substring lookups iterate in order, so
# the full marketing names must stay ahead of the short legacy keys.
_KNOWN_GPU_SPECS: dict[str, GpuSpec] = {
    # --- Full nvidia-smi marketing names (tier 1 candidates) ---
    "NVIDIA A100-SXM4-40GB": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 19.5,
            "peak_tflops_tf32": 156.0,
            "peak_tflops_fp16": 312.0,
            "peak_tflops_bf16": 312.0,
        },
        mem_bw_gbs=1555.0,
    ),
    "NVIDIA A100-SXM4-80GB": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 19.5,
            "peak_tflops_tf32": 156.0,
            "peak_tflops_fp16": 312.0,
            "peak_tflops_bf16": 312.0,
        },
        mem_bw_gbs=2039.0,
    ),
    "NVIDIA A100-PCIE-40GB": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 19.5,
            "peak_tflops_tf32": 156.0,
            "peak_tflops_fp16": 312.0,
            "peak_tflops_bf16": 312.0,
        },
        mem_bw_gbs=1555.0,
    ),
    "NVIDIA A100 80GB PCIe": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 19.5,
            "peak_tflops_tf32": 156.0,
            "peak_tflops_fp16": 312.0,
            "peak_tflops_bf16": 312.0,
        },
        mem_bw_gbs=1935.0,
    ),
    "NVIDIA H100 80GB HBM3": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 67.0,
            "peak_tflops_tf32": 989.0,
            "peak_tflops_fp16": 1979.0,
            "peak_tflops_bf16": 1979.0,
        },
        mem_bw_gbs=3352.0,
    ),
    "NVIDIA H100 PCIe": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 51.0,
            "peak_tflops_tf32": 756.0,
            "peak_tflops_fp16": 1513.0,
            "peak_tflops_bf16": 1513.0,
        },
        mem_bw_gbs=2039.0,
    ),
    "NVIDIA H100 NVL": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 67.0,
            "peak_tflops_tf32": 989.0,
            "peak_tflops_fp16": 1979.0,
            "peak_tflops_bf16": 1979.0,
        },
        mem_bw_gbs=3900.0,
    ),
    "NVIDIA GeForce RTX 4090": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 82.6,
            "peak_tflops_tf32": 82.6,
            "peak_tflops_fp16": 165.2,
            "peak_tflops_bf16": 165.2,
        },
        mem_bw_gbs=1008.0,
    ),
    "NVIDIA GeForce RTX 4080": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 48.7,
            "peak_tflops_tf32": 48.7,
            "peak_tflops_fp16": 97.5,
            "peak_tflops_bf16": 97.5,
        },
        mem_bw_gbs=716.8,
    ),
    "NVIDIA GeForce RTX 3090": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 35.6,
            "peak_tflops_tf32": 71.0,
            "peak_tflops_fp16": 142.0,
            "peak_tflops_bf16": 142.0,
        },
        mem_bw_gbs=936.2,
    ),
    "Tesla V100-SXM2-16GB": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 15.7,
            "peak_tflops_fp16": 125.0,
        },
        mem_bw_gbs=900.0,
    ),
    "Tesla V100-PCIE-16GB": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 14.0,
            "peak_tflops_fp16": 112.0,
        },
        mem_bw_gbs=900.0,
    ),
    # --- Short legacy keys: substring/"assumed" match only, see note above ---
    "A100": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 19.5,
            "peak_tflops_tf32": 156.0,
            "peak_tflops_fp16": 312.0,
            "peak_tflops_bf16": 312.0,
        },
        mem_bw_gbs=1555.0,
        l2_bw_gbs=6000.0,  # 40 MB L2, ~6 TB/s aggregate read
        smem_per_sm_kb=164,
        num_sms=108,
        max_regs_per_sm=65536,
    ),
    "A100-SXM": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 19.5,
            "peak_tflops_tf32": 156.0,
            "peak_tflops_fp16": 312.0,
            "peak_tflops_bf16": 312.0,
        },
        mem_bw_gbs=2039.0,
        l2_bw_gbs=6000.0,
        smem_per_sm_kb=164,
        num_sms=108,
        max_regs_per_sm=65536,
    ),
    "H100": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 67.0,
            "peak_tflops_tf32": 989.0,
            "peak_tflops_fp16": 1979.0,
            "peak_tflops_bf16": 1979.0,
        },
        mem_bw_gbs=3352.0,
        l2_bw_gbs=12000.0,  # 50 MB L2, ~12 TB/s aggregate read
        smem_per_sm_kb=228,
        num_sms=132,
        max_regs_per_sm=65536,
    ),
    "H100-SXM": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 67.0,
            "peak_tflops_tf32": 989.0,
            "peak_tflops_fp16": 1979.0,
            "peak_tflops_bf16": 1979.0,
        },
        mem_bw_gbs=3352.0,
        l2_bw_gbs=12000.0,
        smem_per_sm_kb=228,
        num_sms=132,
        max_regs_per_sm=65536,
    ),
    "RTX 4090": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 82.6,
            "peak_tflops_tf32": 82.6,
            "peak_tflops_fp16": 165.2,
            "peak_tflops_bf16": 165.2,
        },
        mem_bw_gbs=1008.0,
        l2_bw_gbs=3200.0,  # 72 MB L2, ~3.2 TB/s
        smem_per_sm_kb=100,
        num_sms=128,
        max_regs_per_sm=65536,
    ),
    "RTX 4080": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 48.7,
            "peak_tflops_tf32": 48.7,
            "peak_tflops_fp16": 97.5,
            "peak_tflops_bf16": 97.5,
        },
        mem_bw_gbs=716.8,
        l2_bw_gbs=2400.0,  # 64 MB L2, ~2.4 TB/s
        smem_per_sm_kb=100,
        num_sms=76,
        max_regs_per_sm=65536,
    ),
    "RTX 3090": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 35.6,
            "peak_tflops_tf32": 71.0,
            "peak_tflops_fp16": 142.0,
            "peak_tflops_bf16": 142.0,
        },
        mem_bw_gbs=936.2,
        l2_bw_gbs=2400.0,  # 6 MB L2, ~2.4 TB/s
        smem_per_sm_kb=100,
        num_sms=82,
        max_regs_per_sm=65536,
    ),
    "V100": GpuSpec(
        dtype_peaks={
            "peak_tflops_fp32": 15.7,
            "peak_tflops_fp16": 125.0,
        },
        mem_bw_gbs=900.0,
        l2_bw_gbs=3100.0,  # 6 MB L2, ~3.1 TB/s
        smem_per_sm_kb=96,
        num_sms=80,
        max_regs_per_sm=65536,
    ),
}


def _lookup_l2_bw(device_name: str) -> float | None:
    """Try to match a device name against known GPU L2 bandwidth specs."""
    name_upper = (device_name or "").upper()
    for key, spec in _KNOWN_GPU_SPECS.items():
        if spec.l2_bw_gbs is not None and key.upper() in name_upper:
            return spec.l2_bw_gbs
    return None


def _lookup_sm_specs(device_name: str) -> dict[str, int | float] | None:
    """Look up SM resource limits for a GPU."""
    name_upper = (device_name or "").upper()
    for key, spec in _KNOWN_GPU_SPECS.items():
        if spec.smem_per_sm_kb is None or spec.num_sms is None or spec.max_regs_per_sm is None:
            continue
        if key.upper() in name_upper:
            return {
                "max_smem_per_sm_kb": spec.smem_per_sm_kb,
                "num_sms": spec.num_sms,
                "max_regs_per_sm": spec.max_regs_per_sm,
            }
    return None


def _lookup_dtype_peaks_exact(device_name: str) -> tuple[str, dict[str, float]] | None:
    """Exact (case/whitespace-insensitive) match on a full nvidia-smi name -- tier 1."""
    name_norm = (device_name or "").strip().upper()
    if not name_norm:
        return None
    for key, spec in _KNOWN_GPU_SPECS.items():
        if spec.dtype_peaks and key.strip().upper() == name_norm:
            return key, dict(spec.dtype_peaks)
    return None

def _lookup_dtype_peaks_prefix(device_name: str) -> tuple[str, dict[str, float]] | None:
    """Substring match against the legacy short keys -- an unverified "assumed" match."""
    name_upper = (device_name or "").upper()
    for key, spec in _KNOWN_GPU_SPECS.items():
        if spec.dtype_peaks and key.upper() in name_upper:
            return key, dict(spec.dtype_peaks)
    return None

def _lookup_dtype_peaks(device_name: str) -> dict[str, float] | None:
    """Try to match a device name against known GPU dtype peak tables.

    Kept substring-based for backward compatibility: callers such as
    agent.py/pipeline.py/prompt.py pass free-text `task.target_hardware`
    (e.g. "A100"), not a real nvidia-smi name, so a prefix match is the
    intended behavior there.
    """
    match = _lookup_dtype_peaks_prefix(device_name)
    return match[1] if match else None

def _representative_tflops(dtype_peaks: dict[str, float]) -> float | None:
    """Pick a single headline TFLOPS number out of a per-dtype peaks dict."""
    for key in ("peak_tflops_bf16", "peak_tflops_fp16", "peak_tflops_tf32", "peak_tflops_fp32"):
        if key in dtype_peaks:
            return dtype_peaks[key]
    return None

_CACHE_PATH = Path(os.environ.get("PERFLAB_PEAKS_CACHE", str(Path.home() / ".cache" / "perflab" / "peaks.json")))

def cache_path() -> Path:
    return _CACHE_PATH

def _slugify_gpu_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", (name or "").strip()).strip("-").lower()
    return slug or "unknown-gpu"

def gpu_measured_cache_path(gpu_name: str) -> Path:
    """Per-GPU-name sibling of cache_path(), so a measured probe runs once per machine per card."""
    return _CACHE_PATH.parent / f"peaks-{_slugify_gpu_name(gpu_name)}.json"

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


def _physical_cpu_count() -> int | None:
    """Physical core count, excluding SMT/hyperthread siblings.

    Peak FLOPs scale with physical FMA pipelines; using the logical CPU
    count overestimates the compute roof ~2x on hyperthreaded machines.
    Returns None if no method succeeds (caller falls back to logical count).
    """
    try:
        import psutil  # optional dependency
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass

    sys_name = platform.system()
    if sys_name == "Darwin":
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
        if out:
            try:
                return int(out)
            except ValueError:
                pass
    elif sys_name == "Linux":
        # lscpu: unique (core, socket) pairs = physical cores
        out = _run(["bash", "-c",
                    "lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l"])
        if out:
            try:
                n = int(out)
                if n > 0:
                    return n
            except ValueError:
                pass
        # /sys topology: each physical core has one unique thread_siblings_list
        out = _run(["bash", "-c",
                    "cat /sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list"
                    " 2>/dev/null | sort -u | wc -l"])
        if out:
            try:
                n = int(out)
                if n > 0:
                    return n
            except ValueError:
                pass
    return None

# ---------------- CUDA (multi-GPU) ----------------

def _nvidia_smi_query(fields: list[str]) -> list[dict[str,str]] | None:
    q = ",".join(fields)
    out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    res = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        d = {}
        # fields/parts may differ in length for a malformed CSV row; zip() strict=
        # needs Python 3.10+ and this codebase still runs on 3.9.
        for k, v in zip(fields, parts):  # noqa: B905
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

def _computed_gpu_bandwidth(g: dict[str, str]) -> float | None:
    """Tier 2: theoretical peak BW from bus_width/8 x clock x 2 (DDR)."""
    try:
        bus = float(g.get("memory.bus_width", "0"))
        mem_mhz = float(g.get("memory.clock", "0"))
        if bus > 0 and mem_mhz > 0:
            return (bus / 8.0) * (mem_mhz * 1e6) * 2.0 / 1e9
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return None

def _computed_gpu_tflops(g: dict[str, str], cc: str) -> float | None:
    """Tier 2: CUDA-core count x clock derived compute peak."""
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
            return (sms * csm * 2.0 * sm_clock_ghz) / 1000.0
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return None

_GPU_PROBE_SECONDS = 5.0

def _gpu_matmul_tflops_probe(dev: Any) -> float | None:
    """Tier 3: fp16 matmul TFLOPS probe, run for ~_GPU_PROBE_SECONDS."""
    import torch
    try:
        torch.manual_seed(0)
        m = n = k = 4096
        a = torch.randn((m, k), device=dev, dtype=torch.float16)
        b = torch.randn((k, n), device=dev, dtype=torch.float16)
        for _ in range(3):
            _ = a @ b
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        iters = 0
        while time.perf_counter() - t0 < _GPU_PROBE_SECONDS:
            _ = a @ b
            iters += 1
        torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - t0
        if iters == 0 or elapsed <= 0:
            return None
        flops = 2.0 * m * n * k * iters
        return (flops / elapsed) / 1e12
    except (RuntimeError, ValueError, TypeError):
        return None

def _gpu_bandwidth_copy_probe(dev: Any) -> float | None:
    """Tier 3: device-to-device copy_ bandwidth sweep, run for ~_GPU_PROBE_SECONDS."""
    import torch
    try:
        n = 256 * 1024 * 1024 // 4  # 256MB float32
        x = torch.empty((n,), device=dev, dtype=torch.float32)
        y = torch.empty((n,), device=dev, dtype=torch.float32)
        for _ in range(3):
            y.copy_(x)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        iters = 0
        while time.perf_counter() - t0 < _GPU_PROBE_SECONDS:
            y.copy_(x)
            iters += 1
        torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - t0
        if iters == 0 or elapsed <= 0:
            return None
        bytes_moved = x.numel() * x.element_size() * iters
        return (bytes_moved / elapsed) / 1e9
    except (RuntimeError, ValueError, TypeError):
        return None

def _measured_cuda_peaks(name: str, idx: int) -> Peaks | None:
    """Tier 3: measured torch-calibration fallback, cached per-GPU-name so it runs once per machine."""
    try:
        import torch
    except ImportError:
        return None

    cache_file = gpu_measured_cache_path(name)
    if _use_cache():
        try:
            if cache_file.exists():
                c = json.loads(cache_file.read_text(encoding="utf-8"))
                return Peaks(float(c["peak_tflops"]), float(c["peak_mem_bw_gbs"]), "measured", name)
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            logger.warning("Failed to load GPU measured-peaks cache %s", cache_file, exc_info=True)

    dev = torch.device(f"cuda:{idx}")
    tflops = _gpu_matmul_tflops_probe(dev)
    gbs = _gpu_bandwidth_copy_probe(dev)
    if not tflops or not gbs or tflops <= 0 or gbs <= 0:
        return None

    peaks = Peaks(float(tflops), float(gbs), "measured", name)
    if _use_cache():
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"peak_tflops": peaks.peak_tflops, "peak_mem_bw_gbs": peaks.peak_mem_bw_gbs}, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to save GPU measured-peaks cache %s", cache_file, exc_info=True)
    return peaks

def infer_cuda_peaks(preferred_index: int | None = None) -> Peaks | None:
    """Three-tier GPU peak resolution: exact table match, then computed, then measured.

    Peaks.source records which tier produced the number: "table" (tier 1,
    exact nvidia-smi name match), "computed" (tier 2, bus_width/clock derived),
    or "measured" (tier 3, torch calibration probe, cached per GPU name).
    """
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

    dtype_peaks: dict[str, float] | None = None
    exact = _lookup_dtype_peaks_exact(name)
    if exact is not None:
        matched_key, dtype_peaks = exact
        table_tflops = _representative_tflops(dtype_peaks)
        table_bw = _KNOWN_GPU_SPECS[matched_key].mem_bw_gbs
        if table_tflops is not None and table_bw is not None:
            return Peaks(float(table_tflops), float(table_bw), "table", device_id, dtype_peaks=dtype_peaks)
    else:
        prefix = _lookup_dtype_peaks_prefix(name)
        if prefix is not None:
            matched_key, dtype_peaks = prefix
            logger.warning(
                "GPU %r has no exact peaks-table entry; %r matched by substring only -- "
                "treating as unverified and preferring computed/measured peaks over the table",
                name, matched_key,
            )

    bw = _computed_gpu_bandwidth(g)
    tflops = _computed_gpu_tflops(g, cc)
    if bw is not None and tflops is not None:
        return Peaks(float(tflops), float(bw), "computed", device_id, dtype_peaks=dtype_peaks)

    calib = _measured_cuda_peaks(name, idx)
    if calib is not None:
        final_tflops = tflops if tflops is not None else calib.peak_tflops
        final_bw = bw if bw is not None else calib.peak_mem_bw_gbs
        return Peaks(float(final_tflops), float(final_bw), "measured", device_id, dtype_peaks=dtype_peaks)

    return None

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
    # M4 family per Apple's published specs (fp32 TFLOPS, memory bandwidth GB/s)
    "M4": (4.26, 120.0),
    "M4 Pro": (9.2, 273.0),
    "M4 Max": (18.4, 546.0),
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
#
# Two ways to get a CPU compute roof, in preference order:
#
#   1. MEASURE it (`_measured_cpu_peaks`) with a short calibrated fp32 GEMM.
#      Whatever the machine actually sustains is what gets recorded, which
#      sidesteps turbo behavior, AVX-512 licensing and FMA-port counts in one
#      move. Source: "cpu-measured" / "cpu-measured-flops".
#   2. MODEL it (`_estimate_cpu_peaks`) from cores x clock x FLOP/cycle, used
#      whenever the measurement is unavailable or implausible. Source:
#      "cpu-spec".
#
# The roof matters beyond reporting: peak_tflops / peak_mem_bw_gbs is the
# roofline knee, and `_classify_bound()` in perflab/optimizers/prompt.py uses
# that knee to decide whether the LLM is told to chase memory-bound or
# compute-bound optimizations. A 2x error in the roof flips the strategy for
# any kernel near the knee.

# ===== The FLOP/cycle model =====
#
# Every FLOP/cycle number below follows one rule:
#
#     fp32 FLOP/cycle/core = fp32_lanes * 2 * pipes
#
# where the *2 is one multiply plus one add, and `pipes` is the number of
# full-width FP pipelines that can retire such a pair every cycle:
#
#   * FMA hardware: pipes == the FMA unit count. Two on nearly every
#     mainstream core since Haswell / Zen 2; one on Zen 1, Atom/E-cores, and
#     most 512-bit client parts.
#   * Pre-FMA hardware (SSE, AVX1): pipes == 1. The separate FP-mul and
#     FP-add ports co-issue to retire one mul+add pair per cycle. There is no
#     fused op and only a perfectly balanced mul/add mix reaches even that
#     rate, so crediting two pipes here would be wrong.
#
# The previous model used "vector_bits / 32 * 2" unconditionally: that
# hard-codes exactly one FMA unit for every ISA (2x pessimistic on the
# dual-FMA parts that dominate the installed base) while simultaneously
# granting pre-FMA SSE/AVX an FMA they do not have.


@dataclass(frozen=True)
class _CpuIsa:
    """One ISA class of the CPU peak-FLOPS model."""

    name: str
    fp32_lanes: int
    pipes: int
    note: str

    @property
    def flops_per_cycle(self) -> int:
        return self.fp32_lanes * 2 * self.pipes


# Apple P-cores (M1 onward) have four 128-bit NEON FMA pipes:
# 4 lanes * 2 * 4 = 32 fp32 FLOP/cycle/core. Cross-check: M1 Max, 8 P-cores at
# 3.2 GHz -> 819 GFLOP/s, matching the commonly cited ~800 GFLOP/s NEON peak.
# (The old model assumed two pipes and so understated Apple silicon 2x.)
_APPLE_ISA = _CpuIsa("neon-apple", 4, 4, "Apple P-core: 4x 128-bit NEON FMA pipes")

# Generic AArch64: Neoverse N1/V1 and Cortex-A7x have two 128-bit FP/ASIMD
# pipes capable of FMA -> 4 * 2 * 2 = 16 fp32 FLOP/cycle/core.
_NEON_ISA = _CpuIsa("neon", 4, 2, "128-bit NEON, 2 FMA pipes (Neoverse/Cortex-A7x class)")

_SCALAR_ISA = _CpuIsa("scalar", 1, 1, "no usable SIMD flags detected")

# Single-core max turbo is never sustained across all cores. All-core turbo
# typically lands at 70-85% of the single-core ceiling on server parts and
# nearer 90% on desktop parts; 0.75 is a deliberately conservative middle used
# only when no true base clock can be read.
_ALL_CORE_TURBO_DERATE = 0.75

# Intel parts that carry two 512-bit FMA units. Everything else with avx512f
# is assumed to have one (the conservative direction), which is correct for
# Xeon Silver/Bronze, Xeon Gold 51xx/52xx and all 512-bit client parts, and
# happens to give the right answer for Zen 4 as well (its AVX-512 is
# double-pumped over two 256-bit FMAs, i.e. the same 32 fp32 FLOP/cycle as one
# 512-bit unit). It understates Zen 5 desktop parts with a full 512-bit
# datapath; there is no architectural CPUID bit for FMA unit count, so brand
# matching is the only cheap signal and it errs low on purpose.
_AVX512_DUAL_FMA_PATTERNS = (
    # Xeon Platinum 81xx/82xx (SKX/CLX) through 84xx/85xx (SPR/EMR).
    r"platinum\s*8\d{3}",
    # Xeon Gold 6xxx only. Gold 51xx/52xx have a single 512-bit FMA and are
    # deliberately excluded by requiring a leading 6.
    r"gold\s*6\d{3}",
    # Sapphire Rapids workstation (Xeon w3-/w5-/w7-/w9-24xx/34xx).
    r"\bw[3579]-\d{4}",
    # Sapphire Rapids HBM (Xeon Max 94xx).
    r"xeon\s*max\s*9\d{3}",
)

# Skylake-SP / Cascade Lake drop to roughly 60-70% of base clock under the
# heavy AVX-512 license. Ice Lake-SP and later, plus Zen 4/5, either do not
# license-throttle or do so negligibly, so they keep a factor of 1.0. The
# pattern matches the SKX (x1xx) and CLX (x2xx) numbering only.
_AVX512_HEAVY_LICENSE_RE = r"\b(?:platinum|gold|silver|bronze)\s*\d[12]\d{2}\b"
_AVX512_HEAVY_LICENSE_FACTOR = 0.70

# AVX2+FMA parts with only one FMA unit (256-bit ops issue at half rate).
# Everything else gets two, which is right for Haswell onward and Zen 2
# onward.
_AVX2_SINGLE_FMA_PATTERNS = (
    r"ryzen\s+\w+\s+1\d{3}",        # Zen 1 desktop (Ryzen 5 1600, ...)
    r"ryzen\s+threadripper\s+19\d{2}",  # Zen 1 Threadripper
    r"epyc\s+7\d{2}1",              # Zen 1 EPYC (7601, 7351, ...)
    r"\batom\b",                    # Silvermont/Goldmont/Tremont/Gracemont
    r"\bceleron\b",
    r"pentium\s+silver",
)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    low = (name or "").lower()
    return any(re.search(p, low) for p in patterns)


def _avx512_fma_units(cpu_name: str) -> int:
    return 2 if _matches_any(cpu_name, _AVX512_DUAL_FMA_PATTERNS) else 1


def _avx512_clock_factor(cpu_name: str) -> float:
    """Heavy-AVX-512-license clock derate for the SKX/CLX generation."""
    if re.search(_AVX512_HEAVY_LICENSE_RE, (cpu_name or "").lower()):
        return _AVX512_HEAVY_LICENSE_FACTOR
    return 1.0


def _avx2_fma_units(cpu_name: str) -> int:
    return 1 if _matches_any(cpu_name, _AVX2_SINGLE_FMA_PATTERNS) else 2


def _cpu_flags() -> set[str]:
    """CPU feature flags as an exact-token set.

    Returning a set (rather than the raw blob the old code substring-matched)
    is what makes the ISA branches trustworthy: `"avx" in blob` is true on
    every AVX-512 machine, and `"sse" in blob` is true merely because of
    "sse2"/"ssse3"/"sse4_1". Exact membership removes both traps.
    Handles the x86 "flags" line and the AArch64 "Features" line.
    """
    out = _run(["bash", "-c",
                "grep -m1 -E '^(flags|Features)' /proc/cpuinfo | cut -d: -f2"])
    if not out:
        out = _run(["bash", "-c", "lscpu | grep -m1 -i '^Flags:' | cut -d: -f2"])
    return {tok.strip().lower() for tok in (out or "").split() if tok.strip()}


def _cpu_isa_profile(flags: set[str], cpu_name: str, machine: str) -> _CpuIsa:
    """Pick the ISA class -- checked most-capable-first, on exact flag tokens."""
    mach = str(machine or "").lower()
    if mach.startswith(("aarch64", "arm64")):
        if "asimd" in flags or "neon" in flags:
            # SVE lands here too: its vector length is not discoverable from
            # the flag list, so the NEON rate is used as a floor.
            return _NEON_ISA
        return _SCALAR_ISA

    if "avx512f" in flags:
        units = _avx512_fma_units(cpu_name)
        return _CpuIsa(
            "avx512", 16, units,
            f"512-bit FMA x{units}" + ("" if units == 2 else " (assumed; no CPUID bit exposes this)"),
        )
    if "avx2" in flags and "fma" in flags:
        units = _avx2_fma_units(cpu_name)
        return _CpuIsa("avx2+fma", 8, units, f"256-bit FMA x{units}")
    if "avx" in flags:
        # Sandy/Ivy Bridge: 256-bit FP mul and FP add ports co-issue, no FMA.
        return _CpuIsa("avx", 8, 1, "pre-FMA AVX: 256-bit FP mul + FP add co-issue")
    if "sse2" in flags or "sse" in flags:
        # Baseline x86-64 SSE2 has no FMA either; 128-bit mul + add co-issue.
        return _CpuIsa("sse2", 4, 1, "pre-FMA SSE2: 128-bit FP mul + FP add co-issue")
    return _SCALAR_ISA


def _parse_float(text: str | None) -> float | None:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _linux_cpu_name() -> str:
    out = _run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"])
    if not out or not out.strip(" -"):
        out = _run(["bash", "-c", "lscpu | grep -m1 -i '^Model name:' | cut -d: -f2"]) or out
    name = (out or "").strip()
    return name if name and name != "-" else "CPU"


def _linux_sustained_clock_ghz(cpu_name: str) -> tuple[float, str] | None:
    """All-core-sustainable clock, plus a label for where it came from.

    Ordered best-to-worst. The old model read `lscpu`'s "CPU max MHz", which is
    the *single-core* max turbo, and then multiplied it by every physical core
    -- an operating point no machine ever reaches.
    """
    # 1. The true nominal base clock, published by intel_pstate (kHz).
    khz = _parse_float(_run(["bash", "-c",
                             "cat /sys/devices/system/cpu/cpu0/cpufreq/base_frequency 2>/dev/null"]))
    if khz and khz > 0:
        return khz / 1e6, "sysfs-base"

    # 2. Intel/AMD brand strings encode the nominal clock: "... @ 2.30GHz".
    m = re.search(r"@\s*([0-9]+(?:\.[0-9]+)?)\s*GHz", cpu_name or "", re.IGNORECASE)
    val = _parse_float(m.group(1)) if m else None
    if val and val > 0:
        return val, "model-name-base"

    # 3. Only the single-core turbo ceiling is known -- derate it toward
    #    all-core rather than pretending every core sustains it.
    mhz = _parse_float(_run(["bash", "-c", "lscpu | grep 'CPU max MHz' | awk '{print $NF}'"]))
    if mhz and mhz > 0:
        return (mhz / 1000.0) * _ALL_CORE_TURBO_DERATE, "max-turbo-derated"

    # 4. Last resort: an instantaneous reading of one core. Not derated -- it
    #    is already a spot value, not a ceiling.
    mhz = _parse_float(_run(["bash", "-c", "grep -m1 'cpu MHz' /proc/cpuinfo | awk '{print $NF}'"]))
    if mhz and mhz > 0:
        return mhz / 1000.0, "cpuinfo-spot"
    return None


def _estimate_cpu_peaks() -> Peaks | None:
    """Model CPU peaks from cores x sustainable clock x FLOP/cycle.

    This is the fallback for `_measured_cpu_peaks`; see the module section
    header above for the FLOP/cycle rule behind every ISA entry.
    """
    import multiprocessing

    sys_name = platform.system()
    cores: int | None = None
    clock_ghz: float | None = None
    isa: _CpuIsa = _SCALAR_ISA
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

        if "Apple" in cpu_name:
            isa = _APPLE_ISA
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
                peak_tflops = (cores * isa.flops_per_cycle * clock_ghz) / 1000.0
                if bw_gbs is None:
                    bw_gbs = 68.0  # conservative
                return Peaks(peak_tflops, bw_gbs, "cpu-spec", cpu_name)

    elif sys_name == "Linux":
        cpu_name = _linux_cpu_name()

        # Core count — physical cores, not logical: SMT doubles the logical
        # count but not the FMA pipelines, so logical count would
        # overestimate the compute roof ~2x on hyperthreaded machines.
        cores = _physical_cpu_count()
        if cores is None:
            try:
                cores = multiprocessing.cpu_count()
            except (OSError, NotImplementedError):
                pass

        clock = _linux_sustained_clock_ghz(cpu_name)
        clock_source = "unknown"
        if clock is not None:
            clock_ghz, clock_source = clock

        isa = _cpu_isa_profile(_cpu_flags(), cpu_name, platform.machine())
        if isa.name == "avx512" and clock_ghz is not None:
            # Heavy AVX-512 code licenses down on SKX/CLX; fold that into the
            # clock rather than into the FLOP/cycle number, which is a
            # microarchitectural constant.
            factor = _avx512_clock_factor(cpu_name)
            if factor != 1.0:
                clock_ghz *= factor
                clock_source += f"+avx512-license({factor:g})"
        logger.debug(
            "CPU peak model: cores=%s clock=%.3fGHz (%s) isa=%s %d FLOP/cycle (%s)",
            cores, clock_ghz or 0.0, clock_source, isa.name, isa.flops_per_cycle, isa.note,
        )

    if cores and clock_ghz and cores > 0 and clock_ghz > 0:
        peak_tflops = (cores * isa.flops_per_cycle * clock_ghz) / 1000.0
        # Estimate bandwidth: default heuristic for DDR
        # Try to get from dmidecode or lshw, fall back to conservative estimate
        bw_gbs = None
        if sys_name == "Linux":
            # Try lsmem / dmidecode for memory bandwidth
            # sudo -n: never prompt for a password (would block the run
            # waiting on stdin). If credentials aren't cached, sudo exits
            # non-zero, the pipeline yields no output, and _run/the empty
            # checks below fall through to the conservative bw estimate.
            mem_info = _run(["bash", "-c", "sudo -n dmidecode -t memory 2>/dev/null | grep -i 'speed:' | head -1 | awk '{print $2}'"])
            channels_str = _run(["bash", "-c", "sudo -n dmidecode -t memory 2>/dev/null | grep -ic 'size:.*[0-9]'"])
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


# ===== Measured CPU peaks =====
#
# Bump this whenever the probe's methodology changes -- it is part of the
# cache key, so old entries are ignored rather than silently reused.
_CPU_PROBE_VERSION = 1

_CPU_PROBE_TIMEOUT_S = 30.0   # hard wall on the child process
_CPU_PROBE_BUDGET_S = 8.0     # self-imposed deadline inside the child
_CPU_PROBE_WARMUP_S = 0.3     # per kernel, to reach a steady clock
_CPU_PROBE_MIN_TRIAL_S = 0.05
_CPU_PROBE_TRIALS = 7
_CPU_PROBE_GEMM_N = 1024      # 3 x 4 MB operands; AI = N/6 ~ 170 FLOP/byte
_CPU_PROBE_BW_MB = 64         # per buffer, x2 buffers -- far past any LLC

# A run has to clear all of these to be believed.
_CPU_PROBE_MIN_TRIALS = 3
# max/median across trials. A wide spread means the machine was contended or
# thermally unstable during the probe, so the number is not a peak of anything.
_CPU_PROBE_MAX_SPREAD = 3.0
# Absolute plausibility bounds. Deliberately *not* relative to the modeled
# estimate: the model is the thing being distrusted, and a measurement that
# comes in far below it is usually right (cgroup CPU quota, a single-threaded
# BLAS, a shared VM) -- capping it against the model would reintroduce exactly
# the fiction this probe exists to remove.
_CPU_PROBE_TFLOPS_BOUNDS = (1e-4, 200.0)
_CPU_PROBE_GBS_BOUNDS = (0.5, 5000.0)

_CPU_PROBE_MARKER = "PERFLAB_CPU_PROBE "

# Runs out-of-process on purpose: a hard timeout, isolation from BLAS crashes,
# and -- the reason it cannot be done in-process -- BLAS thread-count env vars
# are only read at numpy import time, so the parent cannot set them for an
# already-imported numpy.
_CPU_PROBE_SCRIPT = r'''
import json
import sys
import time


def _emit(obj):
    sys.stdout.write("PERFLAB_CPU_PROBE " + json.dumps(obj) + "\n")
    sys.stdout.flush()


def _timed_trials(call, units_per_call, cfg, deadline):
    """Warm up to a steady clock, size an iteration batch, then time N trials."""
    warm_end = time.perf_counter() + cfg["warmup_s"]
    while time.perf_counter() < warm_end:
        call()
    t0 = time.perf_counter()
    call()
    one = time.perf_counter() - t0
    iters = max(1, int(cfg["min_trial_s"] / one) + 1) if one > 0 else 1
    rates = []
    for _ in range(cfg["trials"]):
        if time.perf_counter() > deadline:
            break
        t0 = time.perf_counter()
        for _ in range(iters):
            call()
        dt = time.perf_counter() - t0
        if dt > 0:
            rates.append(units_per_call * iters / dt)
    return rates, iters


def main():
    cfg = json.loads(sys.argv[1])
    deadline = time.perf_counter() + cfg["budget_s"]
    try:
        import numpy as np
    except Exception as exc:
        _emit({"error": "numpy-unavailable: " + repr(exc)})
        return

    out = {"numpy": np.__version__, "threads": cfg["threads"]}

    # Compute roof: fp32 GEMM. 2*N^3 FLOP over 3*N^2 elements is an arithmetic
    # intensity of N/6 (~170 FLOP/byte at N=1024), an order of magnitude past
    # any CPU knee, and BLAS keeps the blocked tiles resident in L1/L2 -- so
    # this measures the FMA pipelines, not the memory system.
    try:
        n = cfg["gemm_n"]
        rng = np.random.default_rng(0)
        a = rng.standard_normal((n, n), dtype=np.float32)
        b = rng.standard_normal((n, n), dtype=np.float32)
        c = np.empty((n, n), dtype=np.float32)

        def gemm():
            np.matmul(a, b, out=c)

        rates, iters = _timed_trials(gemm, 2.0 * n * n * n, cfg, deadline)
        out["gemm_tflops"] = [r / 1e12 for r in rates]
        out["gemm_n"] = n
        out["gemm_iters_per_trial"] = iters
    except Exception as exc:
        out["gemm_error"] = repr(exc)

    # Bandwidth roof: streaming scale (read x, write y) over buffers far larger
    # than any LLC, fanned out across threads because a single core cannot
    # saturate a modern socket's DRAM controllers. numpy ufuncs drop the GIL,
    # so plain threads really do run concurrently here.
    try:
        import concurrent.futures as cf

        threads = max(1, int(cfg["threads"]))
        nelem = (cfg["bw_mb"] * 1024 * 1024) // 4
        x = np.ones(nelem, dtype=np.float32)
        y = np.empty(nelem, dtype=np.float32)
        s = np.float32(1.0000001)
        step = max(1, nelem // threads)
        bounds = []
        for t in range(threads):
            lo = t * step
            hi = nelem if t == threads - 1 else min(nelem, (t + 1) * step)
            if lo < hi:
                bounds.append((lo, hi))

        def _chunk(lohi):
            lo, hi = lohi
            np.multiply(x[lo:hi], s, out=y[lo:hi])

        pool = cf.ThreadPoolExecutor(max_workers=len(bounds))
        try:
            def stream():
                list(pool.map(_chunk, bounds))

            rates, _ = _timed_trials(stream, 2.0 * x.nbytes, cfg, deadline)
        finally:
            pool.shutdown(wait=True)
        out["stream_gbs"] = [r / 1e9 for r in rates]
        out["stream_mb"] = cfg["bw_mb"]
    except Exception as exc:
        out["stream_error"] = repr(exc)

    _emit(out)


main()
'''


def _cpu_measure_enabled() -> bool:
    return os.environ.get("PERFLAB_CPU_PEAK_MEASURE", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _cpu_probe_threads() -> int:
    import multiprocessing

    n = _physical_cpu_count()
    if not n:
        try:
            n = multiprocessing.cpu_count()
        except (OSError, NotImplementedError):
            n = 1
    return max(1, int(n or 1))


def _cpu_probe_thread_plan(threads: int) -> list[int]:
    """Thread counts to try, best-of wins.

    All cores is not always fastest: SMT oversubscription, vCPU contention
    inside a VM, and P+E heterogeneity can each make a narrower run quicker.
    Measured in this repo's aarch64 dev container, a 10-thread GEMM ran 2.3x
    *slower* than a 5-thread one -- exactly the size of error this whole probe
    exists to eliminate. A roof should be the best the machine can do, so try a
    half-width configuration as well. Capped at two configurations to keep the
    one-time cost near ~3s.
    """
    plan = [threads]
    if threads >= 4:
        plan.append(threads // 2)
    return plan


def _run_cpu_probe(threads: int) -> dict[str, Any] | None:
    """Run the microbenchmark out-of-process. Returns None on any failure."""
    if not sys.executable:
        return None
    cfg = {
        "budget_s": _CPU_PROBE_BUDGET_S,
        "warmup_s": _CPU_PROBE_WARMUP_S,
        "min_trial_s": _CPU_PROBE_MIN_TRIAL_S,
        "trials": _CPU_PROBE_TRIALS,
        "gemm_n": _CPU_PROBE_GEMM_N,
        "bw_mb": _CPU_PROBE_BW_MB,
        "threads": threads,
    }
    env = dict(os.environ)
    # Set before numpy is imported so the BLAS actually honors them; the roof
    # is a whole-socket number, matching what the model computes.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = str(threads)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CPU_PROBE_SCRIPT, json.dumps(cfg)],
            capture_output=True,
            text=True,
            timeout=_CPU_PROBE_TIMEOUT_S,
            env=env,
            # Never inherit a candidate workspace as cwd: `python -c` puts cwd
            # on sys.path, and workspaces hold LLM-written files.
            cwd=tempfile.gettempdir(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.info("CPU peak probe did not complete; falling back to the model", exc_info=True)
        return None

    for line in (proc.stdout or "").splitlines():
        if line.startswith(_CPU_PROBE_MARKER):
            try:
                payload = json.loads(line[len(_CPU_PROBE_MARKER):])
            except json.JSONDecodeError:
                logger.info("CPU peak probe emitted unparseable output")
                return None
            return payload if isinstance(payload, dict) else None
    logger.info("CPU peak probe produced no result (exit=%s)", proc.returncode)
    return None


def _reduce_probe_rates(rates: Any, bounds: tuple[float, float], label: str) -> float | None:
    """Peak = max across trials, but only if the trials are self-consistent."""
    if not isinstance(rates, list):
        return None
    values = [float(r) for r in rates if isinstance(r, (int, float)) and r == r and r > 0]
    if len(values) < _CPU_PROBE_MIN_TRIALS:
        logger.info("CPU peak probe (%s): only %d usable trials", label, len(values))
        return None
    values.sort()
    peak = values[-1]
    median = values[len(values) // 2]
    lo, hi = bounds
    if not (lo <= peak <= hi):
        logger.warning("CPU peak probe (%s): %.4g outside plausible range [%g, %g]", label, peak, lo, hi)
        return None
    if median <= 0 or peak / median > _CPU_PROBE_MAX_SPREAD:
        logger.warning("CPU peak probe (%s): unstable trials (peak %.4g vs median %.4g)", label, peak, median)
        return None
    return peak


def _cpu_measured_cache_key(device: str, threads: int) -> str:
    return f"cpu-measured::v{_CPU_PROBE_VERSION}::{platform.machine()}::{device}::{threads}t"


def _measured_cpu_peaks(model: Peaks | None) -> Peaks | None:
    """Measure the CPU compute roof with a short calibrated fp32 GEMM.

    Preferred over `_estimate_cpu_peaks` because it observes what the machine
    actually sustains, which folds in all-core turbo behavior, AVX-512
    licensing and FMA-port count without having to model any of them.

    Degrades to None (caller falls back to the model) on every failure mode:
    numpy missing, subprocess crash or timeout, too few trials, unstable
    trials, or an implausible absolute result. Cached in the shared peaks
    cache -- including failures, so a machine without numpy does not pay for
    the probe on every run. `perflab peaks --refresh` /
    PERFLAB_PEAKS_NO_CACHE=1 re-runs it.

    Source labels: "cpu-measured" when both roofs are measured,
    "cpu-measured-flops" when only the compute roof is (bandwidth then comes
    from the model, which is the honest description of the mix).
    """
    if not _cpu_measure_enabled():
        return None

    device = model.device if model and model.device else (
        platform.processor() or platform.machine() or "CPU"
    )
    threads = _cpu_probe_threads()
    key = _cpu_measured_cache_key(device, threads)

    cache = _load_cache()
    entry = cache.get(key)
    if isinstance(entry, dict):
        if entry.get("failed"):
            logger.debug("CPU peak probe previously failed (%s); using the model", entry.get("reason"))
            return None
        try:
            return Peaks(
                float(entry["peak_tflops"]),
                float(entry["peak_mem_bw_gbs"]),
                str(entry.get("source", "cpu-measured")),
                str(entry.get("device", device)),
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Ignoring malformed measured-CPU cache entry %s", key)

    tflops: float | None = None
    gbs: float | None = None
    for n_threads in _cpu_probe_thread_plan(threads):
        result = _run_cpu_probe(n_threads)
        if result is None:
            # Process-level failure (timeout, exec error, crash) is
            # environmental; retrying at another width would just burn a
            # second full timeout.
            break
        if result.get("error"):
            logger.info("CPU peak probe unavailable: %s", result["error"])
            break  # numpy missing: retrying at another thread count is pointless
        run_tflops = _reduce_probe_rates(
            result.get("gemm_tflops"), _CPU_PROBE_TFLOPS_BOUNDS, f"gemm@{n_threads}t")
        run_gbs = _reduce_probe_rates(
            result.get("stream_gbs"), _CPU_PROBE_GBS_BOUNDS, f"stream@{n_threads}t")
        if run_tflops is not None and (tflops is None or run_tflops > tflops):
            tflops = run_tflops
        if run_gbs is not None and (gbs is None or run_gbs > gbs):
            gbs = run_gbs

    if tflops is None:
        _store_cpu_measured_failure(cache, key)
        return None

    if gbs is None:
        if model is None or model.peak_mem_bw_gbs <= 0:
            _store_cpu_measured_failure(cache, key)
            return None
        gbs = model.peak_mem_bw_gbs
        source = "cpu-measured-flops"
    else:
        source = "cpu-measured"

    peaks = Peaks(float(tflops), float(gbs), source, device)
    if model is not None and model.peak_tflops > 0:
        logger.info(
            "CPU peak measured at %.3f TFLOPS (%.2fx the modeled %.3f), %.1f GB/s",
            peaks.peak_tflops, peaks.peak_tflops / model.peak_tflops,
            model.peak_tflops, peaks.peak_mem_bw_gbs,
        )
    cache[key] = {
        "peak_tflops": peaks.peak_tflops,
        "peak_mem_bw_gbs": peaks.peak_mem_bw_gbs,
        "source": peaks.source,
        "device": peaks.device,
    }
    _save_cache(cache)
    return peaks


def _store_cpu_measured_failure(cache: dict[str, Any], key: str) -> None:
    cache[key] = {"failed": True, "reason": "probe-unavailable-or-implausible"}
    _save_cache(cache)


def infer_cpu_peaks() -> Peaks | None:
    """Measured CPU peaks first, then the spec model, then torch calibration.

    The model is computed first regardless: it is cheap (a few sysctl/lscpu
    reads), it names the device, and it supplies the bandwidth roof if only the
    compute half of the measurement survives validation.
    """
    spec = _estimate_cpu_peaks()
    measured = _measured_cpu_peaks(spec)
    if measured:
        return measured
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
        "cpu": "CPU peaks are measured by a ~2s cached microbenchmark (source=cpu-measured); set PERFLAB_CPU_PEAK_MEASURE=0 to use the spec model (source=cpu-spec) instead.",
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
        for name, (bf16_tflops, hbm_bw, _hbm_gb) in _KNOWN_TPU_SPECS.items():
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
    except Exception:  # noqa: BLE001 -- best-effort TPU detection, must not abort roofline resolution
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
    except Exception:  # noqa: BLE001 -- best-effort auto-detect, must not abort the caller
        logger.warning("Roofline auto-detect failed", exc_info=True)
    return None
