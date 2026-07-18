from __future__ import annotations

import json
import logging
import os
import platform
import re
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

        # Core count — physical cores, not logical: SMT doubles the logical
        # count but not the FMA pipelines, so logical count would
        # overestimate the compute roof ~2x on hyperthreaded machines.
        cores = _physical_cpu_count()
        if cores is None:
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
