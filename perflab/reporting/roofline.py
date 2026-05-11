from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

logger = logging.getLogger(__name__)

@dataclass
class RooflinePoint:
    ai: float      # FLOP / byte
    tflops: float  # achieved
    gbs: float     # achieved memory traffic rate
    achieved_bw_gbs: float | None = None  # profiler-measured DRAM bandwidth

def _dtype_bytes(dtype: str) -> int:
    d = (dtype or "").lower()
    if d in ("fp16", "float16", "half", "f16"):
        return 2
    if d in ("bf16", "bfloat16"):
        return 2
    if d in ("fp32", "float32", "f32", "float"):
        return 4
    if d in ("fp64", "float64", "double", "f64"):
        return 8
    return 4

def _matmul_flops(M: int, N: int, K: int, batch: int) -> float:
    return 2.0 * float(M) * float(N) * float(K) * float(batch)

def _matmul_bytes(M: int, N: int, K: int, batch: int, bpe: int) -> float:
    # A: (batch,M,K), B: (batch,K,N), C: (batch,M,N)
    # Approx traffic: read A + read B + read+write C (~2*C)
    return float(batch) * float(bpe) * (float(M)*float(K) + float(K)*float(N) + 2.0*float(M)*float(N))

def compute_roofline_point(
    bench: dict[str, Any],
    measured_dram_bytes: float | None = None,
    profiler_flops: float | None = None,
) -> RooflinePoint | None:
    """Compute a roofline data point from benchmark results.

    Supports two modes:
    1. **Matmul mode**: If meta contains M, N, K keys, computes FLOPs and bytes
       from the standard matmul formula (2*M*N*K).
    2. **Generic mode**: If meta contains explicit "flops" and "bytes_moved" keys,
       computes arithmetic intensity directly (AI = flops / bytes_moved). This
       lets task authors provide custom FLOPs accounting in bench.py.

    In both cases, timing is derived from latency_ms.p50 or back-computed from
    the tflops.median field.
    """
    meta = (bench or {}).get("meta", {}) or {}

    # Try matmul-specific mode first
    if all(k in meta for k in ("M", "N", "K")):
        return _compute_matmul_roofline(bench, meta, measured_dram_bytes=measured_dram_bytes)

    # Generic mode: task provides explicit flops and bytes_moved
    flops = meta.get("flops")
    bytes_moved = meta.get("bytes_moved")

    # If meta doesn't have flops, try profiler-provided FLOPS (PyTorch with_flops, JAX HLO cost)
    if flops is None and profiler_flops is not None and profiler_flops > 0:
        flops = profiler_flops
        # If we have profiler FLOPS but no bytes_moved, we can still compute achieved TFLOPS
        # and use measured DRAM bytes if available
        if bytes_moved is None and measured_dram_bytes is not None and measured_dram_bytes > 0:
            bytes_moved = measured_dram_bytes

    if flops is not None and bytes_moved is not None:
        try:
            flops = float(flops)
            bytes_moved = float(bytes_moved)
        except (TypeError, ValueError):
            logger.warning("roofline: meta.flops or meta.bytes_moved is not numeric")
            return None
        if flops <= 0 or bytes_moved <= 0:
            logger.warning("roofline: meta.flops (%s) and meta.bytes_moved (%s) must be positive", flops, bytes_moved)
            return None

        # Sanity check: bytes_moved should be in bytes, not element counts.
        # If bytes_moved < flops/1000 it's suspiciously small — likely elements
        # rather than bytes (e.g., forgot to multiply by dtype size).
        if bytes_moved < flops / 1000:
            logger.warning(
                "roofline: meta.bytes_moved (%.3g) is very small relative to "
                "meta.flops (%.3g) — AI=%.0f FLOP/byte. Verify bytes_moved is "
                "in bytes, not element counts.",
                bytes_moved, flops, flops / bytes_moved,
            )

        seconds = _extract_seconds(bench, meta, flops)
        if seconds is None or seconds <= 0:
            return None

        # Prefer profiler-measured DRAM bytes over theoretical estimate
        effective_bytes = bytes_moved
        if measured_dram_bytes is not None and measured_dram_bytes > 0:
            effective_bytes = measured_dram_bytes

        ai = flops / effective_bytes
        achieved_tflops = flops / seconds / 1e12
        achieved_gbs = effective_bytes / seconds / 1e9
        achieved_bw = measured_dram_bytes / seconds / 1e9 if measured_dram_bytes and measured_dram_bytes > 0 else None

        # Warn on extreme AI values that may indicate misconfigured meta
        if ai > 1000:
            logger.warning(
                "roofline: arithmetic intensity is unusually high (%.0f FLOP/byte). "
                "Verify meta.bytes_moved accounts for all DRAM traffic.", ai,
            )
        elif ai < 0.01:
            logger.warning(
                "roofline: arithmetic intensity is unusually low (%.4f FLOP/byte). "
                "Verify meta.flops is correct.", ai,
            )

        return RooflinePoint(ai=ai, tflops=achieved_tflops, gbs=achieved_gbs, achieved_bw_gbs=achieved_bw)

    return None


def _compute_matmul_roofline(
    bench: dict, meta: dict, measured_dram_bytes: float | None = None,
) -> RooflinePoint | None:
    """Compute roofline point using matmul-specific M, N, K keys."""
    M = int(meta["M"]); N = int(meta["N"]); K = int(meta["K"])
    batch = int(meta.get("batch", 1))
    dtype = str(meta.get("dtype", "fp32"))
    bpe = _dtype_bytes(dtype)

    flops = _matmul_flops(M, N, K, batch)
    seconds = _extract_seconds(bench, meta, flops)
    if seconds is None or seconds <= 0:
        return None

    byt = _matmul_bytes(M, N, K, batch, bpe)
    # Prefer profiler-measured DRAM bytes over theoretical estimate
    effective_bytes = byt
    if measured_dram_bytes is not None and measured_dram_bytes > 0:
        effective_bytes = measured_dram_bytes

    ai = flops / effective_bytes if effective_bytes > 0 else 0.0
    achieved_tflops = flops / seconds / 1e12
    achieved_gbs = effective_bytes / seconds / 1e9
    achieved_bw = measured_dram_bytes / seconds / 1e9 if measured_dram_bytes and measured_dram_bytes > 0 else None
    return RooflinePoint(ai=ai, tflops=achieved_tflops, gbs=achieved_gbs, achieved_bw_gbs=achieved_bw)


def _extract_seconds(bench: dict, meta: dict, flops: float) -> float | None:
    """Best-effort extraction of timing from bench results."""
    # Try p50 latency
    lat = (bench or {}).get("latency_ms", {}) or {}
    if "p50" in lat:
        try:
            seconds = float(lat["p50"]) / 1000.0
            if seconds > 0:
                return seconds
        except (ValueError, TypeError, KeyError):
            pass

    # Infer from tflops.median
    if "tflops" in (bench or {}):
        try:
            achieved = float((bench.get("tflops", {}) or {}).get("median"))
            if achieved > 0:
                return flops / (achieved * 1e12)
        except (ValueError, TypeError, KeyError):
            pass

    return None


def select_peak_tflops(peak_tflops: float, dtype: str | None = None,
                       dtype_peaks: dict[str, float] | None = None) -> float:
    """Select the appropriate peak TFLOPS based on dtype.

    If dtype_peaks is available, uses the precision-specific peak.
    Otherwise falls back to the single peak_tflops value.
    """
    if not dtype_peaks or not dtype:
        return peak_tflops

    d = (dtype or "").lower()

    if d in ("fp16", "float16", "half"):
        return dtype_peaks.get("peak_tflops_fp16", peak_tflops)
    if d in ("bf16", "bfloat16"):
        return dtype_peaks.get("peak_tflops_bf16", peak_tflops)
    if d in ("fp32", "float32", "float"):
        # Use TF32 if available (Ampere+), otherwise FP32
        return dtype_peaks.get("peak_tflops_tf32",
               dtype_peaks.get("peak_tflops_fp32", peak_tflops))
    if d in ("fp64", "float64", "double"):
        return dtype_peaks.get("peak_tflops_fp64", peak_tflops)

    return peak_tflops


def compute_kernel_ceiling(
    occupancy_pct: float,
    peak_tflops: float,
    achieved_tflops: float | None = None,
) -> dict:
    """Compute the theoretical performance ceiling for a specific kernel.

    A kernel at X% occupancy can never exceed X% of hardware peak, regardless
    of other optimizations. This gives the LLM a realistic target.

    Returns dict with kernel_ceiling_tflops, headroom_pct, occupancy_limited.
    """
    ceiling = peak_tflops * (occupancy_pct / 100.0)
    result: dict = {
        "kernel_ceiling_tflops": round(ceiling, 2),
        "occupancy_pct": round(occupancy_pct, 1),
        "peak_tflops": round(peak_tflops, 1),
    }
    if achieved_tflops is not None and ceiling > 0:
        result["achieved_tflops"] = round(achieved_tflops, 3)
        result["pct_of_ceiling"] = round(achieved_tflops / ceiling * 100, 1)
        result["pct_of_peak"] = round(achieved_tflops / peak_tflops * 100, 1)
        result["occupancy_limited"] = occupancy_pct < 50.0
    return result


def write_roofline_png(
    out_png: Path,
    *,
    point: RooflinePoint,
    peak_tflops: float,
    peak_mem_bw_gbs: float,
    title: str,
    dtype_peaks: dict[str, float] | None = None,
    l2_bw_gbs: float | None = None,
    history_points: list[dict] | None = None,
) -> None:
    """history_points: list of dicts with keys 'iteration', 'roofline_ai',
    'roofline_tflops', 'description'. Plotted as a labeled trail behind the
    current point. Baseline (iter 0) and accepted patches only."""
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    all_ai = [point.ai] + [p["roofline_ai"] for p in (history_points or []) if "roofline_ai" in p]
    x_min = max(1e-4, min(all_ai) / 50.0)
    x_max = max(max(all_ai) * 50.0, 1e-2)

    xs = []
    n = 220
    log_min = math.log10(x_min)
    log_max = math.log10(x_max)
    for i in range(n):
        xs.append(10 ** (log_min + (log_max - log_min) * i / (n - 1)))

    # BW line: (GB/s * flop/byte) => GFLOP/s; /1000 => TFLOP/s
    ys_bw = [(peak_mem_bw_gbs * x) / 1000.0 for x in xs]
    ys = [min(peak_tflops, y) for y in ys_bw]

    plt.figure(figsize=(10, 6))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic Intensity (FLOP / byte)")
    plt.ylabel("Performance (TFLOP/s)")
    plt.title(title)

    plt.plot(xs, ys, "k-", linewidth=2, label="Roofline (DRAM)")
    plt.axhline(peak_tflops, linestyle="--", color="gray",
                label=f"Peak compute ≈ {peak_tflops:.1f} TFLOP/s")
    knee_ai = (1000.0 * peak_tflops) / max(1e-9, peak_mem_bw_gbs)
    plt.axvline(knee_ai, linestyle="--", color="gray", alpha=0.5,
                label=f"Knee AI ≈ {knee_ai:.2f}")

    # L2 cache bandwidth ceiling (hierarchical roofline)
    if l2_bw_gbs is not None and l2_bw_gbs > peak_mem_bw_gbs:
        ys_l2 = [min(peak_tflops, (l2_bw_gbs * x) / 1000.0) for x in xs]
        plt.plot(xs, ys_l2, "-", color="#059669", linewidth=1.5, alpha=0.7,
                 label=f"L2 cache ≈ {l2_bw_gbs:.0f} GB/s")

    # Per-dtype ceiling lines
    if dtype_peaks:
        _DTYPE_COLORS = {
            "peak_tflops_fp64": ("#7c3aed", "FP64"),
            "peak_tflops_fp32": ("#dc2626", "FP32"),
            "peak_tflops_tf32": ("#ea580c", "TF32"),
            "peak_tflops_fp16": ("#2563eb", "FP16"),
            "peak_tflops_bf16": ("#0891b2", "BF16"),
        }
        for key, (color, label) in _DTYPE_COLORS.items():
            val = dtype_peaks.get(key)
            if val is not None and val != peak_tflops:
                plt.axhline(val, linestyle=":", color=color, alpha=0.7,
                            label=f"{label} ≈ {val:.1f} TFLOP/s")

    # History trail: baseline + accepted iterations
    trail = [p for p in (history_points or []) if "roofline_ai" in p and "roofline_tflops" in p]
    if trail:
        trail_ai = [p["roofline_ai"] for p in trail]
        trail_tf = [p["roofline_tflops"] for p in trail]
        # Faded connecting line
        plt.plot(trail_ai, trail_tf, "o--", color="#94a3b8", linewidth=1,
                 markersize=5, zorder=4, label="Optimization trail")
        for p in trail:
            label = f"iter {p['iteration']}" if p["iteration"] > 0 else "baseline"
            plt.annotate(
                label,
                (p["roofline_ai"], p["roofline_tflops"]),
                textcoords="offset points",
                xytext=(-8, 6),
                fontsize=7,
                color="#64748b",
            )

    # Annotation text
    annot_lines = [f"AI={point.ai:.3g}", f"{point.tflops:.2f} TFLOP/s"]
    if point.achieved_bw_gbs is not None:
        annot_lines.append(f"BW={point.achieved_bw_gbs:.1f} GB/s (measured)")
    else:
        annot_lines.append(f"{point.gbs:.1f} GB/s")

    plt.scatter([point.ai], [point.tflops], marker="*", s=180, zorder=5,
                color="#f59e0b", label=f"Best ≈ {point.tflops:.2f} TFLOP/s")
    plt.annotate(
        "\n".join(annot_lines),
        (point.ai, point.tflops),
        textcoords="offset points",
        xytext=(10, 10),
    )

    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
