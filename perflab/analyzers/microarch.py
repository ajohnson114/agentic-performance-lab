"""Micro-architectural analysis: derived metrics for deep optimization.

Computes kernel-specific theoretical ceilings, benchmark stability scores,
GPU clock throttle detection, pipeline utilization heatmaps, and instruction
efficiency from SASS — giving the LLM the information it needs to optimize
beyond 80% of peak.
"""
from __future__ import annotations

import math


def compute_benchmark_stability(bench_data: dict) -> dict | None:
    """Compute benchmark stability metrics from bench results.

    Returns dict with cv_pct, is_stable, min_meaningful_improvement_pct,
    and a human-readable assessment.
    """
    times = bench_data.get("times_ms", [])
    if len(times) < 3:
        return None

    mean = sum(times) / len(times)
    if mean <= 0:
        return None

    variance = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
    std = math.sqrt(variance)
    cv = std / mean

    # Minimum meaningful improvement: must exceed 2x the CV to be statistically significant
    min_improvement = cv * 2 * 100

    result: dict = {
        "cv_pct": round(cv * 100, 1),
        "std_ms": round(std, 3),
        "mean_ms": round(mean, 3),
        "n_samples": len(times),
        "is_stable": cv < 0.05,
        "min_meaningful_improvement_pct": round(min_improvement, 1),
    }

    if cv < 0.03:
        result["assessment"] = f"Very stable (CV={cv:.1%}) — improvements >1% are real"
    elif cv < 0.05:
        result["assessment"] = f"Stable (CV={cv:.1%}) — improvements >{min_improvement:.0f}% are reliable"
    elif cv < 0.10:
        result["assessment"] = f"Moderate noise (CV={cv:.1%}) — only improvements >{min_improvement:.0f}% are meaningful"
    else:
        result["assessment"] = f"High noise (CV={cv:.1%}) — results unreliable, increase repeats or reduce system load"

    return result


def detect_clock_throttle(power_data: dict) -> dict | None:
    """Detect GPU clock throttling from power profiler data.

    Returns dict with throttle_detected, clock_drop_pct, effective_peak_ratio,
    and human-readable assessment.
    """
    gpu_power = power_data.get("gpu_power", {})
    samples = gpu_power.get("power_samples", [])

    # Try raw samples first
    watts = [s.get("watts", 0) for s in samples if isinstance(s, dict) and s.get("watts")]

    # Fallback: use aggregate stats if raw samples aren't available
    if len(watts) < 3:
        min_w = gpu_power.get("min_watts")
        max_w = gpu_power.get("max_watts")
        avg_w = gpu_power.get("avg_watts")
        if min_w is not None and max_w is not None and avg_w is not None and max_w > 0:
            watts = [max_w, avg_w, min_w]  # Synthetic 3-point series
        else:
            return None

    if len(watts) < 3:
        return None

    peak_power = max(watts)
    min_power = min(watts)
    avg_power = sum(watts) / len(watts)

    if peak_power <= 0:
        return None

    drop_pct = (peak_power - min_power) / peak_power * 100

    result: dict = {
        "peak_power_w": round(peak_power, 1),
        "min_power_w": round(min_power, 1),
        "avg_power_w": round(avg_power, 1),
        "power_drop_pct": round(drop_pct, 1),
        "throttle_detected": drop_pct > 10.0,
    }

    # Effective peak: if power dropped X%, assume clocks dropped proportionally
    # (simplified — real relationship is non-linear but directionally correct)
    if drop_pct > 10.0:
        effective_ratio = avg_power / peak_power
        result["effective_peak_ratio"] = round(effective_ratio, 2)
        result["assessment"] = (
            f"GPU power dropped {drop_pct:.0f}% during benchmark "
            f"(peak {peak_power:.0f}W → min {min_power:.0f}W). "
            f"Effective peak is ~{effective_ratio:.0%} of theoretical. "
            f"The kernel may be thermally limited — further optimization may "
            f"not improve wall-clock time."
        )
    else:
        result["effective_peak_ratio"] = 1.0
        result["assessment"] = f"No throttling detected — GPU power stable ({drop_pct:.0f}% variation)"

    return result


def format_pipeline_heatmap(ncu_metrics: dict) -> str | None:
    """Format NCU instruction mix as a pipeline utilization heatmap.

    Shows all pipe utilizations so the LLM can identify which execution
    unit is the bottleneck and which are idle.
    """
    inst_mix = ncu_metrics.get("instruction_mix", {})
    tc_util = ncu_metrics.get("tensor_core_utilization_pct")

    # Combine instruction mix with tensor core utilization
    pipes: list[tuple[str, float | None]] = [
        ("FP32/FMA", inst_mix.get("fp32_fma")),
        ("Tensor Core", tc_util),
        ("FP64", inst_mix.get("fp64")),
        ("INT/ALU", inst_mix.get("int_alu")),
        ("SFU (transcendentals)", inst_mix.get("sfu")),
    ]

    active = [(name, val) for name, val in pipes if val is not None]
    if not active:
        return None

    lines: list[str] = ["Pipeline utilization:"]
    for name, val in active:
        # Visual bar: █ for each 10%
        bar_len = int(val / 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        level = "HIGH" if val > 70 else "MED" if val > 30 else "LOW"
        lines.append(f"  {name:22s} {bar} {val:5.1f}% [{level}]")

    return "\n".join(lines)


def build_microarch_summary(
    bench_data: dict,
    profiler_summaries: dict,
    peak_tflops: float | None = None,
) -> dict | None:
    """Build a comprehensive micro-architecture analysis summary.

    Combines benchmark stability, clock throttle detection, kernel ceiling,
    and pipeline utilization into a single summary for the LLM prompt.
    """
    result: dict = {}

    # Benchmark stability
    stability = compute_benchmark_stability(bench_data)
    if stability:
        result["benchmark_stability"] = stability

    # Clock throttle detection
    power_data = profiler_summaries.get("power", {})
    if power_data:
        throttle = detect_clock_throttle(power_data)
        if throttle:
            result["clock_throttle"] = throttle

    # Kernel ceiling (requires NCU occupancy + roofline peak)
    ncu = profiler_summaries.get("ncu", {})
    dominant = ncu.get("dominant_kernel", {})
    occupancy = dominant.get("achieved_occupancy_pct")
    if occupancy is not None and peak_tflops is not None and peak_tflops > 0:
        from perflab.reporting.roofline import compute_kernel_ceiling
        achieved = None
        tflops_data = bench_data.get("tflops", {})
        if isinstance(tflops_data, dict):
            achieved = tflops_data.get("median")
        ceiling = compute_kernel_ceiling(occupancy, peak_tflops, achieved)
        # Adjust ceiling for clock throttle
        throttle = result.get("clock_throttle")
        if throttle and throttle.get("effective_peak_ratio"):
            ratio = throttle["effective_peak_ratio"]
            ceiling["effective_ceiling_tflops"] = round(
                ceiling["kernel_ceiling_tflops"] * ratio, 2
            )
        result["kernel_ceiling"] = ceiling

    # Pipeline utilization heatmap
    if dominant:
        heatmap = format_pipeline_heatmap(dominant)
        if heatmap:
            result["pipeline_heatmap"] = heatmap

    return result if result else None
