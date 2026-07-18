"""Build flag recommendations from ISA detection.

Parses the existing build command and cross-references with detected CPU ISA
features to recommend missing beneficial compiler flags.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlagRecommendation:
    """A recommended build flag with rationale."""
    flag: str           # "-march=native"
    reason: str         # "Machine supports AVX-512 but binary compiled without it"
    impact: str         # "high", "medium", "low"
    category: str       # "isa", "optimization", "debug"


def recommend_build_flags(
    build_cmd: str,
    cpu_isa: dict,
    program_type: str,
) -> list[FlagRecommendation]:
    """Recommend missing beneficial build flags.

    Parses the build command string and checks against detected ISA features
    and optimization best practices.
    """
    recs: list[FlagRecommendation] = []
    cmd_lower = build_cmd.lower()

    if program_type in ("cpp", "cuda"):
        recs.extend(_check_isa_flags(cmd_lower, cpu_isa))
        recs.extend(_check_optimization_flags(cmd_lower, program_type))
        recs.extend(_check_debug_flags(cmd_lower))

    if program_type == "cuda":
        recs.extend(_check_cuda_flags(cmd_lower))

    return recs


def _check_isa_flags(cmd: str, cpu_isa: dict) -> list[FlagRecommendation]:
    """Check for missing ISA-specific flags."""
    recs: list[FlagRecommendation] = []

    has_march_native = "-march=native" in cmd
    has_mavx2 = "-mavx2" in cmd
    has_mavx512 = "-mavx512" in cmd
    has_mfma = "-mfma" in cmd

    max_simd = cpu_isa.get("max_simd_width_bits", 0)

    # ISA: AVX2 available but not enabled
    if cpu_isa.get("avx2") and not has_march_native and not has_mavx2:
        recs.append(FlagRecommendation(
            flag="-march=native",
            reason=f"Machine supports AVX2 ({max_simd}-bit SIMD) but no ISA flags present",
            impact="high",
            category="isa",
        ))

    # ISA: AVX-512 available but only AVX2 enabled
    if cpu_isa.get("avx512f") and has_mavx2 and not has_march_native and not has_mavx512:
        recs.append(FlagRecommendation(
            flag="-march=native",
            reason="Machine supports AVX-512 but only -mavx2 is set",
            impact="medium",
            category="isa",
        ))

    # ISA: FMA available but not enabled
    if cpu_isa.get("fma") and not has_march_native and not has_mfma:
        recs.append(FlagRecommendation(
            flag="-mfma",
            reason="Machine supports FMA (fused multiply-add) but -mfma is not set",
            impact="medium",
            category="isa",
        ))

    return recs


def _check_optimization_flags(cmd: str, program_type: str) -> list[FlagRecommendation]:
    """Check for missing optimization-level flags."""
    recs: list[FlagRecommendation] = []

    has_o3 = "-o3" in cmd
    has_o2 = "-o2" in cmd
    has_o1 = "-o1" in cmd
    has_flto = "-flto" in cmd

    # -O2 but not -O3
    if has_o2 and not has_o3:
        recs.append(FlagRecommendation(
            flag="-O3",
            reason="-O2 is set but -O3 enables more aggressive auto-vectorization and inlining",
            impact="medium",
            category="optimization",
        ))

    # No optimization level at all
    if not has_o3 and not has_o2 and not has_o1 and "-o0" not in cmd:
        # Only recommend if it looks like no -O flag at all
        if not any(f"-o{x}" in cmd for x in "0123s"):
            recs.append(FlagRecommendation(
                flag="-O3",
                reason="No optimization level set — -O3 enables auto-vectorization and inlining",
                impact="high",
                category="optimization",
            ))

    # No LTO
    if not has_flto and (has_o3 or has_o2):
        recs.append(FlagRecommendation(
            flag="-flto",
            reason="Link-time optimization can enable cross-TU inlining and dead code elimination",
            impact="low",
            category="optimization",
        ))

    return recs


def _check_debug_flags(cmd: str) -> list[FlagRecommendation]:
    """Check for debug-related flags that affect benchmark performance."""
    recs: list[FlagRecommendation] = []

    has_g = " -g " in f" {cmd} " or cmd.startswith("-g ") or cmd.endswith(" -g")
    has_gline = "-gline-tables-only" in cmd

    # -g without -gline-tables-only
    if has_g and not has_gline:
        recs.append(FlagRecommendation(
            flag="-gline-tables-only",
            reason="-g generates full debug info (larger binary, slower). Use -gline-tables-only for perf annotate with less overhead",
            impact="low",
            category="debug",
        ))

    # Sanitizers
    if "-fsanitize" in cmd:
        recs.append(FlagRecommendation(
            flag="(remove -fsanitize)",
            reason="Address/memory sanitizers add 2-5x overhead — remove for benchmarking",
            impact="high",
            category="debug",
        ))

    return recs


def _check_cuda_flags(cmd: str) -> list[FlagRecommendation]:
    """Check for CUDA-specific flags."""
    recs: list[FlagRecommendation] = []

    if "-arch=" not in cmd and "--gpu-architecture" not in cmd and "-gencode" not in cmd:
        recs.append(FlagRecommendation(
            flag="-arch=sm_XX",
            reason="No GPU architecture specified — nvcc defaults to an older arch, missing newer features",
            impact="medium",
            category="isa",
        ))

    return recs


def recommend_flags_from_profiling(
    build_cmd: str,
    profiler_summaries: dict,
    program_type: str,
    cpu_isa: dict | None = None,
) -> list[FlagRecommendation]:
    """Recommend compiler flags based on profiler output (dynamic feedback loop).

    Unlike recommend_build_flags() which only checks ISA, this examines actual
    profiling results to suggest flags that address observed bottlenecks.
    """
    recs: list[FlagRecommendation] = []
    cmd_lower = build_cmd.lower()

    if program_type not in ("cpp", "cuda"):
        return recs

    perf = profiler_summaries.get("linux_perf", {})
    tma = perf.get("tma", {})
    tma_l2 = perf.get("tma_level2", {})

    # --- Cache miss driven recommendations ---
    cache_miss_rate = perf.get("cache_miss_rate")
    if cache_miss_rate is not None and cache_miss_rate > 0.05:
        if "-fprefetch-loop-arrays" not in cmd_lower:
            recs.append(FlagRecommendation(
                flag="-fprefetch-loop-arrays",
                reason=f"Cache miss rate is {cache_miss_rate:.1%} — software prefetching may help",
                impact="medium",
                category="optimization",
            ))

    # --- TMA-driven recommendations ---
    frontend_bound = tma.get("frontend_bound_pct", 0)
    if frontend_bound > 25:
        if "-falign-functions" not in cmd_lower:
            recs.append(FlagRecommendation(
                flag="-falign-functions=32",
                reason=f"Frontend Bound is {frontend_bound:.0f}% — function alignment reduces fetch stalls",
                impact="medium",
                category="optimization",
            ))
        if "-falign-loops" not in cmd_lower:
            recs.append(FlagRecommendation(
                flag="-falign-loops=32",
                reason=f"Frontend Bound is {frontend_bound:.0f}% — loop alignment reduces I-cache pressure",
                impact="medium",
                category="optimization",
            ))

    bad_speculation = tma.get("bad_speculation_pct", 0)
    if bad_speculation > 20:
        branch_miss_rate = perf.get("branch_miss_rate")
        if branch_miss_rate and branch_miss_rate > 0.03:
            recs.append(FlagRecommendation(
                flag="-fprofile-generate / -fprofile-use",
                reason=f"Bad Speculation is {bad_speculation:.0f}% with {branch_miss_rate:.1%} branch miss rate — "
                       "PGO trains branch predictors on actual data patterns",
                impact="high",
                category="optimization",
            ))

    # --- TMA Level 2/3 driven ---
    dom_mem = tma_l2.get("dominant_memory_level")
    if dom_mem == "DRAM":
        if "-funroll-loops" not in cmd_lower:
            recs.append(FlagRecommendation(
                flag="-funroll-loops",
                reason="DRAM Bound — loop unrolling increases bytes-per-instruction, better hiding DRAM latency",
                impact="low",
                category="optimization",
            ))

    # --- Vectorization-driven (from compiler remarks) ---
    # If perf shows scalar code in hot loops, suggest march=native
    hotspots = perf.get("hotspots", [])
    annotated = perf.get("annotated_hotspots", [])
    if hotspots and "-march=native" not in cmd_lower:
        # Check if hot assembly lacks SIMD
        for hs in annotated[:3]:
            hot_lines = hs.get("hot_lines", [])
            if hot_lines and any(hl.get("pct", 0) > 10 for hl in hot_lines):
                if cpu_isa and cpu_isa.get("avx2"):
                    recs.append(FlagRecommendation(
                        flag="-march=native",
                        reason="Hot CPU functions detected — -march=native enables AVX2/FMA auto-vectorization",
                        impact="high",
                        category="isa",
                    ))
                    break

    # --- IPC-driven ---
    ipc = perf.get("ipc")
    if ipc is not None and ipc < 0.5 and "-funroll-loops" not in cmd_lower:
        recs.append(FlagRecommendation(
            flag="-funroll-loops",
            reason=f"IPC is very low ({ipc:.2f}) — loop unrolling can improve instruction-level parallelism",
            impact="medium",
            category="optimization",
        ))

    return recs
