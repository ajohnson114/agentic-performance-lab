from __future__ import annotations

from perflab.analyzers.bottleneck_types import AnalysisThresholds, BottleneckDiagnosis


def _analyze_perf(
    summary: dict,
    thresholds: AnalysisThresholds,
    cpu_count: int | None = None,
    program_type: str | None = None,
    source_hints: dict | None = None,
    compiler_remarks: list | None = None,
    cpu_isa: dict | None = None,
) -> list[BottleneckDiagnosis]:
    """Analyze Linux perf summary."""
    findings: list[BottleneckDiagnosis] = []

    ipc = summary.get("ipc")
    cache_miss_rate = summary.get("cache_miss_rate")
    branch_miss_rate = summary.get("branch_miss_rate")
    hotspots = summary.get("hotspots", [])

    if ipc is not None and ipc < thresholds.perf_ipc_low:
        # 0.5 IPC: modern x86 can retire 4-6 IPC; <0.5 means severe memory or pipeline stalls
        confidence = "high" if ipc < 0.5 else "medium"
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Low IPC ({ipc:.2f} instructions per cycle)",
            root_cause="Likely memory-bound or suffering from pipeline stalls",
            confidence=confidence,
            suggested_actions=[
                "Improve cache locality via loop tiling / blocking",
                "Prefetch data to reduce cache miss latency",
                "Reduce branch mispredictions with branchless code",
                "Consider SIMD vectorization for data-parallel work",
            ],
        ))

    if cache_miss_rate is not None and cache_miss_rate > thresholds.perf_cache_miss_rate_high:
        # 10%: one in ten accesses misses cache — each miss costs ~100ns (L3) to ~200ns (DRAM)
        confidence = "high" if cache_miss_rate > 0.10 else "medium"
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"High cache miss rate ({cache_miss_rate:.1%})",
            root_cause="Poor data locality or working set exceeds cache capacity",
            confidence=confidence,
            suggested_actions=[
                "Reorder loops for better spatial locality (e.g. i,k,j for matmul)",
                "Tile loops to fit working set in L1/L2 cache",
                "Use structure-of-arrays instead of array-of-structures",
                "Align data to cache line boundaries",
            ],
        ))

    if branch_miss_rate is not None and branch_miss_rate > thresholds.perf_branch_miss_rate_high:
        # 10%: each misprediction flushes 15-20 pipeline stages; 10% = severe throughput loss
        confidence = "high" if branch_miss_rate > 0.10 else "medium"
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"High branch misprediction rate ({branch_miss_rate:.1%})",
            root_cause="Frequent branch mispredictions cause pipeline flushes",
            confidence=confidence,
            suggested_actions=[
                "Use branchless algorithms (conditional moves, bitwise tricks)",
                "Reduce conditional logic in hot loops",
                "Consider profile-guided optimization (PGO)",
                "Sort data to improve branch prediction accuracy",
            ],
        ))

    if hotspots and hotspots[0].get("pct", 0) > thresholds.perf_hotspot_dominance_pct:
        hs = hotspots[0]
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"CPU hotspot: {hs['function']} ({hs['pct']:.0f}% of samples)",
            root_cause=f"Function '{hs['function']}' in {hs.get('module', '?')} dominates CPU time",
            confidence="high",
            suggested_actions=[
                f"Focus optimization on '{hs['function']}'",
                "Check if this function can be vectorized (SIMD)",
                "Profile with ncu/VTune for micro-architectural insights",
            ],
        ))

    # Rule A — Single-threaded execution
    cpus_utilized = summary.get("cpus_utilized")
    # Only fire for CPU-centric program types (not GPU-centric pytorch/jax/triton)
    single_thread_types = {"cpp", "cuda", "python"}
    if (
        cpus_utilized is not None
        and cpus_utilized < thresholds.perf_cpus_utilized_low
        and cpu_count is not None
        # 4 cores: below this, parallelism gains are marginal — not worth flagging
        and cpu_count >= 4
        and program_type in single_thread_types
    ):
        # 1.1 CPUs on 8+ core system: essentially single-threaded with huge parallelism headroom
        confidence = "high" if cpus_utilized < 1.1 and cpu_count >= 8 else "medium"
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Single-threaded execution on {cpu_count}-core system ({cpus_utilized:.1f} CPUs utilized)",
            root_cause="Program uses only one CPU core despite multi-core hardware being available",
            confidence=confidence,
            suggested_actions=[
                "Add OpenMP parallelization (#pragma omp parallel for)",
                "Use std::thread or C++17 parallel algorithms (std::execution::par)",
                "Compile with -fopenmp to enable OpenMP support",
                "Distribute independent loop iterations across threads",
            ],
        ))

    # Rule B — No SIMD vectorization
    if (
        source_hints is not None
        and source_hints.get("has_simd") is False
        and program_type == "cpp"
        and hotspots
        and hotspots[0].get("pct", 0) > thresholds.perf_hotspot_dominance_pct
    ):
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck="No SIMD vectorization in compute-intensive code",
            root_cause="Source code lacks SIMD intrinsics despite a dominant CPU hotspot",
            confidence="medium",
            suggested_actions=[
                "Use SIMD intrinsics (AVX2/AVX-512 on x86, NEON on ARM)",
                "Compile with -march=native -O3 to enable auto-vectorization",
                "Add auto-vectorization hints (#pragma GCC ivdep, __restrict__)",
                "Use __restrict__ pointers to help the compiler prove no aliasing",
            ],
        ))

    # Rule C — Vectorization width mismatch (from compiler remarks)
    if compiler_remarks and cpu_isa:
        max_simd = cpu_isa.get("max_simd_width_bits", 0)
        if max_simd > 0:
            for remark in compiler_remarks:
                if (
                    hasattr(remark, "category")
                    and remark.category == "vectorize"
                    and remark.status == "applied"
                    and remark.width is not None
                    and remark.width < max_simd
                    and max_simd / remark.width >= thresholds.vec_width_gap_ratio
                ):
                    gap = max_simd / remark.width
                    confidence = "high" if gap >= 4 else "medium"
                    findings.append(BottleneckDiagnosis(
                        rank=0,
                        bottleneck=f"Vectorization width mismatch at {remark.file}:{remark.line} ({remark.width}-bit vs {max_simd}-bit hardware)",
                        root_cause=f"Compiler vectorized at {remark.width}-bit but CPU supports {max_simd}-bit SIMD ({gap:.0f}x gap)",
                        confidence=confidence,
                        suggested_actions=[
                            "Compile with -march=native to enable wider SIMD",
                            "Add __restrict__ qualifiers to pointer parameters",
                            f"Ensure data is aligned to {max_simd // 8}-byte boundaries",
                        ],
                    ))
                    break  # one diagnosis is enough

    # Rule D — Missed vectorization at annotated hotspot
    annotated_hotspots = summary.get("annotated_hotspots", [])
    if compiler_remarks and annotated_hotspots:
        import os as _os
        hot_line_set: set[tuple[str, int]] = set()
        for hs in annotated_hotspots:
            for hl in hs.get("hot_lines", []):
                bn = _os.path.basename(hl.get("file", ""))
                if bn and hl.get("pct", 0) >= thresholds.perf_annotate_hot_line_pct:
                    for offset in range(-thresholds.cross_ref_hotspot_window, thresholds.cross_ref_hotspot_window + 1):
                        hot_line_set.add((bn, hl["line"] + offset))

        for remark in compiler_remarks:
            if (
                hasattr(remark, "category")
                and remark.category == "vectorize"
                and remark.status == "missed"
            ):
                remark_bn = _os.path.basename(remark.file)
                if (remark_bn, remark.line) in hot_line_set:
                    findings.append(BottleneckDiagnosis(
                        rank=0,
                        bottleneck=f"Missed vectorization at CPU hotspot {remark.file}:{remark.line}",
                        root_cause=f"Compiler could not vectorize hot loop: {remark.detail}",
                        confidence="high",
                        suggested_actions=[
                            "Add __restrict__ to pointer parameters to prove no aliasing",
                            "Restructure loop for unit-stride access",
                            "Add #pragma GCC ivdep or #pragma omp simd",
                        ],
                    ))
                    break  # one diagnosis is enough

    # TMA Level 2/3 bottleneck rules
    tma_l2 = summary.get("tma_level2", {})
    if tma_l2:
        dom_mem = tma_l2.get("dominant_memory_level")
        mem_bound = tma_l2.get("memory_bound_pct")
        core_bound = tma_l2.get("core_bound_pct")

        # Memory hierarchy bottleneck identification
        # 20%: Intel TMA guideline — below 20% memory isn't the primary bottleneck
        if mem_bound is not None and mem_bound > 20:
            _MEM_LEVEL_ACTIONS: dict[str, tuple[str, list[str]]] = {
                "L1": (
                    "L1 cache is the memory bottleneck — data working set exceeds L1 capacity",
                    [
                        "Tile inner loops to fit in L1 cache (32-64 KB)",
                        "Ensure innermost loop accesses are stride-1 (row-major for C/C++)",
                        "Reduce data footprint per iteration (fewer arrays, smaller types)",
                    ],
                ),
                "L2": (
                    "L2 cache is the memory bottleneck — tiles or working set spill from L2",
                    [
                        "Increase tile size to improve L1 reuse but ensure tiles still fit in L2 (256KB-1MB)",
                        "Use software prefetching (_mm_prefetch) for L2-to-L1 data movement",
                        "Consider blocking for L2 (outer tile for L2, inner tile for L1)",
                    ],
                ),
                "L3": (
                    "L3/LLC is the memory bottleneck — working set exceeds L3 capacity",
                    [
                        "Restructure algorithm for streaming access (process data in one pass)",
                        "Use non-temporal stores (_mm_stream_ps) for write-only data to bypass cache",
                        "Consider data compression or reduced precision to shrink working set",
                    ],
                ),
                "DRAM": (
                    "DRAM bandwidth is the memory bottleneck — data must be fetched from main memory",
                    [
                        "Reduce total data movement (operator fusion, lower precision, fewer passes)",
                        "Use streaming stores for write-only data to avoid read-for-ownership",
                        "Consider NUMA-aware allocation if on multi-socket system",
                        "Maximize SIMD width to improve bytes-per-instruction ratio",
                    ],
                ),
                "Store": (
                    "Store operations dominate memory stalls — write-back pressure or store-to-load forwarding issues",
                    [
                        "Use non-temporal stores for write-only arrays",
                        "Ensure stores are aligned to cache line boundaries (64 bytes)",
                        "Avoid store-to-load forwarding hazards (read after write to same address)",
                    ],
                ),
            }
            if dom_mem and dom_mem in _MEM_LEVEL_ACTIONS:
                root, actions = _MEM_LEVEL_ACTIONS[dom_mem]
                level_pct = tma_l2.get(f"{dom_mem.lower()}_bound_pct", mem_bound)
                findings.append(BottleneckDiagnosis(
                    rank=0,
                    bottleneck=f"TMA Level 3: {dom_mem} Bound ({level_pct:.0f}% of cycles)",
                    root_cause=root,
                    confidence="high" if level_pct > 30 else "medium",
                    suggested_actions=actions,
                ))

        # Core bound diagnosis
        # 25%: Intel TMA threshold — core-bound is meaningful when >25% of retiring slots
        if core_bound is not None and core_bound > 25:
            actions = [
                "Use SIMD intrinsics or ensure auto-vectorization is active",
                "Reduce instruction count in hot loops (FMA, loop unrolling)",
            ]
            # 10%: integer divide is 20-90 cycles on x86; 10% of core time is worth replacing
            if tma_l2.get("divider_pct") and tma_l2["divider_pct"] > 10:
                actions.append("Reduce division/modulo operations — replace with multiply-by-reciprocal")
            # 20%: execution ports saturated above this — suggests ALU-bound, not just frontend-limited
            if tma_l2.get("port_utilization_pct") and tma_l2["port_utilization_pct"] > 20:
                actions.append("Execution ports are saturated — try wider SIMD or reduce instruction count")
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"TMA Level 2: Core Bound ({core_bound:.0f}% of cycles)",
                root_cause="Execution units are the bottleneck — not enough ILP or SIMD utilization",
                confidence="medium",
                suggested_actions=actions,
            ))

    return findings


def _analyze_io_bottleneck(
    profiler_summaries: dict[str, dict],
    program_type: str,
    thresholds: AnalysisThresholds,
) -> list[BottleneckDiagnosis]:
    """Detect I/O and data loading bottlenecks from profiler data."""
    findings: list[BottleneckDiagnosis] = []

    # Check py-spy hotspots for data loading functions
    pyspy = profiler_summaries.get("pyspy", {})
    hotspots = pyspy.get("hotspots", [])
    io_keywords = {"dataloader", "collate", "fetch", "read", "decode", "pil", "loader", "_worker"}
    io_hotspot_pct = 0.0

    for hs in hotspots:
        func_lower = (hs.get("function", "") + " " + hs.get("location", "")).lower()
        if any(kw in func_lower for kw in io_keywords):
            io_hotspot_pct += hs.get("pct", 0.0)

    if io_hotspot_pct > thresholds.io_hotspot_pct_high:
        # 40%: nearly half the samples in data loading — GPU is clearly starved for data
        confidence = "high" if io_hotspot_pct > 40 else "medium"
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"I/O-bound ({io_hotspot_pct:.0f}% of samples in data loading)",
            root_cause="Data loading and preprocessing dominate execution time",
            confidence=confidence,
            suggested_actions=[
                "Increase DataLoader num_workers for parallel data loading",
                "Enable pin_memory=True for faster CPU-to-GPU transfers",
                "Use persistent_workers=True to avoid worker restart overhead",
                "Consider memory-mapped datasets or pre-loaded tensors",
            ],
        ))

    # Check nsys: if most time is NOT in CUDA kernels AND NOT in CUDA API
    nsys = profiler_summaries.get("nsys", {})
    kernel_time = nsys.get("cuda_kernel_time_ms")
    api_overhead = nsys.get("api_overhead_ms")
    duration = nsys.get("duration_s")

    if kernel_time is not None and duration is not None and duration > 0:
        gpu_ms = kernel_time + (api_overhead or 0)
        total_ms = duration * 1000.0
        non_gpu_fraction = 1.0 - (gpu_ms / total_ms) if total_ms > 0 else 0
        if non_gpu_fraction > thresholds.io_non_gpu_fraction_high and program_type in ("pytorch", "jax"):
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"CPU/IO-bound ({non_gpu_fraction:.0%} time outside GPU)",
                root_cause="Majority of time is spent outside GPU kernels and API — likely data loading or CPU preprocessing",
                confidence="medium",
                suggested_actions=[
                    "Profile data loading separately to identify the bottleneck",
                    "Increase DataLoader num_workers",
                    "Move preprocessing to GPU (torchvision transforms on GPU)",
                    "Use async data prefetching",
                ],
            ))

    return findings
