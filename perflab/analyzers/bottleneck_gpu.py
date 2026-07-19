from __future__ import annotations

from perflab.analyzers.bottleneck_types import AnalysisThresholds, BottleneckDiagnosis


def _analyze_ncu(summary: dict, thresholds: AnalysisThresholds) -> list[BottleneckDiagnosis]:
    """Analyze NVIDIA Nsight Compute (ncu) summary."""
    findings: list[BottleneckDiagnosis] = []

    # Use dominant kernel name for targeted diagnostics when available
    dominant = summary.get("dominant_kernel", {})
    dk_name = dominant.get("name")

    sm_util = summary.get("sm_utilization_pct")
    mem_throughput = summary.get("memory_throughput_pct")
    occupancy = summary.get("achieved_occupancy_pct")

    # Per-kernel: determine if dominant kernel is compute- or memory-bound
    dk_mem = dominant.get("memory_throughput_pct")
    dk_compute = dominant.get("compute_throughput_pct")
    if dk_mem is not None and dk_compute is not None and dk_name:
        if dk_mem > thresholds.ncu_mem_throughput_high and dk_compute < thresholds.ncu_compute_low:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Kernel '{dk_name}' is memory-bound (mem={dk_mem:.0f}%, compute={dk_compute:.0f}%)",
                root_cause="High memory throughput with low compute throughput indicates bandwidth saturation",
                confidence="high",
                suggested_actions=[
                    "Use shared memory tiling to reduce global memory traffic",
                    "Improve data reuse via blocking / loop tiling",
                    "Consider using lower-precision datatypes (FP16, TF32, INT8)",
                    "Ensure coalesced memory access patterns",
                ],
            ))
        elif dk_compute > thresholds.ncu_compute_throughput_high and dk_mem < thresholds.ncu_mem_low:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Kernel '{dk_name}' is compute-bound (compute={dk_compute:.0f}%, mem={dk_mem:.0f}%)",
                root_cause="High compute throughput with low memory throughput indicates ALU saturation",
                confidence="high",
                suggested_actions=[
                    "Use lower-precision datatypes (FP16/TF32) for higher throughput",
                    "Reduce unnecessary computation or use approximations",
                    "Increase occupancy to hide instruction latency",
                ],
            ))

    if sm_util is not None and sm_util < thresholds.ncu_sm_util_low:
        confidence = "high" if sm_util < thresholds.ncu_sm_util_critical else "medium"
        label = f"in kernel '{dk_name}' " if dk_name else ""
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Low SM utilization {label}({sm_util:.0f}%)",
            root_cause="Insufficient parallelism or small kernel launches",
            confidence=confidence,
            suggested_actions=[
                "Increase problem size or batch size to expose more parallelism",
                "Fuse small kernels to reduce launch overhead",
                "Check for serializing dependencies between kernel launches",
            ],
        ))

    if mem_throughput is not None and mem_throughput > thresholds.ncu_mem_bound_high:
        confidence = "high" if mem_throughput > thresholds.ncu_mem_bound_critical else "medium"
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Memory-bound ({mem_throughput:.0f}% memory throughput)",
            root_cause="Kernel is limited by memory bandwidth, not compute",
            confidence=confidence,
            suggested_actions=[
                "Use shared memory tiling to reduce global memory traffic",
                "Improve data reuse via blocking / loop tiling",
                "Consider using lower-precision datatypes (FP16, TF32, INT8)",
                "Ensure coalesced memory access patterns",
            ],
        ))

    if occupancy is not None and occupancy < thresholds.ncu_occupancy_low:
        # 25%: below this, fewer than 1 in 4 warp slots are active — severe latency hiding gap
        confidence = "medium" if occupancy > 25 else "high"
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Low occupancy ({occupancy:.0f}%)",
            root_cause="Too many registers or too much shared memory per block limits active warps",
            confidence=confidence,
            suggested_actions=[
                "Reduce register pressure (fewer local variables, simpler expressions)",
                "Reduce shared memory per block",
                "Tune block dimensions to improve occupancy",
                "Use __launch_bounds__ to hint register usage to compiler",
            ],
        ))

    # High register pressure on dominant kernel
    dk_regs = dominant.get("registers_per_thread")
    if dk_regs is not None and dk_regs > thresholds.ncu_regs_per_thread_high and dk_name:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"High register pressure in '{dk_name}' ({dk_regs} regs/thread)",
            root_cause="Excessive register usage limits occupancy and active warps",
            confidence="medium",
            suggested_actions=[
                f"Add __launch_bounds__(max_threads, min_blocks) to kernel '{dk_name}'",
                "Reduce local variables and simplify expressions",
                "Consider using shared memory instead of registers for some data",
            ],
        ))

    # Low branch efficiency — control divergence
    branch_eff = summary.get("branch_efficiency_pct")
    if branch_eff is not None and branch_eff < thresholds.ncu_branch_efficiency_low:
        # 60%: below this, >40% of branches diverge — a full warp-width serialization penalty
        confidence = "high" if branch_eff < 60.0 else "medium"
        label = f"in kernel '{dk_name}' " if dk_name else ""
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Low branch efficiency {label}({branch_eff:.0f}%)",
            root_cause="Divergent branches within warps cause serialized execution of both paths",
            confidence=confidence,
            suggested_actions=[
                "Refactor conditionals to be warp-uniform (all threads in a warp take the same path)",
                "Use predication or branchless arithmetic instead of if/else",
                "Sort or partition input data so adjacent threads follow the same branch",
                "Replace divergent branches with arithmetic masks (e.g., multiply by condition)",
            ],
        ))

    # Low warp execution efficiency
    warp_exec_eff = summary.get("warp_execution_efficiency_pct")
    if warp_exec_eff is not None and warp_exec_eff < thresholds.ncu_warp_exec_efficiency_low:
        # 60%: below this, >12 of 32 SIMT lanes are wasted per instruction issue
        confidence = "high" if warp_exec_eff < 60.0 else "medium"
        label = f"in kernel '{dk_name}' " if dk_name else ""
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Low warp execution efficiency {label}({warp_exec_eff:.0f}%)",
            root_cause="Many threads per warp are predicated off or inactive, wasting SIMT lanes",
            confidence=confidence,
            suggested_actions=[
                "Apply thread coarsening — assign more work per thread to reduce inactive lanes",
                "Pad work dimensions to warp-size multiples (32 threads)",
                "Restructure computation to minimize predicated-off threads",
                "Ensure grid dimensions evenly divide the problem size",
            ],
        ))

    # Low Tensor Core utilization on capable GPUs
    tc_util = summary.get("tensor_core_utilization_pct")
    dk_tc = dominant.get("tensor_core_utilization_pct")
    # Use per-kernel value if available, fall back to aggregate
    tc_val = dk_tc if dk_tc is not None else tc_util
    if tc_val is not None and tc_val < thresholds.ncu_tc_util_low:
        # 10%: below this, Tensor Cores are essentially unused — 8-16x throughput left on the table
        confidence = "high" if tc_val < 10.0 else "medium"
        label = f"in kernel '{dk_name}' " if dk_name else ""
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Low Tensor Core utilization {label}({tc_val:.0f}%)",
            root_cause="Kernel is using CUDA cores instead of Tensor Cores, leaving significant throughput on the table",
            confidence=confidence,
            suggested_actions=[
                "Use half-precision (FP16/BF16) inputs with WMMA or mma.sync intrinsics",
                "Align matrix dimensions to Tensor Core tile sizes (multiples of 16 for FP16, 8 for TF32)",
                "Enable TF32 for FP32 inputs on Ampere+ (nvcc default, or set CUBLAS_TF32=1)",
                "Use cuBLAS or CUTLASS for drop-in Tensor Core GEMM",
                "Ensure shared memory tiles match WMMA fragment layout (16x16x16 for FP16)",
            ],
        ))

    # Tensor Core capable but no TC metrics detected (using CUDA cores entirely)
    if tc_val is None and dk_compute is not None and dk_compute > thresholds.ncu_compute_throughput_high:
        # Compute-bound on CUDA cores — suggest Tensor Cores
        label = f"in kernel '{dk_name}' " if dk_name else ""
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Compute-bound on CUDA cores {label}— Tensor Cores not engaged",
            root_cause="Kernel saturates CUDA core ALUs but does not use Tensor Cores, "
                       "which offer 4-16x higher throughput for matrix operations",
            confidence="medium",
            suggested_actions=[
                "Switch to FP16/BF16 inputs and use WMMA or mma.sync for Tensor Core access",
                "Use TF32 precision (Ampere+) for transparent Tensor Core acceleration of FP32 matmuls",
                "Replace scalar FMA loops with cooperative matrix operations (wmma::mma_sync)",
                "Consider cuBLAS or CUTLASS for automatic Tensor Core dispatch",
            ],
        ))

    # Dominant warp stall reason diagnosis
    stall_reason = summary.get("dominant_stall_reason")
    stall_pct = summary.get("dominant_stall_pct")
    if stall_reason and stall_pct is not None and stall_pct > thresholds.ncu_stall_pct_high:
        # 50%: majority of warp cycles are stalled on one cause — strong root-cause signal
        confidence = "high" if stall_pct > 50.0 else "medium"
        label = f"in kernel '{dk_name}' " if dk_name else ""
        # Map stall reasons to human-readable diagnosis and actions
        _STALL_DIAGNOSIS: dict[str, tuple[str, list[str]]] = {
            "long_scoreboard": (
                "Warps stall waiting for global memory loads to complete",
                [
                    "Increase data reuse with shared memory tiling to reduce global loads",
                    "Use cp.async (Ampere+) or TMA (Hopper) for asynchronous global→shared copies",
                    "Add software pipelining — prefetch next tile while computing current tile",
                    "Increase occupancy to give the scheduler more warps to hide latency",
                ],
            ),
            "short_scoreboard": (
                "Warps stall waiting for shared memory or L1 cache results",
                [
                    "Reduce shared memory bank conflicts — pad arrays (e.g., float s[32][33])",
                    "Increase data reuse to reduce shared memory traffic",
                    "Restructure access patterns for conflict-free shared memory reads",
                ],
            ),
            "barrier": (
                "Warps stall at __syncthreads() barriers waiting for other warps",
                [
                    "Reduce the frequency of __syncthreads() calls",
                    "Use warp-level primitives (__shfl, cooperative_groups) for intra-warp sync",
                    "Restructure algorithm to reduce inter-warp dependencies",
                ],
            ),
            "lg_throttle": (
                "Global memory pipeline is full — too many outstanding loads/stores",
                [
                    "Reduce the number of concurrent global memory accesses per warp",
                    "Improve memory access coalescing to reduce transaction count",
                    "Use shared memory to stage data instead of repeated global loads",
                ],
            ),
            "memory_throttle": (
                "Memory subsystem is congested — backpressure from DRAM/L2",
                [
                    "Reduce total memory traffic with shared memory tiling",
                    "Improve access locality to reduce L2 pressure",
                    "Use lower precision datatypes to halve memory bandwidth demand",
                ],
            ),
            "not_selected": (
                "Warps are ready but not scheduled — scheduler contention",
                [
                    "This indicates good latency hiding — the scheduler has many eligible warps",
                    "If performance is still low, the bottleneck is elsewhere (compute or memory)",
                ],
            ),
            "math_pipe_throttle": (
                "Compute pipeline is full — instruction throughput limited",
                [
                    "Use Tensor Cores (WMMA/mma.sync) for matrix operations — higher throughput per instruction",
                    "Reduce unnecessary computation or use approximations",
                    "Consider lower precision (FP16/BF16/TF32) for higher compute throughput",
                ],
            ),
            "mio_throttle": (
                "Memory I/O queue is full — too many pending memory instructions",
                [
                    "Reduce the number of concurrent memory operations",
                    "Batch memory accesses with vectorized loads (float4)",
                    "Use shared memory to reduce global memory pressure",
                ],
            ),
            "gmma": (
                "Warps stall waiting for Warp Group MMA (WGMMA/HGMMA) completion on Hopper",
                [
                    "This is expected for compute-heavy GEMM on Hopper — indicates Tensor Cores are in use",
                    "Overlap WGMMA with TMA prefetching using warp specialization (producer/consumer warps)",
                    "Increase the number of pipeline stages to hide WGMMA latency",
                    "If Stall GMMA + Stall Barrier dominate, the kernel is well-pipelined — focus on tile sizes",
                ],
            ),
        }
        root, actions = _STALL_DIAGNOSIS.get(stall_reason, (
            f"Warps frequently stall due to '{stall_reason}'",
            ["Profile with ncu GUI to inspect stall reasons in detail"],
        ))
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Dominant warp stall: {stall_reason} {label}({stall_pct:.0f}%)",
            root_cause=root,
            confidence=confidence,
            suggested_actions=actions,
        ))

    # Shared memory bank conflicts
    bank_conflicts = summary.get("bank_conflicts")
    dk_bc = dominant.get("bank_conflicts")
    bc_val = dk_bc if dk_bc is not None else bank_conflicts
    if bc_val is not None and bc_val > thresholds.ncu_bank_conflicts_high:
        # 1000: at this scale, bank conflicts are serializing many warps (>100 per SM typical)
        confidence = "high" if bc_val > 1000 else "medium"
        label = f"in kernel '{dk_name}' " if dk_name else ""
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Shared memory bank conflicts {label}({bc_val:.0f} conflicts)",
            root_cause="Multiple threads in a warp access the same shared memory bank, "
                       "serializing accesses and reducing shared memory throughput by up to 32x",
            confidence=confidence,
            suggested_actions=[
                "Pad shared memory arrays to avoid bank conflicts (e.g., float s[32][33] instead of s[32][32])",
                "Use swizzled memory layouts for Tensor Core fragments",
                "Restructure access patterns so adjacent threads access different banks",
                "Consider using warp-level primitives (__shfl) instead of shared memory for small data",
            ],
        ))

    # Uncoalesced global memory access
    sectors_per_req = summary.get("sectors_per_request")
    dk_spr = dominant.get("sectors_per_request")
    spr_val = dk_spr if dk_spr is not None else sectors_per_req
    if spr_val is not None and spr_val > thresholds.ncu_sectors_per_request_high:
        # 8.0: 8x ideal means each load generates 8 cache-line transactions — severe coalescing failure
        confidence = "high" if spr_val > 8.0 else "medium"
        label = f"in kernel '{dk_name}' " if dk_name else ""
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Uncoalesced global memory access {label}({spr_val:.1f} sectors/request, ideal is 1.0)",
            root_cause="Adjacent threads are accessing non-adjacent memory locations, "
                       "causing each warp load to generate multiple memory transactions",
            confidence=confidence,
            suggested_actions=[
                "Ensure adjacent threads access adjacent memory addresses (stride-1 access pattern)",
                "Transpose data layout so the innermost dimension is contiguous for warp access",
                "Use Structure of Arrays (SoA) instead of Array of Structures (AoS)",
                "Use vectorized loads (float4) to align transactions to 128-byte boundaries",
            ],
        ))

    # Occupancy limiter diagnosis (actionable "why" for low occupancy)
    if occupancy is not None and occupancy < thresholds.ncu_occupancy_low:
        occ_limit_regs = dominant.get("occupancy_limit_registers_pct")
        occ_limit_smem = dominant.get("occupancy_limit_shared_mem_pct")
        occ_limit_block = dominant.get("occupancy_limit_block_pct")
        # Find the tightest limiter
        limiters: list[tuple[str, float | None]] = [
            ("registers", occ_limit_regs),
            ("shared memory", occ_limit_smem),
            ("block size", occ_limit_block),
        ]
        valid_limiters = [(name, val) for name, val in limiters if val is not None]
        if valid_limiters:
            # The tightest limiter has the lowest theoretical occupancy
            tightest = min(valid_limiters, key=lambda x: x[1])  # type: ignore[arg-type]
            limiter_name, limiter_val = tightest
            label = f"in kernel '{dk_name}' " if dk_name else ""
            _LIMITER_ACTIONS: dict[str, list[str]] = {
                "registers": [
                    "Add __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM) to limit register usage",
                    "Reduce local variables and simplify expressions in the kernel",
                    "Spill some data to shared memory instead of registers",
                    "Use -maxrregcount=N nvcc flag to cap register usage",
                ],
                "shared memory": [
                    "Reduce shared memory allocation per block",
                    "Use dynamic shared memory to allow the runtime to balance allocation",
                    "Split large shared memory arrays across multiple passes",
                    "Consider using L1 cache instead of shared memory for read-only data",
                ],
                "block size": [
                    "Increase threads per block (use multiples of 32, aim for 128-256)",
                    "Ensure block dimensions are not limiting active warps per SM",
                    "Use the CUDA occupancy calculator to find optimal block size",
                ],
            }
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Occupancy limited by {limiter_name} {label}(theoretical max: {limiter_val:.0f}%)",
                root_cause=f"The {limiter_name} usage per block prevents the SM from scheduling "
                           f"enough concurrent warps to hide latency",
                confidence="medium",
                suggested_actions=_LIMITER_ACTIONS.get(limiter_name, [
                    "Profile with ncu occupancy section to identify the specific limiter",
                ]),
            ))

    # FP64 on consumer GPU (1/64th throughput on most consumer GPUs)
    inst_mix = summary.get("instruction_mix", {})
    fp64_util = inst_mix.get("fp64")
    # 10%: FP64 throughput is 1/64th of FP32 on consumer GPUs — even 10% usage is a red flag
    if fp64_util is not None and fp64_util > 10.0:
        label = f"in kernel '{dk_name}' " if dk_name else ""
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Significant FP64 usage {label}({fp64_util:.0f}% of pipe utilization)",
            root_cause="FP64 throughput on consumer GPUs (RTX series) is 1/64th of FP32. "
                       "Even on data-center GPUs (A100), FP64 is 1/2 of FP32",
            # 30%: nearly a third of pipe time is FP64 — major throughput loss
            confidence="high" if fp64_util > 30.0 else "medium",
            suggested_actions=[
                "Convert double-precision (FP64) operations to single-precision (FP32) where accuracy permits",
                "Use FP32 accumulation with FP16/BF16 inputs for maximum throughput",
                "On data-center GPUs (A100, H100), FP64 is viable but still slower than FP32/TF32",
            ],
        ))

    # Register spilling to local memory
    local_mem = summary.get("local_memory_bytes")
    dk_lm = dominant.get("local_memory_bytes")
    lm_val = dk_lm if dk_lm is not None else local_mem
    if lm_val is not None and lm_val > 0:
        # 1024 bytes: at 1KB+, spills are multiple cache lines per thread — significant DRAM traffic
        confidence = "high" if lm_val > 1024 else "medium"
        label = f"in kernel '{dk_name}' " if dk_name else ""
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Register spilling to local memory {label}({lm_val:.0f} bytes)",
            root_cause="Kernel uses more registers than available, causing spills to slow local memory "
                       "(backed by L1/L2/DRAM). Each spill adds memory traffic and latency",
            confidence=confidence,
            suggested_actions=[
                "Add __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM) to cap register usage",
                "Use -maxrregcount=N nvcc flag to limit registers per thread",
                "Reduce local variables — reuse registers, simplify expressions",
                "Move frequently-accessed data to shared memory instead of local variables",
            ],
        ))

    # GPU-side multi-cache-level diagnosis (L1 -> L2 -> DRAM)
    # Synthesize L1/L2 hit rates + memory throughput into a cache hierarchy bottleneck
    dk_l1_hr = dominant.get("l1_hit_rate")
    dk_l2_hr = dominant.get("l2_hit_rate")
    dk_mem_tp = dominant.get("memory_throughput_pct")

    if dk_l1_hr is not None and dk_l2_hr is not None and dk_mem_tp is not None:
        label = f"in kernel '{dk_name}' " if dk_name else ""

        # 50% L1 hit rate: below this, majority of accesses miss L1 and go to L2 or DRAM
        if dk_l1_hr < 50 and dk_l2_hr is not None:
            # Low L1 hit rate — data working set exceeds L1 (shared memory + L1 cache)
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"GPU L1 cache bottleneck {label}(L1 hit: {dk_l1_hr:.0f}%, L2 hit: {dk_l2_hr:.0f}%)",
                root_cause="L1/TEX cache hit rate is low — tiles or working set exceed L1 capacity "
                           "(typically 128-256 KB per SM shared between shared memory and L1 cache)",
                confidence="high" if dk_l1_hr < 30 else "medium",
                suggested_actions=[
                    "Reduce tile size to fit in shared memory + L1 cache per SM",
                    "Increase data reuse per load — each byte loaded from L2 should be used multiple times",
                    "Use shared memory explicitly instead of relying on L1 cache — shared memory has guaranteed capacity",
                    "Ensure coalesced loads to maximize bytes per L1 cache line fill",
                ],
            ))
        # L2 < 50% with mem throughput > 40%: L2 misses are generating real DRAM traffic
        elif dk_l1_hr >= 50 and dk_l2_hr < 50 and dk_mem_tp > 40:
            # Good L1 but low L2 — tiles fit in L1 but aggregate working set exceeds L2
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"GPU L2 cache bottleneck {label}(L1 hit: {dk_l1_hr:.0f}%, L2 hit: {dk_l2_hr:.0f}%)",
                root_cause="L1 cache is effective but L2 hit rate is low — the aggregate working set "
                           "across all SMs exceeds L2 capacity. Data frequently spills to DRAM.",
                confidence="high" if dk_l2_hr < 30 else "medium",
                suggested_actions=[
                    "Reduce total data footprint per kernel launch — smaller tiles reduce L2 pressure",
                    "Improve temporal locality — reuse L2-cached data before it's evicted",
                    "Use L2 cache persistence hints (cudaAccessPolicyWindow on Ampere+) for frequently accessed data",
                    "Apply tile index swizzling to reduce L2 sector conflicts across CTAs",
                    "Consider reducing precision (FP32→FP16) to halve the working set size",
                ],
            ))
        # mem throughput > 70% with good cache hits: DRAM bandwidth wall despite efficient caching
        elif dk_l1_hr >= 50 and dk_l2_hr >= 50 and dk_mem_tp > 70:
            # Good cache hit rates but still high memory throughput — DRAM bandwidth saturated
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"GPU DRAM bandwidth saturated {label}(L1 hit: {dk_l1_hr:.0f}%, L2 hit: {dk_l2_hr:.0f}%, mem throughput: {dk_mem_tp:.0f}%)",
                root_cause="Cache hierarchy is working efficiently but DRAM bandwidth is still the bottleneck — "
                           "the kernel moves more data than the caches can absorb",
                confidence="medium",
                suggested_actions=[
                    "Reduce total bytes moved — fuse operators to eliminate intermediate DRAM writes",
                    "Reduce precision (FP32→FP16/BF16) to halve memory traffic",
                    "Increase compute-to-memory ratio — do more FLOPs per byte loaded (larger tiles, more reuse)",
                    "On Ampere+: use async copy (cp.async) to overlap data movement with compute",
                ],
            ))

    return findings


def _analyze_nsys(summary: dict, thresholds: AnalysisThresholds) -> list[BottleneckDiagnosis]:
    """Analyze NVIDIA Nsight Systems (nsys) summary."""
    findings: list[BottleneckDiagnosis] = []

    kernel_time = summary.get("cuda_kernel_time_ms")
    duration = summary.get("duration_s")
    api_overhead = summary.get("api_overhead_ms")

    if kernel_time is not None and duration is not None and duration > 0:
        gpu_fraction = (kernel_time / 1000.0) / duration
        if gpu_fraction < thresholds.nsys_gpu_fraction_low:
            # 0.3: below 30% GPU time, the GPU is idle >70% — clearly CPU-bound
            confidence = "high" if gpu_fraction < 0.3 else "medium"
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"CPU-bound ({gpu_fraction:.0%} time in GPU kernels)",
                root_cause="GPU is idle most of the time; CPU work or launch overhead dominates",
                confidence=confidence,
                suggested_actions=[
                    "Move more computation to GPU",
                    "Use CUDA graphs to reduce launch overhead",
                    "Overlap CPU and GPU work with async operations",
                    "Increase batch size to amortize CPU overhead",
                ],
            ))

    if api_overhead is not None and duration is not None and duration > 0:
        overhead_fraction = (api_overhead / 1000.0) / duration
        if overhead_fraction > thresholds.nsys_api_overhead_high:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"High CUDA API overhead ({overhead_fraction:.0%} of total time)",
                root_cause="Excessive CUDA API calls (malloc, memcpy, launch) relative to useful work",
                confidence="medium",
                suggested_actions=[
                    "Use memory pools (cudaMallocAsync) to reduce allocation overhead",
                    "Batch small operations into larger kernels",
                    "Use CUDA graphs for repeated launch sequences",
                ],
            ))

    # Single kernel dominating GPU time
    top_kernels = summary.get("top_kernels", [])
    if top_kernels and top_kernels[0].get("pct", 0) > thresholds.nsys_kernel_dominance_pct:
        k = top_kernels[0]
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Single kernel dominates GPU time ('{k['name']}' at {k['pct']:.0f}%)",
            root_cause="One kernel accounts for the vast majority of GPU execution time",
            confidence="high",
            suggested_actions=[
                f"Focus optimization efforts on kernel '{k['name']}'",
                "Profile this kernel with ncu for detailed metrics",
                "Consider algorithmic improvements or kernel fusion",
            ],
        ))

    # Kernel launch overhead from gap analysis
    avg_gap = summary.get("avg_kernel_gap_us")
    if avg_gap is not None and avg_gap > thresholds.nsys_kernel_gap_us:
        # 200us: typical kernel launch is 5-20us; 200us gaps indicate pipeline bubbles
        confidence = "high" if avg_gap > 200 else "medium"
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Kernel launch overhead (avg {avg_gap:.0f} us gap between kernels)",
            root_cause="Significant idle time between consecutive GPU kernel launches",
            confidence=confidence,
            suggested_actions=[
                "Use CUDA graphs to batch launch sequences",
                "Fuse small kernels to reduce launch count",
                "Overlap CPU work with GPU execution using streams",
            ],
        ))

    # Data transfer bottleneck
    memcpy_list = summary.get("memcpy", [])
    if memcpy_list and kernel_time and kernel_time > 0:
        htod_ms = sum(m["total_ms"] for m in memcpy_list if m["direction"] == "HtoD")
        if htod_ms > kernel_time * thresholds.nsys_transfer_ratio:
            pct = htod_ms / kernel_time * 100
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Data transfer bottleneck (HtoD copies = {pct:.0f}% of kernel time)",
                root_cause="Host-to-device memory transfers consume significant time relative to GPU compute",
                confidence="medium",
                suggested_actions=[
                    "Use pinned (page-locked) host memory for faster transfers",
                    "Overlap transfers with computation using CUDA streams",
                    "Reduce transfer volume by keeping data on GPU longer",
                    "Use unified memory or zero-copy memory where appropriate",
                ],
            ))

    # Hot-path cudaMalloc/cudaFree detection
    top_api = summary.get("top_api_calls", [])
    for api in top_api:
        api_name = api.get("name", "")
        api_pct = api.get("pct", 0)
        # 5%: cudaMalloc/cudaFree each take 100-1000us and sync all streams; 5% of API time is significant
        if ("cudaMalloc" in api_name or "cudaFree" in api_name) and api_pct > 5.0:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Memory allocation in hot path ('{api_name}' = {api_pct:.0f}% of API time)",
                root_cause="cudaMalloc/cudaFree synchronize all streams and are extremely expensive "
                           "when called repeatedly. Each call can take 100-1000us",
                # 15%: at this level, malloc/free likely in a hot loop — blocking optimization
                confidence="high" if api_pct > 15.0 else "medium",
                suggested_actions=[
                    "Use cudaMallocAsync / cudaMemPool for stream-ordered allocation (100-1000x faster)",
                    "Pre-allocate buffers and reuse them across iterations",
                    "Use a memory pool allocator (e.g., RAPIDS RMM, PyTorch caching allocator)",
                    "Move allocations outside the hot loop — allocate once, reuse many times",
                ],
            ))
            break  # One finding is enough

    return findings


def _analyze_gpu_attribution(
    nsys_summary: dict,
    perf_summary: dict | None,
    thresholds: AnalysisThresholds,
) -> list[BottleneckDiagnosis]:
    """Generate bottleneck diagnoses from GPU attribution data."""
    findings: list[BottleneckDiagnosis] = []

    try:
        from perflab.analyzers.gpu_attribution import compute_attribution_ranking
    except ImportError:
        return findings

    # Note: no gate on cpu_gpu_correlations here -- nsys can silently omit it
    # (sqlite3.OperationalError, empty rows) while top_kernels/per_stream_gaps/
    # stream_utilization/memcpy are still populated independently.
    # compute_attribution_ranking() defaults correlations to [] internally and
    # still produces gpu-kernel/pipeline-stall findings from that other data.
    ranking = compute_attribution_ranking(nsys_summary, perf_summary)
    for entry in ranking[:3]:  # top 3 attribution entries
        # 20%: a kernel taking >20% of total GPU time is a dominant optimization target
        if entry.category == "gpu-kernel" and entry.gpu_pct > 20:
            suggestions = list(entry.suggestions)
            # 50us: typical cudaLaunchKernel takes 5-10us; >50us signals driver contention
            if entry.launch_overhead_us and entry.launch_overhead_us > 50:
                findings.append(BottleneckDiagnosis(
                    rank=0,
                    bottleneck=f"Kernel '{entry.name}' dominates GPU ({entry.gpu_pct:.0f}%) with high launch overhead ({entry.launch_overhead_us:.0f} us)",
                    root_cause="Kernel launch overhead is significant relative to execution time",
                    confidence="high",
                    suggested_actions=suggestions or ["Use CUDA graphs to batch launches", "Profile with ncu for kernel-level optimization"],
                ))

        elif entry.category == "pipeline-stall":
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=entry.diagnosis,
                root_cause="GPU stream is idle between kernel launches",
                confidence="medium",
                suggested_actions=entry.suggestions or ["Overlap computation across streams", "Use CUDA graphs"],
            ))

    return findings


def _analyze_metal(summary: dict, thresholds: AnalysisThresholds) -> list[BottleneckDiagnosis]:
    """Analyze Apple Metal trace summary."""
    findings: list[BottleneckDiagnosis] = []

    gpu_time = summary.get("gpu_time_total_ms")
    duration = summary.get("duration_s")

    if gpu_time is not None and duration is not None and duration > 0:
        gpu_fraction = (gpu_time / 1000.0) / duration
        if gpu_fraction < thresholds.metal_gpu_fraction_low:
            confidence = "high" if gpu_fraction < 0.3 else "medium"
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"GPU underutilized ({gpu_fraction:.0%} time in GPU work)",
                root_cause="CPU-side overhead or synchronization prevents GPU from staying busy",
                confidence=confidence,
                suggested_actions=[
                    "Reduce CPU-GPU synchronization points",
                    "Batch more work per command buffer",
                    "Use triple buffering to overlap CPU and GPU work",
                    "Increase batch size to amortize per-dispatch overhead",
                ],
            ))

    # Blit (memory transfer) bottleneck
    by_type = summary.get("submissions_by_type", {})
    blit_ms = by_type.get("blit", {}).get("total_ms", 0)
    if gpu_time and gpu_time > 0 and blit_ms > gpu_time * thresholds.metal_blit_ratio:
        pct = blit_ms / gpu_time * 100
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Memory transfer bottleneck (blit = {pct:.0f}% of GPU time)",
            root_cause="Blit (memory copy) submissions consume significant GPU time",
            confidence="medium",
            suggested_actions=[
                "Reduce CPU-GPU data movement by keeping data on GPU longer",
                "Use shared memory or managed buffers where possible",
                "Batch transfers to reduce per-transfer overhead",
            ],
        ))

    # GPU idle time
    gpu_idle_pct = summary.get("gpu_idle_pct")
    if gpu_idle_pct is not None and gpu_idle_pct > thresholds.metal_gpu_idle_pct_high:
        confidence = "high" if gpu_idle_pct > 50 else "medium"
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"GPU idle {gpu_idle_pct:.0f}% of trace time",
            root_cause="Gaps between GPU submissions indicate CPU-side bottleneck or insufficient work batching",
            confidence=confidence,
            suggested_actions=[
                "Increase batch size to give GPU more work per dispatch",
                "Use triple buffering to overlap CPU and GPU work",
                "Reduce command buffer submission overhead",
            ],
        ))

    # Dominant submission
    top_subs = summary.get("top_submissions", [])
    if top_subs and gpu_time and gpu_time > 0:
        top_ms = top_subs[0].get("gpu_time_ms", 0)
        if top_ms > gpu_time * thresholds.metal_submission_dominance:
            pct = top_ms / gpu_time * 100
            label = top_subs[0].get("label", "(unnamed)")
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Dominant submission: '{label}' ({pct:.0f}% of GPU time)",
                root_cause=f"Single GPU submission '{label}' accounts for most GPU execution time",
                confidence="medium",
                suggested_actions=[
                    f"Focus optimization on submission '{label}'",
                    "Profile the associated shader for ALU and memory bottlenecks",
                ],
            ))

    # GPU counter-based analysis
    counters = summary.get("gpu_counters", {})
    alu_util = counters.get("alu_utilization")
    if alu_util is not None and alu_util < thresholds.metal_alu_util_low:
        confidence = "high" if alu_util < 25 else "medium"
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Low ALU utilization ({alu_util:.0f}%)",
            root_cause="GPU compute units are underutilized — likely memory-bound or insufficient parallelism",
            confidence=confidence,
            suggested_actions=[
                "Increase threadgroup size for better occupancy",
                "Reduce memory access latency via threadgroup memory",
                "Use SIMD group functions for data sharing",
            ],
        ))

    return findings


def _analyze_host_device(
    profiler_summaries: dict[str, dict],
    program_type: str,
    thresholds: AnalysisThresholds,
) -> list[BottleneckDiagnosis]:
    """Cross-analyze host (CPU) and device (GPU) data for C++/CUDA programs."""
    findings: list[BottleneckDiagnosis] = []
    nsys = profiler_summaries.get("nsys", {})
    perf = profiler_summaries.get("linux_perf", {})

    kernel_time = nsys.get("cuda_kernel_time_ms")
    total_sync = nsys.get("total_sync_ms")

    # Rule 1: Excessive synchronization
    if total_sync is not None and kernel_time and kernel_time > 0:
        sync_ratio = total_sync / kernel_time
        if sync_ratio > thresholds.host_device_sync_ratio:
            pct = sync_ratio * 100
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Excessive cudaDeviceSynchronize ({pct:.0f}% of kernel time)",
                root_cause="Frequent host-device synchronization stalls the GPU pipeline",
                confidence="high",
                suggested_actions=[
                    "Use CUDA streams to overlap work and reduce synchronization points",
                    "Replace cudaDeviceSynchronize with stream-level cudaStreamSynchronize",
                    "Use CUDA graphs for repeated launch patterns",
                    "Use async memory operations (cudaMemcpyAsync) to avoid implicit syncs",
                ],
            ))

    # Rule 2: Many small kernels
    top_kernels = nsys.get("top_kernels", [])
    if top_kernels:
        avg_us_vals = [k.get("avg_us", 0) for k in top_kernels]
        total_count = sum(k.get("count", 0) for k in top_kernels)
        avg_overall = sum(avg_us_vals) / len(avg_us_vals) if avg_us_vals else 0
        if avg_overall < thresholds.host_device_kernel_dur_low_us and total_count > thresholds.host_device_kernel_count_high:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Kernel launch overhead from many small kernels ({total_count} launches, avg {avg_overall:.1f} us)",
                root_cause="Launching many short kernels causes host-side overhead to dominate",
                confidence="medium",
                suggested_actions=[
                    "Fuse small kernels into larger combined kernels",
                    "Use CUDA graphs to batch repeated launch sequences",
                    "Increase per-kernel work (larger grid, more elements per thread)",
                ],
            ))

    # Rule 3: Data transfer bottleneck (enhanced)
    memcpy_time = nsys.get("memcpy_time_ms")
    if memcpy_time is not None and kernel_time and kernel_time > 0:
        transfer_ratio = memcpy_time / kernel_time
        if transfer_ratio > thresholds.host_device_transfer_ratio:
            pct = transfer_ratio * 100
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Host-device transfer dominates ({pct:.0f}% of kernel time)",
                root_cause="Data transfers between host and device consume significant time",
                confidence="high",
                suggested_actions=[
                    "Use pinned memory (cudaMallocHost) for faster host-device transfers",
                    "Overlap transfers with kernel execution using CUDA streams",
                    "Keep data on device across iterations to eliminate redundant transfers",
                    "Batch small transfers into fewer large transfers",
                ],
            ))

    # Rule 4: Low GPU utilization with CPU hotspot
    gpu_active = nsys.get("gpu_active_pct")
    hotspots = perf.get("hotspots", [])
    if gpu_active is not None and gpu_active < thresholds.host_device_gpu_active_low and hotspots:
        top_hs = hotspots[0]
        # 30%: CPU hotspot consuming 30%+ of samples while GPU is idle = clear CPU bottleneck
        if top_hs.get("pct", 0) > 30:
            fname = top_hs.get("function", "(unknown)")
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"CPU function '{fname}' blocks GPU utilization (GPU active {gpu_active:.0f}%)",
                root_cause=f"CPU hotspot '{fname}' prevents the GPU from staying busy",
                confidence="high",
                suggested_actions=[
                    "Overlap CPU computation with GPU work using CUDA streams",
                    f"Move work from '{fname}' to GPU if possible",
                    "Use async CUDA APIs to avoid blocking the CPU",
                    "Pipeline host preprocessing with device execution",
                ],
            ))

    return findings


def _analyze_cross_profiler_cpu_gpu(
    torch_summary: dict,
    metal_summary: dict,
    thresholds: AnalysisThresholds,
) -> list[BottleneckDiagnosis]:
    """Synthesize CPU/GPU breakdown by cross-referencing torch trace and Metal trace.

    On MPS (Apple Silicon), the torch Chrome trace lacks cat=kernel events for
    Metal GPU work.  The Metal trace profiler provides gpu_time_total_ms while
    the torch profiler still records CPU op time — combining them fills the gap.
    """
    findings: list[BottleneckDiagnosis] = []

    total_cpu_us = sum(
        op.get("total_us", 0) for op in torch_summary.get("top_ops", [])
    )
    total_gpu_ms = metal_summary.get("gpu_time_total_ms", 0)

    if total_cpu_us <= 0 and total_gpu_ms <= 0:
        return findings

    ratio = (total_gpu_ms * 1000) / total_cpu_us if total_cpu_us > 0 else float("inf")
    duration_s = metal_summary.get("duration_s", 1) or 1
    gpu_util = (total_gpu_ms / 1000) / duration_s

    # Write back cpu_vs_gpu into torch_summary so it propagates to the prompt
    torch_summary["cpu_vs_gpu"] = {
        "total_cpu_op_us": round(total_cpu_us, 1),
        "total_gpu_kernel_us": round(total_gpu_ms * 1000, 1),
        "ratio": round(ratio, 3),
        "source": "cross_profiler_mps",
    }

    if ratio < thresholds.cross_gpu_cpu_ratio_low:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"GPU underutilized — CPU dispatch is bottleneck (GPU/CPU ratio={ratio:.2f})",
            root_cause="CPU-side operator dispatch takes much longer than GPU kernel execution (MPS cross-profiler)",
            confidence="high",
            suggested_actions=[
                "Use torch.compile() to fuse operators and reduce dispatch overhead",
                "Reduce Python-level overhead in the training loop",
                "Increase batch size to amortize per-step CPU overhead",
            ],
        ))

    if gpu_util < thresholds.cross_gpu_util_low:
        pct = gpu_util * 100
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"MPS GPU active only {pct:.0f}% of trace time",
            root_cause="Metal GPU is idle for most of the profiled duration",
            confidence="medium",
            suggested_actions=[
                "Increase batch size to give GPU more work per dispatch",
                "Reduce CPU-GPU synchronization points",
                "Batch more work per command buffer",
            ],
        ))

    return findings
