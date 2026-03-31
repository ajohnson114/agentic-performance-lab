from __future__ import annotations

import re

from perflab.analyzers.bottleneck_types import AnalysisThresholds, BottleneckDiagnosis


def _analyze_tpu(
    jax_summary: dict,
    thresholds: AnalysisThresholds,
    system_info: dict | None = None,
) -> list[BottleneckDiagnosis]:
    """Analyze TPU-specific bottlenecks from JAX profiler data and system info."""
    findings: list[BottleneckDiagnosis] = []
    si = system_info or {}

    # Rule 1: Low MXU utilization (from jax profiler trace data if available)
    mxu_util = jax_summary.get("mxu_utilization_pct")
    if mxu_util is not None and mxu_util < thresholds.tpu_mxu_util_low:
        # 15%: MXU is the TPU's primary compute engine — below 15% it's nearly idle
        confidence = "high" if mxu_util < 15.0 else "medium"
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Low TPU MXU utilization ({mxu_util:.1f}%)",
            root_cause="The matrix multiply units (MXUs) are idle — computation is not saturating the TPU's primary compute engine",
            confidence=confidence,
            suggested_actions=[
                "Use bfloat16 dtype — MXUs run bf16 at full throughput, fp32 at half",
                "Pad matrix dimensions to multiples of 128 (TPU tile size) to avoid partial tiles",
                "Increase batch size to give MXUs larger matrices to process",
                "Use jax.lax.scan instead of Python loops to fuse operations into one XLA program",
                "Check for host callbacks or jax.debug.print in the hot path — they stall the pipeline",
            ],
        ))

    # Rule 2: Padding waste from HLO analysis
    hlo_ops = jax_summary.get("hlo_ops", [])
    pad_count = sum(op.get("count", 0) for op in hlo_ops if op.get("op") == "pad")
    total_ops = sum(op.get("count", 0) for op in hlo_ops)
    if total_ops > 0 and pad_count > 0:
        pad_pct = (pad_count / total_ops) * 100
        if pad_pct > thresholds.tpu_padding_waste_pct_high:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Excessive XLA padding operations ({pad_count} pads, {pad_pct:.0f}% of ops)",
                root_cause="XLA is inserting pad operations to align tensors to TPU tile boundaries, wasting compute",
                confidence="medium",
                suggested_actions=[
                    "Pad input dimensions to multiples of 128 in your code to avoid XLA-inserted padding",
                    "Use batch sizes that are multiples of 8 (TPU likes powers of 2)",
                    "For attention: use sequence lengths that are multiples of 128",
                    "Check if reshape/transpose ops are causing unnecessary padding",
                ],
            ))

    # Rule 3: Infeed stalls (data loading bottleneck)
    infeed_pct = jax_summary.get("infeed_stall_pct")
    if infeed_pct is not None and infeed_pct > thresholds.tpu_infeed_stall_pct_high:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"TPU infeed stall ({infeed_pct:.1f}% of step time waiting for data)",
            root_cause="The TPU is idle waiting for data from the host — data pipeline is the bottleneck",
            confidence="high",
            suggested_actions=[
                "Use tf.data with prefetch(tf.data.AUTOTUNE) for data loading",
                "Increase the number of data loading workers",
                "Pre-process and cache data in host memory or on GCS",
                "Use grain dataloader for JAX-native data loading",
            ],
        ))

    # Rule 4: Too many small HLO modules (fragmented computation)
    hlo_modules = jax_summary.get("hlo_module_count", 0)
    # 10 modules: XLA can't fuse/pipeline across module boundaries; >10 fragments the compute
    if hlo_modules > 10:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Fragmented XLA computation ({hlo_modules} HLO modules)",
            root_cause="Many separate XLA programs mean the TPU can't pipeline and fuse operations efficiently",
            confidence="medium",
            suggested_actions=[
                "Consolidate computation into fewer @jax.jit-decorated functions",
                "Use jax.lax.scan/fori_loop to fuse iterative computation",
                "Avoid mixing jitted and un-jitted code in the hot path",
            ],
        ))

    # Rule 5: Not using bfloat16 (check HLO for f32 dominance)
    f32_ops = sum(op.get("count", 0) for op in hlo_ops if "f32" in str(op.get("op", "")).lower())
    bf16_ops = sum(op.get("count", 0) for op in hlo_ops if "bf16" in str(op.get("op", "")).lower() or "bfloat" in str(op.get("op", "")).lower())
    # 20 ops: need enough ops to be meaningful; f32 > 3x bf16 with zero bf16 = clear fp32 dominance
    if total_ops > 20 and f32_ops > bf16_ops * 3 and bf16_ops == 0:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck="Computation uses fp32 — TPU MXUs run fp32 at half the bf16 throughput",
            root_cause="TPU matrix units are optimized for bfloat16; using fp32 halves peak throughput",
            confidence="medium",
            suggested_actions=[
                "Convert model to bfloat16: jnp.bfloat16 or use mixed precision",
                "Use jax.default_matmul_precision('bfloat16') for automatic bf16 matmuls",
                "For training: bf16 forward + fp32 gradient accumulation is standard on TPU",
            ],
        ))

    return findings


def _analyze_jax(summary: dict, thresholds: AnalysisThresholds) -> list[BottleneckDiagnosis]:
    """Analyze JAX/XLA compilation and profiling summary."""
    findings: list[BottleneckDiagnosis] = []

    recomps = summary.get("xla_recompilations", 0)
    compilations = summary.get("xla_compilations", 0)
    compile_time_ms = summary.get("xla_compilation_time_ms", 0)
    duration_s = summary.get("duration_s", 0)

    # Rule 1: Recompilation detected
    if recomps >= thresholds.jax_recompilation_warn:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"XLA recompilation detected ({recomps} recompilations)",
            root_cause="Changing input shapes or types triggers XLA recompilation, stalling execution",
            confidence="high",
            suggested_actions=[
                "Use @jax.jit with static_argnums for arguments that change shape",
                "Pad inputs to fixed shapes to avoid shape-triggered recompilation",
                "Use jax.lax.scan instead of Python loops over varying-length sequences",
                "Check for accidental Python-level tracing (e.g., data-dependent control flow)",
            ],
        ))

    # Rule 2: High compilation overhead
    if (compile_time_ms > thresholds.jax_compilation_time_high_ms
            and duration_s > 0
            and (compile_time_ms / 1000.0) / duration_s > thresholds.jax_compilation_fraction_high):
        frac_pct = (compile_time_ms / 1000.0) / duration_s * 100
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"High XLA compilation overhead ({compile_time_ms:.0f}ms, {frac_pct:.0f}% of runtime)",
            root_cause="XLA compilation consumes a significant fraction of total execution time",
            confidence="high",
            suggested_actions=[
                "Add more warmup iterations before timing to amortize compilation",
                "Enable persistent compilation cache: jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')",
                "Use donate_argnums to reduce memory copies and avoid recompilation",
                "Consider AOT compilation with jax.jit(...).lower(...).compile()",
            ],
        ))

    # Rule 3: Excessive compilations
    if compilations > thresholds.jax_compilations_excessive:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Excessive XLA compilations ({compilations} compilations)",
            root_cause="Many separate XLA compilations indicate un-jitted or fragmented computation",
            confidence="medium",
            suggested_actions=[
                "Wrap compute-heavy functions with @jax.jit",
                "Use jax.lax.scan/fori_loop instead of Python loops",
                "Avoid calling jnp operations outside of jitted functions",
                "Consolidate small jitted functions into larger ones",
            ],
        ))

    return findings


def _analyze_torch_trace(summary: dict, *, device: str | None = None, thresholds: AnalysisThresholds) -> list[BottleneckDiagnosis]:
    """Analyze PyTorch profiler trace summary."""
    findings: list[BottleneckDiagnosis] = []
    is_mps = (device or "").lower() == "mps"

    # CPU vs GPU imbalance
    cpu_vs_gpu = summary.get("cpu_vs_gpu")
    if cpu_vs_gpu:
        ratio = cpu_vs_gpu.get("ratio", 1.0)
        gpu_us = cpu_vs_gpu.get("total_gpu_kernel_us", 0)
        if is_mps and gpu_us == 0:
            # On MPS, torch profiler cannot see Metal GPU kernels — 0 GPU time
            # is expected and does NOT mean the GPU is idle.
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck="MPS GPU timing unavailable (torch profiler cannot trace Metal kernels)",
                root_cause=(
                    "The PyTorch profiler does not emit GPU kernel events on the MPS backend. "
                    "Actual GPU utilization is not zero — it is simply not measurable without "
                    "Xcode Instruments (xctrace 'Metal System Trace')"
                ),
                confidence="medium",
                suggested_actions=[
                    "Install Xcode and run with Metal System Trace for real GPU timing",
                    "Focus on CPU-side optimizations visible in the trace: torch.compile, "
                    "batching, reducing Python overhead",
                    "Use float16 / channels_last for faster MPS kernel dispatch",
                ],
            ))
        elif ratio < thresholds.gpu_cpu_ratio_low:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"GPU underutilized — CPU dispatch is bottleneck (GPU/CPU ratio={ratio:.2f})",
                root_cause="CPU-side operator dispatch takes much longer than GPU kernel execution",
                confidence="high",
                suggested_actions=[
                    "Use torch.compile() to fuse operators and reduce dispatch overhead",
                    "Enable operator fusion via TorchScript or torch.jit.trace",
                    "Reduce Python-level overhead in the training loop",
                    "Increase batch size to amortize per-step CPU overhead",
                ],
            ))

    # Excessive synchronization
    sync_count = summary.get("sync_count", 0)
    total_sync_us = summary.get("total_sync_time_us", 0)
    if sync_count > thresholds.sync_count_warn:
        sync_ms = total_sync_us / 1000.0
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Excessive GPU synchronization ({sync_count} syncs, {sync_ms:.1f} ms)",
            root_cause="Frequent cudaDeviceSynchronize/cudaStreamSynchronize calls stall the pipeline",
            confidence="medium",
            suggested_actions=[
                "Remove unnecessary .item(), .cpu(), or .numpy() calls in the training loop",
                "Use non_blocking=True for CPU-GPU transfers",
                "Avoid printing/logging tensor values during timed runs",
                "Defer result collection to after the training loop",
            ],
        ))

    # Memory allocation overhead
    memory = summary.get("memory", {})
    total_alloc_us = memory.get("total_allocation_time_us", 0)
    top_ops = summary.get("top_ops", [])
    total_op_us = sum(op.get("total_us", 0) for op in top_ops)
    if total_op_us > 0 and total_alloc_us > total_op_us * thresholds.mem_alloc_overhead_pct:
        pct = total_alloc_us / total_op_us * 100
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Memory allocation overhead ({pct:.0f}% of operator time)",
            root_cause="Frequent memory allocations and deallocations during execution",
            confidence="medium",
            suggested_actions=[
                "Pre-allocate tensors and reuse buffers across iterations",
                "Use CUDA memory pools (set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)",
                "Avoid creating temporary tensors in hot loops",
            ],
        ))

    # Dominant GPU kernel
    top_gpu = summary.get("top_gpu_kernels", [])
    if top_gpu and top_gpu[0].get("pct", 0) > thresholds.gpu_kernel_dominance_pct:
        k = top_gpu[0]
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Dominant GPU kernel: '{k['name']}' ({k['pct']:.0f}% of GPU time)",
            root_cause=f"Kernel '{k['name']}' accounts for most GPU execution time",
            confidence="medium",
            suggested_actions=[
                f"Focus optimization on kernel '{k['name']}'",
                "Profile with ncu for detailed per-kernel metrics",
                "Consider using torch.compile() for kernel fusion",
            ],
        ))

    # -- Per-phase training breakdown analysis --
    phases = summary.get("phases", [])
    if phases:
        total_phase_us = sum(p.get("total_us", 0) for p in phases)
        _phase_suggestions: dict[str, list[str]] = {
            "forward": [
                "Use scaled_dot_product_attention (SDPA) for efficient attention",
                "Enable AMP (torch.autocast) for mixed-precision forward pass",
                "Apply torch.compile() for operator fusion",
                "Consider architectural changes (fewer layers, smaller d_model)",
            ],
            "backward": [
                "Enable gradient checkpointing to trade compute for memory",
                "Use AMP (torch.autocast) for mixed-precision backward pass",
                "Apply torch.compile() to fuse backward operators",
                "Use gradient accumulation to reduce peak activation memory per step",
                "Call torch.cuda.empty_cache() after backward pass to defragment memory pools",
            ],
            "optimizer": [
                "Use a fused optimizer (e.g. torch.optim.AdamW with fused=True)",
                "Reduce parameter count or use sparse updates",
            ],
            "data_loading": [
                "Increase DataLoader num_workers for parallel loading",
                "Enable pin_memory=True for faster CPU-to-GPU transfers",
                "Use persistent_workers=True to avoid restart overhead",
                "Consider NVIDIA DALI or memory-mapped datasets",
            ],
        }
        for phase in phases:
            pname = phase.get("name", "")
            pct = phase.get("pct", 0)

            # Phase dominance: >threshold of total
            if pct > thresholds.phase_dominance_pct:
                suggestions = _phase_suggestions.get(pname, [
                    f"Profile the '{pname}' phase in detail for targeted optimization",
                ])
                findings.append(BottleneckDiagnosis(
                    rank=0,
                    bottleneck=f"'{pname}' phase dominates training ({pct:.0f}% of total)",
                    root_cause=f"The {pname} phase accounts for the majority of per-step time",
                    confidence="high",
                    suggested_actions=suggestions,
                ))

            # GPU underutilization within forward/backward
            phase_total = phase.get("total_us", 0)
            phase_gpu = phase.get("gpu_us", 0)
            if pname in ("forward", "backward") and phase_total > 0:
                gpu_frac = phase_gpu / phase_total
                if gpu_frac < thresholds.phase_gpu_fraction_low:
                    findings.append(BottleneckDiagnosis(
                        rank=0,
                        bottleneck=f"GPU underutilized during {pname} phase ({gpu_frac:.0%} GPU)",
                        root_cause=f"CPU dispatch overhead dominates the {pname} phase",
                        confidence="medium",
                        suggested_actions=[
                            "Apply torch.compile() to fuse operators and reduce dispatch overhead",
                            "Increase batch size to amortize per-step CPU overhead",
                        ],
                    ))

    # -- Non-contiguous tensor detection --
    top_ops = summary.get("top_ops", [])
    for op in top_ops[:10]:
        op_name = op.get("name", "")
        op_pct = op.get("pct", 0)
        # 3%: contiguous/clone ops are pure data movement; >3% of op time is actionable waste
        if op_name in ("aten::contiguous", "aten::clone") and op_pct > 3.0:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Non-contiguous tensor copies ('{op_name}' = {op_pct:.0f}% of op time)",
                root_cause=f"'{op_name}' creates a contiguous copy of non-contiguous tensors, "
                           "wasting memory bandwidth on data movement instead of computation",
                confidence="medium" if op_pct < 10.0 else "high",
                suggested_actions=[
                    "Call .contiguous() once before a hot loop, not inside it",
                    "Use channels_last memory format consistently to avoid mixed-format copies",
                    "Preallocate output tensors with the right layout to avoid implicit copies",
                    "Use torch.empty with the correct memory_format instead of clone()",
                ],
            ))
            break  # One finding is enough

    # -- Tensor Core alignment checking --
    for op in top_ops[:5]:
        shapes_str = op.get("shapes", "")
        op_name = op.get("name", "")
        if not shapes_str or "matmul" not in op_name.lower() and "mm" not in op_name.lower():
            continue
        # Extract dimension numbers from shapes like "[[64, 127], [127, 256]]"
        dim_pattern = re.compile(r"\d+")
        dims = [int(d) for d in dim_pattern.findall(shapes_str)]
        # d > 16: skip tiny dims (already fit in one TC tile); d % 8: TC tile alignment (8 for TF32, 16 for FP16)
        misaligned = [d for d in dims if d > 16 and d % 8 != 0]
        if misaligned:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Matmul dimensions not aligned to Tensor Core tiles in '{op_name}' "
                           f"(dims: {misaligned})",
                root_cause="Tensor Core operations require dimensions aligned to multiples of 8 (TF32) "
                           "or 16 (FP16). Misaligned dimensions cause padding waste and reduced throughput",
                confidence="medium",
                suggested_actions=[
                    f"Pad dimensions to nearest multiple of 16: {[((d + 15) // 16) * 16 for d in misaligned]}",
                    "Use torch.nn.functional.pad() before matmul operations",
                    "Design model dimensions (d_model, n_heads, vocab_size) as multiples of 64 or 128",
                ],
            ))
            break

    # -- Memory fragmentation detection --
    memory = summary.get("memory", {})
    peak_mb = memory.get("peak_memory_mb")
    # 1024 MB: fragmentation is only meaningful at scale; below 1GB the allocator handles it fine
    if peak_mb is not None and peak_mb > 1024:
        # Heuristic: if allocation count is high and peak is large,
        # fragmentation is likely. We check if there are many small allocations.
        alloc_count = memory.get("total_allocations", 0)
        alloc_time_us = memory.get("total_allocation_time_us", 0)
        # 500 allocs + 50ms: many small allocs fragmenting the CUDA memory pool with measurable overhead
        if alloc_count > 500 and alloc_time_us > 50000:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Potential memory fragmentation ({alloc_count} allocations, "
                           f"{peak_mb:.0f} MB peak, {alloc_time_us / 1000:.0f} ms alloc time)",
                root_cause="Many small allocations can fragment the CUDA memory pool, causing "
                           "OOM errors despite sufficient total memory and increasing allocation latency",
                confidence="medium",
                suggested_actions=[
                    "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation",
                    "Pre-allocate tensors and reuse them across iterations",
                    "Use torch.cuda.memory.CUDAPluggableAllocator for custom allocation strategies",
                    "Reduce dynamic tensor creation — use fixed-size buffers where possible",
                ],
            ))

    return findings


def _analyze_nvtx_phases(nsys_summary: dict, thresholds: AnalysisThresholds) -> list[BottleneckDiagnosis]:
    """Analyze NVTX annotation ranges for phase-level bottlenecks."""
    findings: list[BottleneckDiagnosis] = []
    ranges = nsys_summary.get("nvtx_ranges", [])
    if not ranges:
        return findings

    total_ms = sum(r.get("duration_ms", 0) for r in ranges)

    # Rule 1: Single NVTX range dominating
    if ranges and total_ms > 0:
        top = ranges[0]
        top_pct = top.get("duration_ms", 0) / total_ms * 100
        if top_pct > thresholds.nvtx_phase_dominance_pct:
            name = top.get("name", "(unnamed)")
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Phase '{name}' dominates execution ({top_pct:.0f}% of NVTX time)",
                root_cause=f"NVTX range '{name}' accounts for the majority of annotated execution time",
                confidence="medium",
                suggested_actions=[
                    f"Focus optimization efforts on the '{name}' phase",
                    "Profile this phase in detail with ncu for GPU-side analysis",
                    "Consider algorithmic improvements within this phase",
                ],
            ))

    # Rule 2: Many short NVTX ranges
    if len(ranges) > thresholds.nvtx_range_count_high:
        avg_ms = total_ms / len(ranges) if len(ranges) > 0 else 0
        if avg_ms < thresholds.nvtx_avg_range_dur_low_ms:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Fine-grained work partitioning ({len(ranges)} NVTX ranges, avg {avg_ms:.2f} ms)",
                root_cause="Many short NVTX ranges may indicate excessive overhead from fine-grained work",
                confidence="low",
                suggested_actions=[
                    "Coarsen work granularity to reduce overhead",
                    "Batch small work units into larger chunks",
                ],
            ))

    return findings


def _analyze_memray(summary: dict, thresholds: AnalysisThresholds) -> list[BottleneckDiagnosis]:
    """Analyze memray memory profiling summary."""
    findings: list[BottleneckDiagnosis] = []

    peak_mb = summary.get("peak_memory_mb")
    if peak_mb is not None and peak_mb > thresholds.memray_peak_mb_warn:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"High peak memory usage ({peak_mb:.0f} MB)",
            root_cause="Large temporary allocations or unbounded data structures",
            confidence="medium",
            suggested_actions=[
                "Use generators/iterators instead of materializing full lists",
                "Process data in chunks rather than loading all at once",
                "Check for memory leaks in loops (appending without clearing)",
            ],
        ))

    top_allocs = summary.get("top_allocators", [])
    total_alloc = summary.get("total_allocated_mb", 0)
    if top_allocs and total_alloc > 0:
        top_size = top_allocs[0].get("size_mb", 0)
        top_pct = (top_size / total_alloc * 100) if total_alloc > 0 else 0
        if top_pct > thresholds.memray_top_allocator_dominance_pct:
            func = top_allocs[0].get("function", "unknown")
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"Dominant memory allocator: {func} ({top_pct:.0f}% of allocations)",
                root_cause=f"Function '{func}' dominates heap allocations",
                confidence="medium",
                suggested_actions=[
                    f"Pre-allocate buffers used by {func}",
                    "Use memory pools or object recycling to reduce allocation pressure",
                    "Consider in-place operations to avoid temporary copies",
                ],
            ))

    return findings


def _analyze_ebpf(summary: dict, thresholds: AnalysisThresholds) -> list[BottleneckDiagnosis]:
    """Analyze eBPF I/O tracing summary."""
    findings: list[BottleneckDiagnosis] = []

    read_lat = summary.get("read_latency", {})
    write_lat = summary.get("write_latency", {})

    # High read latency
    read_p99 = read_lat.get("p99_ns")
    if read_p99 is not None:
        read_p99_us = read_p99 / 1000.0
        if read_p99_us > thresholds.ebpf_read_p99_us_high:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"High I/O read latency (p99={read_p99_us:.0f} \u00b5s)",
                root_cause="Slow disk reads or remote filesystem access",
                confidence="medium",
                suggested_actions=[
                    "Use memory-mapped I/O or pre-load data into memory",
                    "Add prefetching or async I/O for data loading",
                    "Check if data is on a slow storage device (HDD, NFS)",
                ],
            ))

    # High write latency
    write_p99 = write_lat.get("p99_ns")
    if write_p99 is not None:
        write_p99_us = write_p99 / 1000.0
        if write_p99_us > thresholds.ebpf_write_p99_us_high:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"High I/O write latency (p99={write_p99_us:.0f} \u00b5s)",
                root_cause="Slow disk writes or synchronous flushing",
                confidence="medium",
                suggested_actions=[
                    "Buffer writes and flush in batches",
                    "Use async I/O or background writer thread",
                    "Check if filesystem sync is being called too frequently",
                ],
            ))

    # Excessive syscall count
    read_count = summary.get("read_syscalls", 0)
    write_count = summary.get("write_syscalls", 0)
    total_syscalls = read_count + write_count
    if total_syscalls > thresholds.ebpf_syscall_count_high:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Excessive I/O syscalls ({total_syscalls:,} read+write calls)",
            root_cause="Many small I/O operations instead of batched reads/writes",
            confidence="high",
            suggested_actions=[
                "Increase I/O buffer sizes to reduce syscall frequency",
                "Batch small reads into larger reads (e.g., read full blocks)",
                "Use mmap for random access patterns",
            ],
        ))

    return findings


def _analyze_lock_contention(summary: dict, thresholds: AnalysisThresholds) -> list[BottleneckDiagnosis]:
    """Analyze lock contention profiler summary."""
    findings: list[BottleneckDiagnosis] = []

    lock_stats = summary.get("lock_stats", {})
    locks = lock_stats.get("locks", [])

    # Check overall contention ratio
    total_acquired = sum(l.get("acquired", 0) for l in locks)
    total_contended = lock_stats.get("total_contended", 0)
    total_wait_ns = lock_stats.get("total_wait_ns", 0)
    total_wait_ms = total_wait_ns / 1e6

    if total_acquired > 0:
        contention_ratio = total_contended / total_acquired
        if contention_ratio > thresholds.lock_contention_ratio_high:
            findings.append(BottleneckDiagnosis(
                rank=0,
                bottleneck=f"High lock contention ({contention_ratio:.0%} of acquisitions contended)",
                root_cause="Threads frequently block waiting for shared locks",
                confidence="high",
                suggested_actions=[
                    "Reduce critical section size (hold locks for less time)",
                    "Use lock-free data structures where possible",
                    "Partition shared data to reduce lock scope",
                    "Consider reader-writer locks if reads dominate",
                ],
            ))

    if total_wait_ms > thresholds.lock_total_wait_ms_high:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"High lock wait time ({total_wait_ms:.0f} ms total)",
            root_cause="Threads spend significant time blocked on lock acquisition",
            confidence="high",
            suggested_actions=[
                "Profile which locks are most contended and restructure access patterns",
                "Use fine-grained locking instead of a single global lock",
                "Consider lock-free algorithms for hot paths",
            ],
        ))

    # False sharing detection
    c2c = summary.get("c2c_stats", {})
    hitm = c2c.get("total_hitm", 0)
    if hitm > thresholds.lock_false_sharing_hitm_high:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Cache-line false sharing detected ({hitm} HITM events)",
            root_cause="Different threads writing to variables on the same cache line",
            confidence="high",
            suggested_actions=[
                "Pad shared structures to cache-line boundaries (64 bytes)",
                "Use thread-local accumulators and reduce at the end",
                "Align per-thread data with alignas(64) or __attribute__((aligned(64)))",
            ],
        ))

    return findings


def _analyze_thread_sched(summary: dict, thresholds: AnalysisThresholds) -> list[BottleneckDiagnosis]:
    """Analyze thread scheduling profiler summary."""
    findings: list[BottleneckDiagnosis] = []

    latency_entries = summary.get("latency", [])
    timehist = summary.get("timehist", {})

    # Check for high scheduling delays
    if latency_entries:
        avg_delays = [e.get("avg_delay_ms", 0) for e in latency_entries if e.get("avg_delay_ms")]
        if avg_delays:
            max_avg_delay = max(avg_delays)
            if max_avg_delay > thresholds.thread_sched_avg_delay_ms_high:
                worst_thread = next((e.get("task", "?") for e in latency_entries
                                     if e.get("avg_delay_ms") == max_avg_delay), "?")
                findings.append(BottleneckDiagnosis(
                    rank=0,
                    bottleneck=f"High thread scheduling delay ({max_avg_delay:.1f} ms avg for {worst_thread})",
                    root_cause="Thread is frequently preempted or delayed by the OS scheduler",
                    confidence="medium",
                    suggested_actions=[
                        "Set thread affinity to pin threads to specific cores",
                        "Reduce the number of active threads to match physical cores",
                        "Increase thread priority for latency-sensitive threads",
                    ],
                ))

    # Check for excessive migrations
    migrations = timehist.get("migrations")
    if migrations is not None and migrations > thresholds.thread_sched_migrations_high:
        findings.append(BottleneckDiagnosis(
            rank=0,
            bottleneck=f"Excessive thread migrations ({migrations} migrations)",
            root_cause="Threads are frequently moved between CPU cores, invalidating caches",
            confidence="medium",
            suggested_actions=[
                "Use CPU affinity (pthread_setaffinity_np or taskset) to pin threads",
                "Use NUMA-aware allocation to keep data near the processing core",
                "Reduce thread count to reduce scheduler pressure",
            ],
        ))

    return findings


def _analyze_power(summary: dict, thresholds: AnalysisThresholds) -> list[BottleneckDiagnosis]:
    """Analyze power profiler summary for thermal throttling."""
    findings: list[BottleneckDiagnosis] = []

    gpu_power = summary.get("gpu_power", {})
    samples = gpu_power.get("power_samples", [])

    if len(samples) >= 4:
        # Compare early vs late power draw to detect throttling
        n = len(samples)
        early = samples[:n // 4]
        late = samples[-n // 4:]

        early_avg = sum(s.get("watts", 0) for s in early) / len(early) if early else 0
        late_avg = sum(s.get("watts", 0) for s in late) / len(late) if late else 0

        if early_avg > 0 and late_avg > 0:
            drop_pct = (early_avg - late_avg) / early_avg * 100
            if drop_pct > thresholds.power_gpu_throttle_drop_pct:
                findings.append(BottleneckDiagnosis(
                    rank=0,
                    bottleneck=f"Possible GPU thermal throttling ({drop_pct:.0f}% power drop)",
                    root_cause="GPU power draw decreased during the run, suggesting thermal or power throttling",
                    confidence="low",
                    suggested_actions=[
                        "Check GPU temperature with nvidia-smi",
                        "Improve GPU cooling or reduce sustained load",
                        "Set GPU power limit higher if headroom exists",
                    ],
                ))

    return findings
