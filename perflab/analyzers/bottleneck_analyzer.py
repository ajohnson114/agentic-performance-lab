from __future__ import annotations

import re

from perflab.analyzers.bottleneck_types import AnalysisThresholds, BottleneckDiagnosis

# Re-export sub-module functions for backward compatibility.
# Any code doing ``from perflab.analyzers.bottleneck_analyzer import _analyze_ncu``
# will continue to work.
from perflab.analyzers.bottleneck_gpu import (  # noqa: F401
    _analyze_ncu,
    _analyze_nsys,
    _analyze_gpu_attribution,
    _analyze_metal,
    _analyze_host_device,
    _analyze_cross_profiler_cpu_gpu,
)
from perflab.analyzers.bottleneck_cpu import (  # noqa: F401
    _analyze_perf,
    _analyze_io_bottleneck,
)
from perflab.analyzers.bottleneck_system import (  # noqa: F401
    _analyze_torch_trace,
    _analyze_jax,
    _analyze_tpu,
    _analyze_memray,
    _analyze_ebpf,
    _analyze_lock_contention,
    _analyze_thread_sched,
    _analyze_power,
    _analyze_nvtx_phases,
)

# Re-export the dataclasses so existing ``from bottleneck_analyzer import AnalysisThresholds``
# keeps working.
__all__ = [
    "AnalysisThresholds",
    "BottleneckDiagnosis",
    "compute_source_hints",
    "diagnose_bottlenecks",
    "_analyze_ncu",
    "_analyze_nsys",
    "_analyze_gpu_attribution",
    "_analyze_metal",
    "_analyze_host_device",
    "_analyze_cross_profiler_cpu_gpu",
    "_analyze_perf",
    "_analyze_io_bottleneck",
    "_analyze_torch_trace",
    "_analyze_jax",
    "_analyze_tpu",
    "_analyze_memray",
    "_analyze_ebpf",
    "_analyze_lock_contention",
    "_analyze_thread_sched",
    "_analyze_power",
    "_analyze_nvtx_phases",
]


def compute_source_hints(source_files: dict[str, str], program_type: str) -> dict:
    """Compute lightweight source-analysis flags for bottleneck rules."""
    hints: dict = {}
    if program_type != "cpp":
        return hints

    all_content = "\n".join(source_files.values())

    def _in_code(pattern: str) -> bool:
        return bool(re.search(r"^[^#/\n]*" + pattern, all_content, re.MULTILINE))

    hints["has_simd"] = _in_code(r"(?:#include\s*<immintrin|_mm256|_mm512|__m128|vld1|vaddq)")
    hints["has_openmp"] = _in_code(r"#pragma\s+omp")
    hints["has_threading"] = _in_code(r"(?:std::thread|pthread_create|std::async)")

    return hints


def diagnose_bottlenecks(
    profiler_summaries: dict[str, dict],
    program_type: str,
    top_n: int = 3,
    device: str | None = None,
    thresholds: AnalysisThresholds | None = None,
    system_info: dict | None = None,
    source_hints: dict | None = None,
    compiler_remarks: list | None = None,
    cpu_isa: dict | None = None,
) -> list[BottleneckDiagnosis]:
    """Rule-based bottleneck diagnosis from profiler summary data.

    Analyzes profiler summaries (ncu, nsys, linux_perf, metal_trace) and
    returns ranked diagnoses with root causes and suggested actions.

    If *device* is provided (e.g. ``"mps"``), it is used to contextualise
    GPU-related findings — on MPS the torch profiler cannot observe Metal GPU
    kernels, so ``total_gpu_kernel_us == 0`` is expected and should not be
    reported as "GPU underutilized".
    """
    thresholds = thresholds or AnalysisThresholds()
    findings: list[BottleneckDiagnosis] = []

    if "ncu" in profiler_summaries:
        findings.extend(_analyze_ncu(profiler_summaries["ncu"], thresholds))

    if "nsys" in profiler_summaries:
        findings.extend(_analyze_nsys(profiler_summaries["nsys"], thresholds))

    if "linux_perf" in profiler_summaries:
        cpu_count = (system_info or {}).get("cpu_count")
        findings.extend(_analyze_perf(
            profiler_summaries["linux_perf"], thresholds,
            cpu_count=cpu_count, program_type=program_type,
            source_hints=source_hints,
            compiler_remarks=compiler_remarks,
            cpu_isa=cpu_isa,
        ))

    if "metal_trace" in profiler_summaries:
        findings.extend(_analyze_metal(profiler_summaries["metal_trace"], thresholds))

    if "jax" in profiler_summaries:
        findings.extend(_analyze_jax(profiler_summaries["jax"], thresholds))

        # TPU-specific analysis (piggybacks on jax profiler data + system_info)
        is_tpu = (system_info or {}).get("tpu_devices") is not None
        if is_tpu:
            findings.extend(_analyze_tpu(profiler_summaries["jax"], thresholds, system_info=system_info))

    if "torch_profiler" in profiler_summaries:
        findings.extend(_analyze_torch_trace(profiler_summaries["torch_profiler"], device=device, thresholds=thresholds))

    # MPS cross-profiler CPU/GPU synthesis (only when torch trace lacks GPU data)
    if "torch_profiler" in profiler_summaries and "metal_trace" in profiler_summaries:
        torch_cpu_gpu = profiler_summaries["torch_profiler"].get("cpu_vs_gpu", {})
        if not torch_cpu_gpu or torch_cpu_gpu.get("total_gpu_kernel_us", 0) <= 0:
            findings.extend(_analyze_cross_profiler_cpu_gpu(
                profiler_summaries["torch_profiler"],
                profiler_summaries["metal_trace"],
                thresholds,
            ))

    # I/O and data loading detection (cross-profiler)
    findings.extend(_analyze_io_bottleneck(profiler_summaries, program_type, thresholds))

    # Host-device cross-analysis for C++/CUDA programs
    if program_type in ("cpp", "cuda"):
        findings.extend(_analyze_host_device(profiler_summaries, program_type, thresholds))

    # NVTX phase analysis
    nsys = profiler_summaries.get("nsys", {})
    if nsys.get("nvtx_ranges"):
        findings.extend(_analyze_nvtx_phases(nsys, thresholds))

    # GPU attribution analysis
    if "nsys" in profiler_summaries:
        nsys_data = profiler_summaries["nsys"]
        perf_data = profiler_summaries.get("linux_perf")
        findings.extend(_analyze_gpu_attribution(nsys_data, perf_data, thresholds))

    # Memory allocation analysis (memray)
    if "memray" in profiler_summaries:
        findings.extend(_analyze_memray(profiler_summaries["memray"], thresholds))

    # I/O syscall analysis (eBPF)
    if "ebpf" in profiler_summaries:
        findings.extend(_analyze_ebpf(profiler_summaries["ebpf"], thresholds))

    # Lock contention analysis
    if "lock_contention" in profiler_summaries:
        findings.extend(_analyze_lock_contention(profiler_summaries["lock_contention"], thresholds))

    # Thread scheduling analysis
    if "thread_sched" in profiler_summaries:
        findings.extend(_analyze_thread_sched(profiler_summaries["thread_sched"], thresholds))

    # Power / thermal throttling analysis
    if "power" in profiler_summaries:
        findings.extend(_analyze_power(profiler_summaries["power"], thresholds))

    # Sort by confidence (high > medium > low), then by rank placeholder
    confidence_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda d: confidence_order.get(d.confidence, 3))

    # Assign final ranks and trim to top_n
    result = findings[:top_n]
    for i, d in enumerate(result):
        d.rank = i + 1
    return result
