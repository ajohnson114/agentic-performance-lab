from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnalysisThresholds:
    # -- Torch trace --
    gpu_cpu_ratio_low: float = 0.5
    sync_count_warn: int = 10
    mem_alloc_overhead_pct: float = 0.10
    gpu_kernel_dominance_pct: float = 70.0
    phase_dominance_pct: float = 60.0
    phase_gpu_fraction_low: float = 0.3

    # -- NCU --
    ncu_mem_throughput_high: float = 70.0
    ncu_compute_throughput_high: float = 70.0
    ncu_compute_low: float = 40.0
    ncu_mem_low: float = 40.0
    ncu_sm_util_low: float = 50.0
    ncu_sm_util_critical: float = 30.0
    ncu_mem_bound_high: float = 80.0
    ncu_mem_bound_critical: float = 90.0
    ncu_occupancy_low: float = 50.0
    ncu_regs_per_thread_high: int = 128
    ncu_branch_efficiency_low: float = 80.0
    ncu_warp_exec_efficiency_low: float = 80.0
    ncu_tc_util_low: float = 30.0  # Tensor Core utilization threshold
    ncu_bank_conflicts_high: float = 100.0  # flag if bank conflicts > 100
    ncu_sectors_per_request_high: float = 4.0  # >4 sectors/request = poorly coalesced
    ncu_stall_pct_high: float = 30.0  # flag dominant warp stall if > 30%

    # -- NSYS --
    nsys_gpu_fraction_low: float = 0.5
    nsys_api_overhead_high: float = 0.2
    nsys_kernel_dominance_pct: float = 80.0
    nsys_kernel_gap_us: float = 50.0
    nsys_transfer_ratio: float = 0.2
    nsys_gpu_imbalance_pct: float = 30.0  # flag if busiest/idlest GPU active_pct spread > 30pp

    # -- Communication (NCCL) -- shared across nsys and torch-profiler, both
    # of which detect NCCL kernels by name pattern rather than a profiler-
    # specific mechanism, so one threshold covers both sources.
    nccl_time_pct_high: float = 20.0  # flag if NCCL collectives take > 20% of GPU time

    # -- Linux perf --
    perf_ipc_low: float = 1.0
    perf_cache_miss_rate_high: float = 0.05
    perf_branch_miss_rate_high: float = 0.05
    perf_hotspot_dominance_pct: float = 50.0
    perf_cpus_utilized_low: float = 1.5

    # -- Host-device --
    host_device_sync_ratio: float = 0.2
    host_device_kernel_dur_low_us: float = 10.0
    host_device_kernel_count_high: int = 100
    host_device_transfer_ratio: float = 0.3
    host_device_gpu_active_low: float = 50.0

    # -- Metal trace --
    metal_gpu_fraction_low: float = 0.5
    metal_blit_ratio: float = 0.2
    metal_gpu_idle_pct_high: float = 30.0
    metal_submission_dominance: float = 0.7
    metal_alu_util_low: float = 50.0

    # -- Cross-profiler MPS --
    cross_gpu_cpu_ratio_low: float = 0.5
    cross_gpu_util_low: float = 0.3

    # -- I/O --
    io_hotspot_pct_high: float = 20.0
    io_non_gpu_fraction_high: float = 0.5

    # -- JAX / XLA --
    jax_recompilation_warn: int = 1
    jax_compilation_time_high_ms: float = 5000.0
    jax_compilation_fraction_high: float = 0.3
    jax_compilations_excessive: int = 5

    # -- NVTX phases --
    nvtx_phase_dominance_pct: float = 60.0
    nvtx_range_count_high: int = 50
    nvtx_avg_range_dur_low_ms: float = 1.0

    # -- Cross-reference (compiler remarks x perf annotate) --
    perf_annotate_hot_line_pct: float = 5.0    # min % for a line to be "hot"
    vec_width_gap_ratio: float = 2.0            # flag when hw/vec >= this
    cross_ref_hotspot_window: int = 3           # line number matching window

    # -- Memray (memory) --
    memray_peak_mb_warn: float = 4096.0          # flag if peak > 4 GB
    memray_top_allocator_dominance_pct: float = 50.0  # flag if single allocator > 50%

    # -- eBPF (I/O syscalls) --
    ebpf_read_p99_us_high: float = 10000.0       # flag if read p99 > 10ms
    ebpf_write_p99_us_high: float = 10000.0      # flag if write p99 > 10ms
    ebpf_syscall_count_high: int = 10000          # flag if total syscalls > 10K

    # -- Lock contention --
    lock_contention_ratio_high: float = 0.10      # flag if contended/acquired > 10%
    lock_total_wait_ms_high: float = 100.0        # flag if total lock wait > 100ms
    lock_false_sharing_hitm_high: int = 100       # flag if HITM count > 100

    # -- Thread scheduling --
    thread_sched_avg_delay_ms_high: float = 1.0   # flag if avg scheduling delay > 1ms
    thread_sched_migrations_high: int = 50        # flag if thread migrations > 50

    # -- Power / thermal --
    power_gpu_throttle_drop_pct: float = 10.0     # flag if GPU clock drops > 10% (thermal throttling)

    # -- TPU-specific --
    tpu_mxu_util_low: float = 30.0               # flag if MXU utilization < 30%
    tpu_padding_waste_pct_high: float = 20.0      # flag if >20% of compute wasted on padding
    tpu_infeed_stall_pct_high: float = 10.0       # flag if infeed stall > 10% of step time


@dataclass(frozen=True)
class Evidence:
    """Provenance for a :class:`BottleneckDiagnosis`'s ``bottleneck`` claim.

    Note: frozen=True auto-generates ``__hash__``, but ``metrics`` is a plain
    dict, which is unhashable -- hashing an ``Evidence`` (or a diagnosis that
    holds one) raises ``TypeError`` at call time. Equality still works fine.
    Never put one in a ``set`` or use it as a dict key.
    """

    level: str = "derived"
    """Strongest evidence level supporting the claim.

    - ``observed``  — restates a directly measured value with no threshold judgment
    - ``derived``   — deterministic computation or threshold comparison over observed
                      metrics (the common case for these rules)
    - ``inferred``  — heuristic pattern match, source-text hints, or cross-profiler
                      correlation that may be wrong even when the inputs are correct

    ``root_cause`` and ``suggested_actions`` on the owning diagnosis are ALWAYS
    inferred regardless of this field's value, and must be rendered as such.
    """

    rule_id: str = ""
    """Identifier of the rule that fired, e.g. ``ncu_sm_util_low``. Prefer the
    ``AnalysisThresholds`` attribute name when the rule is threshold-driven."""

    metrics: dict[str, float] = field(default_factory=dict)
    """The observed values this diagnosis was computed from, e.g.
    ``{"sm_util_pct": 23.0, "threshold": 50.0}``. These are facts; the surrounding
    prose is not."""

    def to_dict(self) -> dict:
        return {"level": self.level, "rule_id": self.rule_id, "metrics": self.metrics}


@dataclass
class BottleneckDiagnosis:
    rank: int
    bottleneck: str           # e.g. "Low SM utilization (23%)"
    root_cause: str           # e.g. "Insufficient parallelism or small kernel launches"
    confidence: str           # "high" | "medium" | "low"
    suggested_actions: list[str] = field(default_factory=list)

    # --- provenance (added) ---
    evidence: Evidence = field(default_factory=Evidence)
