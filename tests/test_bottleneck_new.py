"""Tests for new bottleneck analysis rules: memray, ebpf, lock_contention, thread_sched, power."""
from __future__ import annotations

from perflab.analyzers.bottleneck_analyzer import (
    AnalysisThresholds,
    _analyze_ebpf,
    _analyze_gpu_attribution,
    _analyze_lock_contention,
    _analyze_memray,
    _analyze_power,
    _analyze_thread_sched,
    _analyze_torch_trace,
    diagnose_bottlenecks,
)


def _default_thresholds() -> AnalysisThresholds:
    return AnalysisThresholds()


# -- memray ------------------------------------------------------------------

def test_analyze_memray_high_peak():
    summary = {"peak_memory_mb": 8000}
    findings = _analyze_memray(summary, _default_thresholds())
    assert len(findings) >= 1
    assert any("peak memory" in f.bottleneck.lower() for f in findings)
    assert any("8000" in f.bottleneck for f in findings)


def test_analyze_memray_dominant_allocator():
    summary = {
        "peak_memory_mb": 100,  # below threshold
        "total_allocated_mb": 1000,
        "top_allocators": [
            {"function": "big_alloc", "size_mb": 600},  # 60% > 50% threshold
        ],
    }
    findings = _analyze_memray(summary, _default_thresholds())
    assert len(findings) >= 1
    assert any("big_alloc" in f.bottleneck for f in findings)
    assert any("60%" in f.bottleneck for f in findings)


def test_analyze_memray_no_issues():
    summary = {
        "peak_memory_mb": 512,
        "total_allocated_mb": 1000,
        "top_allocators": [
            {"function": "small_alloc", "size_mb": 100},  # 10% < 50%
        ],
    }
    findings = _analyze_memray(summary, _default_thresholds())
    assert findings == []


# -- ebpf --------------------------------------------------------------------

def test_analyze_ebpf_high_read_latency():
    # p99_ns = 20_000_000 => 20_000 us > 10_000 us threshold
    summary = {
        "read_latency": {"p99_ns": 20_000_000},
        "write_latency": {},
        "read_syscalls": 0,
        "write_syscalls": 0,
    }
    findings = _analyze_ebpf(summary, _default_thresholds())
    assert len(findings) >= 1
    assert any("read latency" in f.bottleneck.lower() for f in findings)


def test_analyze_ebpf_excessive_syscalls():
    summary = {
        "read_latency": {},
        "write_latency": {},
        "read_syscalls": 30_000,
        "write_syscalls": 20_000,  # total = 50_000 > 10_000
    }
    findings = _analyze_ebpf(summary, _default_thresholds())
    assert len(findings) >= 1
    assert any("syscall" in f.bottleneck.lower() for f in findings)


def test_analyze_ebpf_no_issues():
    summary = {
        "read_latency": {"p99_ns": 1_000_000},  # 1_000 us < 10_000 us
        "write_latency": {"p99_ns": 500_000},
        "read_syscalls": 100,
        "write_syscalls": 50,
    }
    findings = _analyze_ebpf(summary, _default_thresholds())
    assert findings == []


# -- lock contention ---------------------------------------------------------

def test_analyze_lock_contention_high_ratio():
    summary = {
        "lock_stats": {
            "locks": [{"acquired": 1000}],
            "total_contended": 200,  # 200/1000 = 20% > 10%
            "total_wait_ns": 0,
        },
    }
    findings = _analyze_lock_contention(summary, _default_thresholds())
    assert len(findings) >= 1
    assert any("contention" in f.bottleneck.lower() for f in findings)


def test_analyze_lock_contention_false_sharing():
    summary = {
        "lock_stats": {
            "locks": [],
            "total_contended": 0,
            "total_wait_ns": 0,
        },
        "c2c_stats": {
            "total_hitm": 500,  # > 100 threshold
        },
    }
    findings = _analyze_lock_contention(summary, _default_thresholds())
    assert len(findings) >= 1
    assert any("false sharing" in f.bottleneck.lower() for f in findings)
    assert any("500" in f.bottleneck for f in findings)


def test_analyze_lock_contention_no_issues():
    summary = {
        "lock_stats": {
            "locks": [],
            "total_contended": 0,
            "total_wait_ns": 0,
        },
    }
    findings = _analyze_lock_contention(summary, _default_thresholds())
    assert findings == []


# -- thread scheduling -------------------------------------------------------

def test_analyze_thread_sched_high_delay():
    summary = {
        "latency": [
            {"task": "worker-1", "avg_delay_ms": 5.0},  # > 1.0 ms threshold
            {"task": "worker-2", "avg_delay_ms": 0.2},
        ],
        "timehist": {},
    }
    findings = _analyze_thread_sched(summary, _default_thresholds())
    assert len(findings) >= 1
    assert any("scheduling delay" in f.bottleneck.lower() for f in findings)
    assert any("worker-1" in f.bottleneck for f in findings)


def test_analyze_thread_sched_excessive_migrations():
    summary = {
        "latency": [],
        "timehist": {"migrations": 200},  # > 50 threshold
    }
    findings = _analyze_thread_sched(summary, _default_thresholds())
    assert len(findings) >= 1
    assert any("migration" in f.bottleneck.lower() for f in findings)


# -- power / thermal ---------------------------------------------------------

def test_analyze_power_throttling():
    # 8 samples: first 2 at 300W, last 2 at 200W => 33% drop > 10%
    samples = [
        {"watts": 300}, {"watts": 300},
        {"watts": 280}, {"watts": 260},
        {"watts": 240}, {"watts": 220},
        {"watts": 200}, {"watts": 200},
    ]
    summary = {"gpu_power": {"power_samples": samples}}
    findings = _analyze_power(summary, _default_thresholds())
    assert len(findings) >= 1
    assert any("throttl" in f.bottleneck.lower() for f in findings)


def test_analyze_power_no_throttle():
    # Stable power: no drop
    samples = [
        {"watts": 250}, {"watts": 252},
        {"watts": 248}, {"watts": 251},
        {"watts": 249}, {"watts": 250},
        {"watts": 251}, {"watts": 250},
    ]
    summary = {"gpu_power": {"power_samples": samples}}
    findings = _analyze_power(summary, _default_thresholds())
    assert findings == []


# -- torch trace: CPU-only vs MPS vs real-GPU-low-ratio ----------------------

def test_analyze_torch_trace_cpu_only_no_gpu_underutilized_finding():
    """A genuine CPU-only run (no GPU present/used) must not be diagnosed as
    'GPU underutilized' -- there's no GPU to underutilize."""
    summary = {
        "cpu_vs_gpu": {
            "total_cpu_op_us": 50_000.0,
            "total_gpu_kernel_us": 0.0,
            "ratio": 0.0,
        },
    }
    findings = _analyze_torch_trace(summary, device="cpu", thresholds=_default_thresholds())
    assert not any("underutilized" in f.bottleneck.lower() for f in findings)
    assert any("cpu-only" in f.bottleneck.lower() or "cpu only" in f.bottleneck.lower() for f in findings)
    cpu_only = next(f for f in findings if "cpu-only" in f.bottleneck.lower() or "cpu only" in f.bottleneck.lower())
    assert cpu_only.confidence == "low"


def test_analyze_torch_trace_cpu_only_no_device_specified():
    """device=None (device unknown) with zero GPU kernels should also be treated
    as CPU-only, not GPU-underutilized -- the common case for a GPU-less machine."""
    summary = {
        "cpu_vs_gpu": {
            "total_cpu_op_us": 50_000.0,
            "total_gpu_kernel_us": 0.0,
            "ratio": 0.0,
        },
    }
    findings = _analyze_torch_trace(summary, device=None, thresholds=_default_thresholds())
    assert not any("underutilized" in f.bottleneck.lower() for f in findings)


def test_analyze_torch_trace_mps_timing_unavailable_unaffected():
    """MPS's existing 'timing unavailable' explanation must keep working."""
    summary = {
        "cpu_vs_gpu": {
            "total_cpu_op_us": 50_000.0,
            "total_gpu_kernel_us": 0.0,
            "ratio": 0.0,
        },
    }
    findings = _analyze_torch_trace(summary, device="mps", thresholds=_default_thresholds())
    assert any("timing unavailable" in f.bottleneck.lower() for f in findings)
    assert not any("cpu-only" in f.bottleneck.lower() or "cpu only" in f.bottleneck.lower() for f in findings)


def test_analyze_torch_trace_real_gpu_low_ratio_still_flagged():
    """A real GPU with nonzero kernel time but a low ratio is still a genuine
    CPU-dispatch bottleneck and must still be flagged 'GPU underutilized'."""
    summary = {
        "cpu_vs_gpu": {
            "total_cpu_op_us": 100_000.0,
            "total_gpu_kernel_us": 10_000.0,  # ratio = 0.1 < gpu_cpu_ratio_low (0.5)
            "ratio": 0.1,
        },
        "top_gpu_kernels": [{"name": "sgemm_kernel", "total_us": 10_000.0, "count": 5, "pct": 100.0}],
    }
    findings = _analyze_torch_trace(summary, device="cuda", thresholds=_default_thresholds())
    assert any("underutilized" in f.bottleneck.lower() for f in findings)
    underutilized = next(f for f in findings if "underutilized" in f.bottleneck.lower())
    assert underutilized.confidence == "high"


# -- GPU attribution: findings must not be dropped when correlations missing -

def test_analyze_gpu_attribution_without_correlations_still_yields_findings():
    """nsys can omit cpu_gpu_correlations (sqlite3.OperationalError, empty rows)
    while still providing top_kernels/per_stream_gaps/stream_utilization --
    those should still produce findings instead of being silently dropped."""
    nsys_summary = {
        "top_kernels": [{"name": "big_kernel", "pct": 30.0, "total_ms": 100.0}],
        "stream_utilization": {"0": {"active_pct": 20.0, "kernel_count": 5}},
        "per_stream_gaps": {"0": {"max_gap_us": 500.0}},
    }
    findings = _analyze_gpu_attribution(nsys_summary, None, _default_thresholds())
    assert len(findings) >= 1
    assert any("idle" in f.bottleneck.lower() for f in findings)


def test_analyze_gpu_attribution_no_data_returns_empty():
    findings = _analyze_gpu_attribution({}, None, _default_thresholds())
    assert findings == []


# -- integration: diagnose_bottlenecks with all 5 new profilers ---------------

def test_diagnose_bottlenecks_includes_new_profilers():
    profiler_summaries = {
        "memray": {
            "peak_memory_mb": 8000,
        },
        "ebpf": {
            "read_latency": {"p99_ns": 50_000_000},
            "write_latency": {},
            "read_syscalls": 0,
            "write_syscalls": 0,
        },
        "lock_contention": {
            "lock_stats": {
                "locks": [{"acquired": 100}],
                "total_contended": 50,
                "total_wait_ns": 0,
            },
        },
        "thread_sched": {
            "latency": [{"task": "main", "avg_delay_ms": 10.0}],
            "timehist": {"migrations": 200},
        },
        "power": {
            "gpu_power": {
                "power_samples": [
                    {"watts": 300}, {"watts": 300},
                    {"watts": 250}, {"watts": 200},
                    {"watts": 150}, {"watts": 150},
                    {"watts": 150}, {"watts": 150},
                ],
            },
        },
    }
    # Allow enough results to see all profiler categories
    findings = diagnose_bottlenecks(profiler_summaries, program_type="python", top_n=20)
    assert len(findings) >= 5
    all_text = " ".join(f.bottleneck.lower() for f in findings)
    assert "memory" in all_text or "peak" in all_text  # memray
    assert "latency" in all_text or "read" in all_text  # ebpf
    assert "contention" in all_text                      # lock
    assert "scheduling" in all_text or "migration" in all_text  # thread_sched
    assert "throttl" in all_text                         # power
