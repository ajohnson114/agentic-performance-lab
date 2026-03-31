"""Tests for new bottleneck analysis rules: memray, ebpf, lock_contention, thread_sched, power."""
from __future__ import annotations

from perflab.analyzers.bottleneck_analyzer import (
    AnalysisThresholds,
    BottleneckDiagnosis,
    _analyze_ebpf,
    _analyze_lock_contention,
    _analyze_memray,
    _analyze_power,
    _analyze_thread_sched,
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
