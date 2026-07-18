"""Tests for interval-union based busy-time metrics.

Concurrent events (kernels on multiple CUDA streams, nested operator spans)
must not double-count wall-clock time in derived percentages/fractions:
- nsys gpu_active_pct
- torch trace cpu_vs_gpu and per-phase gpu_us
- jax trace host/device split and infeed_stall_pct
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from perflab.profilers.interval_union import union_duration
from perflab.profilers.jax_profiler import _collect_jax_trace_metrics
from perflab.profilers.nsys_profiler import _extract_gpu_utilization
from perflab.profilers.pytorch_profiler import _parse_torch_trace

# ---------------------------------------------------------------------------
# union_duration helper
# ---------------------------------------------------------------------------

class TestUnionDuration:
    def test_empty(self):
        assert union_duration([]) == 0.0

    def test_disjoint_intervals_sum(self):
        assert union_duration([(0, 10), (20, 30)]) == 20

    def test_overlapping_intervals_merge(self):
        # (0,10) and (5,15) cover [0,15] → 15, not 20
        assert union_duration([(0, 10), (5, 15)]) == 15

    def test_identical_intervals_count_once(self):
        assert union_duration([(0, 10), (0, 10), (0, 10)]) == 10

    def test_nested_interval_ignored(self):
        assert union_duration([(0, 100), (10, 20)]) == 100

    def test_unsorted_input(self):
        assert union_duration([(20, 30), (0, 10), (5, 15)]) == 25

    def test_adjacent_intervals(self):
        assert union_duration([(0, 10), (10, 20)]) == 20

    def test_empty_and_inverted_intervals_ignored(self):
        assert union_duration([(5, 5), (10, 3), (0, 4)]) == 4

    def test_float_endpoints(self):
        assert union_duration([(0.0, 1.5), (1.0, 2.0)]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# nsys gpu_active_pct
# ---------------------------------------------------------------------------

def _make_kernel_db(rows):
    """In-memory nsys-like SQLite DB with CUPTI kernel rows (start, end, streamId)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        "(start INTEGER, end INTEGER, streamId INTEGER, "
        " demangledName TEXT, correlationId INTEGER)"
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, streamId) VALUES (?, ?, ?)",
        rows,
    )
    return conn


class TestNsysGpuActivePct:
    def test_concurrent_streams_do_not_double_count(self):
        """Two fully-overlapping kernels on two streams occupy the GPU once."""
        # Span [0, 100]; kernels [0, 50] on two streams + [90, 100]:
        # union busy = 50 + 10 = 60 → 60%. Summing would claim 110%.
        conn = _make_kernel_db([(0, 50, 1), (0, 50, 2), (90, 100, 1)])
        result: dict = {}
        _extract_gpu_utilization(conn, result)
        assert result["gpu_active_pct"] == pytest.approx(60.0)

    def test_cannot_exceed_100_pct(self):
        """Heavy multi-stream overlap must never report >100%."""
        conn = _make_kernel_db([(0, 100, s) for s in range(8)])
        result: dict = {}
        _extract_gpu_utilization(conn, result)
        assert result["gpu_active_pct"] <= 100.0
        assert result["gpu_active_pct"] == pytest.approx(100.0)

    def test_serial_kernels_unchanged(self):
        """Non-overlapping kernels behave as before (plain duty cycle)."""
        conn = _make_kernel_db([(0, 25, 1), (75, 100, 1)])
        result: dict = {}
        _extract_gpu_utilization(conn, result)
        assert result["gpu_active_pct"] == pytest.approx(50.0)

    def test_empty_table(self):
        conn = _make_kernel_db([])
        result: dict = {}
        _extract_gpu_utilization(conn, result)
        assert "gpu_active_pct" not in result


# ---------------------------------------------------------------------------
# torch trace cpu_vs_gpu and phases
# ---------------------------------------------------------------------------

def _write_trace(tmp_path, events):
    trace_path = tmp_path / "torch_trace.json"
    trace_path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return trace_path


class TestTorchTraceCpuVsGpu:
    def test_concurrent_gpu_kernels_use_union(self, tmp_path):
        """Two kernels overlapping on different streams count wall time once."""
        events = [
            # GPU: [1000, 2000] on two streams → union 1000 us, sum 2000 us
            {"ph": "X", "name": "kern_a", "cat": "kernel", "ts": 1000, "dur": 1000},
            {"ph": "X", "name": "kern_b", "cat": "kernel", "ts": 1000, "dur": 1000},
            # CPU op: [0, 500]
            {"ph": "X", "name": "aten::mm", "cat": "cpu_op", "ts": 0, "dur": 500},
        ]
        result = _parse_torch_trace(_write_trace(tmp_path, events))
        cg = result["cpu_vs_gpu"]
        assert cg["total_gpu_kernel_us"] == pytest.approx(1000.0)
        assert cg["total_cpu_op_us"] == pytest.approx(500.0)
        assert cg["ratio"] == pytest.approx(2.0)

    def test_nested_cpu_ops_use_union(self, tmp_path):
        """Nested operator spans (parent encloses child) count wall time once."""
        events = [
            # parent [0, 1000] encloses child [100, 300] → union 1000, sum 1200
            {"ph": "X", "name": "aten::linear", "cat": "cpu_op", "ts": 0, "dur": 1000},
            {"ph": "X", "name": "aten::mm", "cat": "cpu_op", "ts": 100, "dur": 200},
        ]
        result = _parse_torch_trace(_write_trace(tmp_path, events))
        assert result["cpu_vs_gpu"]["total_cpu_op_us"] == pytest.approx(1000.0)

    def test_phase_gpu_us_unions_concurrent_kernels(self, tmp_path):
        """Per-phase GPU time must not exceed the phase span due to overlap."""
        events = [
            {"ph": "X", "name": "## forward ##", "cat": "user_annotation",
             "ts": 0, "dur": 2000},
            # two concurrent kernels inside the phase: union 800, sum 1600
            {"ph": "X", "name": "kern_a", "cat": "kernel", "ts": 100, "dur": 800},
            {"ph": "X", "name": "kern_b", "cat": "kernel", "ts": 100, "dur": 800},
        ]
        result = _parse_torch_trace(_write_trace(tmp_path, events))
        phases = {p["name"]: p for p in result["phases"]}
        assert phases["forward"]["gpu_us"] == pytest.approx(800.0)
        assert phases["forward"]["cpu_us"] == pytest.approx(1200.0)

    def test_serial_events_unchanged(self, tmp_path):
        """Non-overlapping events give the same totals as plain summing."""
        events = [
            {"ph": "X", "name": "kern_a", "cat": "kernel", "ts": 0, "dur": 300},
            {"ph": "X", "name": "kern_b", "cat": "kernel", "ts": 500, "dur": 300},
            {"ph": "X", "name": "aten::mm", "cat": "cpu_op", "ts": 1000, "dur": 400},
            {"ph": "X", "name": "aten::relu", "cat": "cpu_op", "ts": 1500, "dur": 200},
        ]
        result = _parse_torch_trace(_write_trace(tmp_path, events))
        cg = result["cpu_vs_gpu"]
        assert cg["total_gpu_kernel_us"] == pytest.approx(600.0)
        assert cg["total_cpu_op_us"] == pytest.approx(600.0)


# ---------------------------------------------------------------------------
# jax trace host/device split and infeed stall
# ---------------------------------------------------------------------------

class TestJaxTraceMetrics:
    def test_concurrent_device_events_use_union(self, tmp_path):
        trace = {
            "traceEvents": [
                # two overlapping device events [0,100] and [50,150]:
                # union 150, sum 200
                {"cat": "device", "name": "matmul", "ts": 0, "dur": 100},
                {"cat": "device", "name": "conv", "ts": 50, "dur": 100},
                # host event [200, 250]
                {"cat": "host", "name": "prep", "ts": 200, "dur": 50},
            ]
        }
        (tmp_path / "trace.json").write_text(json.dumps(trace))
        result = _collect_jax_trace_metrics(tmp_path)
        assert result["device_time_us"] == pytest.approx(150.0)
        assert result["host_time_us"] == pytest.approx(50.0)
        assert result["device_fraction"] == pytest.approx(150.0 / 200.0, abs=0.001)

    def test_infeed_stall_pct_uses_union_denominator(self, tmp_path):
        trace = {
            "traceEvents": [
                # device busy [0, 80]; nested device event [10, 30] must not
                # inflate the denominator
                {"cat": "device", "name": "matmul", "ts": 0, "dur": 80},
                {"cat": "device", "name": "sub_op", "ts": 10, "dur": 20},
                # infeed [80, 100]
                {"cat": "host", "name": "infeed_enqueue", "ts": 80, "dur": 20},
            ]
        }
        (tmp_path / "trace.json").write_text(json.dumps(trace))
        result = _collect_jax_trace_metrics(tmp_path)
        # infeed 20 / total busy union 100 → 20%, not 20/120 = 16.7%
        assert result["infeed_stall_pct"] == pytest.approx(20.0, abs=0.1)

    def test_events_without_ts_fall_back_to_sum(self, tmp_path):
        """Legacy traces without timestamps keep the old summing behavior."""
        trace = {
            "traceEvents": [
                {"cat": "host", "dur": 5000, "name": "compute"},
                {"cat": "device", "dur": 15000, "name": "matmul"},
                {"cat": "host", "dur": 3000, "name": "prepare"},
            ]
        }
        (tmp_path / "trace.json").write_text(json.dumps(trace))
        result = _collect_jax_trace_metrics(tmp_path)
        assert result["host_time_us"] == 8000.0
        assert result["device_time_us"] == 15000.0
