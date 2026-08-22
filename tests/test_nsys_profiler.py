"""Tests for nsys SQLite extraction, focused on multi-GPU correlation correctness.

CUDA assigns stream IDs per-device, so a device-blind extraction would merge
kernels launched on different GPUs whenever they happen to reuse the same
stream_id. These tests build a minimal in-memory nsys-shaped SQLite DB
spanning two devices to lock in that deviceId is threaded through.
"""
from __future__ import annotations

import sqlite3

from perflab.profilers.nsys_profiler import (
    _extract_cpu_gpu_correlation,
    _extract_device_count,
    _extract_gpu_utilization,
    _extract_nccl_time,
    _extract_per_stream_gaps,
    _extract_top_kernels_by_device,
)


def _make_multi_gpu_db():
    """Two GPUs, each running a 'sgemm' kernel on stream 1 (same stream_id, different device)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        "(correlationId INTEGER, demangledName TEXT, start INTEGER, end INTEGER, "
        " streamId INTEGER, deviceId INTEGER)"
    )
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
        "(correlationId INTEGER, nameId INTEGER, start INTEGER, end INTEGER)"
    )
    conn.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")

    conn.execute("INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel')")
    # GPU 0, stream 1
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100, 'sgemm', 0, 1000, 1, 0)"
    )
    conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (100, 1, 0, 5)")
    # GPU 1, stream 1 -- same stream_id as above, different device
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (200, 'sgemm', 0, 900, 1, 1)"
    )
    conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (200, 1, 0, 5)")
    return conn


class TestCpuGpuCorrelationMultiGpu:
    def test_device_id_present_on_each_correlation(self):
        conn = _make_multi_gpu_db()
        result: dict = {}
        _extract_cpu_gpu_correlation(conn, result)
        corrs = result["cpu_gpu_correlations"]
        assert len(corrs) == 2
        assert {c["device_id"] for c in corrs} == {0, 1}


class TestDeviceCount:
    def test_counts_distinct_devices(self):
        conn = _make_multi_gpu_db()
        result: dict = {}
        _extract_device_count(conn, result)
        assert result["gpu_device_count"] == 2

    def test_single_device(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER)")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0)")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0)")
        result: dict = {}
        _extract_device_count(conn, result)
        assert result["gpu_device_count"] == 1

    def test_empty_table(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INTEGER)")
        result: dict = {}
        _extract_device_count(conn, result)
        assert "gpu_device_count" not in result


class TestPerStreamGapsMultiGpu:
    def test_same_stream_id_different_devices_kept_separate(self):
        """Two devices reusing stream_id=1 must land in separate buckets,
        keyed by device -- not merged as if it were one stream's activity."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, streamId INTEGER, deviceId INTEGER)"
        )
        rows = [
            # GPU 0, stream 1: two kernels with a big gap between them -> low utilization
            (0, 100, 1, 0),
            (10_000, 10_100, 1, 0),
            # GPU 1, stream 1: same stream_id, back-to-back -> high utilization
            (0, 100, 1, 1),
            (100, 200, 1, 1),
        ]
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, streamId, deviceId) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        result: dict = {}
        _extract_per_stream_gaps(conn, result)

        assert set(result["stream_utilization"].keys()) == {"0:1", "1:1"}
        gpu0 = result["stream_utilization"]["0:1"]
        gpu1 = result["stream_utilization"]["1:1"]
        assert gpu0["device_id"] == 0
        assert gpu1["device_id"] == 1
        # GPU 0's huge gap tanks its utilization; GPU 1 stayed busy. If the
        # extraction merged them by stream_id alone, this distinction is lost.
        assert gpu0["active_pct"] < gpu1["active_pct"]


class TestGpuUtilizationMultiGpu:
    def test_per_device_utilization_uses_shared_span(self):
        """A busy GPU 0 and a mostly-idle GPU 1 must be distinguishable --
        the aggregate 'any device busy' union alone would mask GPU 1 being idle."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, deviceId INTEGER)"
        )
        rows = [
            (0, 1000, 0),  # GPU 0: busy the entire trace span
            (0, 100, 1),   # GPU 1: busy only the first 10%
        ]
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, deviceId) VALUES (?, ?, ?)",
            rows,
        )
        result: dict = {}
        _extract_gpu_utilization(conn, result)

        # Aggregate metric looks fully healthy (GPU 0 covers the whole span)...
        assert result["gpu_active_pct"] == 100.0
        # ...but the per-device breakdown reveals GPU 1 sitting mostly idle.
        assert result["gpu_active_pct_by_device"][0] == 100.0
        assert result["gpu_active_pct_by_device"][1] == 10.0

    def test_single_device_no_breakdown(self):
        """Single-GPU traces get no gpu_active_pct_by_device key at all --
        output stays identical to before this feature existed."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, deviceId INTEGER)"
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, deviceId) VALUES (0, 100, 0)"
        )
        result: dict = {}
        _extract_gpu_utilization(conn, result)
        assert "gpu_active_pct_by_device" not in result


class TestTopKernelsByDeviceMultiGpu:
    def test_per_device_breakdown_reveals_different_dominant_kernels(self):
        """GPU 0 is dominated by sgemm, GPU 1 by conv2d -- the device-blind
        top_kernels ranking sums both into one list and can't show this."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(correlationId INTEGER, demangledName TEXT, start INTEGER, end INTEGER, "
            " streamId INTEGER, deviceId INTEGER)"
        )
        rows = [
            (1, "sgemm", 0, 900, 0, 0),
            (2, "relu", 900, 1000, 0, 0),
            (3, "conv2d", 0, 950, 0, 1),
            (4, "relu", 950, 1000, 0, 1),
        ]
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "(correlationId, demangledName, start, end, streamId, deviceId) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        result: dict = {}
        _extract_top_kernels_by_device(conn, result)
        by_device = result["top_kernels_by_device"]
        assert by_device[0][0]["name"] == "sgemm"
        assert by_device[1][0]["name"] == "conv2d"

    def test_single_device_no_breakdown(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(demangledName TEXT, start INTEGER, end INTEGER, deviceId INTEGER)"
        )
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ('sgemm', 0, 100, 0)")
        result: dict = {}
        _extract_top_kernels_by_device(conn, result)
        assert "top_kernels_by_device" not in result


class TestNcclTimeMultiGpu:
    def test_nccl_pct_and_per_device_breakdown(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(demangledName TEXT, start INTEGER, end INTEGER, deviceId INTEGER)"
        )
        rows = [
            # GPU 0: 90% compute, 10% NCCL
            ("sgemm", 0, 900, 0),
            ("ncclKernel_AllReduce_RING", 900, 1000, 0),
            # GPU 1: 50% compute, 50% NCCL -- lagging on the collective
            ("sgemm", 0, 500, 1),
            ("ncclKernel_AllReduce_RING", 500, 1000, 1),
        ]
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "(demangledName, start, end, deviceId) VALUES (?, ?, ?, ?)",
            rows,
        )
        result: dict = {}
        _extract_nccl_time(conn, result)
        assert result["nccl_pct"] == 30.0  # (100 + 500) / 2000
        by_device = result["nccl_pct_by_device"]
        assert by_device[0] == 10.0
        assert by_device[1] == 50.0

    def test_no_nccl_kernels_no_keys(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(demangledName TEXT, start INTEGER, end INTEGER, deviceId INTEGER)"
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ('sgemm', 0, 100, 0)"
        )
        result: dict = {}
        _extract_nccl_time(conn, result)
        assert "nccl_pct" not in result
        assert "nccl_pct_by_device" not in result

    def test_single_device_nccl_no_by_device_breakdown(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(demangledName TEXT, start INTEGER, end INTEGER, deviceId INTEGER)"
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "(demangledName, start, end, deviceId) VALUES (?, ?, ?, ?)",
            [("sgemm", 0, 900, 0), ("ncclAllReduce", 900, 1000, 0)],
        )
        result: dict = {}
        _extract_nccl_time(conn, result)
        assert result["nccl_pct"] == 10.0
        assert "nccl_pct_by_device" not in result
