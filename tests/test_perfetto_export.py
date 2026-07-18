"""Tests for perflab.profilers.perfetto_export."""
from __future__ import annotations

import json
from pathlib import Path

from perflab.profilers.perfetto_export import export_perfetto_trace


class TestExportPerfettoTrace:
    def test_returns_none_with_no_data(self, tmp_path: Path):
        out = tmp_path / "trace.json"
        result = export_perfetto_trace(out)
        assert result is None
        assert not out.exists()

    def test_pyspy_hotspots_produce_events(self, tmp_path: Path):
        out = tmp_path / "trace.json"
        pyspy = {
            "hotspots": [
                {"function": "matmul", "pct": 60.0, "location": "matmul.py:10"},
                {"function": "transpose", "pct": 20.0},
            ],
            "total_samples": 1000,
        }
        result = export_perfetto_trace(out, pyspy_summary=pyspy)
        assert result == out
        assert out.exists()

        data = json.loads(out.read_text())
        assert "traceEvents" in data
        events = data["traceEvents"]

        # Should have thread_name metadata + 2 hotspot events
        x_events = [e for e in events if e.get("ph") == "X"]
        assert len(x_events) == 2
        assert x_events[0]["name"] == "matmul"
        assert x_events[0]["args"]["location"] == "matmul.py:10"
        assert x_events[1]["name"] == "transpose"

    def test_perf_counters_produce_counter_events(self, tmp_path: Path):
        out = tmp_path / "trace.json"
        perf = {"ipc": 1.5, "cache_miss_rate": 0.03}
        result = export_perfetto_trace(out, perf_summary=perf)
        assert result is not None

        data = json.loads(out.read_text())
        c_events = [e for e in data["traceEvents"] if e.get("ph") == "C"]
        assert len(c_events) == 2
        metric_names = {list(e["args"].keys())[0] for e in c_events}
        assert "IPC" in metric_names
        assert "Cache Miss Rate" in metric_names

    def test_perf_hotspots_produce_events(self, tmp_path: Path):
        out = tmp_path / "trace.json"
        perf = {
            "hotspots": [
                {"function": "sgemm", "pct": 45.0, "module": "libcblas.so"},
            ],
        }
        export_perfetto_trace(out, perf_summary=perf)
        data = json.loads(out.read_text())
        x_events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
        assert any(e["name"] == "sgemm" for e in x_events)

    def test_memray_allocators_produce_events(self, tmp_path: Path):
        out = tmp_path / "trace.json"
        memray = {
            "top_allocators": [
                {"function": "numpy.zeros", "size_mb": 512.0, "location": "train.py:42"},
            ],
        }
        export_perfetto_trace(out, memray_summary=memray)
        data = json.loads(out.read_text())
        x_events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
        assert any(e["cat"] == "memory" for e in x_events)

    def test_metadata_sets_process_name(self, tmp_path: Path):
        out = tmp_path / "trace.json"
        export_perfetto_trace(
            out,
            pyspy_summary={"hotspots": [{"function": "f", "pct": 100}], "total_samples": 10},
            metadata={"task_name": "matmul/cpp"},
        )
        data = json.loads(out.read_text())
        m_events = [e for e in data["traceEvents"] if e.get("ph") == "M" and e.get("name") == "process_name"]
        assert len(m_events) == 1
        assert m_events[0]["args"]["name"] == "matmul/cpp"

    def test_combined_sources(self, tmp_path: Path):
        out = tmp_path / "trace.json"
        export_perfetto_trace(
            out,
            pyspy_summary={"hotspots": [{"function": "f1", "pct": 50}], "total_samples": 100},
            perf_summary={"ipc": 1.2, "hotspots": [{"function": "f2", "pct": 30}]},
            memray_summary={"top_allocators": [{"function": "f3", "size_mb": 10}]},
        )
        data = json.loads(out.read_text())
        # Should have events from all three sources
        x_events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
        names = {e["name"] for e in x_events}
        assert "f1" in names
        assert "f2" in names
        assert "f3" in names

    def test_creates_parent_directories(self, tmp_path: Path):
        out = tmp_path / "sub" / "dir" / "trace.json"
        export_perfetto_trace(
            out,
            pyspy_summary={"hotspots": [{"function": "f", "pct": 100}], "total_samples": 10},
        )
        assert out.exists()
