"""Tests for perflab.profilers.memray_profiler."""
from __future__ import annotations

from perflab.profilers.memray_profiler import _parse_memray_stats


class TestParseMemrayStats:
    def test_parse_total_allocations(self):
        text = "Total allocations: 1,234,567\nOther stuff"
        result = _parse_memray_stats(text)
        assert result["total_allocations"] == 1234567

    def test_parse_total_memory_mb(self):
        text = "Total memory allocated: 256.5 MB"
        result = _parse_memray_stats(text)
        assert abs(result["total_allocated_mb"] - 256.5) < 0.1

    def test_parse_total_memory_gb(self):
        text = "Total memory allocated: 2.5 GB"
        result = _parse_memray_stats(text)
        assert abs(result["total_allocated_mb"] - 2560.0) < 0.1

    def test_parse_total_memory_kb(self):
        text = "Total memory allocated: 512 KB"
        result = _parse_memray_stats(text)
        assert abs(result["total_allocated_mb"] - 0.5) < 0.1

    def test_parse_peak_memory(self):
        text = "Peak memory usage: 128.3 MB"
        result = _parse_memray_stats(text)
        assert abs(result["peak_memory_mb"] - 128.3) < 0.1

    def test_parse_empty_text(self):
        result = _parse_memray_stats("")
        assert result == {}

    def test_parse_allocators_table(self):
        text = """Total allocations: 100
Top allocators:
  1.  64.0 MB  50  numpy.zeros  train.py:42
  2.  32.0 MB  25  torch.empty  model.py:10
"""
        result = _parse_memray_stats(text)
        assert result["total_allocations"] == 100
        allocs = result.get("top_allocators", [])
        assert len(allocs) == 2
        assert allocs[0]["function"] == "numpy.zeros"
        assert abs(allocs[0]["size_mb"] - 64.0) < 0.1
        assert allocs[0]["count"] == 50

    def test_no_false_positive_on_unrelated_text(self):
        text = "This is a regular log message with no profiler data"
        result = _parse_memray_stats(text)
        assert "total_allocations" not in result
        assert "peak_memory_mb" not in result
