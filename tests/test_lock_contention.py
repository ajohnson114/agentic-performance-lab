"""Tests for perflab.profilers.lock_contention."""
from __future__ import annotations

from perflab.profilers.lock_contention import (
    _parse_perf_c2c_report,
    _parse_perf_lock_report,
)


class TestParsePerfLockReport:
    def test_parse_basic_lock_stats(self):
        text = """
Name        acquired  contended   avg wait (ns)   total wait (ns)   max wait (ns)
----------- --------  ---------   -------------   ---------------   -------------
mutex_a          500         42           1234             51828           5000
mutex_b         1000         10            500              5000           2000
"""
        result = _parse_perf_lock_report(text)
        assert len(result["locks"]) == 2
        assert result["locks"][0]["name"] == "mutex_a"
        assert result["locks"][0]["acquired"] == 500
        assert result["locks"][0]["contended"] == 42
        assert result["total_contended"] == 52

    def test_parse_empty_text(self):
        result = _parse_perf_lock_report("")
        assert result["locks"] == []
        assert result["total_contended"] == 0

    def test_parse_no_table(self):
        text = "Some random text without lock data"
        result = _parse_perf_lock_report(text)
        assert result["locks"] == []

    def test_total_wait_accumulation(self):
        text = """
Name        acquired  contended   avg wait (ns)   total wait (ns)   max wait (ns)
----------- --------  ---------   -------------   ---------------   -------------
lock1          100         10            100              1000            500
lock2          200         20            200              4000            800
"""
        result = _parse_perf_lock_report(text)
        assert result["total_wait_ns"] == 5000


class TestParsePerfC2cReport:
    def test_parse_hitm_totals(self):
        text = """
Total records : 54321
Total HITM : 1234
Total Store : 9876
"""
        result = _parse_perf_c2c_report(text)
        assert result["total_hitm"] == 1234
        assert result["total_store"] == 9876
        assert result["total_records"] == 54321

    def test_parse_empty_text(self):
        result = _parse_perf_c2c_report("")
        assert result["total_hitm"] == 0
        assert result["false_sharing_lines"] == []

    def test_parse_cacheline_entries(self):
        text = """
Total HITM : 100
0x7fff1234 | 42 | 10
0x7fff5678 | 5 | 20
0x7fff9abc | 0 | 30
"""
        result = _parse_perf_c2c_report(text)
        # Only entries with hitm > 0
        assert len(result["false_sharing_lines"]) == 2
        assert result["false_sharing_lines"][0]["hitm"] == 42
        assert result["false_sharing_lines"][1]["hitm"] == 5

    def test_false_sharing_lines_sorted_by_hitm(self):
        text = """
0x1000 | 5 | 10
0x2000 | 50 | 20
0x3000 | 25 | 15
"""
        result = _parse_perf_c2c_report(text)
        hitms = [e["hitm"] for e in result["false_sharing_lines"]]
        assert hitms == sorted(hitms, reverse=True)

    def test_false_sharing_lines_capped_at_10(self):
        lines = "\n".join(f"0x{i:04x} | {i+1} | {i}" for i in range(20))
        result = _parse_perf_c2c_report(lines)
        assert len(result["false_sharing_lines"]) <= 10
