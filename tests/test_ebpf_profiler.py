"""Tests for perflab.profilers.ebpf_profiler."""
from __future__ import annotations

from perflab.profilers.ebpf_profiler import _parse_bpftrace_output, _parse_histogram


class TestParseBpftraceOutput:
    def test_parse_read_write_counts(self):
        text = "@read_count: 1234\n@write_count: 567\n"
        result = _parse_bpftrace_output(text)
        assert result["read_syscalls"] == 1234
        assert result["write_syscalls"] == 567

    def test_parse_read_write_bytes(self):
        text = "@read_bytes: 1048576\n@write_bytes: 524288\n"
        result = _parse_bpftrace_output(text)
        assert result["read_bytes"] == 1048576
        assert result["write_bytes"] == 524288

    def test_parse_empty_text(self):
        result = _parse_bpftrace_output("")
        assert result == {}

    def test_partial_data(self):
        text = "@read_count: 42\n"
        result = _parse_bpftrace_output(text)
        assert result["read_syscalls"] == 42
        assert "write_syscalls" not in result


class TestParseHistogram:
    def test_returns_none_for_missing_var(self):
        result = _parse_histogram("no histogram here", "@read_ns")
        assert result is None

    def test_parse_simple_histogram(self):
        text = """@read_ns:
[1, 2)  10
[2, 4)  20
[4, 8)  50
[8, 16)  15
[16, 32)  5
"""
        result = _parse_histogram(text, "@read_ns")
        assert result is not None
        assert result["total_count"] == 100
        assert "p50_ns" in result
        assert "p90_ns" in result

    def test_empty_histogram(self):
        text = "@read_ns:\n\nSome other text"
        result = _parse_histogram(text, "@read_ns")
        assert result is None
