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

    def test_suffixed_buckets_use_binary_multipliers(self):
        # bpftrace hist() bounds carry binary-magnitude suffixes:
        # [1K, 2K) means [1024, 2048), not [1, 2).
        text = """@read_ns:
[512, 1K)  10
[1K, 2K)  60
[2K, 4K)  20
[8M, 16M)  10
"""
        result = _parse_histogram(text, "@read_ns")
        assert result is not None
        assert result["total_count"] == 100
        assert result["p50_ns"] == (1024 + 2048) // 2
        assert result["p90_ns"] == (2048 + 4096) // 2
        assert result["p99_ns"] == ((8 << 20) + (16 << 20)) // 2
