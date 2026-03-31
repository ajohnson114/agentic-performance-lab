"""Tests for perflab.profilers.thread_sched."""
from __future__ import annotations

from perflab.profilers.thread_sched import (
    _parse_sched_latency,
    _parse_sched_timehist,
    format_sched_summary,
)


class TestParseSchedLatency:
    def test_parse_basic_line(self):
        text = (
            "-------------------------------------------------\n"
            " Task                  |   Runtime ms  | Switches | Avg delay ms    | Max delay ms    |\n"
            "-------------------------------------------------\n"
            " matmul_bin:12345      |   1234.567 ms |      42  | avg:    0.012 ms | max:    0.456 ms |\n"
            " worker:12346          |    567.890 ms |      10  | avg:    0.005 ms | max:    0.123 ms |\n"
        )
        results = _parse_sched_latency(text)
        assert len(results) == 2
        assert results[0]["task"] == "matmul_bin"
        assert results[0]["pid"] == 12345
        assert results[0]["runtime_ms"] == 1234.567
        assert results[0]["switches"] == 42
        assert results[0]["avg_delay_ms"] == 0.012
        assert results[0]["max_delay_ms"] == 0.456

    def test_empty_input(self):
        assert _parse_sched_latency("") == []

    def test_no_matching_lines(self):
        assert _parse_sched_latency("some random text\nanother line\n") == []


class TestParseSchedTimehist:
    def test_parse_cpu_lines(self):
        text = (
            "CPU 0:   1.234 sec total run time\n"
            "CPU 1:   2.345 sec total run time\n"
            "migrations: 15\n"
        )
        result = _parse_sched_timehist(text)
        assert len(result["cpus"]) == 2
        assert result["cpus"][0]["cpu"] == 0
        assert abs(result["cpus"][0]["run_sec"] - 1.234) < 0.001
        assert abs(result["total_run_ms"] - 3579.0) < 1.0
        assert result["migrations"] == 15

    def test_empty_input(self):
        result = _parse_sched_timehist("")
        assert result["cpus"] == []


class TestFormatSchedSummary:
    def test_format_with_latency(self):
        summary = {
            "latency": [
                {"task": "matmul", "pid": 123, "runtime_ms": 100.0,
                 "switches": 5, "avg_delay_ms": 0.01, "max_delay_ms": 0.1},
            ],
        }
        text = format_sched_summary(summary)
        assert "matmul" in text
        assert "100.0ms" in text

    def test_format_with_timehist(self):
        summary = {
            "timehist": {
                "cpus": [{"cpu": 0, "run_sec": 1.0}],
                "total_run_ms": 1000.0,
                "migrations": 10,
            },
        }
        text = format_sched_summary(summary)
        assert "1000.0ms" in text
        assert "migrations: 10" in text

    def test_format_empty(self):
        assert format_sched_summary({}) == ""
