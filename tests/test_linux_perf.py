"""Tests for perf script parsing, focused on multi-process CPU correctness.

perf inherits into every forked/exec'd child of the traced command by
default, so a multiprocessing.Pool / ProcessPoolExecutor benchmark's samples
already span every worker's pid in the raw perf script output -- these tests
lock in that pid is actually extracted and attributed correctly.
"""
from __future__ import annotations

from perflab.profilers.linux_perf import _parse_perf_script_pid_shares


def _write_script(tmp_path, text):
    path = tmp_path / "perf_script.txt"
    path.write_text(text, encoding="utf-8")
    return path


class TestParsePerfScriptPidShares:
    def test_two_workers_uneven_share(self, tmp_path):
        text = (
            "python 100/100 4324.001: 700000 cycles:\n"
            "\tffffaaaa func_a+0x10 (/path/to/lib.so)\n"
            "\n"
            "python 100/100 4324.002: 700000 cycles:\n"
            "\tffffaaaa func_a+0x10 (/path/to/lib.so)\n"
            "\n"
            "python 100/100 4324.003: 700000 cycles:\n"
            "\tffffaaaa func_a+0x10 (/path/to/lib.so)\n"
            "\n"
            "python 200/200 4324.004: 700000 cycles:\n"
            "\tffffbbbb func_b+0x20 (/path/to/lib.so)\n"
        )
        result = _parse_perf_script_pid_shares(_write_script(tmp_path, text))
        assert result == {100: 75.0, 200: 25.0}

    def test_single_process_no_breakdown(self, tmp_path):
        text = (
            "python 100/100 4324.001: 700000 cycles:\n"
            "\tffffaaaa func_a+0x10 (/path/to/lib.so)\n"
            "\n"
            "python 100/100 4324.002: 700000 cycles:\n"
            "\tffffaaaa func_a+0x10 (/path/to/lib.so)\n"
        )
        result = _parse_perf_script_pid_shares(_write_script(tmp_path, text))
        assert result == {}

    def test_header_without_tid_still_parses(self, tmp_path):
        """Some perf builds/events omit the "/tid" suffix on single-threaded pids."""
        text = (
            "sched-messaging 1414 K 28690.636582:  4590 cycles:\n"
            "\tffffaaaa func_a+0x10 (/path/to/lib.so)\n"
            "\n"
            "sched-messaging 1415 K 28690.636583:  4590 cycles:\n"
            "\tffffbbbb func_b+0x20 (/path/to/lib.so)\n"
        )
        result = _parse_perf_script_pid_shares(_write_script(tmp_path, text))
        assert result == {1414: 50.0, 1415: 50.0}

    def test_empty_file(self, tmp_path):
        result = _parse_perf_script_pid_shares(_write_script(tmp_path, ""))
        assert result == {}

    def test_missing_file(self, tmp_path):
        result = _parse_perf_script_pid_shares(tmp_path / "does_not_exist.txt")
        assert result == {}
