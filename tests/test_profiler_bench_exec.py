"""Tests for the shared benchmark-execution helpers in perflab.profilers.base.

Every profiler launches benchmark runs through bench_argv / run_bench_under /
run_bench_with_sudo_fallback, so these tests pin down the argv composition,
env/timeout passthrough, and the sudo escalation ladder in one place.
"""
from __future__ import annotations

import perflab.profilers.base as base
from perflab.tools.shell import DEFAULT_TIMEOUT_S, CmdResult


class _RunCmdRecorder:
    """Fake run_cmd that records each invocation and replays scripted returncodes."""

    def __init__(self, returncodes=(0,)):
        self.calls: list[dict] = []
        self._returncodes = list(returncodes)

    def __call__(self, cmd, cwd=None, env=None, timeout_s=DEFAULT_TIMEOUT_S, **kwargs):
        self.calls.append({
            "cmd": list(cmd),
            "cwd": cwd,
            "env": env,
            "timeout_s": timeout_s,
        })
        rc = self._returncodes.pop(0) if self._returncodes else 0
        return CmdResult(
            cmd=list(cmd), returncode=rc, stdout="", stderr="", duration_s=0.01,
        )


class TestBenchArgv:
    def test_simple_split(self):
        assert base.bench_argv("python3 bench.py --n 5") == [
            "python3", "bench.py", "--n", "5",
        ]

    def test_quoted_arg_stays_single_token(self):
        assert base.bench_argv('python3 bench.py --flag "a b"') == [
            "python3", "bench.py", "--flag", "a b",
        ]

    def test_empty_command(self):
        assert base.bench_argv("") == []


class TestRunBenchUnder:
    def test_wrapper_prepended_to_bench_argv(self, monkeypatch, tmp_path):
        rec = _RunCmdRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)

        res = base.run_bench_under(
            ["perf", "stat", "--"], "python3 bench.py --n 5", cwd=tmp_path,
        )

        assert len(rec.calls) == 1
        assert rec.calls[0]["cmd"] == [
            "perf", "stat", "--", "python3", "bench.py", "--n", "5",
        ]
        assert rec.calls[0]["cwd"] == tmp_path
        assert res.returncode == 0

    def test_empty_wrapper_runs_bench_bare(self, monkeypatch, tmp_path):
        rec = _RunCmdRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)

        base.run_bench_under([], 'python3 bench.py --flag "a b"', cwd=tmp_path)

        assert rec.calls[0]["cmd"] == ["python3", "bench.py", "--flag", "a b"]

    def test_env_passthrough(self, monkeypatch, tmp_path):
        rec = _RunCmdRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)
        env = {"PERFLAB_TORCH_PROFILE": "1"}

        base.run_bench_under([], "python3 bench.py", cwd=tmp_path, env=env)

        assert rec.calls[0]["env"] is env

    def test_env_defaults_to_none(self, monkeypatch, tmp_path):
        rec = _RunCmdRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)

        base.run_bench_under(["perf", "record", "--"], "python3 bench.py", cwd=tmp_path)

        assert rec.calls[0]["env"] is None

    def test_timeout_passthrough(self, monkeypatch, tmp_path):
        rec = _RunCmdRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)

        base.run_bench_under(
            ["perf", "sched", "record", "--"], "python3 bench.py",
            cwd=tmp_path, timeout_s=300,
        )

        assert rec.calls[0]["timeout_s"] == 300

    def test_default_timeout_matches_run_cmd_default(self, monkeypatch, tmp_path):
        rec = _RunCmdRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)

        base.run_bench_under([], "python3 bench.py", cwd=tmp_path)

        assert rec.calls[0]["timeout_s"] == DEFAULT_TIMEOUT_S

    def test_returns_run_cmd_result(self, monkeypatch, tmp_path):
        rec = _RunCmdRecorder(returncodes=[7])
        monkeypatch.setattr(base, "run_cmd", rec)

        res = base.run_bench_under([], "python3 bench.py", cwd=tmp_path)

        assert isinstance(res, CmdResult)
        assert res.returncode == 7


class TestRunBenchWithSudoFallback:
    def test_retries_with_sudo_on_failure_without_artifact(self, monkeypatch, tmp_path):
        rec = _RunCmdRecorder(returncodes=[1, 0])
        monkeypatch.setattr(base, "run_cmd", rec)
        monkeypatch.setattr(base.shutil, "which", lambda name: "/usr/bin/sudo")
        artifact = tmp_path / "out.json"  # never created

        res, used_sudo = base.run_bench_with_sudo_fallback(
            ["py-spy", "record", "--"], "python3 bench.py",
            tmp_path, expect_artifact=artifact,
        )

        assert used_sudo is True
        assert len(rec.calls) == 2
        assert rec.calls[0]["cmd"] == [
            "py-spy", "record", "--", "python3", "bench.py",
        ]
        assert rec.calls[1]["cmd"] == [
            "sudo", "-n", "py-spy", "record", "--", "python3", "bench.py",
        ]
        # Result comes from the sudo retry
        assert res.returncode == 0

    def test_no_retry_on_success(self, monkeypatch, tmp_path):
        rec = _RunCmdRecorder(returncodes=[0])
        monkeypatch.setattr(base, "run_cmd", rec)
        monkeypatch.setattr(base.shutil, "which", lambda name: "/usr/bin/sudo")

        res, used_sudo = base.run_bench_with_sudo_fallback(
            ["py-spy", "record", "--"], "python3 bench.py",
            tmp_path, expect_artifact=tmp_path / "out.json",
        )

        assert used_sudo is False
        assert len(rec.calls) == 1
        assert res.returncode == 0

    def test_no_retry_when_artifact_was_produced(self, monkeypatch, tmp_path):
        # Nonzero exit but the artifact exists: profiler output is usable,
        # so no escalation.
        rec = _RunCmdRecorder(returncodes=[1])
        monkeypatch.setattr(base, "run_cmd", rec)
        monkeypatch.setattr(base.shutil, "which", lambda name: "/usr/bin/sudo")
        artifact = tmp_path / "out.json"
        artifact.write_text("{}", encoding="utf-8")

        res, used_sudo = base.run_bench_with_sudo_fallback(
            ["py-spy", "record", "--"], "python3 bench.py",
            tmp_path, expect_artifact=artifact,
        )

        assert used_sudo is False
        assert len(rec.calls) == 1
        assert res.returncode == 1

    def test_no_retry_when_sudo_missing(self, monkeypatch, tmp_path):
        rec = _RunCmdRecorder(returncodes=[1])
        monkeypatch.setattr(base, "run_cmd", rec)
        monkeypatch.setattr(base.shutil, "which", lambda name: None)

        res, used_sudo = base.run_bench_with_sudo_fallback(
            ["py-spy", "record", "--"], "python3 bench.py",
            tmp_path, expect_artifact=tmp_path / "out.json",
        )

        assert used_sudo is False
        assert len(rec.calls) == 1
        assert res.returncode == 1

    def test_env_forwarded_to_both_attempts(self, monkeypatch, tmp_path):
        rec = _RunCmdRecorder(returncodes=[1, 0])
        monkeypatch.setattr(base, "run_cmd", rec)
        monkeypatch.setattr(base.shutil, "which", lambda name: "/usr/bin/sudo")
        env = {"XLA_FLAGS": "--xla_dump_hlo_as_text"}

        base.run_bench_with_sudo_fallback(
            ["wrapper"], "python3 bench.py",
            tmp_path, expect_artifact=tmp_path / "out.json", env=env,
        )

        assert rec.calls[0]["env"] is env
        assert rec.calls[1]["env"] is env


class TestPySpyLadder:
    """The py-spy escalation ladder must keep its exact historical sequence:
    native -> native+sudo -> non-native -> non-native+sudo."""

    def test_full_ladder_on_repeated_failure(self, monkeypatch, tmp_path):
        from perflab.profilers.python_pyspy import PySpyProfiler

        rec = _RunCmdRecorder(returncodes=[1, 1, 1, 1])
        monkeypatch.setattr(base, "run_cmd", rec)
        monkeypatch.setattr(base.shutil, "which", lambda name: "/usr/bin/sudo")

        artifacts_dir = tmp_path / "artifacts"
        result = PySpyProfiler().run("python3 bench.py", tmp_path, artifacts_dir)

        out = str((artifacts_dir / "pyspy_speedscope.json").resolve())
        native = ["py-spy", "record", "--native", "--format", "speedscope",
                  "-o", out, "--", "python3", "bench.py"]
        plain = ["py-spy", "record", "--format", "speedscope",
                 "-o", out, "--", "python3", "bench.py"]
        assert [c["cmd"] for c in rec.calls] == [
            native,
            ["sudo", "-n"] + native,
            plain,
            ["sudo", "-n"] + plain,
        ]
        assert result.summary["native_mode"] is False
        assert result.summary["returncode"] == 1

    def test_ladder_stops_after_first_success(self, monkeypatch, tmp_path):
        from perflab.profilers.python_pyspy import PySpyProfiler

        rec = _RunCmdRecorder(returncodes=[0])
        monkeypatch.setattr(base, "run_cmd", rec)
        monkeypatch.setattr(base.shutil, "which", lambda name: "/usr/bin/sudo")

        result = PySpyProfiler().run("python3 bench.py", tmp_path, tmp_path / "artifacts")

        assert len(rec.calls) == 1
        assert rec.calls[0]["cmd"][:3] == ["py-spy", "record", "--native"]
        assert result.summary["native_mode"] is True
        assert result.summary["returncode"] == 0
