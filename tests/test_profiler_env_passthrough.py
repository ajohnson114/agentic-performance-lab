"""task.yaml env_passthrough vars must reach profiled benchmark runs.

run_benchmark/run_correctness already forward task.constraints.env_passthrough
(DATA_ROOT/HF_HOME/...) from the parent environment, but profiled runs funnel
through profilers.base.run_bench_under, which uses the allowlist env mode and
so drops everything not named. bench_env_passthrough() re-introduces the named
vars for the duration of the profiler loop; these tests pin that at the
run_bench_under seam and end-to-end through run_pipeline.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import perflab.profilers as profilers_pkg
import perflab.profilers.base as base
from perflab.profilers.base import (
    ProfileResult,
    bench_env_passthrough,
    run_bench_under,
)
from perflab.runners.pipeline import run_pipeline
from perflab.task_spec import TaskSpec
from perflab.tools.shell import CmdResult


class _EnvRecorder:
    """Fake run_cmd capturing the env overlay each call received."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, cmd, cwd=None, env=None, timeout_s=None, **kwargs):
        self.calls.append({
            "cmd": list(cmd),
            "env": dict(env) if env else env,
            "env_mode": kwargs.get("env_mode"),
        })
        return CmdResult(cmd=list(cmd), returncode=0, stdout="", stderr="", duration_s=0.0)


class TestRunBenchUnderPassthrough:
    def test_named_var_present_in_environ_is_forwarded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATA_ROOT", "/data/x")
        rec = _EnvRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)

        with bench_env_passthrough(["DATA_ROOT"]):
            run_bench_under([], "python3 bench.py", cwd=tmp_path)

        assert rec.calls[0]["env"] == {"LC_ALL": "C", "DATA_ROOT": "/data/x"}
        # Still the untrusted-code env mode.
        assert rec.calls[0]["env_mode"] == "allowlist"

    def test_unnamed_and_absent_vars_are_dropped(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATA_ROOT", "/data/x")
        monkeypatch.setenv("SECRET_TOKEN", "shh")  # in environ but not named
        monkeypatch.delenv("HF_HOME", raising=False)  # named but absent
        rec = _EnvRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)

        with bench_env_passthrough(["DATA_ROOT", "HF_HOME"]):
            run_bench_under([], "python3 bench.py", cwd=tmp_path)

        # Only the named-and-present var rides along.
        assert rec.calls[0]["env"] == {"LC_ALL": "C", "DATA_ROOT": "/data/x"}

    def test_explicit_env_overrides_passthrough(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATA_ROOT", "/data/from-environ")
        rec = _EnvRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)

        with bench_env_passthrough(["DATA_ROOT"]):
            run_bench_under(
                [], "python3 bench.py", cwd=tmp_path,
                env={"DATA_ROOT": "/data/explicit"},
            )

        assert rec.calls[0]["env"]["DATA_ROOT"] == "/data/explicit"

    def test_outside_context_nothing_is_forwarded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATA_ROOT", "/data/x")
        rec = _EnvRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)

        run_bench_under([], "python3 bench.py", cwd=tmp_path)

        assert rec.calls[0]["env"] == {"LC_ALL": "C"}

    def test_context_resets_after_block(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATA_ROOT", "/data/x")
        rec = _EnvRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)

        with bench_env_passthrough(["DATA_ROOT"]):
            run_bench_under([], "python3 bench.py", cwd=tmp_path)
        run_bench_under([], "python3 bench.py", cwd=tmp_path)

        assert rec.calls[0]["env"] == {"LC_ALL": "C", "DATA_ROOT": "/data/x"}
        assert rec.calls[1]["env"] == {"LC_ALL": "C"}

    def test_none_resets_to_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATA_ROOT", "/data/x")
        rec = _EnvRecorder()
        monkeypatch.setattr(base, "run_cmd", rec)

        with bench_env_passthrough(None):
            run_bench_under([], "python3 bench.py", cwd=tmp_path)

        assert rec.calls[0]["env"] == {"LC_ALL": "C"}


# --------------------------------------------------------------------------
# Pipeline wires the context around the profiler loop
# --------------------------------------------------------------------------

_BENCH_OK = textwrap.dedent("""\
    import json, os, time
    payload = {"ok": True, "throughput": {"median": 1.0},
               "nonce": os.urandom(8).hex(), "ts": time.time()}
    os.makedirs("out", exist_ok=True)
    with open("out/bench.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
""")


def _make_task(tmp_path: Path, env_passthrough: list[str]) -> TaskSpec:
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    passthrough_yaml = "[" + ", ".join(env_passthrough) + "]"
    (ws / "task.yaml").write_text(textwrap.dedent(f"""\
        name: passthrough-test
        program_type: python
        build: null
        correctness:
          cmd: "python tests.py"
          expected_exit: 0
        benchmark:
          cmd: "python bench.py"
          metric:
            name: throughput.median
            mode: maximize
          warmup: 1
          repeats: 2
        edit_policy:
          allowed_paths:
            - "*.py"
        constraints:
          rlimit_as_gb: 0
          env_passthrough: {passthrough_yaml}
        contract:
          required_bench_fields:
            - ok
    """), encoding="utf-8")
    (ws / "bench.py").write_text(_BENCH_OK, encoding="utf-8")
    (ws / "tests.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    return TaskSpec.load(ws / "task.yaml")


class _CtxCheckProfiler:
    """Mock profiler that records the bench_env_passthrough contextvar value."""

    name = "ctxcheck"

    def __init__(self) -> None:
        self.seen: tuple[str, ...] | None = None

    def is_available(self) -> bool:
        return True

    def run(self, bench_cmd, cwd, artifacts_dir) -> ProfileResult:
        self.seen = base._BENCH_ENV_PASSTHROUGH.get()
        return ProfileResult(name=self.name, artifacts={}, summary={"ok": True})


def test_pipeline_sets_passthrough_context_around_profiler_run(tmp_path, monkeypatch):
    task = _make_task(tmp_path, ["DATA_ROOT", "HF_HOME"])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifacts_dir = run_dir / "artifacts"

    prof = _CtxCheckProfiler()
    monkeypatch.setattr(profilers_pkg, "select_profilers", lambda task: [prof])

    run_pipeline(task, run_dir, artifacts_dir, do_profiles=True)

    assert prof.seen == ("DATA_ROOT", "HF_HOME")
    # And the context is torn down once the pipeline returns.
    assert base._BENCH_ENV_PASSTHROUGH.get() == ()
