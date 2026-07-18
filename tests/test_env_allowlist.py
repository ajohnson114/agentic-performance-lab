"""Tests for Fix 2a: allowlist-based environment sanitization for agent-run
(candidate-patched) subprocesses.

Covers:
  1. perflab.tools.shell.agent_subprocess_env — allowlist filtering
  2. perflab.tools.shell.run_cmd(env_mode=...) — allowlist vs blocklist wiring
  3. perflab.task_spec — Constraints.env_passthrough parsing
  4. perflab.runners.correctness.run_correctness — allowlist + passthrough
  5. perflab.runners.benchmark.run_benchmark — allowlist + passthrough
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from perflab.tools.shell import agent_subprocess_env, run_cmd

# Vars a real agent-run subprocess must never see, mirroring the spec's
# examples (cloud creds, VCS/model-hub tokens, DB creds, ssh agent socket).
SECRET_VARS = {
    "AWS_ACCESS_KEY_ID": "AKIAFAKE",
    "AWS_SECRET_ACCESS_KEY": "fake-secret",
    "GITHUB_TOKEN": "ghp_fake",
    "HF_TOKEN": "hf_fake",
    "DATABASE_URL": "postgres://fake",
    "SSH_AUTH_SOCK": "/tmp/fake.sock",
}


def _set_secrets(monkeypatch) -> None:
    for k, v in SECRET_VARS.items():
        monkeypatch.setenv(k, v)


# ---------------------------------------------------------------------------
# 1. agent_subprocess_env
# ---------------------------------------------------------------------------

class TestAgentSubprocessEnv:
    def test_secrets_stripped(self, monkeypatch):
        _set_secrets(monkeypatch)
        env = agent_subprocess_env()
        for key in SECRET_VARS:
            assert key not in env

    def test_path_and_cuda_forwarded(self, monkeypatch):
        _set_secrets(monkeypatch)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        env = agent_subprocess_env()
        assert "PATH" in env
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_perflab_task_prefix_forwarded(self, monkeypatch):
        monkeypatch.setenv("PERFLAB_TASK_FOO", "bar")
        env = agent_subprocess_env()
        assert env["PERFLAB_TASK_FOO"] == "bar"

    def test_unrelated_var_not_forwarded(self, monkeypatch):
        monkeypatch.setenv("SOME_RANDOM_UNRELATED_VAR", "x")
        env = agent_subprocess_env()
        assert "SOME_RANDOM_UNRELATED_VAR" not in env

    def test_extra_overrides_and_adds(self, monkeypatch):
        _set_secrets(monkeypatch)
        env = agent_subprocess_env(extra={"PATH": "/custom/bin", "MY_EXTRA": "1"})
        assert env["PATH"] == "/custom/bin"
        assert env["MY_EXTRA"] == "1"


# ---------------------------------------------------------------------------
# 2. run_cmd env_mode wiring (real subprocess)
# ---------------------------------------------------------------------------

class TestRunCmdEnvMode:
    def test_allowlist_mode_strips_secret_from_real_subprocess(self, monkeypatch):
        _set_secrets(monkeypatch)
        res = run_cmd(
            ["python3", "-c",
             "import os; print('present' if 'AWS_SECRET_ACCESS_KEY' in os.environ else 'absent')"],
            env_mode="allowlist", skip_preexec=True,
        )
        assert res.returncode == 0
        assert "absent" in res.stdout

    def test_blocklist_mode_default_still_inherits_unknown_vars(self, monkeypatch):
        # This is the pre-Fix-2a behavior, intentionally still available for
        # trusted tool invocations (profilers, compilers).
        _set_secrets(monkeypatch)
        res = run_cmd(
            ["python3", "-c",
             "import os; print('present' if 'AWS_SECRET_ACCESS_KEY' in os.environ else 'absent')"],
            skip_preexec=True,
        )
        assert res.returncode == 0
        assert "present" in res.stdout


# ---------------------------------------------------------------------------
# 3. Constraints.env_passthrough parsing
# ---------------------------------------------------------------------------

class TestConstraintsEnvPassthrough:
    def test_default_empty(self, sample_task_yaml: Path):
        from perflab.task_spec import TaskSpec
        task = TaskSpec.load(sample_task_yaml)
        assert task.constraints.env_passthrough == []

    def test_parses_declared_vars(self, tmp_workspace: Path):
        from perflab.task_spec import TaskSpec
        task_file = tmp_workspace / "task.yaml"
        task_file.write_text(textwrap.dedent("""\
            name: passthrough-test
            program_type: python
            correctness:
              cmd: "python tests.py"
            benchmark:
              cmd: "python bench.py --json out/bench.json"
              metric:
                name: value
            constraints:
              env_passthrough:
                - MY_VAR
                - ANOTHER_VAR
        """), encoding="utf-8")
        task = TaskSpec.load(task_file)
        assert task.constraints.env_passthrough == ["MY_VAR", "ANOTHER_VAR"]


# ---------------------------------------------------------------------------
# 4. run_correctness — allowlist + env_passthrough
# ---------------------------------------------------------------------------

class TestRunCorrectnessEnvHandling:
    def test_secrets_not_inherited_by_default(self, tmp_workspace: Path, monkeypatch):
        from perflab.runners.correctness import run_correctness
        _set_secrets(monkeypatch)
        script = tmp_workspace / "tests.py"
        script.write_text(
            "import os, sys\n"
            "sys.exit(0 if 'AWS_SECRET_ACCESS_KEY' not in os.environ else 1)\n"
        )
        res = run_correctness(f"python3 {script}", cwd=tmp_workspace)
        assert res.returncode == 0

    def test_env_passthrough_forwards_named_var(self, tmp_workspace: Path, monkeypatch):
        from perflab.runners.correctness import run_correctness
        monkeypatch.setenv("MY_CUSTOM_VAR", "hello123")
        script = tmp_workspace / "tests.py"
        script.write_text(
            "import os, sys\n"
            "sys.exit(0 if os.environ.get('MY_CUSTOM_VAR') == 'hello123' else 1)\n"
        )
        res = run_correctness(
            f"python3 {script}", cwd=tmp_workspace, env_passthrough=["MY_CUSTOM_VAR"],
        )
        assert res.returncode == 0

    def test_without_passthrough_var_not_forwarded(self, tmp_workspace: Path, monkeypatch):
        from perflab.runners.correctness import run_correctness
        monkeypatch.setenv("MY_CUSTOM_VAR", "hello123")
        script = tmp_workspace / "tests.py"
        script.write_text(
            "import os, sys\n"
            "sys.exit(0 if 'MY_CUSTOM_VAR' not in os.environ else 1)\n"
        )
        res = run_correctness(f"python3 {script}", cwd=tmp_workspace)
        assert res.returncode == 0


# ---------------------------------------------------------------------------
# 5. run_benchmark — allowlist + env_passthrough
# ---------------------------------------------------------------------------

def _write_bench_script(ws: Path) -> Path:
    script = ws / "bench.py"
    script.write_text(textwrap.dedent("""\
        import json, os
        data = {
            "ok": True,
            "aws_secret_present": "AWS_SECRET_ACCESS_KEY" in os.environ,
            "custom_var": os.environ.get("MY_CUSTOM_VAR"),
            "throughput": {"median": 1.0},
        }
        with open("out/bench.json", "w") as f:
            json.dump(data, f)
    """), encoding="utf-8")
    return script


class TestRunBenchmarkEnvHandling:
    def test_secrets_not_inherited_and_passthrough_forwards(self, tmp_workspace: Path, monkeypatch):
        from perflab.runners.benchmark import run_benchmark
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
        monkeypatch.setenv("MY_CUSTOM_VAR", "hello123")
        script = _write_bench_script(tmp_workspace)
        _, bench = run_benchmark(
            f"python3 {script}", cwd=tmp_workspace, env_passthrough=["MY_CUSTOM_VAR"],
        )
        assert bench["aws_secret_present"] is False
        assert bench["custom_var"] == "hello123"

    def test_without_passthrough_custom_var_absent(self, tmp_workspace: Path, monkeypatch):
        from perflab.runners.benchmark import run_benchmark
        monkeypatch.setenv("MY_CUSTOM_VAR", "hello123")
        script = _write_bench_script(tmp_workspace)
        _, bench = run_benchmark(f"python3 {script}", cwd=tmp_workspace)
        assert bench["custom_var"] is None
