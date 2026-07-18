"""Tests for Fix 2b: opt-in OS-level isolation for candidate execution.

Covers:
  1. perflab.tools.isolation -- level normalization/resolution, wrap_command
     fallback behavior (all runnable without bwrap installed)
  2. perflab.config -- IsolationSection defaults, YAML/env overlay, template
  3. perflab.runners.benchmark / perflab.runners.correctness -- isolation
     parameter wiring (default-None no-op, cwd-overrides-policy.workspace)
  4. bwrap-gated acceptance tests (skipped when bwrap isn't installed, e.g.
     on this macOS dev machine -- these are written to run correctly on a
     Linux CI runner with bwrap installed, per the Fix 2b spec)
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import socket
import subprocess
import textwrap
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from perflab.config import (
    DEFAULT_CONFIG_TEMPLATE,
    PerfLabConfig,
    _overlay_env,
    _overlay_yaml,
)
from perflab.tools import isolation as isolation_mod
from perflab.tools.isolation import (
    IsolationPolicy,
    default_level_for_host,
    normalize_level,
    resolve_level,
    resolve_policy,
    wrap_command,
)

# ---------------------------------------------------------------------------
# 1a. Level normalization / resolution (pure functions)
# ---------------------------------------------------------------------------


class TestResolvePolicy:
    """resolve_policy: the shared CLI/MCP task.yaml+config resolution."""

    def _write_task(self, tmp_path, extra: str = "") -> Path:
        p = tmp_path / "task.yaml"
        p.write_text("name: t\n" + extra, encoding="utf-8")
        return p

    def test_none_everywhere_returns_no_policy(self, tmp_path):
        assert resolve_policy(self._write_task(tmp_path), "none") is None

    def test_task_yaml_level_wins_over_config(self, tmp_path):
        task = self._write_task(tmp_path, "isolation:\n  level: restricted\n")
        policy = resolve_policy(task, "none")
        assert policy is not None
        assert policy.level == "restricted"

    def test_cli_level_wins_over_task_yaml(self, tmp_path):
        task = self._write_task(tmp_path, "isolation:\n  level: restricted\n")
        policy = resolve_policy(task, "none", cli_level="strict")
        assert policy is not None
        assert policy.level == "strict"

    def test_config_level_used_when_task_silent(self, tmp_path):
        policy = resolve_policy(self._write_task(tmp_path), "restricted")
        assert policy is not None
        assert policy.level == "restricted"

    def test_network_flag_carried_from_constraints(self, tmp_path):
        task = self._write_task(
            tmp_path, "isolation:\n  level: restricted\nconstraints:\n  network: true\n"
        )
        policy = resolve_policy(task, "none")
        assert policy is not None
        assert policy.network is True

    def test_invalid_level_raises(self, tmp_path):
        task = self._write_task(tmp_path, "isolation:\n  level: banana\n")
        with pytest.raises(ValueError):
            resolve_policy(task, "none")

    def test_missing_task_file_falls_back_to_config(self, tmp_path):
        policy = resolve_policy(tmp_path / "does-not-exist.yaml", "restricted")
        assert policy is not None
        assert policy.level == "restricted"


class TestNormalizeLevel:
    def test_none_input_is_none_level(self):
        assert normalize_level(None) == "none"
        assert normalize_level("") == "none"

    def test_valid_levels_pass_through(self):
        assert normalize_level("none") == "none"
        assert normalize_level("restricted") == "restricted"
        assert normalize_level("strict") == "strict"

    def test_case_and_whitespace_insensitive(self):
        assert normalize_level("  Restricted  ") == "restricted"
        assert normalize_level("STRICT") == "strict"

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            normalize_level("docker")


class TestResolveLevel:
    def test_cli_flag_wins(self):
        assert resolve_level("strict", "restricted", "none") == "strict"

    def test_task_yaml_wins_over_config(self):
        assert resolve_level(None, "restricted", "strict") == "restricted"

    def test_config_default_used_when_nothing_else_set(self):
        assert resolve_level(None, None, "restricted") == "restricted"

    def test_all_unset_defaults_to_none(self):
        assert resolve_level(None, None, None) == "none"


# ---------------------------------------------------------------------------
# 1b. IsolationPolicy dataclass
# ---------------------------------------------------------------------------


class TestIsolationPolicy:
    def test_defaults(self):
        policy = IsolationPolicy()
        assert policy.level == "none"
        assert policy.workspace is None
        assert policy.run_output_dir is None
        assert policy.network is False

    def test_replace_overrides_workspace_only(self, tmp_path):
        policy = IsolationPolicy(level="restricted", workspace=tmp_path / "a", network=True)
        replaced = dataclasses.replace(policy, workspace=tmp_path / "b")
        assert replaced.workspace == tmp_path / "b"
        assert replaced.level == "restricted"
        assert replaced.network is True


# ---------------------------------------------------------------------------
# 1c. wrap_command fallback behavior (no bwrap required)
# ---------------------------------------------------------------------------


class TestWrapCommandFallbacks:
    def test_level_none_returns_cmd_unchanged(self):
        cmd = ["python3", "bench.py"]
        assert wrap_command(cmd, IsolationPolicy(level="none")) == cmd

    def test_macos_falls_back_to_none(self, caplog):
        cmd = ["python3", "bench.py"]
        with patch.object(isolation_mod.platform, "system", return_value="Darwin"):
            result = wrap_command(cmd, IsolationPolicy(level="restricted"))
        assert result == cmd
        assert "macOS" in caplog.text

    def test_unknown_platform_falls_back_to_none(self, caplog):
        cmd = ["python3", "bench.py"]
        with patch.object(isolation_mod.platform, "system", return_value="Windows"):
            result = wrap_command(cmd, IsolationPolicy(level="restricted"))
        assert result == cmd
        assert "Windows" in caplog.text

    def test_linux_without_bwrap_falls_back_to_none(self, caplog):
        cmd = ["python3", "bench.py"]
        with patch.object(isolation_mod.platform, "system", return_value="Linux"), \
             patch.object(isolation_mod.shutil, "which", return_value=None):
            result = wrap_command(cmd, IsolationPolicy(level="restricted"))
        assert result == cmd
        assert "bwrap" in caplog.text

    def test_linux_with_unusable_bwrap_falls_back_to_none(self, caplog):
        cmd = ["python3", "bench.py"]
        with patch.object(isolation_mod.platform, "system", return_value="Linux"), \
             patch.object(isolation_mod.shutil, "which", return_value="/usr/bin/bwrap"), \
             patch.object(isolation_mod, "_bwrap_usable", return_value=False):
            result = wrap_command(cmd, IsolationPolicy(level="restricted"))
        assert result == cmd

    def test_linux_with_usable_bwrap_prefixes_cmd(self, tmp_path):
        cmd = ["python3", "bench.py"]
        ws = tmp_path / "ws"
        ws.mkdir()
        with patch.object(isolation_mod.platform, "system", return_value="Linux"), \
             patch.object(isolation_mod.shutil, "which", return_value="/usr/bin/bwrap"), \
             patch.object(isolation_mod, "_bwrap_usable", return_value=True), \
             patch.object(isolation_mod, "_readonly_bind_paths", return_value=[]), \
             patch.object(isolation_mod, "_nvidia_device_paths", return_value=[]):
            result = wrap_command(cmd, IsolationPolicy(level="restricted", workspace=ws))
        assert result[0] == "/usr/bin/bwrap"
        assert result[-2:] == cmd
        assert "--unshare-net" in result  # network defaults to False
        assert "--die-with-parent" in result
        assert "--bind" in result and str(ws) in result

    def test_network_true_omits_unshare_net(self, tmp_path):
        cmd = ["python3", "bench.py"]
        ws = tmp_path / "ws"
        ws.mkdir()
        with patch.object(isolation_mod.platform, "system", return_value="Linux"), \
             patch.object(isolation_mod.shutil, "which", return_value="/usr/bin/bwrap"), \
             patch.object(isolation_mod, "_bwrap_usable", return_value=True), \
             patch.object(isolation_mod, "_readonly_bind_paths", return_value=[]), \
             patch.object(isolation_mod, "_nvidia_device_paths", return_value=[]):
            result = wrap_command(cmd, IsolationPolicy(level="restricted", workspace=ws, network=True))
        assert "--unshare-net" not in result

    def test_bare_python_resolved_to_sys_executable_when_wrapped(self, tmp_path):
        import sys
        cmd = ["python", "bench.py"]
        ws = tmp_path / "ws"
        ws.mkdir()
        with patch.object(isolation_mod.platform, "system", return_value="Linux"), \
             patch.object(isolation_mod.shutil, "which", return_value="/usr/bin/bwrap"), \
             patch.object(isolation_mod, "_bwrap_usable", return_value=True), \
             patch.object(isolation_mod, "_readonly_bind_paths", return_value=[]), \
             patch.object(isolation_mod, "_nvidia_device_paths", return_value=[]):
            result = wrap_command(cmd, IsolationPolicy(level="restricted", workspace=ws))
        assert result[-2:] == [sys.executable, "bench.py"]

    def test_strict_without_seccomp_support_warns_and_still_wraps(self, tmp_path, caplog):
        cmd = ["python3", "bench.py"]
        ws = tmp_path / "ws"
        ws.mkdir()
        with patch.object(isolation_mod.platform, "system", return_value="Linux"), \
             patch.object(isolation_mod.shutil, "which", return_value="/usr/bin/bwrap"), \
             patch.object(isolation_mod, "_bwrap_usable", return_value=True), \
             patch.object(isolation_mod, "_bwrap_supports_seccomp", return_value=False), \
             patch.object(isolation_mod, "_readonly_bind_paths", return_value=[]), \
             patch.object(isolation_mod, "_nvidia_device_paths", return_value=[]):
            result = wrap_command(cmd, IsolationPolicy(level="strict", workspace=ws))
        assert result[0] == "/usr/bin/bwrap"
        assert "restricted" in caplog.text.lower()


class TestDefaultLevelForHost:
    def test_restricted_when_bwrap_usable(self):
        with patch.object(isolation_mod, "_bwrap_usable", return_value=True):
            assert default_level_for_host() == "restricted"

    def test_none_when_bwrap_unusable(self):
        with patch.object(isolation_mod, "_bwrap_usable", return_value=False):
            assert default_level_for_host() == "none"


# ---------------------------------------------------------------------------
# 2. Config wiring (IsolationSection)
# ---------------------------------------------------------------------------


class TestIsolationConfig:
    def test_default_is_none(self):
        cfg = PerfLabConfig()
        assert cfg.isolation.level == "none"

    def test_yaml_overlay(self):
        cfg = PerfLabConfig()
        _overlay_yaml(cfg, {"isolation": {"level": "restricted"}})
        assert cfg.isolation.level == "restricted"

    def test_yaml_overlay_partial_leaves_other_sections_untouched(self):
        cfg = PerfLabConfig()
        _overlay_yaml(cfg, {"isolation": {"level": "strict"}})
        assert cfg.isolation.level == "strict"
        assert cfg.benchmark.warmup == 3

    def test_env_overlay(self):
        cfg = PerfLabConfig()
        with patch.dict(os.environ, {"PERFLAB_ISOLATION_LEVEL": "restricted"}):
            _overlay_env(cfg)
        assert cfg.isolation.level == "restricted"

    def test_env_overrides_yaml(self):
        cfg = PerfLabConfig()
        _overlay_yaml(cfg, {"isolation": {"level": "strict"}})
        with patch.dict(os.environ, {"PERFLAB_ISOLATION_LEVEL": "none"}):
            _overlay_env(cfg)
        assert cfg.isolation.level == "none"

    def test_to_dict_includes_isolation(self):
        cfg = PerfLabConfig()
        d = cfg.to_dict()
        assert d["isolation"]["level"] == "none"

    def test_template_documents_isolation(self):
        data = yaml.safe_load(DEFAULT_CONFIG_TEMPLATE)
        assert data["isolation"]["level"] == "none"
        assert "restricted" in DEFAULT_CONFIG_TEMPLATE
        assert "strict" in DEFAULT_CONFIG_TEMPLATE


# ---------------------------------------------------------------------------
# 3. Runner wiring (isolation param on run_benchmark / run_correctness)
# ---------------------------------------------------------------------------


def _write_bench_script(ws: Path) -> Path:
    script = ws / "bench.py"
    script.write_text(textwrap.dedent("""\
        import json
        with open("out/bench.json", "w") as f:
            json.dump({"ok": True, "throughput": {"median": 1.0}}, f)
    """), encoding="utf-8")
    return script


class TestRunBenchmarkIsolationWiring:
    def test_default_isolation_none_is_unaffected(self, tmp_workspace: Path):
        """isolation not passed at all -- must behave exactly as before Fix 2b."""
        from perflab.runners.benchmark import run_benchmark
        script = _write_bench_script(tmp_workspace)
        res, bench = run_benchmark(f"python3 {script}", cwd=tmp_workspace)
        assert res.returncode == 0
        assert bench["ok"] is True

    def test_isolation_none_policy_is_also_unaffected(self, tmp_workspace: Path):
        from perflab.runners.benchmark import run_benchmark
        script = _write_bench_script(tmp_workspace)
        res, bench = run_benchmark(
            f"python3 {script}", cwd=tmp_workspace,
            isolation=IsolationPolicy(level="none"),
        )
        assert res.returncode == 0
        assert bench["ok"] is True

    def test_wrap_command_called_with_cwd_as_workspace(self, tmp_workspace: Path):
        """The candidate benchmarks in its own temp copy of the workspace (cwd) --
        run_benchmark must bind *that* directory, not whatever workspace path
        happened to be on the IsolationPolicy when it was constructed."""
        from perflab.runners import benchmark as benchmark_mod

        script = _write_bench_script(tmp_workspace)
        stale_workspace = tmp_workspace.parent / "stale"
        policy = IsolationPolicy(level="restricted", workspace=stale_workspace)

        captured = {}
        def fake_wrap_command(cmd, pol):
            captured["policy"] = pol
            return cmd

        with patch.object(benchmark_mod, "wrap_command", side_effect=fake_wrap_command):
            benchmark_mod.run_benchmark(f"python3 {script}", cwd=tmp_workspace, isolation=policy)

        assert captured["policy"].workspace == tmp_workspace
        assert captured["policy"].level == "restricted"


class TestRunCorrectnessIsolationWiring:
    def test_default_isolation_none_is_unaffected(self, tmp_workspace: Path):
        from perflab.runners.correctness import run_correctness
        script = tmp_workspace / "tests.py"
        script.write_text("import sys; sys.exit(0)\n")
        res = run_correctness(f"python3 {script}", cwd=tmp_workspace)
        assert res.returncode == 0

    def test_wrap_command_called_with_cwd_as_workspace(self, tmp_workspace: Path):
        from perflab.runners import correctness as correctness_mod

        script = tmp_workspace / "tests.py"
        script.write_text("import sys; sys.exit(0)\n")
        stale_workspace = tmp_workspace.parent / "stale"
        policy = IsolationPolicy(level="restricted", workspace=stale_workspace)

        captured = {}
        def fake_wrap_command(cmd, pol):
            captured["policy"] = pol
            return cmd

        with patch.object(correctness_mod, "wrap_command", side_effect=fake_wrap_command):
            res = correctness_mod.run_correctness(f"python3 {script}", cwd=tmp_workspace, isolation=policy)

        assert res.returncode == 0
        assert captured["policy"].workspace == tmp_workspace

    def test_run_correctness_twice_wiring(self, tmp_workspace: Path):
        from perflab.runners import correctness as correctness_mod

        script = tmp_workspace / "tests.py"
        script.write_text("import sys; sys.exit(0)\n")
        policy = IsolationPolicy(level="restricted", workspace=tmp_workspace.parent / "stale")

        calls = []
        def fake_wrap_command(cmd, pol):
            calls.append(pol)
            return cmd

        with patch.object(correctness_mod, "wrap_command", side_effect=fake_wrap_command):
            res, warnings = correctness_mod.run_correctness_twice(
                f"python3 {script}", cwd=tmp_workspace, isolation=policy,
            )

        assert res.returncode == 0
        assert warnings == []
        assert len(calls) == 1  # wrap_command applies once; same wrapped args reused for both runs
        assert calls[0].workspace == tmp_workspace


# ---------------------------------------------------------------------------
# 4. bwrap-gated acceptance tests
# ---------------------------------------------------------------------------

_BWRAP_MISSING = shutil.which("bwrap") is None


def _free_local_server() -> tuple[socket.socket, int]:
    """Bind a TCP server on 127.0.0.1 and accept (and drop) connections."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    port = srv.getsockname()[1]

    def _accept_loop():
        try:
            while True:
                conn, _ = srv.accept()
                conn.close()
        except OSError:
            pass

    threading.Thread(target=_accept_loop, daemon=True).start()
    return srv, port


@pytest.mark.skipif(_BWRAP_MISSING, reason="bwrap not installed")
class TestBwrapAcceptance:
    def test_restricted_blocks_write_outside_workspace(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        pwned = Path.home() / "perflab_isolation_test_pwned"
        if pwned.exists():
            pwned.unlink()
        try:
            policy = IsolationPolicy(level="restricted", workspace=ws, network=False)
            cmd = wrap_command(
                ["python3", "-c", f"open({str(pwned)!r}, 'w').write('pwned')"],
                policy,
            )
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            assert result.returncode != 0
            assert not pwned.exists()
        finally:
            if pwned.exists():
                pwned.unlink()

    def test_restricted_blocks_network_without_network_flag(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        srv, port = _free_local_server()
        try:
            policy = IsolationPolicy(level="restricted", workspace=ws, network=False)
            cmd = wrap_command(
                ["python3", "-c",
                 f"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=3)"],
                policy,
            )
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            assert result.returncode != 0
        finally:
            srv.close()

    def test_restricted_allows_network_with_network_flag(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        srv, port = _free_local_server()
        try:
            policy = IsolationPolicy(level="restricted", workspace=ws, network=True)
            cmd = wrap_command(
                ["python3", "-c",
                 f"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=3).close()"],
                policy,
            )
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            assert result.returncode == 0, result.stderr
        finally:
            srv.close()

    def test_matmul_cpp_builds_and_runs_under_restricted(self, tmp_path):
        from perflab.runners.benchmark import run_benchmark
        from perflab.runners.correctness import run_correctness

        src = Path(__file__).resolve().parents[1] / "tasks" / "matmul" / "cpp"
        ws = tmp_path / "matmul_cpp"
        shutil.copytree(src, ws)

        build = subprocess.run(
            ["g++", "-O2", "-o", "matmul_bin", "matmul.cpp"],
            cwd=ws, capture_output=True, text=True, timeout=120,
        )
        assert build.returncode == 0, build.stderr

        policy = IsolationPolicy(level="restricted", workspace=ws, network=False)

        correctness_res = run_correctness("python tests.py", cwd=ws, isolation=policy)
        assert correctness_res.returncode == 0, correctness_res.stderr

        bench_res, bench = run_benchmark(
            "python bench.py --json out/bench.json", cwd=ws, isolation=policy,
        )
        assert bench_res.returncode == 0, bench_res.stderr
        assert bench["ok"] is True


# ---------------------------------------------------------------------------
# 5. Agent-loop wiring (isolation policy threaded from AgentConfig into the
#    candidate benchmark/correctness subprocess calls)
# ---------------------------------------------------------------------------


def _fake_cmd_result(returncode: int = 0):
    from types import SimpleNamespace
    return SimpleNamespace(returncode=returncode, stdout="", stderr="", rlimits_applied=None)


class _NoOpEventLog:
    """Duck-typed AgentEventLog stand-in: swallows every event method."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _make_agent_ctx(task, tmp_path, isolation_policy):
    from types import SimpleNamespace

    from perflab.optimizers.agent import AgentConfig, AgentContext
    from perflab.optimizers.progress import PrintProgress

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return AgentContext(
        task=task,
        config=AgentConfig(isolation=isolation_policy),
        llm_config=None,
        provider=None,
        progress=PrintProgress(),
        ws=task.workspace,
        rp=SimpleNamespace(run_dir=run_dir, artifacts_dir=run_dir / "artifacts"),
        event_log=_NoOpEventLog(),
    )


class TestAgentLoopIsolationWiring:
    def test_evaluate_passes_config_isolation_to_run_correctness(
        self, tmp_workspace: Path, sample_task_yaml: Path, tmp_path, monkeypatch,
    ):
        from perflab.optimizers.phases import evaluate as evaluate_mod
        from perflab.task_spec import TaskSpec

        task = TaskSpec.load(sample_task_yaml)
        policy = IsolationPolicy(level="restricted")
        ctx = _make_agent_ctx(task, tmp_path, policy)

        captured = {}

        def fake_correctness(cmd, **kwargs):
            captured["isolation"] = kwargs.get("isolation")
            return _fake_cmd_result(returncode=1)  # fail -> early return, no benchmark

        monkeypatch.setattr(evaluate_mod, "run_correctness", fake_correctness)
        cand, errors = evaluate_mod.evaluate_single_candidate(
            ctx, 0, blocks=[], reasoning="", use_fast=False,
        )
        assert captured["isolation"] is policy
        assert "correctness failed" in cand.description

    def test_evaluate_passes_config_isolation_to_run_benchmark(
        self, tmp_workspace: Path, sample_task_yaml: Path, tmp_path, monkeypatch,
    ):
        from perflab.optimizers.phases import evaluate as evaluate_mod
        from perflab.task_spec import TaskSpec

        task = TaskSpec.load(sample_task_yaml)
        policy = IsolationPolicy(level="restricted")
        ctx = _make_agent_ctx(task, tmp_path, policy)

        captured = {}
        monkeypatch.setattr(
            evaluate_mod, "run_correctness", lambda cmd, **kw: _fake_cmd_result(0),
        )

        def fake_benchmark(cmd, **kwargs):
            captured["isolation"] = kwargs.get("isolation")
            raise RuntimeError("stop here")

        monkeypatch.setattr(evaluate_mod, "run_benchmark", fake_benchmark)
        evaluate_mod.evaluate_single_candidate(ctx, 0, blocks=[], reasoning="", use_fast=False)
        assert captured["isolation"] is policy

    def test_prescreen_passes_config_isolation_to_run_correctness(
        self, tmp_workspace: Path, sample_task_yaml: Path, tmp_path, monkeypatch,
    ):
        from perflab.optimizers.phases import prescreen as prescreen_mod
        from perflab.task_spec import TaskSpec

        task = TaskSpec.load(sample_task_yaml)
        policy = IsolationPolicy(level="restricted")
        ctx = _make_agent_ctx(task, tmp_path, policy)

        captured = {}

        def fake_correctness(cmd, **kwargs):
            captured["isolation"] = kwargs.get("isolation")
            return _fake_cmd_result(0)

        monkeypatch.setattr(prescreen_mod, "run_correctness", fake_correctness)
        results = prescreen_mod.run(ctx, [[]], [""])
        assert results[0]["passed"] is True
        assert captured["isolation"] is policy

    def test_pipeline_threads_isolation_to_both_runners(
        self, tmp_workspace: Path, sample_task_yaml: Path, tmp_path, monkeypatch,
    ):
        import json as _json

        from perflab.runners import pipeline as pipeline_mod
        from perflab.task_spec import TaskSpec

        task = TaskSpec.load(sample_task_yaml)
        policy = IsolationPolicy(level="restricted")
        captured = {}

        def fake_correctness(cmd, **kwargs):
            captured["correctness"] = kwargs.get("isolation")
            return _fake_cmd_result(0)

        def fake_benchmark(cmd, cwd, **kwargs):
            captured["benchmark"] = kwargs.get("isolation")
            bench = {"ok": True, "throughput": {"median": 1.0}}
            (cwd / "out").mkdir(exist_ok=True)
            (cwd / "out" / "bench.json").write_text(_json.dumps(bench), encoding="utf-8")
            return _fake_cmd_result(0), bench

        monkeypatch.setattr(pipeline_mod, "run_correctness", fake_correctness)
        monkeypatch.setattr(pipeline_mod, "run_benchmark", fake_benchmark)

        run_dir = tmp_path / "pipeline_run"
        run_dir.mkdir()
        pipeline_mod.run_pipeline(
            task, run_dir, run_dir / "artifacts", isolation=policy,
        )
        assert captured["correctness"] is policy
        assert captured["benchmark"] is policy
