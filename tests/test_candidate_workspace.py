"""Candidate evaluation must never execute in the real workspace.

A candidate's benchmark/correctness processes run candidate code that can
write arbitrary files at runtime (including protected ones like tests.py).
Evaluation therefore happens in a disposable temp copy of the workspace, and
the real workspace's protected files are hash-verified against a run-start
snapshot.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

from perflab.optimizers.patch import (
    SearchReplaceBlock,
    snapshot_protected_files,
    verify_protected_files,
)
from perflab.optimizers.phases import evaluate as evaluate_mod
from perflab.task_spec import TaskSpec
from perflab.tools.shell import CmdResult


class _NoOpEventLog:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _ok(returncode: int = 0) -> CmdResult:
    return CmdResult(
        cmd=[], returncode=returncode, stdout="", stderr="", duration_s=0.01,
    )


def _bench_dict(value: float = 2.0) -> dict:
    return {
        "ok": True,
        "throughput": {"median": value},
        "meta": {"warmup": 1, "repeats": 5},
    }


def _make_ctx(task, tmp_path, **config_kwargs):
    from perflab.optimizers.agent import AgentConfig, AgentContext
    from perflab.optimizers.progress import PrintProgress

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return AgentContext(
        task=task,
        config=AgentConfig(**config_kwargs),
        llm_config=None,
        provider=None,
        progress=PrintProgress(),
        ws=task.workspace,
        rp=SimpleNamespace(run_dir=run_dir, artifacts_dir=run_dir / "artifacts"),
        event_log=_NoOpEventLog(),
    )


class TestEvaluateRunsInTempCopy:
    def _setup(self, tmp_workspace: Path, sample_task_yaml: Path):
        (tmp_workspace / "algo.py").write_text(
            "def f():\n    return 1\n", encoding="utf-8",
        )
        (tmp_workspace / "tests.py").write_text("print('tests')\n", encoding="utf-8")
        task = TaskSpec.load(sample_task_yaml)
        task.anti_gaming.determinism_rerun = False
        return task

    def test_correctness_and_benchmark_run_outside_real_ws(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        task = self._setup(tmp_workspace, sample_task_yaml)
        ctx = _make_ctx(task, tmp_path)
        captured: dict = {}

        def fake_correctness(cmd, cwd, **kwargs):
            captured["correctness_cwd"] = Path(cwd)
            # The patch must already be applied in the eval copy...
            captured["patched_content"] = (Path(cwd) / "algo.py").read_text(encoding="utf-8")
            # ...while the real workspace still has the original.
            captured["real_ws_content"] = (tmp_workspace / "algo.py").read_text(encoding="utf-8")
            return _ok()

        def fake_benchmark(cmd, cwd, **kwargs):
            captured["benchmark_cwd"] = Path(cwd)
            # Simulate a hostile candidate process rewriting tests.py at runtime
            (Path(cwd) / "tests.py").write_text("POISONED\n", encoding="utf-8")
            return _ok(), _bench_dict()

        monkeypatch.setattr(evaluate_mod, "run_correctness", fake_correctness)
        monkeypatch.setattr(evaluate_mod, "run_benchmark", fake_benchmark)

        blocks = [SearchReplaceBlock("algo.py", "return 1", "return 2")]
        cand, errors = evaluate_mod.evaluate_single_candidate(
            ctx, 0, blocks, reasoning="", use_fast=False,
        )

        assert cand.value == 2.0 and errors == []
        # Ran in a temp copy, not the real workspace
        assert captured["correctness_cwd"] != tmp_workspace
        assert captured["benchmark_cwd"] != tmp_workspace
        assert "return 2" in captured["patched_content"]
        assert "return 2" not in captured["real_ws_content"]
        # The runtime tests.py rewrite poisoned only the discarded copy
        assert (tmp_workspace / "tests.py").read_text(encoding="utf-8") == "print('tests')\n"
        assert (tmp_workspace / "algo.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"
        # The temp copy is cleaned up
        assert not captured["benchmark_cwd"].exists()

    def test_full_rebench_runs_outside_real_ws(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        task = self._setup(tmp_workspace, sample_task_yaml)
        ctx = _make_ctx(task, tmp_path)
        ctx.baseline_val = 1.0
        ctx.best_value = 1.0
        rebench_cwds: list[Path] = []

        def fake_benchmark(cmd, cwd, **kwargs):
            rebench_cwds.append(Path(cwd))
            return _ok(), _bench_dict(3.0)

        monkeypatch.setattr(evaluate_mod, "run_benchmark", fake_benchmark)
        monkeypatch.setattr(evaluate_mod, "snapshot_workspace", lambda *a, **k: None)

        blocks = [SearchReplaceBlock("algo.py", "return 1", "return 2")]
        cand = evaluate_mod.BeamCandidate(
            iteration=1, index=0, blocks=blocks, description="candidate 1", value=2.0,
        )
        accepted, _, _ = evaluate_mod.accept_best(ctx, [cand], use_fast=True)

        assert accepted is True
        assert len(rebench_cwds) == 1
        assert rebench_cwds[0] != tmp_workspace
        # Acceptance applies the patch to the real workspace
        assert "return 2" in (tmp_workspace / "algo.py").read_text(encoding="utf-8")


class TestProtectedFileGuard:
    def test_verify_detects_and_restores_tampered_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "tests.py").write_text("original tests\n", encoding="utf-8")
        (ws / "bench.py").write_text("original bench\n", encoding="utf-8")
        (ws / "task.yaml").write_text("name: t\n", encoding="utf-8")
        snap = tmp_path / "snap"

        hashes = snapshot_protected_files(ws, snap)
        assert set(hashes) == {"tests.py", "bench.py", "task.yaml"}

        (ws / "tests.py").write_text("if True: pass  # gamed\n", encoding="utf-8")
        tampered = verify_protected_files(ws, snap, hashes)

        assert tampered == ["tests.py"]
        assert (ws / "tests.py").read_text(encoding="utf-8") == "original tests\n"

    def test_verify_restores_deleted_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "tests.py").write_text("original\n", encoding="utf-8")
        snap = tmp_path / "snap"
        hashes = snapshot_protected_files(ws, snap)

        (ws / "tests.py").unlink()
        tampered = verify_protected_files(ws, snap, hashes)

        assert tampered == ["tests.py"]
        assert (ws / "tests.py").read_text(encoding="utf-8") == "original\n"

    def test_verify_clean_workspace_reports_nothing(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "tests.py").write_text("original\n", encoding="utf-8")
        snap = tmp_path / "snap"
        hashes = snapshot_protected_files(ws, snap)

        assert verify_protected_files(ws, snap, hashes) == []

    def test_snapshot_covers_nested_protected_files(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "sub").mkdir(parents=True)
        (ws / "sub" / "tests.py").write_text("nested\n", encoding="utf-8")
        snap = tmp_path / "snap"

        hashes = snapshot_protected_files(ws, snap)
        assert set(hashes) == {"sub/tests.py"}

        (ws / "sub" / "tests.py").write_text("tampered\n", encoding="utf-8")
        assert verify_protected_files(ws, snap, hashes) == ["sub/tests.py"]
        assert (ws / "sub" / "tests.py").read_text(encoding="utf-8") == "nested\n"

    def test_verify_restores_nested_file_after_parent_dir_deleted(self, tmp_path):
        # Candidate code can delete a protected file's enclosing directory.
        # verify must recreate the directory, restore the file, and not raise.
        ws = tmp_path / "ws"
        (ws / "sub").mkdir(parents=True)
        (ws / "sub" / "tests.py").write_text("nested\n", encoding="utf-8")
        snap = tmp_path / "snap"
        hashes = snapshot_protected_files(ws, snap)

        shutil.rmtree(ws / "sub")
        assert not (ws / "sub").exists()

        tampered = verify_protected_files(ws, snap, hashes)

        assert tampered == ["sub/tests.py"]
        assert (ws / "sub").is_dir()
        assert (ws / "sub" / "tests.py").read_text(encoding="utf-8") == "nested\n"

    def test_verify_missing_snapshot_reports_tampered_without_raising(
        self, tmp_path, caplog,
    ):
        # If the snapshot copy itself is gone, the restore cannot succeed, but
        # detection is authoritative: the file is still reported as tampered and
        # the failed restore degrades to a warning rather than an exception.
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "tests.py").write_text("original\n", encoding="utf-8")
        snap = tmp_path / "snap"
        hashes = snapshot_protected_files(ws, snap)

        (ws / "tests.py").write_text("gamed\n", encoding="utf-8")
        (snap / "tests.py").unlink()

        with caplog.at_level(logging.WARNING):
            tampered = verify_protected_files(ws, snap, hashes)

        assert tampered == ["tests.py"]
        assert "Failed to restore" in caplog.text
        assert "tests.py" in caplog.text


class TestOutDirNotCopied:
    """out/runs grows every iteration (bench json, profiler traces, snapshots);
    disposable candidate copies must exclude its contents or every
    prescreen/eval copy gets slower as the run progresses. The out dir itself
    stays, empty, because bench harnesses write out/bench.json into it."""

    def _fill_out_dir(self, tmp_workspace: Path) -> None:
        run_dir = tmp_workspace / "out" / "runs" / "run-1"
        run_dir.mkdir(parents=True)
        (run_dir / "trace.bin").write_bytes(b"x" * 1024)
        (tmp_workspace / "algo.py").write_text(
            "def f():\n    return 1\n", encoding="utf-8",
        )

    def test_patched_workspace_copy_excludes_out_contents(
        self, tmp_workspace, sample_task_yaml,
    ):
        self._fill_out_dir(tmp_workspace)
        task = TaskSpec.load(sample_task_yaml)
        with evaluate_mod._patched_workspace_copy(
            tmp_workspace, [], "perflab_test_", task.out_dir,
        ) as temp_ws:
            assert (temp_ws / "algo.py").exists()
            assert (temp_ws / "out").is_dir()
            assert list((temp_ws / "out").iterdir()) == []

    def test_prescreen_copy_excludes_out_contents(
        self, tmp_workspace, sample_task_yaml, monkeypatch,
    ):
        from perflab.optimizers.phases import prescreen as prescreen_mod

        self._fill_out_dir(tmp_workspace)
        task = TaskSpec.load(sample_task_yaml)
        seen: dict = {}

        def fake_correctness(cmd, cwd, **kwargs):
            seen["out_entries"] = list((Path(cwd) / "out").iterdir())
            seen["has_algo"] = (Path(cwd) / "algo.py").exists()
            return _ok()

        monkeypatch.setattr(prescreen_mod, "run_correctness", fake_correctness)
        result = prescreen_mod._prescreen_candidate(0, [], "", task, tmp_workspace)

        assert result["passed"] is True
        assert seen["has_algo"] is True
        assert seen["out_entries"] == []

    def test_out_dir_outside_workspace_ignores_nothing(self, tmp_path):
        from perflab.optimizers.patch import workspace_copy_ignore

        ws = tmp_path / "ws"
        (ws / "sub").mkdir(parents=True)
        ignore = workspace_copy_ignore(ws, tmp_path / "elsewhere")
        assert ignore(str(ws), ["sub", "algo.py"]) == set()
        assert ignore(str(ws / "sub"), ["f.txt"]) == set()

    def test_out_dir_equal_to_workspace_ignores_nothing(self, tmp_path):
        from perflab.optimizers.patch import workspace_copy_ignore

        ws = tmp_path / "ws"
        ws.mkdir()
        ignore = workspace_copy_ignore(ws, ws)
        assert ignore(str(ws), ["algo.py"]) == set()
