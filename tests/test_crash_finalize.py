"""run_agent must finalize (reports, run meta) even when the iteration loop crashes.

A crash hours into a run previously lost all reports and finalize output;
now the failure path finalizes with partial state (status="failed") before
re-raising, and Ctrl-C gets the same treatment.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from perflab.llm.config import LLMConfig
from perflab.optimizers.agent import AgentConfig, run_agent
from perflab.optimizers.phases import baseline as baseline_mod
from perflab.optimizers.phases import finalize as finalize_mod
from perflab.optimizers.phases import generate as generate_mod
from perflab.task_spec import TaskSpec


def _write_task(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    task_file = ws / "task.yaml"
    task_file.write_text(textwrap.dedent("""\
        name: crash-task
        program_type: python
        build: null
        correctness:
          cmd: "python tests.py"
        benchmark:
          cmd: "python bench.py --json out/bench.json"
          metric:
            name: throughput.median
            mode: maximize
    """), encoding="utf-8")
    return task_file


class _SilentProgress:
    def on_message(self, msg: str) -> None:
        pass


def _patch_phases(monkeypatch, generate_behavior, finalize_behavior):
    def fake_baseline(ctx):
        ctx.baseline_val = 10.0
        ctx.best_value = 10.0

    monkeypatch.setattr(baseline_mod, "run", fake_baseline)
    monkeypatch.setattr(generate_mod, "run", generate_behavior)
    monkeypatch.setattr(finalize_mod, "run", finalize_behavior)


def _run(tmp_path) -> object:
    task_file = _write_task(tmp_path)
    task = TaskSpec.load(task_file)
    return run_agent(
        task, task_file,
        AgentConfig(max_iters=2, early_stop=False),
        LLMConfig(provider="ollama", model="test"),
        progress=_SilentProgress(),
        provider=SimpleNamespace(),
    )


class TestCrashFinalize:
    def test_crash_mid_iteration_still_finalizes_as_failed(self, tmp_path, monkeypatch):
        finalize_calls: list[str] = []

        def boom(ctx):
            raise RuntimeError("boom")

        _patch_phases(
            monkeypatch, boom,
            lambda ctx, status="completed": finalize_calls.append(status),
        )
        with pytest.raises(RuntimeError, match="boom"):
            _run(tmp_path)
        assert finalize_calls == ["failed"]

    def test_keyboard_interrupt_still_finalizes(self, tmp_path, monkeypatch):
        finalize_calls: list[str] = []

        def interrupt(ctx):
            raise KeyboardInterrupt

        _patch_phases(
            monkeypatch, interrupt,
            lambda ctx, status="completed": finalize_calls.append(status),
        )
        with pytest.raises(KeyboardInterrupt):
            _run(tmp_path)
        assert finalize_calls == ["failed"]

    def test_crash_records_reason_in_early_stop_reason(self, tmp_path, monkeypatch):
        seen: list[str | None] = []

        def boom(ctx):
            raise RuntimeError("boom")

        _patch_phases(
            monkeypatch, boom,
            lambda ctx, status="completed": seen.append(ctx.early_stop_reason),
        )
        with pytest.raises(RuntimeError):
            _run(tmp_path)
        assert seen and seen[0] is not None and "boom" in seen[0]

    def test_finalize_crash_does_not_mask_original_error(self, tmp_path, monkeypatch):
        def boom(ctx):
            raise RuntimeError("original error")

        def broken_finalize(ctx, status="completed"):
            raise ValueError("finalize also broke")

        _patch_phases(monkeypatch, boom, broken_finalize)
        with pytest.raises(RuntimeError, match="original error"):
            _run(tmp_path)

    def test_clean_run_finalizes_as_completed(self, tmp_path, monkeypatch):
        finalize_calls: list[str] = []

        # Every iteration reports an LLM failure -> loop runs to completion
        _patch_phases(
            monkeypatch,
            lambda ctx: SimpleNamespace(llm_failed=True),
            lambda ctx, status="completed": finalize_calls.append(status),
        )
        result = _run(tmp_path)
        assert finalize_calls == ["completed"]
        assert result.baseline_value == 10.0
