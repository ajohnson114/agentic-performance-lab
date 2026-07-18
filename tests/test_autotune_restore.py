"""Tests for the auto-tune sweep's tuning.yaml restore behavior.

A sweep writes each candidate's knobs into tuning.yaml before benchmarking
it. If the sweep aborts mid-trial, the file must be restored to its
pre-sweep contents -- not left holding whatever losing candidate's values
were written last.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from perflab.optimizers.phases import autotune
from perflab.optimizers.propose_params import load_knobs, save_knobs
from perflab.task_spec import TaskSpec

ORIGINAL_KNOBS = {"block_size": 32, "sweep": {"block_size": [64, 128]}}


class _NoOpEventLog:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _make_ctx(task):
    return SimpleNamespace(
        task=task,
        progress=SimpleNamespace(on_message=lambda msg: None),
        event_log=_NoOpEventLog(),
        iteration=1,
        best_value=1.0,
        baseline_val=1.0,
        history=[],
        config=SimpleNamespace(isolation=None),
    )


@pytest.fixture
def sweep_task(sample_task_yaml):
    task = TaskSpec.load(sample_task_yaml)
    save_knobs(task.workspace / "tuning.yaml", dict(ORIGINAL_KNOBS))
    return task


def _pass_correctness(cmd, **kwargs):
    return SimpleNamespace(returncode=0, stdout="", stderr="", rlimits_applied=None)


def test_sweep_error_restores_pre_sweep_knobs(sweep_task, monkeypatch):
    """A benchmark crash mid-sweep must restore the ORIGINAL knobs (including
    the sweep section, so an aborted sweep is retryable), not persist the
    last candidate's values."""
    monkeypatch.setattr("perflab.runners.correctness.run_correctness", _pass_correctness)

    def exploding_benchmark(cmd, **kwargs):
        raise RuntimeError("benchmark crashed mid-sweep")

    monkeypatch.setattr("perflab.runners.benchmark.run_benchmark", exploding_benchmark)

    result = autotune._auto_tune_sweep(_make_ctx(sweep_task))
    assert result is None
    knobs_path = sweep_task.workspace / "tuning.yaml"
    assert load_knobs(knobs_path) == ORIGINAL_KNOBS


def test_sweep_no_improvement_restores_originals(sweep_task, monkeypatch):
    """A completed sweep with no improvement restores the original knobs
    verbatim, including the sweep section -- tuning.yaml is user config, and
    later accepted patches may shift the optimum, so tuning must stay enabled."""
    monkeypatch.setattr("perflab.runners.correctness.run_correctness", _pass_correctness)

    def flat_benchmark(cmd, **kwargs):
        return (
            SimpleNamespace(returncode=0, stdout="", stderr="", rlimits_applied=None),
            {"ok": True, "throughput": {"median": 0.5}},  # worse than best_value=1.0
        )

    monkeypatch.setattr("perflab.runners.benchmark.run_benchmark", flat_benchmark)

    result = autotune._auto_tune_sweep(_make_ctx(sweep_task))
    assert result is None
    knobs = load_knobs(sweep_task.workspace / "tuning.yaml")
    assert knobs == ORIGINAL_KNOBS


def test_sweep_improvement_keeps_best_knobs(sweep_task, monkeypatch):
    monkeypatch.setattr("perflab.runners.correctness.run_correctness", _pass_correctness)

    def improving_benchmark(cmd, **kwargs):
        return (
            SimpleNamespace(returncode=0, stdout="", stderr="", rlimits_applied=None),
            {"ok": True, "throughput": {"median": 2.0}},  # better than best_value=1.0
        )

    monkeypatch.setattr("perflab.runners.benchmark.run_benchmark", improving_benchmark)

    result = autotune._auto_tune_sweep(_make_ctx(sweep_task))
    assert result == 2.0
    knobs = load_knobs(sweep_task.workspace / "tuning.yaml")
    assert knobs["block_size"] in (64, 128)
    # The sweep section survives a winning sweep so future iterations can retune
    assert knobs["sweep"] == ORIGINAL_KNOBS["sweep"]
