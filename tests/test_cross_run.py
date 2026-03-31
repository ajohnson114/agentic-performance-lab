"""Tests for perflab.optimizers.cross_run."""
from __future__ import annotations

import json
from pathlib import Path

from perflab.optimizers.cross_run import (
    load_prior_event_insights,
    load_prior_run_context,
)


def _create_run(runs_dir: Path, run_id: str, report_data: dict, events: list[dict] | None = None) -> Path:
    """Helper to create a mock run directory."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(json.dumps(report_data), encoding="utf-8")
    if events:
        lines = [json.dumps(e) for e in events]
        (run_dir / "agent_events.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


class TestLoadPriorRunContext:
    def test_no_prior_runs(self, tmp_path: Path):
        result = load_prior_run_context(tmp_path)
        assert result is None

    def test_empty_runs_dir(self, tmp_path: Path):
        (tmp_path / "runs").mkdir()
        result = load_prior_run_context(tmp_path)
        assert result is None

    def test_loads_single_prior_run(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        _create_run(runs_dir, "run_001", {
            "run_id": "run_001",
            "task_name": "matmul",
            "baseline_value": 1.0,
            "best_value": 2.0,
            "best_iter": 3,
            "metric_name": "gflops",
            "rows": [
                {"iter": 0, "value": 1.0, "accepted": True, "notes": "baseline"},
                {"iter": 1, "value": 2.0, "accepted": True, "notes": "Used tiling"},
            ],
        })
        result = load_prior_run_context(tmp_path, current_run_id="run_002")
        assert result is not None
        assert "run_001" in result
        assert "tiling" in result.lower()

    def test_skips_current_run(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        _create_run(runs_dir, "run_001", {"run_id": "run_001", "rows": []})
        _create_run(runs_dir, "run_002", {"run_id": "run_002", "rows": []})
        result = load_prior_run_context(tmp_path, current_run_id="run_001")
        assert result is not None
        assert "run_001" not in result
        assert "run_002" in result

    def test_limits_to_last_3_runs(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        for i in range(5):
            _create_run(runs_dir, f"run_{i:03d}", {
                "run_id": f"run_{i:03d}",
                "rows": [],
            })
        result = load_prior_run_context(tmp_path)
        assert result is not None
        # Should contain last 3: run_002, run_003, run_004
        assert "run_004" in result

    def test_includes_what_worked(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        _create_run(runs_dir, "run_001", {
            "run_id": "run_001",
            "baseline_value": 1.0,
            "best_value": 3.0,
            "metric_name": "tflops",
            "rows": [
                {"iter": 0, "value": 1.0, "accepted": True, "notes": "baseline"},
                {"iter": 1, "value": 2.0, "accepted": True, "notes": "vectorization"},
                {"iter": 2, "value": 1.5, "accepted": False, "notes": "no improvement (loop unroll)"},
                {"iter": 3, "value": 3.0, "accepted": True, "notes": "cache tiling"},
            ],
        })
        result = load_prior_run_context(tmp_path)
        assert "vectorization" in result
        assert "cache tiling" in result
        assert "loop unroll" in result


class TestLoadPriorEventInsights:
    def test_no_events(self, tmp_path: Path):
        result = load_prior_event_insights(tmp_path)
        assert result == []

    def test_extracts_accepted_events(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        _create_run(runs_dir, "run_001", {"rows": []}, events=[
            {"event": "candidate_accepted", "iteration": 1, "value": 2.0, "description": "tiling patch"},
            {"event": "other_event", "data": "ignored"},
        ])
        insights = load_prior_event_insights(tmp_path)
        assert len(insights) == 1
        assert insights[0]["type"] == "success"
        assert insights[0]["description"] == "tiling patch"

    def test_extracts_early_stop(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        _create_run(runs_dir, "run_001", {"rows": []}, events=[
            {"event": "early_stop", "iteration": 5, "reason": "convergence"},
        ])
        insights = load_prior_event_insights(tmp_path)
        assert insights[0]["type"] == "early_stop"

    def test_caps_at_20_insights(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        events = [
            {"event": "candidate_accepted", "iteration": i, "value": float(i), "description": f"patch_{i}"}
            for i in range(30)
        ]
        _create_run(runs_dir, "run_001", {"rows": []}, events=events)
        insights = load_prior_event_insights(tmp_path)
        assert len(insights) <= 20
