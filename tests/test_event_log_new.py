"""Tests for new event log methods: error_feedback, roofline_detected, updated llm_request."""
from __future__ import annotations

import json
from pathlib import Path

from perflab.optimizers.event_log import AgentEventLog, replay_events


def _read_events(run_dir: Path) -> list[dict]:
    path = run_dir / "agent_events.jsonl"
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


# -- error_feedback -----------------------------------------------------------

def test_error_feedback_event(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    errors = [
        {"type": "compilation", "description": "undefined reference to foo"},
        {"type": "runtime", "description": "segfault at line 42"},
    ]
    log.error_feedback(iteration=3, errors=errors)

    events = _read_events(tmp_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "error_feedback"
    assert ev["iteration"] == 3
    assert ev["error_count"] == 2
    assert len(ev["errors"]) == 2
    assert ev["errors"][0]["type"] == "compilation"
    assert ev["errors"][0]["description"] == "undefined reference to foo"
    assert ev["errors"][1]["type"] == "runtime"
    assert "timestamp" in ev


# -- roofline_detected --------------------------------------------------------

def test_roofline_detected_event(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    log.roofline_detected(
        peak_tflops=312.0,
        peak_mem_bw_gbs=2039.0,
        source="specs",
        device="H100",
    )

    events = _read_events(tmp_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "roofline_detected"
    assert ev["iteration"] is None
    assert ev["peak_tflops"] == 312.0
    assert ev["peak_mem_bw_gbs"] == 2039.0
    assert ev["source"] == "specs"
    assert ev["device"] == "H100"


# -- llm_request with prompt_token_budget -------------------------------------

def test_llm_request_with_budget(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    log.llm_request(
        iteration=1,
        prompt_length_chars=12000,
        n_candidates_requested=3,
        model="claude-opus-4-20250514",
        provider="anthropic",
        prompt_token_budget=5000,
    )

    events = _read_events(tmp_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "llm_request"
    assert ev["prompt_token_budget"] == 5000
    assert ev["model"] == "claude-opus-4-20250514"
    assert ev["provider"] == "anthropic"


def test_llm_request_without_budget(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    log.llm_request(
        iteration=1,
        prompt_length_chars=8000,
        n_candidates_requested=2,
        model="gpt-4",
        provider="openai",
        prompt_token_budget=0,
    )

    events = _read_events(tmp_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "llm_request"
    assert "prompt_token_budget" not in ev


# -- auto_tune_sweep ----------------------------------------------------------

def test_auto_tune_sweep_event(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    log.auto_tune_sweep(
        iteration=2,
        candidates_tried=8,
        best_value=123.4,
        best_knobs={"BLOCK_SIZE": 128, "num_warps": 4},
        improvement=3.4,
    )

    events = _read_events(tmp_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "auto_tune_sweep"
    assert ev["iteration"] == 2
    assert ev["candidates_tried"] == 8
    assert ev["best_value"] == 123.4
    assert ev["best_knobs"] == {"BLOCK_SIZE": 128, "num_warps": 4}
    assert ev["improvement"] == 3.4


def test_replay_events_auto_tune_sweep(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    log.auto_tune_sweep(
        iteration=2,
        candidates_tried=8,
        best_value=123.4,
        best_knobs={"BLOCK_SIZE": 128},
        improvement=3.4,
    )

    output = replay_events(tmp_path)
    assert "Auto-tune sweep" in output
    assert "8 candidates" in output
    assert "123.4" in output
    assert "BLOCK_SIZE=128" in output


# -- replay_events with error_feedback ----------------------------------------

def test_replay_events_error_feedback(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    log.error_feedback(iteration=2, errors=[
        {"type": "build", "description": "missing header file"},
    ])

    output = replay_events(tmp_path)
    assert "Error feedback" in output
    assert "1 error" in output
    assert "build" in output
    assert "missing header file" in output


# -- replay_events with roofline_detected -------------------------------------

def test_replay_events_roofline(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    log.roofline_detected(
        peak_tflops=19.5,
        peak_mem_bw_gbs=900.0,
        source="nvidia-smi",
        device="A100",
    )

    output = replay_events(tmp_path)
    assert "Roofline" in output
    assert "19.500" in output or "19.5" in output
    assert "900.0" in output
    assert "nvidia-smi" in output
    assert "A100" in output


# -- replay_events with drift_check ------------------------------------------

def test_replay_events_drift_check(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    log.drift_check(iteration=3, clean_value=1.05, last_accepted_value=1.00, drift_pct=5.0)

    output = replay_events(tmp_path)
    assert "Drift check" in output
    assert "5.0%" in output


def test_replay_events_drift_check_warning(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    log.drift_check(iteration=3, clean_value=1.10, last_accepted_value=1.00, drift_pct=10.0)

    output = replay_events(tmp_path)
    assert "WARNING" in output
    assert "10.0%" in output


# -- cost_limit_reached --------------------------------------------------------

def test_cost_limit_reached_event(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    log.cost_limit_reached(iteration=5, estimated_cost_usd=12.34, max_cost_usd=10.0)

    events = _read_events(tmp_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "cost_limit_reached"
    assert ev["iteration"] == 5
    assert ev["estimated_cost_usd"] == 12.34
    assert ev["max_cost_usd"] == 10.0


def test_replay_events_cost_limit_reached(tmp_path: Path):
    log = AgentEventLog(run_dir=tmp_path)
    log.cost_limit_reached(iteration=5, estimated_cost_usd=12.34, max_cost_usd=10.0)

    output = replay_events(tmp_path)
    assert "COST LIMIT REACHED" in output
    assert "12.34" in output
    assert "10.00" in output
