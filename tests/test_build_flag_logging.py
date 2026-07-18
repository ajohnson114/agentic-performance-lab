"""The generate phase logs build-flag recommendations from the structured data
returned by build_iteration_prompt — not by scraping the rendered prompt text."""
from __future__ import annotations

import json
from types import SimpleNamespace

from perflab.llm.base import Message
from perflab.optimizers.event_log import AgentEventLog
from perflab.optimizers.phases import generate


def _make_ctx(tmp_path, build_cmd: str | None):
    provider = SimpleNamespace(
        complete=lambda msgs, temperature, max_tokens: SimpleNamespace(
            content="no candidates here",
            usage={"input_tokens": 5, "output_tokens": 7},
        ),
    )
    build = SimpleNamespace(cmd=build_cmd) if build_cmd else None
    return SimpleNamespace(
        task=SimpleNamespace(
            build=build,
            constraints=SimpleNamespace(prompt_token_budget=0),
        ),
        config=SimpleNamespace(n_candidates=2),
        llm_config=SimpleNamespace(
            provider="test", model="test-model", temperature=0.2, max_tokens=64,
        ),
        provider=provider,
        progress=SimpleNamespace(on_message=lambda m: None),
        event_log=AgentEventLog(run_dir=tmp_path),
        iteration=1,
        total_llm_calls=0,
        total_llm_latency=0.0,
        total_input_tokens=0,
        total_output_tokens=0,
        user_actions=[],
    )


def _read_events(tmp_path):
    path = tmp_path / "agent_events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_flags_event_logged_from_structured_data(tmp_path, monkeypatch):
    flags = [
        {"flag": "-march=native", "reason": "AVX2 available", "impact": "high", "category": "isa"},
        {"flag": "-flto", "reason": "cross-TU inlining", "impact": "medium", "category": "lto"},
    ]
    # The rendered messages deliberately never mention the flags: the event
    # must come from the structured recommendations, not from prompt wording.
    messages = [
        Message(role="system", content="system prompt"),
        Message(role="user", content="user prompt"),
    ]
    monkeypatch.setattr(generate, "build_iteration_prompt", lambda ctx: (messages, flags))

    ctx = _make_ctx(tmp_path, build_cmd="g++ -O2 main.cpp")
    generate.run(ctx)

    events = [e for e in _read_events(tmp_path) if e["event_type"] == "build_flags_state"]
    assert len(events) == 1
    ev = events[0]
    assert ev["iteration"] == 1
    assert ev["build_cmd"] == "g++ -O2 main.cpp"
    assert [f["flag"] for f in ev["flags_recommended"]] == ["-march=native", "-flto"]
    assert ev["flags_recommended"][0]["impact"] == "high"
    assert ev["flags_source"] == "static+profiler"


def test_build_flags_event_empty_when_no_recommendations(tmp_path, monkeypatch):
    messages = [
        Message(role="system", content="system prompt"),
        Message(role="user", content="user prompt"),
    ]
    monkeypatch.setattr(generate, "build_iteration_prompt", lambda ctx: (messages, None))

    ctx = _make_ctx(tmp_path, build_cmd="g++ -O2 main.cpp")
    generate.run(ctx)

    events = [e for e in _read_events(tmp_path) if e["event_type"] == "build_flags_state"]
    assert len(events) == 1
    assert events[0]["flags_recommended"] == []


def test_no_build_flags_event_without_build(tmp_path, monkeypatch):
    messages = [
        Message(role="system", content="system prompt"),
        Message(role="user", content="user prompt"),
    ]
    monkeypatch.setattr(generate, "build_iteration_prompt", lambda ctx: (messages, None))

    ctx = _make_ctx(tmp_path, build_cmd=None)
    generate.run(ctx)

    events = [e for e in _read_events(tmp_path) if e["event_type"] == "build_flags_state"]
    assert events == []
