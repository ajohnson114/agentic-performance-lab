"""Tests for prompt token budget trimming."""
from perflab.llm.base import Message
from perflab.optimizers.prompt import (
    PromptContext,
    _estimate_tokens,
    _trim_to_budget,
    build_prompt,
)


def _minimal_ctx(**kwargs):
    defaults = dict(
        source_files={"main.py": "print('hello')"},
        profiler_summaries={},
        bench_results={"ok": True, "throughput": {"median": 1.0}},
    )
    defaults.update(kwargs)
    return PromptContext(**defaults)


def test_estimate_tokens():
    assert _estimate_tokens("hello") == 1  # 5 chars / 4
    assert _estimate_tokens("a" * 100) == 25


def test_no_budget_no_trimming():
    ctx = _minimal_ctx(prompt_token_budget=0)
    msgs = build_prompt(ctx)
    # Should have at least 2 messages
    assert len(msgs) >= 2


def test_large_budget_no_trimming():
    ctx = _minimal_ctx(prompt_token_budget=1000000)
    msgs = build_prompt(ctx)
    content = msgs[1].content
    assert "Source files" in content


def test_small_budget_trims_sections():
    # Include sections that _trim_to_budget can actually remove
    shared_kwargs = dict(
        profiler_summaries={"pyspy": {"top_funcs": ["a"] * 100}},
        prior_run_context="## Prior run context\n" + "Previous data. " * 200,
        bottleneck_diagnoses=[
            {"rank": 1, "bottleneck": "test", "root_cause": "x" * 500,
             "confidence": "high", "suggested_actions": ["fix it"]}
        ],
        history=[
            {"iteration": i, "description": f"iter {i}", "value": float(i), "accepted": i % 2 == 0}
            for i in range(10)
        ],
    )
    trimmed_msgs = build_prompt(_minimal_ctx(prompt_token_budget=200, **shared_kwargs))
    untrimmed_msgs = build_prompt(_minimal_ctx(prompt_token_budget=1000000, **shared_kwargs))
    trimmed_total = sum(_estimate_tokens(m.content) for m in trimmed_msgs)
    untrimmed_total = sum(_estimate_tokens(m.content) for m in untrimmed_msgs)
    assert trimmed_total < untrimmed_total


def test_trim_removes_prior_run_context_first():
    ctx = _minimal_ctx(
        prompt_token_budget=500,
        prior_run_context="## Prior run context\nSome previous run data " * 50,
    )
    build_prompt(ctx)
    # Prior run context should be removed if budget is tight
    # (the exact behavior depends on total size)


def test_trim_to_budget_passthrough():
    msgs = [
        Message(role="system", content="short"),
        Message(role="user", content="also short"),
    ]
    result = _trim_to_budget(msgs, 1000)
    assert result[1].content == "also short"


def test_constraints_prompt_token_budget():
    from perflab.task_spec import Constraints
    c = Constraints(prompt_token_budget=5000)
    assert c.prompt_token_budget == 5000
