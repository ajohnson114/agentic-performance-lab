"""Tests for error feedback to LLM prompt."""
from perflab.optimizers.prompt import PromptContext, build_prompt


def _minimal_ctx(**kwargs):
    defaults = dict(
        source_files={"main.py": "print('hello')"},
        profiler_summaries={},
        bench_results={"ok": True, "throughput": {"median": 1.0}},
    )
    defaults.update(kwargs)
    return PromptContext(**defaults)


def test_no_errors_no_section():
    ctx = _minimal_ctx(last_errors=None)
    msgs = build_prompt(ctx)
    content = msgs[1].content
    assert "Errors from previous iteration" not in content


def test_errors_rendered_in_prompt():
    errors = [
        {"type": "correctness", "description": "candidate 1 failed", "output": "AssertionError: wrong result"},
        {"type": "contract_violation", "description": "candidate 2: min_repeats violated", "output": ""},
    ]
    ctx = _minimal_ctx(last_errors=errors)
    msgs = build_prompt(ctx)
    content = msgs[1].content
    assert "Errors from previous iteration" in content
    assert "correctness error" in content
    assert "candidate 1 failed" in content
    assert "AssertionError" in content
    assert "contract_violation error" in content


def test_error_output_truncated():
    long_output = "x" * 5000
    errors = [{"type": "build", "description": "build failed", "output": long_output}]
    ctx = _minimal_ctx(last_errors=errors)
    msgs = build_prompt(ctx)
    content = msgs[1].content
    assert "truncated" in content
    # Should not contain the full 5000 chars
    assert "x" * 3000 not in content


def test_empty_errors_list_no_section():
    ctx = _minimal_ctx(last_errors=[])
    msgs = build_prompt(ctx)
    content = msgs[1].content
    assert "Errors from previous iteration" not in content


def test_errors_appear_before_history():
    errors = [{"type": "correctness", "description": "test", "output": "err"}]
    history = [{"iteration": 1, "description": "baseline", "value": 1.0, "accepted": True}]
    ctx = _minimal_ctx(last_errors=errors, history=history)
    msgs = build_prompt(ctx)
    content = msgs[1].content
    err_idx = content.find("Errors from previous iteration")
    hist_idx = content.find("Optimization history")
    assert err_idx < hist_idx
