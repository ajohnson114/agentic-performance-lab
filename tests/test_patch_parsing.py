"""Truncated LLM output must never become an applied patch.

A response cut off at max_tokens can end mid-edit-block; a REPLACE truncated
at the cut point would silently delete the tail of the matched region and can
still benchmark "faster". The parser drops incomplete blocks (surfacing a
warning), and the generate phase turns finish_reason=length/max_tokens into
error feedback for the next iteration.
"""
from __future__ import annotations

from types import SimpleNamespace

from perflab.llm.base import Message
from perflab.optimizers.patch import parse_patch_response
from perflab.optimizers.phases import generate

COMPLETE_BLOCK = """\
FILE: algo.py
<<<<<<< SEARCH
def f():
    return 1
=======
def f():
    return 2
>>>>>>> REPLACE
"""


class TestParsePatchResponseTruncation:
    def test_complete_block_parses(self):
        warnings: list[str] = []
        blocks = parse_patch_response(COMPLETE_BLOCK, warnings=warnings)
        assert len(blocks) == 1
        assert blocks[0].file_path == "algo.py"
        assert blocks[0].replace == "def f():\n    return 2"
        assert warnings == []

    def test_block_truncated_mid_replace_is_dropped(self):
        truncated = """\
FILE: algo.py
<<<<<<< SEARCH
def f():
    slow_path()
    return compute()
=======
def f():
"""
        warnings: list[str] = []
        blocks = parse_patch_response(truncated, warnings=warnings)
        # Accepting the partial block would delete slow_path() + return.
        assert blocks == []
        assert len(warnings) == 1
        assert "algo.py" in warnings[0]
        assert ">>>>>>> REPLACE" in warnings[0]

    def test_block_truncated_mid_search_is_dropped(self):
        truncated = """\
FILE: algo.py
<<<<<<< SEARCH
def f():
    return 1
"""
        warnings: list[str] = []
        blocks = parse_patch_response(truncated, warnings=warnings)
        assert blocks == []
        assert len(warnings) == 1
        assert "=======" in warnings[0]

    def test_earlier_complete_blocks_survive_truncated_tail(self):
        response = COMPLETE_BLOCK + """\
FILE: other.py
<<<<<<< SEARCH
x = 1
=======
x = 2
"""
        warnings: list[str] = []
        blocks = parse_patch_response(response, warnings=warnings)
        assert len(blocks) == 1
        assert blocks[0].file_path == "algo.py"
        assert len(warnings) == 1
        assert "other.py" in warnings[0]

    def test_no_warnings_list_still_drops_block(self):
        truncated = "FILE: a.py\n<<<<<<< SEARCH\nfoo\n=======\nbar"
        assert parse_patch_response(truncated) == []


def _make_generate_ctx(tmp_path, content: str, finish_reason: str | None):
    from perflab.optimizers.event_log import AgentEventLog

    provider = SimpleNamespace(
        complete=lambda msgs, temperature, max_tokens: SimpleNamespace(
            content=content,
            usage={"input_tokens": 5, "output_tokens": 7},
            finish_reason=finish_reason,
        ),
    )
    return SimpleNamespace(
        task=SimpleNamespace(
            build=None,
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


def _stub_prompt(monkeypatch):
    messages = [
        Message(role="system", content="system prompt"),
        Message(role="user", content="user prompt"),
    ]
    monkeypatch.setattr(generate, "build_iteration_prompt", lambda ctx: (messages, None))


class TestGenerateTruncationFeedback:
    def test_length_finish_reason_produces_generation_error(self, tmp_path, monkeypatch):
        _stub_prompt(monkeypatch)
        ctx = _make_generate_ctx(tmp_path, COMPLETE_BLOCK, finish_reason="length")

        result = generate.run(ctx)

        types = [e["type"] for e in result.generation_errors]
        assert "truncated_output" in types

    def test_max_tokens_stop_reason_produces_generation_error(self, tmp_path, monkeypatch):
        # Anthropic reports truncation as stop_reason="max_tokens"
        _stub_prompt(monkeypatch)
        ctx = _make_generate_ctx(tmp_path, COMPLETE_BLOCK, finish_reason="max_tokens")

        result = generate.run(ctx)

        assert any(e["type"] == "truncated_output" for e in result.generation_errors)

    def test_dropped_incomplete_block_produces_generation_error(self, tmp_path, monkeypatch):
        _stub_prompt(monkeypatch)
        truncated_response = COMPLETE_BLOCK + (
            "FILE: other.py\n<<<<<<< SEARCH\nx = 1\n=======\nx ="
        )
        ctx = _make_generate_ctx(tmp_path, truncated_response, finish_reason="stop")

        result = generate.run(ctx)

        incomplete = [e for e in result.generation_errors if e["type"] == "incomplete_block"]
        assert len(incomplete) == 1
        assert "other.py" in incomplete[0]["description"]
        # The complete candidate still made it through
        assert len(result.candidate_blocks) == 1

    def test_clean_stop_produces_no_generation_errors(self, tmp_path, monkeypatch):
        _stub_prompt(monkeypatch)
        ctx = _make_generate_ctx(tmp_path, COMPLETE_BLOCK, finish_reason="stop")

        result = generate.run(ctx)

        assert result.generation_errors == []
        assert len(result.candidate_blocks) == 1
