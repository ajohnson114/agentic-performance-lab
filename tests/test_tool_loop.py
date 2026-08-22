"""Tests for the generate-phase tool-calling loop (perflab.optimizers.tool_loop).

Focused on loop mechanics -- turn counting, message threading, usage
summation, provider fallback, event logging -- not on what any individual
diagnostic tool does (that's tests/test_optimizer_tools.py's job). Tool
execution itself is monkeypatched here so these tests stay isolated from
the real tool registry.
"""
from __future__ import annotations

from types import SimpleNamespace

from perflab.llm.base import CompletionResult, Message, ToolCall
from perflab.optimizers import tool_loop
from perflab.optimizers.event_log import AgentEventLog
from perflab.optimizers.tool_loop import run_tool_loop


def _make_ctx(tmp_path):
    return SimpleNamespace(
        event_log=AgentEventLog(run_dir=tmp_path),
        iteration=1,
    )


def _messages():
    return [
        Message(role="system", content="system prompt"),
        Message(role="user", content="user prompt"),
    ]


class TestProviderFallback:
    def test_no_supports_tools_falls_back_to_plain_complete(self, tmp_path):
        """A provider without supports_tools() at all (older provider, or a
        bare test double) must not crash -- it degrades to a single plain
        call, same as today's pre-tool-loop behavior."""
        calls = []

        def complete(messages, *, temperature, max_tokens, tools=None):
            calls.append(tools)
            return CompletionResult(content="patch text", usage={"input_tokens": 1})

        provider = SimpleNamespace(complete=complete)  # no supports_tools attribute
        result = run_tool_loop(provider, _messages(), _make_ctx(tmp_path), temperature=0.2, max_tokens=64)

        assert result.content == "patch text"
        assert calls == [None]  # never offered tools

    def test_supports_tools_false_falls_back_to_plain_complete(self, tmp_path):
        provider = SimpleNamespace(
            supports_tools=lambda: False,
            complete=lambda messages, *, temperature, max_tokens, tools=None: CompletionResult(content="ok"),
        )
        result = run_tool_loop(provider, _messages(), _make_ctx(tmp_path), temperature=0.2, max_tokens=64)
        assert result.content == "ok"


class TestToolLoopMechanics:
    def test_no_tool_calls_returns_immediately(self, tmp_path):
        provider = SimpleNamespace(
            supports_tools=lambda: True,
            complete=lambda messages, *, temperature, max_tokens, tools=None: CompletionResult(
                content="final patch", usage={"input_tokens": 10, "output_tokens": 20},
            ),
        )
        result = run_tool_loop(provider, _messages(), _make_ctx(tmp_path), temperature=0.2, max_tokens=64)
        assert result.content == "final patch"
        assert result.usage == {"input_tokens": 10, "output_tokens": 20}

    def test_one_tool_call_then_final_answer(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tool_loop, "execute_tool", lambda tc, ctx: '{"bottleneck": "memory-bound"}')

        responses = [
            CompletionResult(
                content="", usage={"input_tokens": 5, "output_tokens": 5},
                tool_calls=[ToolCall(id="call_1", name="get_bottlenecks", arguments={})],
            ),
            CompletionResult(content="final patch", usage={"input_tokens": 8, "output_tokens": 12}),
        ]
        calls = []

        def complete(messages, *, temperature, max_tokens, tools=None):
            calls.append(list(messages))
            return responses.pop(0)

        provider = SimpleNamespace(supports_tools=lambda: True, complete=complete)
        result = run_tool_loop(provider, _messages(), _make_ctx(tmp_path), temperature=0.2, max_tokens=64)

        assert result.content == "final patch"
        # Usage summed across both turns, not just the last one.
        assert result.usage == {"input_tokens": 13, "output_tokens": 17}
        # Second call's conversation includes the assistant tool-call turn
        # and the tool-result turn appended after the first response.
        second_call_messages = calls[1]
        assert second_call_messages[-2].role == "assistant"
        assert second_call_messages[-2].tool_calls[0].name == "get_bottlenecks"
        assert second_call_messages[-1].role == "tool"
        assert second_call_messages[-1].tool_call_id == "call_1"
        assert second_call_messages[-1].content == '{"bottleneck": "memory-bound"}'

    def test_multiple_tool_calls_in_one_turn_all_executed(self, tmp_path, monkeypatch):
        executed = []

        def fake_execute(tc, ctx):
            executed.append(tc.name)
            return f'{{"tool": "{tc.name}"}}'

        monkeypatch.setattr(tool_loop, "execute_tool", fake_execute)

        responses = [
            CompletionResult(
                content="", usage={},
                tool_calls=[
                    ToolCall(id="1", name="get_bottlenecks", arguments={}),
                    ToolCall(id="2", name="run_profiler", arguments={"profiler_name": "ncu"}),
                ],
            ),
            CompletionResult(content="done", usage={}),
        ]
        provider = SimpleNamespace(
            supports_tools=lambda: True,
            complete=lambda messages, *, temperature, max_tokens, tools=None: responses.pop(0),
        )
        run_tool_loop(provider, _messages(), _make_ctx(tmp_path), temperature=0.2, max_tokens=64)
        assert executed == ["get_bottlenecks", "run_profiler"]

    def test_max_turns_exhausted_forces_final_answer_without_tools(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tool_loop, "execute_tool", lambda tc, ctx: "{}")

        call_count = {"n": 0}

        def complete(messages, *, temperature, max_tokens, tools=None):
            call_count["n"] += 1
            if tools is not None:
                # Keeps requesting tools forever if allowed to.
                return CompletionResult(
                    content="", usage={},
                    tool_calls=[ToolCall(id=str(call_count["n"]), name="get_bottlenecks", arguments={})],
                )
            # The forced final call (tools=None) must answer.
            return CompletionResult(content="forced final answer", usage={})

        provider = SimpleNamespace(supports_tools=lambda: True, complete=complete)
        result = run_tool_loop(
            provider, _messages(), _make_ctx(tmp_path),
            temperature=0.2, max_tokens=64, max_turns=3,
        )
        assert result.content == "forced final answer"
        # 3 tool-offering turns + 1 forced final call, no more.
        assert call_count["n"] == 4

    def test_tool_calls_logged_to_event_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tool_loop, "execute_tool", lambda tc, ctx: '{"ok": true}')

        responses = [
            CompletionResult(
                content="", usage={},
                tool_calls=[ToolCall(id="call_1", name="get_bottlenecks", arguments={"top_n": 3})],
            ),
            CompletionResult(content="final", usage={}),
        ]
        provider = SimpleNamespace(
            supports_tools=lambda: True,
            complete=lambda messages, *, temperature, max_tokens, tools=None: responses.pop(0),
        )
        ctx = _make_ctx(tmp_path)
        run_tool_loop(provider, _messages(), ctx, temperature=0.2, max_tokens=64)

        events_path = tmp_path / "agent_events.jsonl"
        assert events_path.exists()
        lines = events_path.read_text(encoding="utf-8").strip().splitlines()
        tool_events = [line for line in lines if '"tool_call"' in line]
        assert len(tool_events) == 1
        assert "get_bottlenecks" in tool_events[0]
