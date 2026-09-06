"""Tests for the Anthropic provider's tool-calling support (Phase 1 of the
tool-using optimizer agent). Mocking follows tests/test_llm_providers.py's
convention: patch.dict("sys.modules", {"anthropic": mock_anthropic}) with a
MagicMock SDK and SimpleNamespace response objects.

This file is purely additive to the diagnostic/prompt-building side and does
not touch the benchmark/accept/contract layer.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from perflab.llm.anthropic_provider import AnthropicProvider
from perflab.llm.base import Message, ToolCall, ToolSpec


def _mock_anthropic(resp) -> MagicMock:
    # complete() streams internally (see anthropic_provider.py) and calls
    # stream.get_final_message() to get the same Message shape create() used
    # to return directly -- so the mock's __enter__ result is what needs
    # get_final_message configured, not messages.create.
    mock_anthropic = MagicMock()
    stream_cm = mock_anthropic.Anthropic.return_value.messages.stream
    stream_cm.return_value.__enter__.return_value.get_final_message.return_value = resp
    return mock_anthropic


class TestSupportsTools:
    def test_supports_tools_returns_true(self):
        assert AnthropicProvider(api_key="sk-x").supports_tools() is True


class TestToolsKwargShape:
    def test_complete_with_tools_passes_correctly_shaped_tools_kwarg(self):
        resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        mock_anthropic = _mock_anthropic(resp)
        tools = [
            ToolSpec(
                name="get_sass",
                description="Show SASS for a kernel",
                parameters={
                    "type": "object",
                    "properties": {"kernel": {"type": "string"}},
                    "required": ["kernel"],
                },
            )
        ]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            AnthropicProvider(api_key="sk-x").complete(
                [Message("user", "show me the sass")], tools=tools
            )
        create_kwargs = mock_anthropic.Anthropic.return_value.messages.stream.call_args.kwargs
        assert create_kwargs["tools"] == [
            {
                "name": "get_sass",
                "description": "Show SASS for a kernel",
                "input_schema": {
                    "type": "object",
                    "properties": {"kernel": {"type": "string"}},
                    "required": ["kernel"],
                },
            }
        ]

    def test_complete_without_tools_omits_tools_kwarg(self):
        resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        mock_anthropic = _mock_anthropic(resp)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            AnthropicProvider(api_key="sk-x").complete([Message("user", "hi")])
        create_kwargs = mock_anthropic.Anthropic.return_value.messages.stream.call_args.kwargs
        assert "tools" not in create_kwargs


class TestToolUseResponseParsing:
    def test_tool_use_block_parsed_into_tool_calls(self):
        resp = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Let me check that."),
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_01abc",
                    name="run_ncu",
                    input={"kernel": "matmul_kernel"},
                ),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        )
        mock_anthropic = _mock_anthropic(resp)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = AnthropicProvider(api_key="sk-x").complete(
                [Message("user", "profile this kernel")],
                tools=[ToolSpec(name="run_ncu", description="run ncu", parameters={})],
            )
        # finish_reason "tool_use" is not an error path -- it's passed through.
        assert result.finish_reason == "tool_use"
        # text content is still collected alongside the tool call.
        assert result.content == "Let me check that."
        assert result.tool_calls == [
            ToolCall(id="toolu_01abc", name="run_ncu", arguments={"kernel": "matmul_kernel"})
        ]

    def test_no_tool_use_block_leaves_tool_calls_none(self):
        resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="just an answer")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=2, output_tokens=2),
        )
        mock_anthropic = _mock_anthropic(resp)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = AnthropicProvider(api_key="sk-x").complete([Message("user", "hi")])
        assert result.tool_calls is None
        assert result.content == "just an answer"

    def test_multiple_tool_use_blocks_all_collected(self):
        resp = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use", id="toolu_1", name="tool_a", input={"x": 1}
                ),
                SimpleNamespace(
                    type="tool_use", id="toolu_2", name="tool_b", input={"y": 2}
                ),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        mock_anthropic = _mock_anthropic(resp)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = AnthropicProvider(api_key="sk-x").complete([Message("user", "hi")])
        assert result.content == ""
        assert result.tool_calls == [
            ToolCall(id="toolu_1", name="tool_a", arguments={"x": 1}),
            ToolCall(id="toolu_2", name="tool_b", arguments={"y": 2}),
        ]


class TestToolResultRoundTrip:
    def test_assistant_tool_calls_and_tool_result_produce_anthropic_shaped_messages(self):
        resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        mock_anthropic = _mock_anthropic(resp)
        messages = [
            Message("user", "show me the sass for matmul_kernel"),
            Message(
                "assistant",
                "Let me look.",
                tool_calls=[
                    ToolCall(id="toolu_01abc", name="get_sass", arguments={"kernel": "matmul_kernel"})
                ],
            ),
            Message("tool", "<sass output>", tool_call_id="toolu_01abc"),
        ]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            AnthropicProvider(api_key="sk-x").complete(messages)
        create_kwargs = mock_anthropic.Anthropic.return_value.messages.stream.call_args.kwargs
        assert create_kwargs["messages"] == [
            {"role": "user", "content": "show me the sass for matmul_kernel"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me look."},
                    {
                        "type": "tool_use",
                        "id": "toolu_01abc",
                        "name": "get_sass",
                        "input": {"kernel": "matmul_kernel"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01abc",
                        "content": "<sass output>",
                    }
                ],
            },
        ]

    def test_assistant_tool_calls_with_empty_content_omits_text_block(self):
        resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        mock_anthropic = _mock_anthropic(resp)
        messages = [
            Message("user", "go"),
            Message(
                "assistant",
                "",
                tool_calls=[ToolCall(id="toolu_1", name="run_ncu", arguments={})],
            ),
            Message("tool", "result", tool_call_id="toolu_1"),
        ]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            AnthropicProvider(api_key="sk-x").complete(messages)
        create_kwargs = mock_anthropic.Anthropic.return_value.messages.stream.call_args.kwargs
        assistant_turn = create_kwargs["messages"][1]
        assert assistant_turn["content"] == [
            {"type": "tool_use", "id": "toolu_1", "name": "run_ncu", "input": {}}
        ]

    def test_consecutive_tool_results_batched_into_one_user_turn(self):
        resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        mock_anthropic = _mock_anthropic(resp)
        messages = [
            Message("user", "run both"),
            Message(
                "assistant",
                "",
                tool_calls=[
                    ToolCall(id="toolu_1", name="tool_a", arguments={}),
                    ToolCall(id="toolu_2", name="tool_b", arguments={}),
                ],
            ),
            Message("tool", "result a", tool_call_id="toolu_1"),
            Message("tool", "result b", tool_call_id="toolu_2"),
        ]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            AnthropicProvider(api_key="sk-x").complete(messages)
        create_kwargs = mock_anthropic.Anthropic.return_value.messages.stream.call_args.kwargs
        sent_messages = create_kwargs["messages"]
        # Exactly 3 turns -- user, assistant, and ONE batched user turn with
        # both tool_result blocks -- not 4 (which would mean one turn per
        # tool result).
        assert len(sent_messages) == 3
        tool_result_turn = sent_messages[2]
        assert tool_result_turn["role"] == "user"
        assert tool_result_turn["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result a"},
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "result b"},
        ]

    def test_non_consecutive_tool_results_not_batched(self):
        # A user message between two tool results must split them into
        # separate turns rather than merging across the intervening turn.
        resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        mock_anthropic = _mock_anthropic(resp)
        messages = [
            Message("tool", "result a", tool_call_id="toolu_1"),
            Message("user", "interjection"),
            Message("tool", "result b", tool_call_id="toolu_2"),
        ]
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            AnthropicProvider(api_key="sk-x").complete(messages)
        create_kwargs = mock_anthropic.Anthropic.return_value.messages.stream.call_args.kwargs
        sent_messages = create_kwargs["messages"]
        assert len(sent_messages) == 3
        assert sent_messages[0]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result a"}
        ]
        assert sent_messages[1] == {"role": "user", "content": "interjection"}
        assert sent_messages[2]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "result b"}
        ]


class TestRegressionNoToolsPath:
    """The existing single-shot path (no tools involved anywhere) must keep
    working exactly as before."""

    def test_plain_conversation_unaffected(self):
        resp = SimpleNamespace(
            content=[SimpleNamespace(text="hello "), SimpleNamespace(text="world")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=4),
        )
        mock_anthropic = _mock_anthropic(resp)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = AnthropicProvider(api_key="sk-x").complete(
                [Message("system", "be terse"), Message("user", "hi")]
            )
        assert result.content == "hello world"
        assert result.finish_reason == "end_turn"
        assert result.tool_calls is None
        create_kwargs = mock_anthropic.Anthropic.return_value.messages.stream.call_args.kwargs
        assert create_kwargs["system"] == "be terse"
        assert create_kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert "tools" not in create_kwargs
