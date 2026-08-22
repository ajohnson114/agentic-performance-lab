"""Tests for the OpenAI provider's tool-calling support (Phase 1 tool-using agent).

Mirrors the mocking conventions in tests/test_llm_providers.py: the openai SDK is
faked via patch.dict("sys.modules", ...) and response objects are built with
SimpleNamespace so attribute access matches the real SDK's shape (a missing
attribute raises AttributeError, just like a real SDK object with no tool_calls
field -- getattr-with-default in the provider is what's under test there).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from perflab.llm.base import Message, ToolCall, ToolSpec
from perflab.llm.openai_provider import OpenAIProvider


def _tool_call_obj(id_: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=arguments))


class TestOpenAISupportsTools:
    def test_supports_tools_true(self):
        assert OpenAIProvider(api_key="sk-x").supports_tools() is True


class TestOpenAIToolSchemaTranslation:
    def test_complete_passes_tools_kwarg_in_openai_shape(self):
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=None,
        )
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = resp
        tool = ToolSpec(
            name="get_sass",
            description="Show the SASS for a kernel",
            parameters={
                "type": "object",
                "properties": {"kernel": {"type": "string"}},
                "required": ["kernel"],
            },
        )
        with patch.dict("sys.modules", {"openai": mock_openai}):
            OpenAIProvider(api_key="sk-x").complete([Message("user", "hi")], tools=[tool])
        create_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
        assert create_kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_sass",
                    "description": "Show the SASS for a kernel",
                    "parameters": tool.parameters,
                },
            }
        ]

    def test_complete_without_tools_omits_tools_kwarg(self):
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=None,
        )
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = resp
        with patch.dict("sys.modules", {"openai": mock_openai}):
            OpenAIProvider(api_key="sk-x").complete([Message("user", "hi")])
        create_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
        assert "tools" not in create_kwargs


class TestOpenAIToolCallParsing:
    def test_response_tool_calls_parsed_into_completion_result(self):
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            _tool_call_obj("call_1", "get_sass", json.dumps({"kernel": "matmul"})),
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        )
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = resp
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = OpenAIProvider(api_key="sk-x").complete([Message("user", "hi")])
        assert result.content == ""
        assert result.finish_reason == "tool_calls"
        assert result.tool_calls == [
            ToolCall(id="call_1", name="get_sass", arguments={"kernel": "matmul"})
        ]

    def test_response_without_tool_calls_leaves_field_none(self):
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi there"), finish_reason="stop")],
            usage=None,
        )
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = resp
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = OpenAIProvider(api_key="sk-x").complete([Message("user", "hi")])
        assert result.tool_calls is None

    def test_malformed_tool_call_arguments_raise_loudly(self):
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[_tool_call_obj("call_1", "get_sass", "{not valid json")],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        )
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = resp
        with patch.dict("sys.modules", {"openai": mock_openai}):
            with pytest.raises(ValueError, match="malformed tool-call arguments"):
                OpenAIProvider(api_key="sk-x").complete([Message("user", "hi")])


class TestOpenAIToolResultRoundTrip:
    def test_tool_result_messages_produce_openai_shaped_payload(self):
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done"), finish_reason="stop")],
            usage=None,
        )
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = resp

        assistant_msg = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_1", name="get_sass", arguments={"kernel": "matmul"})],
        )
        tool_result_msg = Message(role="tool", content="SASS dump here", tool_call_id="call_1")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            OpenAIProvider(api_key="sk-x").complete(
                [Message("user", "show me sass"), assistant_msg, tool_result_msg]
            )

        create_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
        assert create_kwargs["messages"] == [
            {"role": "user", "content": "show me sass"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_sass",
                            "arguments": json.dumps({"kernel": "matmul"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "SASS dump here"},
        ]

    def test_multiple_tool_results_each_stay_own_message(self):
        # Unlike Anthropic, OpenAI keeps a real "tool" role and does not need
        # consecutive tool results batched into a single turn.
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done"), finish_reason="stop")],
            usage=None,
        )
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = resp

        assistant_msg = Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="call_1", name="get_sass", arguments={"kernel": "matmul"}),
                ToolCall(id="call_2", name="run_ncu", arguments={"kernel": "matmul"}),
            ],
        )
        tool_result_1 = Message(role="tool", content="sass output", tool_call_id="call_1")
        tool_result_2 = Message(role="tool", content="ncu output", tool_call_id="call_2")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            OpenAIProvider(api_key="sk-x").complete(
                [Message("user", "go"), assistant_msg, tool_result_1, tool_result_2]
            )

        create_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
        messages = create_kwargs["messages"]
        assert messages[-2] == {"role": "tool", "tool_call_id": "call_1", "content": "sass output"}
        assert messages[-1] == {"role": "tool", "tool_call_id": "call_2", "content": "ncu output"}


class TestOpenAINoToolsRegression:
    def test_plain_message_shape_unchanged(self):
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi there"), finish_reason="stop")],
            usage=None,
        )
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = resp
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = OpenAIProvider(api_key="sk-x").complete(
                [Message("system", "be terse"), Message("user", "hi")]
            )
        create_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
        assert create_kwargs["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
        assert result.content == "hi there"
        assert result.tool_calls is None
