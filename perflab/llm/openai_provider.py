from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from perflab.llm.base import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_S,
    CompletionResult,
    Message,
    ToolCall,
    ToolSpec,
)
from perflab.llm.config import PROVIDER_DEFAULT_MODELS


def _to_openai_message(m: Message) -> dict:
    """Translate a provider-agnostic Message into OpenAI's Chat Completions
    message shape. Branches only for the tool-calling fields; a plain message
    (no tool_calls, not role="tool") keeps the original {"role", "content"}
    shape exactly."""
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ],
        }
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    return {"role": m.role, "content": m.content}


def _to_openai_tool(ts: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {"name": ts.name, "description": ts.description, "parameters": ts.parameters},
    }


def _parse_tool_calls(message: object) -> list[ToolCall] | None:
    """Extract ToolCall objects from a Chat Completions response message.

    choice.message.tool_calls is absent entirely on plain-text responses (the
    SDK object has no attribute), so this must tolerate that via getattr
    rather than assuming the field always exists.
    """
    raw_tool_calls = getattr(message, "tool_calls", None)
    if not raw_tool_calls:
        return None
    tool_calls = []
    for tc in raw_tool_calls:
        try:
            arguments = json.loads(tc.function.arguments)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"OpenAI returned malformed tool-call arguments for "
                f"{tc.function.name!r} (id={tc.id!r}): not valid JSON: {tc.function.arguments!r}"
            ) from e
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
    return tool_calls


@dataclass
class OpenAIProvider:
    """Uses the openai SDK. Supports native OpenAI and OpenAI-compatible endpoints."""

    name: str = "openai"
    model: str = PROVIDER_DEFAULT_MODELS["openai"]
    api_key: str = ""
    api_base: str | None = None

    def is_available(self) -> bool:
        try: #Checks to see if optional dependency is installed and if API key is set. If not, this provider will be skipped.
            import openai  # noqa: F401
            return bool(self.api_key) #noqa: F401 suppresses the Flake8 "imported but unused" warning
        except ImportError:
            return False

    def _client(self):
        import openai
        kwargs: dict = {
            "api_key": self.api_key,
            "timeout": DEFAULT_TIMEOUT_S,
            "max_retries": DEFAULT_MAX_RETRIES,
        }
        if self.api_base:
            kwargs["base_url"] = self.api_base
        return openai.OpenAI(**kwargs)

    def supports_tools(self) -> bool:
        return True

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> CompletionResult:
        client = self._client()
        kwargs: dict = {
            "model": self.model,
            "messages": [_to_openai_message(m) for m in messages],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if stop:
            kwargs["stop"] = list(stop)
        if tools:
            kwargs["tools"] = [_to_openai_tool(ts) for ts in tools]

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return CompletionResult(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
            raw=resp,
            tool_calls=_parse_tool_calls(choice.message),
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
    ) -> Iterator[str]:
        client = self._client()
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if stop:
            kwargs["stop"] = list(stop)

        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
