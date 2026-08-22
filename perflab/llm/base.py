from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

# Shared HTTP hardening for all providers: the timeout bounds a hung request so
# the agent loop can't wedge, and retries let transient 429/5xx/network blips be
# absorbed by the SDK instead of aborting an optimizer iteration.
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_MAX_RETRIES = 3


@dataclass
class ToolSpec:
    """Provider-agnostic tool definition, translated into each provider's own
    tool-schema format by that provider's complete() implementation."""
    name: str
    description: str
    parameters: dict  # JSON Schema for the tool's input object


@dataclass
class ToolCall:
    """The model's request to invoke one tool, extracted from a CompletionResult."""
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    # Set on an assistant message that requested tool calls (mirrors the
    # turn as the provider actually produced it, needed when replaying the
    # conversation back on the next request).
    tool_calls: list[ToolCall] | None = None
    # Set on a "tool" role message: which ToolCall.id this result answers.
    tool_call_id: str | None = None


@dataclass
class CompletionResult:
    content: str
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: object = None
    # Populated instead of (or alongside) content when the model wants to
    # invoke tools rather than finalize its answer. A caller building a tool
    # loop checks this before treating `content` as the final answer.
    tool_calls: list[ToolCall] | None = None


class LLMProvider(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> CompletionResult: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
    ) -> Iterator[str]: ...

    def supports_tools(self) -> bool:
        """False means a tool loop must fall back to a single-shot call
        (see perflab.llm.ollama_provider / mcp_sampling_provider) -- not
        every provider/model combination can do function calling."""
        ...
