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
class Message:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class CompletionResult:
    content: str
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: object = None


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
