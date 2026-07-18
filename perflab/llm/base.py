from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol


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
