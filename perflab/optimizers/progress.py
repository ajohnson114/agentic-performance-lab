from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentProgress(Protocol):
    def on_message(self, message: str) -> None: ...


class PrintProgress:
    def on_message(self, message: str) -> None:
        print(message)


class ListProgress:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def on_message(self, message: str) -> None:
        self.messages.append(message)
