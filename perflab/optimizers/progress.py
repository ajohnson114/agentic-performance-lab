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


def fmt_usage(usage: dict) -> str:
    """Format token usage as 'in=1234, out=567, total=1801'."""
    inp = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
    out = usage.get("output_tokens") or usage.get("completion_tokens", 0)
    total = inp + out
    return f"in={inp}, out={out}, total={total}"


def fmt_elapsed(seconds: float) -> str:
    """Format seconds as '4.2s' or '2m15s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"
