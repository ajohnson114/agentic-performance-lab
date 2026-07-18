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


def _usage_count(usage: dict, key: str, fallback_key: str) -> int:
    """Read a token count, preferring `key` and falling back to `fallback_key`.

    Uses explicit None checks: a provider may legitimately report 0 tokens,
    which `usage.get(key) or usage.get(fallback_key, 0)` would misread as missing.
    """
    val = usage.get(key)
    if val is None:
        val = usage.get(fallback_key)
    return val if val is not None else 0


def usage_input_tokens(usage: dict) -> int:
    """Input/prompt token count from a provider usage dict."""
    return _usage_count(usage, "input_tokens", "prompt_tokens")


def usage_output_tokens(usage: dict) -> int:
    """Output/completion token count from a provider usage dict."""
    return _usage_count(usage, "output_tokens", "completion_tokens")


def fmt_usage(usage: dict) -> str:
    """Format token usage as 'in=1234, out=567, total=1801'."""
    inp = usage_input_tokens(usage)
    out = usage_output_tokens(usage)
    total = inp + out
    return f"in={inp}, out={out}, total={total}"


def fmt_elapsed(seconds: float) -> str:
    """Format seconds as '4.2s' or '2m15s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"
