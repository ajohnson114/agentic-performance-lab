"""LLM provider that delegates to an MCP client's own LLM via sampling."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from perflab.llm.base import CompletionResult, Message


@dataclass
class MCPSamplingProvider:
    """Bridges the MCP sampling protocol to the LLMProvider interface.

    The ``_sample_fn`` is an async callable created in ``mcp_server.py`` that
    calls ``ctx.sample()`` on the MCP context.  The ``_loop`` is the event
    loop where the MCP server runs, so we can schedule coroutines from the
    synchronous agent thread.
    """

    name: str = "mcp-sampling"
    _sample_fn: Callable[..., Any] | None = field(default=None, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)

    def is_available(self) -> bool:
        return self._sample_fn is not None and self._loop is not None

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
    ) -> CompletionResult:
        if self._sample_fn is None or self._loop is None:
            raise RuntimeError(
                "MCPSamplingProvider.complete() called before the MCP sample "
                "function/event loop were set -- check is_available() first."
            )

        # Extract system message (first message with role="system")
        system_prompt: str | None = None
        user_messages: list[Message] = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                user_messages.append(m)

        # Call the async sample function from a sync context
        future = asyncio.run_coroutine_threadsafe(
            self._sample_fn(
                messages=user_messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            self._loop,
        )
        text = future.result(timeout=300)

        return CompletionResult(
            content=text,
            finish_reason="stop",
            usage={},
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
        raise NotImplementedError("MCPSamplingProvider does not support streaming")
