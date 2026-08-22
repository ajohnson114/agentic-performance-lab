"""Multi-turn tool-calling wrapper around LLMProvider.complete().

Phase 1 of making the generate-phase LLM call tool-using instead of
single-shot: the model can pull diagnostic tools (perflab.optimizers.tools)
on demand -- inspect a kernel dossier, diff two iterations, or re-run a
specific profiler for more detail -- before proposing its patch candidates.

This module is deliberately narrow. It wraps ONE provider.complete() call
site (phases/generate.py's candidate-generation call) with a bounded
request/tool-execute/respond loop and nothing else. It never touches the
benchmark, correctness, contract, or accept/reject path -- every tool in
DIAGNOSTIC_TOOLS is read-only, and this loop has no way to influence which
candidate gets accepted. See perflab/optimizers/tools.py's module docstring
for the anti-gaming boundary this is built to respect.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from perflab.llm.base import CompletionResult, LLMProvider, Message
from perflab.optimizers.tools import DIAGNOSTIC_TOOLS, execute_tool

if TYPE_CHECKING:
    from perflab.optimizers.agent import AgentContext
    from perflab.optimizers.event_log import AgentEventLog

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 6


def _merge_usage(total: dict[str, int], turn_usage: dict[str, int]) -> dict[str, int]:
    merged = dict(total)
    for key, val in turn_usage.items():
        merged[key] = merged.get(key, 0) + val
    return merged


def run_tool_loop(
    provider: LLMProvider,
    messages: list[Message],
    ctx: AgentContext,
    *,
    temperature: float,
    max_tokens: int,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> CompletionResult:
    """Run the generate-phase LLM call with tool access, if the provider supports it.

    Falls back to a single plain provider.complete() call (today's exact
    behavior) when the provider doesn't support tool calling -- Ollama and
    the MCP-sampling provider both report supports_tools() == False, so
    this is a no-op wrapper for them, not a degraded path.

    Returns a CompletionResult whose .content is the model's final text
    (same contract callers already had from a bare provider.complete()) and
    whose .usage is the SUM across every turn in the loop, so existing
    token/cost accounting in phases/generate.py doesn't need to change.
    """
    # getattr, not a direct call: test doubles and any provider written
    # before this Protocol method existed won't have it, and "no tool
    # support" is the correct, safe default for anything that doesn't
    # explicitly declare otherwise.
    supports_tools = getattr(provider, "supports_tools", None)
    if supports_tools is None or not supports_tools():
        return provider.complete(messages, temperature=temperature, max_tokens=max_tokens)

    event_log: AgentEventLog | None = ctx.event_log
    iteration = ctx.iteration
    conversation = list(messages)
    total_usage: dict[str, int] = {}

    for turn in range(max_turns):
        result = provider.complete(
            conversation, temperature=temperature, max_tokens=max_tokens,
            tools=DIAGNOSTIC_TOOLS,
        )
        total_usage = _merge_usage(total_usage, result.usage)

        if not result.tool_calls:
            return CompletionResult(
                content=result.content, finish_reason=result.finish_reason,
                usage=total_usage, raw=result.raw,
            )

        conversation.append(Message(
            role="assistant", content=result.content, tool_calls=result.tool_calls,
        ))
        for tc in result.tool_calls:
            tool_result = execute_tool(tc, ctx)
            conversation.append(Message(
                role="tool", content=tool_result, tool_call_id=tc.id,
            ))
            if event_log is not None:
                try:
                    event_log.tool_call(iteration, turn, tc.name, tc.arguments, tool_result)
                except Exception:  # noqa: BLE001 -- best-effort event log entry, must not block the tool loop
                    logger.warning("Failed to log tool_call event", exc_info=True)

    # Ran out of turns while the model kept requesting tools. Force one
    # final call with no tools offered so it can't loop forever -- it must
    # answer with whatever it's learned so far.
    logger.warning(
        "Tool loop hit max_turns=%d still requesting tools at iteration %d; forcing a final answer",
        max_turns, iteration,
    )
    final = provider.complete(conversation, temperature=temperature, max_tokens=max_tokens)
    total_usage = _merge_usage(total_usage, final.usage)
    return CompletionResult(
        content=final.content, finish_reason=final.finish_reason,
        usage=total_usage, raw=final.raw,
    )
