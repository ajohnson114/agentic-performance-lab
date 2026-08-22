from __future__ import annotations

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

# Sampling parameters (temperature/top_p/top_k) were REMOVED from the Anthropic
# API starting with Claude Opus 4.7: sending one is a hard 400, not a warning.
# They remain valid on the older models below.
#
# This is an allowlist rather than a denylist on purpose -- it fails safe. A
# request that omits temperature is valid on every model past and present, so
# an unrecognized (i.e. newer) model id defaults to omitting it. A denylist
# would silently start 400-ing the whole agent loop the next time Anthropic
# ships a model.
_SAMPLING_PARAM_MODELS: tuple[str, ...] = (
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-opus-4-0",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-sonnet-4-0",
    "claude-haiku-4-5",
    "claude-3-",
)


def accepts_sampling_params(model: str) -> bool:
    """True if this model still accepts temperature/top_p/top_k.

    Note the ordering hazard this avoids: "claude-sonnet-4-6" must not be
    matched by a "claude-sonnet-5" style check, so we compare against full
    version-qualified prefixes rather than family names.
    """
    return model.startswith(_SAMPLING_PARAM_MODELS)


@dataclass
class AnthropicProvider:
    """Uses the anthropic SDK. Extracts system message into separate parameter."""

    name: str = "anthropic"
    model: str = PROVIDER_DEFAULT_MODELS["anthropic"]
    api_key: str = ""

    def is_available(self) -> bool:
        try:
            import anthropic  # noqa: F401
            return bool(self.api_key)
        except ImportError:
            return False

    def _client(self):
        import anthropic
        return anthropic.Anthropic(
            api_key=self.api_key,
            timeout=DEFAULT_TIMEOUT_S,
            max_retries=DEFAULT_MAX_RETRIES,
        )

    @staticmethod
    def _split_messages(messages: Sequence[Message]) -> tuple[str, list[dict]]:
        """Extract system message and format the rest for Anthropic API.

        Three shapes beyond a plain user/assistant turn:
          - An assistant Message with tool_calls set becomes an assistant turn
            whose content is a list of blocks: a text block first (if content
            is non-empty), then one tool_use block per ToolCall.
          - A Message with role="tool" becomes a user-role turn (Anthropic has
            no separate tool role) containing a tool_result block.
          - Consecutive tool messages are batched into a single user turn with
            multiple tool_result blocks, since Anthropic expects all results
            for one assistant turn to arrive together rather than as separate
            user turns.
        """
        system_text = ""
        api_msgs: list[dict] = []
        last_was_tool_result = False
        for m in messages:
            if m.role == "system":
                system_text = m.content
                last_was_tool_result = False
                continue
            if m.role == "tool":
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }
                if last_was_tool_result:
                    api_msgs[-1]["content"].append(tool_result_block)
                else:
                    api_msgs.append({"role": "user", "content": [tool_result_block]})
                    last_was_tool_result = True
                continue
            last_was_tool_result = False
            if m.tool_calls:
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    )
                api_msgs.append({"role": m.role, "content": blocks})
            else:
                api_msgs.append({"role": m.role, "content": m.content})
        return system_text, api_msgs

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
        system_text, api_msgs = self._split_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": api_msgs,
            "max_tokens": max_tokens,
        }
        # Omitted entirely on Opus 4.7+ / Opus 5 / Sonnet 5 / Fable 5, where any
        # sampling parameter is a 400. The caller still passes one (it comes
        # from llm.temperature, which has a dataclass default and so cannot be
        # switched off from a config file) -- dropping it has to happen here.
        if accepts_sampling_params(self.model):
            kwargs["temperature"] = temperature
        if system_text:
            kwargs["system"] = system_text
        if stop:
            kwargs["stop_sequences"] = list(stop)
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]

        resp = client.messages.create(**kwargs)
        content = ""
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            # tool_use blocks carry no .text -- keyed off .type rather than
            # hasattr(block, "text") so a turn with both reasoning text and
            # tool calls collects each into the right bucket.
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
            elif hasattr(block, "text"):
                content += block.text

        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            }
        return CompletionResult(
            content=content,
            finish_reason=resp.stop_reason,
            usage=usage,
            raw=resp,
            # None (not []) when the model didn't call a tool, matching the
            # "populated when the model wants to invoke tools" contract in
            # CompletionResult -- callers can `if result.tool_calls:` directly.
            tool_calls=tool_calls or None,
        )

    def supports_tools(self) -> bool:
        return True

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
        system_text, api_msgs = self._split_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": api_msgs,
            "max_tokens": max_tokens,
        }
        # Omitted entirely on Opus 4.7+ / Opus 5 / Sonnet 5 / Fable 5, where any
        # sampling parameter is a 400. The caller still passes one (it comes
        # from llm.temperature, which has a dataclass default and so cannot be
        # switched off from a config file) -- dropping it has to happen here.
        if accepts_sampling_params(self.model):
            kwargs["temperature"] = temperature
        if system_text:
            kwargs["system"] = system_text
        if stop:
            kwargs["stop_sequences"] = list(stop)

        with client.messages.stream(**kwargs) as stream:
            yield from stream.text_stream
