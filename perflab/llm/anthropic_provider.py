from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from perflab.llm.base import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT_S, CompletionResult, Message
from perflab.llm.config import PROVIDER_DEFAULT_MODELS


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
        """Extract system message and format the rest for Anthropic API."""
        system_text = ""
        api_msgs: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_text = m.content
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
    ) -> CompletionResult:
        client = self._client()
        system_text, api_msgs = self._split_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": api_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text:
            kwargs["system"] = system_text
        if stop:
            kwargs["stop_sequences"] = list(stop)

        resp = client.messages.create(**kwargs)
        content = ""
        for block in resp.content:
            if hasattr(block, "text"):
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
        system_text, api_msgs = self._split_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": api_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text:
            kwargs["system"] = system_text
        if stop:
            kwargs["stop_sequences"] = list(stop)

        with client.messages.stream(**kwargs) as stream:
            yield from stream.text_stream
