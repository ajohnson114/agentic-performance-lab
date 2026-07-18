from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from perflab.llm.base import CompletionResult, Message
from perflab.llm.config import PROVIDER_DEFAULT_MODELS


@dataclass
class OpenAIProvider:
    """Uses the openai SDK. Supports native OpenAI and OpenAI-compatible endpoints."""

    name: str = "openai"
    model: str = PROVIDER_DEFAULT_MODELS["openai"]
    api_key: str = ""
    api_base: str | None = None

    def is_available(self) -> bool:
        try: #Checks to see if optional dependency is installed and if API key is set. If not, this provider will be skipped.
            import openai  # noqa: F401
            return bool(self.api_key) #noqa: F401 suppresses the Flake8 "imported but unused" warning
        except ImportError:
            return False

    def _client(self):
        import openai
        kwargs = {"api_key": self.api_key}
        if self.api_base:
            kwargs["base_url"] = self.api_base
        return openai.OpenAI(**kwargs)

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
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if stop:
            kwargs["stop"] = list(stop)

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return CompletionResult(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
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
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if stop:
            kwargs["stop"] = list(stop)

        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
