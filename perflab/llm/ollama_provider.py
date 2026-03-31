from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Iterator, Sequence
from urllib.parse import urlparse

from perflab.llm.base import CompletionResult, Message

# Allowed hosts for Ollama API to prevent SSRF.
_OLLAMA_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _validate_ollama_url(api_base: str) -> None:
    """Validate that api_base points to a local Ollama instance."""
    parsed = urlparse(api_base)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Ollama api_base must use http or https scheme, got: {parsed.scheme!r}")
    host = parsed.hostname or ""
    if host not in _OLLAMA_ALLOWED_HOSTS:
        raise ValueError(
            f"Ollama api_base host must be one of {_OLLAMA_ALLOWED_HOSTS}, "
            f"got: {host!r}. Set PERFLAB_OLLAMA_ALLOW_REMOTE=1 to override."
        )
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    allowed_ports = {11434}
    extra = os.environ.get("PERFLAB_OLLAMA_ALLOWED_PORTS", "")
    if extra:
        for p in extra.split(","):
            p = p.strip()
            if p.isdigit():
                allowed_ports.add(int(p))
    if port not in allowed_ports:
        raise ValueError(
            f"Ollama api_base port must be one of {sorted(allowed_ports)}, "
            f"got: {port}. Set PERFLAB_OLLAMA_ALLOWED_PORTS={port} to allow it."
        )


@dataclass
class OllamaProvider:
    """Raw HTTP calls to Ollama REST API (no SDK dependency)."""

    name: str = "ollama"
    model: str = "llama3.2"
    api_base: str = "http://localhost:11434"

    def __post_init__(self) -> None:
        if not os.environ.get("PERFLAB_OLLAMA_ALLOW_REMOTE"):
            _validate_ollama_url(self.api_base)

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            url = f"{self.api_base.rstrip('/')}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
    ) -> CompletionResult:
        url = f"{self.api_base.rstrip('/')}/api/chat"
        payload: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        if stop:
            payload["options"]["stop"] = list(stop)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Ollama API returned HTTP {resp.status}: {resp.read().decode('utf-8', errors='replace')[:500]}"
                )
            body = json.loads(resp.read().decode("utf-8"))

        message = body.get("message", {})
        usage = {}
        if "eval_count" in body:
            usage["completion_tokens"] = body["eval_count"]
        if "prompt_eval_count" in body:
            usage["prompt_tokens"] = body["prompt_eval_count"]
            usage["total_tokens"] = body.get("prompt_eval_count", 0) + body.get("eval_count", 0)

        return CompletionResult(
            content=message.get("content", ""),
            finish_reason=body.get("done_reason"),
            usage=usage,
            raw=body,
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
        url = f"{self.api_base.rstrip('/')}/api/chat"
        payload: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        if stop:
            payload["options"]["stop"] = list(stop)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Ollama API returned HTTP {resp.status}: {resp.read().decode('utf-8', errors='replace')[:500]}"
                )
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = chunk.get("message", {})
                if content := msg.get("content"):
                    yield content
                if chunk.get("done"):
                    break
