"""Tests for the LLM provider layer -- availability, client hardening, parsing, retries.

SDK-backed providers (anthropic/openai) are mocked via patch.dict("sys.modules", ...)
so tests run without the real SDKs; OllamaProvider is exercised by patching
urllib.request.urlopen.
"""
from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from perflab.llm.anthropic_provider import AnthropicProvider
from perflab.llm.base import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT_S, Message
from perflab.llm.ollama_provider import OllamaProvider, _validate_ollama_url
from perflab.llm.openai_provider import OpenAIProvider


class TestAnthropicProvider:
    def test_is_available_false_when_api_key_empty(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            assert AnthropicProvider(api_key="").is_available() is False

    def test_is_available_false_when_sdk_missing(self):
        # None in sys.modules makes `import anthropic` raise ImportError
        with patch.dict("sys.modules", {"anthropic": None}):
            assert AnthropicProvider(api_key="sk-x").is_available() is False

    def test_is_available_true_with_sdk_and_key(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            assert AnthropicProvider(api_key="sk-x").is_available() is True

    def test_client_passes_key_timeout_and_retries(self):
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            AnthropicProvider(api_key="sk-x")._client()
        kwargs = mock_anthropic.Anthropic.call_args.kwargs
        assert kwargs["api_key"] == "sk-x"
        assert kwargs["timeout"] == DEFAULT_TIMEOUT_S
        assert kwargs["max_retries"] == DEFAULT_MAX_RETRIES

    def test_complete_parses_content_finish_reason_and_usage(self):
        resp = SimpleNamespace(
            content=[SimpleNamespace(text="hello "), SimpleNamespace(text="world")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=4),
        )
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.return_value = resp
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = AnthropicProvider(api_key="sk-x").complete(
                [Message("system", "be terse"), Message("user", "hi")]
            )
        assert result.content == "hello world"
        assert result.finish_reason == "end_turn"
        assert result.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        }
        # system message is extracted into the separate parameter
        create_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args.kwargs
        assert create_kwargs["system"] == "be terse"
        assert create_kwargs["messages"] == [{"role": "user", "content": "hi"}]


class TestOpenAIProvider:
    def test_is_available_false_when_api_key_empty(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            assert OpenAIProvider(api_key="").is_available() is False

    def test_is_available_false_when_sdk_missing(self):
        with patch.dict("sys.modules", {"openai": None}):
            assert OpenAIProvider(api_key="sk-x").is_available() is False

    def test_is_available_true_with_sdk_and_key(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            assert OpenAIProvider(api_key="sk-x").is_available() is True

    def test_client_passes_key_timeout_and_retries(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            OpenAIProvider(api_key="sk-x")._client()
        kwargs = mock_openai.OpenAI.call_args.kwargs
        assert kwargs["api_key"] == "sk-x"
        assert kwargs["timeout"] == DEFAULT_TIMEOUT_S
        assert kwargs["max_retries"] == DEFAULT_MAX_RETRIES
        assert "base_url" not in kwargs

    def test_client_passes_api_base_when_set(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            OpenAIProvider(api_key="sk-x", api_base="http://localhost:8000/v1")._client()
        kwargs = mock_openai.OpenAI.call_args.kwargs
        assert kwargs["base_url"] == "http://localhost:8000/v1"
        assert kwargs["timeout"] == DEFAULT_TIMEOUT_S
        assert kwargs["max_retries"] == DEFAULT_MAX_RETRIES

    def test_complete_parses_content_finish_reason_and_usage(self):
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hi there"), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        )
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = resp
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = OpenAIProvider(api_key="sk-x").complete([Message("user", "hi")])
        assert result.content == "hi there"
        assert result.finish_reason == "stop"
        assert result.usage == {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        }

    def test_complete_none_content_becomes_empty_string(self):
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="stop")],
            usage=None,
        )
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = resp
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = OpenAIProvider(api_key="sk-x").complete([Message("user", "hi")])
        assert result.content == ""
        assert result.usage == {}


_OLLAMA_BODY = {
    "message": {"content": "pong"},
    "done_reason": "stop",
    "eval_count": 5,
    "prompt_eval_count": 7,
}


def _ollama_response(body: dict) -> MagicMock:
    """Mocked urlopen return value: context manager with .status and .read()."""
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = json.dumps(body).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://localhost:11434/api/chat", code, "err", None, None)


@patch("perflab.llm.ollama_provider.time.sleep")
@patch("urllib.request.urlopen")
class TestOllamaProviderRetries:
    def test_complete_retries_after_urlerror(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            urllib.error.URLError("connection refused"),
            _ollama_response(_OLLAMA_BODY),
        ]
        result = OllamaProvider().complete([Message("user", "ping")])
        assert result.content == "pong"
        assert result.finish_reason == "stop"
        assert result.usage == {
            "completion_tokens": 5,
            "prompt_tokens": 7,
            "total_tokens": 12,
        }
        assert mock_urlopen.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    def test_complete_http_404_raises_without_retry(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = _http_error(404)
        with pytest.raises(urllib.error.HTTPError):
            OllamaProvider().complete([Message("user", "ping")])
        assert mock_urlopen.call_count == 1
        mock_sleep.assert_not_called()

    def test_complete_http_500_is_retried(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [_http_error(500), _ollama_response(_OLLAMA_BODY)]
        result = OllamaProvider().complete([Message("user", "ping")])
        assert result.content == "pong"
        assert mock_urlopen.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    def test_complete_raises_after_retries_exhausted(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = urllib.error.URLError("still down")
        with pytest.raises(urllib.error.URLError):
            OllamaProvider().complete([Message("user", "ping")])
        assert mock_urlopen.call_count == 3  # initial + 2 retries
        assert [c.args[0] for c in mock_sleep.call_args_list] == [1.0, 4.0]

    def test_stream_retries_establishing_connection(self, mock_urlopen, mock_sleep):
        chunks = [
            {"message": {"content": "po"}},
            {"message": {"content": "ng"}, "done": True},
        ]
        resp = _ollama_response({})
        resp.__iter__.return_value = iter(
            [json.dumps(c).encode("utf-8") + b"\n" for c in chunks]
        )
        mock_urlopen.side_effect = [urllib.error.URLError("connection refused"), resp]
        out = list(OllamaProvider().stream([Message("user", "ping")]))
        assert out == ["po", "ng"]
        assert mock_urlopen.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    def test_stream_never_retries_after_first_chunk(self, mock_urlopen, mock_sleep):
        def lines():
            yield json.dumps({"message": {"content": "po"}}).encode("utf-8")
            raise urllib.error.URLError("connection dropped mid-stream")

        resp = _ollama_response({})
        resp.__iter__.return_value = lines()
        mock_urlopen.return_value = resp

        stream = OllamaProvider().stream([Message("user", "ping")])
        assert next(stream) == "po"
        with pytest.raises(urllib.error.URLError):
            next(stream)
        assert mock_urlopen.call_count == 1
        mock_sleep.assert_not_called()


class TestValidateOllamaUrl:
    def test_localhost_default_port_ok(self):
        _validate_ollama_url("http://localhost:11434")

    def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            _validate_ollama_url("ftp://localhost:11434")

    def test_rejects_remote_host(self):
        with pytest.raises(ValueError, match="host"):
            _validate_ollama_url("http://evil.example.com:11434")

    def test_rejects_disallowed_port(self):
        with pytest.raises(ValueError, match="port"):
            _validate_ollama_url("http://localhost:8080")

    def test_extra_port_allowed_via_env(self):
        with patch.dict("os.environ", {"PERFLAB_OLLAMA_ALLOWED_PORTS": "8080"}):
            _validate_ollama_url("http://localhost:8080")
