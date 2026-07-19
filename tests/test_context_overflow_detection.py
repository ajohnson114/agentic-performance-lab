"""Tests for _is_context_overflow_error's "max_tokens" handling.

The bare substring "max_tokens" also appears in unrelated output-cap errors
(e.g. "max_tokens exceeds model's output limit"), which prompt-trimming
cannot fix -- trimming the prompt does nothing to lower a too-large
max_tokens request parameter. These tests pin the co-occurrence rule that
distinguishes a genuine context-overflow message from an output-cap message,
without regressing any of the other true-positive substrings covered in
tests/test_agent_state.py::TestContextOverflowDetection.
"""
from __future__ import annotations

from perflab.optimizers.phases.generate import _is_context_overflow_error


def test_max_tokens_with_context_indicator_matches():
    exc = Exception("max_tokens plus prompt length exceeds the context window")
    assert _is_context_overflow_error(exc) is True


def test_max_tokens_output_cap_error_does_not_match():
    exc = Exception("max_tokens exceeds model's output limit")
    assert _is_context_overflow_error(exc) is False
