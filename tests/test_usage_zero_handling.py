"""Legitimate zero values must not be treated as missing.

Covers the token-usage helpers (a provider may genuinely report 0 tokens) and
the evaluate phase's rejection bookkeeping (a candidate can score a real 0.0).
"""
from __future__ import annotations

from types import SimpleNamespace

from perflab.optimizers.progress import fmt_usage, usage_input_tokens, usage_output_tokens

# -- usage helpers ------------------------------------------------------------


def test_zero_primary_key_is_kept():
    # `usage.get("input_tokens") or usage.get("prompt_tokens", 0)` would
    # wrongly fall through to 99 here.
    assert usage_input_tokens({"input_tokens": 0, "prompt_tokens": 99}) == 0
    assert usage_output_tokens({"output_tokens": 0, "completion_tokens": 99}) == 0


def test_fallback_key_used_when_primary_missing():
    assert usage_input_tokens({"prompt_tokens": 7}) == 7
    assert usage_output_tokens({"completion_tokens": 3}) == 3


def test_zero_fallback_key_is_kept():
    assert usage_input_tokens({"prompt_tokens": 0}) == 0
    assert usage_output_tokens({"completion_tokens": 0}) == 0


def test_primary_key_preferred_over_fallback():
    assert usage_input_tokens({"input_tokens": 11, "prompt_tokens": 22}) == 11
    assert usage_output_tokens({"output_tokens": 5, "completion_tokens": 9}) == 5


def test_empty_usage_returns_zero():
    assert usage_input_tokens({}) == 0
    assert usage_output_tokens({}) == 0


def test_fmt_usage_reports_real_zeros():
    usage = {"input_tokens": 0, "output_tokens": 0, "prompt_tokens": 5, "completion_tokens": 6}
    assert fmt_usage(usage) == "in=0, out=0, total=0"


# -- evaluate: rejected-candidate history value -------------------------------


class _NoOpEventLog:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_reject_history_records_genuine_zero_metric(tmp_path):
    """A best candidate whose metric is a real 0.0 must be recorded as 0.0 in
    the no-improvement history entry, not silently replaced by ctx.best_value."""
    from perflab.optimizers.phases.evaluate import BeamCandidate, accept_best

    ctx = SimpleNamespace(
        task=SimpleNamespace(
            benchmark=SimpleNamespace(
                metric=SimpleNamespace(name="throughput.median", mode="maximize"),
            ),
            constraints=SimpleNamespace(regression_tolerance=0.02),
        ),
        ws=tmp_path,
        rp=SimpleNamespace(run_dir=tmp_path),
        iteration=2,
        progress=SimpleNamespace(on_message=lambda m: None),
        event_log=_NoOpEventLog(),
        history=[],
        baseline_val=1.0,
        best_value=1.0,
        best_iter=1,
        accepted_patches=[],
        accepted_count=0,
        sec_metric=None,
        config=SimpleNamespace(isolation=None, top_k=2),
    )
    cand = BeamCandidate(
        iteration=2, index=0, blocks=[], description="candidate 1: 1 blocks", value=0.0,
    )

    accepted, rel_improvement, accepted_value = accept_best(
        ctx, [cand], use_fast=False,
    )

    assert accepted is False
    assert rel_improvement is None and accepted_value is None
    assert len(ctx.history) == 1
    entry = ctx.history[0]
    assert entry["accepted"] is False
    assert entry["value"] == 0.0
