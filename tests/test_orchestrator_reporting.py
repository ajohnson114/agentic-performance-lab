"""A rejected knob must say why, at the verbosity the CLI actually runs at.

`perflab optimize` now rejects configurations the old bare-ratio search would
have accepted and written into tuning.yaml. That is the intended behavior
change, but the CLI configures logging at WARNING (`cli.py::_setup`), so a
reason emitted at INFO is invisible: the user watches a sweep reject everything
and is told nothing. A sweep that silently stops picking winners is a support
ticket; one that says "improvement within noise (CV=6.7% at n=5 ...)" is an
answer, and the answer names the fix — more repeats, a quieter host, or a wider
tolerance.

The split is deliberate rather than "log everything loudly": a trial that never
cleared the tolerance is the ordinary outcome of a sweep and would be pure
noise at WARNING.
"""
from __future__ import annotations

import logging

from perflab.analyzers.decision import DEFAULT_RULE
from perflab.orchestrator import _judge

CLI_DEFAULT_LEVEL = logging.WARNING  # cli.py::_setup without --verbose

NOISY = {  # ~6.7% CV: a 5% "win" is not resolvable at n=5
    "candidate_samples": [105.0, 98.0, 112.0, 101.0, 109.0],
    "incumbent_samples": [100.0, 92.0, 108.0, 95.0, 105.0],
}


def _judge_at_cli_level(caplog, **kwargs):
    with caplog.at_level(CLI_DEFAULT_LEVEL, logger="perflab.orchestrator"):
        return _judge(DEFAULT_RULE, mode="maximize", tolerance=0.02, **kwargs)


class TestNoiseRejectionIsVisible:
    def test_beat_tolerance_but_lost_to_noise_is_reported_at_cli_default(self, caplog):
        verdict = _judge_at_cli_level(
            caplog, candidate=105.0, incumbent=100.0, label="trial 3", **NOISY
        )
        assert verdict.improved is False
        assert verdict.beats_tolerance is True, "precondition: it did clear 2%"
        assert caplog.records, "rejection was invisible at the CLI's own log level"
        assert "within noise" in caplog.text
        assert "trial 3" in caplog.text

    def test_the_message_names_what_to_change(self, caplog):
        """A reason without a remedy is just a refusal."""
        _judge_at_cli_level(
            caplog, candidate=105.0, incumbent=100.0, label="t", **NOISY
        )
        # CV and n tell the user which lever moves: quieter host, or more repeats.
        assert "CV=" in caplog.text and "n=" in caplog.text
        assert "need" in caplog.text


class TestRoutineOutcomesStayQuiet:
    def test_failing_the_tolerance_is_not_escalated(self, caplog):
        """The ordinary result of a sweep. Escalating it would drown the signal."""
        verdict = _judge_at_cli_level(
            caplog,
            candidate=100.5, incumbent=100.0, label="trial 4",
            candidate_samples=[100.5] * 5, incumbent_samples=[100.0] * 5,
        )
        assert verdict.improved is False
        assert verdict.beats_tolerance is False
        assert not caplog.records, f"routine rejection escalated: {caplog.text}"

    def test_a_genuine_win_is_accepted_without_a_warning(self, caplog):
        verdict = _judge_at_cli_level(
            caplog,
            candidate=300.0, incumbent=100.0, label="trial 5",
            candidate_samples=[300.0, 299.0, 301.0, 300.0, 300.0],
            incumbent_samples=[100.0, 99.0, 101.0, 100.0, 100.0],
        )
        assert verdict.improved is True
        assert not caplog.records


class TestUnverifiedAcceptIsStillDisclosed:
    def test_accept_without_samples_says_so(self, caplog):
        """No per-repeat samples means only the tolerance was checked. The user
        should be able to find that out rather than assume the gate ran."""
        with caplog.at_level(logging.INFO, logger="perflab.orchestrator"):
            verdict = _judge(
                DEFAULT_RULE, candidate=105.0, incumbent=100.0, mode="maximize",
                tolerance=0.02, candidate_samples=[], incumbent_samples=[],
                label="trial 6",
            )
        assert verdict.improved is True
        assert verdict.verified is False
        assert "without variance verification" in caplog.text
