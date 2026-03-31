"""Tests for perflab.optimizers.convergence."""
from __future__ import annotations

from perflab.optimizers.convergence import ConvergenceDetector


class TestConvergenceDetector:
    def test_no_stop_initially(self):
        det = ConvergenceDetector()
        stop, reason = det.should_stop()
        assert stop is False
        assert reason == ""

    def test_stop_after_max_failures(self):
        det = ConvergenceDetector()
        for _ in range(5):
            det.record_failure()
        stop, reason = det.should_stop()
        assert stop is True
        assert "5" in reason

    def test_improvement_resets_failures(self):
        det = ConvergenceDetector()
        for _ in range(4):
            det.record_failure()
        det.record_improvement(0.10)
        # After improvement, failures reset; add 2 more → still below 5
        det.record_failure()
        det.record_failure()
        stop, _ = det.should_stop()
        assert stop is False

    def test_stop_on_diminishing_returns(self):
        det = ConvergenceDetector()
        det.record_improvement(0.01)  # 1% < 3%
        det.record_improvement(0.02)  # 2% < 3%
        stop, reason = det.should_stop()
        assert stop is True
        assert "converging" in reason.lower()

    def test_large_improvements_no_stop(self):
        det = ConvergenceDetector()
        det.record_improvement(0.10)  # 10% > 3%
        det.record_improvement(0.15)  # 15% > 3%
        stop, _ = det.should_stop()
        assert stop is False

    def test_custom_thresholds(self):
        det = ConvergenceDetector(max_consecutive_failures=2)
        det.record_failure()
        det.record_failure()
        stop, _ = det.should_stop()
        assert stop is True
