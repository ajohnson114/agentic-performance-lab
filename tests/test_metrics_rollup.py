"""Tests for perflab.analyzers.metrics_rollup."""
from __future__ import annotations

from perflab.analyzers.metrics_rollup import (
    compute_run_summary,
    is_improvement,
)


class TestComputeRunSummary:
    def test_empty_history(self):
        s = compute_run_summary([], baseline_value=10.0, mode="maximize")
        assert s.best_value == 10.0
        assert s.median_speedup == 1.0
        assert s.total_iterations == 0
        assert s.time_to_first_improvement is None
        assert s.success_rate == 0.0

    def test_maximize_single_improvement(self):
        history = [
            {"iteration": 0, "value": 100.0, "accepted": True},
            {"iteration": 1, "value": 150.0, "accepted": True},
        ]
        s = compute_run_summary(history, baseline_value=100.0, mode="maximize")
        assert s.best_value == 150.0
        # speedup for iter1 = 150/100 = 1.5
        assert s.time_to_first_improvement == 1

    def test_minimize_mode(self):
        history = [
            {"iteration": 0, "value": 200.0, "accepted": True},
            {"iteration": 1, "value": 100.0, "accepted": True},
        ]
        s = compute_run_summary(history, baseline_value=200.0, mode="minimize")
        assert s.best_value == 100.0
        # speedup for minimize = baseline/val = 200/100 = 2.0
        assert s.median_speedup == 2.0  # median of [1.0, 2.0] = sorted[1] = 2.0

    def test_no_accepted_iterations(self):
        history = [
            {"iteration": 0, "value": 10.0, "accepted": False},
            {"iteration": 1, "value": 9.0, "accepted": False},
        ]
        s = compute_run_summary(history, baseline_value=10.0, mode="maximize")
        assert s.success_rate == 0.0
        # No accepted → fallback to [1.0]
        assert s.median_speedup == 1.0

    def test_multiple_accepted(self):
        history = [
            {"iteration": 0, "value": 100.0, "accepted": True},
            {"iteration": 1, "value": 120.0, "accepted": True},
            {"iteration": 2, "value": 130.0, "accepted": True},
            {"iteration": 3, "value": 110.0, "accepted": False},
        ]
        s = compute_run_summary(history, baseline_value=100.0, mode="maximize")
        # accepted speedups: [1.0, 1.2, 1.3] → sorted: [1.0, 1.2, 1.3]
        assert s.median_speedup == 1.2  # index 3//2 = 1
        assert s.success_rate == 2 / 3  # 2 accepted out of 3 non-baseline
        assert s.total_iterations == 3

    def test_zero_baseline(self):
        history = [
            {"iteration": 0, "value": 0.0, "accepted": True},
            {"iteration": 1, "value": 5.0, "accepted": True},
        ]
        s = compute_run_summary(history, baseline_value=0.0, mode="maximize")
        # When baseline is 0, speedup stays 1.0
        assert s.median_speedup == 1.0


class TestIsImprovement:
    def test_maximize_above_tolerance(self):
        # new=1.1 > best * (1.0 + 0.01) = 1.0 * 1.01 = 1.01
        assert is_improvement(1.1, 1.0, "maximize", 0.01) is True

    def test_maximize_below_tolerance(self):
        # new=1.005 > best * 1.01 = 1.01? No.
        assert is_improvement(1.005, 1.0, "maximize", 0.01) is False

    def test_minimize_below_tolerance(self):
        # new=0.9 < best * (1.0 - 0.01) = 1.0 * 0.99 = 0.99
        assert is_improvement(0.9, 1.0, "minimize", 0.01) is True

    def test_minimize_above_tolerance(self):
        # new=0.995 < 0.99? No.
        assert is_improvement(0.995, 1.0, "minimize", 0.01) is False
