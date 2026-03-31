"""Tests for perflab.analyzers.bench_stats."""
from __future__ import annotations

from perflab.analyzers.bench_stats import (
    BenchStats,
    compute_bench_stats,
    extract_repeated_values,
    format_bench_stats_for_prompt,
)


class TestComputeBenchStats:
    def test_none_for_single_value(self):
        assert compute_bench_stats([42.0]) is None

    def test_none_for_empty(self):
        assert compute_bench_stats([]) is None

    def test_basic_stats(self):
        stats = compute_bench_stats([100.0, 100.0, 100.0, 100.0])
        assert stats is not None
        assert stats.n == 4
        assert stats.mean == 100.0
        assert stats.cv == 0.0
        assert not stats.is_noisy

    def test_noisy_detection(self):
        # Values with ~20% CV
        stats = compute_bench_stats([80.0, 100.0, 120.0], cv_threshold=0.10)
        assert stats is not None
        assert stats.is_noisy
        assert "variance" in stats.warning.lower()

    def test_not_noisy_within_threshold(self):
        stats = compute_bench_stats([99.0, 100.0, 101.0], cv_threshold=0.10)
        assert stats is not None
        assert not stats.is_noisy

    def test_confidence_interval(self):
        stats = compute_bench_stats([10.0, 20.0, 30.0])
        assert stats is not None
        assert stats.ci_95_low < stats.mean
        assert stats.ci_95_high > stats.mean

    def test_median_even(self):
        stats = compute_bench_stats([1.0, 2.0, 3.0, 4.0])
        assert stats is not None
        assert stats.median == 2.5

    def test_median_odd(self):
        stats = compute_bench_stats([1.0, 3.0, 5.0])
        assert stats is not None
        assert stats.median == 3.0

    def test_custom_threshold(self):
        stats = compute_bench_stats([90.0, 100.0, 110.0], cv_threshold=0.20)
        assert stats is not None
        assert not stats.is_noisy


class TestExtractRepeatedValues:
    def test_raw_values(self):
        bench = {"throughput": {"median": 100.0, "raw_values": [95.0, 100.0, 105.0]}}
        vals = extract_repeated_values(bench, "throughput.median")
        assert vals == [95.0, 100.0, 105.0]

    def test_samples_key(self):
        bench = {"throughput": {"median": 100.0, "samples": [95.0, 105.0]}}
        vals = extract_repeated_values(bench, "throughput.median")
        assert vals == [95.0, 105.0]

    def test_missing_raw(self):
        bench = {"throughput": {"median": 100.0}}
        vals = extract_repeated_values(bench, "throughput.median")
        assert vals == []

    def test_single_part_metric(self):
        vals = extract_repeated_values({"value": 42.0}, "value")
        assert vals == []


class TestFormatBenchStats:
    def test_noisy_format(self):
        stats = BenchStats(
            n=5, mean=100.0, median=100.0, std=20.0, cv=0.20,
            ci_95_low=82.0, ci_95_high=118.0, is_noisy=True,
            warning="High measurement variance detected: CV=20.0%"
        )
        text = format_bench_stats_for_prompt(stats)
        assert "WARNING" in text
        assert "CV=20.0%" in text

    def test_clean_format(self):
        stats = BenchStats(
            n=5, mean=100.0, median=100.0, std=1.0, cv=0.01,
            ci_95_low=99.1, ci_95_high=100.9, is_noisy=False,
        )
        text = format_bench_stats_for_prompt(stats)
        assert "95% CI" in text
        assert "WARNING" not in text
