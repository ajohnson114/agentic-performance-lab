"""Tests for perflab.analyzers.diff_flamegraph."""
from __future__ import annotations

from pathlib import Path

from perflab.analyzers.diff_flamegraph import (
    _extract_func_pcts,
    compute_diff_stacks,
    generate_diff_svg,
)


class TestExtractFuncPcts:
    def test_extract_pyspy_hotspots(self):
        summary = {
            "pyspy": {
                "hotspots": [
                    {"function": "matmul", "pct": 60.0},
                    {"function": "relu", "pct": 20.0},
                ]
            }
        }
        result = _extract_func_pcts(summary)
        assert result["matmul"] == 60.0
        assert result["relu"] == 20.0

    def test_extract_perf_hotspots(self):
        summary = {
            "linux_perf": {
                "hotspots": [
                    {"function": "sgemm", "pct": 80.0},
                ]
            }
        }
        result = _extract_func_pcts(summary)
        assert result["sgemm"] == 80.0

    def test_empty_summary(self):
        assert _extract_func_pcts({}) == {}


class TestComputeDiffStacks:
    def test_hotter_function(self):
        before = {"pyspy": {"hotspots": [{"function": "f", "pct": 10.0}]}}
        after = {"pyspy": {"hotspots": [{"function": "f", "pct": 50.0}]}}
        diffs = compute_diff_stacks(before, after)
        assert len(diffs) == 1
        assert diffs[0]["function"] == "f"
        assert diffs[0]["direction"] == "hotter"
        assert diffs[0]["delta"] == 40.0

    def test_cooler_function(self):
        before = {"pyspy": {"hotspots": [{"function": "f", "pct": 80.0}]}}
        after = {"pyspy": {"hotspots": [{"function": "f", "pct": 20.0}]}}
        diffs = compute_diff_stacks(before, after)
        assert diffs[0]["direction"] == "cooler"
        assert diffs[0]["delta"] == -60.0

    def test_new_function(self):
        before = {"pyspy": {"hotspots": []}}
        after = {"pyspy": {"hotspots": [{"function": "new_func", "pct": 30.0}]}}
        diffs = compute_diff_stacks(before, after)
        assert len(diffs) == 1
        assert diffs[0]["function"] == "new_func"
        assert diffs[0]["direction"] == "hotter"

    def test_removed_function(self):
        before = {"pyspy": {"hotspots": [{"function": "old_func", "pct": 50.0}]}}
        after = {"pyspy": {"hotspots": []}}
        diffs = compute_diff_stacks(before, after)
        assert diffs[0]["direction"] == "cooler"

    def test_small_delta_filtered(self):
        before = {"pyspy": {"hotspots": [{"function": "f", "pct": 10.0}]}}
        after = {"pyspy": {"hotspots": [{"function": "f", "pct": 10.3}]}}
        diffs = compute_diff_stacks(before, after)
        assert len(diffs) == 0

    def test_sorted_by_abs_delta(self):
        before = {"pyspy": {"hotspots": [
            {"function": "a", "pct": 10.0},
            {"function": "b", "pct": 50.0},
        ]}}
        after = {"pyspy": {"hotspots": [
            {"function": "a", "pct": 40.0},
            {"function": "b", "pct": 10.0},
        ]}}
        diffs = compute_diff_stacks(before, after)
        assert abs(diffs[0]["delta"]) >= abs(diffs[1]["delta"])

    def test_empty_inputs(self):
        assert compute_diff_stacks({}, {}) == []


class TestGenerateDiffSvg:
    def test_generates_svg_file(self, tmp_path: Path):
        diffs = [
            {"function": "hot_func", "before_pct": 10.0, "after_pct": 50.0, "delta": 40.0, "direction": "hotter"},
            {"function": "cool_func", "before_pct": 60.0, "after_pct": 20.0, "delta": -40.0, "direction": "cooler"},
        ]
        out = tmp_path / "diff.svg"
        result = generate_diff_svg(diffs, out)
        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert "<svg" in content
        assert "hot_func" in content
        assert "cool_func" in content

    def test_empty_diffs_returns_none(self, tmp_path: Path):
        out = tmp_path / "diff.svg"
        result = generate_diff_svg([], out)
        assert result is None

    def test_svg_contains_legend(self, tmp_path: Path):
        diffs = [{"function": "f", "before_pct": 0, "after_pct": 30, "delta": 30, "direction": "hotter"}]
        out = tmp_path / "diff.svg"
        result = generate_diff_svg(diffs, out)
        content = result.read_text()
        assert "hotter" in content.lower()
        assert "cooler" in content.lower()
