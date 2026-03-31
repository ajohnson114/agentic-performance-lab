"""Tests for perflab.reporting.plots."""
from __future__ import annotations

from pathlib import Path

from perflab.reporting.plots import plot_metric_history


class TestPlotMetricHistory:
    def test_creates_valid_png(self, tmp_path: Path):
        out = tmp_path / "metric.png"
        plot_metric_history(out, [0, 1, 2], [10.0, 12.0, 11.0], "throughput")
        assert out.exists()
        data = out.read_bytes()
        assert len(data) > 100  # non-trivial PNG
        assert data[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG header

    def test_with_baseline(self, tmp_path: Path):
        out = tmp_path / "metric.png"
        plot_metric_history(out, [0, 1, 2], [10.0, 12.0, 11.0], "throughput", baseline_val=10.0)
        data = out.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_single_point(self, tmp_path: Path):
        """Should handle a single data point without crashing."""
        out = tmp_path / "metric.png"
        plot_metric_history(out, [0], [10.0], "latency_ms")
        assert out.exists()
        assert out.read_bytes()[:4] == b"\x89PNG"

    def test_many_points(self, tmp_path: Path):
        """Should handle many data points (axis labels don't overlap)."""
        out = tmp_path / "metric.png"
        plot_metric_history(out, list(range(100)), [float(i) for i in range(100)], "throughput")
        assert out.exists()
        assert out.stat().st_size > 0
