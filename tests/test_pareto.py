"""Tests for Pareto frontier computation and plotting."""
from __future__ import annotations

from pathlib import Path

from perflab.reporting.plots import compute_pareto_frontier, plot_pareto_frontier


class TestComputeParetoFrontier:
    def test_simple_frontier(self):
        points = [
            {"primary": 10.0, "secondary": 5.0},
            {"primary": 8.0, "secondary": 3.0},
            {"primary": 6.0, "secondary": 2.0},
            {"primary": 7.0, "secondary": 4.0},  # dominated by (8, 3)
        ]
        frontier = compute_pareto_frontier(points, "maximize", "minimize")
        # (10, 5), (8, 3), (6, 2) are on frontier; (7, 4) is dominated
        assert len(frontier) == 3
        primary_vals = [p["primary"] for p in frontier]
        assert 7.0 not in primary_vals

    def test_all_on_frontier(self):
        points = [
            {"primary": 10.0, "secondary": 1.0},
            {"primary": 5.0, "secondary": 0.5},
        ]
        frontier = compute_pareto_frontier(points, "maximize", "minimize")
        assert len(frontier) == 2

    def test_single_point(self):
        points = [{"primary": 5.0, "secondary": 3.0}]
        frontier = compute_pareto_frontier(points, "maximize", "minimize")
        assert len(frontier) == 1

    def test_empty_points(self):
        frontier = compute_pareto_frontier([], "maximize", "minimize")
        assert frontier == []

    def test_minimize_both(self):
        points = [
            {"primary": 1.0, "secondary": 5.0},
            {"primary": 2.0, "secondary": 3.0},
            {"primary": 3.0, "secondary": 1.0},
            {"primary": 2.5, "secondary": 4.0},  # dominated by (2, 3)
        ]
        frontier = compute_pareto_frontier(points, "minimize", "minimize")
        assert len(frontier) == 3
        primary_vals = [p["primary"] for p in frontier]
        assert 2.5 not in primary_vals

    def test_maximize_both(self):
        points = [
            {"primary": 10.0, "secondary": 10.0},
            {"primary": 8.0, "secondary": 12.0},
            {"primary": 5.0, "secondary": 5.0},  # dominated
        ]
        frontier = compute_pareto_frontier(points, "maximize", "maximize")
        assert len(frontier) == 2

    def test_frontier_sorted_by_primary(self):
        points = [
            {"primary": 5.0, "secondary": 1.0},
            {"primary": 10.0, "secondary": 5.0},
            {"primary": 8.0, "secondary": 3.0},
        ]
        frontier = compute_pareto_frontier(points, "maximize", "minimize")
        primary_vals = [p["primary"] for p in frontier]
        assert primary_vals == sorted(primary_vals, reverse=True)


class TestPlotParetoFrontier:
    def test_generates_png(self, tmp_path: Path):
        points = [
            {"primary": 10.0, "secondary": 5.0, "label": "iter1"},
            {"primary": 8.0, "secondary": 3.0, "label": "iter2"},
            {"primary": 6.0, "secondary": 2.0, "label": "iter3"},
        ]
        frontier = compute_pareto_frontier(points, "maximize", "minimize")
        out = tmp_path / "pareto.png"
        plot_pareto_frontier(out, points, frontier, "TFLOPS", "Latency (ms)")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_points_no_crash(self, tmp_path: Path):
        out = tmp_path / "pareto.png"
        plot_pareto_frontier(out, [], [], "A", "B")
        assert not out.exists()

    def test_single_point(self, tmp_path: Path):
        points = [{"primary": 5.0, "secondary": 3.0}]
        frontier = [{"primary": 5.0, "secondary": 3.0}]
        out = tmp_path / "pareto.png"
        plot_pareto_frontier(out, points, frontier, "Speed", "Power")
        assert out.exists()
