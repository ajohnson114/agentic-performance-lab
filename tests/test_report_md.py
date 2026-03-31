"""Tests for perflab.reporting.report_md."""
from __future__ import annotations

from pathlib import Path

from perflab.reporting.report_md import write_report_md


def _minimal_data(**overrides):
    data = {
        "task_name": "matmul_opt",
        "run_id": "run-001",
        "metric_name": "gflops",
        "metric_mode": "maximize",
        "best_value": 120.0,
        "best_iter": 3,
        "baseline_value": 100.0,
        "rows": [
            {"iter": 0, "value": 100.0, "accepted": True, "notes": "baseline"},
            {"iter": 1, "value": 110.0, "accepted": True, "notes": "first try"},
            {"iter": 2, "value": 105.0, "accepted": False, "notes": "regression"},
            {"iter": 3, "value": 120.0, "accepted": True, "notes": "best"},
        ],
        "run_summary": {
            "baseline_value": 100.0,
            "best_value": 120.0,
            "median_speedup": 1.15,
            "p90_speedup": 1.20,
            "time_to_first_improvement": 1,
            "success_rate": 0.67,
            "total_iterations": 3,
        },
        "latest_artifacts": {},
    }
    data.update(overrides)
    return data


class TestWriteReportMd:
    def test_creates_file(self, tmp_path: Path):
        p = tmp_path / "report.md"
        write_report_md(p, _minimal_data())
        assert p.exists()

    def test_contains_task_name(self, tmp_path: Path):
        p = tmp_path / "report.md"
        write_report_md(p, _minimal_data())
        text = p.read_text()
        assert "# PerfLab Report — matmul_opt" in text

    def test_best_metric_section(self, tmp_path: Path):
        p = tmp_path / "report.md"
        write_report_md(p, _minimal_data())
        text = p.read_text()
        assert "120" in text
        assert "1.20x" in text  # speedup = 120/100

    def test_iterations_table(self, tmp_path: Path):
        p = tmp_path / "report.md"
        write_report_md(p, _minimal_data())
        text = p.read_text()
        assert "baseline" in text
        assert "first try" in text
        assert "## Iterations" in text

    def test_bottleneck_diagnosis_section(self, tmp_path: Path):
        p = tmp_path / "report.md"
        data = _minimal_data(bottleneck_diagnoses=[
            {"rank": 1, "bottleneck": "Low SM util", "root_cause": "small kernels", "confidence": "high"},
        ])
        write_report_md(p, data)
        text = p.read_text()
        assert "Bottleneck diagnosis" in text
        assert "Low SM util" in text

    def test_early_stop_section(self, tmp_path: Path):
        p = tmp_path / "report.md"
        data = _minimal_data(early_stop_reason="5 consecutive failures")
        write_report_md(p, data)
        text = p.read_text()
        assert "Early stop" in text
        assert "5 consecutive failures" in text

    def test_no_speedup_column_when_absent(self, tmp_path: Path):
        p = tmp_path / "report.md"
        # Rows without speedup key → simpler table
        data = _minimal_data()
        write_report_md(p, data)
        text = p.read_text()
        # No speedup in rows → should use the simple table format
        assert "| iter | value | accepted | notes |" in text
