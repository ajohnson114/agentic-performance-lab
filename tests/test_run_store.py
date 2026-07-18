"""Tests for perflab.memory.run_store and CLI list-runs / compare commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perflab.memory.run_store import RunStore

# ── RunStore.new_run ──────────────────────────────────────────────

class TestNewRun:
    def test_creates_run_dirs(self, tmp_path: Path):
        store = RunStore(tmp_path)
        rp = store.new_run("matmul/python")
        assert rp.run_dir.exists()
        assert rp.artifacts_dir.exists()
        assert rp.logs_dir.exists()

    def test_writes_meta_json(self, tmp_path: Path):
        store = RunStore(tmp_path)
        rp = store.new_run("matmul/python", program_type="python")
        meta = json.loads((rp.run_dir / "meta.json").read_text())
        assert meta["task"] == "matmul/python"
        assert meta["program_type"] == "python"
        assert meta["run_id"] == rp.run_id

    def test_appends_to_index(self, tmp_path: Path):
        store = RunStore(tmp_path)
        store.new_run("task_a")
        store.new_run("task_b")
        lines = store.index_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["task"] == "task_a"
        assert json.loads(lines[1])["task"] == "task_b"

    def test_run_id_is_unique(self, tmp_path: Path):
        store = RunStore(tmp_path)
        ids = {store.new_run("t").run_id for _ in range(5)}
        assert len(ids) == 5

    def test_program_type_optional(self, tmp_path: Path):
        store = RunStore(tmp_path)
        rp = store.new_run("t")
        meta = json.loads((rp.run_dir / "meta.json").read_text())
        assert "program_type" not in meta


# ── RunStore.list_runs ────────────────────────────────────────────

class TestListRuns:
    def test_empty_store(self, tmp_path: Path):
        store = RunStore(tmp_path)
        assert store.list_runs() == []

    def test_returns_newest_first(self, tmp_path: Path):
        store = RunStore(tmp_path)
        r1 = store.new_run("t")
        r2 = store.new_run("t")
        runs = store.list_runs()
        assert runs[0]["run_id"] == r2.run_id
        assert runs[1]["run_id"] == r1.run_id

    def test_filter_by_task(self, tmp_path: Path):
        store = RunStore(tmp_path)
        store.new_run("alpha")
        store.new_run("beta")
        store.new_run("alpha")
        runs = store.list_runs(task="alpha")
        assert len(runs) == 2
        assert all(r["task"] == "alpha" for r in runs)

    def test_limit(self, tmp_path: Path):
        store = RunStore(tmp_path)
        for _ in range(5):
            store.new_run("t")
        runs = store.list_runs(limit=3)
        assert len(runs) == 3

    def test_enriches_from_meta_json(self, tmp_path: Path):
        store = RunStore(tmp_path)
        rp = store.new_run("t")
        # Write extra fields to meta.json that aren't in index
        store.update_meta(rp.run_id, {"best_value": 42.0, "status": "done"})
        runs = store.list_runs()
        assert runs[0]["best_value"] == 42.0
        assert runs[0]["status"] == "done"

    def test_handles_corrupt_index_lines(self, tmp_path: Path):
        store = RunStore(tmp_path)
        store.new_run("t")
        # Append a corrupt line
        with store.index_path.open("a") as f:
            f.write("not-json\n")
        runs = store.list_runs()
        assert len(runs) == 1

    def test_handles_blank_lines(self, tmp_path: Path):
        store = RunStore(tmp_path)
        store.new_run("t")
        with store.index_path.open("a") as f:
            f.write("\n\n")
        runs = store.list_runs()
        assert len(runs) == 1


# ── RunStore.get_run ──────────────────────────────────────────────

class TestGetRun:
    def test_returns_meta(self, tmp_path: Path):
        store = RunStore(tmp_path)
        rp = store.new_run("t")
        data = store.get_run(rp.run_id)
        assert data["meta"]["task"] == "t"
        assert data["run_id"] == rp.run_id

    def test_not_found_raises(self, tmp_path: Path):
        store = RunStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.get_run("nonexistent-id")

    def test_loads_report_json(self, tmp_path: Path):
        store = RunStore(tmp_path)
        rp = store.new_run("t")
        report = {"best_value": 99.0, "bottleneck_diagnoses": []}
        (rp.run_dir / "report.json").write_text(json.dumps(report))
        data = store.get_run(rp.run_id)
        assert data["report"]["best_value"] == 99.0

    def test_loads_bench_json(self, tmp_path: Path):
        store = RunStore(tmp_path)
        rp = store.new_run("t")
        bench = {"wall_time": 1.23}
        (rp.run_dir / "bench.json").write_text(json.dumps(bench))
        data = store.get_run(rp.run_id)
        assert data["bench"]["wall_time"] == 1.23

    def test_loads_profiler_summaries(self, tmp_path: Path):
        store = RunStore(tmp_path)
        rp = store.new_run("t")
        (rp.artifacts_dir / "pyspy_summary.json").write_text(
            json.dumps({"top_func": "main"})
        )
        data = store.get_run(rp.run_id)
        assert "pyspy" in data["profiler_summaries"]
        assert data["profiler_summaries"]["pyspy"]["top_func"] == "main"

    def test_missing_optional_files(self, tmp_path: Path):
        store = RunStore(tmp_path)
        rp = store.new_run("t")
        data = store.get_run(rp.run_id)
        assert data["report"] is None
        assert data["bench"] is None
        assert data["profiler_summaries"] == {}


# ── RunStore.compare_runs ─────────────────────────────────────────

class TestCompareRuns:
    def test_basic_comparison(self, tmp_path: Path):
        store = RunStore(tmp_path)
        ra = store.new_run("t")
        rb = store.new_run("t")
        (ra.run_dir / "report.json").write_text(json.dumps({"best_value": 10.0}))
        (rb.run_dir / "report.json").write_text(json.dumps({"best_value": 20.0}))
        result = store.compare_runs(ra.run_id, rb.run_id)
        assert result["value_a"] == 10.0
        assert result["value_b"] == 20.0
        assert result["delta"] == 10.0
        assert result["ratio"] == 2.0

    def test_no_values(self, tmp_path: Path):
        store = RunStore(tmp_path)
        ra = store.new_run("t")
        rb = store.new_run("t")
        result = store.compare_runs(ra.run_id, rb.run_id)
        assert result["delta"] is None
        assert result["ratio"] is None

    def test_bottleneck_diff(self, tmp_path: Path):
        store = RunStore(tmp_path)
        ra = store.new_run("t")
        rb = store.new_run("t")
        (ra.run_dir / "report.json").write_text(json.dumps({
            "bottleneck_diagnoses": [{"bottleneck": "mem_bound"}, {"bottleneck": "low_gpu_util"}]
        }))
        (rb.run_dir / "report.json").write_text(json.dumps({
            "bottleneck_diagnoses": [{"bottleneck": "low_gpu_util"}, {"bottleneck": "cache_miss"}]
        }))
        result = store.compare_runs(ra.run_id, rb.run_id)
        assert result["resolved_bottlenecks"] == ["mem_bound"]
        assert result["new_bottlenecks"] == ["cache_miss"]

    def test_zero_value_a(self, tmp_path: Path):
        store = RunStore(tmp_path)
        ra = store.new_run("t")
        rb = store.new_run("t")
        (ra.run_dir / "report.json").write_text(json.dumps({"best_value": 0.0}))
        (rb.run_dir / "report.json").write_text(json.dumps({"best_value": 5.0}))
        result = store.compare_runs(ra.run_id, rb.run_id)
        assert result["ratio"] is None

    def test_extracts_metric_context(self, tmp_path: Path):
        store = RunStore(tmp_path)
        ra = store.new_run("matmul/python")
        rb = store.new_run("matmul/python")
        (ra.run_dir / "report.json").write_text(json.dumps({
            "best_value": 100.0, "metric_name": "gflops", "metric_mode": "maximize",
            "task_name": "matmul/python",
        }))
        (rb.run_dir / "report.json").write_text(json.dumps({
            "best_value": 200.0, "metric_name": "gflops", "metric_mode": "maximize",
        }))
        result = store.compare_runs(ra.run_id, rb.run_id)
        assert result["metric_name"] == "gflops"
        assert result["metric_mode"] == "maximize"
        assert result["task_name"] == "matmul/python"

    def test_extracts_status(self, tmp_path: Path):
        store = RunStore(tmp_path)
        ra = store.new_run("t")
        rb = store.new_run("t")
        store.update_meta(ra.run_id, {"status": "profiled"})
        store.update_meta(rb.run_id, {"status": "completed"})
        result = store.compare_runs(ra.run_id, rb.run_id)
        assert result["status_a"] == "profiled"
        assert result["status_b"] == "completed"

    def test_profile_vs_agent_run(self, tmp_path: Path):
        """Comparing a profile run to an agent run works correctly."""
        store = RunStore(tmp_path)
        # Profile run: has report.json but status=profiled
        rp = store.new_run("matmul/python")
        store.update_meta(rp.run_id, {"status": "profiled", "best_value": 50.0})
        (rp.run_dir / "report.json").write_text(json.dumps({
            "best_value": 50.0, "metric_name": "gflops", "metric_mode": "maximize",
            "task_name": "matmul/python", "bottleneck_diagnoses": [{"bottleneck": "no_vectorization"}],
        }))
        # Agent run: status=completed, improved value
        ra = store.new_run("matmul/python")
        store.update_meta(ra.run_id, {"status": "completed", "best_value": 150.0})
        (ra.run_dir / "report.json").write_text(json.dumps({
            "best_value": 150.0, "metric_name": "gflops", "metric_mode": "maximize",
            "task_name": "matmul/python", "bottleneck_diagnoses": [],
        }))
        result = store.compare_runs(rp.run_id, ra.run_id)
        assert result["status_a"] == "profiled"
        assert result["status_b"] == "completed"
        assert result["value_a"] == 50.0
        assert result["value_b"] == 150.0
        assert result["delta"] == 100.0
        assert result["ratio"] == 3.0
        assert result["resolved_bottlenecks"] == ["no_vectorization"]
        assert result["new_bottlenecks"] == []

    def test_task_name_fallback_to_meta(self, tmp_path: Path):
        """task_name falls back to 'task' field in meta when report lacks task_name."""
        store = RunStore(tmp_path)
        ra = store.new_run("matmul/cpp")
        rb = store.new_run("matmul/cpp")
        result = store.compare_runs(ra.run_id, rb.run_id)
        assert result["task_name"] == "matmul/cpp"


# ── RunStore.update_meta ──────────────────────────────────────────

class TestUpdateMeta:
    def test_updates_existing_meta(self, tmp_path: Path):
        store = RunStore(tmp_path)
        rp = store.new_run("t")
        store.update_meta(rp.run_id, {"status": "completed", "best_value": 5.5})
        meta = json.loads((rp.run_dir / "meta.json").read_text())
        assert meta["status"] == "completed"
        assert meta["best_value"] == 5.5
        assert meta["task"] == "t"  # original fields preserved

    def test_creates_meta_if_missing(self, tmp_path: Path):
        store = RunStore(tmp_path)
        rp = store.new_run("t")
        (rp.run_dir / "meta.json").unlink()
        store.update_meta(rp.run_id, {"status": "partial"})
        meta = json.loads((rp.run_dir / "meta.json").read_text())
        assert meta["status"] == "partial"


# ── CLI list-runs command ─────────────────────────────────────────

class TestListRunsCLI:
    def test_no_runs(self, tmp_path: Path):
        from typer.testing import CliRunner

        from perflab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["list-runs", "--out-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_shows_runs(self, tmp_path: Path):
        from typer.testing import CliRunner

        from perflab.cli import app

        store = RunStore(tmp_path)
        rp = store.new_run("matmul/python", program_type="python")
        store.update_meta(rp.run_id, {"status": "done", "best_value": 42.0})

        runner = CliRunner()
        result = runner.invoke(app, ["list-runs", "--out-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert rp.run_id in result.output
        assert "matmul/python" in result.output
        assert "best=42" in result.output
        assert "type=python" in result.output

    def test_task_filter(self, tmp_path: Path):
        from typer.testing import CliRunner

        from perflab.cli import app

        store = RunStore(tmp_path)
        store.new_run("alpha")
        store.new_run("beta")

        runner = CliRunner()
        result = runner.invoke(app, ["list-runs", "--out-dir", str(tmp_path), "--task", "alpha"])
        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "beta" not in result.output

    def test_limit_flag(self, tmp_path: Path):
        from typer.testing import CliRunner

        from perflab.cli import app

        store = RunStore(tmp_path)
        for i in range(5):
            store.new_run(f"task_{i}")

        runner = CliRunner()
        result = runner.invoke(app, ["list-runs", "--out-dir", str(tmp_path), "--limit", "2"])
        assert result.exit_code == 0
        # Should only show 2 runs
        lines = [line for line in result.output.strip().splitlines() if line.strip().startswith("20")]
        assert len(lines) == 2


# ── CLI compare command ───────────────────────────────────────────

class TestCompareCLI:
    def _make_run(self, store, task="t", status="completed", best_value=None,
                  metric_name=None, metric_mode=None, bottlenecks=None):
        rp = store.new_run(task)
        store.update_meta(rp.run_id, {"status": status})
        report = {"best_value": best_value} if best_value is not None else {}
        if metric_name:
            report["metric_name"] = metric_name
        if metric_mode:
            report["metric_mode"] = metric_mode
        if bottlenecks is not None:
            report["bottleneck_diagnoses"] = [{"bottleneck": b} for b in bottlenecks]
        report["task_name"] = task
        (rp.run_dir / "report.json").write_text(json.dumps(report))
        return rp

    def test_basic_output(self, tmp_path: Path):
        from typer.testing import CliRunner

        from perflab.cli import app

        store = RunStore(tmp_path)
        ra = self._make_run(store, task="matmul/python", status="profiled",
                            best_value=10.0, metric_name="gflops", metric_mode="maximize")
        rb = self._make_run(store, task="matmul/python", status="completed",
                            best_value=30.0, metric_name="gflops", metric_mode="maximize")

        runner = CliRunner()
        result = runner.invoke(app, ["compare", ra.run_id, rb.run_id, "--out-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "matmul/python" in result.output
        assert "gflops" in result.output
        assert "profiled" in result.output
        assert "completed" in result.output
        assert "10" in result.output
        assert "30" in result.output
        assert "Improvement" in result.output

    def test_minimize_metric(self, tmp_path: Path):
        from typer.testing import CliRunner

        from perflab.cli import app

        store = RunStore(tmp_path)
        ra = self._make_run(store, best_value=100.0, metric_name="latency_ms", metric_mode="minimize")
        rb = self._make_run(store, best_value=50.0, metric_name="latency_ms", metric_mode="minimize")

        runner = CliRunner()
        result = runner.invoke(app, ["compare", ra.run_id, rb.run_id, "--out-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Speedup" in result.output

    def test_minimize_regression(self, tmp_path: Path):
        from typer.testing import CliRunner

        from perflab.cli import app

        store = RunStore(tmp_path)
        ra = self._make_run(store, best_value=50.0, metric_name="latency_ms", metric_mode="minimize")
        rb = self._make_run(store, best_value=100.0, metric_name="latency_ms", metric_mode="minimize")

        runner = CliRunner()
        result = runner.invoke(app, ["compare", ra.run_id, rb.run_id, "--out-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Slowdown" in result.output

    def test_not_found(self, tmp_path: Path):
        from typer.testing import CliRunner

        from perflab.cli import app

        store = RunStore(tmp_path)
        ra = self._make_run(store, best_value=10.0)

        runner = CliRunner()
        result = runner.invoke(app, ["compare", ra.run_id, "nonexistent", "--out-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_bottleneck_diff_display(self, tmp_path: Path):
        from typer.testing import CliRunner

        from perflab.cli import app

        store = RunStore(tmp_path)
        ra = self._make_run(store, best_value=10.0, bottlenecks=["mem_bound", "no_simd"])
        rb = self._make_run(store, best_value=20.0, bottlenecks=["no_simd", "cache_miss"])

        runner = CliRunner()
        result = runner.invoke(app, ["compare", ra.run_id, rb.run_id, "--out-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Resolved bottlenecks" in result.output
        assert "mem_bound" in result.output
        assert "New bottlenecks" in result.output
        assert "cache_miss" in result.output

    def test_no_values(self, tmp_path: Path):
        from typer.testing import CliRunner

        from perflab.cli import app

        store = RunStore(tmp_path)
        ra = store.new_run("t")
        rb = store.new_run("t")

        runner = CliRunner()
        result = runner.invoke(app, ["compare", ra.run_id, rb.run_id, "--out-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "N/A" in result.output
