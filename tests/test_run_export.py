"""Tests for `perflab export` -- bundling a run for the trip off a remote box.

The load-bearing behaviour is the slim filter: it must drop the heavyweight
raw captures and must NOT drop the parsed summaries or the browser-openable
traces, because those are the entire point of exporting at all.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from perflab.cli import app
from perflab.memory.run_export import export_run, human_bytes, is_slim_excluded

runner = CliRunner()

# One file per artifact class the profilers actually emit, so the slim filter
# is asserted against real names rather than invented ones.
_HEAVY = {
    "artifacts/ncu_report.ncu-rep": "ncu",
    "artifacts/nsys_report.nsys-rep": "nsys",
    "artifacts/nsys_report.sqlite": "sqlite",
    "artifacts/perf.data": "perf",
    "artifacts/perf_lock.data": "lock",
    "artifacts/perf_c2c.data": "c2c",
    "artifacts/perf_sched.data": "sched",
    "artifacts/perf_script.txt": "script",
    "artifacts/jax_trace/plugins/x.xplane.pb": "xplane",
}
_KEPT = {
    "meta.json": '{"run_id": "20260810-120000-abcd1234"}',
    "dashboard.html": "<h1>dash</h1>",
    "report.md": "# report",
    "bench.json": "{}",
    "system_info.json": "{}",
    "perfetto_trace.json": "{}",
    "snapshots/optimized.zip": "zip",
    "artifacts/ncu_summary.json": "{}",
    "artifacts/nsys_summary.json": "{}",
    "artifacts/linux_perf_summary.json": "{}",
    "artifacts/pyspy_speedscope.json": "{}",
    "artifacts/torch_trace.json": "{}",
    "artifacts/memray_flamegraph.html": "<html>",
    "artifacts/perf_annotate.txt": "annotate",
    "artifacts/xla_hlo_dump/module.txt": "hlo",
}


def _make_run(tmp_path: Path, run_id: str = "20260810-120000-abcd1234") -> Path:
    run_dir = tmp_path / "out" / "runs" / run_id
    for rel, content in {**_KEPT, **_HEAVY}.items():
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    # Instruments writes a .trace as a directory, not a file -- it must be
    # pruned whole rather than walked into.
    bundle = run_dir / "artifacts" / "metal_trace.trace"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "run1.core").write_text("core", encoding="utf-8")
    return run_dir


def _members(archive: Path) -> set[str]:
    with tarfile.open(archive) as tar:
        return {m.name for m in tar.getmembers()}


class TestSlimFilter:
    @pytest.mark.parametrize("name", [
        "ncu_report.ncu-rep", "nsys_report.nsys-rep", "nsys_report.sqlite",
        "perf.data", "perf_lock.data", "perf_sched.data", "perf_script.txt",
        "metal_trace.trace", "x.xplane.pb",
    ])
    def test_excludes_raw_captures(self, name: str) -> None:
        assert is_slim_excluded(name)

    @pytest.mark.parametrize("name", [
        "dashboard.html", "ncu_summary.json", "pyspy_speedscope.json",
        "torch_trace.json", "perfetto_trace.json", "memray_flamegraph.html",
        "perf_annotate.txt", "optimized.zip", "module.txt", "meta.json",
    ])
    def test_keeps_everything_readable_elsewhere(self, name: str) -> None:
        assert not is_slim_excluded(name)


class TestExportRun:
    def test_full_export_includes_everything(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        result = export_run(run_dir, tmp_path / "full.tgz")
        names = _members(result.path)
        for rel in {**_KEPT, **_HEAVY}:
            assert f"{run_dir.name}/{rel}" in names
        assert result.skipped_count == 0

    def test_slim_drops_heavy_keeps_the_rest(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        result = export_run(run_dir, tmp_path / "slim.tgz", slim=True)
        names = _members(result.path)
        for rel in _KEPT:
            assert f"{run_dir.name}/{rel}" in names, f"slim dropped {rel}"
        for rel in _HEAVY:
            assert f"{run_dir.name}/{rel}" not in names, f"slim kept {rel}"

    def test_slim_prunes_trace_bundle_directory(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        result = export_run(run_dir, tmp_path / "slim.tgz", slim=True)
        assert not any(".trace/" in n for n in _members(result.path))
        # The file inside the pruned bundle is still counted as dropped.
        assert result.skipped_count == len(_HEAVY) + 1

    def test_archive_rooted_at_run_id(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        result = export_run(run_dir, tmp_path / "out.tgz")
        assert all(n.startswith(f"{run_dir.name}/") for n in _members(result.path))

    def test_rejects_non_run_directory(self, tmp_path: Path) -> None:
        (tmp_path / "nope").mkdir()
        with pytest.raises(ValueError, match="no meta.json"):
            export_run(tmp_path / "nope", tmp_path / "x.tgz")

    def test_rejects_destination_inside_run_dir(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        with pytest.raises(ValueError, match="inside the run directory"):
            export_run(run_dir, run_dir / "self.tgz")

    def test_counts_are_consistent(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        result = export_run(run_dir, tmp_path / "slim.tgz", slim=True)
        assert result.file_count == len(_KEPT)
        assert result.source_bytes > 0
        assert result.archive_bytes > 0


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch working directory -- the default archive path is cwd-relative."""
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return workdir


class TestExportCli:
    def test_defaults_to_newest_run(self, tmp_path: Path, cwd: Path) -> None:
        _make_run(tmp_path, "20260810-090000-aaaaaaaa")
        newest = _make_run(tmp_path, "20260810-120000-bbbbbbbb")
        result = runner.invoke(app, ["export", "--out-dir", str(tmp_path / "out")])
        assert result.exit_code == 0, result.output
        assert (cwd / f"{newest.name}.tgz").is_file()

    def test_accepts_bare_run_id(self, tmp_path: Path, cwd: Path) -> None:
        _make_run(tmp_path, "20260810-120000-bbbbbbbb")
        older = _make_run(tmp_path, "20260810-090000-aaaaaaaa")
        result = runner.invoke(
            app, ["export", older.name, "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, result.output
        assert (cwd / f"{older.name}.tgz").is_file()

    def test_slim_reports_what_it_dropped(self, tmp_path: Path, cwd: Path) -> None:
        _make_run(tmp_path)
        result = runner.invoke(
            app, ["export", "--slim", "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, result.output
        assert "Slim dropped" in result.output

    def test_refuses_to_overwrite_without_force(self, tmp_path: Path, cwd: Path) -> None:
        run_dir = _make_run(tmp_path)
        archive = cwd / f"{run_dir.name}.tgz"
        archive.write_text("existing")

        result = runner.invoke(app, ["export", "--out-dir", str(tmp_path / "out")])
        assert result.exit_code == 1
        assert "already exists" in result.output
        assert archive.read_text() == "existing"

        forced = runner.invoke(
            app, ["export", "--force", "--out-dir", str(tmp_path / "out")]
        )
        assert forced.exit_code == 0, forced.output
        assert archive.read_bytes()[:2] == b"\x1f\x8b"  # gzip magic

    def test_dest_option_controls_path(self, tmp_path: Path, cwd: Path) -> None:
        _make_run(tmp_path)
        result = runner.invoke(
            app, ["export", "-o", "bundle.tgz", "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, result.output
        assert (cwd / "bundle.tgz").is_file()

    def test_unknown_run_id_exits_nonzero(self, tmp_path: Path) -> None:
        _make_run(tmp_path)
        result = runner.invoke(
            app, ["export", "nope-not-a-run", "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code != 0


class TestHumanBytes:
    @pytest.mark.parametrize("n,expected", [
        (96, "96 B"),
        (2048, "2.0 KB"),
        (5 * 1024 * 1024, "5.0 MB"),
        (3 * 1024 * 1024 * 1024, "3.0 GB"),
    ])
    def test_formats(self, n: int, expected: str) -> None:
        assert human_bytes(n) == expected
