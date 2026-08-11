"""Tests for `perflab diff` -- before/after source across a run's snapshots.

The subtle one is iteration ordering: lexicographic sorting picks iter10 <
iter2, which would silently diff against the wrong "last accepted" snapshot on
any run that accepted more than nine candidates.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from perflab.cli import app
from perflab.memory.run_diff import (
    diff_snapshots,
    list_labels,
    read_snapshot,
    resolve_labels,
)

runner = CliRunner()

_BASELINE = "def matmul(a, b):\n    return naive(a, b)\n"
_OPTIMIZED = "def matmul(a, b):\n    return blocked(a, b)\n"


def _write_snapshot(run_dir: Path, label: str, sources: dict[str, str]) -> None:
    snap_dir = run_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(snap_dir / f"{label}.zip", "w") as zf:
        for rel, content in sources.items():
            zf.writestr(rel, content)


def _make_run(
    tmp_path: Path,
    run_id: str = "20260810-120000-abcd1234",
    *,
    iters: tuple[int, ...] = (3,),
) -> Path:
    run_dir = tmp_path / "out" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(f'{{"run_id": "{run_id}"}}', encoding="utf-8")
    _write_snapshot(run_dir, "baseline", {"matmul.py": _BASELINE})
    for n in iters:
        _write_snapshot(run_dir, f"iter{n}", {"matmul.py": f"# iter{n}\n{_OPTIMIZED}"})
    return run_dir


class TestLabelOrdering:
    def test_baseline_first_then_iterations_numerically(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path, iters=(2, 10, 1))
        assert list_labels(run_dir) == ["baseline", "iter1", "iter2", "iter10"]

    def test_last_accepted_is_highest_iteration_not_lexicographic(
        self, tmp_path: Path
    ) -> None:
        run_dir = _make_run(tmp_path, iters=(2, 10))
        assert resolve_labels(run_dir) == ("baseline", "iter10")

    def test_unknown_labels_sort_last_without_crashing(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path, iters=(1,))
        _write_snapshot(run_dir, "manual", {"matmul.py": _OPTIMIZED})
        assert list_labels(run_dir) == ["baseline", "iter1", "manual"]


class TestResolveLabels:
    def test_explicit_endpoints_respected(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path, iters=(1, 2))
        assert resolve_labels(run_dir, "iter1", "iter2") == ("iter1", "iter2")

    def test_to_is_none_when_nothing_was_accepted(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path, iters=())
        assert resolve_labels(run_dir) == ("baseline", None)

    def test_rejects_unknown_label_and_lists_available(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path, iters=(3,))
        with pytest.raises(ValueError, match="Available: baseline, iter3"):
            resolve_labels(run_dir, "iter99", None)

    def test_rejects_run_without_snapshots(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "out" / "runs" / "empty"
        run_dir.mkdir(parents=True)
        with pytest.raises(ValueError, match="No snapshots"):
            resolve_labels(run_dir)


class TestDiffSnapshots:
    def test_reports_the_actual_change(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        result = diff_snapshots(run_dir, "baseline", "iter3")
        assert "-    return naive(a, b)" in result.text
        assert "+    return blocked(a, b)" in result.text
        assert result.files_changed == 1

    def test_counts_exclude_file_headers(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        result = diff_snapshots(run_dir, "baseline", "iter3")
        # One line swapped plus the "# iter3" line added.
        assert result.insertions == 2
        assert result.deletions == 1

    def test_identical_snapshots_produce_nothing(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        _write_snapshot(run_dir, "iter4", {"matmul.py": _BASELINE})
        result = diff_snapshots(run_dir, "baseline", "iter4")
        assert result.files_changed == 0
        assert result.text == ""

    def test_added_file_shown_against_dev_null(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        _write_snapshot(
            run_dir, "iter5", {"matmul.py": _BASELINE, "kernel.py": "x = 1\n"}
        )
        result = diff_snapshots(run_dir, "baseline", "iter5")
        assert "--- /dev/null" in result.text
        assert "+++ iter5/kernel.py" in result.text

    def test_removed_file_shown_against_dev_null(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        _write_snapshot(run_dir, "iter6", {})
        result = diff_snapshots(run_dir, "baseline", "iter6")
        assert "+++ /dev/null" in result.text

    def test_missing_trailing_newline_does_not_merge_files(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        _write_snapshot(run_dir, "iter7", {"matmul.py": "no trailing newline"})
        result = diff_snapshots(run_dir, "baseline", "iter7")
        assert result.text.endswith("\n")
        assert all(line for line in result.text.splitlines()[:1])

    def test_read_snapshot_rejects_unknown_label(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path)
        with pytest.raises(ValueError, match="No snapshot 'nope'"):
            read_snapshot(run_dir, "nope")


class TestDiffCli:
    def test_defaults_to_baseline_vs_last_accepted(self, tmp_path: Path) -> None:
        _make_run(tmp_path, iters=(2, 10))
        result = runner.invoke(app, ["diff", "--out-dir", str(tmp_path / "out")])
        assert result.exit_code == 0, result.output
        assert "baseline -> iter10" in result.output
        assert "+    return blocked(a, b)" in result.output
        assert "1 file(s) changed, 2 insertion(s), 1 deletion(s)" in result.output

    def test_defaults_to_newest_run(self, tmp_path: Path) -> None:
        _make_run(tmp_path, "20260810-090000-aaaaaaaa", iters=(1,))
        _make_run(tmp_path, "20260810-120000-bbbbbbbb", iters=(7,))
        result = runner.invoke(app, ["diff", "--out-dir", str(tmp_path / "out")])
        assert result.exit_code == 0, result.output
        assert "20260810-120000-bbbbbbbb: baseline -> iter7" in result.output

    def test_accepts_bare_run_id(self, tmp_path: Path) -> None:
        _make_run(tmp_path, "20260810-120000-bbbbbbbb", iters=(7,))
        older = _make_run(tmp_path, "20260810-090000-aaaaaaaa", iters=(1,))
        result = runner.invoke(
            app, ["diff", older.name, "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, result.output
        assert f"{older.name}: baseline -> iter1" in result.output

    def test_explicit_endpoints(self, tmp_path: Path) -> None:
        _make_run(tmp_path, iters=(1, 2))
        result = runner.invoke(
            app,
            ["diff", "--from", "iter1", "--to", "iter2", "--out-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 0, result.output
        assert "iter1 -> iter2" in result.output

    def test_list_shows_snapshots(self, tmp_path: Path) -> None:
        _make_run(tmp_path, iters=(1, 10))
        result = runner.invoke(
            app, ["diff", "--list", "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, result.output
        assert result.output.index("iter1") < result.output.index("iter10")

    def test_run_with_no_accepts_explains_itself(self, tmp_path: Path) -> None:
        _make_run(tmp_path, iters=())
        result = runner.invoke(app, ["diff", "--out-dir", str(tmp_path / "out")])
        assert result.exit_code == 0, result.output
        assert "no iteration was accepted" in result.output

    def test_unknown_label_exits_nonzero(self, tmp_path: Path) -> None:
        _make_run(tmp_path, iters=(1,))
        result = runner.invoke(
            app, ["diff", "--to", "iter99", "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        assert "Available: baseline, iter1" in result.output

    def test_run_without_snapshots_exits_nonzero(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "out" / "runs" / "20260810-120000-abcd1234"
        run_dir.mkdir(parents=True)
        (run_dir / "meta.json").write_text("{}", encoding="utf-8")
        result = runner.invoke(app, ["diff", "--out-dir", str(tmp_path / "out")])
        assert result.exit_code == 1
        assert "No snapshots" in result.output
