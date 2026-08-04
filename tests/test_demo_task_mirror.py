"""`tasks/` must stay identical to `perflab/demo_tasks/`.

`perflab/demo_tasks/` is the source of truth — the wheel ships it as package
data, and package data has to live inside the package. `tasks/` is the short
path every doc, README example and CI invocation uses. It was a symlink, which
kept them identical for free but made every bench.py look like a duplicate
module to anything walking the tree (`mypy .` fails outright on it).

Two real copies is the deliberate trade. Duplication without enforcement is
just drift on a delay, so this test is the enforcement — and being a test
rather than a bespoke CI job means it fails locally before it ever reaches CI.

Fix drift with:  python scripts/sync_demo_tasks.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "sync_demo_tasks.py"

sys.path.insert(0, str(SCRIPT.parent))
import sync_demo_tasks as sync_mod  # noqa: E402


class TestMirrorIsInSync:
    def test_no_drift(self):
        missing, extra, differing = sync_mod.diff()
        assert not (missing or extra or differing), (
            "tasks/ has drifted from perflab/demo_tasks/:\n"
            + sync_mod.describe(missing, extra, differing)
            + "\n\nFix with: python scripts/sync_demo_tasks.py"
        )

    def test_mirror_is_a_real_directory_not_a_symlink(self):
        """The symlink is what broke `mypy .`; don't let it come back."""
        assert sync_mod.MIRROR.is_dir()
        assert not sync_mod.MIRROR.is_symlink()

    def test_source_of_truth_is_non_empty(self):
        """Guards against the check passing vacuously if a path moves."""
        assert len(sync_mod.collect(sync_mod.SOURCE)) > 50


class TestDriftIsActuallyDetected:
    """A checker that cannot fail is not a checker."""

    def test_content_change_is_caught(self, tmp_path, monkeypatch):
        src, dst = tmp_path / "src", tmp_path / "dst"
        (src / "t").mkdir(parents=True)
        (dst / "t").mkdir(parents=True)
        (src / "t" / "a.py").write_text("x = 1\n")
        (dst / "t" / "a.py").write_text("x = 2\n")
        monkeypatch.setattr(sync_mod, "SOURCE", src)
        monkeypatch.setattr(sync_mod, "MIRROR", dst)
        _missing, _extra, differing = sync_mod.diff()
        assert differing == [Path("t/a.py")]

    def test_same_size_edit_is_caught(self, tmp_path, monkeypatch):
        """Compared by content, not size+mtime — a same-length edit is the
        drift most likely to slip through a shallow comparison."""
        src, dst = tmp_path / "src", tmp_path / "dst"
        src.mkdir(), dst.mkdir()
        (src / "a.yaml").write_text("M: 256\n")
        (dst / "a.yaml").write_text("M: 512\n")
        monkeypatch.setattr(sync_mod, "SOURCE", src)
        monkeypatch.setattr(sync_mod, "MIRROR", dst)
        assert sync_mod.diff()[2] == [Path("a.yaml")]

    def test_missing_and_extra_files_are_caught(self, tmp_path, monkeypatch):
        src, dst = tmp_path / "src", tmp_path / "dst"
        src.mkdir(), dst.mkdir()
        (src / "only_src.py").write_text("a\n")
        (dst / "only_dst.py").write_text("b\n")
        monkeypatch.setattr(sync_mod, "SOURCE", src)
        monkeypatch.setattr(sync_mod, "MIRROR", dst)
        missing, extra, _differing = sync_mod.diff()
        assert missing == [Path("only_src.py")]
        assert extra == [Path("only_dst.py")]

    def test_generated_output_is_ignored(self, tmp_path, monkeypatch):
        """Each tree writes its own out/ and __pycache__; copying those across
        would be wrong, and flagging them would make the check cry wolf."""
        src, dst = tmp_path / "src", tmp_path / "dst"
        (src / "t" / "out").mkdir(parents=True)
        (dst / "t" / "__pycache__").mkdir(parents=True)
        (src / "t" / "out" / "bench.json").write_text("{}")
        (dst / "t" / "__pycache__" / "m.py").write_text("cached")
        monkeypatch.setattr(sync_mod, "SOURCE", src)
        monkeypatch.setattr(sync_mod, "MIRROR", dst)
        assert sync_mod.diff() == ([], [], [])


class TestSyncRepairsDrift:
    def test_sync_makes_them_match(self, tmp_path, monkeypatch):
        src, dst = tmp_path / "src", tmp_path / "dst"
        src.mkdir(), dst.mkdir()
        (src / "keep.py").write_text("new\n")
        (dst / "keep.py").write_text("old\n")
        (dst / "stale.py").write_text("remove me\n")
        monkeypatch.setattr(sync_mod, "SOURCE", src)
        monkeypatch.setattr(sync_mod, "MIRROR", dst)
        sync_mod.sync()
        assert sync_mod.diff() == ([], [], [])
        assert (dst / "keep.py").read_text() == "new\n"
        assert not (dst / "stale.py").exists()


def test_check_mode_exit_codes():
    """The command the failure message tells you to run must behave."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "in sync" in result.stdout


@pytest.mark.parametrize("path", ["matmul/python/task.yaml", "matmul/cpp/matmul.cpp"])
def test_documented_short_paths_resolve(path):
    """README and CI invoke tasks/... — those paths must exist for real."""
    assert (REPO_ROOT / "tasks" / path).is_file()
