"""Tests for `perflab tasks list` / `perflab tasks copy` and the underlying
importlib.resources-based lookup of the bundled demo tasks.

These never run a benchmark or touch an LLM -- they only exercise the CLI's
listing/copying logic and its resolution of perflab/demo_tasks/ (the former
repo-root tasks/ directory, now packaged as data and re-exposed at the repo
root via a symlink -- see pyproject.toml [tool.setuptools.package-data]).
"""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from perflab.cli import _demo_tasks_root, _iter_bundled_tasks, app

runner = CliRunner()


class TestDemoTasksRootResolution:
    """The importlib.resources lookup must find the real directory on disk
    in this (editable) install, exactly as it would find packaged data
    inside an installed wheel."""

    def test_root_resolves_to_a_directory(self) -> None:
        root = _demo_tasks_root()
        assert root.is_dir()

    def test_root_contains_matmul_python(self) -> None:
        root = _demo_tasks_root()
        assert (root / "matmul" / "python" / "task.yaml").is_file()
        assert (root / "matmul" / "python" / "bench.py").is_file()

    def test_iter_bundled_tasks_includes_matmul_python(self) -> None:
        names = [name for name, _ in _iter_bundled_tasks()]
        assert "matmul/python" in names
        # Sorted, and every entry actually has a task.yaml on disk.
        assert names == sorted(names)

    def test_iter_bundled_tasks_finds_all_eighteen(self) -> None:
        # 17 real demo tasks + the _sample template.
        names = [name for name, _ in _iter_bundled_tasks()]
        assert len(names) == 18


class TestTasksList:
    def test_exits_zero(self) -> None:
        result = runner.invoke(app, ["tasks", "list"])
        assert result.exit_code == 0

    def test_contains_matmul_python(self) -> None:
        result = runner.invoke(app, ["tasks", "list"])
        assert "matmul/python" in result.output

    def test_contains_a_description_derived_from_task_yaml(self) -> None:
        result = runner.invoke(app, ["tasks", "list"])
        # matmul/python's task.yaml has name: "matmul_python_pure"
        assert "matmul_python_pure" in result.output

    def test_hints_at_copy_command(self) -> None:
        result = runner.invoke(app, ["tasks", "list"])
        assert "perflab tasks copy" in result.output


class TestTasksCopy:
    def test_copy_lands_task_files(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["tasks", "copy", "matmul/python", str(tmp_path)])
        assert result.exit_code == 0, result.output

        copied = tmp_path / "matmul" / "python"
        assert (copied / "task.yaml").is_file()
        assert (copied / "bench.py").is_file()
        assert (copied / "matmul.py").is_file()
        assert (copied / "tests.py").is_file()

    def test_copy_excludes_pycache_and_out(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["tasks", "copy", "matmul/python", str(tmp_path)])
        assert result.exit_code == 0, result.output

        copied = tmp_path / "matmul" / "python"
        assert not (copied / "__pycache__").exists()
        assert not (copied / "out").exists()

    def test_copy_prints_agent_hint(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["tasks", "copy", "matmul/python", str(tmp_path)])
        assert result.exit_code == 0, result.output
        expected_task_yaml = tmp_path / "matmul" / "python" / "task.yaml"
        assert "perflab agent" in result.output
        assert str(expected_task_yaml) in result.output

    def test_copy_default_dest_is_cwd(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["tasks", "copy", "matmul/python"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "matmul" / "python" / "task.yaml").is_file()

    def test_copy_refuses_to_overwrite_existing_target(self, tmp_path: Path) -> None:
        first = runner.invoke(app, ["tasks", "copy", "matmul/python", str(tmp_path)])
        assert first.exit_code == 0, first.output

        second = runner.invoke(app, ["tasks", "copy", "matmul/python", str(tmp_path)])
        assert second.exit_code != 0
        assert "already exists" in second.output
        # Original copy must be left untouched, not partially clobbered.
        assert (tmp_path / "matmul" / "python" / "task.yaml").is_file()

    def test_copy_unknown_name_errors_with_valid_list(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["tasks", "copy", "not/a/real/task", str(tmp_path)])
        assert result.exit_code != 0
        assert "unknown task" in result.output
        # Helpful error should list at least one real task name.
        assert "matmul/python" in result.output
        assert not (tmp_path / "not").exists()
