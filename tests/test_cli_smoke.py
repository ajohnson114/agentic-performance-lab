"""CLI-level smoke tests for the typer app in perflab/cli.py.

These never run benchmarks, probe hardware, or touch an LLM: they only
exercise argument parsing, help output, and cheap offline commands plus a
few negative paths.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from perflab.cli import app

runner = CliRunner()


class TestTopLevelHelp:
    def test_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_lists_expected_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        expected = [
            "profile",
            "optimize",
            "agent",
            "doctor",
            "peaks",
            "ci-check",
            "replay",
            "list-runs",
            "compare",
            "thresholds",
            "init",
            "show-task",
        ]
        for cmd in expected:
            assert cmd in result.output, f"command {cmd!r} missing from --help output"


# Every registered subcommand's --help must parse and exit 0. Commands that
# probe hardware (doctor, peaks) or run benchmarks (profile, optimize, agent,
# ci-check) are only exercised via --help here.
_SUBCOMMANDS = [
    "profile",
    "optimize",
    "agent",
    "doctor",
    "peaks",
    "init",
    "init-config",
    "ci-check",
    "replay",
    "list-runs",
    "compare",
    "thresholds",
    "show-config",
    "show-config-template",
    "show-task",
    "show-task-schema",
    "show-tuning-schema",
]


class TestSubcommandHelp:
    @pytest.mark.parametrize("cmd", _SUBCOMMANDS)
    def test_subcommand_help_exits_zero(self, cmd: str) -> None:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed:\n{result.output}"


class TestCheapOfflineCommands:
    def test_thresholds_prints_defaults(self) -> None:
        result = runner.invoke(app, ["thresholds"])
        assert result.exit_code == 0
        assert "Analysis Thresholds" in result.output
        # A known threshold field from AnalysisThresholds should be listed
        assert "ncu_tc_util_low" in result.output

    def test_thresholds_section_filter(self) -> None:
        result = runner.invoke(app, ["thresholds", "--section", "ncu"])
        assert result.exit_code == 0
        assert "NCU" in result.output

    def test_show_task_schema(self) -> None:
        result = runner.invoke(app, ["show-task-schema"])
        assert result.exit_code == 0
        assert "task.yaml Schema Reference" in result.output
        assert "program_type" in result.output

    def test_show_tuning_schema(self) -> None:
        result = runner.invoke(app, ["show-tuning-schema"])
        assert result.exit_code == 0
        assert "tuning.yaml Schema Reference" in result.output

    def test_show_config_template(self) -> None:
        result = runner.invoke(app, ["show-config-template"])
        assert result.exit_code == 0
        assert result.output.strip()

    def test_list_runs_empty_out_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["list-runs", "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 0
        assert "No runs found." in result.output

    def test_show_task_renders_sample_task(self, sample_task_yaml: Path) -> None:
        result = runner.invoke(app, ["show-task", str(sample_task_yaml)])
        assert result.exit_code == 0
        assert "Task: test-task" in result.output
        assert "throughput.median" in result.output
        assert "Correctness:" in result.output


class TestNegativeCases:
    def test_profile_nonexistent_task_yaml(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist" / "task.yaml"
        result = runner.invoke(app, ["profile", str(missing)])
        assert result.exit_code == 2
        assert "task file not found" in result.output

    def test_ci_check_nonexistent_task_yaml(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.yaml"
        result = runner.invoke(app, ["ci-check", str(missing)])
        assert result.exit_code == 2
        assert "task file not found" in result.output

    def test_replay_nonexistent_run_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["replay", str(tmp_path / "no-such-run")])
        assert result.exit_code == 1
        assert "Run directory not found" in result.output

    def test_compare_unknown_runs(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["compare", "run-a", "run-b", "--out-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_profile_missing_argument_is_usage_error(self) -> None:
        result = runner.invoke(app, ["profile"])
        assert result.exit_code == 2
