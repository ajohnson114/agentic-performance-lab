"""Tests for perflab.analyzers.build_overrides."""
from __future__ import annotations

import json
from pathlib import Path

from perflab.analyzers.build_overrides import (
    ALLOWED_FLAGS,
    apply_build_overrides,
    load_build_overrides,
    load_build_overrides_with_feedback,
)


class TestLoadBuildOverrides:
    def test_no_file(self, tmp_path: Path):
        assert load_build_overrides(tmp_path) == []

    def test_valid_flags(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": ["-O3", "-march=native"]}),
            encoding="utf-8",
        )
        result = load_build_overrides(tmp_path)
        assert "-O3" in result
        assert "-march=native" in result

    def test_filters_disallowed_flags(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": ["-O3", "-rm -rf /", "--evil"]}),
            encoding="utf-8",
        )
        result = load_build_overrides(tmp_path)
        assert result == ["-O3"]

    def test_invalid_json(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text("not json", encoding="utf-8")
        assert load_build_overrides(tmp_path) == []

    def test_empty_flags(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": []}),
            encoding="utf-8",
        )
        assert load_build_overrides(tmp_path) == []

    def test_non_string_flags_filtered(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": ["-O3", 42, None, "-flto"]}),
            encoding="utf-8",
        )
        result = load_build_overrides(tmp_path)
        assert result == ["-O3", "-flto"]


class TestApplyBuildOverrides:
    def test_appends_flags(self):
        cmd = "g++ -O2 -o bin main.cpp"
        result = apply_build_overrides(cmd, ["-march=native", "-flto"])
        assert "-march=native" in result
        assert "-flto" in result

    def test_no_duplicates(self):
        cmd = "g++ -O2 -march=native -o bin main.cpp"
        result = apply_build_overrides(cmd, ["-march=native"])
        assert result == cmd  # No change

    def test_empty_overrides(self):
        cmd = "g++ -O2 -o bin main.cpp"
        assert apply_build_overrides(cmd, []) == cmd

    def test_mixed_new_and_existing(self):
        cmd = "g++ -O2 -o bin main.cpp"
        result = apply_build_overrides(cmd, ["-O2", "-flto"])
        assert result.count("-O2") == 1
        assert "-flto" in result


class TestAllowlist:
    def test_common_flags_present(self):
        assert "-O3" in ALLOWED_FLAGS
        assert "-march=native" in ALLOWED_FLAGS
        assert "-fopenmp" in ALLOWED_FLAGS

    def test_dangerous_flags_absent(self):
        assert "-rm" not in ALLOWED_FLAGS
        assert "--evil" not in ALLOWED_FLAGS


class TestSyntaxValidation:
    def test_flag_without_leading_dash_rejected(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": ["O3", "march=native"]}),
            encoding="utf-8",
        )
        result = load_build_overrides_with_feedback(tmp_path)
        assert result.accepted == []
        assert len(result.rejected) == 2
        assert all("invalid syntax" in r.reason for r in result.rejected)

    def test_non_string_flag_rejected_with_reason(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": [42, None]}),
            encoding="utf-8",
        )
        result = load_build_overrides_with_feedback(tmp_path)
        assert result.accepted == []
        assert len(result.rejected) == 2
        assert all("not a string" in r.reason for r in result.rejected)

    def test_valid_flags_still_pass(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": ["-O3", "-flto", "-march=native"]}),
            encoding="utf-8",
        )
        result = load_build_overrides_with_feedback(tmp_path)
        assert result.accepted == ["-O3", "-flto", "-march=native"]
        assert result.rejected == []

    def test_not_in_allowlist_rejected_with_reason(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": ["-O3", "--some-unknown-flag"]}),
            encoding="utf-8",
        )
        result = load_build_overrides_with_feedback(tmp_path)
        assert result.accepted == ["-O3"]
        assert len(result.rejected) == 1
        assert result.rejected[0].flag == "--some-unknown-flag"
        assert "not in allowlist" in result.rejected[0].reason

    def test_fast_math_rejected_with_reason_when_disallowed(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": ["-ffast-math"]}),
            encoding="utf-8",
        )
        result = load_build_overrides_with_feedback(tmp_path)
        assert result.accepted == []
        assert len(result.rejected) == 1
        assert "fast-math" in result.rejected[0].reason


class TestConflictDetection:
    def test_conflicting_optimization_flags_warn(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": ["-O2", "-O3"]}),
            encoding="utf-8",
        )
        result = load_build_overrides_with_feedback(tmp_path)
        assert "-O2" in result.accepted
        assert "-O3" in result.accepted
        assert len(result.warnings) == 1
        assert "-O2" in result.warnings[0]
        assert "-O3" in result.warnings[0]
        assert "Conflicting" in result.warnings[0]

    def test_single_optimization_flag_no_warning(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": ["-O3"]}),
            encoding="utf-8",
        )
        result = load_build_overrides_with_feedback(tmp_path)
        assert result.accepted == ["-O3"]
        assert result.warnings == []

    def test_feedback_includes_reason_for_each_rejection(self, tmp_path: Path):
        (tmp_path / "build_overrides.json").write_text(
            json.dumps({"flags": ["badflag", "-ffast-math", "--unknown", "-O3"]}),
            encoding="utf-8",
        )
        result = load_build_overrides_with_feedback(tmp_path)
        assert result.accepted == ["-O3"]
        assert len(result.rejected) == 3
        reasons = {r.flag: r.reason for r in result.rejected}
        assert "invalid syntax" in reasons["badflag"]
        assert "fast-math" in reasons["-ffast-math"]
        assert "not in allowlist" in reasons["--unknown"]
