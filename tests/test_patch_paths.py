"""Tests for workspace path-containment checks in perflab.optimizers.patch.

Covers the Fix 1 remediation: string-prefix containment was replaced with
Path.is_relative_to(), which closes the sibling-directory escape
("../<workspace>-evil/x.py") and adds explicit rejection of absolute paths
and traversal tricks that dodge allowed_paths matching.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from perflab.optimizers.patch import SearchReplaceBlock, validate_patch


def _block(file_path: str) -> SearchReplaceBlock:
    return SearchReplaceBlock(file_path=file_path, search="x", replace="y")


class TestWorkspaceEscape:
    def test_sibling_directory_with_shared_prefix_is_rejected(self, tmp_path):
        workspace = tmp_path / "proj"
        workspace.mkdir()
        evil = tmp_path / "proj-evil"
        evil.mkdir()
        (evil / "x.py").write_text("x")

        errors = validate_patch(
            [_block("../proj-evil/x.py")], allowed_paths=[], workspace=workspace
        )
        assert len(errors) == 1
        assert "escapes workspace" in errors[0]

    def test_absolute_path_is_rejected(self, tmp_path):
        workspace = tmp_path / "proj"
        workspace.mkdir()

        errors = validate_patch(
            [_block("/etc/hosts")], allowed_paths=[], workspace=workspace
        )
        assert len(errors) == 1
        assert "relative to the workspace" in errors[0]

    def test_traversal_to_protected_file_is_rejected(self, tmp_path):
        workspace = tmp_path / "proj"
        (workspace / "src").mkdir(parents=True)
        (workspace / "bench.py").write_text("x")

        errors = validate_patch(
            [_block("src/../bench.py")], allowed_paths=[], workspace=workspace
        )
        assert len(errors) == 1
        assert "protected file" in errors[0]

    def test_traversal_within_workspace_matches_allowed_paths(self, tmp_path):
        workspace = tmp_path / "proj"
        (workspace / "src").mkdir(parents=True)
        (workspace / "src" / "kernel.cu").write_text("__global__ void k() {}")

        errors = validate_patch(
            [_block("./src/kernel.cu")],
            allowed_paths=["src/*.cu"],
            workspace=workspace,
        )
        # The search text "x" won't match the file contents, but it must get
        # past path validation (no "escapes" / "not in allowed_paths" errors).
        assert not any("escapes" in e or "allowed_paths" in e for e in errors)

    def test_dotdot_variant_still_blocked_by_allowed_paths(self, tmp_path):
        workspace = tmp_path / "proj"
        (workspace / "src").mkdir(parents=True)
        (workspace / "other").mkdir(parents=True)
        (workspace / "other" / "kernel.cu").write_text("x")

        # This stays inside the workspace, so it isn't an escape -- but it must
        # still be checked against allowed_paths using the resolved relative
        # path, not the literal traversal string.
        errors = validate_patch(
            [_block("src/../other/kernel.cu")],
            allowed_paths=["src/*.cu"],
            workspace=workspace,
        )
        assert any("not in allowed_paths" in e for e in errors)

    @pytest.mark.skipif(
        not hasattr(Path, "symlink_to"), reason="platform lacks symlink support"
    )
    def test_symlink_escaping_workspace_is_rejected(self, tmp_path):
        workspace = tmp_path / "proj"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text("x")

        link = workspace / "link_dir"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks not supported in this environment")

        errors = validate_patch(
            [_block("link_dir/secret.py")], allowed_paths=[], workspace=workspace
        )
        assert len(errors) == 1
        assert "escapes workspace" in errors[0]
