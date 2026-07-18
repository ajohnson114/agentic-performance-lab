"""Tests for search-text uniqueness enforcement and fuzzy-correction surfacing
in perflab.optimizers.patch.

Covers two patch-safety hardenings:
  1. A SEARCH block matching more than one location is rejected (both for
     exact matches and for fuzzy-corrected matches) with an error telling the
     LLM to include more surrounding context.
  2. Fuzzy auto-corrections are no longer silent: validate_patch() appends a
     notice describing the correction to the caller-supplied `notices` list.
"""
from __future__ import annotations

import pytest

from perflab.optimizers.patch import SearchReplaceBlock, apply_patch, validate_patch

UNIQUE_CONTENT = """\
import math

def compute(a, b):
    result = a + b
    return result
"""

DUPLICATED_CONTENT = """\
def f1(a, b):
    x = a + b
    return x

def f2(a, b):
    x = a + b
    return x
"""


def _workspace(tmp_path, content):
    workspace = tmp_path / "proj"
    workspace.mkdir()
    (workspace / "main.py").write_text(content)
    return workspace


class TestExactMatchUniqueness:
    def test_unique_exact_match_passes(self, tmp_path):
        workspace = _workspace(tmp_path, UNIQUE_CONTENT)
        block = SearchReplaceBlock(
            file_path="main.py",
            search="    result = a + b",
            replace="    result = a * b",
        )
        notices: list[str] = []
        errors = validate_patch([block], allowed_paths=[], workspace=workspace, notices=notices)
        assert errors == []
        assert notices == []

    def test_multi_occurrence_exact_match_is_rejected(self, tmp_path):
        workspace = _workspace(tmp_path, DUPLICATED_CONTENT)
        block = SearchReplaceBlock(
            file_path="main.py",
            search="    x = a + b\n    return x",
            replace="    x = a * b\n    return x",
        )
        errors = validate_patch([block], allowed_paths=[], workspace=workspace)
        assert len(errors) == 1
        assert "matches 2 locations" in errors[0]
        assert "more surrounding context" in errors[0]

    def test_error_message_names_file_and_block(self, tmp_path):
        workspace = _workspace(tmp_path, DUPLICATED_CONTENT)
        block = SearchReplaceBlock(
            file_path="main.py",
            search="    x = a + b\n    return x",
            replace="    x = 1\n    return x",
        )
        errors = validate_patch([block], allowed_paths=[], workspace=workspace)
        assert "Block 0" in errors[0]
        assert "main.py" in errors[0]
        assert "Ambiguous edit rejected" in errors[0]

    def test_disambiguated_search_passes(self, tmp_path):
        workspace = _workspace(tmp_path, DUPLICATED_CONTENT)
        # Adding the enclosing def line makes the match unique
        block = SearchReplaceBlock(
            file_path="main.py",
            search="def f2(a, b):\n    x = a + b\n    return x",
            replace="def f2(a, b):\n    x = a * b\n    return x",
        )
        errors = validate_patch([block], allowed_paths=[], workspace=workspace)
        assert errors == []


class TestFuzzyMatchUniqueness:
    def test_fuzzy_correction_is_surfaced_via_notices(self, tmp_path):
        workspace = _workspace(tmp_path, UNIQUE_CONTENT)
        # Whitespace near-miss: "a+b" instead of "a + b"
        block = SearchReplaceBlock(
            file_path="main.py",
            search="    result = a+b\n    return result",
            replace="    result = a * b\n    return result",
        )
        notices: list[str] = []
        errors = validate_patch([block], allowed_paths=[], workspace=workspace, notices=notices)
        assert errors == []
        assert len(notices) == 1
        assert "exact SEARCH text not found" in notices[0]
        assert "main.py" in notices[0]
        assert "auto-corrected" in notices[0]
        assert "% similar" in notices[0]
        # The block was rewritten to the actual file content
        assert block.search == "    result = a + b\n    return result"

    def test_fuzzy_correction_silent_without_notices_list(self, tmp_path):
        # Backwards compatible: omitting `notices` still validates fine
        workspace = _workspace(tmp_path, UNIQUE_CONTENT)
        block = SearchReplaceBlock(
            file_path="main.py",
            search="    result = a+b\n    return result",
            replace="    result = a * b\n    return result",
        )
        errors = validate_patch([block], allowed_paths=[], workspace=workspace)
        assert errors == []

    def test_multi_occurrence_fuzzy_match_is_rejected(self, tmp_path):
        workspace = _workspace(tmp_path, DUPLICATED_CONTENT)
        # Near-miss that fuzzy-corrects to "    x = a + b\n    return x",
        # which appears twice in the file.
        block = SearchReplaceBlock(
            file_path="main.py",
            search="    x = a+b\n    return x",
            replace="    x = a * b\n    return x",
        )
        errors = validate_patch([block], allowed_paths=[], workspace=workspace)
        assert len(errors) == 1
        assert "did not match exactly" in errors[0]
        assert "fuzzy match appears 2 times" in errors[0]
        assert "more surrounding context" in errors[0]

    def test_exact_match_produces_no_notice(self, tmp_path):
        workspace = _workspace(tmp_path, UNIQUE_CONTENT)
        block = SearchReplaceBlock(
            file_path="main.py",
            search="import math",
            replace="import math\nimport os",
        )
        notices: list[str] = []
        errors = validate_patch([block], allowed_paths=[], workspace=workspace, notices=notices)
        assert errors == []
        assert notices == []


class TestApplyPatchGuard:
    def test_apply_patch_raises_on_ambiguous_search(self, tmp_path):
        workspace = _workspace(tmp_path, DUPLICATED_CONTENT)
        block = SearchReplaceBlock(
            file_path="main.py",
            search="    x = a + b\n    return x",
            replace="    x = a * b\n    return x",
        )
        with pytest.raises(ValueError, match="matches 2 locations"):
            apply_patch([block], workspace)
        # File must be untouched
        assert (workspace / "main.py").read_text() == DUPLICATED_CONTENT

    def test_apply_patch_applies_unique_match(self, tmp_path):
        workspace = _workspace(tmp_path, UNIQUE_CONTENT)
        block = SearchReplaceBlock(
            file_path="main.py",
            search="    result = a + b",
            replace="    result = a * b",
        )
        apply_patch([block], workspace)
        assert "a * b" in (workspace / "main.py").read_text()
