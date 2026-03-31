from __future__ import annotations

import difflib
import fnmatch
import shutil
from dataclasses import dataclass, field
from pathlib import Path


PROTECTED_FILENAMES = {"tests.py", "bench.py", "task.yaml"}


@dataclass
class SearchReplaceBlock:
    file_path: str
    search: str
    replace: str


@dataclass
class Patch:
    description: str
    blocks: list[SearchReplaceBlock] = field(default_factory=list)


SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDER_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that some LLMs wrap around edit blocks."""
    import re
    # Remove ```python, ```yaml, ```, etc. lines
    return re.sub(r"^```\w*\s*$", "", text, flags=re.MULTILINE)


def parse_patch_response(response: str) -> list[SearchReplaceBlock]:
    """Parse LLM output containing search/replace blocks.

    Expected format per block:
        FILE: <path>
        <<<<<<< SEARCH
        <search text>
        =======
        <replace text>
        >>>>>>> REPLACE
    """
    response = _strip_code_fences(response)
    blocks: list[SearchReplaceBlock] = []
    lines = response.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for FILE: marker or a SEARCH marker
        if line.strip().startswith("FILE:"):
            file_path = line.strip().split("FILE:", 1)[1].strip()
            i += 1
        elif line.strip() == SEARCH_MARKER:
            # If we hit SEARCH without FILE:, try to find file_path
            # from previous context -- skip block if we can't
            file_path = ""
            # Search backwards for a FILE: line
            for j in range(i - 1, max(i - 5, -1), -1):
                if lines[j].strip().startswith("FILE:"):
                    file_path = lines[j].strip().split("FILE:", 1)[1].strip()
                    break
            if not file_path:
                # Skip malformed block
                i += 1
                continue
        else:
            i += 1
            continue

        # Now expect <<<<<<< SEARCH
        if i < len(lines) and lines[i].strip() == SEARCH_MARKER:
            i += 1
        elif line.strip() == SEARCH_MARKER:
            # Already consumed the SEARCH marker above
            pass
        else:
            continue

        # Collect search text until =======
        search_lines: list[str] = []
        while i < len(lines) and lines[i].strip() != DIVIDER_MARKER:
            search_lines.append(lines[i])
            i += 1
        if i >= len(lines):
            break
        i += 1  # skip =======

        # Collect replace text until >>>>>>> REPLACE
        replace_lines: list[str] = []
        while i < len(lines) and lines[i].strip() != REPLACE_MARKER:
            replace_lines.append(lines[i])
            i += 1
        if i < len(lines):
            i += 1  # skip >>>>>>> REPLACE

        blocks.append(SearchReplaceBlock(
            file_path=file_path,
            search="\n".join(search_lines),
            replace="\n".join(replace_lines),
        ))

    return blocks


def _diagnose_match_failure(
    search_text: str,
    file_content: str,
    file_path: str,
    max_diagnostic_length: int = 500,
) -> str:
    """Produce a diagnostic explaining why a search/replace match failed.

    Uses difflib.SequenceMatcher to find the closest matching region in the file
    and reports the difference to help the LLM self-correct.
    """
    search_lines = search_text.splitlines()
    file_lines = file_content.splitlines()
    preview_n = min(3, len(search_lines))
    search_preview = search_lines[:preview_n]

    if not file_lines or not search_lines:
        return (
            f"Search text not found in '{file_path}'. "
            f"The file is {'empty' if not file_lines else 'non-empty'} and "
            f"the search text is {'empty' if not search_lines else 'non-empty'}."
        )

    # Sliding window: compare search text against windows of the file
    # Use character-level matching for more accurate similarity
    best_ratio = 0.0
    best_start = 0
    window_size = len(search_lines)

    for start in range(max(1, len(file_lines) - window_size + 1)):
        end = min(start + window_size, len(file_lines))
        window_text = "\n".join(file_lines[start:end])
        ratio = difflib.SequenceMatcher(
            None, search_text, window_text
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = start

    if best_ratio > 0.5:
        best_window = file_lines[best_start : best_start + window_size]
        line_num = best_start + 1  # 1-indexed

        # Find first differing line
        diff_lines: list[str] = []
        for i, (expected, found) in enumerate(
            zip(search_preview, best_window[:preview_n])
        ):
            if expected != found:
                diff_lines.append(f"  EXPECTED: {expected!r}")
                diff_lines.append(f"  FOUND:    {found!r}")
                # Find column of first difference
                for col, (a, b) in enumerate(zip(expected, found)):
                    if a != b:
                        diff_lines.append(f"  {'':>10}{' ' * col}^ difference at column {col}")
                        break
                else:
                    # One is a prefix of the other
                    shorter = min(len(expected), len(found))
                    diff_lines.append(f"  {'':>10}{' ' * shorter}^ length differs")
                break

        diagnostic = (
            f"Search text not found in '{file_path}'. "
            f"Closest match at line {line_num} ({best_ratio:.0%} similar):\n"
            + "\n".join(diff_lines)
        )
    else:
        search_preview_str = "\n".join(f"  {l}" for l in search_preview)
        diagnostic = (
            f"Search text not found in '{file_path}'. "
            f"No similar region found — the search text may reference code that "
            f"doesn't exist in the current version of this file.\n"
            f"First {preview_n} lines of search text:\n{search_preview_str}"
        )

    if len(diagnostic) > max_diagnostic_length:
        diagnostic = diagnostic[: max_diagnostic_length - 3] + "..."
    return diagnostic


def _fuzzy_match_and_correct(
    block: SearchReplaceBlock,
    content: str,
    min_similarity: float = 0.80,
) -> bool:
    """Try to fuzzy-match the search text and correct it to the actual file content.

    When the LLM produces a SEARCH block that is close but not exact (e.g. it
    changes a docstring, adds a parameter, or tweaks whitespace), this function
    finds the closest matching region in the file and rewrites block.search
    in-place so that the subsequent apply_patch() will succeed.

    Returns True if the block was corrected, False if no close match was found.
    """
    search_lines = block.search.splitlines()
    file_lines = content.splitlines()

    if not search_lines or not file_lines:
        return False

    base_window = len(search_lines)
    best_ratio = 0.0
    best_start = 0
    best_end = 0

    # Try windows of size base ±2 to handle added/removed lines
    for delta in (0, -1, 1, -2, 2):
        window_size = base_window + delta
        if window_size < 1:
            continue
        for start in range(max(1, len(file_lines) - window_size + 1)):
            end = start + window_size
            if end > len(file_lines):
                continue
            window_text = "\n".join(file_lines[start:end])
            ratio = difflib.SequenceMatcher(
                None, block.search, window_text
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_end = end

    if best_ratio >= min_similarity:
        corrected = "\n".join(file_lines[best_start:best_end])
        # Make sure the corrected text is actually in the file (it should be,
        # since we built it from file_lines) and is unique enough.
        if corrected in content:
            block.search = corrected
            return True

    return False


def validate_patch(
    blocks: list[SearchReplaceBlock],
    allowed_paths: list[str],
    workspace: Path,
) -> list[str]:
    """Validate patch blocks against policy and file contents.

    Returns a list of error strings (empty means valid).
    """
    errors: list[str] = []
    for idx, block in enumerate(blocks):
        # Defense-in-depth: reject edits targeting protected files
        basename = Path(block.file_path).name
        if basename in PROTECTED_FILENAMES:
            errors.append(
                f"Block {idx}: '{block.file_path}' is a protected file "
                f"(basename '{basename}' is in PROTECTED_FILENAMES)"
            )
            continue

        # Check path doesn't escape workspace
        try:
            resolved = (workspace / block.file_path).resolve()
            if not str(resolved).startswith(str(workspace.resolve())):
                errors.append(f"Block {idx}: path '{block.file_path}' escapes workspace")
                continue
        except Exception as exc:
            errors.append(f"Block {idx}: invalid path '{block.file_path}': {exc}")
            continue

        # Check against allowed_paths via fnmatch
        if allowed_paths:
            matched = any(
                fnmatch.fnmatch(block.file_path, pattern)
                for pattern in allowed_paths
            )
            if not matched:
                errors.append(
                    f"Block {idx}: path '{block.file_path}' not in allowed_paths {allowed_paths}"
                )
                continue

        # Check that the file exists and contains the search text
        target = workspace / block.file_path
        if not target.exists():
            errors.append(f"Block {idx}: file '{block.file_path}' does not exist")
            continue

        content = target.read_text(encoding="utf-8")
        if block.search not in content:
            # Try fuzzy matching — auto-correct near-miss SEARCH blocks
            if not _fuzzy_match_and_correct(block, content):
                diagnostic = _diagnose_match_failure(
                    block.search, content, block.file_path
                )
                errors.append(f"Block {idx}: {diagnostic}")

    return errors


def apply_patch(blocks: list[SearchReplaceBlock], workspace: Path) -> None:
    """Apply search/replace blocks (one occurrence per block)."""
    for block in blocks:
        target = workspace / block.file_path
        content = target.read_text(encoding="utf-8")
        # Replace only first occurrence
        content = content.replace(block.search, block.replace, 1)
        target.write_text(content, encoding="utf-8")


def backup_files(
    blocks: list[SearchReplaceBlock],
    workspace: Path,
    backup_dir: Path,
) -> dict[str, Path]:
    """Copy affected files before patching. Returns {file_path: backup_path}."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_up: dict[str, Path] = {}
    for block in blocks:
        if block.file_path in backed_up:
            continue
        src = workspace / block.file_path
        if src.exists():
            dst = backup_dir / block.file_path.replace("/", "__")
            shutil.copy2(src, dst)
            backed_up[block.file_path] = dst
    return backed_up


def restore_files(backed_up: dict[str, Path], workspace: Path) -> None:
    """Restore files from backup."""
    for file_path, backup_path in backed_up.items():
        dst = workspace / file_path
        shutil.copy2(backup_path, dst)
