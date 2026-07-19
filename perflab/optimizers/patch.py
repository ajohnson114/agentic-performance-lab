from __future__ import annotations

import difflib
import fnmatch
import hashlib
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perflab.task_spec import TaskSpec

logger = logging.getLogger(__name__)

PROTECTED_FILENAMES = {"tests.py", "bench.py", "task.yaml"}


def read_source_files(task: TaskSpec) -> dict[str, str]:
    """Read all files matching edit_policy.allowed_paths.

    Rejects symlinks that resolve outside the workspace to prevent
    information disclosure (e.g. workspace/link.py -> /etc/passwd).
    """
    sources: dict[str, str] = {}
    ws = task.workspace
    ws_resolved = str(ws.resolve())
    for pattern in task.edit_policy.allowed_paths:
        # Expand glob patterns relative to workspace
        for p in sorted(ws.rglob("*")):
            if not p.is_file():
                continue
            # Reject symlinks that escape workspace
            try:
                resolved = p.resolve()
                if not str(resolved).startswith(ws_resolved + "/") and str(resolved) != ws_resolved:
                    continue
            except OSError:
                continue
            rel = str(p.relative_to(ws))
            if fnmatch.fnmatch(rel, pattern):
                try:
                    sources[rel] = p.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    pass
    return sources


def workspace_copy_ignore(
    ws: Path, out_dir: Path,
) -> Callable[[str, list[str]], set[str]]:
    """Build a shutil.copytree ignore callback that skips out_dir's contents.

    out_dir accumulates run artifacts (out/runs gains bench json, profiler
    traces, and snapshots every iteration), so copying it into each
    disposable candidate workspace makes every prescreen/eval copy slower and
    larger as a run progresses. The directory itself is kept -- empty --
    because bench harnesses write out/bench.json relative to the workspace
    without necessarily creating the directory first. Ignores nothing when
    out_dir lies outside the workspace or equals it (degenerate config).
    """
    try:
        out_resolved = out_dir.resolve()
        ws_resolved = ws.resolve()
    except OSError:
        return lambda src, names: set()
    if out_resolved == ws_resolved:
        return lambda src, names: set()

    def _ignore(src: str, names: list[str]) -> set[str]:
        try:
            if Path(src).resolve() == out_resolved:
                return set(names)
        except OSError:
            pass
        return set()

    return _ignore


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


def parse_patch_response(
    response: str, warnings: list[str] | None = None,
) -> list[SearchReplaceBlock]:
    """Parse LLM output containing search/replace blocks.

    Expected format per block:
        FILE: <path>
        <<<<<<< SEARCH
        <search text>
        =======
        <replace text>
        >>>>>>> REPLACE

    A block whose closing marker never arrives (LLM output truncated
    mid-block) is dropped, never returned partially: a REPLACE cut off at
    the truncation point would silently delete the tail of the matched
    region and could still benchmark "faster". If `warnings` is provided,
    a note is appended for every dropped incomplete block.
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
            if warnings is not None:
                warnings.append(
                    f"Dropped incomplete edit block for '{file_path}': no "
                    f"'{DIVIDER_MARKER}' divider before end of output "
                    f"(response truncated?)"
                )
            break
        i += 1  # skip =======

        # Collect replace text until >>>>>>> REPLACE
        replace_lines: list[str] = []
        terminated = False
        while i < len(lines):
            if lines[i].strip() == REPLACE_MARKER:
                terminated = True
                i += 1
                break
            replace_lines.append(lines[i])
            i += 1

        if not terminated:
            if warnings is not None:
                warnings.append(
                    f"Dropped incomplete edit block for '{file_path}': no "
                    f"'{REPLACE_MARKER}' marker before end of output "
                    f"(response truncated?)"
                )
            break

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
        # strict=False: the window can be shorter than the preview near end-of-file;
        # length mismatches are exactly what the diagnostics below report.
        for expected, found in zip(search_preview, best_window[:preview_n], strict=False):
            if expected != found:
                diff_lines.append(f"  EXPECTED: {expected!r}")
                diff_lines.append(f"  FOUND:    {found!r}")
                # strict=False: lines may differ in length; the for-else reports the
                # prefix case when no differing column is found.
                for col, (a, b) in enumerate(zip(expected, found, strict=False)):
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
        search_preview_str = "\n".join(f"  {line}" for line in search_preview)
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
) -> tuple[float, int] | None:
    """Try to fuzzy-match the search text and correct it to the actual file content.

    When the LLM produces a SEARCH block that is close but not exact (e.g. it
    changes a docstring, adds a parameter, or tweaks whitespace), this function
    finds the closest matching region in the file and rewrites block.search
    in-place so that the subsequent apply_patch() will succeed.

    Returns (similarity_ratio, 1-indexed_start_line) if the block was corrected,
    or None if no close match was found.
    """
    search_lines = block.search.splitlines()
    file_lines = content.splitlines()

    if not search_lines or not file_lines:
        return None

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
        # since we built it from file_lines). Uniqueness of the corrected text
        # is enforced by validate_patch(), same as for exact matches.
        if corrected in content:
            block.search = corrected
            return (best_ratio, best_start + 1)

    return None


def _ambiguity_error(idx: int, file_path: str, n_matches: int, fuzzy: bool) -> str:
    prefix = (
        f"Block {idx}: SEARCH text did not match exactly and its closest "
        f"fuzzy match appears {n_matches} times in '{file_path}'."
        if fuzzy
        else f"Block {idx}: SEARCH text matches {n_matches} locations in '{file_path}'."
    )
    return (
        f"{prefix} Ambiguous edit rejected — include more surrounding context "
        f"(unchanged lines before/after the target) in the SEARCH block so it "
        f"matches exactly one location."
    )


def validate_patch(
    blocks: list[SearchReplaceBlock],
    allowed_paths: list[str],
    workspace: Path,
    notices: list[str] | None = None,
) -> list[str]:
    """Validate patch blocks against policy and file contents.

    Returns a list of error strings (empty means valid).

    If `notices` is provided, non-fatal warnings are appended to it — currently
    a note whenever a block's SEARCH text did not match exactly and was
    auto-corrected via fuzzy matching, so the correction is visible in logs
    rather than silent.
    """
    errors: list[str] = []
    workspace_root = workspace.resolve()
    for idx, block in enumerate(blocks):
        # Reject absolute paths early; `workspace / "/etc/passwd"` would otherwise
        # resolve to "/etc/passwd" since the absolute right-hand side wins in `/`.
        if Path(block.file_path).is_absolute():
            errors.append(
                f"Block {idx}: path '{block.file_path}' must be relative to the workspace"
            )
            continue

        # Check path doesn't escape workspace. Resolve symlinks and ".." segments
        # first, then use is_relative_to rather than a string-prefix comparison --
        # a prefix check would let "../<workspace>-evil/x.py" through, since
        # e.g. "/home/u/proj-evil" startswith "/home/u/proj".
        try:
            resolved = (workspace / block.file_path).resolve()
        except OSError as exc:
            errors.append(f"Block {idx}: invalid path '{block.file_path}': {exc}")
            continue
        if not resolved.is_relative_to(workspace_root):
            errors.append(f"Block {idx}: path '{block.file_path}' escapes workspace")
            continue

        # From here on, match against the resolved path relative to the workspace
        # so traversal tricks (./src/x.py, src/../src/x.py) can't dodge a check.
        rel = resolved.relative_to(workspace_root).as_posix()

        # Defense-in-depth: reject edits targeting protected files
        basename = Path(rel).name
        if basename in PROTECTED_FILENAMES:
            errors.append(
                f"Block {idx}: '{block.file_path}' is a protected file "
                f"(basename '{basename}' is in PROTECTED_FILENAMES)"
            )
            continue

        # Check against allowed_paths via fnmatch
        if allowed_paths:
            matched = any(
                fnmatch.fnmatch(rel, pattern)
                for pattern in allowed_paths
            )
            if not matched:
                errors.append(
                    f"Block {idx}: path '{block.file_path}' not in allowed_paths {allowed_paths}"
                )
                continue

        # Check that the file exists and contains the search text
        target = resolved
        if not target.exists():
            errors.append(f"Block {idx}: file '{block.file_path}' does not exist")
            continue

        content = target.read_text(encoding="utf-8")

        # Reject full-file rewrites: SEARCH must not cover the vast majority of the file.
        # When the model puts the entire file in SEARCH it technically "works" but defeats
        # the purpose of surgical edits — the agent loop will ask it to try again.
        file_line_count = content.count("\n") + 1
        search_line_count = block.search.count("\n") + 1
        if file_line_count > 20 and search_line_count / file_line_count > 0.70:
            errors.append(
                f"Block {idx}: SEARCH block spans {search_line_count}/{file_line_count} lines "
                f"({search_line_count / file_line_count:.0%}) of '{block.file_path}'. "
                f"Make surgical edits — isolate only the specific lines that need to change, "
                f"not the entire file or function."
            )
            continue

        # Anti-new-kernel check: reject REPLACE blocks that introduce __global__ kernels
        # when none exist in SEARCH. This catches gaming (adding a no-op fast path from
        # scratch) while allowing legitimate kernel refactors (e.g. splitting naive into
        # tiled + keeping naive for selftest).
        import re
        search_has_global = "__global__" in block.search
        replace_has_global = "__global__" in block.replace
        search_has_def = re.search(r"^\s*(static\s+)?\w+\s+\w+\s*\(", block.search, re.MULTILINE)
        replace_has_def = re.search(r"^\s*(static\s+)?\w+\s+\w+\s*\(", block.replace, re.MULTILINE)

        if replace_has_global and not search_has_global:
            errors.append(
                f"Block {idx}: REPLACE introduces __global__ kernel(s) not present in SEARCH. "
                f"Only modify existing kernels; do not add new ones from scratch."
            )
            continue
        if replace_has_def and not search_has_def:
            errors.append(
                f"Block {idx}: REPLACE introduces new function definition(s) not in SEARCH. "
                f"Modify existing functions only, never create new ones."
            )
            continue

        if block.search in content:
            # Require a unique match: replacing "the first occurrence" of an
            # ambiguous SEARCH can silently edit the wrong site.
            n_matches = content.count(block.search)
            if n_matches > 1:
                errors.append(
                    _ambiguity_error(idx, block.file_path, n_matches, fuzzy=False)
                )
        else:
            # Try fuzzy matching — auto-correct near-miss SEARCH blocks
            fuzzy_result = _fuzzy_match_and_correct(block, content)
            if fuzzy_result is None:
                diagnostic = _diagnose_match_failure(
                    block.search, content, block.file_path
                )
                errors.append(f"Block {idx}: {diagnostic}")
                continue

            ratio, line_num = fuzzy_result
            # The corrected SEARCH must be unique too.
            n_matches = content.count(block.search)
            if n_matches > 1:
                errors.append(
                    _ambiguity_error(idx, block.file_path, n_matches, fuzzy=True)
                )
                continue
            if notices is not None:
                notices.append(
                    f"Block {idx}: exact SEARCH text not found in "
                    f"'{block.file_path}'; auto-corrected to the closest "
                    f"matching region at line {line_num} ({ratio:.0%} similar). "
                    f"The corrected text was used for the edit."
                )

    return errors


def apply_patch(blocks: list[SearchReplaceBlock], workspace: Path) -> None:
    """Apply search/replace blocks (one occurrence per block).

    Defense-in-depth: raises ValueError if a block's search text matches more
    than once — validate_patch() rejects such patches, so hitting this means a
    caller skipped validation or the file changed underneath us.
    """
    for block in blocks:
        target = workspace / block.file_path
        content = target.read_text(encoding="utf-8")
        n_matches = content.count(block.search)
        if n_matches > 1:
            raise ValueError(
                f"Ambiguous patch: search text matches {n_matches} locations "
                f"in '{block.file_path}' (patches must match exactly once)"
            )
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


def _protected_files(workspace: Path) -> list[Path]:
    return sorted(
        p for p in workspace.rglob("*")
        if p.is_file() and p.name in PROTECTED_FILENAMES
    )


def snapshot_protected_files(workspace: Path, snapshot_dir: Path) -> dict[str, str]:
    """Snapshot protected files (tests.py/bench.py/task.yaml) for tamper detection.

    Patch validation blocks *edits* to these files, but candidate code
    executed in the real workspace (post-accept re-profiling, autotune,
    drift checks) can still rewrite them at runtime. Copies every protected
    file into snapshot_dir and returns {relative_path: sha256} for
    verify_protected_files() to check against.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for p in _protected_files(workspace):
        rel = p.relative_to(workspace).as_posix()
        shutil.copy2(p, snapshot_dir / rel.replace("/", "__"))
        hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def verify_protected_files(
    workspace: Path, snapshot_dir: Path, hashes: dict[str, str],
) -> list[str]:
    """Compare protected files against their snapshot, restoring any that changed.

    Returns the relative paths that had been modified or deleted since the
    snapshot (empty = clean). Detection via the returned list is
    authoritative; restoration is best-effort. Each tampered file is restored
    from the snapshot so later correctness checks stay honest, but the restore
    never raises: candidate code may have deleted a protected file's enclosing
    directory (so ``copy2`` would hit FileNotFoundError) or the snapshot copy
    itself may be gone. The parent directory is recreated first, and any
    restore failure degrades to a logged warning -- the file stays flagged as
    tampered either way. The per-iteration caller relies on this: a crash here
    would take down the very run this guard protects.
    """
    tampered: list[str] = []
    for rel, expected in hashes.items():
        target = workspace / rel
        actual = (
            hashlib.sha256(target.read_bytes()).hexdigest()
            if target.exists() else None
        )
        if actual != expected:
            tampered.append(rel)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot_dir / rel.replace("/", "__"), target)
            except OSError as exc:
                logger.warning(
                    "Failed to restore tampered protected file '%s' from "
                    "snapshot: %s. File remains flagged as tampered.",
                    rel, exc,
                )
    return tampered
