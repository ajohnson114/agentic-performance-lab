"""Before/after source diffs between the snapshots a run leaves behind.

``snapshot_workspace`` already zips every ``allowed_paths`` file at the
baseline and again at each *accepted* iteration, so a finished run carries the
complete source at every point where the metric actually moved. What was
missing was a way to read them back: the dashboard's "What changed" section
renders each edit's search/replace blocks truncated to 500 characters, which
is a summary of a change rather than the change.

A candidate is only ever accepted if it beat the incumbent, so the
highest-numbered ``iter<N>`` snapshot is by construction the best one. That
makes it the default right-hand side, with the baseline on the left.
"""
from __future__ import annotations

import difflib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

SNAPSHOT_DIR = "snapshots"
BASELINE_LABEL = "baseline"

_ITER_RE = re.compile(r"^iter(\d+)$")


def _label_sort_key(label: str) -> tuple[int, int, str]:
    """Order labels chronologically: baseline first, then iterations numerically.

    Lexicographic order would put iter10 before iter2, which silently picks the
    wrong "last accepted" snapshot on any run past nine accepts.
    """
    if label == BASELINE_LABEL:
        return (0, 0, "")
    m = _ITER_RE.match(label)
    if m:
        return (1, int(m.group(1)), "")
    return (2, 0, label)


def list_labels(run_dir: Path) -> list[str]:
    """Snapshot labels present in *run_dir*, chronologically ordered."""
    snap_dir = run_dir / SNAPSHOT_DIR
    if not snap_dir.is_dir():
        return []
    labels = [p.stem for p in snap_dir.glob("*.zip")]
    return sorted(labels, key=_label_sort_key)


def read_snapshot(run_dir: Path, label: str) -> dict[str, str]:
    """Read a snapshot zip into {relative path: text}."""
    zip_path = run_dir / SNAPSHOT_DIR / f"{label}.zip"
    if not zip_path.is_file():
        raise ValueError(f"No snapshot {label!r} in {run_dir / SNAPSHOT_DIR}")
    sources: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            sources[name] = zf.read(name).decode("utf-8", errors="replace")
    return sources


def resolve_labels(
    run_dir: Path,
    from_label: str | None = None,
    to_label: str | None = None,
) -> tuple[str, str | None]:
    """Fill in the default endpoints: baseline -> last accepted iteration.

    Returns (from, to). *to* is None when the run has a baseline but never
    accepted anything -- a real outcome (no candidate beat the incumbent), not
    an error, so the caller decides how to report it.
    """
    labels = list_labels(run_dir)
    if not labels:
        raise ValueError(f"No snapshots in {run_dir / SNAPSHOT_DIR}")

    for requested in (from_label, to_label):
        if requested is not None and requested not in labels:
            raise ValueError(
                f"No snapshot {requested!r} in this run. Available: {', '.join(labels)}"
            )

    resolved_from = from_label or (
        BASELINE_LABEL if BASELINE_LABEL in labels else labels[0]
    )
    if to_label is not None:
        return resolved_from, to_label

    later = [x for x in labels if x != resolved_from]
    return resolved_from, (later[-1] if later else None)


@dataclass
class DiffResult:
    text: str
    files_changed: int
    insertions: int
    deletions: int


def diff_snapshots(run_dir: Path, from_label: str, to_label: str) -> DiffResult:
    """Unified diff between two snapshots of a run, with git-style counts."""
    before = read_snapshot(run_dir, from_label)
    after = read_snapshot(run_dir, to_label)

    chunks: list[str] = []
    files_changed = insertions = deletions = 0

    for rel in sorted(set(before) | set(after)):
        old = before.get(rel)
        new = after.get(rel)
        if old == new:
            continue
        files_changed += 1
        lines = difflib.unified_diff(
            (old or "").splitlines(keepends=True),
            (new or "").splitlines(keepends=True),
            fromfile=f"{from_label}/{rel}" if old is not None else "/dev/null",
            tofile=f"{to_label}/{rel}" if new is not None else "/dev/null",
        )
        for line in lines:
            # A source file with no trailing newline yields a final chunk
            # without one; left as-is it would run into the next file's header.
            if not line.endswith("\n"):
                line += "\n"
            chunks.append(line)
            if line.startswith("+") and not line.startswith("+++"):
                insertions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1

    return DiffResult(
        text="".join(chunks),
        files_changed=files_changed,
        insertions=insertions,
        deletions=deletions,
    )
