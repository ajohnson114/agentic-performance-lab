#!/usr/bin/env python3
"""Keep the top-level ``tasks/`` tree in sync with ``perflab/demo_tasks/``.

Why two copies exist
--------------------
``perflab/demo_tasks/`` is the source of truth: the wheel ships it as package
data, and Python package data has to live inside the package directory.
``tasks/`` exists because it is the short path every doc, README example and CI
invocation uses. It used to be a symlink, which meant one copy -- but a symlink
makes ``mypy .`` see every bench.py twice as a duplicate module, and it is a
constant surprise to tools that walk the tree.

So the files are duplicated on purpose, and duplication without enforcement is
just drift on a delay. This script is the enforcement:

    python scripts/sync_demo_tasks.py --check   # exit 1 if they diverge
    python scripts/sync_demo_tasks.py           # make tasks/ match the source

``tests/test_demo_task_mirror.py`` calls ``--check`` logic directly, so the
existing CI test job fails on drift -- no separate CI job needed, and it is
caught locally before it ever reaches CI.

The comparison and the fix share one implementation deliberately. A checker
that disagrees with its own fixer is worse than no checker.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "perflab" / "demo_tasks"
MIRROR = REPO_ROOT / "tasks"

#: Only task *sources* are mirrored. Anything generated belongs to whichever
#: tree produced it and must never be copied across.
MIRRORED_SUFFIXES = frozenset({
    ".py", ".yaml", ".yml", ".cpp", ".cu", ".h", ".cuh", ".md", ".txt", ".json",
})

#: Directories that hold build or run output. ``out/`` is where benchmarks write
#: bench.json and run artifacts; both trees generate their own and they are
#: gitignored.
EXCLUDED_DIR_NAMES = frozenset({"out", "__pycache__", ".pytest_cache"})


def _is_excluded(rel: Path) -> bool:
    return any(
        part in EXCLUDED_DIR_NAMES or part.endswith(".dSYM") for part in rel.parts
    )


def collect(root: Path) -> dict[Path, Path]:
    """Map relative path -> absolute path for every mirrored source file."""
    if not root.exists():
        return {}
    found: dict[Path, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in MIRRORED_SUFFIXES:
            continue
        rel = path.relative_to(root)
        if _is_excluded(rel):
            continue
        found[rel] = path
    return found


def diff() -> tuple[list[Path], list[Path], list[Path]]:
    """Return (missing_from_mirror, extra_in_mirror, differing_content)."""
    src, dst = collect(SOURCE), collect(MIRROR)
    missing = sorted(set(src) - set(dst))
    extra = sorted(set(dst) - set(src))
    differing = sorted(
        rel for rel in (set(src) & set(dst))
        # shallow=False: compare contents, not size+mtime. A same-size edit is
        # exactly the drift this exists to catch.
        if not filecmp.cmp(src[rel], dst[rel], shallow=False)
    )
    return missing, extra, differing


def describe(missing: list[Path], extra: list[Path], differing: list[Path]) -> str:
    lines: list[str] = []
    if missing:
        lines.append(f"  {len(missing)} file(s) in perflab/demo_tasks/ but not tasks/:")
        lines += [f"    - {p}" for p in missing[:10]]
    if extra:
        lines.append(f"  {len(extra)} file(s) in tasks/ but not perflab/demo_tasks/:")
        lines += [f"    + {p}" for p in extra[:10]]
    if differing:
        lines.append(f"  {len(differing)} file(s) with differing contents:")
        lines += [f"    ~ {p}" for p in differing[:10]]
    total = len(missing) + len(extra) + len(differing)
    if total > 30:
        lines.append(f"  ... and {total - 30} more")
    return "\n".join(lines)


def sync() -> int:
    """Make tasks/ match perflab/demo_tasks/. Returns the number of changes."""
    missing, extra, differing = diff()
    for rel in missing + differing:
        target = MIRROR / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / rel, target)
    for rel in extra:
        (MIRROR / rel).unlink()
    # Remove directories left empty by deletions, deepest first.
    for path in sorted(MIRROR.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    return len(missing) + len(extra) + len(differing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="report drift and exit 1 instead of fixing it",
    )
    args = parser.parse_args()

    if args.check:
        missing, extra, differing = diff()
        if not (missing or extra or differing):
            print(f"tasks/ is in sync with perflab/demo_tasks/ ({len(collect(SOURCE))} files)")
            return 0
        print("tasks/ has drifted from perflab/demo_tasks/ (the source of truth):")
        print(describe(missing, extra, differing))
        print("\nFix with: python scripts/sync_demo_tasks.py")
        return 1

    changed = sync()
    print(
        f"tasks/ synced from perflab/demo_tasks/ ({changed} file(s) updated)"
        if changed else "tasks/ already in sync"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
