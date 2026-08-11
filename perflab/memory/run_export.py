"""Bundle a run directory into a portable archive.

A run is produced wherever the accelerator is -- typically a rented cloud GPU
instance -- and read wherever the person is. Moving it is therefore a routine
step, and the thing that makes it awkward is size: a single ``ncu --set full``
report or an ``nsys`` SQLite export can outweigh everything else in the run
combined.

``slim`` applies one rule: *keep what a different machine can actually open.*
A ``.ncu-rep`` needs Nsight Compute, a ``perf.data`` needs perf plus the
original binaries and kernel symbols, an Instruments ``.trace`` needs macOS --
none of those arrive in a useful state, and every profiler has already written
its parsed findings to ``<name>_summary.json``, which is what the dashboard and
the LLM actually read. What stays is everything that opens in a browser or a
text editor: the self-contained ``dashboard.html``, the speedscope profile, the
torch and perfetto Chrome traces, the memray flame graph, the HLO text dump,
and the source snapshots.
"""
from __future__ import annotations

import fnmatch
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path

# Matched against the base name of each file *and* each directory, so bundle
# directories (Instruments writes metal_trace.trace as a directory) are pruned
# whole rather than walked into.
SLIM_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "*.ncu-rep",       # Nsight Compute report -- needs ncu to open
    "*.nsys-rep",      # Nsight Systems report -- needs nsys to open
    "*.sqlite",        # nsys SQLite export -- parser input for nsys_summary.json
    "perf*.data",      # perf record/lock/c2c/sched -- needs perf + the original host
    "perf_script.txt", # raw perf script dump -- parser input, grows with sample count
    "*.trace",         # Instruments Metal trace bundle -- needs macOS + Instruments
    "*.pb",            # XLA/TensorBoard xplane traces -- needs TensorBoard
)


def is_slim_excluded(name: str) -> bool:
    """Whether a file or directory base name is dropped by a slim export."""
    return any(fnmatch.fnmatch(name, pat) for pat in SLIM_EXCLUDE_PATTERNS)


@dataclass
class ExportResult:
    path: Path
    file_count: int
    source_bytes: int
    archive_bytes: int
    skipped_count: int
    skipped_bytes: int


def export_run(run_dir: Path, dest: Path, *, slim: bool = False) -> ExportResult:
    """Write *run_dir* to *dest* as a gzipped tar rooted at the run's own name.

    Extracting the archive yields a single ``<run_id>/`` directory, so several
    exports can be unpacked side by side and handed straight to ``perflab view``.
    """
    run_dir = run_dir.resolve()
    if not (run_dir / "meta.json").is_file():
        raise ValueError(f"Not a run directory (no meta.json): {run_dir}")

    dest = dest.resolve()
    # Writing the archive into the tree being archived would have os.walk pick
    # up the partially-written file (or not, depending on walk order) -- a
    # nondeterministic self-inclusion bug rather than an error.
    if dest == run_dir or run_dir in dest.parents:
        raise ValueError(f"Archive destination is inside the run directory: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    root = run_dir.name
    file_count = source_bytes = skipped_count = skipped_bytes = 0

    with tarfile.open(dest, "w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(run_dir):
            here = Path(dirpath)
            if slim:
                kept = []
                for name in sorted(dirnames):
                    if is_slim_excluded(name):
                        count, size = _dir_stats(here / name)
                        skipped_count += count
                        skipped_bytes += size
                    else:
                        kept.append(name)
                dirnames[:] = kept
            else:
                dirnames[:] = sorted(dirnames)

            for name in sorted(filenames):
                path = here / name
                size = _size_of(path)
                if slim and is_slim_excluded(name):
                    skipped_count += 1
                    skipped_bytes += size
                    continue
                arcname = f"{root}/{path.relative_to(run_dir).as_posix()}"
                tar.add(path, arcname=arcname, recursive=False)
                file_count += 1
                source_bytes += size

    return ExportResult(
        path=dest,
        file_count=file_count,
        source_bytes=source_bytes,
        archive_bytes=dest.stat().st_size,
        skipped_count=skipped_count,
        skipped_bytes=skipped_bytes,
    )


def human_bytes(n: int) -> str:
    """Format a byte count for terminal output (1.4 GB, 812 KB, 96 B)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover -- loop always returns


def _size_of(path: Path) -> int:
    """Size of *path* without following symlinks; 0 if it cannot be stat'd."""
    try:
        return os.lstat(path).st_size
    except OSError:
        return 0


def _dir_stats(path: Path) -> tuple[int, int]:
    """(file count, total bytes) under *path*, used to report what slim dropped."""
    count = 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            count += 1
            total += _size_of(Path(dirpath) / name)
    return count, total
