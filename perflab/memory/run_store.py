from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from perflab.optimizers.patch import read_source_files

if TYPE_CHECKING:
    from perflab.task_spec import TaskSpec

logger = logging.getLogger(__name__)


def validate_run_id(run_id: str) -> str:
    """Reject run_ids that are not a single plain path segment.

    run_id is joined onto runs_root and reaches these methods from CLI
    arguments and MCP clients — a value like "../../etc" would read or
    write outside the runs directory.
    """
    if not run_id or run_id in (".", "..") or run_id != Path(run_id).name:
        raise ValueError(f"Invalid run_id: {run_id!r} (must be a plain directory name)")
    return run_id


@dataclass
class RunPaths:
    run_id: str
    run_dir: Path
    artifacts_dir: Path
    logs_dir: Path


def snapshot_workspace(task: TaskSpec, run_dir: Path, label: str) -> Path | None:
    """Zip all allowed_paths files to run_dir/snapshots/<label>.zip."""
    import zipfile

    sources = read_source_files(task)
    if not sources:
        return None

    snap_dir = run_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    zip_path = snap_dir / f"{label}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path, content in sources.items():
            zf.writestr(rel_path, content)

    return zip_path


_profiler_summary_cache: dict[str, tuple[dict[str, float], dict[str, dict]]] = {}


def load_profiler_summaries(artifacts_dir: Path) -> dict[str, dict]:
    """Load all *_summary.json files from artifacts dir.

    Each profiler writes its own ``<name>_summary.json`` keyed by profiler
    name (e.g. ``nsys``, ``torch_profiler``, ``pyspy``).  When multiple
    profilers report overlapping metrics (e.g. GPU kernel time from both
    NSys and torch profiler), each profiler's data is kept in its own
    namespace — consumers choose which source to use based on context
    (e.g. ``nsys`` for correlationId data, ``torch_profiler`` for operator
    breakdown).  There is no merging or conflict resolution at load time.

    Results are cached by directory path + file mtimes. Subsequent calls
    return the cached result if no summary files have been modified.
    """
    cache_key = str(artifacts_dir)
    summaries: dict[str, dict] = {}
    if not artifacts_dir.exists():
        return summaries

    # Build mtime fingerprint for cache validation
    current_mtimes: dict[str, float] = {}
    for p in artifacts_dir.glob("*_summary.json"):
        try:
            current_mtimes[str(p)] = p.stat().st_mtime
        except OSError:
            pass

    if cache_key in _profiler_summary_cache:
        cached_mtimes, cached_summaries = _profiler_summary_cache[cache_key]
        if cached_mtimes == current_mtimes:
            return cached_summaries

    for p in artifacts_dir.glob("*_summary.json"):
        try:
            summaries[p.stem.replace("_summary", "")] = json.loads(
                p.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load profiler summary %s", p, exc_info=True)

    _profiler_summary_cache[cache_key] = (current_mtimes, summaries)
    return summaries

class RunStore:
    def __init__(self, out_root: Path):
        self.out_root = out_root
        self.runs_root = out_root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.runs_root / "index.jsonl"

    def new_run(self, task_name: str, program_type: str | None = None) -> RunPaths:
        ts = time.strftime("%Y%m%d-%H%M%S")
        rid = f"{ts}-{uuid.uuid4().hex[:8]}"
        run_dir = self.runs_root / rid
        artifacts_dir = run_dir / "artifacts"
        logs_dir = run_dir / "logs"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        meta: dict = {"run_id": rid, "task": task_name, "created_at": ts}
        if program_type is not None:
            meta["program_type"] = program_type
        self._append_index(meta)
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return RunPaths(run_id=rid, run_dir=run_dir, artifacts_dir=artifacts_dir, logs_dir=logs_dir)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def list_runs(self, task: str | None = None, limit: int = 50) -> list[dict]:
        """List runs from index.jsonl, newest first. Optionally filter by task name."""
        if not self.index_path.exists():
            return []
        entries: list[dict] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if task is not None and entry.get("task") != task:
                continue
            entries.append(entry)

        # Newest first
        entries.reverse()
        entries = entries[:limit]

        # Enrich from meta.json where possible
        for entry in entries:
            rid = entry.get("run_id", "")
            meta_path = self.runs_root / rid / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    for key in ("best_value", "status", "program_type"):
                        if key in meta and key not in entry:
                            entry[key] = meta[key]
                except (json.JSONDecodeError, OSError):
                    logger.warning("Failed to load meta.json for run %s", rid, exc_info=True)
        return entries

    def get_run(self, run_id: str) -> dict:
        """Load full run data: meta, report, bench, profiler summaries.

        Raises ValueError on a malformed run_id, FileNotFoundError if the
        run directory does not exist.
        """
        run_dir = self.runs_root / validate_run_id(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Meta
        meta: dict = {}
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        # Report (optional)
        report: dict | None = None
        report_path = run_dir / "report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load report.json for run %s", run_id, exc_info=True)

        # Bench (optional)
        bench: dict | None = None
        bench_path = run_dir / "bench.json"
        if bench_path.exists():
            try:
                bench = json.loads(bench_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load bench.json for run %s", run_id, exc_info=True)

        # Profiler summaries
        profiler_summaries: dict[str, dict] = {}
        artifacts_dir = run_dir / "artifacts"
        if artifacts_dir.exists():
            for p in artifacts_dir.glob("*_summary.json"):
                try:
                    profiler_summaries[p.stem.replace("_summary", "")] = json.loads(
                        p.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    logger.warning("Failed to load profiler summary %s", p, exc_info=True)

        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "meta": meta,
            "report": report,
            "bench": bench,
            "profiler_summaries": profiler_summaries,
        }

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict:
        """Compare two runs: extract best values, compute delta/speedup, diff bottlenecks."""
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)

        def _best(run: dict) -> float | None:
            report = run.get("report")
            if report and "best_value" in report:
                return report["best_value"]
            meta = run.get("meta", {})
            return meta.get("best_value")

        def _field(run: dict, key: str) -> str | None:
            """Extract a field from report or meta, preferring report."""
            report = run.get("report")
            if report and key in report:
                return report[key]
            meta = run.get("meta", {})
            return meta.get(key)

        val_a = _best(run_a)
        val_b = _best(run_b)

        delta: float | None = None
        ratio: float | None = None
        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            ratio = val_b / val_a if val_a != 0 else None

        # Bottleneck diff
        def _bottlenecks(run: dict) -> list[str]:
            report = run.get("report")
            if not report:
                return []
            return [d.get("bottleneck", "") for d in report.get("bottleneck_diagnoses", [])]

        bn_a = set(_bottlenecks(run_a))
        bn_b = set(_bottlenecks(run_b))

        # Extract shared context (metric name/mode, task)
        metric_name = _field(run_a, "metric_name") or _field(run_b, "metric_name")
        metric_mode = _field(run_a, "metric_mode") or _field(run_b, "metric_mode")
        task_name = _field(run_a, "task_name") or _field(run_b, "task_name") \
            or _field(run_a, "task") or _field(run_b, "task")
        status_a = _field(run_a, "status")
        status_b = _field(run_b, "status")

        return {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "task_name": task_name,
            "metric_name": metric_name,
            "metric_mode": metric_mode,
            "status_a": status_a,
            "status_b": status_b,
            "value_a": val_a,
            "value_b": val_b,
            "delta": delta,
            "ratio": ratio,
            "resolved_bottlenecks": sorted(bn_a - bn_b),
            "new_bottlenecks": sorted(bn_b - bn_a),
        }

    def update_meta(self, run_id: str, updates: dict) -> None:
        """Merge updates into the run's meta.json."""
        meta_path = self.runs_root / validate_run_id(run_id) / "meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to read existing meta.json for run %s", run_id, exc_info=True)
        meta.update(updates)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _append_index(self, obj: dict) -> None:
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")
