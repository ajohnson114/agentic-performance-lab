from __future__ import annotations

import logging
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from perflab.profilers.base import ProfileResult, run_bench_under
from perflab.profilers.interval_union import union_duration
from perflab.tools.shell import run_cmd

logger = logging.getLogger(__name__)


@dataclass
class NsysProfiler:
    name: str = "nsys"

    def is_available(self) -> bool:
        return shutil.which("nsys") is not None

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        report_base = artifacts_dir / "nsys_report"

        # nsys profile with stats
        profile_res = run_bench_under([
            "nsys", "profile",
            "--stats=true",
            f"--output={report_base}",
            "--force-overwrite=true",
        ], bench_cmd, cwd=cwd)

        nsys_rep = Path(f"{report_base}.nsys-rep")
        sqlite_path = Path(f"{report_base}.sqlite")

        # Export to SQLite for programmatic analysis
        if nsys_rep.exists() and not sqlite_path.exists():
            export_cmd = [
                "nsys", "export",
                "--type=sqlite",
                f"--output={sqlite_path}",
                str(nsys_rep),
            ]
            run_cmd(export_cmd, cwd=cwd)

        # Prefer SQLite parsing; fall back to regex-based stdout parsing
        if sqlite_path.exists():
            summary = _parse_nsys_sqlite(sqlite_path)
        else:
            summary = _parse_nsys_stats(profile_res.stdout + "\n" + profile_res.stderr)

        summary["returncode"] = profile_res.returncode
        summary["duration_s"] = profile_res.duration_s

        artifacts: dict[str, str] = {}
        if nsys_rep.exists():
            artifacts["nsys_rep"] = str(nsys_rep)
        if sqlite_path.exists():
            artifacts["nsys_sqlite"] = str(sqlite_path)

        return ProfileResult(name=self.name, artifacts=artifacts, summary=summary)


def _parse_nsys_sqlite(sqlite_path: Path) -> dict:
    """Extract structured profiling data from the nsys SQLite export.

    Queries the well-defined tables that nsys exports rather than parsing
    version-dependent stdout text.
    """
    result: dict = {}
    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
    except (sqlite3.Error, OSError):
        logger.warning("Failed to open nsys SQLite database %s", sqlite_path, exc_info=True)
        return result

    try:
        _extract_top_kernels(conn, result)
    except sqlite3.Error:
        logger.warning("Failed to extract top kernels from nsys", exc_info=True)

    try:
        _extract_memcpy(conn, result)
    except sqlite3.Error:
        logger.warning("Failed to extract memcpy data from nsys", exc_info=True)

    try:
        _extract_top_api_calls(conn, result)
    except sqlite3.Error:
        logger.warning("Failed to extract API calls from nsys", exc_info=True)

    try:
        _extract_gpu_utilization(conn, result)
    except sqlite3.Error:
        logger.warning("Failed to extract GPU utilization from nsys", exc_info=True)

    try:
        _extract_kernel_gaps(conn, result)
    except sqlite3.Error:
        logger.warning("Failed to extract kernel gaps from nsys", exc_info=True)

    try:
        _extract_nvtx_ranges(conn, result)
    except sqlite3.Error:
        logger.warning("Failed to extract NVTX ranges from nsys", exc_info=True)

    try:
        _extract_kernel_launch_dims(conn, result)
    except sqlite3.Error:
        logger.warning("Failed to extract kernel launch dims from nsys", exc_info=True)

    try:
        _extract_cuda_sync_overhead(conn, result)
    except sqlite3.Error:
        logger.warning("Failed to extract CUDA sync overhead from nsys", exc_info=True)

    try:
        _extract_cpu_gpu_correlation(conn, result)
    except sqlite3.Error:
        logger.warning("Failed to extract CPU-GPU correlation from nsys", exc_info=True)

    try:
        _extract_callchain_context(conn, result)
    except sqlite3.Error:
        logger.warning("Failed to extract call chain context from nsys", exc_info=True)

    try:
        _extract_per_stream_gaps(conn, result)
    except sqlite3.Error:
        logger.warning("Failed to extract per-stream gaps from nsys", exc_info=True)

    conn.close()
    return result


def _extract_top_kernels(conn: sqlite3.Connection, result: dict) -> None:
    """Top-10 GPU kernels by total time."""
    cur = conn.execute("""
        SELECT demangledName,
               COUNT(*)       AS count,
               SUM(end-start) AS total_ns,
               AVG(end-start) AS avg_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY demangledName
        ORDER BY total_ns DESC
        LIMIT 10
    """)
    rows = cur.fetchall()
    if not rows:
        return

    total_kernel_ns = sum(r["total_ns"] for r in rows)
    # Get the true total across *all* kernels, not just top-10
    all_total = conn.execute(
        "SELECT SUM(end-start) AS t FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    total_all_ns = all_total["t"] if all_total and all_total["t"] else total_kernel_ns

    kernels = []
    for r in rows:
        pct = (r["total_ns"] / total_all_ns * 100.0) if total_all_ns > 0 else 0.0
        kernels.append({
            "name": r["demangledName"] or "(unknown)",
            "count": r["count"],
            "total_ms": r["total_ns"] / 1e6,
            "avg_us": r["avg_ns"] / 1e3,
            "pct": round(pct, 1),
        })

    result["top_kernels"] = kernels
    result["cuda_kernel_time_ms"] = total_all_ns / 1e6


def _extract_memcpy(conn: sqlite3.Connection, result: dict) -> None:
    """Memory transfer summary grouped by direction."""
    kind_map = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
    cur = conn.execute("""
        SELECT copyKind,
               COUNT(*)       AS count,
               SUM(bytes)     AS total_bytes,
               SUM(end-start) AS total_ns
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        GROUP BY copyKind
    """)
    rows = cur.fetchall()
    if not rows:
        return

    memcpy = []
    total_memcpy_ns = 0
    for r in rows:
        total_memcpy_ns += r["total_ns"] or 0
        memcpy.append({
            "direction": kind_map.get(r["copyKind"], f"kind_{r['copyKind']}"),
            "count": r["count"],
            "total_bytes": r["total_bytes"] or 0,
            "total_ms": (r["total_ns"] or 0) / 1e6,
        })

    result["memcpy"] = memcpy
    result["memcpy_time_ms"] = total_memcpy_ns / 1e6


def _extract_top_api_calls(conn: sqlite3.Connection, result: dict) -> None:
    """Top-10 CUDA runtime API calls by total time."""
    # Try joining with StringIds for readable names (schema varies by version)
    try:
        cur = conn.execute("""
            SELECT s.value AS name,
                   COUNT(*)       AS count,
                   SUM(r.end-r.start) AS total_ns
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON r.nameId = s.id
            GROUP BY s.value
            ORDER BY total_ns DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        # Fallback: try cbid column, or nameId without join
        try:
            cur = conn.execute("""
                SELECT COALESCE(nameId, cbid) AS name,
                       COUNT(*)       AS count,
                       SUM(end-start) AS total_ns
                FROM CUPTI_ACTIVITY_KIND_RUNTIME
                GROUP BY name
                ORDER BY total_ns DESC
                LIMIT 10
            """)
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return

    if not rows:
        return

    total_api_ns = sum(r["total_ns"] for r in rows)
    api_calls = []
    for r in rows:
        api_calls.append({
            "name": str(r["name"]),
            "count": r["count"],
            "total_ms": r["total_ns"] / 1e6,
        })

    result["top_api_calls"] = api_calls
    result["api_overhead_ms"] = total_api_ns / 1e6


def _extract_gpu_utilization(conn: sqlite3.Connection, result: dict) -> None:
    """GPU active percentage = union of kernel busy intervals / trace span.

    Kernels can run concurrently on multiple streams, so summing per-kernel
    durations double-counts overlap (and can exceed 100%). The GPU is
    "active" whenever at least one kernel is running, i.e. the interval union.
    """
    rows = conn.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    intervals = [(r["start"], r["end"]) for r in rows
                 if r["start"] is not None and r["end"] is not None]
    if not intervals:
        return

    busy_ns = union_duration(intervals)
    span_ns = max(e for _, e in intervals) - min(s for s, _ in intervals)
    if span_ns > 0:
        result["gpu_active_pct"] = round(busy_ns / span_ns * 100.0, 1)


def _extract_kernel_gaps(conn: sqlite3.Connection, result: dict) -> None:
    """Average and max gap between consecutive kernel launches."""
    cur = conn.execute("""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start
    """)
    rows = cur.fetchall()
    if len(rows) < 2:
        return

    gaps_ns: list[int] = []
    prev_end = rows[0]["end"]
    for r in rows[1:]:
        gap = r["start"] - prev_end
        if gap > 0:
            gaps_ns.append(gap)
        prev_end = max(prev_end, r["end"])

    if gaps_ns:
        result["avg_kernel_gap_us"] = round(sum(gaps_ns) / len(gaps_ns) / 1e3, 2)
        result["max_kernel_gap_us"] = round(max(gaps_ns) / 1e3, 2)


def _extract_nvtx_ranges(conn: sqlite3.Connection, result: dict) -> None:
    """Top NVTX annotation ranges by duration, with timestamps for temporal matching."""
    # Schema varies across nsys versions; try common table names
    for table in ("NVTX_EVENTS", "NVTX_RANGES"):
        try:
            cur = conn.execute(f"""
                SELECT text, start, end, (end-start) AS duration_ns
                FROM {table}
                WHERE end > start
                ORDER BY duration_ns DESC
                LIMIT 50
            """)
            rows = cur.fetchall()
            break
        except sqlite3.OperationalError:
            rows = []

    if not rows:
        return

    total_ns = sum(r["duration_ns"] for r in rows[:10])
    ranges = []
    for r in rows:
        pct = (r["duration_ns"] / total_ns * 100.0) if total_ns > 0 else 0.0
        ranges.append({
            "name": r["text"] or "(unnamed)",
            "duration_ms": r["duration_ns"] / 1e6,
            "pct": round(pct, 1),
            "start_ns": r["start"],
            "end_ns": r["end"],
        })

    result["nvtx_ranges"] = ranges


def _extract_kernel_launch_dims(conn: sqlite3.Connection, result: dict) -> None:
    """Enrich top_kernels with average grid/block dimensions."""
    try:
        cur = conn.execute("""
            SELECT demangledName,
                   gridX, gridY, gridZ,
                   blockX, blockY, blockZ,
                   (end-start) AS duration_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            ORDER BY duration_ns DESC
            LIMIT 20
        """)
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return

    if not rows:
        return

    # Aggregate per-kernel
    kernel_dims: dict[str, dict] = {}
    for r in rows:
        name = r["demangledName"] or "(unknown)"
        if name not in kernel_dims:
            kernel_dims[name] = {"grid_sizes": [], "block_sizes": [], "threads": []}
        grid_size = (r["gridX"] or 1) * (r["gridY"] or 1) * (r["gridZ"] or 1)
        block_size = (r["blockX"] or 1) * (r["blockY"] or 1) * (r["blockZ"] or 1)
        kernel_dims[name]["grid_sizes"].append(grid_size)
        kernel_dims[name]["block_sizes"].append(block_size)
        kernel_dims[name]["threads"].append(grid_size * block_size)

    # Enrich existing top_kernels entries
    top_kernels = result.get("top_kernels", [])
    for entry in top_kernels:
        dims = kernel_dims.get(entry["name"])
        if dims:
            entry["avg_grid_size"] = round(sum(dims["grid_sizes"]) / len(dims["grid_sizes"]), 1)
            entry["avg_block_size"] = round(sum(dims["block_sizes"]) / len(dims["block_sizes"]), 1)
            entry["avg_threads_per_launch"] = round(sum(dims["threads"]) / len(dims["threads"]), 1)


def _extract_cuda_sync_overhead(conn: sqlite3.Connection, result: dict) -> None:
    """Track cudaDeviceSynchronize / cudaStreamSynchronize overhead."""
    try:
        cur = conn.execute("""
            SELECT s.value AS name, COUNT(*) AS count, SUM(r.end-r.start) AS total_ns
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON r.nameId = s.id
            WHERE s.value LIKE '%Synchronize%' OR s.value LIKE '%StreamSync%'
            GROUP BY s.value
        """)
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        # Fallback without StringIds join
        return

    if not rows:
        return

    sync_calls = []
    total_sync_ns = 0
    for r in rows:
        ns = r["total_ns"] or 0
        total_sync_ns += ns
        sync_calls.append({
            "name": str(r["name"]),
            "count": r["count"],
            "total_ms": ns / 1e6,
        })

    result["sync_calls"] = sync_calls
    result["total_sync_ms"] = total_sync_ns / 1e6


# ---------------------------------------------------------------------------
# Legacy regex-based parser (fallback when SQLite export is unavailable)
# ---------------------------------------------------------------------------

def _parse_nsys_stats(stdout: str) -> dict:
    """Extract CUDA kernel time, memcpy time, API overhead from nsys stats output."""
    result: dict = {}

    # Look for summary sections in nsys output
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        # CUDA Kernel Statistics
        if "CUDA Kernel Statistics" in line or "kern" in line.lower():
            # Try to extract total kernel time from nearby lines
            for j in range(i + 1, min(i + 20, len(lines))):
                m = re.search(r"Total\s*[:\s]+([\d.]+)\s*(ms|us|ns|s)", lines[j], re.IGNORECASE)
                if m:
                    val = float(m.group(1))
                    unit = m.group(2).lower()
                    result["cuda_kernel_time_ms"] = _to_ms(val, unit)
                    break

        # Memory copy statistics
        if "memcpy" in line.lower() or "CUDA Memory Operation" in line:
            for j in range(i + 1, min(i + 20, len(lines))):
                m = re.search(r"Total\s*[:\s]+([\d.]+)\s*(ms|us|ns|s)", lines[j], re.IGNORECASE)
                if m:
                    val = float(m.group(1))
                    unit = m.group(2).lower()
                    result["memcpy_time_ms"] = _to_ms(val, unit)
                    break

        # API overhead
        if "CUDA API" in line:
            for j in range(i + 1, min(i + 20, len(lines))):
                m = re.search(r"Total\s*[:\s]+([\d.]+)\s*(ms|us|ns|s)", lines[j], re.IGNORECASE)
                if m:
                    val = float(m.group(1))
                    unit = m.group(2).lower()
                    result["api_overhead_ms"] = _to_ms(val, unit)
                    break

    return result


def _to_ms(val: float, unit: str) -> float:
    """Convert a time value to milliseconds."""
    if unit == "s":
        return val * 1000.0
    if unit == "ms":
        return val
    if unit == "us":
        return val / 1000.0
    if unit == "ns":
        return val / 1_000_000.0
    return val


# ---------------------------------------------------------------------------
# CPU→GPU correlation and per-stream analysis
# ---------------------------------------------------------------------------

def _extract_cpu_gpu_correlation(conn: sqlite3.Connection, result: dict) -> None:
    """Link CPU CUDA API calls to GPU kernel launches via correlationId.

    Works across all CUDA-based frameworks (PyTorch, Triton, JAX/XLA, raw CUDA)
    because they all go through cudaLaunchKernel at the CUDA runtime layer.
    """
    try:
        cur = conn.execute("""
            SELECT
                r.correlationId,
                s.value AS api_name,
                r.start AS cpu_start,
                r.end   AS cpu_end,
                k.demangledName AS kernel_name,
                k.start AS gpu_start,
                k.end   AS gpu_end,
                k.streamId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON r.nameId = s.id
            JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON r.correlationId = k.correlationId
            WHERE s.value LIKE 'cudaLaunch%'
            ORDER BY r.start
        """)
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return

    if not rows:
        return

    correlations = []
    for r in rows:
        gpu_start = r["gpu_start"]
        gpu_end = r["gpu_end"]
        cpu_start = r["cpu_start"]
        cpu_end = r["cpu_end"]
        gpu_dur = gpu_end - gpu_start
        launch_overhead = gpu_start - cpu_start

        correlations.append({
            "correlation_id": r["correlationId"],
            "api_name": r["api_name"],
            "kernel_name": r["kernel_name"] or "(unknown)",
            "cpu_start_ns": cpu_start,
            "cpu_end_ns": cpu_end,
            "gpu_start_ns": gpu_start,
            "gpu_end_ns": gpu_end,
            "gpu_duration_ns": gpu_dur,
            "launch_overhead_ns": max(0, launch_overhead),
            "stream_id": r["streamId"],
        })

    result["cpu_gpu_correlations"] = correlations


def _extract_callchain_context(conn: sqlite3.Connection, result: dict) -> None:
    """Walk CPU call chains from cudaLaunchKernel up to user-code frames.

    NSys records the host call stack at each CUDA runtime call via callchainId.
    We walk up the chain to find the first frame that isn't in CUDA runtime,
    driver, or framework internals — that's the user code that triggered the
    kernel launch.

    Enriches cpu_gpu_correlations with 'caller_function' and 'caller_module'.
    """
    correlations = result.get("cpu_gpu_correlations")
    if not correlations:
        return

    # Check which call chain tables exist (schema varies across nsys versions)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    # We need both a callchain mapping and a symbol table
    has_callchain = "CUPTI_ACTIVITY_KIND_RUNTIME" in tables
    has_symbols = "StringIds" in tables

    if not has_callchain or not has_symbols:
        return

    # Try to get callchainId from RUNTIME events — not all nsys versions have it
    try:
        test = conn.execute(
            "SELECT callchainId FROM CUPTI_ACTIVITY_KIND_RUNTIME LIMIT 1"
        ).fetchone()
        if test is None or test[0] is None:
            return
    except sqlite3.OperationalError:
        return

    # Build correlation_id -> callchainId lookup
    try:
        cur = conn.execute("""
            SELECT correlationId, callchainId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE callchainId IS NOT NULL
        """)
        corr_to_chain: dict[int, int] = {}
        for r in cur.fetchall():
            corr_to_chain[r["correlationId"]] = r["callchainId"]
    except sqlite3.OperationalError:
        return

    if not corr_to_chain:
        return

    # Load call chain entries — try common table names across nsys versions
    chain_entries: dict[int, list[dict]] = {}
    for chain_table, _symbol_col in [
        ("CallchainIds", "symbol"),
        ("CALLCHAIN", "symbol"),
    ]:
        if chain_table not in tables:
            continue
        try:
            # Get column names to determine schema
            col_info = conn.execute(f"PRAGMA table_info({chain_table})").fetchall()
            col_names = {c["name"] for c in col_info}

            if "symbolName" in col_names:
                # Newer nsys: symbolName is directly in the table
                cur = conn.execute(f"""
                    SELECT id, symbolName AS symbol, module
                    FROM {chain_table}
                    ORDER BY id
                """)
            elif "nameId" in col_names and "StringIds" in tables:
                # Older nsys: join with StringIds
                cur = conn.execute(f"""
                    SELECT c.id, s.value AS symbol,
                           COALESCE(m.value, '') AS module
                    FROM {chain_table} c
                    LEFT JOIN StringIds s ON c.nameId = s.id
                    LEFT JOIN StringIds m ON c.moduleId = m.id
                    ORDER BY c.id
                """)
            else:
                continue

            for r in cur.fetchall():
                chain_id = r["id"]
                chain_entries.setdefault(chain_id, []).append({
                    "symbol": r["symbol"] or "",
                    "module": r.get("module") or "",
                })
            if chain_entries:
                break
        except sqlite3.OperationalError:
            continue

    if not chain_entries:
        return

    # Patterns indicating internal frames to skip
    _INTERNAL_PREFIXES = (
        "cuda", "libcuda", "libcudart", "libnvidia",
        "libcublas", "libcublasLt", "libcudnn", "libcufft",
        "libnccl", "libcutlass",
        "c10::", "at::", "torch::", "THC",  # PyTorch internals
        "pybind11::", "_PyCFunction",
        "libtorch", "libc10",
    )
    _INTERNAL_MODULES = (
        "libcuda", "libcudart", "libnvidia", "libcublas",
        "libc10", "libtorch", "libgomp", "libstdc++",
    )

    def _is_internal(symbol: str, module: str) -> bool:
        s = symbol.lower()
        m = module.lower()
        for prefix in _INTERNAL_PREFIXES:
            if s.startswith(prefix.lower()):
                return True
        for mod in _INTERNAL_MODULES:
            if mod.lower() in m:
                return True
        return False

    # Enrich each correlation with caller info
    for corr in correlations:
        chain_id = corr_to_chain.get(corr["correlation_id"])
        if chain_id is None:
            continue

        frames = chain_entries.get(chain_id, [])
        for frame in frames:
            symbol = frame["symbol"]
            module = frame.get("module", "")
            if symbol and not _is_internal(symbol, module):
                corr["caller_function"] = symbol
                corr["caller_module"] = module
                break


def _extract_per_stream_gaps(conn: sqlite3.Connection, result: dict) -> None:
    """Compute kernel gaps per CUDA stream for pipeline stall detection."""
    try:
        cur = conn.execute("""
            SELECT streamId, start, end
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            ORDER BY streamId, start
        """)
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return

    if len(rows) < 2:
        return

    # Group by stream
    streams: dict[int, list[tuple[int, int]]] = {}
    for r in rows:
        sid = r["streamId"]
        streams.setdefault(sid, []).append((r["start"], r["end"]))

    per_stream_gaps: dict[int, dict] = {}
    stream_utilization: dict[int, dict] = {}

    for sid, intervals in streams.items():
        if not intervals:
            continue

        intervals.sort()
        gaps_ns: list[int] = []
        total_kernel_ns = 0
        prev_end = intervals[0][1]
        total_kernel_ns += intervals[0][1] - intervals[0][0]

        for start, end in intervals[1:]:
            gap = start - prev_end
            if gap > 0:
                gaps_ns.append(gap)
            total_kernel_ns += end - start
            prev_end = max(prev_end, end)

        span_ns = intervals[-1][1] - intervals[0][0]

        if gaps_ns:
            per_stream_gaps[sid] = {
                "avg_gap_us": round(sum(gaps_ns) / len(gaps_ns) / 1e3, 2),
                "max_gap_us": round(max(gaps_ns) / 1e3, 2),
                "num_gaps": len(gaps_ns),
                "total_gap_ms": round(sum(gaps_ns) / 1e6, 2),
                "num_kernels": len(intervals),
            }

        if span_ns > 0:
            stream_utilization[sid] = {
                "active_pct": round(total_kernel_ns / span_ns * 100.0, 1),
                "kernel_count": len(intervals),
                "total_kernel_ms": round(total_kernel_ns / 1e6, 2),
            }

    if per_stream_gaps:
        result["per_stream_gaps"] = per_stream_gaps
    if stream_utilization:
        result["stream_utilization"] = stream_utilization
