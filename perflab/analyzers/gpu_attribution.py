"""GPU attribution engine: CPU→GPU call graph, edge weighting, unified ranking.

Builds a weighted call graph linking CPU-side CUDA API calls to GPU kernel
launches via correlationId from NSys SQLite. Works across all CUDA-based
frameworks (PyTorch, Triton, JAX/XLA, raw CUDA) because they all go through
cudaLaunchKernel at the CUDA runtime layer.

Attribution linking strategies (in priority order):
1. correlationId join (NSys SQLite) — exact CPU API → GPU kernel mapping
2. Call chain walking (NSys callchainId) — user-code caller above cudaLaunchKernel
3. Temporal NVTX matching — kernel start within NVTX range time window
4. Torch trace cross-reference — PyTorch operator → kernel by timestamp overlap
5. Py-spy temporal join — Python function on CPU during cudaLaunchKernel call
"""
from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CpuGpuEdge:
    """An edge in the CPU→GPU call graph."""
    api_name: str               # CPU-side API call
    kernel_name: str            # GPU kernel launched
    stream_id: int
    count: int                  # number of launches
    total_gpu_ms: float         # total GPU time attributed to this edge
    avg_launch_overhead_us: float  # avg time from CPU call to GPU execution start
    pct_of_total_gpu: float     # % of total GPU time
    framework_op: str | None = None  # enriched: "aten::matmul", "triton_fused_relu", etc.
    caller_function: str | None = None  # user-code function from call chain walking


@dataclass
class AttributionEntry:
    """A ranked attribution entry combining CPU and GPU evidence."""
    rank: int
    category: str       # "gpu-kernel", "cpu-hotspot", "launch-overhead", "pipeline-stall", "transfer"
    name: str           # kernel name, function name, or stream id
    gpu_time_ms: float
    gpu_pct: float      # % of total GPU time
    cpu_pct: float | None = None       # % of CPU samples (from perf hotspots), if linkable
    launch_overhead_us: float | None = None
    stream_id: int | None = None
    caller_function: str | None = None  # user-code function that triggered this kernel
    framework_op: str | None = None     # framework-level op (e.g. aten::mm)
    diagnosis: str = ""      # human-readable description
    suggestions: list[str] = field(default_factory=list)


def build_cpu_gpu_call_graph(correlations: list[dict]) -> list[CpuGpuEdge]:
    """Build CPU→GPU call graph from correlation data.

    Groups correlations by (api_name, kernel_name, stream_id) and computes
    aggregate stats per edge, sorted by total GPU time descending.
    Propagates caller_function from call chain walking when available.
    """
    if not correlations:
        return []

    # Group by (api_name, kernel_name, stream_id)
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for c in correlations:
        key = (c["api_name"], c["kernel_name"], c.get("stream_id", 0))
        groups.setdefault(key, []).append(c)

    total_gpu_ns = sum(c.get("gpu_duration_ns", 0) for c in correlations)
    if total_gpu_ns <= 0:
        total_gpu_ns = 1  # avoid division by zero

    edges: list[CpuGpuEdge] = []
    for (api_name, kernel_name, stream_id), items in groups.items():
        count = len(items)
        total_ns = sum(it.get("gpu_duration_ns", 0) for it in items)
        overhead_ns = [it.get("launch_overhead_ns", 0) for it in items]
        avg_overhead_us = (sum(overhead_ns) / count / 1000.0) if count > 0 else 0.0

        # Pick the most common caller_function across items in this group
        caller = _most_common_caller(items)

        edges.append(CpuGpuEdge(
            api_name=api_name,
            kernel_name=kernel_name,
            stream_id=stream_id,
            count=count,
            total_gpu_ms=total_ns / 1e6,
            avg_launch_overhead_us=avg_overhead_us,
            pct_of_total_gpu=total_ns / total_gpu_ns * 100.0,
            caller_function=caller,
        ))

    edges.sort(key=lambda e: e.total_gpu_ms, reverse=True)
    return edges


def _most_common_caller(items: list[dict]) -> str | None:
    """Return the most frequently occurring caller_function across correlation items."""
    counts: dict[str, int] = {}
    for it in items:
        caller = it.get("caller_function")
        if caller:
            counts[caller] = counts.get(caller, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)  # type: ignore[arg-type]


def compute_attribution_ranking(
    nsys_summary: dict,
    perf_summary: dict | None = None,
    torch_summary: dict | None = None,
    pyspy_summary: dict | None = None,
) -> list[AttributionEntry]:
    """Compute unified attribution ranking combining GPU and CPU evidence.

    Uses multiple linking strategies:
    1. correlationId call graph (exact CPU API → GPU kernel)
    2. Call chain walking (user-code caller from NSys callchainId)
    3. Temporal NVTX matching (kernel within NVTX range time window)
    4. Torch trace cross-reference (operator → kernel by timestamp overlap)
    5. Py-spy temporal join (Python function on CPU during kernel launch)

    Ranking heuristic:
        score = (gpu_pct * 2.0)       -- GPU time is the primary optimization target (2x weight)
              + (cpu_pct * 0.5)        -- CPU evidence is corroborating signal only (0.5x weight)
              + stall_penalty + overhead_penalty
    """
    entries: list[AttributionEntry] = []

    correlations = nsys_summary.get("cpu_gpu_correlations", [])
    top_kernels = nsys_summary.get("top_kernels", [])
    per_stream_gaps = nsys_summary.get("per_stream_gaps", {})
    stream_util = nsys_summary.get("stream_utilization", {})
    nvtx_ranges = nsys_summary.get("nvtx_ranges", [])

    # Build call graph (includes caller_function from call chain walking)
    edges = build_cpu_gpu_call_graph(correlations) if correlations else []

    # CPU hotspot lookup (linux perf)
    cpu_hotspots: dict[str, float] = {}
    if perf_summary:
        for h in perf_summary.get("hotspots", []):
            cpu_hotspots[h.get("function", "")] = h.get("pct", 0)

    # Build temporal indices for cross-referencing
    nvtx_temporal = _build_nvtx_temporal_index(nvtx_ranges)
    torch_op_temporal = _build_torch_op_temporal_index(torch_summary)
    pyspy_temporal = _build_pyspy_temporal_index(pyspy_summary)

    # Build kernel → correlation timestamp lookup for temporal matching
    kernel_timestamps: dict[str, list[tuple[int, int]]] = {}
    for c in correlations:
        kname = c.get("kernel_name", "")
        if kname:
            kernel_timestamps.setdefault(kname, []).append(
                (c.get("cpu_start_ns", 0), c.get("gpu_start_ns", 0))
            )

    # 1. GPU kernel dominance from top_kernels
    for kernel in top_kernels:
        gpu_pct = kernel.get("pct", 0)
        total_ms = kernel.get("total_ms", 0)
        name = kernel.get("name", "(unknown)")

        # Find matching edge for overhead info + caller
        overhead_us: float | None = None
        stream_id: int | None = None
        caller_function: str | None = None
        framework_op: str | None = None
        for edge in edges:
            if edge.kernel_name == name:
                overhead_us = edge.avg_launch_overhead_us
                stream_id = edge.stream_id
                caller_function = edge.caller_function
                framework_op = edge.framework_op
                break

        # Strategy 1: CPU match via fuzzy name matching (original approach)
        cpu_pct: float | None = None
        for func, pct in cpu_hotspots.items():
            if _fuzzy_match(func, name):
                cpu_pct = pct
                break

        # Strategy 2: CPU match via call chain caller (if fuzzy failed)
        if cpu_pct is None and caller_function:
            for func, pct in cpu_hotspots.items():
                if _fuzzy_match(func, caller_function):
                    cpu_pct = pct
                    break

        # Strategy 3: Temporal NVTX matching (if no framework_op yet)
        if framework_op is None and nvtx_temporal:
            timestamps = kernel_timestamps.get(name, [])
            framework_op = _match_kernel_to_nvtx_temporal(timestamps, nvtx_temporal)

        # Strategy 4: Torch trace cross-reference
        if framework_op is None and torch_op_temporal:
            timestamps = kernel_timestamps.get(name, [])
            framework_op = _match_kernel_to_torch_op_temporal(timestamps, torch_op_temporal)

        # Strategy 5: Py-spy temporal join (Python function during launch)
        pyspy_caller: str | None = None
        if pyspy_temporal:
            timestamps = kernel_timestamps.get(name, [])
            pyspy_caller = _match_kernel_to_pyspy_temporal(timestamps, pyspy_temporal)
            # Use py-spy caller if we don't have one from call chain
            if caller_function is None and pyspy_caller:
                caller_function = pyspy_caller

        # Score
        # 50μs launch overhead threshold: typical cudaLaunchKernel takes ~5-10μs;
        # >50μs indicates driver/runtime contention worth flagging for CUDA graphs.
        # 5.0 bonus: enough to promote a kernel with overhead above non-overhead entries at similar gpu_pct
        overhead_penalty = 5.0 if overhead_us and overhead_us > 50 else 0.0
        # gpu_pct * 2.0: GPU time is the primary signal; cpu_pct * 0.5: corroborating CPU evidence
        score = (gpu_pct * 2.0) + ((cpu_pct or 0) * 0.5) + overhead_penalty

        suggestions: list[str] = []
        diagnosis = f"Kernel '{name}' consumes {gpu_pct:.0f}% of GPU time ({total_ms:.1f} ms)"

        if framework_op:
            diagnosis += f" [from {framework_op}]"
        if caller_function:
            diagnosis += f" [called by {caller_function}]"

        # 20%: kernel consuming >20% of GPU time warrants detailed ncu profiling
        if gpu_pct > 20:
            suggestions.append("Profile with ncu for detailed kernel analysis")
        if overhead_us and overhead_us > 50:
            suggestions.append(f"Launch overhead is {overhead_us:.0f} us — consider CUDA graphs")
            diagnosis += f" with {overhead_us:.0f} us launch overhead"
        if cpu_pct:
            diagnosis += f"; CPU function matches at {cpu_pct:.0f}% of CPU samples"

        entries.append(AttributionEntry(
            rank=0,  # assigned later
            category="gpu-kernel",
            name=name,
            gpu_time_ms=total_ms,
            gpu_pct=gpu_pct,
            cpu_pct=cpu_pct,
            launch_overhead_us=overhead_us,
            stream_id=stream_id,
            caller_function=caller_function,
            framework_op=framework_op,
            diagnosis=diagnosis,
            suggestions=suggestions,
        ))

    # 2. Pipeline stall entries
    stall_entries = detect_pipeline_stalls(per_stream_gaps, stream_util)
    entries.extend(stall_entries)

    # 3. Transfer bottleneck entries
    memcpy = nsys_summary.get("memcpy", [])
    kernel_time_ms = nsys_summary.get("cuda_kernel_time_ms", 0)
    for mc in memcpy:
        total_ms = mc.get("total_ms", 0)
        direction = mc.get("direction", "unknown")
        # 0.1: transfer >10% of kernel time is significant — data movement vs compute imbalance
        if kernel_time_ms > 0 and total_ms / kernel_time_ms > 0.1:
            transfer_pct = total_ms / kernel_time_ms * 100
            entries.append(AttributionEntry(
                rank=0,
                category="transfer",
                name=f"memcpy_{direction}",
                gpu_time_ms=total_ms,
                gpu_pct=0,
                diagnosis=f"{direction} transfers take {total_ms:.1f} ms ({transfer_pct:.0f}% of kernel time)",
                suggestions=["Use pinned memory (cudaMallocHost)", "Use async transfers (cudaMemcpyAsync)"],
            ))

    # Sort by score and assign ranks
    def _score(e: AttributionEntry) -> float:
        # gpu_pct * 2.0: GPU time is primary ranking signal; cpu_pct * 0.5: secondary CPU evidence
        base = (e.gpu_pct * 2.0) + ((e.cpu_pct or 0) * 0.5)
        overhead = 5.0 if e.launch_overhead_us and e.launch_overhead_us > 50 else 0
        # Boost entries with richer attribution — caller/framework info makes
        # them more actionable for the LLM (can reference user code directly).
        # +2.0 for caller is larger than +1.0 for framework_op because knowing
        # "train_step() triggers this kernel" is more actionable than "aten::mm".
        attribution_bonus = 2.0 if e.caller_function else 0
        attribution_bonus += 1.0 if e.framework_op else 0
        return base + overhead + attribution_bonus

    entries.sort(key=_score, reverse=True)
    for i, entry in enumerate(entries):
        entry.rank = i + 1

    return entries


def enrich_with_framework_context(
    edges: list[CpuGpuEdge],
    nvtx_ranges: list[dict] | None = None,
    correlations: list[dict] | None = None,
    program_type: str = "cuda",
) -> list[CpuGpuEdge]:
    """Enrich call graph edges with framework-level context.

    Uses temporal NVTX matching (kernel launch within NVTX range time window)
    before falling back to name-based heuristics.
    """
    # Build temporal index from NVTX ranges
    nvtx_temporal = _build_nvtx_temporal_index(nvtx_ranges) if nvtx_ranges else []

    # Build kernel → timestamps lookup from correlations
    kernel_timestamps: dict[str, list[tuple[int, int]]] = {}
    if correlations:
        for c in correlations:
            kname = c.get("kernel_name", "")
            if kname:
                kernel_timestamps.setdefault(kname, []).append(
                    (c.get("cpu_start_ns", 0), c.get("gpu_start_ns", 0))
                )

    for edge in edges:
        # Strategy 1: Temporal NVTX matching (preferred)
        if nvtx_temporal:
            timestamps = kernel_timestamps.get(edge.kernel_name, [])
            temporal_match = _match_kernel_to_nvtx_temporal(timestamps, nvtx_temporal)
            if temporal_match:
                edge.framework_op = temporal_match
                continue

        # Strategy 2: Name-based NVTX matching (fallback)
        if nvtx_ranges:
            for nvtx in nvtx_ranges:
                name = nvtx.get("name", "")
                if name and name != "(unnamed)":
                    if _nvtx_matches_kernel(name, edge.kernel_name):
                        edge.framework_op = name
                        break

        # Strategy 3: Kernel name heuristics (final fallback)
        if edge.framework_op is None:
            edge.framework_op = _infer_framework_op(edge.kernel_name, program_type)

    return edges


def detect_pipeline_stalls(
    per_stream_gaps: dict,
    stream_utilization: dict,
) -> list[AttributionEntry]:
    """Detect per-stream pipeline stalls from gap analysis."""
    entries: list[AttributionEntry] = []

    if not per_stream_gaps and not stream_utilization:
        return entries

    for stream_id_str, util in (stream_utilization or {}).items():
        stream_id = int(stream_id_str) if isinstance(stream_id_str, str) else stream_id_str
        active_pct = util.get("active_pct", 100)
        kernel_count = util.get("kernel_count", 0)

        gaps = (per_stream_gaps or {}).get(stream_id_str, {})
        if not gaps:
            gaps = (per_stream_gaps or {}).get(stream_id, {})
        max_gap_us = gaps.get("max_gap_us", 0)

        # 50% active: stream is idle more than busy — clear pipeline stall.
        # 100μs gap: kernel launch is ~5-20μs; >100μs between kernels = pipeline bubble.
        if active_pct < 50 and max_gap_us > 100:
            entries.append(AttributionEntry(
                rank=0,
                category="pipeline-stall",
                name=f"stream_{stream_id}",
                gpu_time_ms=0,
                gpu_pct=0,
                stream_id=stream_id,
                diagnosis=(
                    f"Stream {stream_id} is idle {100 - active_pct:.0f}% of the time "
                    f"(max gap: {max_gap_us:.0f} us, {kernel_count} kernels)"
                ),
                suggestions=[
                    "Overlap computation with other streams",
                    "Fuse small kernels to reduce gaps",
                    "Use CUDA graphs for repeated launch patterns",
                ],
            ))

    # Detect multi-stream serialization
    if len(stream_utilization or {}) > 1:
        utils = list((stream_utilization or {}).values())
        active_pcts = [u.get("active_pct", 0) for u in utils]
        # 40pp spread: streams with >40% utilization gap indicate serialization or load imbalance
        if active_pcts and max(active_pcts) - min(active_pcts) > 40:
            entries.append(AttributionEntry(
                rank=0,
                category="pipeline-stall",
                name="multi-stream-serialization",
                gpu_time_ms=0,
                gpu_pct=0,
                diagnosis=(
                    f"Multi-stream imbalance detected: utilization ranges from "
                    f"{min(active_pcts):.0f}% to {max(active_pcts):.0f}%"
                ),
                suggestions=[
                    "Balance work across streams",
                    "Check for implicit synchronization between streams",
                ],
            ))

    return entries


# ---------------------------------------------------------------------------
# Temporal matching helpers
# ---------------------------------------------------------------------------

def _build_nvtx_temporal_index(
    nvtx_ranges: list[dict] | None,
) -> list[tuple[int, int, str]]:
    """Build sorted list of (start_ns, end_ns, name) from NVTX ranges.

    Only includes ranges that have timestamps (from updated nsys extraction).
    Sorted by start time for efficient temporal matching.
    """
    if not nvtx_ranges:
        return []
    index: list[tuple[int, int, str]] = []
    for r in nvtx_ranges:
        start = r.get("start_ns")
        end = r.get("end_ns")
        name = r.get("name", "")
        if start is not None and end is not None and name and name != "(unnamed)":
            index.append((int(start), int(end), name))
    index.sort()
    return index


def _build_torch_op_temporal_index(
    torch_summary: dict | None,
) -> list[tuple[int, int, str]]:
    """Build temporal index from PyTorch profiler trace operator events.

    Uses the raw Chrome trace events (cpu_op category) with their timestamps.
    The torch profiler records operator start/duration in microseconds.
    """
    if not torch_summary:
        return []

    # The torch trace stores raw events with ts (microseconds) and dur
    raw_events = torch_summary.get("_raw_cpu_ops")
    if not raw_events:
        return []

    index: list[tuple[int, int, str]] = []
    for ev in raw_events:
        name = ev.get("name", "")
        ts_us = ev.get("ts", 0)
        dur_us = ev.get("dur", 0)
        if name and dur_us > 0:
            # Convert microseconds to nanoseconds to match nsys timestamps
            start_ns = int(ts_us * 1000)
            end_ns = int((ts_us + dur_us) * 1000)
            index.append((start_ns, end_ns, name))
    index.sort()
    return index


def _build_pyspy_temporal_index(
    pyspy_summary: dict | None,
) -> list[tuple[int, int, str]]:
    """Build temporal index from py-spy speedscope samples.

    Each sample has a timestamp and the Python function that was on the CPU.
    """
    if not pyspy_summary:
        return []

    samples = pyspy_summary.get("timed_samples")
    if not samples:
        return []

    index: list[tuple[int, int, str]] = []
    for s in samples:
        ts_ns = s.get("ts_ns", 0)
        dur_ns = s.get("dur_ns", 0)
        func = s.get("function", "")
        if func and ts_ns > 0:
            # If no duration, assume sample interval (~10ms default for py-spy)
            if dur_ns <= 0:
                dur_ns = 10_000_000  # 10ms default
            index.append((ts_ns, ts_ns + dur_ns, func))
    index.sort()
    return index


def _find_best_enclosing(
    query: int,
    index: list[tuple[int, int, str]],
) -> str | None:
    """Find the shortest interval in *index* that contains *query*.

    *index* must be sorted by start time (element 0). Uses bisect to skip
    intervals that start after the query, then scans backwards with an early
    exit: once the gap ``query - start`` exceeds the current best duration,
    no earlier interval can be shorter, so we stop.

    O(log n) bisect + typically O(1-3) backward scan for non-overlapping
    intervals; O(k) for k nested intervals (rare, and k is usually small).
    """
    if not index:
        return None

    starts = [s for s, _, _ in index]
    pos = bisect.bisect_right(starts, query)

    best_name: str | None = None
    best_duration = float("inf")

    for i in range(pos - 1, -1, -1):
        s, e, name = index[i]
        if e >= query:
            duration = e - s
            if duration < best_duration:
                best_duration = duration
                best_name = name
        # Intervals starting further left can only be wider than (query - s).
        # If that already exceeds our best, no earlier interval can win.
        if best_name is not None and (query - s) > best_duration:
            break

    return best_name


def _match_kernel_to_nvtx_temporal(
    kernel_timestamps: list[tuple[int, int]],
    nvtx_index: list[tuple[int, int, str]],
) -> str | None:
    """Find the NVTX range that temporally contains a kernel launch.

    A kernel is attributed to an NVTX range if the cudaLaunchKernel CPU-side
    call (cpu_start_ns) falls within the NVTX range's [start, end] window.
    Returns the most specific (shortest duration) matching range.
    """
    if not kernel_timestamps or not nvtx_index:
        return None
    cpu_start, _ = kernel_timestamps[0]
    if cpu_start <= 0:
        return None
    return _find_best_enclosing(cpu_start, nvtx_index)


def _match_kernel_to_torch_op_temporal(
    kernel_timestamps: list[tuple[int, int]],
    torch_op_index: list[tuple[int, int, str]],
) -> str | None:
    """Find the PyTorch operator that temporally contains a kernel launch.

    Note: torch trace timestamps and nsys timestamps use different clocks,
    so we match by relative position within the trace rather than absolute
    time when clocks diverge significantly.
    """
    if not kernel_timestamps or not torch_op_index:
        return None
    cpu_start, _ = kernel_timestamps[0]
    if cpu_start <= 0:
        return None
    return _find_best_enclosing(cpu_start, torch_op_index)


def _match_kernel_to_pyspy_temporal(
    kernel_timestamps: list[tuple[int, int]],
    pyspy_index: list[tuple[int, int, str]],
) -> str | None:
    """Find the Python function that was on the CPU when a kernel was launched.

    Py-spy samples at ~100Hz, so each sample covers ~10ms. A kernel launch
    is attributed to a py-spy sample if the cudaLaunchKernel CPU timestamp
    falls within the sample's time window.
    """
    if not kernel_timestamps or not pyspy_index:
        return None
    cpu_start, _ = kernel_timestamps[0]
    if cpu_start <= 0:
        return None
    return _find_best_enclosing(cpu_start, pyspy_index)


# ---------------------------------------------------------------------------
# Name-based helpers
# ---------------------------------------------------------------------------

def _fuzzy_match(cpu_func: str, kernel_name: str) -> bool:
    """Fuzzy match CPU function name against GPU kernel name."""
    cpu_lower = cpu_func.lower()
    kernel_lower = kernel_name.lower()

    # Direct substring
    if cpu_lower in kernel_lower or kernel_lower in cpu_lower:
        return True

    # Extract base name (strip template args, namespaces)
    cpu_base = cpu_lower.split("::")[-1].split("<")[0].split("(")[0]
    kernel_base = kernel_lower.split("::")[-1].split("<")[0].split("(")[0]

    if cpu_base and kernel_base and (cpu_base in kernel_base or kernel_base in cpu_base):
        return True

    return False


def _nvtx_matches_kernel(nvtx_name: str, kernel_name: str) -> bool:
    """Check if an NVTX annotation is related to a kernel."""
    nvtx_lower = nvtx_name.lower()
    kernel_lower = kernel_name.lower()

    # PyTorch NVTX ranges like "aten::mm" matching kernel "volta_sgemm"
    # These are not directly matchable by name, but temporal overlap is used
    # For now, check direct substring match
    nvtx_base = nvtx_lower.replace("aten::", "").replace("torch.", "")
    kernel_base = kernel_lower.split("::")[-1].split("<")[0]

    return nvtx_base in kernel_base or kernel_base in nvtx_base


def _infer_framework_op(kernel_name: str, program_type: str) -> str | None:
    """Infer framework operation from kernel name heuristics."""
    name = kernel_name.lower()

    if program_type == "pytorch":
        # Triton-compiled kernels in torch.compile
        if "triton_" in name:
            # e.g. triton_poi_fused_relu_0 → "triton:fused_relu"
            parts = name.split("triton_")
            if len(parts) > 1:
                return f"triton:{parts[1]}"
        # cuBLAS kernels
        if "gemm" in name or "sgemm" in name or "hgemm" in name:
            return "aten::mm"
        if "cutlass" in name:
            return "cutlass_gemm"

    elif program_type == "triton":
        if "triton_" in name:
            parts = name.split("triton_")
            if len(parts) > 1:
                return parts[1]

    elif program_type == "jax":
        if "xla" in name:
            return "xla_computation"

    return None


# ---------------------------------------------------------------------------
# Unified kernel dossier: attribution + NCU metrics + SASS
# ---------------------------------------------------------------------------

@dataclass
class KernelDossier:
    """Unified view of a GPU kernel combining attribution, NCU metrics, and SASS."""
    name: str                     # display name (demangled or NSys name)
    gpu_pct: float                # % of total GPU time (from attribution)
    gpu_time_ms: float            # total GPU time in ms
    # NCU metrics (None if not matched)
    ncu_metrics: dict | None = None
    # SASS snippet (None if not matched)
    sass_snippet: str | None = None
    sass_instruction_count: int | None = None
    # Attribution metadata
    launch_overhead_us: float | None = None
    caller_function: str | None = None  # user-code function from call chain / py-spy
    framework_op: str | None = None     # framework-level op (e.g. aten::mm) from temporal matching
    diagnosis: str = ""
    suggestions: list[str] = field(default_factory=list)


def build_kernel_dossiers(
    gpu_attribution: list[dict] | None,
    ncu_summary: dict | None,
    sass_entries: list[dict] | None,
    *,
    max_kernels: int = 3,
) -> list[KernelDossier]:
    """Join GPU attribution ranking with NCU per-kernel metrics and SASS.

    The join is by fuzzy kernel name matching:
    - Attribution uses NSys names (e.g., "volta_sgemm_128x128_nn")
    - NCU uses demangled names (e.g., "sgemm_naive(int, int, ...)")
    - SASS uses mangled names (e.g., "_Z12sgemm_naivePfS_S_iii")

    Returns dossiers sorted by GPU time % (attribution ranking order).
    """
    if not gpu_attribution:
        return []

    ncu_kernels = (ncu_summary or {}).get("kernels", [])
    sass_list = sass_entries or []

    dossiers: list[KernelDossier] = []
    for attrib in gpu_attribution[:max_kernels]:
        attrib_name = attrib.get("name", "")
        if not attrib_name:
            continue

        # Match NCU per-kernel data
        ncu_match = _match_kernel(attrib_name, ncu_kernels, key="name")
        ncu_metrics = None
        if ncu_match:
            ncu_metrics = {
                k: v for k, v in ncu_match.items()
                if k not in ("name", "invocations", "source_file", "function")
            }

        # Match SASS snippet
        sass_match = _match_kernel(attrib_name, sass_list, key="kernel")
        sass_snippet = sass_match.get("snippet") if sass_match else None
        sass_count = sass_match.get("instruction_count") if sass_match else None

        dossiers.append(KernelDossier(
            name=attrib_name,
            gpu_pct=attrib.get("gpu_pct", 0),
            gpu_time_ms=attrib.get("gpu_time_ms", 0),
            ncu_metrics=ncu_metrics,
            sass_snippet=sass_snippet,
            sass_instruction_count=sass_count,
            launch_overhead_us=attrib.get("launch_overhead_us"),
            caller_function=attrib.get("caller_function"),
            framework_op=attrib.get("framework_op"),
            diagnosis=attrib.get("diagnosis", ""),
            suggestions=attrib.get("suggestions", []),
        ))

    return dossiers


def _match_kernel(
    target_name: str,
    candidates: list[dict],
    key: str = "name",
) -> dict | None:
    """Find the best fuzzy match for a kernel name in a list of candidates.

    Returns the matched dict or None.
    """
    if not target_name or not candidates:
        return None

    target_lower = target_name.lower()
    target_base = target_lower.split("::")[-1].split("<")[0].split("(")[0]

    best: dict | None = None
    best_score = 0

    for c in candidates:
        cname = c.get(key, "")
        if not cname:
            continue
        cname_lower = cname.lower()
        cname_base = cname_lower.split("::")[-1].split("<")[0].split("(")[0]

        score = 0
        # Exact match
        if target_lower == cname_lower:
            score = 100
        # Direct substring
        elif target_lower in cname_lower or cname_lower in target_lower:
            score = 80
        # Base name match
        elif target_base and cname_base and (target_base in cname_base or cname_base in target_base):
            score = 60
        # Partial base overlap (e.g., "sgemm" matches "sgemm_naive_kernel")
        elif target_base and cname_base:
            # Check if any significant token (>3 chars) appears in both
            for token in target_base.replace("_", " ").split():
                if len(token) > 3 and token in cname_base:
                    score = 40
                    break

        if score > best_score:
            best_score = score
            best = c

    return best
