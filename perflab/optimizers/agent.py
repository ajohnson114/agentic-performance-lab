from __future__ import annotations

import fnmatch
import json
import logging
import shlex
import shutil
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# Suppress JAX fork() warnings — harmless in agent context where fork is controlled
warnings.filterwarnings("ignore", message=".*os.fork.*multithreaded.*", category=RuntimeWarning)

logger = logging.getLogger(__name__)

from perflab.analyzers.bottleneck_analyzer import compute_source_hints, diagnose_bottlenecks
from perflab.analyzers.compiler_diagnostics import (
    CompilerDiagnostics,
    cross_reference_diagnostics,
)
from perflab.analyzers.metrics_rollup import compute_run_summary, is_improvement
from perflab.analyzers.user_actions import UserAction, extract_build_suggestions
from perflab.llm.base import LLMProvider
from perflab.llm.config import LLMConfig, create_provider
from perflab.memory.run_store import RunStore
from perflab.optimizers.convergence import ConvergenceDetector
from perflab.optimizers.event_log import AgentEventLog
from perflab.optimizers.patch import (
    SearchReplaceBlock,
    apply_patch,
    backup_files,
    restore_files,
    validate_patch,
)
from perflab.optimizers.progress import AgentProgress, PrintProgress
from perflab.optimizers.prompt import PromptContext, build_prompt, parse_candidates
from perflab.reporting.generate import ReportParams, generate_reports
from perflab.runners.benchmark import metric_value, run_benchmark, validate_contract
from perflab.runners.correctness import run_correctness
from perflab.task_spec import TaskSpec


@dataclass
class BeamCandidate:
    iteration: int
    index: int
    blocks: list[SearchReplaceBlock]
    description: str
    reasoning: str = ""
    value: float | None = None
    accepted: bool = False


@dataclass
class AgentConfig:
    n_candidates: int = 6
    top_k: int = 2
    max_iters: int = 12
    diversity_penalty: float = 0.1
    early_stop: bool = True
    fast_screen: bool = True
    max_wall_time_s: int = 3600


@dataclass
class AgentContext:
    """All state for an optimization run, threaded through every phase.

    The Context pattern replaces the 30+ local variables and 18-parameter
    function signatures in the agent loop with a single structured object.
    Each handler reads what it needs and writes what it produces.

    Fields are organized by lifecycle:
      - Immutable inputs: set at construction, never changed
      - Baseline state: set once during the baseline phase
      - Evolving state: mutated across iterations by handlers
      - Per-iteration transient: reset at the start of each iteration
      - Accumulating metadata: counters and logs that grow monotonically
    """

    # --- Immutable inputs (set at construction) ---
    task: TaskSpec
    config: AgentConfig
    llm_config: LLMConfig
    provider: LLMProvider
    progress: AgentProgress
    ws: Path                                              # task.workspace shorthand
    rp: object                                            # RunPath from run_store.new_run()
    event_log: AgentEventLog
    expert_suggestion: str | None = None

    # --- Baseline state (set once after baseline phase) ---
    baseline_val: float = 0.0
    sec_metric: object | None = None                      # SecondaryMetricSpec or None
    baseline_sec_val: float | None = None
    sysinfo: dict = field(default_factory=dict)
    hardware_mismatch: str | None = None
    prior_run_context: str | None = None

    # --- Evolving optimization state (mutated across iterations) ---
    iteration: int = 0
    best_value: float = 0.0
    best_iter: int = 0
    accepted_count: int = 0
    history: list[dict] = field(default_factory=list)
    accepted_patches: list[dict] = field(default_factory=list)
    failure_memory: list[dict] = field(default_factory=list)
    latest_diagnostics: CompilerDiagnostics | None = None
    prev_summaries: dict | None = None
    profiler_summaries: dict = field(default_factory=dict)

    # --- Per-iteration transient (reset each iteration) ---
    last_errors: list[dict] = field(default_factory=list)
    promising_alternatives: list[dict] = field(default_factory=list)

    # --- Accumulating metadata ---
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_latency: float = 0.0
    user_actions: list[dict] = field(default_factory=list)
    early_stop_reason: str | None = None
    convergence: ConvergenceDetector | None = None
    wall_start: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        """Serialize context state for resumability (excludes non-serializable objects)."""
        d = {
            "iteration": self.iteration,
            "best_value": self.best_value,
            "best_iter": self.best_iter,
            "baseline_val": self.baseline_val,
            "accepted_count": self.accepted_count,
            "history": self.history,
            "accepted_patches": self.accepted_patches,
            "failure_memory": self.failure_memory,
            "last_errors": self.last_errors,
            "promising_alternatives": self.promising_alternatives,
            "total_llm_calls": self.total_llm_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_llm_latency": self.total_llm_latency,
            "user_actions": self.user_actions,
            "early_stop_reason": self.early_stop_reason,
        }
        if self.latest_diagnostics is not None:
            d["latest_diagnostics"] = self.latest_diagnostics.to_dict()
        return d

    def load_dict(self, data: dict) -> None:
        """Restore serialized state for resumability."""
        for key, val in data.items():
            if key == "latest_diagnostics" and isinstance(val, dict):
                self.latest_diagnostics = CompilerDiagnostics.from_dict(val)
            elif hasattr(self, key):
                setattr(self, key, val)


@dataclass
class AgentResult:
    best_value: float
    best_iter: int
    baseline_value: float
    history: list[dict]
    run_dir: Path


def _fmt_usage(usage: dict) -> str:
    """Format token usage as 'in=1234, out=567, total=1801'."""
    inp = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
    out = usage.get("output_tokens") or usage.get("completion_tokens", 0)
    total = inp + out
    return f"in={inp}, out={out}, total={total}"


def _is_context_overflow_error(exc: Exception) -> bool:
    """Detect if an LLM exception is a context window overflow.

    Each provider surfaces this differently — OpenAI uses "context_length_exceeded",
    Anthropic uses "prompt is too long", Ollama uses "too many tokens". Rather than
    importing provider-specific exception types (which may not be installed), we
    match against known error message substrings.

    If a provider changes its error format, the worst case is a missed catch —
    the error propagates normally instead of triggering emergency trimming.
    """
    exc_str = str(exc).lower()
    return any(k in exc_str for k in (
        "context_length", "context length", "maximum context",
        "token limit", "too many tokens", "max_tokens",
        "prompt is too long", "request too large",
    ))


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as '4.2s' or '2m15s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _calc_speedup(value: float, baseline: float) -> float:
    """Compute speedup ratio (value / baseline), returning 1.0 if baseline is zero."""
    return value / baseline if baseline != 0 else 1.0


def _check_hardware_mismatch(
    target_hardware: str | None,
    system_info: dict,
) -> str | None:
    """Check if target_hardware matches any detected GPU.

    Returns a mismatch message string if there's a mismatch, or None if OK.
    """
    if not target_hardware:
        return None
    gpus = system_info.get("nvidia_gpus") or []
    if not gpus:
        return None
    target_lower = target_hardware.lower()
    for gpu in gpus:
        gpu_name = (gpu.get("name") or "").lower()
        if target_lower in gpu_name or gpu_name in target_lower:
            return None
    detected_name = gpus[0].get("name", "unknown GPU")
    return (
        f'Hardware mismatch: task targets "{target_hardware}" '
        f'but detected GPU is "{detected_name}"'
    )


def _warn_build_flag_mismatch(
    build_cmd: str,
    diags: list,
    progress: AgentProgress,
) -> None:
    """Warn if bottleneck suggestions require build flags missing from the build command."""
    cmd_lower = build_cmd.lower()
    # Map: (keyword in suggested_actions) -> (required flag, update hint)
    checks = [
        ("openmp", "-fopenmp",
         'Bottleneck suggests OpenMP but build command lacks -fopenmp. '
         'Update task.yaml build.cmd to include -fopenmp if you want the agent to use OpenMP.'),
        ("-march=native", "-march=native",
         'Bottleneck suggests -march=native but it is not in the build command. '
         'Update task.yaml build.cmd to include -march=native for auto-vectorization.'),
        ("avx", "-mavx",
         'Bottleneck suggests AVX intrinsics but no AVX flags in build command. '
         'Update task.yaml build.cmd to include -march=native or -mavx2.'),
    ]
    warned: set[str] = set()
    for diag in diags:
        for action in (diag.suggested_actions or []):
            action_lower = action.lower()
            for keyword, flag, message in checks:
                if keyword in action_lower and flag not in cmd_lower and flag not in warned:
                    progress.on_message(f"[agent] WARNING: {message}")
                    warned.add(flag)


def _resolve_roofline_peaks(task: TaskSpec) -> dict | None:
    """Resolve roofline peaks: use task.yaml config if set, else auto-detect."""
    from perflab.roofline_peaks import resolve_roofline
    return resolve_roofline(task)


def _snapshot_workspace(task: TaskSpec, run_dir: Path, label: str) -> Path | None:
    """Zip all allowed_paths files to run_dir/snapshots/<label>.zip."""
    import zipfile

    sources = _read_source_files(task)
    if not sources:
        return None

    snap_dir = run_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    zip_path = snap_dir / f"{label}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path, content in sources.items():
            zf.writestr(rel_path, content)

    return zip_path


def _read_source_files(task: TaskSpec) -> dict[str, str]:
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


def _build_dossiers(
    gpu_attribution: list[dict] | None,
    profiler_summaries: dict[str, dict],
    sass_entries: list[dict] | None,
) -> list | None:
    """Build unified kernel dossiers joining attribution + NCU + SASS.

    Returns None if GPU attribution is not available (no NSys correlation data).
    """
    if not gpu_attribution:
        return None
    try:
        from perflab.analyzers.gpu_attribution import build_kernel_dossiers
        ncu_summary = profiler_summaries.get("ncu")
        dossiers = build_kernel_dossiers(
            gpu_attribution, ncu_summary, sass_entries, max_kernels=3,
        )
        return dossiers if dossiers else None
    except (ImportError, KeyError, TypeError):
        logger.warning("GPU kernel dossier construction failed", exc_info=True)
        return None


def _auto_tune_sweep(
    ctx: AgentContext,
    max_trials: int = 15,
) -> float | None:
    """Run a parameter sweep on tuning.yaml if it has a sweep section.

    Returns the new best value if the sweep found something better,
    or None if no sweep was run or no improvement was found.
    """
    task = ctx.task
    rp = ctx.rp
    progress = ctx.progress
    event_log = ctx.event_log
    iteration = ctx.iteration
    current_best = ctx.best_value

    knobs_path = task.workspace / "tuning.yaml"
    if not knobs_path.exists():
        return None

    try:
        from perflab.optimizers.propose_params import load_knobs, save_knobs, generate_sweep_candidates, sample_candidates
        from perflab.runners.benchmark import run_benchmark
        from perflab.runners.correctness import run_correctness

        knobs = load_knobs(knobs_path)
        if not knobs.get("sweep"):
            return None

        candidates = generate_sweep_candidates(knobs)
        if not candidates:
            return None

        # Limit trials
        if len(candidates) > max_trials:
            candidates = sample_candidates(candidates, max_trials)

        progress.on_message(f"[agent] Auto-tuning: sweeping {len(candidates)} parameter combinations...")

        best_sweep_value = current_best
        best_sweep_knobs: dict | None = None
        original_knobs = dict(knobs)

        for i, cand in enumerate(candidates):
            # Write candidate knobs
            save_knobs(knobs_path, cand.new_knobs)

            # Build if needed
            if task.build:
                import shlex
                from perflab.tools.shell import run_cmd
                build_res = run_cmd(shlex.split(task.build.cmd), cwd=task.workspace)
                if build_res.returncode != 0:
                    continue

            # Correctness check
            cres = run_correctness(
                task.correctness.cmd, cwd=task.workspace,
                program_type=task.program_type,
                rlimit_as_gb=task.constraints.rlimit_as_gb,
            )
            if cres.returncode != task.correctness.expected_exit:
                continue

            # Benchmark
            _, bench_data = run_benchmark(
                task.benchmark.cmd, cwd=task.workspace,
                program_type=task.program_type,
                rlimit_as_gb=task.constraints.rlimit_as_gb,
            )

            from perflab.runners.benchmark import metric_value
            value = metric_value(bench_data, task.benchmark.metric.name)

            is_better = (
                (task.benchmark.metric.mode == "maximize" and value > best_sweep_value) or
                (task.benchmark.metric.mode == "minimize" and value < best_sweep_value)
            )
            if is_better:
                best_sweep_value = value
                best_sweep_knobs = dict(cand.new_knobs)
                progress.on_message(
                    f"[agent]   Sweep {i+1}/{len(candidates)}: {cand.description} → {value:.6g} (NEW BEST)"
                )

        # Restore best configuration or original
        if best_sweep_knobs is not None:
            save_knobs(knobs_path, best_sweep_knobs)
            event_log._write("auto_tune_sweep", iteration, {
                "candidates_tried": len(candidates),
                "best_value": best_sweep_value,
                "best_knobs": best_sweep_knobs,
                "improvement": best_sweep_value - current_best,
            })
            progress.on_message(
                f"[agent] Auto-tune: found better params ({best_sweep_value:.6g} vs {current_best:.6g})"
            )
            return best_sweep_value
        else:
            # Restore original knobs
            save_knobs(knobs_path, {k: v for k, v in original_knobs.items() if k != "sweep"})
            return None

    except Exception as exc:
        progress.on_message(f"[agent] Auto-tune sweep failed: {exc}")
        # Restore original knobs on error
        try:
            save_knobs(knobs_path, load_knobs(knobs_path))
        except Exception:
            logger.warning("Failed to restore knobs after sweep error", exc_info=True)
        return None


def _build_hlo_attribution(profiler_summaries: dict[str, dict]) -> list[dict] | None:
    """Build HLO attribution for JAX/TPU tasks."""
    jax_summary = profiler_summaries.get("jax", {})
    if not jax_summary.get("hlo_ops"):
        return None
    try:
        from perflab.analyzers.hlo_attribution import compute_hlo_attribution
        attrib = compute_hlo_attribution(jax_summary, jax_summary)
        if attrib and attrib.entries:
            return [
                {
                    "op": e.op, "count": e.count,
                    "pct_of_ops": e.pct_of_ops,
                    "category": e.category,
                    "estimated_device_pct": e.estimated_device_pct,
                    "diagnosis": e.diagnosis,
                    "suggestions": e.suggestions,
                }
                for e in attrib.entries[:5]
            ]
    except (ImportError, KeyError, TypeError):
        logger.warning("HLO attribution failed", exc_info=True)
    return None


def _extract_data_hints(task: TaskSpec) -> dict | None:
    """Extract data hints from task spec as a dict for the prompt."""
    import dataclasses
    dh = task.data_hints
    result = {}
    for f in dataclasses.fields(dh):
        val = getattr(dh, f.name)
        if val is not None:
            result[f.name] = val
    return result if result else None


def _build_microarch(
    bench_data: dict,
    profiler_summaries: dict[str, dict],
    roofline_dict: dict | None,
) -> dict | None:
    """Build micro-architecture analysis summary for the prompt."""
    try:
        from perflab.analyzers.microarch import build_microarch_summary
        peak = (roofline_dict or {}).get("peak_tflops")
        return build_microarch_summary(bench_data, profiler_summaries, peak_tflops=peak)
    except (ImportError, KeyError, TypeError):
        logger.warning("Microarch summary construction failed", exc_info=True)
        return None


def _extract_memray_summary(profiler_summaries: dict[str, dict]) -> dict | None:
    """Extract memray summary for the LLM prompt if data is present."""
    s = profiler_summaries.get("memray", {})
    if not s or s.get("returncode", 1) != 0:
        return None
    if s.get("top_allocators") or s.get("peak_memory_mb"):
        return {
            "peak_memory_mb": s.get("peak_memory_mb", 0),
            "total_allocated_mb": s.get("total_allocated_mb", 0),
            "total_allocations": s.get("total_allocations", 0),
            "top_allocators": s.get("top_allocators", []),
        }
    return None


def _extract_lock_contention_summary(profiler_summaries: dict[str, dict]) -> dict | None:
    """Extract lock contention summary for the LLM prompt if contention was found."""
    s = profiler_summaries.get("lock_contention", {})
    if not s:
        return None
    lock_stats = s.get("lock_stats", {})
    c2c_stats = s.get("c2c_stats", {})
    if lock_stats.get("total_contended", 0) > 0 or c2c_stats.get("total_hitm", 0) > 0:
        return {"lock_stats": lock_stats, "c2c_stats": c2c_stats}
    return None


def _extract_gpu_memory_summary(profiler_summaries: dict[str, dict]) -> dict | None:
    """Extract GPU memory utilization for the LLM prompt."""
    s = profiler_summaries.get("power", {})
    gpu_mem = s.get("gpu_memory")
    if gpu_mem and gpu_mem.get("total_mib", 0) > 0:
        return gpu_mem
    return None


def _extract_ebpf_summary(profiler_summaries: dict[str, dict]) -> dict | None:
    """Extract eBPF syscall/IO tracing summary for the LLM prompt."""
    s = profiler_summaries.get("ebpf", {})
    if not s or s.get("returncode", 1) != 0:
        return None
    has_data = (
        s.get("read_syscalls", 0) > 0
        or s.get("write_syscalls", 0) > 0
        or s.get("read_latency")
        or s.get("write_latency")
    )
    if not has_data:
        return None
    return {
        "read_syscalls": s.get("read_syscalls", 0),
        "write_syscalls": s.get("write_syscalls", 0),
        "read_bytes": s.get("read_bytes", 0),
        "write_bytes": s.get("write_bytes", 0),
        "read_latency": s.get("read_latency"),
        "write_latency": s.get("write_latency"),
    }


_profiler_summary_cache: dict[str, tuple[dict[str, float], dict[str, dict]]] = {}


def _load_profiler_summaries(artifacts_dir: Path) -> dict[str, dict]:
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


def _build_gpu_context(
    profiler_summaries: dict[str, dict],
    progress: AgentProgress,
) -> list[dict] | None:
    """Extract GPU attribution dicts from profiler summaries."""
    gpu_attrib_dicts: list[dict] | None = None
    nsys_summary = profiler_summaries.get("nsys", {})
    if nsys_summary.get("cpu_gpu_correlations"):
        try:
            from perflab.analyzers.gpu_attribution import compute_attribution_ranking
            perf_for_attrib = profiler_summaries.get("linux_perf")
            torch_for_attrib = profiler_summaries.get("torch_profiler")
            pyspy_for_attrib = profiler_summaries.get("pyspy")
            ranking = compute_attribution_ranking(
                nsys_summary, perf_for_attrib,
                torch_summary=torch_for_attrib,
                pyspy_summary=pyspy_for_attrib,
            )
            if ranking:
                gpu_attrib_dicts = [
                    {
                        "rank": e.rank,
                        "category": e.category,
                        "name": e.name,
                        "gpu_time_ms": e.gpu_time_ms,
                        "gpu_pct": e.gpu_pct,
                        "launch_overhead_us": e.launch_overhead_us,
                        "caller_function": e.caller_function,
                        "framework_op": e.framework_op,
                        "diagnosis": e.diagnosis,
                        "suggestions": e.suggestions,
                    }
                    for e in ranking[:5]
                ]
                top = ranking[0]
                caller_info = f" [called by {top.caller_function}]" if top.caller_function else ""
                framework_info = f" [from {top.framework_op}]" if top.framework_op else ""
                progress.on_message(f"[agent] GPU attribution: top kernel is {top.name} ({top.gpu_pct:.0f}% GPU time){framework_info}{caller_info}")
        except Exception:
            logger.warning("GPU attribution failed", exc_info=True)
    return gpu_attrib_dicts


def _build_diagnostics_context(
    profiler_summaries: dict[str, dict],
    latest_diagnostics: CompilerDiagnostics | None,
    cpu_isa: str | None,
    gpu_attrib_dicts: list[dict] | None,
    task: TaskSpec,
    bench_data: dict,
    source_files: dict[str, str],
    sysinfo: dict | None,
    progress: AgentProgress,
) -> tuple[list[dict] | None, list[dict] | None]:
    """Run bottleneck diagnosis and cross-reference compiler remarks.

    Returns (diag_dicts, cross_ref_dicts).
    """
    diag_dicts: list[dict] | None = None
    device = bench_data.get("meta", {}).get("device", None)
    src_hints = compute_source_hints(source_files, task.program_type) if source_files else None
    if profiler_summaries:
        diags = diagnose_bottlenecks(
            profiler_summaries, task.program_type, device=device,
            top_n=task.constraints.top_n,
            thresholds=task.analysis_thresholds,
            system_info=sysinfo, source_hints=src_hints,
            compiler_remarks=latest_diagnostics.remarks if latest_diagnostics else None,
            cpu_isa=cpu_isa,
        )
        if diags:
            diag_dicts = [
                {
                    "rank": d.rank,
                    "bottleneck": d.bottleneck,
                    "root_cause": d.root_cause,
                    "confidence": d.confidence,
                    "suggested_actions": d.suggested_actions,
                }
                for d in diags
            ]
            progress.on_message(f"[agent] Bottleneck diagnosis: {diags[0].bottleneck} ({diags[0].confidence} confidence)")

            # Warn if bottleneck suggestions require build flags not in the build command
            if task.build:
                _warn_build_flag_mismatch(task.build.cmd, diags, progress)

    # Cross-reference compiler remarks with profiler hotspots (+ GPU attribution for CUDA)
    cross_ref_dicts: list[dict] | None = None
    if latest_diagnostics and latest_diagnostics.remarks:
        perf_summary = profiler_summaries.get("linux_perf", {})
        insights = cross_reference_diagnostics(
            latest_diagnostics.remarks, perf_summary, cpu_isa=cpu_isa,
            gpu_attribution=gpu_attrib_dicts,
        )
        if insights:
            cross_ref_dicts = [
                {
                    "priority": ins.priority,
                    "category": ins.category,
                    "source_location": ins.source_location,
                    "description": ins.description,
                    "suggestion": ins.suggestion,
                    "perf_pct": ins.perf_pct,
                }
                for ins in insights
            ]
            progress.on_message(f"[agent] Compiler cross-ref: {len(insights)} insights (top: {insights[0].category})")

    return diag_dicts, cross_ref_dicts


def _build_assembly_context(
    task: TaskSpec,
    profiler_summaries: dict[str, dict],
    artifacts_dir: Path,
    progress: AgentProgress,
) -> tuple[list[dict] | None, list[dict] | None]:
    """Extract hot assembly and SASS disassembly.

    Returns (hot_asm, sass_asm).
    """
    # Hot loop assembly from perf annotate
    hot_asm: list[dict] | None = None
    if task.program_type in ("cpp", "cuda"):
        try:
            from perflab.profilers.linux_perf import extract_hot_assembly
            annotate_path = artifacts_dir / "perf_annotate.txt"
            snippets = extract_hot_assembly(annotate_path, max_functions=3, context_lines=8)
            if snippets:
                hot_asm = snippets
                top = snippets[0]
                progress.on_message(f"[agent] Hot assembly: {top['function']} ({top['hot_pct']:.1f}% CPU)")
        except Exception:
            logger.warning("Hot assembly extraction failed", exc_info=True)

    # CUDA SASS disassembly from cuobjdump
    sass_asm: list[dict] | None = None
    if task.program_type == "cuda" and task.build:
        from perflab.profilers import extract_sass_from_build
        sass_snippets = extract_sass_from_build(
            task.build.cmd, task.workspace, artifacts_dir,
            max_kernels=2, context_lines=10,
        )
        if sass_snippets:
            sass_asm = sass_snippets
            progress.on_message(f"[agent] SASS: {sass_snippets[0]['kernel']} ({sass_snippets[0]['instruction_count']} instructions)")

    return hot_asm, sass_asm


def _build_roofline_context(
    task: TaskSpec,
    bench_data: dict,
    profiler_summaries: dict[str, dict],
    it: int,
    progress: AgentProgress,
    event_log: AgentEventLog,
    ctx: "AgentContext",
) -> dict | None:
    """Build roofline dict with optional achieved_bw, dtype_peaks, and computed AI."""
    roofline_dict: dict | None = None
    if task.roofline:
        roofline_dict = {
            "peak_tflops": task.roofline.peak_tflops,
            "peak_mem_bw_gbs": task.roofline.peak_mem_bw_gbs,
        }
        # Add achieved bandwidth from ncu summary
        ncu_s = profiler_summaries.get("ncu", {})
        if ncu_s.get("achieved_bw_gbs") is not None:
            roofline_dict["achieved_bw_gbs"] = ncu_s["achieved_bw_gbs"]
        # Add per-dtype peaks and L2 bandwidth
        from perflab.roofline_peaks import _lookup_dtype_peaks, _lookup_l2_bw
        dp = _lookup_dtype_peaks(task.target_hardware or "")
        l2_bw = _lookup_l2_bw(task.target_hardware or "")
        if l2_bw is not None:
            roofline_dict["peak_l2_bw_gbs"] = l2_bw
        if dp:
            roofline_dict["dtype_peaks"] = dp
        # Compute AI from roofline point for bound classification
        # Try profiler-provided FLOPS (PyTorch with_flops, JAX HLO cost analysis)
        profiler_flops: float | None = None
        torch_summary = profiler_summaries.get("torch_profiler", {})
        if torch_summary.get("total_flops"):
            profiler_flops = torch_summary["total_flops"]
        jax_summary = profiler_summaries.get("jax", {})
        if profiler_flops is None and jax_summary.get("hlo_cost_flops"):
            profiler_flops = jax_summary["hlo_cost_flops"]

        try:
            from perflab.reporting.roofline import compute_roofline_point
            measured_dram = ncu_s.get("dram_bytes_total")
            rp_point = compute_roofline_point(
                bench_data,
                measured_dram_bytes=measured_dram,
                profiler_flops=profiler_flops,
            )
            if rp_point is not None:
                roofline_dict["computed_ai"] = rp_point.ai
                roofline_dict["computed_achieved_tflops"] = rp_point.tflops
                # Stash on the current history entry so we can plot the trail later
                if ctx.history:
                    ctx.history[-1]["roofline_ai"] = rp_point.ai
                    ctx.history[-1]["roofline_tflops"] = rp_point.tflops
        except Exception:
            logger.warning("Roofline point computation failed", exc_info=True)
    else:
        # Auto-detect roofline peaks (CPU spec-based or GPU)
        try:
            from perflab.roofline_peaks import infer_peaks
            auto_peaks = infer_peaks(task.target_hardware or "auto")
            if auto_peaks:
                roofline_dict = {
                    "peak_tflops": auto_peaks.peak_tflops,
                    "peak_mem_bw_gbs": auto_peaks.peak_mem_bw_gbs,
                }
                if it == 1:
                    progress.on_message(f"[agent] Auto-detected roofline: {auto_peaks.peak_tflops:.3f} TFLOPS, {auto_peaks.peak_mem_bw_gbs:.1f} GB/s ({auto_peaks.source}: {auto_peaks.device})")
                    event_log.roofline_detected(auto_peaks.peak_tflops, auto_peaks.peak_mem_bw_gbs, auto_peaks.source, auto_peaks.device)
        except Exception:
            logger.warning("Roofline auto-detect failed", exc_info=True)

    return roofline_dict


def _regenerate_roofline_with_history(ctx: "AgentContext") -> None:
    """Regenerate roofline.png with the full optimization trail from ctx.history.

    Only runs if the task has a roofline spec and at least one history entry
    has roofline data. Overwrites the last per-iteration PNG so the dashboard
    picks up the trail automatically.
    """
    task = ctx.task
    if task.roofline is None:
        return

    trail = [e for e in ctx.history if "roofline_ai" in e and "roofline_tflops" in e]
    if not trail:
        return

    # Gold star = best accepted iteration, not just last profiled entry.
    # Fall back to trail[-1] if best_iter has no roofline data (e.g. NCU skipped that iter).
    best_entries = [e for e in trail if e.get("iteration") == ctx.best_iter]
    final = best_entries[0] if best_entries else trail[-1]
    roof_png = ctx.rp.artifacts_dir / "roofline.png"

    try:
        from perflab.reporting.roofline import RooflinePoint, write_roofline_png
        from perflab.roofline_peaks import _lookup_dtype_peaks, _lookup_l2_bw

        pt = RooflinePoint(
            ai=final["roofline_ai"],
            tflops=final["roofline_tflops"],
            gbs=final["roofline_tflops"] * 1000.0 / max(final["roofline_ai"], 1e-9),
        )
        write_roofline_png(
            roof_png,
            point=pt,
            peak_tflops=float(task.roofline.peak_tflops),
            peak_mem_bw_gbs=float(task.roofline.peak_mem_bw_gbs),
            title=task.roofline.title or f"Roofline — {task.name}",
            dtype_peaks=_lookup_dtype_peaks(task.target_hardware or ""),
            l2_bw_gbs=_lookup_l2_bw(task.target_hardware or ""),
            history_points=trail,  # full trail; gold star renders on top of final dot
        )
    except Exception:
        logger.warning("Failed to regenerate roofline with history trail", exc_info=True)


def _build_iteration_prompt(ctx: AgentContext) -> list:
    """Build prompt messages for one iteration.

    Reads from ctx and returns the messages list.
    Side effect: updates ctx.prev_summaries for the next iteration's profile diff.
    """
    task = ctx.task
    rp = ctx.rp
    it = ctx.iteration
    progress = ctx.progress
    event_log = ctx.event_log
    sysinfo = ctx.sysinfo
    latest_diagnostics = ctx.latest_diagnostics
    prev_summaries = ctx.prev_summaries

    source_files = _read_source_files(task)
    profiler_summaries = _load_profiler_summaries(rp.artifacts_dir)

    bench_data = {}
    bench_json = rp.run_dir / "bench.json"
    if bench_json.exists():
        bench_data = json.loads(bench_json.read_text(encoding="utf-8"))

    cpu_isa = sysinfo.get("cpu_isa") if sysinfo else None

    # GPU attribution (must run before diagnostics since cross-ref uses it)
    gpu_attrib_dicts = _build_gpu_context(profiler_summaries, progress)

    # Bottleneck diagnosis + compiler cross-referencing
    diag_dicts, cross_ref_dicts = _build_diagnostics_context(
        profiler_summaries, latest_diagnostics, cpu_isa, gpu_attrib_dicts,
        task, bench_data, source_files, sysinfo, progress,
    )

    # Profile diff from previous iteration
    profile_diff_str: str | None = None
    if it > 1 and prev_summaries is not None:
        try:
            from perflab.analyzers.profile_diff import compute_profile_diff, format_profile_diff
            deltas = compute_profile_diff(
                prev_summaries, profiler_summaries,
                task.benchmark.metric.mode,
            )
            if deltas:
                profile_diff_str = format_profile_diff(deltas)
                improved = [d for d in deltas if d.get("direction") == "improved"]
                regressed = [d for d in deltas if d.get("direction") == "regressed"]
                if improved or regressed:
                    parts = []
                    if improved:
                        parts.append(f"{len(improved)} improved")
                    if regressed:
                        parts.append(f"{len(regressed)} regressed")
                    progress.on_message(f"[agent] Profile diff: {', '.join(parts)}")
        except Exception:
            logger.warning("Profile diff computation failed", exc_info=True)

    # Build flag recommendations (static ISA + dynamic profiler-driven)
    build_flag_dicts: list[dict] | None = None
    if task.build:
        try:
            from perflab.analyzers.build_flags import recommend_build_flags, recommend_flags_from_profiling
            all_recs = []
            if cpu_isa:
                all_recs.extend(recommend_build_flags(task.build.cmd, cpu_isa, task.program_type))
            # Dynamic: recommendations based on profiler output
            prof_recs = recommend_flags_from_profiling(
                task.build.cmd, profiler_summaries, task.program_type, cpu_isa=cpu_isa,
            )
            all_recs.extend(prof_recs)
            # Deduplicate by flag name
            seen_flags: set[str] = set()
            deduped: list = []
            for r in all_recs:
                if r.flag not in seen_flags:
                    seen_flags.add(r.flag)
                    deduped.append(r)
            if deduped:
                build_flag_dicts = [
                    {"flag": r.flag, "reason": r.reason, "impact": r.impact, "category": r.category}
                    for r in deduped
                ]
                progress.on_message(f"[agent] Build flag recommendations: {', '.join(r.flag for r in deduped)}")
        except Exception:
            logger.warning("Build flag recommendation failed", exc_info=True)

    # Assembly and SASS
    hot_asm, sass_asm = _build_assembly_context(task, profiler_summaries, rp.artifacts_dir, progress)

    # Roofline
    roofline_dict = _build_roofline_context(task, bench_data, profiler_summaries, it, progress, event_log, ctx)

    prompt_ctx = PromptContext(
        source_files=source_files,
        profiler_summaries=profiler_summaries,
        bench_results=bench_data,
        roofline=roofline_dict,
        history=ctx.history,
        allowed_paths=task.edit_policy.allowed_paths,
        n_candidates=ctx.config.n_candidates,
        target_hardware=task.target_hardware,
        program_type=task.program_type,
        expert_suggestion=ctx.expert_suggestion,
        bottleneck_diagnoses=diag_dicts,
        compiler_diagnostics=ctx.latest_diagnostics.summary if ctx.latest_diagnostics and ctx.latest_diagnostics.summary else None,
        cross_referenced_insights=cross_ref_dicts,
        gpu_attribution=gpu_attrib_dicts,
        profile_diff=profile_diff_str,
        build_flag_recommendations=build_flag_dicts,
        prior_run_context=ctx.prior_run_context if it == 1 else None,
        prompt_token_budget=task.constraints.prompt_token_budget,
        max_history=task.constraints.max_history,
        model=ctx.llm_config.model,
        last_errors=ctx.last_errors if ctx.last_errors else None,
        hot_loop_assembly=hot_asm,
        cuda_sass=sass_asm,
        kernel_dossiers=_build_dossiers(gpu_attrib_dicts, profiler_summaries, sass_asm),
        microarch_summary=_build_microarch(bench_data, profiler_summaries, roofline_dict),
        data_hints=_extract_data_hints(task),
        hlo_attribution=_build_hlo_attribution(profiler_summaries),
        memray_summary=_extract_memray_summary(profiler_summaries),
        lock_contention_summary=_extract_lock_contention_summary(profiler_summaries),
        gpu_memory_summary=_extract_gpu_memory_summary(profiler_summaries),
        ebpf_summary=_extract_ebpf_summary(profiler_summaries),
        allow_fast_math=task.constraints.allow_fast_math,
        accuracy_tolerance=task.constraints.accuracy_tolerance,
        failure_memory=ctx.failure_memory if ctx.failure_memory else None,
        promising_alternatives=ctx.promising_alternatives if ctx.promising_alternatives else None,
    )
    messages = build_prompt(prompt_ctx)
    ctx.prev_summaries = profiler_summaries
    return messages


def _try_accept_best(
    ctx: AgentContext,
    candidates: list[BeamCandidate],
    backup_dir: Path,
    use_fast: bool,
) -> tuple[bool, float | None]:
    """Find the best improving candidate, accept it, re-profile.

    Mutates ctx.history, ctx.accepted_patches, ctx.accepted_count, ctx.best_value,
    ctx.best_iter, and ctx.latest_diagnostics in place.
    Returns (accepted, rel_improvement) -- True + relative improvement for success,
    or (False, None) for no improvement.
    """
    task = ctx.task
    ws = ctx.ws
    rp = ctx.rp
    it = ctx.iteration
    progress = ctx.progress
    event_log = ctx.event_log

    scored = [c for c in candidates if c.value is not None]
    if task.benchmark.metric.mode == "maximize":
        scored.sort(key=lambda c: c.value, reverse=True)
    else:
        scored.sort(key=lambda c: c.value)

    for cand in scored:
        if not is_improvement(
            cand.value, ctx.best_value,
            task.benchmark.metric.mode,
            task.constraints.regression_tolerance,
        ):
            continue

        # If we used fast screening, re-benchmark with full precision
        if use_fast:
            progress.on_message("[agent]   Re-benchmarking top candidate with full precision...")
            backed_up = backup_files(cand.blocks, ws, backup_dir)
            try:
                apply_patch(cand.blocks, ws)
                _, bench_full = run_benchmark(task.benchmark.cmd, cwd=ws, fast_mode=False, program_type=task.program_type, rlimit_as_gb=task.constraints.rlimit_as_gb)
                contract_errors = validate_contract(bench_full, task.contract)
                if contract_errors:
                    progress.on_message(f"[agent]   Contract violation on full re-bench: {contract_errors}")
                    continue
                full_val = metric_value(bench_full, task.benchmark.metric.name)
                progress.on_message(f"[agent]   Full benchmark: {task.benchmark.metric.name} = {full_val:.6g}")
                cand.value = full_val
            finally:
                restore_files(backed_up, ws)

            # Re-check improvement with full benchmark value
            if not is_improvement(
                cand.value, ctx.best_value,
                task.benchmark.metric.mode,
                task.constraints.regression_tolerance,
            ):
                progress.on_message("[agent]   Full benchmark did not confirm improvement, skipping")
                continue

        # Compute relative improvement BEFORE updating best_value
        old_best = ctx.best_value

        # Re-apply permanently
        progress.on_message(f"[agent]   ACCEPTING {cand.description} (value={cand.value:.6g})")
        apply_patch(cand.blocks, ws)
        delta = cand.value - ctx.baseline_val
        speedup = _calc_speedup(cand.value, ctx.baseline_val)
        ctx.best_value = cand.value
        ctx.best_iter = it
        cand.accepted = True

        event_log.candidate_accepted(it, cand.index, cand.value, delta, speedup, cand.description)

        # Gaming detector: warn if suspiciously large speedup on first iteration
        if it == 1 and speedup > 10.0:
            progress.on_message(f"[agent] WARNING: suspiciously large speedup ({speedup:.1f}x) on first iteration — possible gaming")

        hist_entry = {
            "iteration": it,
            "description": cand.description,
            "value": cand.value,
            "accepted": True,
            "delta": delta,
            "speedup": speedup,
        }
        if cand.reasoning:
            hist_entry["reasoning"] = cand.reasoning
        # Track secondary metric if available
        if ctx.sec_metric:
            try:
                bench_json_path = rp.run_dir / "bench.json"
                if bench_json_path.exists():
                    bench_for_sec = json.loads(bench_json_path.read_text(encoding="utf-8"))
                    hist_entry["secondary_value"] = metric_value(bench_for_sec, ctx.sec_metric.name)
            except (KeyError, TypeError):
                logger.debug("Secondary metric extraction failed for iteration %d", it, exc_info=True)
        ctx.history.append(hist_entry)

        # Track for post-optimization summary
        ctx.accepted_patches.append({
            "iteration": it,
            "description": cand.description,
            "reasoning": cand.reasoning,
            "blocks": [{"file_path": b.file_path, "search": b.search[:500], "replace": b.replace[:500]} for b in cand.blocks],
            "value": cand.value,
        })

        ctx.accepted_count += 1

        # Snapshot accepted code
        _snapshot_workspace(task, rp.run_dir, f"iter{it}")

        # Auto-tune: if tuning.yaml has sweep parameters, run a quick sweep
        # to find optimal parameter values for the new code
        sweep_result = _auto_tune_sweep(ctx)
        if sweep_result is not None:
            ctx.best_value = sweep_result
            ctx.history.append({
                "iteration": it,
                "description": "Auto-tune sweep found better parameters",
                "value": ctx.best_value,
                "accepted": True,
                "delta": ctx.best_value - ctx.baseline_val,
                "speedup": _calc_speedup(ctx.best_value, ctx.baseline_val),
            })

        # Re-profile with accepted changes
        _, _, _, diag = _run_bench(ctx, do_profiles=True, capture_diagnostics=True)
        if diag is not None:
            ctx.latest_diagnostics = diag

        # Drift detection: every 3 accepted patches, re-run benchmark clean
        if ctx.accepted_count % 3 == 0:
            try:
                progress.on_message(f"[agent]   Drift check (accepted #{ctx.accepted_count})...")
                _, drift_bench = run_benchmark(task.benchmark.cmd, cwd=ws, fast_mode=False, program_type=task.program_type, rlimit_as_gb=task.constraints.rlimit_as_gb)
                drift_val = metric_value(drift_bench, task.benchmark.metric.name)
                drift_pct = abs(drift_val - cand.value) / abs(cand.value) * 100 if cand.value else 0
                event_log.drift_check(it, drift_val, cand.value, drift_pct)
                if drift_pct > 5:
                    progress.on_message(f"[agent]   WARNING: drift of {drift_pct:.1f}% detected")
            except Exception as exc:
                progress.on_message(f"[agent]   Drift check failed: {exc}")

        rel_improvement = abs(cand.value - old_best) / abs(old_best) if old_best != 0 else 0
        shutil.rmtree(backup_dir, ignore_errors=True)
        return (True, rel_improvement)

    # No candidate improved
    shutil.rmtree(backup_dir, ignore_errors=True)
    progress.on_message("[agent]   No improving candidate this iteration")
    best_desc = scored[0].description if scored else "no valid candidates"
    best_val = scored[0].value if scored else ctx.best_value
    reject_val = best_val or ctx.best_value
    ctx.history.append({
        "iteration": it,
        "description": f"no improvement ({best_desc})",
        "value": reject_val,
        "accepted": False,
        "delta": reject_val - ctx.baseline_val,
        "speedup": _calc_speedup(reject_val, ctx.baseline_val),
    })
    return (False, None)


def _prescreen_candidate(
    ci: int,
    blocks: list[SearchReplaceBlock],
    reasoning: str,
    task: TaskSpec,
    ws: Path,
    it: int,
) -> dict:
    """Prescreen a candidate: validate patch + build + correctness (no benchmark).

    Creates a temporary workspace copy to avoid file conflicts with other
    parallel prescreens. Returns a dict with 'passed', 'error', and metadata.
    """
    import shutil as _shutil
    import tempfile
    from perflab.tools.shell import run_cmd

    result: dict = {
        "ci": ci,
        "blocks": blocks,
        "reasoning": reasoning,
        "passed": False,
        "error": None,
    }

    # Validate patch
    validation_errors = validate_patch(blocks, task.edit_policy.allowed_paths, ws)
    if validation_errors:
        result["error"] = {"type": "validation", "description": validation_errors[0], "output": ""}
        return result

    # Create temp workspace copy for isolation
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"perflab_prescreen_{ci}_"))
        temp_ws = temp_dir / "ws"
        _shutil.copytree(ws, temp_ws, dirs_exist_ok=True)

        # Apply patch to temp copy
        apply_patch(blocks, temp_ws)

        # Build in temp copy
        # skip_preexec=True because we're inside ThreadPoolExecutor —
        # preexec_fn + fork() in a multithreaded process is undefined behavior.
        if task.build is not None:
            import shlex
            bres = run_cmd(shlex.split(task.build.cmd), cwd=temp_ws, timeout_s=120, skip_preexec=True)
            if bres.returncode != task.build.expected_exit:
                result["error"] = {
                    "type": "build",
                    "description": f"Build failed (exit code {bres.returncode})",
                    "output": bres.stderr[:1000],
                }
                return result

        # Correctness test in temp copy
        cres = run_correctness(
            task.correctness.cmd, cwd=temp_ws,
            program_type=task.program_type,
            rlimit_as_gb=task.constraints.rlimit_as_gb,
            skip_preexec=True,
        )
        if cres.returncode != task.correctness.expected_exit:
            result["error"] = {
                "type": "correctness",
                "description": f"Correctness failed (exit code {cres.returncode})",
                "output": cres.stderr[:1000],
            }
            return result

        result["passed"] = True
        return result

    except Exception as exc:
        result["error"] = {
            "type": "prescreen_error",
            "description": str(exc)[:500],
            "output": "",
        }
        return result
    finally:
        if temp_dir and temp_dir.exists():
            try:
                _shutil.rmtree(temp_dir)
            except Exception:
                pass


def prescreen_candidates_parallel(
    ctx: AgentContext,
    candidate_blocks: list[list[SearchReplaceBlock]],
    candidate_reasoning: list[str],
    max_workers: int = 4,
) -> list[dict]:
    """Prescreen candidates in parallel (build + correctness, no benchmark).

    Returns list of prescreen results. Candidates that pass can proceed
    to the sequential benchmark phase.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    task = ctx.task
    ws = ctx.ws
    it = ctx.iteration
    progress = ctx.progress

    n = len(candidate_blocks)
    if n == 0:
        return []

    progress.on_message(f"[agent] Prescreening {n} candidates in parallel (build+test)...")

    results: list[dict] = [{} for _ in range(n)]  # Pre-allocate for ordered results

    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as executor:
        futures = {}
        for ci, blocks in enumerate(candidate_blocks):
            reasoning = candidate_reasoning[ci] if ci < len(candidate_reasoning) else ""
            future = executor.submit(
                _prescreen_candidate, ci, blocks, reasoning, task, ws, it,
            )
            futures[future] = ci

        for future in as_completed(futures):
            ci = futures[future]
            try:
                results[ci] = future.result()
            except Exception as exc:
                results[ci] = {
                    "ci": ci,
                    "blocks": candidate_blocks[ci],
                    "reasoning": candidate_reasoning[ci] if ci < len(candidate_reasoning) else "",
                    "passed": False,
                    "error": {"type": "prescreen_error", "description": str(exc), "output": ""},
                }

    passed = sum(1 for r in results if r.get("passed"))
    failed = n - passed
    progress.on_message(f"[agent] Prescreen: {passed} passed, {failed} failed")

    return results


def _evaluate_single_candidate(
    ctx: AgentContext,
    ci: int,
    blocks: list[SearchReplaceBlock],
    reasoning: str,
    use_fast: bool,
) -> tuple[BeamCandidate, list[dict]]:
    """Evaluate a single candidate: validate, apply, correctness, benchmark.

    Returns (candidate, errors) where errors is a list of error dicts for feedback.
    """
    task = ctx.task
    ws = ctx.ws
    rp = ctx.rp
    it = ctx.iteration
    progress = ctx.progress
    event_log = ctx.event_log

    errors: list[dict] = []
    desc = f"candidate {ci + 1}: {len(blocks)} blocks"
    screen_label = " (fast screen)" if use_fast else ""
    progress.on_message(f"[agent]   Evaluating {desc}{screen_label}...")

    # Log patch content
    event_log.candidate_patch(it, ci, [
        {"file_path": b.file_path, "search": b.search, "replace": b.replace}
        for b in blocks
    ])

    # Validate
    validation_errors = validate_patch(blocks, task.edit_policy.allowed_paths, ws)
    event_log.candidate_validation(it, ci, not validation_errors, validation_errors)

    if validation_errors:
        progress.on_message(f"[agent]   Validation errors: {validation_errors}")
        return BeamCandidate(
            iteration=it, index=ci, blocks=blocks,
            description=f"{desc} (INVALID: {validation_errors[0]})",
            reasoning=reasoning,
        ), errors

    # Backup -> apply -> correctness -> benchmark -> restore
    backup_dir = rp.run_dir / "backups" / f"iter{it}"
    backed_up = backup_files(blocks, ws, backup_dir)
    try:
        apply_patch(blocks, ws)

        # Correctness check
        cres = run_correctness(task.correctness.cmd, cwd=ws, program_type=task.program_type, rlimit_as_gb=task.constraints.rlimit_as_gb)
        event_log.candidate_correctness(
            it, ci, cres.returncode == task.correctness.expected_exit,
            cres.returncode, cres.stderr,
        )

        if cres.returncode != task.correctness.expected_exit:
            progress.on_message(f"[agent]   Correctness FAILED (rc={cres.returncode})")
            errors.append({
                "type": "correctness",
                "description": f"candidate {ci + 1} failed correctness (exit code {cres.returncode})",
                "output": (cres.stderr or "")[:3000],
            })
            return BeamCandidate(
                iteration=it, index=ci, blocks=blocks,
                description=f"{desc} (correctness failed)",
                reasoning=reasoning,
            ), errors

        # Benchmark (fast screen or full depending on mode)
        try:
            _, bench = run_benchmark(task.benchmark.cmd, cwd=ws, fast_mode=use_fast, program_type=task.program_type, rlimit_as_gb=task.constraints.rlimit_as_gb)
        except Exception as exc:
            errors.append({
                "type": "benchmark",
                "description": f"candidate {ci + 1} benchmark failed: {exc}",
                "output": str(exc)[:3000],
            })
            progress.on_message(f"[agent]   Benchmark FAILED: {exc}")
            return BeamCandidate(
                iteration=it, index=ci, blocks=blocks,
                description=f"{desc} (benchmark failed)",
                reasoning=reasoning,
            ), errors

        # Contract validation
        contract_errors = validate_contract(bench, task.contract)
        if contract_errors:
            progress.on_message(f"[agent]   Contract violation: {contract_errors}")
            errors.append({
                "type": "contract_violation",
                "description": f"candidate {ci + 1}: {contract_errors[0]}",
                "output": "",
            })
            return BeamCandidate(
                iteration=it, index=ci, blocks=blocks,
                description=f"{desc} (contract violation)",
                reasoning=reasoning,
            ), errors

        val = metric_value(bench, task.benchmark.metric.name)
        progress.on_message(f"[agent]   {task.benchmark.metric.name} = {val:.6g}{screen_label}")

        event_log.candidate_benchmark(it, ci, val, task.benchmark.metric.name)

        return BeamCandidate(
            iteration=it, index=ci, blocks=blocks,
            description=desc, reasoning=reasoning, value=val,
        ), errors
    finally:
        restore_files(backed_up, ws)
        shutil.rmtree(backup_dir, ignore_errors=True)


def run_agent(
    task: TaskSpec,
    task_file: Path,
    config: AgentConfig,
    llm_config: LLMConfig,
    expert_suggestion: str | None = None,
    progress: AgentProgress | None = None,
    provider: LLMProvider | None = None,
) -> AgentResult:
    """Run the agentic beam-search optimizer.

    1. Baseline profile + benchmark
    2. Per iteration: build prompt -> LLM -> parse candidates -> evaluate each
    3. Accept best improving candidate, re-apply permanently, re-profile
    4. Generate reports
    """
    # Validate contract structure before spending time on benchmarks
    contract_errors = task.contract.validate()
    if contract_errors:
        raise ValueError(f"Invalid contract in task.yaml: {'; '.join(contract_errors)}")

    wall_start = time.monotonic()

    if progress is None:
        progress = PrintProgress()

    if provider is None:
        provider = create_provider(llm_config)
    ws = task.workspace
    run_store = RunStore(task.out_dir)
    rp = run_store.new_run(task.name, program_type=task.program_type)

    # System info capture
    sysinfo: dict = {}
    try:
        from perflab.tools.sysinfo import collect_system_info, warn_if_noisy
        sysinfo = collect_system_info()
        (rp.run_dir / "system_info.json").write_text(
            json.dumps(sysinfo, indent=2), encoding="utf-8"
        )
        for warning in warn_if_noisy():
            progress.on_message(f"[agent] WARNING: {warning}")
    except Exception as exc:
        progress.on_message(f"[agent] Failed to collect system info: {exc}")

    # Hardware mismatch check
    hardware_mismatch: str | None = None
    if task.target_hardware and sysinfo:
        hardware_mismatch = _check_hardware_mismatch(task.target_hardware, sysinfo)
        if hardware_mismatch:
            progress.on_message(f"[agent] \u26a0 {hardware_mismatch}")
            progress.on_message("[agent]   Roofline analysis and optimization hints may be inaccurate.")
            progress.on_message("[agent]   Set target_hardware to null for auto-detection, or update it to match your hardware.")

    # Save resolved configuration for run reproducibility
    try:
        from perflab.config import load_config
        resolved_cfg = load_config()
        resolved_cfg.save(rp.run_dir / "resolved_config.json")
    except Exception:
        logger.debug("Failed to save resolved config", exc_info=True)

    # Event logging
    event_log = AgentEventLog(rp.run_dir)

    # Construct AgentContext
    ctx = AgentContext(
        task=task,
        config=config,
        llm_config=llm_config,
        provider=provider,
        progress=progress,
        ws=ws,
        rp=rp,
        event_log=event_log,
        expert_suggestion=expert_suggestion,
        sysinfo=sysinfo,
        hardware_mismatch=hardware_mismatch,
        convergence=ConvergenceDetector() if config.early_stop else None,
        wall_start=wall_start,
    )

    # --- Baseline ---
    progress.on_message("[agent] Baseline run...")
    bench_base, bench_wall, prof_wall, latest_diagnostics = _run_bench(ctx, do_profiles=True, capture_diagnostics=True)
    ctx.baseline_val = metric_value(bench_base, task.benchmark.metric.name)
    ctx.best_value = ctx.baseline_val
    ctx.latest_diagnostics = latest_diagnostics

    # Secondary metric tracking (optional, for Pareto optimization)
    ctx.sec_metric = task.benchmark.secondary_metric
    if ctx.sec_metric:
        try:
            ctx.baseline_sec_val = metric_value(bench_base, ctx.sec_metric.name)
        except (KeyError, TypeError):
            ctx.sec_metric = None  # Not available in bench.json, disable

    baseline_entry: dict = {
        "iteration": 0,
        "description": "baseline",
        "value": ctx.baseline_val,
        "accepted": True,
        "delta": 0.0,
        "speedup": 1.0,
    }
    if ctx.baseline_sec_val is not None:
        baseline_entry["secondary_value"] = ctx.baseline_sec_val
    if bench_wall is not None:
        baseline_entry["bench_wall_time_s"] = bench_wall
    if bench_wall is not None and prof_wall is not None and bench_wall > 0:
        baseline_entry["profiling_overhead_pct"] = (prof_wall - bench_wall) / bench_wall * 100
    ctx.history.append(baseline_entry)
    progress.on_message(f"[agent] Baseline {task.benchmark.metric.name} = {ctx.baseline_val:.6g}")
    _bench_s = f"bench={_fmt_elapsed(bench_wall)}" if bench_wall is not None else ""
    _prof_s = f", profiling={_fmt_elapsed(prof_wall)}" if prof_wall is not None else ""
    if _bench_s:
        progress.on_message(f"[agent] Baseline timing: {_bench_s}{_prof_s}")

    # Snapshot baseline code
    _snapshot_workspace(task, rp.run_dir, "baseline")

    profiler_summaries = _load_profiler_summaries(rp.artifacts_dir)

    # Preserve baseline artifacts before they get overwritten by re-profiling
    baseline_artifacts_dir = rp.run_dir / "artifacts_baseline"
    if rp.artifacts_dir.exists() and not baseline_artifacts_dir.exists():
        shutil.copytree(rp.artifacts_dir, baseline_artifacts_dir)

    event_log.baseline_complete(
        ctx.baseline_val, bench_base, list(profiler_summaries.keys()),
    )

    # --- Cross-run learning: load prior run context ---
    try:
        from perflab.optimizers.cross_run import load_prior_run_context
        ctx.prior_run_context = load_prior_run_context(task.out_dir, current_run_id=rp.run_id)
        if ctx.prior_run_context:
            progress.on_message(f"[agent] Loaded context from prior runs")
    except Exception:
        logger.warning("Failed to load prior run context", exc_info=True)

    # --- Resume from checkpoint if available ---
    checkpoint_path = rp.run_dir / "checkpoint.json"
    if checkpoint_path.exists():
        try:
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            ctx.load_dict(saved)
            progress.on_message(f"[agent] Resumed from checkpoint (iteration {ctx.iteration})")
        except (json.JSONDecodeError, OSError, KeyError):
            logger.warning("Failed to load checkpoint, starting fresh", exc_info=True)

    # --- Iterate ---
    start_iter = ctx.iteration + 1 if ctx.iteration > 0 else 1
    for it in range(start_iter, config.max_iters + 1):
        ctx.iteration = it
        ctx.last_errors = ctx.last_errors  # preserve from previous iteration
        iteration_errors: list[dict] = []
        # Wall-clock budget check
        elapsed = time.monotonic() - ctx.wall_start
        if elapsed > config.max_wall_time_s:
            reason = f"Wall-clock budget exceeded ({elapsed:.0f}s > {config.max_wall_time_s}s)"
            progress.on_message(f"[agent] {reason}")
            event_log.early_stop(it, reason)
            ctx.history.append({"iteration": it, "description": f"early stop: {reason}", "value": ctx.best_value, "accepted": False, "delta": ctx.best_value - ctx.baseline_val, "speedup": _calc_speedup(ctx.best_value, ctx.baseline_val)})
            break

        iter_start = time.monotonic()
        remaining = max(0, config.max_wall_time_s - elapsed)
        progress.on_message(f"\n[agent] === Iteration {it}/{config.max_iters} === [elapsed {_fmt_elapsed(elapsed)}, {_fmt_elapsed(remaining)} remaining]")

        if ctx.last_errors:
            progress.on_message(f"[agent] Feeding {len(ctx.last_errors)} error(s) from previous iteration back to LLM")
            event_log.error_feedback(it, ctx.last_errors)
        if task.constraints.prompt_token_budget > 0:
            progress.on_message(f"[agent] Prompt token budget: {task.constraints.prompt_token_budget}")

        # Build prompt context
        messages = _build_iteration_prompt(ctx)

        # Log build flag state for this iteration
        if task.build:
            try:
                build_cmd_current = task.build.cmd
                # Extract flag recommendations from the prompt (they're embedded in the messages)
                flag_recs_for_log: list[dict] = []
                for msg in messages:
                    if "Build flag" in msg.content and "flag" in msg.content.lower():
                        # Extract just the flag names mentioned
                        import re as _re
                        flag_matches = _re.findall(r"`(-\S+)`", msg.content)
                        flag_recs_for_log = [{"flag": f} for f in flag_matches[:10]]
                        break
                event_log.build_flags_state(it, build_cmd_current, flag_recs_for_log)
            except Exception:
                logger.warning("Failed to log build flags state", exc_info=True)

        # Call LLM
        prompt_chars = sum(len(m.content) for m in messages)
        progress.on_message(f"[agent] Querying LLM ({llm_config.provider}:{llm_config.model}) for {config.n_candidates} candidates...")
        event_log.llm_request(it, prompt_chars, config.n_candidates, llm_config.model, llm_config.provider, prompt_token_budget=task.constraints.prompt_token_budget)

        llm_t0 = time.monotonic()
        try:
            result = provider.complete(
                messages,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
            )
        except Exception as llm_exc:
            is_context_error = _is_context_overflow_error(llm_exc)
            if is_context_error:
                # Emergency trim: aggressively cut the prompt and retry
                progress.on_message("[agent] Context window overflow — trimming prompt and retrying...")
                from perflab.optimizers.prompt import _trim_to_budget, _estimate_tokens
                current_tokens = sum(_estimate_tokens(m.content) for m in messages)
                emergency_budget = current_tokens // 2  # halve the prompt
                messages = _trim_to_budget(messages, emergency_budget)
                try:
                    result = provider.complete(
                        messages,
                        temperature=llm_config.temperature,
                        max_tokens=llm_config.max_tokens,
                    )
                except Exception:
                    progress.on_message(f"[agent] LLM call failed even after trimming: {llm_exc}")
                    event_log.iteration_complete(it, ctx.best_value, False)
                    continue
            else:
                raise
        llm_latency = time.monotonic() - llm_t0
        ctx.total_llm_calls += 1
        ctx.total_llm_latency += llm_latency
        ctx.total_input_tokens += result.usage.get("input_tokens") or result.usage.get("prompt_tokens", 0)
        ctx.total_output_tokens += result.usage.get("output_tokens") or result.usage.get("completion_tokens", 0)
        progress.on_message(f"[agent] LLM response ({_fmt_elapsed(llm_latency)}): {len(result.content)} chars, tokens: {_fmt_usage(result.usage)}")

        # Parse candidates (with reasoning)
        parsed = parse_candidates(result.content, config.n_candidates)
        candidate_blocks = [blocks for _, blocks in parsed]
        candidate_reasoning = [reasoning for reasoning, _ in parsed]

        event_log.llm_response(it, result.content, result.usage, len(candidate_blocks))

        # Extract build suggestions from LLM reasoning
        for reasoning in candidate_reasoning:
            for ua in extract_build_suggestions(reasoning, iteration=it):
                if ua.flag not in {a["flag"] for a in ctx.user_actions}:
                    ctx.user_actions.append({
                        "suggestion": ua.suggestion,
                        "flag": ua.flag,
                        "iteration": ua.iteration,
                        "source": ua.source,
                    })

        if not candidate_blocks:
            progress.on_message("[agent] No valid candidates parsed from LLM response")
            ctx.history.append({
                "iteration": it,
                "description": "no candidates parsed",
                "value": ctx.best_value,
                "accepted": False,
                "delta": ctx.best_value - ctx.baseline_val,
                "speedup": _calc_speedup(ctx.best_value, ctx.baseline_val),
            })
            if ctx.convergence:
                ctx.convergence.record_failure()
            event_log.iteration_complete(it, ctx.best_value, False)
            # Check convergence
            if ctx.convergence:
                should_stop, reason = ctx.convergence.should_stop()
                if should_stop:
                    ctx.early_stop_reason = reason
                    progress.on_message(f"[agent] {reason}")
                    event_log.early_stop(it, reason)
                    ctx.history.append({"iteration": it, "description": f"early stop: {reason}", "value": ctx.best_value, "accepted": False, "delta": ctx.best_value - ctx.baseline_val, "speedup": _calc_speedup(ctx.best_value, ctx.baseline_val)})
                    break
            continue

        progress.on_message(f"[agent] Parsed {len(candidate_blocks)} candidates")

        # Phase 1: Parallel prescreen (build + correctness, no benchmark)
        prescreen_results = prescreen_candidates_parallel(
            ctx, candidate_blocks, candidate_reasoning,
        )

        # Collect prescreen errors into failure tracking
        for pr in prescreen_results:
            if not pr.get("passed") and pr.get("error"):
                iteration_errors.append(pr["error"])

        # Phase 2: Sequential benchmark (GPU-bound, only for survivors)
        candidates: list[BeamCandidate] = []
        backup_dir = rp.run_dir / "backups" / f"iter{it}"
        use_fast = config.fast_screen and len(candidate_blocks) > 1

        survivors = [pr for pr in prescreen_results if pr.get("passed")]
        failed = [pr for pr in prescreen_results if not pr.get("passed")]

        # Create BeamCandidates for failed prescreens
        for pr in failed:
            ci = pr["ci"]
            err_desc = pr.get("error", {}).get("description", "prescreen failed")
            candidates.append(BeamCandidate(
                iteration=it, index=ci, blocks=pr["blocks"],
                description=f"candidate {ci + 1}: {len(pr['blocks'])} blocks ({err_desc})",
                reasoning=pr.get("reasoning", ""),
            ))
            event_log.candidate_validation(it, ci, False, [err_desc])

        # Benchmark survivors sequentially (GPU contention)
        for pr in survivors:
            ci = pr["ci"]
            blocks = pr["blocks"]
            reasoning = pr.get("reasoning", "")
            candidate, eval_errors = _evaluate_single_candidate(
                ctx, ci, blocks, reasoning, use_fast,
            )
            candidates.append(candidate)
            iteration_errors.extend(eval_errors)

        # Find best improving candidate and accept it
        accepted_any, rel_improvement = _try_accept_best(
            ctx, candidates, backup_dir, use_fast,
        )
        if accepted_any and ctx.convergence and rel_improvement is not None:
            ctx.convergence.record_improvement(rel_improvement)
        elif not accepted_any and ctx.convergence:
            ctx.convergence.record_failure()

        ctx.last_errors = iteration_errors

        # Collect promising alternatives (improved but not best)
        ctx.promising_alternatives = []
        if accepted_any:
            for cand in candidates:
                if cand.value is not None and cand.value > ctx.baseline_val and not cand.accepted:
                    ctx.promising_alternatives.append({
                        "description": cand.description,
                        "reasoning": cand.reasoning,
                        "value": cand.value,
                        "improvement": round(cand.value / ctx.baseline_val, 2) if ctx.baseline_val > 0 else 0,
                    })
            # Sort by value descending
            ctx.promising_alternatives.sort(key=lambda x: x.get("value", 0), reverse=True)
            ctx.promising_alternatives = ctx.promising_alternatives[:3]  # Top 3

        # Accumulate structured failure memory from this iteration's errors
        for ci, cand in enumerate(candidates):
            if cand.value is None:  # Failed candidate
                reasoning_text = cand.reasoning or cand.description
                for err in iteration_errors:
                    if f"candidate {ci + 1}" in err.get("description", ""):
                        ctx.failure_memory.append({
                            "iteration": it,
                            "strategy": reasoning_text[:200],
                            "failure_type": err.get("type", "unknown"),
                            "reason": err.get("description", "")[:300],
                            "profiler_context": err.get("output", "")[:200] if err.get("output") else None,
                        })
                        break

        iter_elapsed = time.monotonic() - iter_start
        event_log.iteration_complete(it, ctx.best_value, accepted_any)
        progress.on_message(f"[agent] Iteration {it} completed in {_fmt_elapsed(iter_elapsed)}")

        # Save checkpoint for crash recovery
        try:
            checkpoint_path.write_text(
                json.dumps(ctx.to_dict(), indent=2), encoding="utf-8",
            )
        except (OSError, TypeError):
            logger.warning("Failed to save checkpoint", exc_info=True)

        # Check convergence
        if ctx.convergence:
            should_stop, reason = ctx.convergence.should_stop()
            if should_stop:
                ctx.early_stop_reason = reason
                progress.on_message(f"[agent] {reason}")
                event_log.early_stop(it, reason)
                ctx.history.append({
                    "iteration": it,
                    "description": f"early stop: {reason}",
                    "value": ctx.best_value,
                    "accepted": False,
                    "delta": ctx.best_value - ctx.baseline_val,
                    "speedup": _calc_speedup(ctx.best_value, ctx.baseline_val),
                })
                break

    # --- Post-optimization explanation ---
    optimization_summary_text: str | None = None
    bench_data_final = {}
    bench_json = rp.run_dir / "bench.json"
    if bench_json.exists():
        bench_data_final = json.loads(bench_json.read_text(encoding="utf-8"))
    profiler_summaries_final = _load_profiler_summaries(rp.artifacts_dir)

    if ctx.best_value != ctx.baseline_val and ctx.accepted_patches:
        try:
            optimization_summary_text, summary_usage, summary_latency = _generate_optimization_summary(
                ctx,
                device=bench_data_final.get("meta", {}).get("device", "unknown"),
                profiler_summaries=profiler_summaries_final,
            )
            ctx.total_llm_calls += 1
            ctx.total_llm_latency += summary_latency
            ctx.total_input_tokens += summary_usage.get("input_tokens") or summary_usage.get("prompt_tokens", 0)
            ctx.total_output_tokens += summary_usage.get("output_tokens") or summary_usage.get("completion_tokens", 0)
            if optimization_summary_text:
                (rp.run_dir / "optimization_summary.md").write_text(
                    optimization_summary_text, encoding="utf-8",
                )
        except Exception as exc:
            progress.on_message(f"[agent] Failed to generate optimization summary: {exc}")

    event_log.run_complete(ctx.best_value, ctx.best_iter, ctx.baseline_val, config.max_iters, ctx.total_llm_calls)

    # --- Generate reports ---
    progress.on_message("\n[agent] Generating reports...")
    llm_stats = {
        "model": llm_config.model,
        "provider": llm_config.provider,
        "total_calls": ctx.total_llm_calls,
        "total_input_tokens": ctx.total_input_tokens,
        "total_output_tokens": ctx.total_output_tokens,
        "total_llm_latency_s": ctx.total_llm_latency,
    }
    # Determine detected GPU name for hardware mismatch reporting
    detected_hw: str | None = None
    if sysinfo.get("nvidia_gpus"):
        detected_hw = sysinfo["nvidia_gpus"][0].get("name")

    # Regenerate the roofline PNG with the full optimization trail before reporting
    _regenerate_roofline_with_history(ctx)

    report_data = generate_reports(ReportParams(
        run_dir=rp.run_dir,
        run_id=rp.run_id,
        task_name=task.name,
        metric_name=task.benchmark.metric.name,
        metric_mode=task.benchmark.metric.mode,
        program_type=task.program_type,
        history=ctx.history,
        baseline_val=ctx.baseline_val,
        best_value=ctx.best_value,
        best_iter=ctx.best_iter,
        early_stop_reason=ctx.early_stop_reason,
        optimization_summary_text=optimization_summary_text,
        analysis_thresholds=task.analysis_thresholds,
        accepted_patches=ctx.accepted_patches,
        roofline_peaks=_resolve_roofline_peaks(task),
        llm_stats=llm_stats,
        target_hardware=task.target_hardware,
        detected_hardware=detected_hw,
        build_cmd=task.build.cmd if task.build else None,
        secondary_metric_name=task.benchmark.secondary_metric.name if task.benchmark.secondary_metric else None,
        secondary_metric_mode=task.benchmark.secondary_metric.mode if task.benchmark.secondary_metric else None,
        top_n=task.constraints.top_n,
        user_actions=ctx.user_actions or None,
    ))

    # Update meta with completion status
    run_store.update_meta(rp.run_id, {
        "status": "completed",
        "best_value": ctx.best_value,
        "baseline_value": ctx.baseline_val,
        "completed_at": time.strftime("%Y%m%d-%H%M%S"),
        "program_type": task.program_type,
    })

    # --- Surface user actions (build suggestions the LLM couldn't apply) ---
    if ctx.user_actions:
        (rp.run_dir / "user_actions.json").write_text(
            json.dumps(ctx.user_actions, indent=2), encoding="utf-8",
        )
        progress.on_message("\n[agent] === USER ACTION REQUIRED ===")
        progress.on_message("[agent] The optimizer suggested build/compilation changes that require manual task.yaml updates:")
        for a in ctx.user_actions:
            progress.on_message(f"[agent]   • {a['suggestion']} (iter {a['iteration']})")
        progress.on_message(f"[agent] Details saved to {rp.run_dir / 'user_actions.json'}")

    total_wall = time.monotonic() - ctx.wall_start
    progress.on_message(f"[agent] Done in {_fmt_elapsed(total_wall)}. Best {task.benchmark.metric.name} = {ctx.best_value:.6g} at iter {ctx.best_iter}")
    progress.on_message(f"[agent] Reports: {rp.run_dir}")

    return AgentResult(
        best_value=ctx.best_value,
        best_iter=ctx.best_iter,
        baseline_value=ctx.baseline_val,
        history=ctx.history,
        run_dir=rp.run_dir,
    )


def _generate_optimization_summary(
    ctx: AgentContext,
    device: str = "unknown",
    profiler_summaries: dict[str, dict] | None = None,
) -> tuple[str | None, dict, float]:
    """Call LLM to generate a concise explanation of what was optimized and why.

    Returns (summary_text, usage_dict, latency_seconds).
    """
    from perflab.llm.base import Message

    provider = ctx.provider
    llm_config = ctx.llm_config
    task = ctx.task
    baseline_val = ctx.baseline_val
    best_value = ctx.best_value
    accepted_patches = ctx.accepted_patches

    speedup = _calc_speedup(best_value, baseline_val)
    patches_desc = []
    for p in accepted_patches:
        patches_desc.append(f"- Iter {p['iteration']}: {p['description']}")
        if p.get("reasoning"):
            patches_desc.append(f"  Reasoning: {p['reasoning'][:300]}")

    # Build profiler context so the LLM doesn't hallucinate hardware claims
    profiler_context = f"Device: {device}\n"
    if profiler_summaries:
        torch_summary = profiler_summaries.get("torch_profiler", {})
        cpu_gpu = torch_summary.get("cpu_vs_gpu", {})
        cpu_us = cpu_gpu.get("total_cpu_op_us", 0)
        gpu_us = cpu_gpu.get("total_gpu_kernel_us", 0)
        if cpu_us > 0 or gpu_us > 0:
            profiler_context += f"CPU time: {cpu_us/1000:.1f}ms, GPU kernel time: {gpu_us/1000:.1f}ms\n"
        if device == "mps":
            profiler_context += (
                "Note: This runs on Apple Silicon (MPS). The torch profiler shows 0 GPU "
                "kernel time because MPS Metal operations are not visible to the profiler. "
                "Do NOT claim GPU utilization improvements — say 'device' or 'MPS' instead.\n"
            )

    user_content = (
        f"You previously optimized this code. Here is a summary of what was changed:\n\n"
        f"Accepted patches:\n" + "\n".join(patches_desc) + "\n\n"
        f"{profiler_context}\n"
        f"Before: {baseline_val:.6g} {task.benchmark.metric.name}\n"
        f"After: {best_value:.6g} {task.benchmark.metric.name}\n"
        f"Speedup: {speedup:.2f}x\n\n"
        f"Write a concise but comprehensive technical explanation structured as:\n"
        f"1. WHAT WORKED: For each accepted patch, explain what optimization was applied "
        f"and the technical reason it improved performance (e.g., reduced cache misses, "
        f"enabled vectorization, eliminated redundant computation).\n"
        f"2. WHY IT WORKED: Explain the underlying performance principle — what hardware "
        f"or algorithmic bottleneck was addressed.\n\n"
        f"Reference specific code changes. "
        f"Be accurate about the hardware — only mention GPU if profiler data confirms GPU activity."
    )
    messages = [
        Message(role="system", content="You are a performance engineering expert. Write concise technical summaries."),
        Message(role="user", content=user_content),
    ]
    t0 = time.monotonic()
    result = provider.complete(messages, temperature=0.3, max_tokens=512)
    latency = time.monotonic() - t0
    text = result.content.strip() if result.content else None
    return text, result.usage, latency


def _run_bench(
    ctx: AgentContext,
    do_profiles: bool,
    capture_diagnostics: bool = False,
) -> tuple[dict, float | None, float | None, CompilerDiagnostics | None]:
    """Run correctness + benchmark + optional profiles.

    Delegates to the shared pipeline in runners/pipeline.py.
    Returns (bench_dict, bench_wall_time_s, profiled_wall_time_s, diagnostics).
    """
    from perflab.runners.pipeline import run_pipeline

    result = run_pipeline(
        task=ctx.task,
        run_dir=ctx.rp.run_dir,
        artifacts_dir=ctx.rp.artifacts_dir,
        do_profiles=do_profiles,
        capture_diagnostics=capture_diagnostics,
        apply_build_overrides=True,
        progress_fn=ctx.progress.on_message,
    )
    return result.bench, result.bench_wall_s, result.profile_wall_s, result.diagnostics
