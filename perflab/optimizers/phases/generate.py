from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from perflab.analyzers.bottleneck_analyzer import compute_source_hints, diagnose_bottlenecks
from perflab.analyzers.compiler_diagnostics import CompilerDiagnostics, cross_reference_diagnostics
from perflab.analyzers.user_actions import extract_build_suggestions
from perflab.memory.run_store import load_profiler_summaries
from perflab.optimizers.event_log import AgentEventLog
from perflab.optimizers.patch import SearchReplaceBlock, read_source_files
from perflab.optimizers.progress import AgentProgress, fmt_elapsed, fmt_usage
from perflab.optimizers.prompt import PromptContext, build_prompt, parse_candidates
from perflab.task_spec import TaskSpec

if TYPE_CHECKING:
    from perflab.optimizers.agent import AgentContext

logger = logging.getLogger(__name__)


@dataclass
class GenerateResult:
    """Output of one generate-phase pass: parsed LLM candidates for this iteration."""

    candidate_blocks: list[list[SearchReplaceBlock]] = field(default_factory=list)
    candidate_reasoning: list[str] = field(default_factory=list)
    llm_failed: bool = False


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
        except Exception:  # noqa: BLE001 -- best-effort prompt enrichment, must not block prompt building
            logger.warning("GPU attribution failed", exc_info=True)
    return gpu_attrib_dicts


def _build_diagnostics_context(
    profiler_summaries: dict[str, dict],
    latest_diagnostics: CompilerDiagnostics | None,
    cpu_isa: dict | None,
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
        except Exception:  # noqa: BLE001 -- best-effort prompt enrichment, must not block prompt building
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
    ctx: AgentContext,
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
        except Exception:  # noqa: BLE001 -- best-effort prompt enrichment, must not block prompt building
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
        except Exception:  # noqa: BLE001 -- best-effort prompt enrichment, must not block prompt building
            logger.warning("Roofline auto-detect failed", exc_info=True)

    return roofline_dict


def build_iteration_prompt(ctx: AgentContext) -> list:
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

    source_files = read_source_files(task)
    profiler_summaries = load_profiler_summaries(rp.artifacts_dir)

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
                improved = [d for d in deltas if d.direction == "improved"]
                regressed = [d for d in deltas if d.direction == "regressed"]
                if improved or regressed:
                    parts = []
                    if improved:
                        parts.append(f"{len(improved)} improved")
                    if regressed:
                        parts.append(f"{len(regressed)} regressed")
                    progress.on_message(f"[agent] Profile diff: {', '.join(parts)}")
        except Exception:  # noqa: BLE001 -- best-effort prompt enrichment, must not block prompt building
            logger.warning("Profile diff computation failed", exc_info=True)

    # Build flag recommendations (static ISA + dynamic profiler-driven)
    build_flag_dicts: list[dict] | None = None
    if task.build:
        try:
            from perflab.analyzers.build_flags import (
                recommend_build_flags,
                recommend_flags_from_profiling,
            )
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
        except Exception:  # noqa: BLE001 -- best-effort prompt enrichment, must not block prompt building
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


def run(ctx: AgentContext) -> GenerateResult:
    """Generate phase: build the iteration prompt, call the LLM, and parse candidates.

    Mutates ctx.total_llm_calls / total_llm_latency / total_input_tokens /
    total_output_tokens / user_actions / prev_summaries.
    """
    task = ctx.task
    config = ctx.config
    llm_config = ctx.llm_config
    provider = ctx.provider
    progress = ctx.progress
    event_log = ctx.event_log
    it = ctx.iteration

    messages = build_iteration_prompt(ctx)

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
        except Exception:  # noqa: BLE001 -- best-effort event log entry, must not block the LLM call
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
            from perflab.optimizers.prompt import _estimate_tokens, _trim_to_budget
            current_tokens = sum(_estimate_tokens(m.content) for m in messages)
            emergency_budget = current_tokens // 2  # halve the prompt
            messages = _trim_to_budget(messages, emergency_budget)
            try:
                result = provider.complete(
                    messages,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens,
                )
            except Exception:  # noqa: BLE001 -- provider-agnostic: any exception here means trimming didn't help, report failure and skip the iteration
                progress.on_message(f"[agent] LLM call failed even after trimming: {llm_exc}")
                return GenerateResult(llm_failed=True)
        else:
            raise
    llm_latency = time.monotonic() - llm_t0
    ctx.total_llm_calls += 1
    ctx.total_llm_latency += llm_latency
    ctx.total_input_tokens += result.usage.get("input_tokens") or result.usage.get("prompt_tokens", 0)
    ctx.total_output_tokens += result.usage.get("output_tokens") or result.usage.get("completion_tokens", 0)
    progress.on_message(f"[agent] LLM response ({fmt_elapsed(llm_latency)}): {len(result.content)} chars, tokens: {fmt_usage(result.usage)}")

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

    return GenerateResult(candidate_blocks=candidate_blocks, candidate_reasoning=candidate_reasoning)
