"""On-demand diagnostic tools for the generate-phase LLM tool loop.

Phase 1 of making the optimizer agent tool-using: instead of stuffing every
possible diagnostic into one giant prompt, the LLM can pull specific data
on demand while it's deciding what to try next -- "show me the bottleneck
ranking", "give me the dossier for this kernel", "diff iteration 0 vs now",
and (the important one) "re-run ncu, I want more detail before I decide".

This module is purely additive to the diagnostic/prompt-building side. It
must NEVER touch the benchmark/accept/contract/anti-gaming layer -- nothing
here can influence whether a candidate patch is accepted or rejected. Every
tool either reads data already gathered by a normal profiling pass
(``ctx.profiler_summaries``) or re-runs one of the profiler instances
``perflab.profilers.select_profilers`` already returns for this task's
program_type, against the CURRENT already-accepted workspace -- never a
not-yet-proposed candidate, and never a new measurement mechanism.

Profiling itself stays broad-by-default (every applicable profiler still
runs in full during the normal baseline/re-profile passes); what's adaptive
here is the agent's ability to re-run or dig deeper into one SPECIFIC
profiler once it has seen where the bottleneck actually is.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from perflab.llm.base import ToolCall, ToolSpec

if TYPE_CHECKING:
    from perflab.optimizers.agent import AgentContext

logger = logging.getLogger(__name__)


DIAGNOSTIC_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="get_bottlenecks",
        description=(
            "Diagnose performance bottlenecks from the profiler data gathered so "
            "far (NCU, NSys, linux perf, Metal trace, JAX/TPU trace, memray, "
            "eBPF, etc.). Returns ranked findings with root cause, confidence, "
            "and suggested actions. Use this to decide where to focus before "
            "proposing a patch, or call it again after run_profiler to see how "
            "the diagnosis changes with fresh data."
        ),
        parameters={
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Maximum number of ranked bottleneck findings to return. "
                        "Defaults to the task's configured top_n (usually 3)."
                    ),
                },
            },
            "required": [],
        },
    ),
    ToolSpec(
        name="get_kernel_dossier",
        description=(
            "Get a unified deep-dive on one GPU kernel: its share of total GPU "
            "time, launch overhead, the framework op / user-code caller that "
            "triggered it, per-kernel NCU metrics (occupancy, memory "
            "throughput, etc. -- if an ncu profile has been captured), and a "
            "SASS disassembly snippet (for CUDA tasks with a build step). "
            "Requires NSys GPU attribution data -- run the nsys profiler first "
            "(via run_profiler) if it hasn't run yet. Kernel names are matched "
            "fuzzily against attribution/NCU/SASS data, so an exact string is "
            "not required -- use the name as it appears in get_bottlenecks or "
            "prior profiler output."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kernel_name": {
                    "type": "string",
                    "description": (
                        "Name (or distinctive substring) of the GPU kernel to "
                        "inspect, e.g. as it appears in NSys top_kernels or GPU "
                        "attribution output."
                    ),
                },
            },
            "required": ["kernel_name"],
        },
    ),
    ToolSpec(
        name="get_profile_diff",
        description=(
            "Compare profiler metrics (IPC, cache misses, GPU utilization, "
            "hotspot shifts, etc.) between two points in this run. Only two "
            "snapshots are available: iteration 0 is the baseline (before any "
            "changes), and the current iteration number is the latest "
            "profiler data (including anything run_profiler has already added "
            "this turn). PerfLab does not keep a separate historical archive "
            "for every intermediate iteration, so any other iteration number "
            "returns an error instead of silently wrong data."
        ),
        parameters={
            "type": "object",
            "properties": {
                "iteration_a": {
                    "type": "integer",
                    "description": (
                        "First iteration to compare. Use 0 for the baseline "
                        "(pre-optimization) snapshot."
                    ),
                },
                "iteration_b": {
                    "type": "integer",
                    "description": (
                        "Second iteration to compare. Use the current "
                        "iteration number for the latest data."
                    ),
                },
            },
            "required": ["iteration_a", "iteration_b"],
        },
    ),
    ToolSpec(
        name="run_profiler",
        description=(
            "Re-run one specific profiler against the CURRENT accepted code "
            "(never a candidate patch) to get fresh, more detailed diagnostic "
            "data before deciding what to try next -- e.g. run ncu again for "
            "deeper per-kernel metrics on a kernel you just identified as the "
            "bottleneck. Purely diagnostic: it never affects benchmarking, "
            "acceptance, or the contract -- it only updates what get_bottlenecks "
            "/ get_kernel_dossier can see for the rest of this turn. Only "
            "profilers already applicable to this task's program_type can be "
            "run; call with an unknown or unavailable name to see what's "
            "actually available."
        ),
        parameters={
            "type": "object",
            "properties": {
                "profiler_name": {
                    "type": "string",
                    "description": (
                        "Name of the profiler to run, e.g. 'ncu', 'nsys', "
                        "'linux_perf', 'pyspy', 'torch_profiler', 'jax', "
                        "'metal_trace', 'memray', 'ebpf', 'lock_contention', "
                        "'thread_sched', 'power'. Must be one already selected "
                        "for this task's program_type."
                    ),
                },
            },
            "required": ["profiler_name"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool 1: get_bottlenecks
# ---------------------------------------------------------------------------

def _read_bench_device(ctx: AgentContext) -> str | None:
    """Best-effort device string ("cuda", "mps", ...) from this run's bench.json.

    diagnose_bottlenecks uses this to avoid misreporting "GPU underutilized"
    on MPS, where the torch profiler cannot observe Metal GPU kernels.
    Best-effort like the rest of the diagnostic layer: a missing/corrupt
    bench.json just yields no device hint, never an exception.
    """
    bench_json = ctx.rp.run_dir / "bench.json"
    if not bench_json.exists():
        return None
    try:
        data = json.loads(bench_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    device = data.get("meta", {}).get("device")
    return device if isinstance(device, str) else None


def _tool_get_bottlenecks(ctx: AgentContext, args: dict[str, Any]) -> dict:
    """Rule-based bottleneck diagnosis from ctx.profiler_summaries.

    Mirrors perflab.server.analysis.get_bottlenecks's call shape and dict
    formatting, but reads profiler summaries already in memory (the latest
    full profiling pass, plus anything run_profiler has added this turn)
    instead of loading a stored run from RunStore.
    """
    from perflab.analyzers.bottleneck_analyzer import diagnose_bottlenecks

    top_n = args.get("top_n")
    if not isinstance(top_n, int) or top_n <= 0:
        top_n = ctx.task.constraints.top_n

    summaries = ctx.profiler_summaries or {}
    if not summaries:
        return {
            "bottlenecks": [],
            "note": "No profiler data available yet -- run a profiler first.",
        }

    diags = diagnose_bottlenecks(
        summaries,
        ctx.task.program_type,
        top_n=top_n,
        device=_read_bench_device(ctx),
        thresholds=ctx.task.analysis_thresholds,
        system_info=ctx.sysinfo or None,
    )
    return {
        "bottlenecks": [
            {
                "rank": d.rank,
                "bottleneck": d.bottleneck,
                "root_cause": d.root_cause,
                "confidence": d.confidence,
                "suggested_actions": d.suggested_actions,
                "evidence": d.evidence.to_dict(),
            }
            for d in diags
        ],
    }


# ---------------------------------------------------------------------------
# Tool 2: get_kernel_dossier
# ---------------------------------------------------------------------------

def _tool_get_kernel_dossier(ctx: AgentContext, args: dict[str, Any]) -> dict:
    """Unified attribution + NCU + SASS dossier for one named GPU kernel.

    Reuses perflab.analyzers.gpu_attribution.compute_attribution_ranking /
    _match_kernel / build_kernel_dossiers -- the same functions
    perflab.optimizers.phases.generate uses to build kernel dossiers for the
    prompt -- but scoped to a single kernel_name and reading
    ctx.profiler_summaries in-process instead of the whole-prompt path.
    """
    from perflab.analyzers.gpu_attribution import (
        _match_kernel,
        build_kernel_dossiers,
        compute_attribution_ranking,
    )

    kernel_name = args.get("kernel_name")
    if not kernel_name or not isinstance(kernel_name, str):
        return {"error": "kernel_name is required and must be a non-empty string."}

    summaries = ctx.profiler_summaries or {}
    nsys_summary = summaries.get("nsys")
    if not nsys_summary or not nsys_summary.get("cpu_gpu_correlations"):
        return {
            "error": (
                "No NSys GPU attribution data available yet. Kernel dossiers "
                "require CUDA workloads profiled with Nsight Systems -- run "
                "the nsys profiler first (e.g. via run_profiler)."
            ),
        }

    ranking = compute_attribution_ranking(
        nsys_summary,
        summaries.get("linux_perf"),
        torch_summary=summaries.get("torch_profiler"),
        pyspy_summary=summaries.get("pyspy"),
    )
    if not ranking:
        return {"error": "GPU attribution ranking is empty; no kernels to look up."}

    attrib_dicts = [dataclasses.asdict(e) for e in ranking]
    match = _match_kernel(kernel_name, attrib_dicts, key="name")
    if match is None:
        available = [
            a["name"] for a in attrib_dicts if a.get("category") == "gpu-kernel"
        ][:10]
        return {
            "error": f"Kernel {kernel_name!r} not found in GPU attribution ranking.",
            "available_kernels": available,
        }

    ncu_summary = summaries.get("ncu")

    # SASS is a best-effort enrichment (requires cuobjdump + a CUDA build) --
    # its own failure must not blow away the attribution/NCU data that's
    # already useful on its own.
    sass_entries: list[dict] | None = None
    if ctx.task.program_type == "cuda" and ctx.task.build is not None:
        try:
            from perflab.profilers import extract_sass_from_build
            sass_dir = ctx.rp.artifacts_dir / "tool_calls" / "sass"
            sass_entries = extract_sass_from_build(
                ctx.task.build.cmd, ctx.ws, sass_dir, max_kernels=5, context_lines=15,
            )
        except Exception:  # noqa: BLE001 -- best-effort; the dossier is still useful without SASS
            logger.warning("SASS extraction failed for kernel dossier", exc_info=True)
            sass_entries = None

    dossiers = build_kernel_dossiers([match], ncu_summary, sass_entries, max_kernels=1)
    if not dossiers:
        return {"error": f"Could not build a dossier for kernel {kernel_name!r}."}
    return dataclasses.asdict(dossiers[0])


# ---------------------------------------------------------------------------
# Tool 3: get_profile_diff
# ---------------------------------------------------------------------------

def _resolve_iteration_summaries(
    ctx: AgentContext, iteration: int,
) -> tuple[dict[str, dict], str | None]:
    """Map a requested iteration number to a profiler-summaries snapshot.

    PerfLab does not persist a separate profiler-summary archive per
    iteration: ``ctx.rp.artifacts_dir`` is overwritten by every re-profiling
    pass (see perflab.optimizers.phases.evaluate.reprofile_after_accept), and
    only two fixed points survive:

    * iteration 0 -- the baseline snapshot, preserved on disk under
      ``artifacts_baseline/`` (copied once, right after the baseline phase;
      see perflab.optimizers.phases.baseline and how
      perflab.reporting.generate reads it for the report's own profile-diff
      section).
    * the current iteration -- ``ctx.profiler_summaries``, the latest
      in-memory summaries, kept live-updated within a tool loop by
      run_profiler.

    Any other iteration number is not resolvable from what PerfLab keeps on
    disk today, so it returns a clear error rather than silently comparing
    against the wrong (or stale) data.
    """
    if iteration == 0:
        baseline_dir = ctx.rp.run_dir / "artifacts_baseline"
        summaries: dict[str, dict] = {}
        if baseline_dir.exists():
            for p in baseline_dir.glob("*_summary.json"):
                try:
                    summaries[p.stem.replace("_summary", "")] = json.loads(
                        p.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    continue
        if not summaries:
            return {}, "No baseline profiler summaries found (artifacts_baseline/ missing or empty)."
        return summaries, None

    if iteration == ctx.iteration:
        summaries = ctx.profiler_summaries or {}
        if not summaries:
            return {}, f"No profiler summaries available yet for the current iteration ({iteration})."
        return summaries, None

    return {}, (
        f"Iteration {iteration} is not available for diffing. Only two "
        f"profiler-summary snapshots are kept: the baseline (iteration 0) and "
        f"the latest reprofiled state (currently iteration {ctx.iteration}). "
        f"Per-iteration history beyond those two points is not persisted."
    )


def _tool_get_profile_diff(ctx: AgentContext, args: dict[str, Any]) -> dict:
    """Diff profiler summaries between two iterations.

    Reuses perflab.analyzers.profile_diff.compute_profile_diff /
    compute_hotspot_diff -- the same functions perflab.reporting.generate
    uses for the report's baseline-vs-final profile-diff section, and
    perflab.optimizers.phases.generate uses for the previous-vs-current
    iteration diff in the prompt.
    """
    from perflab.analyzers.profile_diff import compute_hotspot_diff, compute_profile_diff

    try:
        iteration_a = int(args["iteration_a"])
        iteration_b = int(args["iteration_b"])
    except (KeyError, TypeError, ValueError):
        return {"error": "iteration_a and iteration_b are required integer arguments."}

    summaries_a, err_a = _resolve_iteration_summaries(ctx, iteration_a)
    if err_a:
        return {"error": err_a}
    summaries_b, err_b = _resolve_iteration_summaries(ctx, iteration_b)
    if err_b:
        return {"error": err_b}

    metric_mode = ctx.task.benchmark.metric.mode
    deltas = compute_profile_diff(summaries_a, summaries_b, metric_mode=metric_mode)
    hotspots = compute_hotspot_diff(summaries_a, summaries_b)
    return {
        "iteration_a": iteration_a,
        "iteration_b": iteration_b,
        "deltas": [dataclasses.asdict(d) for d in deltas],
        "hotspot_shifts": [dataclasses.asdict(h) for h in hotspots],
    }


# ---------------------------------------------------------------------------
# Tool 4: run_profiler
# ---------------------------------------------------------------------------

def _tool_run_profiler(ctx: AgentContext, args: dict[str, Any]) -> dict:
    """Re-run one already-selected profiler against the current workspace.

    Only ever runs one of the Profiler instances
    perflab.profilers.select_profilers already returns for this task's
    program_type -- never a new measurement mechanism -- and only against
    ctx.task.benchmark.cmd on the CURRENT accepted workspace (ctx.ws), the
    same command the normal accept/benchmark path uses for measurement, but
    this call itself is not part of that path: nothing here is scored,
    compared, or fed to the accept/contract/anti-gaming layer. On success,
    the fresh summary is merged into ctx.profiler_summaries so a later
    get_bottlenecks / get_kernel_dossier call in the same tool loop sees it.
    """
    from perflab.profilers import select_profilers
    from perflab.profilers.base import bench_env_passthrough

    profiler_name = args.get("profiler_name")
    if not profiler_name or not isinstance(profiler_name, str):
        return {"error": "profiler_name is required and must be a non-empty string."}

    profilers = select_profilers(ctx.task)
    by_name = {p.name: p for p in profilers}

    profiler = by_name.get(profiler_name)
    if profiler is None:
        return {
            "error": (
                f"Unknown profiler {profiler_name!r} for program_type "
                f"{ctx.task.program_type!r}."
            ),
            "available_profilers": sorted(by_name),
        }
    if not profiler.is_available():
        available = sorted(name for name, p in by_name.items() if p.is_available())
        return {
            "error": (
                f"Profiler {profiler_name!r} is applicable to this task but not "
                f"available on this machine (missing binary/driver/permissions)."
            ),
            "available_profilers": available,
        }

    # Scoped subdir so this on-demand run never collides with the regular
    # phase-driven profiling artifacts under ctx.rp.artifacts_dir.
    scoped_dir = ctx.rp.artifacts_dir / "tool_calls" / f"{profiler_name}_{uuid.uuid4().hex[:8]}"
    scoped_dir.mkdir(parents=True, exist_ok=True)

    with bench_env_passthrough(ctx.task.constraints.env_passthrough):
        result = profiler.run(ctx.task.benchmark.cmd, cwd=ctx.ws, artifacts_dir=scoped_dir)

    try:
        (scoped_dir / f"{profiler_name}_summary.json").write_text(
            json.dumps(result.summary, indent=2, default=str), encoding="utf-8",
        )
    except OSError:
        logger.warning("Failed to persist tool-call profiler summary", exc_info=True)

    # Mutate ctx so the rest of THIS tool loop sees the fresh data.
    ctx.profiler_summaries[profiler_name] = result.summary

    return {
        "profiler": profiler_name,
        "summary": result.summary,
        "artifacts_dir": str(scoped_dir),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Callable[[AgentContext, dict[str, Any]], dict]] = {
    "get_bottlenecks": _tool_get_bottlenecks,
    "get_kernel_dossier": _tool_get_kernel_dossier,
    "get_profile_diff": _tool_get_profile_diff,
    "run_profiler": _tool_run_profiler,
}


def execute_tool(tool_call: ToolCall, ctx: AgentContext) -> str:
    """Dispatch one LLM tool call to its implementation and return a JSON string.

    Always returns a string (this becomes the ``content`` of a ``role="tool"``
    Message) -- a bad tool call (unknown name, bad args, or an exception
    raised inside the implementation) never propagates and never crashes the
    agent iteration; it comes back as ``{"error": "..."}`` instead.
    """
    handler = _HANDLERS.get(tool_call.name)
    if handler is None:
        return json.dumps({
            "error": f"Unknown tool {tool_call.name!r}.",
            "available_tools": sorted(_HANDLERS),
        })

    args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
    try:
        result = handler(ctx, args)
    except Exception as exc:  # noqa: BLE001 -- a bad tool call must not crash the agent iteration
        logger.warning("Tool %s failed", tool_call.name, exc_info=True)
        return json.dumps({"error": f"{tool_call.name} failed: {exc}"})

    try:
        return json.dumps(result, default=str)
    except TypeError:
        logger.warning("Tool %s produced non-JSON-serializable output", tool_call.name)
        return json.dumps({"error": f"{tool_call.name} produced non-serializable output."})
