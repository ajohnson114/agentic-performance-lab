"""Shared build -> correctness -> benchmark -> profile pipeline.

Extracts the common execution sequence used by both the LLM agent loop
(agent.py) and the grid-search orchestrator (orchestrator.py), eliminating
~120 lines of duplicated build/bench/profile logic.
"""
from __future__ import annotations

import json
import logging
import shlex
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from perflab.analyzers.compiler_diagnostics import (
    CompilerDiagnostics,
    detect_compiler,
    get_diagnostic_build_flags,
    get_diagnostic_env_vars,
    parse_compiler_output,
)
from perflab.runners.benchmark import run_benchmark, validate_contract
from perflab.runners.correctness import run_correctness
from perflab.task_spec import TaskSpec
from perflab.tools.isolation import IsolationPolicy
from perflab.tools.shell import run_cmd

if TYPE_CHECKING:
    from perflab.optimizers.agent import AgentContext

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a full build -> correctness -> benchmark -> profile run."""

    bench: dict
    bench_wall_s: float | None = None
    profile_wall_s: float | None = None
    diagnostics: CompilerDiagnostics | None = None
    artifacts: dict[str, str] = field(default_factory=dict)


def run_pipeline(
    task: TaskSpec,
    run_dir: Path,
    artifacts_dir: Path,
    *,
    do_profiles: bool = False,
    capture_diagnostics: bool = False,
    apply_build_overrides: bool = False,
    validate_contract_spec: bool = False,
    save_logs: bool = False,
    progress_fn: Callable[[str], None] | None = None,
    isolation: IsolationPolicy | None = None,
) -> PipelineResult:
    """Execute the build -> correctness -> benchmark -> profile pipeline.

    Args:
        task: Task specification.
        run_dir: Run directory (for logs, bench.json copy).
        artifacts_dir: Directory for profiler artifacts and summaries.
        do_profiles: Run profilers after benchmarking.
        capture_diagnostics: Capture compiler diagnostics (remark flags, env vars).
        apply_build_overrides: Apply build_overrides.json if present.
        validate_contract_spec: Raise on contract violations in bench.json.
        save_logs: Write stdout/stderr logs to run_dir/logs/.
        progress_fn: Optional callback for progress messages.
        isolation: Optional OS-level sandbox policy applied to the
            correctness and benchmark subprocesses (perflab.tools.isolation).
            Build and profiler invocations are NOT wrapped: profilers (perf,
            nsys, ncu) need ptrace/driver/sysfs access a sandbox denies, and
            build commands are task-defined, not candidate-authored.

    Returns:
        PipelineResult with bench dict, timing, diagnostics, and artifacts.
    """

    def _msg(text: str) -> None:
        if progress_fn:
            progress_fn(text)

    ws = task.workspace
    (ws / "out").mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build (optional)
    # ------------------------------------------------------------------
    build_stderr = ""
    detected_compiler = "gcc"
    if task.build is not None:
        build_cmd_parts = shlex.split(task.build.cmd)

        # Apply build_overrides.json if present (agent-proposed compiler flags)
        if apply_build_overrides:
            try:
                from perflab.analyzers.build_overrides import (
                    apply_build_overrides as _apply_overrides,
                )
                from perflab.analyzers.build_overrides import (
                    load_build_overrides,
                )
                overrides = load_build_overrides(ws, allow_fast_math=task.constraints.allow_fast_math)
                if overrides:
                    build_cmd_str = _apply_overrides(task.build.cmd, overrides)
                    build_cmd_parts = shlex.split(build_cmd_str)
            except Exception:  # noqa: BLE001 -- best-effort optional feature, fall back to the unmodified build command
                logger.warning("Build override application failed", exc_info=True)

        if capture_diagnostics and build_cmd_parts:
            detected_compiler = detect_compiler(task.build.cmd)
            base = Path(build_cmd_parts[0]).name
            if base in ("g++", "gcc", "c++", "clang++", "clang", "nvcc"):
                build_cmd_parts.extend(
                    get_diagnostic_build_flags(task.program_type, compiler=detected_compiler)
                )

        bres = run_cmd(build_cmd_parts, cwd=ws)
        build_stderr = bres.stderr
        if save_logs:
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "build.stdout.txt").write_text(bres.stdout, encoding="utf-8")
            (logs_dir / "build.stderr.txt").write_text(bres.stderr, encoding="utf-8")
        if bres.returncode != task.build.expected_exit:
            raise RuntimeError(f"Build failed with code {bres.returncode}")

    # ------------------------------------------------------------------
    # 2. Correctness
    # ------------------------------------------------------------------
    cres = run_correctness(
        task.correctness.cmd, cwd=ws,
        program_type=task.program_type,
        rlimit_as_gb=task.constraints.rlimit_as_gb,
        isolation=isolation,
    )
    if save_logs:
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "correctness.stdout.txt").write_text(cres.stdout, encoding="utf-8")
        (logs_dir / "correctness.stderr.txt").write_text(cres.stderr, encoding="utf-8")
    if cres.returncode != task.correctness.expected_exit:
        raise RuntimeError(f"Correctness failed with code {cres.returncode}")

    # ------------------------------------------------------------------
    # 3. Benchmark (timed)
    # ------------------------------------------------------------------
    bench_env = (
        get_diagnostic_env_vars(task.program_type, compiler=detected_compiler)
        if capture_diagnostics else None
    )
    t0 = time.perf_counter()
    bench_res, bench = run_benchmark(
        task.benchmark.cmd, cwd=ws, env=bench_env,
        program_type=task.program_type,
        rlimit_as_gb=task.constraints.rlimit_as_gb,
        isolation=isolation,
    )
    bench_wall = time.perf_counter() - t0
    bench_stderr = bench_res.stderr if capture_diagnostics else ""
    if save_logs:
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "bench.stdout.txt").write_text(bench_res.stdout, encoding="utf-8")
        (logs_dir / "bench.stderr.txt").write_text(bench_res.stderr, encoding="utf-8")
    shutil.copy2(ws / "out" / "bench.json", run_dir / "bench.json")

    # Contract validation (optional — orchestrator does this, agent does it elsewhere)
    if validate_contract_spec:
        contract_errors = validate_contract(bench, task.contract)
        if contract_errors:
            raise RuntimeError(f"Contract violation: {contract_errors}")

    # ------------------------------------------------------------------
    # 4. Compiler diagnostics
    # ------------------------------------------------------------------
    diagnostics: CompilerDiagnostics | None = None
    if capture_diagnostics:
        diagnostics = parse_compiler_output(
            task.program_type, build_stderr, bench_stderr, compiler=detected_compiler,
        )
        if diagnostics.summary:
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "compiler_diagnostics.txt").write_text(
                diagnostics.summary, encoding="utf-8",
            )

    # ------------------------------------------------------------------
    # 5. Artifacts collection
    # ------------------------------------------------------------------
    artifacts: dict[str, str] = {}

    # SASS extraction for CUDA tasks
    if do_profiles and task.program_type == "cuda" and task.build:
        try:
            from perflab.profilers import extract_sass_from_build
            sass_results = extract_sass_from_build(task.build.cmd, ws, artifacts_dir)
            if sass_results:
                artifacts["sass_dump"] = str(artifacts_dir / "sass_dump.txt")
        except Exception:  # noqa: BLE001 -- best-effort optional artifact, must not abort the pipeline
            logger.warning("SASS extraction failed", exc_info=True)

    # ------------------------------------------------------------------
    # 6. Roofline (optional)
    # ------------------------------------------------------------------
    try:
        if task.roofline is not None:
            from perflab.reporting.roofline import compute_roofline_point, write_roofline_png
            from perflab.roofline_peaks import _lookup_dtype_peaks, _lookup_l2_bw

            ncu_summary_path = artifacts_dir / "ncu_summary.json"
            measured_dram: float | None = None
            if ncu_summary_path.exists():
                try:
                    ncu_data = json.loads(ncu_summary_path.read_text(encoding="utf-8"))
                    measured_dram = ncu_data.get("dram_bytes_total")
                except json.JSONDecodeError:
                    logger.warning("Failed to parse NCU summary", exc_info=True)

            pt = compute_roofline_point(bench, measured_dram_bytes=measured_dram)
            if pt is not None:
                roof_png = artifacts_dir / "roofline.png"
                title = task.roofline.title or f"Roofline \u2014 {task.name}"
                dtype_peaks = _lookup_dtype_peaks(task.target_hardware or "")
                l2_bw = _lookup_l2_bw(task.target_hardware or "")
                write_roofline_png(
                    roof_png,
                    point=pt,
                    peak_tflops=float(task.roofline.peak_tflops),
                    peak_mem_bw_gbs=float(task.roofline.peak_mem_bw_gbs),
                    title=title,
                    dtype_peaks=dtype_peaks,
                    l2_bw_gbs=l2_bw,
                )
                artifacts["roofline"] = str(roof_png)
    except Exception as exc:  # noqa: BLE001 -- best-effort optional artifact, must not abort the pipeline
        _msg(f"[pipeline] Roofline generation failed: {exc}")

    # ------------------------------------------------------------------
    # 7. Profilers (timed)
    # ------------------------------------------------------------------
    prof_wall: float | None = None
    if do_profiles:
        tp0 = time.perf_counter()
        from perflab.profilers import select_profilers

        for profiler in select_profilers(task):
            if not profiler.is_available():
                artifacts[profiler.name] = "(not available)"
                continue
            try:
                pr = profiler.run(task.benchmark.cmd, cwd=ws, artifacts_dir=artifacts_dir)
                (artifacts_dir / f"{profiler.name}_summary.json").write_text(
                    json.dumps(pr.summary, indent=2), encoding="utf-8",
                )
                for k, v in pr.artifacts.items():
                    artifacts[k] = str(artifacts_dir / Path(v).name)
            except Exception as exc:  # noqa: BLE001 -- a single failed profiler must not abort the others
                error_summary = {"error": f"{profiler.name} failed: {exc}"}
                (artifacts_dir / f"{profiler.name}_summary.json").write_text(
                    json.dumps(error_summary, indent=2), encoding="utf-8",
                )
                artifacts[profiler.name] = f"(error: {exc})"
                _msg(f"[pipeline] Profiler {profiler.name} failed: {exc}")

        prof_wall = time.perf_counter() - tp0

    return PipelineResult(
        bench=bench,
        bench_wall_s=bench_wall,
        profile_wall_s=prof_wall,
        diagnostics=diagnostics,
        artifacts=artifacts,
    )


def run_pipeline_for_ctx(
    ctx: AgentContext,
    do_profiles: bool,
    capture_diagnostics: bool = False,
) -> tuple[dict, float | None, float | None, CompilerDiagnostics | None]:
    """Run correctness + benchmark + optional profiles for an agent iteration.

    Thin adapter from AgentContext to run_pipeline(), shared by the baseline
    phase (initial profiling) and the evaluate phase (post-accept re-profiling).
    Returns (bench_dict, bench_wall_time_s, profiled_wall_time_s, diagnostics).
    """
    result = run_pipeline(
        task=ctx.task,
        run_dir=ctx.rp.run_dir,
        artifacts_dir=ctx.rp.artifacts_dir,
        do_profiles=do_profiles,
        capture_diagnostics=capture_diagnostics,
        apply_build_overrides=True,
        progress_fn=ctx.progress.on_message,
        isolation=ctx.config.isolation,
    )
    return result.bench, result.bench_wall_s, result.profile_wall_s, result.diagnostics
