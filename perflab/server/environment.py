"""MCP tools for environment readiness and CI regression checks."""
from __future__ import annotations

from pathlib import Path

from perflab.server.core import mcp

# ===========================================================================
# Environment
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": True})
def get_peaks(target: str = "auto", cuda_index: int | None = None) -> dict:
    """Show inferred roofline peaks and detected hardware devices (CUDA GPUs, Metal/MPS GPUs, TPU)."""
    from perflab.roofline_peaks import infer_peaks, list_cuda_gpus, list_metal_gpus, selection_hints

    result: dict = {
        "target": target,
        "cuda_gpus": list_cuda_gpus(),
        "metal_gpus": list_metal_gpus(),
    }

    peaks = infer_peaks(target, preferred_cuda_index=cuda_index)
    if peaks:
        result["peaks"] = {
            "device": peaks.device,
            "source": peaks.source,
            "peak_tflops": peaks.peak_tflops,
            "peak_mem_bw_gbs": peaks.peak_mem_bw_gbs,
        }
        if peaks.dtype_peaks:
            result["peaks"]["dtype_peaks"] = peaks.dtype_peaks
    else:
        result["peaks"] = None

    result["hints"] = selection_hints()
    return result


@mcp.tool(annotations={"readOnlyHint": True})
def doctor_check(check_profilers: bool = True, check_llm: bool = True, check_all: bool = False) -> dict:
    """Check environment readiness: Python version, packages, profiler tools, hardware detection, LLM config."""
    from perflab.doctor import run_doctor

    results = run_doctor(check_profilers=check_profilers, check_llm=check_llm, check_all=check_all)

    checks = [
        {"name": r.name, "status": r.status, "message": r.message}
        for r in results
    ]
    passes = sum(1 for r in results if r.status == "pass")
    warns = sum(1 for r in results if r.status == "warn")
    fails = sum(1 for r in results if r.status == "fail")

    return {
        "checks": checks,
        "summary": {"passed": passes, "warnings": warns, "failures": fails},
        "ready": fails == 0,
    }


# ===========================================================================
# CI regression checks
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
def ci_check(task_yaml: str, baseline_file: str | None = None) -> dict:
    """Run a CI regression check: benchmark current code against a saved baseline.

    Returns pass/fail, regression %, tolerance %, bench variance warnings,
    and profiler metric regressions (when NCU data is available in both
    the baseline and a recent profile run).
    """
    from perflab.ci import run_ci_check
    from perflab.task_spec import TaskSpec

    task = TaskSpec.load(Path(task_yaml))
    bp = Path(baseline_file) if baseline_file else None
    result = run_ci_check(task, bp)
    return result.to_dict()


@mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
def save_ci_baseline(task_yaml: str, baseline_file: str | None = None) -> dict:
    """Run benchmark and save result as CI baseline for future regression checks.

    Automatically includes NCU profiler data from the most recent profile
    run (if available) for future profiler regression detection.
    """
    from perflab.ci import save_baseline
    from perflab.task_spec import TaskSpec

    task = TaskSpec.load(Path(task_yaml))
    bp = Path(baseline_file) if baseline_file else None
    saved_path = save_baseline(task, bp)
    return {"baseline_saved": str(saved_path)}
