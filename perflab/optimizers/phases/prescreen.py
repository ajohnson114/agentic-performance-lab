from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from perflab.optimizers.patch import SearchReplaceBlock, apply_patch, validate_patch
from perflab.runners.correctness import run_correctness
from perflab.task_spec import TaskSpec
from perflab.tools.isolation import IsolationPolicy

if TYPE_CHECKING:
    from perflab.optimizers.agent import AgentContext

__all__ = ["run"]


def _prescreen_candidate(
    ci: int,
    blocks: list[SearchReplaceBlock],
    reasoning: str,
    task: TaskSpec,
    ws: Path,
    it: int,
    isolation: IsolationPolicy | None = None,
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
        "notices": [],
    }

    # Validate patch
    validation_errors = validate_patch(
        blocks, task.edit_policy.allowed_paths, ws, notices=result["notices"]
    )
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
            isolation=isolation,
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

    except Exception as exc:  # noqa: BLE001 -- untrusted candidate's build/correctness can fail in arbitrary ways; must feed back as a prescreen error, not crash the parallel pool
        # Exception text can embed candidate-controlled subprocess output --
        # keep it in "output" (sanitized at prompt-render time), not "description".
        result["error"] = {
            "type": "prescreen_error",
            "description": f"prescreen failed ({type(exc).__name__})",
            "output": str(exc)[:1000],
        }
        return result
    finally:
        if temp_dir and temp_dir.exists():
            try:
                _shutil.rmtree(temp_dir)
            except Exception:  # noqa: BLE001 -- best-effort temp dir cleanup, nothing more to do if this fails
                pass


def run(
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
                ctx.config.isolation,
            )
            futures[future] = ci

        for future in as_completed(futures):
            ci = futures[future]
            try:
                results[ci] = future.result()
            except Exception as exc:  # noqa: BLE001 -- worker thread can raise arbitrary errors; must feed back as a prescreen error, not crash the pool
                results[ci] = {
                    "ci": ci,
                    "blocks": candidate_blocks[ci],
                    "reasoning": candidate_reasoning[ci] if ci < len(candidate_reasoning) else "",
                    "passed": False,
                    "error": {"type": "prescreen_error", "description": str(exc), "output": ""},
                }

    for r in results:
        for note in r.get("notices", []):
            progress.on_message(
                f"[agent]   Patch note (candidate {r.get('ci', 0) + 1}): {note}"
            )

    passed = sum(1 for r in results if r.get("passed"))
    failed = n - passed
    progress.on_message(f"[agent] Prescreen: {passed} passed, {failed} failed")

    return results
