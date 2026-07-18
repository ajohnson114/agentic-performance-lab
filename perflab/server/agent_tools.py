"""MCP tools for profiling and agent optimization runs.

Holds the background job state (single-slot executor, agent lock, job
registry) shared by start_agent / optimize_task / get_agent_progress.
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastmcp import Context

from perflab.server.core import _guard_output_size, mcp

# ---------------------------------------------------------------------------
# Background agent run management
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=1)
_active_runs: dict[str, dict] = {}
_lock = threading.Lock()
_agent_lock = threading.Lock()

# _active_runs must not grow forever: terminal (completed/failed) jobs beyond
# this cap are evicted oldest-first whenever a new job is registered. In-flight
# jobs are never evicted.
_MAX_TERMINAL_RUNS = 50


def _register_job(job_id: str, entry: dict) -> None:
    """Insert a job entry, evicting the oldest terminal jobs beyond the cap."""
    with _lock:
        _active_runs[job_id] = entry
        terminal = [
            jid for jid, info in _active_runs.items()
            if info.get("status") in ("completed", "failed")
        ]
        for jid in terminal[: max(0, len(terminal) - _MAX_TERMINAL_RUNS)]:
            del _active_runs[jid]


# ===========================================================================
# Optimization — profiling and agent runs
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
def profile_task(task_yaml: str) -> dict:
    """Run baseline profiling for a task. Returns profiler summary paths."""
    from perflab.memory.run_store import load_profiler_summaries
    from perflab.orchestrator import profile_only
    from perflab.task_spec import TaskSpec

    task_file = Path(task_yaml)
    task = TaskSpec.load(task_file)
    run_dir = profile_only(task)

    return _guard_output_size({
        "run_dir": str(run_dir),
        "profiler_summaries": load_profiler_summaries(run_dir / "artifacts"),
    })


def _resolve_isolation(task_file: Path):
    """Resolve the sandbox policy for agent runs: task.yaml > perflab.yaml/config.

    Mirrors the CLI ``perflab agent`` resolution (minus the CLI flag) so
    MCP-launched runs honor the same isolation the task/config request —
    previously these tools built AgentConfig without isolation, silently
    running candidate code unsandboxed even when task.yaml asked for a
    sandbox.
    """
    from perflab.config import load_config
    from perflab.tools.isolation import resolve_policy

    return resolve_policy(task_file, load_config().isolation.level)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def start_agent(
    task_yaml: str,
    iters: int = 8,
    candidates: int = 4,
    suggest: str | None = None,
) -> dict:
    """Launch an agent optimization run in the background. Returns a job_id for tracking.

    Requires a configured LLM provider (run `perflab init` first).
    Use `get_agent_progress` to poll for status. Preferred for production use
    — non-blocking, full token tracking, progress polling.
    """
    job_id = uuid.uuid4().hex[:12]

    # Single concurrency model: take the agent lock up front so a rejected
    # start never consumes the executor slot or spawns a thread. The worker
    # releases the lock in its finally (threading.Lock allows cross-thread
    # release).
    if not _agent_lock.acquire(timeout=0):
        error = "Another agent run is already in progress."
        _register_job(job_id, {"status": "failed", "error": error, "progress": None})
        return {"job_id": job_id, "status": "failed", "error": error}

    def _run() -> None:
        from perflab.llm.config import LLMConfig
        from perflab.optimizers.agent import AgentConfig, run_agent
        from perflab.optimizers.progress import ListProgress
        from perflab.task_spec import TaskSpec

        progress = ListProgress()
        with _lock:
            _active_runs[job_id]["progress"] = progress
            _active_runs[job_id]["status"] = "running"

        try:
            task_file = Path(task_yaml)
            task = TaskSpec.load(task_file)
            llm_config = LLMConfig.load()
            isolation = _resolve_isolation(task_file)
            config = AgentConfig(
                n_candidates=candidates,
                max_iters=iters,
                isolation=isolation,
            )
            result = run_agent(
                task, task_file, config, llm_config,
                expert_suggestion=suggest,
                progress=progress,
            )
            with _lock:
                _active_runs[job_id]["status"] = "completed"
                _active_runs[job_id]["result"] = {
                    "best_value": result.best_value,
                    "best_iter": result.best_iter,
                    "baseline_value": result.baseline_value,
                    "run_dir": str(result.run_dir),
                    "isolation": isolation.level if isolation else "none",
                }
        except Exception as exc:  # noqa: BLE001 -- top-level safety net for a background job; any failure must be reported, not crash the thread
            with _lock:
                _active_runs[job_id]["status"] = "failed"
                _active_runs[job_id]["error"] = str(exc)
        finally:
            _agent_lock.release()

    _register_job(job_id, {"status": "starting", "progress": None})
    _executor.submit(_run)
    return {"job_id": job_id, "status": "starting"}


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def optimize_task(
    task_yaml: str,
    iters: int = 8,
    candidates: int = 4,
    suggest: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Run optimization using the client's own LLM via MCP sampling (no API key needed).

    This tool uses your MCP client's LLM to generate optimization candidates.
    No `perflab init` or API key is required.

    Caveats compared to start_agent:
    - Blocks the MCP connection for the duration of the run (may be several minutes)
    - Token usage statistics are unavailable
    - Requires your MCP client to support the sampling protocol
    """
    from perflab.llm.base import Message as PerfLabMessage
    from perflab.llm.config import LLMConfig
    from perflab.llm.mcp_sampling_provider import MCPSamplingProvider
    from perflab.optimizers.agent import AgentConfig, run_agent
    from perflab.optimizers.progress import ListProgress
    from perflab.task_spec import TaskSpec

    if ctx is None:
        return {"error": "No MCP context available. This tool must be invoked by an MCP client."}

    # Pre-flight: verify sampling works
    try:
        preflight = await ctx.sample("Respond with OK.", max_tokens=16)
        if not preflight.text:
            return {"error": "MCP sampling pre-flight returned empty response. Your client may not support sampling."}
    except Exception as exc:  # noqa: BLE001 -- feature-detection probe against an arbitrary MCP client, report any failure as unsupported
        return {"error": f"MCP sampling not supported by your client: {exc}"}

    # Concurrency guard
    if not _agent_lock.acquire(timeout=0):
        return {"error": "Another agent run is already in progress. Wait for it to finish."}

    try:
        loop = asyncio.get_running_loop()

        async def sample_fn(
            *,
            messages: list[PerfLabMessage],
            system_prompt: str | None = None,
            temperature: float = 0.7,
            max_tokens: int = 4096,
        ) -> str:
            # Convert PerfLab Messages to the format ctx.sample() expects
            from fastmcp.server.context import SamplingMessage
            from mcp.types import TextContent

            sampling_msgs = []
            for m in messages:
                # System content arrives via system_prompt, so anything else maps to user
                role: Literal["user", "assistant"] = (
                    "assistant" if m.role == "assistant" else "user"
                )
                sampling_msgs.append(
                    SamplingMessage(
                        role=role,
                        content=TextContent(type="text", text=m.content),
                    )
                )

            result = await ctx.sample(
                messages=sampling_msgs,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result.text or ""

        provider = MCPSamplingProvider(
            name="mcp-sampling",
            _sample_fn=sample_fn,
            _loop=loop,
        )

        llm_config = LLMConfig(
            provider="mcp-sampling",
            model="client-llm",
        )

        task_file = Path(task_yaml)
        task = TaskSpec.load(task_file)
        progress = ListProgress()
        isolation = _resolve_isolation(task_file)
        config = AgentConfig(
            n_candidates=candidates,
            max_iters=iters,
            isolation=isolation,
        )

        result = await asyncio.to_thread(
            run_agent,
            task, task_file, config, llm_config,
            expert_suggestion=suggest,
            progress=progress,
            provider=provider,
        )

        return {
            "status": "completed",
            "best_value": result.best_value,
            "best_iter": result.best_iter,
            "baseline_value": result.baseline_value,
            "run_dir": str(result.run_dir),
            "isolation": isolation.level if isolation else "none",
            "messages": progress.messages[-20:],
        }
    except Exception as exc:  # noqa: BLE001 -- top-level safety net for the MCP tool call; any failure must be reported, not crash the connection
        return {"status": "failed", "error": str(exc)}
    finally:
        _agent_lock.release()


@mcp.tool(annotations={"readOnlyHint": True})
def get_agent_progress(job_id: str) -> dict:
    """Check status and recent progress messages for a background agent run."""
    with _lock:
        run_info = _active_runs.get(job_id)
        if run_info is None:
            return {"error": f"Unknown job_id: {job_id}"}

        status = run_info["status"]
        messages: list[str] = []
        progress = run_info.get("progress")
        if progress is not None:
            messages = progress.messages[-20:]

        result: dict = {
            "job_id": job_id,
            "status": status,
            "recent_messages": messages,
        }
        if "result" in run_info:
            result["result"] = run_info["result"]
        if "error" in run_info:
            result["error"] = run_info["error"]
        return result
