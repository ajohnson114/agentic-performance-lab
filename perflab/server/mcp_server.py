"""PerfLab MCP server — exposes profiling and optimization tools via FastMCP.

Trust boundary: this server is designed for LOCAL, single-user use. Its tools
execute task-defined build/benchmark commands with the invoking user's
privileges, so any client that can call them can run code as that user —
do not expose the server to untrusted remote clients. Path-shaped inputs
that are joined onto known roots (run_id, create_task's category/name) are
validated as single path segments; task_yaml/tasks_root/out_dir select which
local files to operate on and are trusted as the user's own choice, exactly
like the equivalent CLI arguments.

This module is the stable entry point and import surface. The tool
implementations live in sibling modules — authoring, runs, analysis,
environment, agent_tools — which all register on the shared FastMCP
instance in perflab.server.core; importing this module registers every
tool. All names (including the ``_``-prefixed helpers used by tests) are
re-exported here.
"""
from __future__ import annotations

from perflab.server.agent_tools import (  # noqa: F401
    _MAX_TERMINAL_RUNS,
    _active_runs,
    _agent_lock,
    _executor,
    _lock,
    _register_job,
    _resolve_isolation,
    get_agent_progress,
    optimize_task,
    profile_task,
    start_agent,
)
from perflab.server.analysis import (  # noqa: F401
    get_bottlenecks,
    get_build_recommendations,
    get_gpu_attribution,
    get_hlo_attribution,
    get_profile_diff,
    get_roofline_analysis,
    get_thresholds,
)
from perflab.server.authoring import (  # noqa: F401
    create_task,
    lint_bench_script,
    list_tasks,
    show_task,
    show_task_authoring_guide,
    show_task_schema,
    show_tuning_schema,
    suggest_contract,
    suggest_profilers,
    suggest_thresholds,
    validate_task,
)
from perflab.server.core import (  # noqa: F401
    _MAX_OUTPUT_BYTES,
    _PROGRAM_TYPES,
    _guard_output_size,
    _to_dicts,
    mcp,
)
from perflab.server.environment import (  # noqa: F401
    ci_check,
    doctor_check,
    get_peaks,
    save_ci_baseline,
)
from perflab.server.runs import (  # noqa: F401
    compare_runs,
    get_run,
    get_run_section,
    list_runs,
    replay_run,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
