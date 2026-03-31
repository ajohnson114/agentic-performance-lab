from __future__ import annotations

import logging
from pathlib import Path
from perflab.tools.shell import run_cmd, CmdResult
from perflab.runners.benchmark import _resolve_rlimit

_logger = logging.getLogger(__name__)


def run_correctness(
    cmd: str,
    cwd: Path,
    program_type: str | None = None,
    rlimit_as_gb: float | None = None,
    skip_preexec: bool = False,
) -> CmdResult:
    """Run correctness tests. Disables RLIMIT_AS for GPU program types.

    rlimit_as_gb overrides the default when set in task.yaml constraints.
    skip_preexec: If True, skip preexec_fn (use when called from threads).
    """
    import shlex
    rlimit = _resolve_rlimit(program_type, rlimit_as_gb)
    return run_cmd(shlex.split(cmd), cwd=cwd, timeout_s=60, rlimit_as_bytes=rlimit, skip_preexec=skip_preexec)


def run_correctness_twice(
    cmd: str,
    cwd: Path,
    program_type: str | None = None,
    rlimit_as_gb: float | None = None,
    skip_preexec: bool = False,
    expected_exit: int = 0,
) -> tuple[CmdResult, list[str]]:
    """Run correctness tests twice with different random seeds.

    The second run uses PERFLAB_DETERMINISM_SEED=42 to give the test harness
    a signal to vary its random inputs if it supports it. This catches:
      - No-op kernels that rely on buffer reuse from a prior run
      - Kernels that only work for specific input patterns

    Returns (first_result, warnings). The first_result is the primary
    correctness check result. Warnings are non-empty if the second run
    behaves differently from the first.
    """
    import shlex
    rlimit = _resolve_rlimit(program_type, rlimit_as_gb)
    args = shlex.split(cmd)

    # First run: normal
    res1 = run_cmd(args, cwd=cwd, timeout_s=60, rlimit_as_bytes=rlimit, skip_preexec=skip_preexec)

    if res1.returncode != expected_exit:
        return res1, []

    # Second run: with different seed env var to invalidate caches
    res2 = run_cmd(
        args, cwd=cwd, timeout_s=60, rlimit_as_bytes=rlimit,
        skip_preexec=skip_preexec,
        env={"PERFLAB_DETERMINISM_SEED": "42"},
    )

    warnings: list[str] = []
    if res2.returncode != expected_exit:
        warnings.append(
            f"Determinism check failed: correctness passed on first run "
            f"(rc={res1.returncode}) but failed on second run with different "
            f"seed (rc={res2.returncode}). The kernel may rely on buffer "
            f"reuse or specific input patterns. stderr: {res2.stderr[:500]}"
        )
        _logger.warning("Determinism re-run failed: %s", warnings[0])

    return res1, warnings
