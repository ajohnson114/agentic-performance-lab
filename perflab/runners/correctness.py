from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

from perflab.runners.benchmark import _resolve_rlimit
from perflab.tools.isolation import IsolationPolicy, wrap_command
from perflab.tools.shell import CmdResult, run_cmd

_logger = logging.getLogger(__name__)


def _passthrough_env(
    env_passthrough: list[str] | None,
    accuracy_tolerance: str | None = None,
) -> dict[str, str]:
    """Forward task.yaml-declared env_passthrough vars from the current process env.

    accuracy_tolerance (task.yaml constraints.accuracy_tolerance) is exported
    as PERFLAB_ACCURACY_TOLERANCE so tests.py harnesses can loosen their
    comparison thresholds to the task author's declared bound.
    """
    env = {name: os.environ[name] for name in env_passthrough or [] if name in os.environ}
    if accuracy_tolerance:
        env["PERFLAB_ACCURACY_TOLERANCE"] = accuracy_tolerance
    return env


def _maybe_wrap(
    cmd_args: list[str],
    cwd: Path,
    isolation: IsolationPolicy | None,
    extra_fds: list[int] | None = None,
) -> list[str]:
    """Apply OS-level sandboxing (Fix 2b) if an isolation policy was given.

    Binds this call's own ``cwd`` as the read-write workspace rather than
    trusting ``isolation.workspace`` -- see the equivalent note in
    perflab.runners.benchmark.run_benchmark. Default (isolation=None) is a
    no-op, so existing callers are unaffected.

    extra_fds collects fds the wrapped argv references (strict's seccomp
    filter memfd); the caller must forward them to run_cmd's pass_fds, and a
    wrapped argv is single-use -- wrap again for every spawn (run_cmd closes
    the fds afterwards).
    """
    if isolation is None:
        return cmd_args
    effective_policy = dataclasses.replace(isolation, workspace=cwd)
    return wrap_command(cmd_args, effective_policy, extra_fds=extra_fds)


def run_correctness(
    cmd: str,
    cwd: Path,
    program_type: str | None = None,
    rlimit_as_gb: float | None = None,
    skip_preexec: bool = False,
    env_passthrough: list[str] | None = None,
    isolation: IsolationPolicy | None = None,
    accuracy_tolerance: str | None = None,
) -> CmdResult:
    """Run correctness tests. Disables RLIMIT_AS for GPU program types.

    rlimit_as_gb overrides the default when set in task.yaml constraints.
    skip_preexec: If True, skip preexec_fn (use when called from threads).

    This runs candidate-patched (LLM-authored) tests.py, so the subprocess
    environment is built via the allowlist (agent_subprocess_env), not the
    blocklist used for trusted tool invocations. env_passthrough names extra
    task.yaml-declared vars (task.constraints.env_passthrough) to forward.
    accuracy_tolerance (task.yaml constraints.accuracy_tolerance) is exported
    as PERFLAB_ACCURACY_TOLERANCE — see _passthrough_env.

    isolation (Fix 2b): optional OS-level sandboxing (see perflab.tools.
    isolation), layered on top of the protections above. Defaults to None
    (no sandboxing beyond rlimits), matching pre-Fix-2b behavior exactly.
    """
    import shlex
    rlimit = _resolve_rlimit(program_type, rlimit_as_gb)
    extra = _passthrough_env(env_passthrough, accuracy_tolerance)
    spawn_fds: list[int] = []
    cmd_args = _maybe_wrap(shlex.split(cmd), cwd, isolation, extra_fds=spawn_fds)
    return run_cmd(
        cmd_args, cwd=cwd, env=extra if extra else None,
        timeout_s=60, rlimit_as_bytes=rlimit, skip_preexec=skip_preexec,
        env_mode="allowlist", pass_fds=spawn_fds,
    )


def run_correctness_twice(
    cmd: str,
    cwd: Path,
    program_type: str | None = None,
    rlimit_as_gb: float | None = None,
    skip_preexec: bool = False,
    expected_exit: int = 0,
    env_passthrough: list[str] | None = None,
    isolation: IsolationPolicy | None = None,
    accuracy_tolerance: str | None = None,
) -> tuple[CmdResult, list[str]]:
    """Run correctness tests twice with different random seeds.

    The second run uses PERFLAB_DETERMINISM_SEED=42 to give the test harness
    a signal to vary its random inputs if it supports it. This catches:
      - No-op kernels that rely on buffer reuse from a prior run
      - Kernels that only work for specific input patterns

    Returns (first_result, warnings). The first_result is the primary
    correctness check result. Warnings are non-empty if the second run
    behaves differently from the first.

    Both runs execute candidate-patched code, so the subprocess environment
    is built via the allowlist (agent_subprocess_env) -- see run_correctness.
    isolation (Fix 2b): see run_correctness; defaults to None (no-op).
    """
    import shlex
    rlimit = _resolve_rlimit(program_type, rlimit_as_gb)
    extra = _passthrough_env(env_passthrough, accuracy_tolerance)

    # Wrapped argvs are single-use under strict isolation (run_cmd closes the
    # seccomp fd the argv references), so each run wraps the command afresh.
    fds1: list[int] = []
    args1 = _maybe_wrap(shlex.split(cmd), cwd, isolation, extra_fds=fds1)

    # First run: normal
    res1 = run_cmd(
        args1, cwd=cwd, env=extra if extra else None,
        timeout_s=60, rlimit_as_bytes=rlimit, skip_preexec=skip_preexec,
        env_mode="allowlist", pass_fds=fds1,
    )

    if res1.returncode != expected_exit:
        return res1, []

    # Second run: with different seed env var to invalidate caches
    fds2: list[int] = []
    args2 = _maybe_wrap(shlex.split(cmd), cwd, isolation, extra_fds=fds2)
    res2_env = {**extra, "PERFLAB_DETERMINISM_SEED": "42"}
    res2 = run_cmd(
        args2, cwd=cwd, timeout_s=60, rlimit_as_bytes=rlimit,
        skip_preexec=skip_preexec,
        env=res2_env,
        env_mode="allowlist", pass_fds=fds2,
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
