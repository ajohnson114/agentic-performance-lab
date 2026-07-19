from __future__ import annotations

import contextvars
import logging
import shlex
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from perflab.runners.correctness import _passthrough_env
from perflab.tools.shell import DEFAULT_TIMEOUT_S, CmdResult, run_cmd

logger = logging.getLogger(__name__)

# Task-declared env_passthrough names (task.constraints.env_passthrough) that
# every profiler's benchmark run should forward from the parent process env.
# The profiler loop in the pipeline sets this via bench_env_passthrough() so
# that vars like DATA_ROOT/HF_HOME reach profiled runs exactly as they already
# reach the normal benchmark/correctness runs. run_bench_under is the single
# funnel every profiler executes the benchmark through, so resolving the
# passthrough here covers every current and future profiler.
_BENCH_ENV_PASSTHROUGH: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "perflab_bench_env_passthrough", default=(),
)


@contextmanager
def bench_env_passthrough(names: Sequence[str] | None) -> Iterator[None]:
    """Forward task.yaml env_passthrough vars to benchmark runs within the block.

    While active, run_bench_under overlays the named vars (those present in
    os.environ) onto every profiled benchmark run's environment, matching what
    run_benchmark/run_correctness already do for non-profiled runs. The
    previous value is restored on exit (nested blocks work); None resets to ().
    """
    token = _BENCH_ENV_PASSTHROUGH.set(tuple(names or ()))
    try:
        yield
    finally:
        _BENCH_ENV_PASSTHROUGH.reset(token)


@dataclass
class ProfileResult:
    name: str
    artifacts: dict[str, str]  # logical_name -> relative path
    summary: dict

class Profiler(Protocol):
    name: str
    def is_available(self) -> bool: ...
    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult: ...


def bench_argv(bench_cmd: str) -> list[str]:
    """Parse the task's benchmark command exactly once, the same way for every profiler."""
    return shlex.split(bench_cmd)


def run_bench_under(
    wrapper: Sequence[str],
    bench_cmd: str,
    cwd: Path,
    *,
    env: Mapping[str, str] | None = None,
    timeout_s: int | None = DEFAULT_TIMEOUT_S,
) -> CmdResult:
    """Run the benchmark under a profiler wrapper: [*wrapper, *bench_argv(bench_cmd)].

    An empty wrapper runs the benchmark bare. timeout_s is passed through to
    run_cmd unchanged and defaults to run_cmd's own default.

    The benchmark is candidate-patched (LLM-authored) code, so the subprocess
    environment is built via the allowlist (agent_subprocess_env), matching
    the benchmark/correctness runners -- the profiler wrapper itself only
    needs PATH/HOME/toolchain vars, all of which the allowlist forwards.
    LC_ALL=C is forced (unless the caller overrides it) because the
    perf/tma/power/lock parsers assume period decimal separators.

    Task.yaml env_passthrough vars (task.constraints.env_passthrough, e.g.
    DATA_ROOT/HF_HOME) active via bench_env_passthrough() are resolved from
    os.environ and overlaid next, so profiled runs see the same extra vars the
    non-profiled benchmark/correctness runners forward. Explicit `env` entries
    are overlaid last and win over both the locale and the passthrough vars.
    """
    passthrough_env = _passthrough_env(list(_BENCH_ENV_PASSTHROUGH.get()))
    return run_cmd(
        [*wrapper, *bench_argv(bench_cmd)], cwd=cwd,
        env={"LC_ALL": "C", **passthrough_env, **(env or {})}, timeout_s=timeout_s,
        env_mode="allowlist",
    )


def _sudo_allowed() -> bool:
    """Whether profilers may retry under ``sudo -n`` (opt-in, default off).

    The sudo retry re-runs the benchmark -- candidate-patched, untrusted
    code -- as root, outside every rlimit/env protection. On hosts with
    NOPASSWD sudo (common on cloud GPU boxes) a silent fallback would
    escalate routinely, so it requires explicit opt-in via
    profiler.allow_sudo in perflab.yaml or PERFLAB_PROFILER_ALLOW_SUDO=1.
    """
    try:
        from perflab.config import load_config
        return load_config().profiler.allow_sudo
    except Exception:  # noqa: BLE001 -- config load failure must fail closed, never escalate
        return False


def run_bench_with_sudo_fallback(
    wrapper: Sequence[str],
    bench_cmd: str,
    cwd: Path,
    *,
    expect_artifact: Path,
    env: Mapping[str, str] | None = None,
) -> tuple[CmdResult, bool]:
    """run_bench_under, retrying once under ``sudo -n`` when needed and allowed.

    The retry fires only when the first run failed (nonzero returncode), the
    expected artifact was not produced, sudo is on PATH, and sudo escalation
    has been explicitly enabled (see _sudo_allowed) — running the candidate
    benchmark as root is never a silent default. Returns (result, used_sudo).

    Environment handling (LC_ALL, task.yaml env_passthrough, and the explicit
    `env` overlay) is delegated entirely to run_bench_under; both the first
    attempt and the sudo retry inherit it identically.
    """
    res = run_bench_under(wrapper, bench_cmd, cwd, env=env)
    if res.returncode != 0 and not expect_artifact.exists() and shutil.which("sudo"):
        if not _sudo_allowed():
            logger.warning(
                "Profiler %s failed without elevated permissions; NOT retrying "
                "under sudo (runs candidate code as root). Opt in with "
                "profiler.allow_sudo: true in perflab.yaml or "
                "PERFLAB_PROFILER_ALLOW_SUDO=1.", wrapper[0] if wrapper else "?",
            )
            return res, False
        res = run_bench_under(["sudo", "-n", *wrapper], bench_cmd, cwd, env=env)
        return res, True
    return res, False
