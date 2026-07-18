from __future__ import annotations

import shlex
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from perflab.tools.shell import DEFAULT_TIMEOUT_S, CmdResult, run_cmd


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

    An empty wrapper runs the benchmark bare. env and timeout_s are passed
    through to run_cmd unchanged; timeout_s defaults to run_cmd's own default.
    """
    return run_cmd(
        [*wrapper, *bench_argv(bench_cmd)], cwd=cwd, env=env, timeout_s=timeout_s,
    )


def run_bench_with_sudo_fallback(
    wrapper: Sequence[str],
    bench_cmd: str,
    cwd: Path,
    *,
    expect_artifact: Path,
    env: Mapping[str, str] | None = None,
) -> tuple[CmdResult, bool]:
    """run_bench_under, retrying once under ``sudo -n`` when needed.

    The retry fires only when the first run failed (nonzero returncode), the
    expected artifact was not produced, and sudo is on PATH — the escalation
    ladder profilers previously hand-rolled. Returns (result, used_sudo).
    """
    res = run_bench_under(wrapper, bench_cmd, cwd, env=env)
    if res.returncode != 0 and not expect_artifact.exists() and shutil.which("sudo"):
        res = run_bench_under(["sudo", "-n", *wrapper], bench_cmd, cwd, env=env)
        return res, True
    return res, False
