"""compute-sanitizer gate: catch CUDA memory/race bugs before a candidate is accepted.

Why this exists
================
The correctness test's numerical check tells you the answer was right on the
one input the test happened to run. It does not tell you the kernel that
produced it was free of undefined behavior. An out-of-bounds shared-memory
access, or a race between threads writing shared memory without (or with a
misplaced) ``__syncthreads()``, routinely produces the right answer on
today's driver, GPU and launch configuration while remaining UB that a
different driver, a different occupancy, or a different compiler flag can
flip into silent corruption. Exactly the optimizations PerfLab's CUDA tasks
invite -- shared-memory tiling, warp-level primitives, double buffering --
are where this bug class lives, and a numeric-output check structurally
cannot see it.

This module wraps a CUDA task's correctness command in NVIDIA's
``compute-sanitizer`` (memcheck: out-of-bounds/misaligned/illegal-address
accesses; racecheck: shared-memory hazards) and treats a detected error as a
rejected candidate. It runs once, against the candidate about to be
accepted, between "correctness test passed" and "candidate accepted" -- see
``perflab.optimizers.phases.evaluate.accept_best``.

Why the correctness command, not the benchmark command: sanitizer
instrumentation adds heavy overhead (racecheck especially), and the
correctness command already exercises the same kernel code paths at a much
smaller problem size purely to check numerics. Wrapping the benchmark's many
repeats at full problem size in a serializing memory checker would multiply
the accept-time cost for no additional coverage.

Scope: only tasks whose build step invokes ``nvcc`` -- see
:func:`uses_cuda_build`. This deliberately is not gated on ``program_type``:
``reduction/cpp_cuda`` is declared ``program_type: cpp`` (its host code is
C++) but its build step is ``nvcc ... reduce.cu``, so it is exactly the kind
of hand-written-CUDA task this gate exists for. PyTorch/JAX/Triton tasks
dispatch to vetted library- or compiler-generated kernels rather than
agent-hand-written CUDA -- a different risk profile, and a much heavier
sanitizer target (a full framework import) -- so they are out of scope here.

Availability: like every other profiling tool in PerfLab, a missing
compute-sanitizer binary degrades to "skipped, logged once" rather than
blocking every CUDA task on a host (including this project's own CI, which
has no GPU) that has no CUDA toolkit installed. See
``optimizers.phases.evaluate._cuda_sanitizer_gate`` for that half; this
module only answers "is compute-sanitizer available and what did it find."

Parsing: compute-sanitizer prints a stable summary line after the target
process exits, regardless of the target's own exit code -- so a
correctness assertion that fails for an unrelated reason does not
masquerade as a sanitizer finding, and a sanitizer-flagged error is not
masked by the target happening to exit 0. The summary line's exact text is
NOT tool-agnostic, though: memcheck prints ``ERROR SUMMARY: N error(s)``,
but racecheck prints its own ``RACECHECK SUMMARY: N hazards displayed (X
errors, Y warnings)`` instead (confirmed on real hardware) -- see
``_SUMMARY_PATTERNS``, tried in order. When no pattern matches (a crash
before either tool could print, a timeout, an unexpected output format),
the result is treated as unverified and NOT clean -- fail closed on
ambiguity, the same stance every statistical gate in this codebase takes
(see ``analyzers.decision``): a missing signal is never silently treated as
a passing one.
"""
from __future__ import annotations

import dataclasses
import os
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from perflab.runners.benchmark import _resolve_rlimit
from perflab.tools.isolation import IsolationPolicy, wrap_command
from perflab.tools.shell import run_cmd

#: A task "uses CUDA" for this gate's purposes when its build step invokes
#: nvcc -- independent of program_type (see module docstring).
_NVCC_RE = re.compile(r"(?:^|[\s/])nvcc(?:\.exe)?\b")

#: compute-sanitizer's summary line format differs by tool. memcheck (and,
#: per NVIDIA's docs, initcheck/synccheck) print "ERROR SUMMARY: N error(s)".
#: racecheck does NOT -- confirmed on real hardware (Nsight Compute / CUDA
#: 12.8, H100): it prints its own "RACECHECK SUMMARY: N hazards displayed
#: (X errors, Y warnings)" instead. A single ERROR-SUMMARY-only regex made
#: every racecheck run "inconclusive" regardless of whether it was actually
#: clean -- rejecting every real candidate on real hardware, silently,
#: because every unit test fixture for racecheck used the wrong (copied
#: from memcheck) summary format and never caught it. Patterns are tried in
#: order; the first one that matches the combined output wins.
_SUMMARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ERROR SUMMARY:\s*(\d+)\s*error", re.IGNORECASE),
    re.compile(r"RACECHECK SUMMARY:.*?\(\s*(\d+)\s*errors?", re.IGNORECASE),
)

#: memcheck catches out-of-bounds/misaligned/illegal-address accesses;
#: racecheck catches shared-memory hazards (the bug class a missing or
#: misplaced __syncthreads() produces). Both, because they catch disjoint
#: bug classes and neither subsumes the other.
DEFAULT_TOOLS: tuple[str, ...] = ("memcheck", "racecheck")

#: Generous relative to these tasks' correctness commands (small selftest
#: sizes), because sanitizer instrumentation -- racecheck especially -- adds
#: heavy per-access overhead. Overridable via
#: constraints.compute_sanitizer_timeout_s.
DEFAULT_TIMEOUT_S = 180


def uses_cuda_build(build_cmd: str | None) -> bool:
    """True when a task's build step compiles with nvcc (a real CUDA binary).

    Not gated on program_type -- see the module docstring's "Scope" section.
    """
    return build_cmd is not None and bool(_NVCC_RE.search(build_cmd))


def compute_sanitizer_available() -> bool:
    """True when the ``compute-sanitizer`` binary is on PATH."""
    return shutil.which("compute-sanitizer") is not None


@dataclass(frozen=True)
class SanitizerToolResult:
    """The outcome of one ``compute-sanitizer --tool <tool>`` invocation."""

    tool: str
    #: False only when the summary line could not be found at all -- a
    #: crash, a timeout, or an output format this parser doesn't recognize.
    #: Distinct from `errors == 0`: `ran=False` means "inconclusive", not
    #: "clean".
    ran: bool
    #: Parsed error count, or -1 when `ran` is True but... this field is
    #: always consistent with `ran` (see `ran`'s docstring) -- -1 never
    #: appears when ran=True. Kept as a plain int (not Optional) so callers
    #: comparing `errors == 0` don't need a None-check on the common path.
    errors: int
    returncode: int
    #: Tail of the tool's combined stdout+stderr, for the rejection message
    #: and event log -- not the full output, which can be large under
    #: racecheck.
    output: str
    duration_s: float

    @property
    def clean(self) -> bool:
        return self.ran and self.errors == 0


@dataclass(frozen=True)
class SanitizerReport:
    """The outcome of running every configured tool against one candidate."""

    #: False: the task has no CUDA build, or the check is disabled --
    #: nothing to run, and `clean` below is vacuously True.
    applicable: bool
    #: False: compute-sanitizer isn't installed. Distinct from `applicable`
    #: so callers can tell "nothing to check" from "wanted to check but couldn't".
    available: bool
    results: list[SanitizerToolResult] = field(default_factory=list)

    @property
    def ran(self) -> bool:
        return self.applicable and self.available and bool(self.results)

    @property
    def clean(self) -> bool:
        """True unless a tool that actually ran found (or failed to rule
        out) an error. Vacuously True when nothing ran -- callers that need
        to distinguish "clean" from "unchecked" should consult `ran` too."""
        return all(r.clean for r in self.results)

    @property
    def reason(self) -> str:
        """Human-readable explanation of what failed, or "" when `clean`."""
        bad = [r for r in self.results if not r.clean]
        if not bad:
            return ""
        parts = []
        for r in bad:
            if not r.ran:
                parts.append(f"{r.tool}: inconclusive (no ERROR SUMMARY in output -- crash, timeout, or unrecognized format)")
            else:
                parts.append(f"{r.tool}: {r.errors} error(s)")
        return "compute-sanitizer found problems -- " + "; ".join(parts)


_OUTPUT_TAIL_CHARS = 3000


def _wrap_isolation(
    cmd_args: list[str], cwd: Path, isolation: IsolationPolicy | None,
    extra_fds: list[int],
) -> list[str]:
    """Apply OS-level sandboxing if given. Mirrors runners.correctness._maybe_wrap.

    extra_fds collects the strict-mode seccomp filter fd (if any); the
    caller must forward it to run_cmd's pass_fds, same contract as every
    other isolation-wrapped spawn in this codebase.
    """
    if isolation is None:
        return cmd_args
    return wrap_command(
        cmd_args, dataclasses.replace(isolation, workspace=cwd), extra_fds=extra_fds,
    )


def _run_one_tool(
    tool: str,
    cmd: str,
    cwd: Path,
    *,
    timeout_s: int,
    program_type: str | None,
    rlimit_as_gb: float | None,
    env_passthrough: list[str] | None,
    isolation: IsolationPolicy | None,
) -> SanitizerToolResult:
    argv = [
        "compute-sanitizer", "--tool", tool, "--target-processes", "all",
        *shlex.split(cmd),
    ]
    spawn_fds: list[int] = []
    argv = _wrap_isolation(argv, cwd, isolation, spawn_fds)
    extra = {name: os.environ[name] for name in env_passthrough or [] if name in os.environ}
    rlimit = _resolve_rlimit(program_type, rlimit_as_gb)

    res = run_cmd(
        argv, cwd=cwd, env=extra if extra else None,
        timeout_s=timeout_s, rlimit_as_bytes=rlimit,
        env_mode="allowlist", pass_fds=spawn_fds,
    )
    combined = f"{res.stdout}\n{res.stderr}"
    match = None
    for pattern in _SUMMARY_PATTERNS:
        match = pattern.search(combined)
        if match:
            break
    ran = match is not None
    errors = int(match.group(1)) if match else -1
    return SanitizerToolResult(
        tool=tool, ran=ran, errors=errors, returncode=res.returncode,
        output=combined[-_OUTPUT_TAIL_CHARS:], duration_s=res.duration_s,
    )


def run_compute_sanitizer(
    cmd: str,
    cwd: Path,
    *,
    tools: list[str] | tuple[str, ...] = DEFAULT_TOOLS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    program_type: str | None = None,
    rlimit_as_gb: float | None = None,
    env_passthrough: list[str] | None = None,
    isolation: IsolationPolicy | None = None,
) -> SanitizerReport:
    """Run every tool in ``tools`` against ``cmd``, sequentially.

    ``applicable``/``available`` are always True in the returned report --
    the caller (``evaluate._cuda_sanitizer_gate``) is responsible for those
    two checks *before* calling this, since they gate whether it's worth
    building the candidate in the first place. This function's job is only
    "run the tools and report what they found."

    ``cmd`` runs candidate-patched (LLM-authored) code, so the subprocess
    environment is allowlist-only (``agent_subprocess_env``), matching
    ``runners.correctness.run_correctness``.
    """
    results = [
        _run_one_tool(
            tool, cmd, cwd, timeout_s=timeout_s, program_type=program_type,
            rlimit_as_gb=rlimit_as_gb, env_passthrough=env_passthrough,
            isolation=isolation,
        )
        for tool in tools
    ]
    return SanitizerReport(applicable=True, available=True, results=results)


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_TOOLS",
    "SanitizerReport",
    "SanitizerToolResult",
    "compute_sanitizer_available",
    "run_compute_sanitizer",
    "uses_cuda_build",
]
