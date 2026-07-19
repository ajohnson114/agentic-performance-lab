from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger(__name__)

# Marker preexec_fn writes to fd 2 (captured as stderr) when a resource limit
# fails to apply. Matched on a single line since exception messages (e.g.
# OSError's "[Errno 11] ...") may themselves contain "]"; greedy matching
# extends to the rightmost "]" before the newline so those aren't truncated.
_RLIMIT_MARKER_RE = re.compile(r"\[perflab-rlimit-failed (.*)\]\n?")


@dataclass
class CmdResult:
    cmd: Sequence[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    # None = no preexec_fn ran (non-Linux, or skip_preexec=True) -- not applicable.
    # True = preexec_fn ran and all rlimits applied cleanly.
    # False = preexec_fn ran but one or more rlimits failed to apply.
    rlimits_applied: bool | None = None


def _make_linux_preexec(
    rlimit_as_bytes: int | None, rlimit_nproc: int = 512,
) -> Callable[[], None] | None:
    """Return a preexec_fn that sets resource limits on Linux, or None."""
    if platform.system() != "Linux":
        return None

    def _preexec() -> None:
        import resource

        # preexec_fn runs post-fork/pre-exec: it cannot log through the
        # parent's logger, but fd 2 is already dup'd onto the child's
        # stderr pipe at this point, so writing there surfaces failures
        # to run_cmd via the captured CmdResult.stderr.
        failures = []
        for name, res, lim in (
            ("AS", resource.RLIMIT_AS, rlimit_as_bytes),
            ("NPROC", resource.RLIMIT_NPROC, rlimit_nproc),
            ("NOFILE", resource.RLIMIT_NOFILE, 1024),
        ):
            if lim is None:
                continue
            try:
                resource.setrlimit(res, (lim, lim))
            except (ValueError, OSError) as exc:
                failures.append(f"{name}:{exc}")
        if failures:
            os.write(2, f"[perflab-rlimit-failed {' '.join(failures)}]\n".encode())

    return _preexec


# Default wall-clock timeout for subprocesses launched via run_cmd. Benchmark
# and profiler invocations previously defaulted to no timeout, so a wedged
# candidate (deadlock, infinite loop) hung the whole stage forever. 600 s is
# generous headroom over the longest expected legitimate run (benchmark runs
# cap at 300 s, thread_sched records at 300 s) while still bounding a hang.
# Callers may pass an explicit timeout_s (or None to disable, e.g. for
# interactive/debug use).
DEFAULT_TIMEOUT_S = 600

# Returncode used for CmdResult when a command is killed on timeout.
# Mirrors the GNU coreutils `timeout` convention (124 = timed out).
TIMEOUT_RETURNCODE = 124

# Default address space limit for CPU-only tasks.
DEFAULT_RLIMIT_AS_BYTES = 4 * 1024**3  # 4 GB

# Default address space limit for GPU tasks. CUDA runtimes and JIT compilers
# map large virtual address regions, so the cap is much higher than CPU tasks.
# This prevents runaway allocation while still allowing normal GPU workloads.
DEFAULT_GPU_RLIMIT_AS_BYTES = 32 * 1024**3  # 32 GB


# Environment variable prefixes that should not be inherited by benchmark
# subprocesses to prevent accidental secret leakage.
_SECRET_ENV_PREFIXES = ("PERFLAB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def _sanitize_env(env: dict[str, str] | None) -> dict[str, str]:
    """Build subprocess environment, stripping secret keys.

    Blocklist model: kept for trusted tool invocations (profilers, compilers)
    where dropping unknown vars could break toolchains. Subprocesses that
    execute candidate-patched (agent/LLM-authored) code should use
    agent_subprocess_env() instead -- see env_mode on run_cmd.
    """
    base = dict(os.environ)
    for key in list(base):
        if any(key.startswith(prefix) for prefix in _SECRET_ENV_PREFIXES):
            del base[key]
    # Force the C locale: the perf/tma/power/lock parsers assume period
    # decimal separators and comma thousands separators. Under a comma-decimal
    # locale (e.g. de_DE), perf prints "4002,12 msec" and comma-stripping
    # would silently read it as 400212 — a 100x error. LC_ALL outranks every
    # LANG/LC_* var; an explicit caller env may still override it.
    base["LC_ALL"] = "C"
    if env:
        base.update(env)
    return base


# Environment variable prefixes forwarded to subprocesses that execute
# agent/LLM-patched candidate code (benchmark & correctness runners).
# Allowlist, not blocklist: candidate code is untrusted, so unknown vars
# (AWS/GITHUB/HF tokens, DATABASE_URL, SSH_AUTH_SOCK, etc.) are dropped by
# default rather than requiring them to be individually blocked.
_AGENT_ENV_ALLOWLIST_PREFIXES = (
    "PATH", "HOME", "LANG", "LC_", "TERM", "TMPDIR", "USER", "SHELL",
    "PYTHON", "VIRTUAL_ENV", "CONDA_",              # interpreter resolution
    "CUDA_", "NVIDIA_", "LD_LIBRARY_PATH",          # GPU runtimes
    "XLA_", "JAX_", "TPU_", "TF_",                  # JAX/TPU
    "TORCH_", "TRITON_", "OMP_", "MKL_", "OPENBLAS_",
    "PERFLAB_TASK_",                                 # task-declared vars
)


def agent_subprocess_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build subprocess environment for agent-run (candidate) code: allowlist only.

    extra (fast-screen bench flags, determinism seeds, task.yaml
    env_passthrough forwards) is applied last and wins over inherited values.
    """
    base = {
        k: v for k, v in os.environ.items()
        if any(k == p or k.startswith(p) for p in _AGENT_ENV_ALLOWLIST_PREFIXES)
    }
    if extra:
        base.update(extra)
    return base


def _coerce_output(raw: object) -> str:
    """Normalize TimeoutExpired.stdout/.stderr, which may be None, str, or
    bytes depending on platform and where the timeout fired."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode(errors="replace")
    return str(raw)


def _resolve_python(cmd: Sequence[str]) -> list[str]:
    """Replace bare 'python' with sys.executable so venv Python is used."""
    cmd = list(cmd)
    if cmd and cmd[0] == "python":
        cmd[0] = sys.executable
    return cmd


def run_cmd(
    cmd: Sequence[str],
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: int | None = DEFAULT_TIMEOUT_S,
    rlimit_as_bytes: int | None = DEFAULT_RLIMIT_AS_BYTES,
    rlimit_nproc: int = 512,
    skip_preexec: bool = False,
    env_mode: str = "blocklist",
    pass_fds: Sequence[int] = (),
) -> CmdResult:
    """Run a command and return the result.

    skip_preexec: If True, skip preexec_fn. Use this when calling from
    threads (e.g. ThreadPoolExecutor) because preexec_fn + fork() in a
    multithreaded process has undefined behavior (Python docs).

    pass_fds: file descriptors the spawned command must inherit (currently
    the seccomp filter memfd that isolation.wrap_command references by number
    in a ``bwrap --seccomp FD`` argument). run_cmd takes ownership: every fd
    here is closed in the parent once the subprocess has finished (or failed
    to spawn), so callers open fresh fds per call and never reuse them.

    env_mode: "blocklist" (default) inherits the full environment minus a
    handful of known secret prefixes -- use for trusted tool invocations
    (profilers, compilers). "allowlist" inherits only a known-safe prefix
    set (see agent_subprocess_env) -- use when running candidate-patched
    code (benchmark/correctness runners), since that code is untrusted and
    arbitrary secrets must not leak into it. In both modes, `env` is
    overlaid last and wins over inherited values.

    timeout_s: Wall-clock limit for the subprocess (default
    DEFAULT_TIMEOUT_S). On expiry the child is killed and a failure
    CmdResult is returned (returncode=TIMEOUT_RETURNCODE, timeout message
    appended to stderr) -- subprocess.TimeoutExpired never escapes. Pass
    None to disable the timeout entirely.
    """
    preexec = None if skip_preexec else _make_linux_preexec(rlimit_as_bytes, rlimit_nproc=rlimit_nproc)
    cmd = _resolve_python(cmd)

    if env_mode == "allowlist":
        run_env = agent_subprocess_env(env)
    else:
        run_env = _sanitize_env(dict(env) if env else None)

    t0 = time.time()
    try:
        try:
            p = subprocess.run(
                list(cmd),
                cwd=str(cwd) if cwd else None,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=preexec,
                pass_fds=tuple(pass_fds),
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run has already killed the child before re-raising.
            t1 = time.time()
            _logger.warning(
                "command timed out after %ss and was killed: %s", timeout_s, list(cmd),
            )
            stdout = _coerce_output(exc.stdout)
            stderr = _coerce_output(exc.stderr)
            msg = f"[perflab-timeout] command timed out after {timeout_s}s and was killed"
            stderr = f"{stderr}\n{msg}" if stderr else msg
            return CmdResult(
                cmd=cmd, returncode=TIMEOUT_RETURNCODE, stdout=stdout, stderr=stderr,
                duration_s=t1 - t0, rlimits_applied=None,
            )
    finally:
        # Ownership contract (see docstring): pass_fds are closed here on
        # every path -- normal exit, timeout return, and spawn failure alike.
        for fd in pass_fds:
            try:
                os.close(fd)
            except OSError:
                pass
    t1 = time.time()

    stdout, stderr = p.stdout, p.stderr
    rlimits_applied: bool | None = None
    if preexec is not None:
        match = _RLIMIT_MARKER_RE.search(stderr)
        if match:
            rlimits_applied = False
            _logger.warning("rlimit application failed for %s: %s", list(cmd), match.group(1))
            stderr = _RLIMIT_MARKER_RE.sub("", stderr)
            stdout = _RLIMIT_MARKER_RE.sub("", stdout)
        else:
            rlimits_applied = True

    return CmdResult(
        cmd=cmd, returncode=p.returncode, stdout=stdout, stderr=stderr,
        duration_s=t1 - t0, rlimits_applied=rlimits_applied,
    )
