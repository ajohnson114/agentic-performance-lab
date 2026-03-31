from __future__ import annotations
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

@dataclass
class CmdResult:
    cmd: Sequence[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


def _make_linux_preexec(rlimit_as_bytes: int | None, rlimit_nproc: int = 512) -> callable | None:
    """Return a preexec_fn that sets resource limits on Linux, or None."""
    if platform.system() != "Linux":
        return None

    def _preexec() -> None:
        try:
            import resource
            if rlimit_as_bytes is not None:
                resource.setrlimit(resource.RLIMIT_AS, (rlimit_as_bytes, rlimit_as_bytes))
            resource.setrlimit(resource.RLIMIT_NPROC, (rlimit_nproc, rlimit_nproc))
            resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
        except Exception:
            pass  # Best-effort; silently skip on unsupported platforms

    return _preexec


# Default address space limit for CPU-only tasks.
DEFAULT_RLIMIT_AS_BYTES = 4 * 1024**3  # 4 GB

# Default address space limit for GPU tasks. CUDA runtimes and JIT compilers
# map large virtual address regions, so the cap is much higher than CPU tasks.
# This prevents runaway allocation while still allowing normal GPU workloads.
DEFAULT_GPU_RLIMIT_AS_BYTES = 32 * 1024**3  # 32 GB


# Environment variable prefixes that should not be inherited by benchmark
# subprocesses to prevent accidental secret leakage.
_SECRET_ENV_PREFIXES = ("PERFLAB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def _sanitize_env(env: dict[str, str] | None) -> dict[str, str] | None:
    """Build subprocess environment, stripping secret keys."""
    base = dict(os.environ)
    for key in list(base):
        if any(key.startswith(prefix) for prefix in _SECRET_ENV_PREFIXES):
            del base[key]
    if env:
        base.update(env)
    return base


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
    timeout_s: int | None = None,
    rlimit_as_bytes: int | None = DEFAULT_RLIMIT_AS_BYTES,
    rlimit_nproc: int = 512,
    skip_preexec: bool = False,
) -> CmdResult:
    """Run a command and return the result.

    skip_preexec: If True, skip preexec_fn. Use this when calling from
    threads (e.g. ThreadPoolExecutor) because preexec_fn + fork() in a
    multithreaded process has undefined behavior (Python docs).
    """
    preexec = None if skip_preexec else _make_linux_preexec(rlimit_as_bytes, rlimit_nproc=rlimit_nproc)
    cmd = _resolve_python(cmd)

    t0 = time.time()
    p = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=_sanitize_env(dict(env) if env else None),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        preexec_fn=preexec,
    )
    t1 = time.time()
    return CmdResult(cmd=cmd, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr, duration_s=t1 - t0)
