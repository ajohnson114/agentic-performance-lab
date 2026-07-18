"""OS-level sandboxing for candidate (LLM-authored) subprocess execution.

Fix 2b: tiered, opt-in isolation layered on top of the rlimit (perflab.tools.
shell._make_linux_preexec) and env-allowlist (perflab.tools.shell.
agent_subprocess_env) protections that already exist. rlimits cap CPU/memory/
fd usage but do nothing about filesystem writes outside the workspace or
network egress -- that's the gap this module closes.

Full container orchestration (Docker) was explicitly ruled out: image
management contradicts the "local-first CLI" premise, and container
cold-start would pollute the fast-screen timing tier (see DESIGN.md,
"Two-tier benchmarking"). Instead, isolation here is a launch-time wrapper
(Bubblewrap on Linux) around the same subprocess perflab would have run
anyway -- cheap, optional, and (per the benchmark-noise A/B tracked in
DESIGN.md) intended to add negligible steady-state overhead.

Levels:
  none       -- current behavior (rlimits only). The compiled-in default.
  restricted -- bwrap: read-only bind of /usr, /lib, the venv, and any
                detected CUDA/driver paths; read-write bind of the calling
                workspace only; --unshare-net unless network=True;
                --die-with-parent. Falls back to 'none' (with a logged
                warning) if bwrap is missing, not on Linux, or user
                namespaces are unusable.
  strict     -- restricted + a seccomp filter denying ptrace/mount/bpf/keyctl,
                if bwrap on this host supports --seccomp. perflab does not
                ship a compiled BPF filter yet (see wrap_command docstring),
                so strict currently behaves like restricted plus a warning.

macOS has no bwrap. sandbox-exec exists but is deprecated, and a correct
profile is a substantial undertaking on its own -- rather than ship a profile
that only partially confines candidate code (and so *looks* sandboxed
without being sandboxed), restricted/strict on macOS fall back to 'none'
with an unambiguous warning. A real sandbox-exec implementation is a
documented follow-up, not attempted here.
"""
from __future__ import annotations

import functools
import glob
import logging
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

logger = logging.getLogger(__name__)

IsolationLevel = Literal["none", "restricted", "strict"]

_VALID_LEVELS: tuple[IsolationLevel, ...] = ("none", "restricted", "strict")

# Read-only paths bound into the sandbox so the interpreter, its native
# extensions, and the shared library loader are visible without giving
# candidate code write access to anything outside its workspace. /bin and
# /sbin are included alongside /usr because some distros still keep them as
# real directories rather than symlinks into /usr.
_RO_BIND_CANDIDATES: tuple[str, ...] = ("/usr", "/lib", "/lib64", "/bin", "/sbin")

# CUDA/driver install locations vary by distro (Debian multiarch libdir vs.
# RHEL/Arch single libdir), so several glob patterns are tried and only ones
# that actually exist on this host are bound.
_CUDA_PATH_GLOBS: tuple[str, ...] = (
    "/usr/local/cuda*",
    "/usr/lib/x86_64-linux-gnu/libcuda*",
    "/usr/lib64/libcuda*",
    "/usr/lib/nvidia*",
)


def normalize_level(level: str | None) -> IsolationLevel:
    """Validate and normalize a level string. None/'' -> 'none'."""
    if not level:
        return "none"
    lvl = level.strip().lower()
    if lvl not in _VALID_LEVELS:
        raise ValueError(f"Invalid isolation level {level!r}; must be one of {_VALID_LEVELS}")
    return cast(IsolationLevel, lvl)


def resolve_level(
    cli_level: str | None,
    task_level: str | None,
    config_level: str | None,
) -> IsolationLevel:
    """Resolve the effective isolation level: CLI flag > task.yaml > config.

    Mirrors the resolution order already used for iters/candidates/etc. in
    ``perflab agent`` (perflab/cli.py). A pure function so it's testable
    without invoking Typer.
    """
    return normalize_level(cli_level or task_level or config_level)


@dataclass
class IsolationPolicy:
    """Describes how a candidate subprocess should be sandboxed.

    ``level`` is the *requested* level -- wrap_command() may silently
    downgrade to a weaker level (logging why) if this host can't support
    what was asked; it never raises, since an isolation failure should
    degrade to unsandboxed execution rather than block the benchmark/
    correctness run (perflab is a local dev tool, not a hosted multi-tenant
    sandbox with a security SLA).

    ``workspace`` is the read-write directory candidate code is allowed to
    touch. Callers running a specific subprocess (run_benchmark,
    run_correctness) should treat their own ``cwd`` as authoritative for
    this -- see the note in those modules.
    """
    level: IsolationLevel = "none"
    workspace: Path | None = None
    run_output_dir: Path | None = None
    network: bool = False  # True if task.yaml set constraints.network: true


def default_level_for_host() -> IsolationLevel:
    """The level the spec designates as default (Linux + usable bwrap).

    Not currently wired in as the *actual* config default -- see DESIGN.md:
    the spec requires an A/B benchmark-noise check (restricted vs. none on
    tasks/matmul/cpp, confirming <1% median runtime delta) before flipping
    the shipped default from 'none' to this. This function exists so that
    check (and any future `perflab doctor`-style readiness report) has
    something to call once that validation has actually been run.
    """
    return "restricted" if _bwrap_usable() else "none"


@functools.lru_cache(maxsize=1)
def _bwrap_usable() -> bool:
    """Whether bwrap is installed AND can actually create a sandbox.

    A plain ``shutil.which`` check isn't enough: bwrap can be installed but
    fail at runtime if unprivileged user namespaces are disabled (common on
    hardened kernels), so this runs a cheap, no-op sandbox as a real
    capability probe. Cached -- this would otherwise run once per candidate
    in a beam-search iteration.
    """
    if platform.system() != "Linux":
        return False
    bwrap_bin = shutil.which("bwrap")
    if not bwrap_bin:
        return False
    try:
        probe = subprocess.run(
            [bwrap_bin, "--unshare-user", "--die-with-parent",
             "--ro-bind", "/", "/", "true"],
            capture_output=True, timeout=5,
        )
        return probe.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@functools.lru_cache(maxsize=1)
def _bwrap_supports_seccomp() -> bool:
    bwrap_bin = shutil.which("bwrap")
    if not bwrap_bin:
        return False
    try:
        help_result = subprocess.run(
            [bwrap_bin, "--help"], capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "--seccomp" in help_result.stdout


def _readonly_bind_paths() -> list[Path]:
    paths = [Path(p) for p in _RO_BIND_CANDIDATES if Path(p).exists()]
    # sys.prefix covers the active venv root (bin/, lib/, site-packages/);
    # sys.base_prefix additionally covers the system Python install when
    # not running inside a venv (base_prefix == prefix in that case, so the
    # set dedupes it for us).
    for p in {Path(sys.prefix), Path(sys.base_prefix)}:
        if p.exists() and p not in paths:
            paths.append(p)
    for pattern in _CUDA_PATH_GLOBS:
        for match in glob.glob(pattern):
            mp = Path(match)
            if mp.exists() and mp not in paths:
                paths.append(mp)
    return paths


def _nvidia_device_paths() -> list[str]:
    return sorted(glob.glob("/dev/nvidia*"))


def _resolve_python(cmd: list[str]) -> list[str]:
    """Replace a bare leading 'python' with sys.executable.

    Mirrors perflab.tools.shell._resolve_python. That resolution happens in
    run_cmd based on cmd[0], but once wrap_command prefixes cmd with a bwrap
    invocation, cmd[0] is "bwrap" rather than "python" and run_cmd's own
    check can't see through the wrapper -- so it has to happen here instead,
    before the bwrap prefix is added.
    """
    cmd = list(cmd)
    if cmd and cmd[0] == "python":
        cmd[0] = sys.executable
    return cmd


def wrap_command(cmd: list[str], policy: IsolationPolicy) -> list[str]:
    """Return ``cmd``, optionally prefixed with a bwrap invocation.

    Falls back to returning ``cmd`` unchanged whenever the requested level
    can't be honored on this host (non-Linux, bwrap missing/unusable), always
    logging why so the fallback isn't silent.

    Strict's seccomp layer (deny ptrace/mount/bpf/keyctl) is intentionally
    not implemented even when bwrap supports --seccomp: a correct filter
    needs per-architecture syscall numbers and BPF program validation, which
    needs either a libseccomp binding or hand-rolled bytecode -- neither is a
    dependency of this project. Shipping a guessed filter risks silently
    failing open on exactly the syscalls it's meant to deny, which is worse
    than not having it (same reasoning as the macOS sandbox-exec omission
    above). strict therefore currently runs with restricted's protections
    plus a warning; adding a real filter is a follow-up once there's a
    Linux+bwrap target to validate it against.
    """
    if policy.level == "none":
        return cmd

    system = platform.system()
    if system == "Darwin":
        logger.warning(
            "Isolation level %r requested but restricted/strict isolation is "
            "not implemented on macOS yet (sandbox-exec is deprecated and a "
            "correct profile is a substantial separate undertaking) -- "
            "falling back to 'none'. Candidate code runs unsandboxed on this host.",
            policy.level,
        )
        return cmd
    if system != "Linux":
        logger.warning(
            "OS-level isolation is only implemented for Linux (bwrap); this "
            "host reports %r -- falling back to 'none'. Candidate code runs "
            "unsandboxed on this host.", system,
        )
        return cmd

    bwrap_bin = shutil.which("bwrap")
    if not bwrap_bin or not _bwrap_usable():
        reason = "not found on PATH" if not bwrap_bin else (
            "installed but unusable (unprivileged user namespaces are likely disabled)"
        )
        logger.warning(
            "Isolation level %r requested but bwrap is %s -- falling back to "
            "'none'. Candidate code runs unsandboxed on this host.",
            policy.level, reason,
        )
        return cmd

    cmd = _resolve_python(cmd)
    args = [bwrap_bin]

    for p in _readonly_bind_paths():
        args += ["--ro-bind", str(p), str(p)]

    args += ["--proc", "/proc", "--dev", "/dev"]
    for dev in _nvidia_device_paths():
        args += ["--dev-bind", dev, dev]
    # A tmpfs, not a host bind: gives candidate code the writable /tmp many
    # programs assume exists, without exposing (or persisting anything on)
    # the real filesystem -- doesn't weaken the "workspace + run output dir
    # only" read-write contract below.
    args += ["--tmpfs", "/tmp"]

    if not policy.network:
        args.append("--unshare-net")

    args.append("--die-with-parent")

    if policy.workspace is not None:
        args += ["--bind", str(policy.workspace), str(policy.workspace)]
    if policy.run_output_dir is not None and policy.run_output_dir != policy.workspace:
        args += ["--bind", str(policy.run_output_dir), str(policy.run_output_dir)]

    if policy.level == "strict":
        if _bwrap_supports_seccomp():
            logger.warning(
                "Isolation level 'strict' requested: this bwrap build "
                "supports --seccomp but perflab does not yet ship a compiled "
                "BPF filter -- running with 'restricted'-equivalent "
                "protections only (no ptrace/mount/bpf/keyctl denial layer)."
            )
        else:
            logger.warning(
                "Isolation level 'strict' requested but this bwrap build has "
                "no --seccomp support -- running with 'restricted'-equivalent "
                "protections only."
            )

    return [*args, "--", *cmd]
