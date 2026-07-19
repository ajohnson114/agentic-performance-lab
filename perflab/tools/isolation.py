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
anyway -- cheap, optional, and intended to add negligible steady-state
overhead (validated in CI: see DESIGN.md for how the shipped "auto" default
is exercised on every push).

Levels:
  auto       -- resolve to whatever default_level_for_host() decides:
                'restricted' if this host has usable bwrap (Linux + working
                user namespaces), else 'none'. The compiled-in default as of
                2026-07-19 (see DESIGN.md) -- accepted from any source (CLI
                flag, task.yaml, config) and resolved by resolve_policy()
                before an IsolationPolicy is ever constructed, so nothing
                downstream (wrap_command, IsolationPolicy.level) ever sees
                the literal string 'auto'.
  none       -- unsandboxed (rlimits only).
  restricted -- bwrap: read-only bind of /usr, /lib, the venv, and any
                detected CUDA/driver paths; read-write bind of the calling
                workspace only; --unshare-net unless network=True (in which
                case /etc/resolv.conf, /etc/ssl/certs, and /etc/hosts are
                also read-only bound, so DNS/TLS work inside the sandbox);
                --die-with-parent. Falls back to 'none' (with a logged
                warning) if bwrap is missing, not on Linux, or user
                namespaces are unusable.
  strict     -- restricted + a seccomp syscall-denial filter (ptrace, the
                mount family, bpf, keyctl, namespace manipulation, module
                loading -- see perflab.tools.seccomp.DENIED_SYSCALL_NAMES),
                compiled in-process and handed to ``bwrap --seccomp`` as an
                inherited memfd. Falls back to restricted-equivalent (with a
                warning) if this bwrap build lacks --seccomp or the filter
                can't be built for this architecture.

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

import yaml

from perflab.tools import seccomp

logger = logging.getLogger(__name__)

IsolationLevel = Literal["none", "restricted", "strict"]

# What a level string may resolve to *before* "auto" is expanded via
# default_level_for_host(). normalize_level/resolve_level return this; by the
# time an IsolationPolicy is constructed (resolve_policy), "auto" has always
# been replaced with a concrete IsolationLevel.
RequestedIsolationLevel = Literal["none", "restricted", "strict", "auto"]

_VALID_LEVELS: tuple[RequestedIsolationLevel, ...] = ("none", "restricted", "strict", "auto")

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

# DNS/TLS config, read-only bound only when a task's constraints.network is
# true (i.e. --unshare-net is skipped -- see wrap_command). None of
# _RO_BIND_CANDIDATES cover /etc, so without these a network-allowed task
# could still reach the network namespace-wise but couldn't resolve hostnames
# or verify TLS certs from inside the sandbox. Bound as-is (no symlink
# resolution): on distros where /etc/ssl/certs is itself a symlink to another
# directory, --ro-bind on the symlink path still exposes whatever it points
# at, the same way it would outside the sandbox.
_NETWORK_RO_BIND_CANDIDATES: tuple[str, ...] = (
    "/etc/resolv.conf",
    "/etc/ssl/certs",
    "/etc/hosts",
)


def normalize_level(level: str | None) -> RequestedIsolationLevel:
    """Validate and normalize a level string. None/'' -> 'none'.

    Returns the *requested* level, which may be the literal string "auto" --
    callers that need a concrete IsolationLevel (never "auto") should go
    through resolve_effective_level() or resolve_policy().
    """
    if not level:
        return "none"
    lvl = level.strip().lower()
    if lvl not in _VALID_LEVELS:
        raise ValueError(f"Invalid isolation level {level!r}; must be one of {_VALID_LEVELS}")
    return cast(RequestedIsolationLevel, lvl)


def resolve_level(
    cli_level: str | None,
    task_level: str | None,
    config_level: str | None,
) -> RequestedIsolationLevel:
    """Resolve the requested isolation level: CLI flag > task.yaml > config.

    Mirrors the resolution order already used for iters/candidates/etc. in
    ``perflab agent`` (perflab/cli.py). A pure function so it's testable
    without invoking Typer. May return "auto" -- see resolve_effective_level().
    """
    return normalize_level(cli_level or task_level or config_level)


def resolve_effective_level(requested: RequestedIsolationLevel) -> IsolationLevel:
    """Expand "auto" to a concrete level via default_level_for_host().

    Every other requested level passes through unchanged. This is the single
    place "auto" gets resolved -- resolve_policy() calls it so an
    IsolationPolicy never carries the literal string "auto".
    """
    if requested == "auto":
        return default_level_for_host()
    return requested


def read_task_isolation(task_file: Path) -> tuple[str | None, bool]:
    """Read ``isolation.level`` and ``constraints.network`` from a task.yaml.

    Read via raw yaml.safe_load rather than through TaskSpec/Constraints;
    promoting these to proper TaskSpec fields (with schema validation and
    ``perflab show-task`` visibility) is a documented follow-up.
    """
    try:
        data = yaml.safe_load(task_file.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return None, False
    if not isinstance(data, dict):
        return None, False
    isolation_data = data.get("isolation")
    level = isolation_data.get("level") if isinstance(isolation_data, dict) else None
    constraints_data = data.get("constraints")
    network = bool(constraints_data.get("network", False)) if isinstance(constraints_data, dict) else False
    return level, network


def resolve_policy(
    task_file: Path,
    config_level: str | None,
    cli_level: str | None = None,
) -> IsolationPolicy | None:
    """Resolve the effective sandbox policy: CLI flag > task.yaml > config.

    "auto" (from any of the three sources) is expanded to a concrete level via
    resolve_effective_level()/default_level_for_host() before it ever reaches
    an IsolationPolicy. Returns None when the resolved level is "none" --
    which happens either because "none" was explicitly requested, or because
    "auto" resolved to "none" on a host without usable bwrap. Shared by
    ``perflab agent`` and the MCP server's agent tools so every entrypoint
    that runs candidate code applies the same policy — the MCP tools
    previously built AgentConfig without isolation, silently downgrading a
    task.yaml that asked for a sandbox.

    Raises ValueError (from normalize_level) on an invalid level string.
    """
    task_level, network = read_task_isolation(task_file)
    requested = resolve_level(cli_level, task_level, config_level)
    level = resolve_effective_level(requested)
    if level == "none":
        return None
    return IsolationPolicy(level=level, network=network)


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
    """The level "auto" resolves to on this host: restricted if bwrap is
    usable, else none.

    Wired in as the actual config/CLI/task.yaml default as of 2026-07-19 --
    ``IsolationSection.level`` defaults to "auto" (perflab/config.py) and
    resolve_policy()/resolve_effective_level() call this to expand it. See
    DESIGN.md for why the originally-planned one-off benchmark-noise A/B was
    superseded by CI as the validation mechanism (bwrap acceptance tests plus
    a real-task ci-check run under ``restricted``, both exercised on every
    push). macOS (no bwrap) is unaffected: this always returns "none" there.
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


def _network_bind_paths() -> list[Path]:
    """Existing DNS/TLS config paths from _NETWORK_RO_BIND_CANDIDATES."""
    return [Path(p) for p in _NETWORK_RO_BIND_CANDIDATES if Path(p).exists()]


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


def wrap_command(
    cmd: list[str],
    policy: IsolationPolicy,
    extra_fds: list[int] | None = None,
) -> list[str]:
    """Return ``cmd``, optionally prefixed with a bwrap invocation.

    Falls back to returning ``cmd`` unchanged whenever the requested level
    can't be honored on this host (non-Linux, bwrap missing/unusable), always
    logging why so the fallback isn't silent.

    ``strict`` adds a seccomp syscall-denial filter (perflab.tools.seccomp)
    via ``bwrap --seccomp FD``. The fd is a fresh memfd holding the compiled
    BPF program; bwrap reads it in the child, so it must survive the spawn:
    wrap_command appends it to ``extra_fds`` and the caller must hand those
    fds to run_cmd's ``pass_fds`` (which forwards them to the subprocess and
    closes them in the parent afterwards -- one wrap_command call per spawn,
    never reuse a wrapped argv for a second run). A caller that passes
    extra_fds=None is declaring it can't carry fds across the spawn, so the
    seccomp layer is skipped with a warning rather than emitting a --seccomp
    argument pointing at an fd the child will never see.
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

    if policy.network:
        # Task explicitly allowed network access (constraints.network: true),
        # so --unshare-net is skipped below -- bind the DNS/TLS config
        # candidate code needs to actually use that network from inside the
        # sandbox (see _NETWORK_RO_BIND_CANDIDATES).
        for p in _network_bind_paths():
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
        args += _seccomp_args(extra_fds)

    return [*args, "--", *cmd]


def _seccomp_args(extra_fds: list[int] | None) -> list[str]:
    """bwrap args for strict's seccomp layer, or [] (with a warning) when it
    can't be applied. Appends the filter fd to extra_fds on success -- see
    wrap_command's docstring for the fd lifetime contract."""
    if not _bwrap_supports_seccomp():
        logger.warning(
            "Isolation level 'strict' requested but this bwrap build has "
            "no --seccomp support -- running with 'restricted'-equivalent "
            "protections only."
        )
        return []
    if extra_fds is None:
        logger.warning(
            "Isolation level 'strict' requested but this caller cannot carry "
            "file descriptors across the spawn (extra_fds=None) -- running "
            "with 'restricted'-equivalent protections only."
        )
        return []
    try:
        fd = seccomp.filter_memfd()
    except (seccomp.SeccompUnavailableError, OSError) as exc:
        logger.warning(
            "Isolation level 'strict' requested but the seccomp filter could "
            "not be prepared (%s) -- running with 'restricted'-equivalent "
            "protections only.", exc,
        )
        return []
    extra_fds.append(fd)
    return ["--seccomp", str(fd)]
