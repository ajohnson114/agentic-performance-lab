from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_logger = logging.getLogger(__name__)

# Marker preexec_fn writes to fd 2 (captured as stderr) when a resource limit
# fails to apply. Matched on a single line since exception messages (e.g.
# OSError's "[Errno 11] ...") may themselves contain "]"; greedy matching
# extends to the rightmost "]" before the newline so those aren't truncated.
_RLIMIT_MARKER_RE = re.compile(r"\[perflab-rlimit-failed (.*)\]\n?")

# Same mechanism for the CPU-environment half of preexec_fn (affinity, nice).
# Kept as a separate marker so a pinning failure is not reported to the user as
# an rlimit failure -- they have different causes and different fixes.
_CPUENV_MARKER_RE = re.compile(r"\[perflab-cpuenv-failed (.*)\]\n?")


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
    # None = no CPU affinity/nice was requested for this command (or no
    # preexec_fn ran). True = requested and applied cleanly. False = requested
    # but sched_setaffinity/nice failed in the child.
    cpu_env_applied: bool | None = None


def _make_linux_preexec(
    rlimit_as_bytes: int | None,
    rlimit_nproc: int = 512,
    *,
    cpus: Sequence[int] = (),
    nice_adj: int | None = None,
) -> Callable[[], None] | None:
    """Return a preexec_fn that sets the child's CPU environment and resource
    limits on Linux, or None off Linux.

    cpus: CPU ids to pin the child to via sched_setaffinity(0, cpus). Affinity
    survives execve and is inherited by the child's own threads and children,
    so pinning the process we spawn also pins everything the benchmark forks
    (including a bwrap wrapper's payload under --isolation).

    nice_adj: nice(2) adjustment, applied only when the caller has already
    established that this process may actually apply it (see _can_renice) --
    a positive adjustment would *lower* the benchmark's priority, which is the
    opposite of what a measurement wants.
    """
    if platform.system() != "Linux":
        return None

    def _preexec() -> None:
        import resource

        # preexec_fn runs post-fork/pre-exec: it cannot log through the
        # parent's logger, but fd 2 is already dup'd onto the child's
        # stderr pipe at this point, so writing there surfaces failures
        # to run_cmd via the captured CmdResult.stderr.

        # CPU environment first: a benchmark that runs on the wrong cores is
        # mismeasured, which is worse than one that runs without a memory cap.
        cpu_failures = []
        if cpus:
            try:
                # Linux-only API; this whole preexec is gated on Linux above.
                # (typeshed hides it on the macOS dev box, hence the ignore.)
                os.sched_setaffinity(0, cpus)  # type: ignore[attr-defined]
            except (OSError, ValueError, AttributeError) as exc:
                cpu_failures.append(f"AFFINITY:{exc}")
        if nice_adj:
            try:
                os.nice(nice_adj)
            except OSError as exc:
                cpu_failures.append(f"NICE:{exc}")
        if cpu_failures:
            os.write(2, f"[perflab-cpuenv-failed {' '.join(cpu_failures)}]\n".encode())

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


# ---------------------------------------------------------------------------
# CPU environment control for measured subprocesses
# ---------------------------------------------------------------------------
#
# PerfLab's analyzer tells users to pin threads with taskset; this is the code
# that makes PerfLab pin its own. An unpinned benchmark is rescheduled across
# cores mid-run, which invalidates caches, crosses SMT siblings and (on
# multi-socket boxes) NUMA nodes -- run-to-run spread from that routinely
# exceeds the size of the optimizations the agent is trying to find.
#
# THE INVARIANT THAT MATTERS: the plan is resolved once per process and cached
# (resolve_cpu_plan), and the runners read it themselves rather than taking it
# as a per-call argument. There is deliberately no per-call override anywhere
# in the benchmark path -- if the baseline could be measured on a different
# core set than the candidates, every candidate would gain or lose speed for
# free, a systematic bias far worse than the noise pinning removes. See
# tests/test_cpu_pinning.py::TestBaselineCandidateParity.

_SYS_CPU = Path("/sys/devices/system/cpu")

# Nice adjustment for measured subprocesses. Applied ONLY when this process can
# actually raise priority (root, or a raised RLIMIT_NICE). Without that, nice(2)
# can only *lower* priority, which would make measurements worse, so we skip it
# and say so rather than pretending it did something.
_BENCH_NICE_ADJ = -5

# Values of the pinning spec that mean "leave the scheduler alone".
_PINNING_OFF = frozenset({"off", "none", "false", "no", "0", "disabled"})


@dataclass(frozen=True)
class CpuPlan:
    """The resolved CPU environment applied to every measured subprocess.

    cpus: CPU ids to pin to. Empty means pinning is not in effect; ``reason``
    then explains why (non-Linux, disabled, unusable topology, ...).
    omp_env: OMP_* vars to overlay on the benchmark environment. CPU affinity
    alone does not bind OpenMP threads -- libgomp/libiomp will happily migrate
    them within the mask -- so OMP_PROC_BIND/OMP_PLACES are set alongside it.
    Vars already present in the parent environment are omitted: an explicit
    ``export OMP_NUM_THREADS=...`` is a deliberate choice and outranks ours.
    nice_adj: nice(2) adjustment, or None when the host does not permit
    raising priority (the common case for an unprivileged user).
    governors/turbo/advice: detected frequency-scaling state and the exact
    commands to fix it. PerfLab reports these and never applies them -- see
    _log_cpu_plan.
    """

    cpus: tuple[int, ...] = ()
    omp_env: Mapping[str, str] = field(default_factory=dict)
    nice_adj: int | None = None
    reason: str = ""
    governors: tuple[str, ...] = ()
    turbo: str | None = None  # "on" | "off" | None (unknown/unavailable)
    advice: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.cpus)


def _read_sys(path: Path) -> str | None:
    """Read a sysfs file, returning None if it is missing or unreadable."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None


def _parse_cpu_list(text: str) -> list[int]:
    """Parse a Linux CPU list ("0-3,8", "2,4,6") into explicit ids.

    Malformed fragments are dropped rather than raising: this parses both
    sysfs (trusted) and a user-supplied constraints value (typo-prone), and a
    bad character in perflab.yaml must not abort the run.
    """
    out: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                continue
            if 0 <= lo <= hi:
                out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out


def _physical_cores(available: set[int]) -> list[tuple[int, int, tuple[int, ...]]]:
    """Collapse logical CPUs to physical cores: [(package_id, rep_cpu, siblings)].

    Reads topology/thread_siblings_list, so two hyperthreads of one physical
    core yield a single entry. That matters: spreading a 2-thread benchmark
    across two SMT siblings of the same core is far slower (and noisier) than
    spreading it across two physical cores, and a naive "first N cpus" pick
    lands on siblings on every Intel/AMD box with SMT enabled.

    The representative is the lowest-numbered sibling that is actually inside
    ``available`` (the process's own affinity mask, which already reflects any
    cgroup cpuset), so a container restricted to odd CPUs still gets a valid
    pick. Cores with no usable sibling are dropped.
    """
    seen: set[tuple[int, ...]] = set()
    cores: list[tuple[int, int, tuple[int, ...]]] = []
    for cpu in sorted(available):
        raw = _read_sys(_SYS_CPU / f"cpu{cpu}" / "topology" / "thread_siblings_list")
        siblings = tuple(sorted(_parse_cpu_list(raw))) if raw else ()
        if not siblings:
            siblings = (cpu,)  # no topology exposed (VM/container): 1 cpu == 1 core
        if siblings in seen:
            continue
        seen.add(siblings)
        usable = [c for c in siblings if c in available]
        if not usable:
            continue
        pkg_raw = _read_sys(
            _SYS_CPU / f"cpu{usable[0]}" / "topology" / "physical_package_id"
        )
        try:
            pkg = int(pkg_raw) if pkg_raw is not None else 0
        except ValueError:
            pkg = 0
        cores.append((pkg, usable[0], siblings))
    return cores


def _select_cpus(available: set[int], want: int | None) -> tuple[tuple[int, ...], str]:
    """Choose the CPU set to pin to. Returns (cpus, human-readable rationale).

    Policy, in order:
      1. One logical CPU per *physical* core (never two SMT siblings).
      2. Stay inside a single package/socket -- the one with the most usable
         cores. Cross-socket scheduling is itself a noise source, and a
         benchmark that fits on one socket should not straddle two.
      3. Avoid the physical core hosting CPU 0, which is where the kernel's
         default IRQ affinity and most housekeeping work land. Relaxed only if
         excluding it would leave too few cores.
      4. ``want`` cores if given (a single-threaded task wants 1), else every
         core that survives (1)-(3), so an OpenMP task keeps its parallelism.
    """
    cores = _physical_cores(available)
    if not cores:
        return (), "no usable CPU topology"

    by_pkg: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for pkg, rep, siblings in cores:
        by_pkg.setdefault(pkg, []).append((rep, siblings))
    # Largest package wins; ties break toward the lowest package id.
    pkg = sorted(by_pkg, key=lambda p: (-len(by_pkg[p]), p))[0]
    pool = by_pkg[pkg]

    notes: list[str] = []
    if len(by_pkg) > 1:
        notes.append(f"package {pkg} of {len(by_pkg)}")

    preferred = sorted(rep for rep, siblings in pool if 0 not in siblings)
    with_core0 = sorted(rep for rep, _ in pool)
    need = want if want is not None else 1
    if len(preferred) >= need and preferred:
        chosen_pool = preferred
        notes.append("core 0 excluded")
    else:
        chosen_pool = with_core0
        if len(with_core0) > len(preferred):
            notes.append("core 0 included (too few cores without it)")

    if want is None:
        chosen = tuple(chosen_pool)
        notes.append(f"all {len(chosen)} physical cores")
    else:
        chosen = tuple(chosen_pool[:want])
        if len(chosen) < want:
            notes.append(f"requested {want} cores, only {len(chosen)} available")
        else:
            notes.append(f"{want} physical core{'s' if want != 1 else ''}")

    smt = sum(1 for _, _, siblings in cores if len(siblings) > 1)
    if smt:
        notes.append("SMT siblings excluded")
    return chosen, "; ".join(notes)


def _can_renice() -> bool:
    """True when this process may actually *raise* priority (nice < 0).

    Unprivileged processes can only lower their priority, so applying nice(2)
    without this check would make the measurement worse, not better.
    """
    try:
        if os.geteuid() == 0:
            return True
    except AttributeError:  # pragma: no cover -- non-POSIX
        return False
    try:
        import resource

        # RLIMIT_NICE is Linux-only; absent on the macOS dev box's typeshed.
        soft, _hard = resource.getrlimit(resource.RLIMIT_NICE)  # type: ignore[attr-defined]
    except (ImportError, AttributeError, ValueError, OSError):
        return False
    if soft == resource.RLIM_INFINITY:
        return True
    # RLIMIT_NICE soft value X permits a nice floor of 20 - X.
    return 20 - soft <= _BENCH_NICE_ADJ


def _cpu_freq_state() -> tuple[tuple[str, ...], str | None, tuple[str, ...]]:
    """Detect scaling governor + turbo/boost state. Returns (governors, turbo, fixes).

    Pure detection. PerfLab never flips a machine-wide setting on the user's
    behalf -- same stance as perflab/profilers/base.py declining to re-run
    py-spy under sudo -- so the third element is the exact command to opt in.
    """
    governors: set[str] = set()
    try:
        for path in sorted(_SYS_CPU.glob("cpu[0-9]*/cpufreq/scaling_governor")):
            gov = _read_sys(path)
            if gov:
                governors.add(gov)
    except OSError:
        pass

    turbo: str | None = None
    intel_no_turbo = _SYS_CPU / "intel_pstate" / "no_turbo"
    generic_boost = _SYS_CPU / "cpufreq" / "boost"
    raw = _read_sys(intel_no_turbo)
    if raw in ("0", "1"):
        turbo = "off" if raw == "1" else "on"
    else:
        raw = _read_sys(generic_boost)
        if raw in ("0", "1"):
            turbo = "on" if raw == "1" else "off"

    fixes: list[str] = []
    if governors and governors != {"performance"}:
        fixes.append("sudo cpupower frequency-set -g performance")
    if turbo == "on":
        if intel_no_turbo.exists():
            fixes.append(f"echo 1 | sudo tee {intel_no_turbo}")
        else:
            fixes.append(f"echo 0 | sudo tee {generic_boost}")
    return tuple(sorted(governors)), turbo, tuple(fixes)


def _build_cpu_plan(spec: str) -> CpuPlan:
    """Resolve a pinning spec into a concrete CpuPlan. Never raises."""
    governors, turbo, advice = _cpu_freq_state()

    def _off(reason: str) -> CpuPlan:
        """A no-op plan: pinning is not applied, but the frequency findings
        (which are independent of pinning) are still reported."""
        return CpuPlan(reason=reason, governors=governors, turbo=turbo, advice=advice)

    if platform.system() != "Linux":
        return _off(
            f"unavailable on {platform.system()} "
            "(no sched_setaffinity / cpufreq governor)"
        )

    normalized = (spec or "auto").strip().lower()
    if normalized in _PINNING_OFF:
        return _off("disabled by configuration")

    try:
        available = set(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError) as exc:
        return _off(f"sched_getaffinity unavailable ({exc})")
    if not available:
        return _off("empty CPU affinity mask")

    if normalized == "auto":
        chosen, rationale = _select_cpus(available, None)
    elif normalized.isdigit():
        want = int(normalized)
        if want <= 0:
            return _off(f"disabled ({spec!r} requests 0 cores)")
        chosen, rationale = _select_cpus(available, want)
    else:
        requested = _parse_cpu_list(normalized)
        chosen = tuple(sorted(set(requested) & available))
        if not chosen:
            return _off(
                f"explicit cpu list {spec!r} has no CPU inside this process's "
                f"affinity mask {sorted(available)}"
            )
        dropped = sorted(set(requested) - available)
        rationale = f"explicit cpu list {spec!r}" + (
            f" ({len(dropped)} outside the affinity mask dropped)" if dropped else ""
        )

    if not chosen:
        return _off(f"no CPUs selected ({rationale})")

    # Affinity confines the process; it does not bind OpenMP threads within
    # the mask. PROC_BIND=close + PLACES=cores is the spec-defined way to get
    # one thread per core, and both libgomp and libiomp intersect it with the
    # affinity mask we just set.
    omp_env = {
        key: value
        for key, value in (
            ("OMP_PROC_BIND", "close"),
            ("OMP_PLACES", "cores"),
            ("OMP_NUM_THREADS", str(len(chosen))),
        )
        if key not in os.environ
    }
    return CpuPlan(
        cpus=chosen,
        omp_env=omp_env,
        nice_adj=_BENCH_NICE_ADJ if _can_renice() else None,
        reason=rationale,
        governors=governors,
        turbo=turbo,
        advice=advice,
    )


_cached_cpu_plan: CpuPlan | None = None
_task_cpu_pinning: str | None = None


def set_task_cpu_pinning(spec: str | None) -> None:
    """Publish the loaded task's ``constraints.cpu_pinning`` (None = unset).

    Called from TaskSpec.load so the task's setting reaches every measured
    subprocess without being threaded through call sites -- which is what keeps
    baseline and candidate runs on identical cores by construction. Changing it
    invalidates the cached plan.
    """
    global _task_cpu_pinning, _cached_cpu_plan
    if spec == _task_cpu_pinning:
        return
    _task_cpu_pinning = spec
    _cached_cpu_plan = None


def _pinning_spec() -> str:
    """Resolve the active spec: env var > task.yaml > perflab.yaml > "auto"."""
    env_spec = os.environ.get("PERFLAB_CPU_PINNING", "").strip()
    if env_spec:
        return env_spec
    if _task_cpu_pinning:
        return _task_cpu_pinning
    try:
        from perflab.config import load_config

        return str(load_config().benchmark.cpu_pinning)
    except Exception:  # noqa: BLE001 -- config is best-effort; "auto" is the default
        return "auto"


def _log_cpu_plan(plan: CpuPlan) -> None:
    """Log the resolved plan once, including the governor finding."""
    if plan.enabled:
        nice_note = (
            f", nice {plan.nice_adj}"
            if plan.nice_adj
            else ", nice not applied (needs root or a raised RLIMIT_NICE: "
            "`ulimit -e 30`, or run as root)"
        )
        _logger.info(
            "Benchmark CPU pinning: cpus=%s (%s)%s",
            ",".join(str(c) for c in plan.cpus), plan.reason, nice_note,
        )
    else:
        _logger.info("Benchmark CPU pinning not applied: %s", plan.reason)

    findings = []
    if plan.governors and set(plan.governors) != {"performance"}:
        findings.append(f"scaling governor is {'/'.join(plan.governors)}, not performance")
    if plan.turbo == "on":
        findings.append("turbo/boost is enabled (clocks drift with temperature)")
    if findings:
        _logger.warning(
            "CPU frequency scaling will add run-to-run spread: %s. PerfLab does "
            "not change machine-wide settings for you; to opt in, run: %s",
            "; ".join(findings), " && ".join(plan.advice) or "(no fix detected)",
        )


def resolve_cpu_plan(*, force: bool = False) -> CpuPlan:
    """Return the process-wide CPU plan for measured subprocesses (cached).

    Every benchmark and correctness run in a PerfLab session resolves through
    this one function, so the baseline and every candidate are measured under
    an identical CPU environment. Pass force=True only in tests.
    """
    global _cached_cpu_plan
    if _cached_cpu_plan is not None and not force:
        return _cached_cpu_plan
    plan = _build_cpu_plan(_pinning_spec())
    _log_cpu_plan(plan)
    _cached_cpu_plan = plan
    return plan


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
    cpu_affinity: Sequence[int] | None = None,
    nice_adj: int | None = None,
) -> CmdResult:
    """Run a command and return the result.

    skip_preexec: If True, skip preexec_fn. Use this when calling from
    threads (e.g. ThreadPoolExecutor) because preexec_fn + fork() in a
    multithreaded process has undefined behavior (Python docs). Note this
    also skips CPU pinning -- callers that need a *measured* run must not set
    it (run_benchmark never does).

    cpu_affinity / nice_adj: CPU environment for the child (Linux only, see
    CpuPlan). Callers running a benchmark should pass resolve_cpu_plan()'s
    values rather than inventing their own, so every measured run shares one
    policy. Default None leaves scheduling untouched, which is what build and
    profiler-tool invocations want.

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
    cpus = tuple(cpu_affinity or ())
    preexec = (
        None if skip_preexec
        else _make_linux_preexec(
            rlimit_as_bytes, rlimit_nproc=rlimit_nproc, cpus=cpus, nice_adj=nice_adj,
        )
    )
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
    cpu_env_applied: bool | None = None
    if preexec is not None:
        match = _RLIMIT_MARKER_RE.search(stderr)
        if match:
            rlimits_applied = False
            _logger.warning("rlimit application failed for %s: %s", list(cmd), match.group(1))
            stderr = _RLIMIT_MARKER_RE.sub("", stderr)
            stdout = _RLIMIT_MARKER_RE.sub("", stdout)
        else:
            rlimits_applied = True

        if cpus or nice_adj:
            cpu_match = _CPUENV_MARKER_RE.search(stderr)
            if cpu_match:
                cpu_env_applied = False
                # A silently-unpinned run is a mismeasurement, not a nuisance:
                # warn loudly so it is visible next to the numbers it affects.
                _logger.warning(
                    "CPU pinning/priority failed for %s: %s -- this run was NOT "
                    "measured under the pinning policy", list(cmd), cpu_match.group(1),
                )
                stderr = _CPUENV_MARKER_RE.sub("", stderr)
                stdout = _CPUENV_MARKER_RE.sub("", stdout)
            else:
                cpu_env_applied = True

    return CmdResult(
        cmd=cmd, returncode=p.returncode, stdout=stdout, stderr=stderr,
        duration_s=t1 - t0, rlimits_applied=rlimits_applied,
        cpu_env_applied=cpu_env_applied,
    )
