"""Structured configuration for PerfLab.

Loads settings from (in priority order):
1. Environment variables (PERFLAB_*)  — always win
2. Project-local ``./perflab.yaml``   — per-project overrides
3. User-level ``~/.config/perflab/config.yaml`` — global defaults
4. Dataclass defaults                 — sensible out-of-the-box

Some env vars exist only as subprocess transport — they are set by PerfLab
when launching benchmarks/profilers and are NOT part of this config file.
See the ``# Subprocess-only`` comments in the default YAML template.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from perflab.llm.config import DEFAULT_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LLMSection:
    """LLM provider settings."""
    provider: str = "openai"
    model: str = DEFAULT_MODEL
    api_base: str = ""
    temperature: float = 0.7
    max_tokens: int = 16000
    # api_key is deliberately NOT stored in this config — it comes from
    # PERFLAB_API_KEY env var only, so it never gets written to disk or
    # serialized into run artifacts.


@dataclass
class BenchmarkSection:
    """Benchmark harness defaults (can be overridden per-task via task.yaml)."""
    warmup: int = 3
    repeats: int = 20


@dataclass
class ProfilerSection:
    """Profiler behavior settings."""
    torch_with_flops: bool = True
    peaks_cache: str = str(Path.home() / ".cache" / "perflab" / "peaks.json")
    peaks_no_cache: bool = False
    # Allow profilers (py-spy et al.) to retry a failed run under `sudo -n`.
    # Off by default: the retried command re-runs the candidate-patched
    # benchmark as root, outside every rlimit/env protection.
    allow_sudo: bool = False


@dataclass
class MPSSection:
    """Apple Silicon MPS device selection."""
    device_match: str = ""
    device_index: int | None = None


@dataclass
class OllamaSection:
    """Ollama provider security settings."""
    allow_remote: bool = False
    allowed_ports: list[int] = field(default_factory=list)


@dataclass
class AgentSection:
    """Agent loop defaults (can be overridden per-task in task.yaml or via CLI flags).

    n_candidates/max_iters are kept in sync with perflab.task_spec.AgentSpec,
    which is the actual source of truth for these two: cli.py's resolution
    chain (``candidates or task.agent.n_candidates or ga.n_candidates``) only
    ever falls through to this section's values when task.agent.n_candidates/
    max_iters are falsy, but AgentSpec's own dataclass defaults (3 / 12) are
    always truthy -- so unless a task.yaml explicitly sets one to 0, this
    section's n_candidates/max_iters are structurally unreachable defaults.
    """
    n_candidates: int = 3          # LLM candidates per iteration
    max_iters: int = 12            # maximum optimization iterations
    max_wall_time_s: int = 3600    # wall-clock budget in seconds
    fast_screen: bool = True       # use fast benchmark screening for candidates
    max_history: int = 3           # recent iterations included in LLM prompt
    prompt_token_budget: int = 0   # 0 = unlimited; cap prompt to this many tokens


@dataclass
class IsolationSection:
    """OS-level sandboxing for candidate (LLM-authored) subprocess execution.

    See perflab/tools/isolation.py for the "auto" | "none" | "restricted" |
    "strict" tiers. Default is "auto" (resolves to "restricted" on a host
    with usable bwrap, else "none") as of 2026-07-19 -- CI (bwrap acceptance
    tests plus a real-task ci-check run under "restricted") is the ongoing
    validation mechanism; see DESIGN.md.
    """
    level: str = "auto"  # "auto" | "none" | "restricted" | "strict"


@dataclass
class PerfLabConfig:
    """Top-level PerfLab configuration.

    Combines all config sections into a single typed structure.
    Loaded once at startup, accessible throughout the session.
    """
    llm: LLMSection = field(default_factory=LLMSection)
    benchmark: BenchmarkSection = field(default_factory=BenchmarkSection)
    profiler: ProfilerSection = field(default_factory=ProfilerSection)
    mps: MPSSection = field(default_factory=MPSSection)
    ollama: OllamaSection = field(default_factory=OllamaSection)
    agent: AgentSection = field(default_factory=AgentSection)
    isolation: IsolationSection = field(default_factory=IsolationSection)
    # Sparse dict of AnalysisThresholds field overrides.
    # Only specified keys override defaults; unset keys use AnalysisThresholds defaults.
    # task.yaml analysis_thresholds override these (task > config > defaults).
    analysis_thresholds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe, no api_key)."""
        return dataclasses.asdict(self)

    def save(self, path: Path) -> None:
        """Save resolved config to JSON (for run reproducibility)."""
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_USER_CONFIG_PATH = Path.home() / ".config" / "perflab" / "config.yaml"
_PROJECT_CONFIG_NAME = "perflab.yaml"

# Singleton — loaded once, reused across the session
_cached_config: PerfLabConfig | None = None


def _find_project_config() -> Path | None:
    """Walk up from cwd to find a project-level perflab.yaml."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / _PROJECT_CONFIG_NAME
        if candidate.exists():
            return candidate
        # Stop at filesystem root or home directory
        if parent == Path.home() or parent == parent.parent:
            break
    return None


def _safe_set(obj: object, attr: str, raw: object, cast) -> None:
    """Set obj.attr to cast(raw), warning instead of crashing on a bad value.

    A typo in perflab.yaml (e.g. ``temperature: abc``) must degrade to the
    default, not abort every command at config-load time. Mirrors the per-key
    ValueError guards in _overlay_env.
    """
    try:
        setattr(obj, attr, cast(raw))
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid config value %s=%r in perflab.yaml", attr, raw)


def _overlay_yaml(cfg: PerfLabConfig, data: dict) -> None:
    """Overlay a parsed YAML dict onto a PerfLabConfig instance."""
    if not isinstance(data, dict):
        return

    llm = data.get("llm", {})
    if isinstance(llm, dict):
        for key in ("provider", "model", "api_base", "temperature", "max_tokens"):
            if key in llm:
                _safe_set(cfg.llm, key, llm[key], type(getattr(cfg.llm, key)))

    bench = data.get("benchmark", {})
    if isinstance(bench, dict):
        if "warmup" in bench:
            _safe_set(cfg.benchmark, "warmup", bench["warmup"], int)
        if "repeats" in bench:
            _safe_set(cfg.benchmark, "repeats", bench["repeats"], int)

    prof = data.get("profiler", {})
    if isinstance(prof, dict):
        if "torch_with_flops" in prof:
            cfg.profiler.torch_with_flops = bool(prof["torch_with_flops"])
        if "peaks_cache" in prof:
            cfg.profiler.peaks_cache = str(prof["peaks_cache"])
        if "peaks_no_cache" in prof:
            cfg.profiler.peaks_no_cache = bool(prof["peaks_no_cache"])
        if "allow_sudo" in prof:
            cfg.profiler.allow_sudo = bool(prof["allow_sudo"])

    mps = data.get("mps", {})
    if isinstance(mps, dict):
        if "device_match" in mps:
            cfg.mps.device_match = str(mps["device_match"] or "")
        if "device_index" in mps:
            val = mps["device_index"]
            _safe_set(cfg.mps, "device_index", val, lambda v: int(v) if v is not None else None)

    ollama = data.get("ollama", {})
    if isinstance(ollama, dict):
        if "allow_remote" in ollama:
            cfg.ollama.allow_remote = bool(ollama["allow_remote"])
        if "allowed_ports" in ollama:
            _safe_set(cfg.ollama, "allowed_ports", ollama["allowed_ports"],
                      lambda v: [int(p) for p in (v or [])])

    agent = data.get("agent", {})
    if isinstance(agent, dict):
        for key in ("n_candidates", "max_iters", "max_wall_time_s", "max_history", "prompt_token_budget"):
            if key in agent:
                _safe_set(cfg.agent, key, agent[key], int)
        if "fast_screen" in agent:
            cfg.agent.fast_screen = bool(agent["fast_screen"])

    isolation = data.get("isolation", {})
    if isinstance(isolation, dict):
        if "level" in isolation:
            cfg.isolation.level = str(isolation["level"])

    thresholds = data.get("analysis_thresholds", {})
    if isinstance(thresholds, dict):
        cfg.analysis_thresholds.update(thresholds)


def _overlay_env(cfg: PerfLabConfig) -> None:
    """Overlay environment variables onto the config (highest priority)."""
    # LLM (api_key handled separately in LLMConfig — never stored in PerfLabConfig)
    if v := os.environ.get("PERFLAB_LLM_PROVIDER"):
        cfg.llm.provider = v
    if v := os.environ.get("PERFLAB_LLM_MODEL"):
        cfg.llm.model = v
    if v := os.environ.get("PERFLAB_API_BASE"):
        cfg.llm.api_base = v

    # Benchmark
    if v := os.environ.get("PERFLAB_BENCH_WARMUP"):
        try:
            cfg.benchmark.warmup = int(v)
        except ValueError:
            pass
    if v := os.environ.get("PERFLAB_BENCH_REPEATS"):
        try:
            cfg.benchmark.repeats = int(v)
        except ValueError:
            pass

    # Profiler
    if v := os.environ.get("PERFLAB_PEAKS_CACHE"):
        cfg.profiler.peaks_cache = v
    if os.environ.get("PERFLAB_PEAKS_NO_CACHE", "").strip() == "1":
        cfg.profiler.peaks_no_cache = True
    if os.environ.get("PERFLAB_PROFILER_ALLOW_SUDO", "").strip() == "1":
        cfg.profiler.allow_sudo = True

    # MPS
    if v := os.environ.get("PERFLAB_MPS_DEVICE_MATCH"):
        cfg.mps.device_match = v
    if v := os.environ.get("PERFLAB_MPS_DEVICE_INDEX"):
        try:
            cfg.mps.device_index = int(v)
        except ValueError:
            pass

    # Ollama
    if os.environ.get("PERFLAB_OLLAMA_ALLOW_REMOTE"):
        cfg.ollama.allow_remote = True
    if v := os.environ.get("PERFLAB_OLLAMA_ALLOWED_PORTS"):
        try:
            cfg.ollama.allowed_ports = [int(p) for p in v.split(",") if p.strip()]
        except ValueError:
            pass

    # Isolation
    if v := os.environ.get("PERFLAB_ISOLATION_LEVEL"):
        cfg.isolation.level = v


def load_config(*, force_reload: bool = False) -> PerfLabConfig:
    """Load and return the PerfLabConfig singleton.

    Resolution order: defaults → user config → project config → env vars.
    Results are cached — subsequent calls return the same instance unless
    ``force_reload=True``.
    """
    global _cached_config
    if _cached_config is not None and not force_reload:
        return _cached_config

    cfg = PerfLabConfig()

    # Layer 1: User-level config (~/.config/perflab/config.yaml)
    if _USER_CONFIG_PATH.exists():
        try:
            data = yaml.safe_load(_USER_CONFIG_PATH.read_text(encoding="utf-8"))
            _overlay_yaml(cfg, data)
        except (yaml.YAMLError, OSError):
            logger.warning("Failed to load user config %s", _USER_CONFIG_PATH, exc_info=True)

    # Layer 2: Project-level config (./perflab.yaml, walks up)
    project_config = _find_project_config()
    if project_config:
        try:
            data = yaml.safe_load(project_config.read_text(encoding="utf-8"))
            _overlay_yaml(cfg, data)
        except (yaml.YAMLError, OSError):
            logger.warning("Failed to load project config %s", project_config, exc_info=True)

    # Layer 3: Environment variables (always win)
    _overlay_env(cfg)

    _cached_config = cfg
    return cfg


def create_project_config(directory: Path | None = None) -> Path:
    """Create a perflab.yaml in the given directory with the default template.

    Returns the path to the created file.
    """
    target = (directory or Path.cwd()) / _PROJECT_CONFIG_NAME
    target.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    return target


def create_user_config() -> Path:
    """Create the user-level config at ~/.config/perflab/config.yaml.

    Returns the path to the created file.
    """
    _USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _USER_CONFIG_PATH.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    return _USER_CONFIG_PATH


# ---------------------------------------------------------------------------
# Default YAML template (for `perflab init` and documentation)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_TEMPLATE = f"""\
# PerfLab Configuration
# =====================
# Resolution order: env vars > ./perflab.yaml > ~/.config/perflab/config.yaml > defaults
#
# This file controls PerfLab's behavior. Place it at:
#   ~/.config/perflab/config.yaml   (user-level defaults)
#   ./perflab.yaml                  (project-level overrides)

llm:
  provider: openai           # "openai", "anthropic", or "ollama"
  model: {DEFAULT_MODEL}            # Model identifier
  api_base: ""               # Custom API endpoint (leave empty for default)
  temperature: 0.7
  max_tokens: 16000
  # NOTE: api_key is NOT stored here for security — always use:
  #   export PERFLAB_API_KEY=sk-...

benchmark:
  warmup: 3                  # Warmup iterations before timing
  repeats: 20                # Timed repetitions per benchmark

profiler:
  torch_with_flops: true     # Enable per-operator FLOPS counting
  peaks_cache: "~/.cache/perflab/peaks.json"
  peaks_no_cache: false      # Set true to force re-detection
  allow_sudo: false          # Allow profilers to retry under `sudo -n` when a run
                             # fails (e.g. py-spy needing elevated permissions).
                             # WARNING: the retried command re-runs the
                             # candidate-patched benchmark as root.

mps:
  device_match: ""           # Substring match for Metal device name
  device_index: null         # Integer index for Metal device selection

ollama:
  allow_remote: false        # Allow connecting to non-localhost Ollama
  allowed_ports: []          # Extra allowed ports (default: 11434 only)

agent:
  n_candidates: 3            # LLM candidates generated per iteration
  max_iters: 12              # Maximum optimization iterations
  max_wall_time_s: 3600      # Wall-clock budget (seconds). 3600 = 1 hour
  fast_screen: true          # Quick benchmark to rank candidates before full eval
  max_history: 3             # How many past iterations the LLM sees in its prompt
  prompt_token_budget: 0     # 0 = unlimited. Set to cap prompt size for small models

isolation:
  level: auto                # "auto" | "none" | "restricted" | "strict" -- sandbox
                              # candidate (LLM-authored) subprocess execution on top
                              # of the rlimit + env-allowlist protections that always
                              # apply.
                              # auto:       (default) restricted if this host has
                              #             usable bwrap (Linux + working user
                              #             namespaces), else none. See
                              #             default_level_for_host().
                              # none:       rlimits only, unsandboxed.
                              # restricted: Bubblewrap (Linux only) -- read-only bind
                              #             of /usr, /lib, the venv, and CUDA/driver
                              #             paths; read-write bind of the workspace
                              #             only; network unshared unless task.yaml
                              #             sets constraints.network: true. Falls back
                              #             to none (with a logged warning) if bwrap
                              #             isn't installed/usable, or on non-Linux.
                              # strict:     restricted + seccomp (ptrace/mount/bpf/
                              #             keyctl denied) where bwrap supports it.
                              # See perflab/tools/isolation.py and DESIGN.md.

# ---------------------------------------------------------------------------
# Analysis thresholds (optional — override bottleneck detection sensitivity)
# ---------------------------------------------------------------------------
# These control when the bottleneck analyzer fires diagnoses. Only specify
# values you want to change — unset keys use AnalysisThresholds defaults.
# task.yaml analysis_thresholds override these (task > config > defaults).
#
# Uncomment and adjust for your hardware/domain:
#
# analysis_thresholds:
#   # GPU occupancy: lower for HPC kernels that intentionally use many registers
#   ncu_occupancy_low: 50.0          # default 50% — flag kernels below this
#
#   # Tensor Core utilization: raise for inference, lower for training
#   ncu_tc_util_low: 30.0            # default 30% — flag TC-capable but unused
#
#   # Cache miss rate: lower for latency-sensitive, raise for throughput workloads
#   perf_cache_miss_rate_high: 0.05  # default 5% — flag above this
#
#   # IPC: lower for memory-bound codes where low IPC is expected
#   perf_ipc_low: 1.0               # default 1.0 — flag below this
#
#   # GPU active percentage: lower for interactive/serving, raise for batch
#   nsys_gpu_fraction_low: 0.5      # default 50% — flag below this
#
#   # Lock contention: raise for known-contentious architectures
#   lock_contention_ratio_high: 0.10 # default 10% — contended/acquired ratio
#
#   # Full list: see AnalysisThresholds in perflab/analyzers/bottleneck_analyzer.py

# ---------------------------------------------------------------------------
# Subprocess-only environment variables (NOT configurable here)
# ---------------------------------------------------------------------------
# These env vars are set automatically by PerfLab when launching benchmarks
# and profilers. They cross process boundaries and must stay as env vars.
# They are documented here for reference but cannot be set in this file.
#
# PERFLAB_API_KEY            — LLM API key (security: never written to disk)
# PERFLAB_DETERMINISM_SEED   — Set to 42 during correctness testing
# PERFLAB_ACCURACY_TOLERANCE — task.yaml constraints.accuracy_tolerance,
#                              forwarded to correctness subprocesses
# PERFLAB_TORCH_PROFILE      — Set to "1" to enable torch profiler capture
# PERFLAB_TORCH_TRACE_PATH   — Path for torch profiler Chrome trace output
# PERFLAB_TORCH_WITH_FLOPS   — Set to "1" for per-op FLOPS counting
# PERFLAB_JAX_TRACE_DIR      — Directory for JAX profiler trace output
# PERFLAB_CXXFLAGS           — Diagnostic compiler flags for C++ builds
# PERFLAB_NVCCFLAGS          — Diagnostic compiler flags for CUDA builds
# PERFLAB_BENCH_WARMUP       — Overridden during fast screening (set to 0)
# PERFLAB_BENCH_REPEATS      — Overridden during fast screening (set to 2)
"""
