"""Build flag override mechanism.

Allows the agent to write a build_overrides.json file with additional
compiler flags. The runner picks these up and appends them to the build
command, scoped to a safe allowlist.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# Allowlist of flags the agent may add via overrides
ALLOWED_FLAGS = frozenset({
    # Optimization levels
    "-O2", "-O3", "-Ofast",
    # Architecture targeting
    "-march=native", "-mtune=native",
    # Vectorization / SIMD
    "-mavx", "-mavx2", "-mavx512f", "-mfma", "-msse4.2",
    # OpenMP
    "-fopenmp",
    # Loop optimizations
    "-funroll-loops", "-ftree-vectorize",
    # LTO
    "-flto",
    # PGO
    "-fprofile-generate", "-fprofile-use",
    # Debug info (for profiling)
    "-g", "-pg",
    # Warnings (harmless)
    "-Wall", "-Wextra",
    # Alignment
    "-falign-functions=32", "-falign-loops=32",
    # Prefetch
    "-fprefetch-loop-arrays",
    # CUDA / nvcc — line info (required for ncu source correlation)
    "-lineinfo", "--generate-line-info",
})

# Flags that require explicit permission (constraints.allow_fast_math: true)
FAST_MATH_FLAGS = frozenset({
    "-ffast-math",
    "-Ofast",           # implies -ffast-math
    "--use_fast_math",  # nvcc equivalent
    "-fno-math-errno",  # subset of fast-math
    "-funsafe-math-optimizations",
    "-ffinite-math-only",
    "-fno-trapping-math",
    "-fassociative-math",
    "-freciprocal-math",
})

# CUDA parameterized flags: matched by regex because their values vary.
# Each pattern must match the entire flag string.
_CUDA_ALLOWED_PATTERNS: list[re.Pattern[str]] = [
    # Target architecture: sm_70 (Volta/V100), sm_75 (Turing/T4),
    # sm_80 (Ampere/A100), sm_86 (Ampere/RTX30), sm_89 (Ada/RTX40/L4),
    # sm_90 / sm_90a (Hopper/H100), native (auto-detect)
    re.compile(r"^-arch=(sm_\d+[a-z]?|native)$"),
    re.compile(r"^--gpu-architecture=(sm_\d+[a-z]?|native)$"),
    # Register cap — limits spilling, affects occupancy
    re.compile(r"^--maxrregcount=\d+$"),
    # PTX assembler flags passed via -Xptxas (e.g. -Xptxas -v)
    re.compile(r"^-Xptxas\s+\S.*$"),
    # Shared memory bank size (4 or 8 bytes)
    re.compile(r"^-Xptxas\s+--bankmode=(4|8)$"),
]

# Groups of mutually exclusive flags — only one per group should be used
_CONFLICTING_GROUPS: list[frozenset[str]] = [
    frozenset({"-O2", "-O3", "-Ofast"}),
]

# Regex for syntactically valid compiler flags: must start with - or --
_FLAG_SYNTAX_RE = re.compile(r"^--?[A-Za-z]")


@dataclass
class RejectedFlag:
    """A flag that was rejected during validation, with a reason."""
    flag: str
    reason: str


@dataclass
class BuildOverrideResult:
    """Result of loading and validating build overrides."""
    accepted: list[str] = field(default_factory=list)
    rejected: list[RejectedFlag] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _detect_conflicts(flags: list[str]) -> list[str]:
    """Return warnings for any mutually exclusive flags present together."""
    warnings: list[str] = []
    for group in _CONFLICTING_GROUPS:
        present = [f for f in flags if f in group]
        if len(present) > 1:
            warnings.append(
                f"Conflicting flags: {', '.join(sorted(present))}. "
                f"Only the last one will take effect."
            )
    return warnings


def load_build_overrides(workspace: Path, allow_fast_math: bool = False) -> list[str]:
    """Load build flag overrides from workspace/build_overrides.json.

    Returns a list of validated flags, filtering out anything not
    in the allowlist. Fast-math flags are only allowed when
    allow_fast_math=True (from constraints.allow_fast_math in task.yaml).
    """
    return load_build_overrides_with_feedback(workspace, allow_fast_math).accepted


def load_build_overrides_with_feedback(
    workspace: Path, allow_fast_math: bool = False
) -> BuildOverrideResult:
    """Load build flag overrides with detailed feedback on rejections.

    Returns a BuildOverrideResult containing accepted flags, rejected
    flags with reasons, and warnings (e.g. conflicting flags).
    """
    result = BuildOverrideResult()
    override_path = workspace / "build_overrides.json"
    if not override_path.exists():
        return result

    try:
        data = json.loads(override_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return result

    if not isinstance(data, dict):
        return result

    flags = data.get("flags", [])
    if not isinstance(flags, list):
        return result

    effective_allowed = ALLOWED_FLAGS | (FAST_MATH_FLAGS if allow_fast_math else frozenset())

    for flag in flags:
        if not isinstance(flag, str):
            result.rejected.append(RejectedFlag(
                flag=repr(flag), reason="not a string"
            ))
            continue

        flag = flag.strip()
        if not flag:
            continue

        # Syntax validation: must look like a compiler flag
        if not _FLAG_SYNTAX_RE.match(flag):
            result.rejected.append(RejectedFlag(
                flag=flag, reason="invalid syntax: flags must start with '-' or '--'"
            ))
            continue

        # Fast-math gate
        if flag in FAST_MATH_FLAGS and not allow_fast_math:
            result.rejected.append(RejectedFlag(
                flag=flag, reason="fast-math flag not allowed (allow_fast_math is false)"
            ))
            continue

        # Allowlist check: exact match or CUDA parameterized pattern
        if flag not in effective_allowed and not any(
            p.match(flag) for p in _CUDA_ALLOWED_PATTERNS
        ):
            result.rejected.append(RejectedFlag(
                flag=flag, reason="not in allowlist"
            ))
            continue

        result.accepted.append(flag)

    # Check for conflicting flags among accepted
    result.warnings = _detect_conflicts(result.accepted)

    return result


def apply_build_overrides(build_cmd: str, overrides: list[str]) -> str:
    """Append validated override flags to a build command string.

    Flags already present in the command are not duplicated.
    """
    if not overrides:
        return build_cmd

    existing = set(build_cmd.split())
    new_flags = [f for f in overrides if f not in existing]
    if not new_flags:
        return build_cmd

    return build_cmd + " " + " ".join(new_flags)
