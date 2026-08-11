"""Forbidden-construct policy: reject optimizations a task deliberately withholds.

This complements the existing anti-gaming checks. Those catch a candidate that
games the *measurement* (zero-variance timings, shrunken problem sizes). This
catches a candidate that reaches for a tool the task rules out.

The motivating case: `matmul/cpp` pins its build line at a fixed -O flag, and
`task.yaml` is protected so the agent cannot change it. An agent nonetheless
wrote `#pragma GCC optimize("O3","unroll-loops")` into the one file it *was*
allowed to edit, silently overriding the pinned build. Measured, that pragma was
worth 4.0x on top of a 9.7x algorithmic rewrite -- so most of the headline
number came from routing around a constraint rather than from optimization.

The check is deliberately general. A task says what is off-limits:

    anti_gaming:
      forbidden_constructs: ["optimization_pragmas", "openmp"]
      forbidden_patterns: ["\\\\bmy_fast_lib\\\\b"]     # custom regex escape hatch

Only text a candidate *introduces* is checked (the REPLACE side of each edit
block), not the file it edits. A baseline that already contains a construct is
a task-authoring choice, not a candidate violation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ForbiddenConstruct:
    """A named bundle of patterns, with prose the prompt can show the model."""

    description: str
    patterns: tuple[str, ...]


# Named presets. Keys are what a task.yaml writes in forbidden_constructs.
FORBIDDEN_CONSTRUCTS: dict[str, ForbiddenConstruct] = {
    "optimization_pragmas": ForbiddenConstruct(
        description=(
            "compiler optimization-level pragmas/attributes "
            "(they override the pinned build command)"
        ),
        patterns=(
            r"#\s*pragma\s+GCC\s+optimize",
            r"#\s*pragma\s+GCC\s+target",
            r"#\s*pragma\s+GCC\s+push_options",
            r"#\s*pragma\s+GCC\s+ivdep",
            r"#\s*pragma\s+clang\s+optimize",
            r"#\s*pragma\s+clang\s+loop",
            r"__attribute__\s*\(\s*\(\s*optimize",
            r"__attribute__\s*\(\s*\(\s*target",
        ),
    ),
    "openmp": ForbiddenConstruct(
        description="OpenMP parallelism",
        patterns=(
            r"#\s*pragma\s+omp\b",
            r"#\s*include\s*[<\"]omp\.h[>\"]",
            r"\bomp_set_num_threads\b",
            r"\bomp_get_\w+\b",
        ),
    ),
    "threading": ForbiddenConstruct(
        description="multi-threading or multi-processing",
        patterns=(
            r"\bstd::thread\b",
            r"\bstd::async\b",
            r"\bpthread_create\b",
            r"\bThreadPoolExecutor\b",
            r"\bProcessPoolExecutor\b",
            r"\bmultiprocessing\b",
            r"\bthreading\.Thread\b",
        ),
    ),
    "blas": ForbiddenConstruct(
        description="BLAS/LAPACK or vendor math libraries",
        patterns=(
            r"\bcblas_\w+",
            r"#\s*include\s*[<\"]cblas\.h[>\"]",
            r"#\s*include\s*[<\"]lapacke?\.h[>\"]",
            r"Accelerate/Accelerate\.h",
            r"\bopenblas\b",
            r"\bmkl_\w+",
            r"\bcublas\w*",
        ),
    ),
    "simd_intrinsics": ForbiddenConstruct(
        description="hand-written SIMD intrinsics",
        patterns=(
            r"#\s*include\s*[<\"]immintrin\.h[>\"]",
            r"#\s*include\s*[<\"]x86intrin\.h[>\"]",
            r"#\s*include\s*[<\"]arm_neon\.h[>\"]",
            r"\b_mm\d*_\w+",
            r"\bv(?:ld|st)\dq?_\w+",
            r"\bvmlaq_\w+",
        ),
    ),
    "inline_asm": ForbiddenConstruct(
        description="inline assembly",
        patterns=(
            r"\b__asm__\b",
            r"\basm\s+volatile\b",
        ),
    ),
}


@dataclass(frozen=True)
class CompiledRule:
    """One compiled pattern plus the label reported when it fires."""

    label: str
    pattern: re.Pattern[str]


def validate_spec(constructs: list[str], patterns: list[str]) -> list[str]:
    """Return config errors for a task's forbidden_* settings (empty = valid)."""
    errors: list[str] = []
    for name in constructs:
        if name not in FORBIDDEN_CONSTRUCTS:
            known = ", ".join(sorted(FORBIDDEN_CONSTRUCTS))
            errors.append(
                f"unknown forbidden_construct {name!r} (known: {known})"
            )
    for pat in patterns:
        try:
            re.compile(pat)
        except re.error as exc:
            errors.append(f"invalid forbidden_pattern {pat!r}: {exc}")
    return errors


def compile_rules(constructs: list[str], patterns: list[str]) -> list[CompiledRule]:
    """Compile named constructs + custom patterns into checkable rules.

    Unknown names and bad regexes are skipped here -- validate_spec() reports
    them at task-load time, so silently dropping them keeps a config typo from
    crashing an in-flight run.
    """
    rules: list[CompiledRule] = []
    for name in constructs:
        construct = FORBIDDEN_CONSTRUCTS.get(name)
        if construct is None:
            continue
        for pat in construct.patterns:
            rules.append(CompiledRule(name, re.compile(pat, re.IGNORECASE)))
    for pat in patterns:
        try:
            rules.append(CompiledRule("custom", re.compile(pat, re.IGNORECASE)))
        except re.error:
            continue
    return rules


def check_text(text: str, rules: list[CompiledRule]) -> list[str]:
    """Return one message per distinct rule the text violates."""
    if not rules or not text:
        return []
    violations: list[str] = []
    seen: set[str] = set()
    for rule in rules:
        m = rule.pattern.search(text)
        if m is None:
            continue
        key = f"{rule.label}:{m.group(0)}"
        if key in seen:
            continue
        seen.add(key)
        if rule.label == "custom":
            violations.append(f"matches forbidden pattern (found {m.group(0)!r})")
        else:
            construct = FORBIDDEN_CONSTRUCTS[rule.label]
            violations.append(
                f"uses forbidden construct '{rule.label}' -- "
                f"{construct.description} (found {m.group(0)!r})"
            )
    return violations


def describe(constructs: list[str], patterns: list[str]) -> list[str]:
    """Human-readable lines for the prompt, so the model knows the rules."""
    lines: list[str] = []
    for name in constructs:
        construct = FORBIDDEN_CONSTRUCTS.get(name)
        if construct is not None:
            lines.append(f"{name}: {construct.description}")
    for pat in patterns:
        lines.append(f"custom pattern: {pat}")
    return lines
