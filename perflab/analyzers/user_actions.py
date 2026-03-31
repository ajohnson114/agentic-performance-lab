"""Extract build/compilation suggestions from LLM reasoning text.

The LLM sometimes suggests build flag changes (e.g., "compile with -fopenmp",
"add -ffast-math") that it cannot apply because task.yaml is a protected file.
This module extracts those suggestions so they can be surfaced to the user as
manual action items.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class UserAction:
    """A build/compilation suggestion the user needs to apply manually."""

    suggestion: str  # Human-readable description
    flag: str  # The specific flag, e.g. "-fopenmp"
    iteration: int  # Which agent iteration produced this
    source: str  # "llm_reasoning" or "build_flag_analysis"


# Compiler flags we look for in LLM reasoning text.
# Ordered longest-first so greedy matching picks the most specific flag.
_FLAG_PATTERNS: list[re.Pattern[str]] = [
    # GCC/Clang flags
    re.compile(r"(?<!\w)(-f(?:openmp|fast-math|lto|unroll-loops|PIC|pic|tree-vectorize|no-math-errno|finite-math-only|associative-math))(?!\w)"),
    re.compile(r"(?<!\w)(-march=\S+)"),
    re.compile(r"(?<!\w)(-mtune=\S+)"),
    re.compile(r"(?<!\w)(-mavx\S*)"),
    re.compile(r"(?<!\w)(-mfma)(?!\w)"),
    re.compile(r"(?<!\w)(-msse\S*)"),
    re.compile(r"(?<!\w)(-O[0-3s])(?!\w)"),
    re.compile(r"(?<!\w)(-std=c\+\+\d+)"),
    re.compile(r"(?<!\w)(-DNDEBUG)(?!\w)"),
    re.compile(r"(?<!\w)(-l(?:pthread|m|gomp|omp))(?!\w)"),
    # CUDA / nvcc flags
    re.compile(r"(?<!\w)(--use_fast_math)(?!\w)"),
    re.compile(r"(?<!\w)(-arch=sm_\d+)"),
    re.compile(r"(?<!\w)(--gpu-architecture=\S+)"),
    re.compile(r"(?<!\w)(-Xptxas\s+\S+)"),
    re.compile(r"(?<!\w)(--maxrregcount=\d+)"),
]

# Phrases that signal the LLM is talking about build changes.
_BUILD_PHRASES = re.compile(
    r"(?:compile\s+with|add\s+.*?(?:flag|to\s+.*?build)|build\s+command|task\.yaml|"
    r"compilation\s+flag|linker\s+flag|link\s+with|CMakeLists|Makefile|"
    r"compiler\s+flag|nvcc\s+flag|build\.cmd)",
    re.IGNORECASE,
)


def extract_build_suggestions(text: str, iteration: int = 0) -> list[UserAction]:
    """Extract build-related suggestions from LLM reasoning text.

    Scans each line for known compiler flags. When a flag is found in a line
    that also contains a build-related phrase (or the flag itself is strong
    enough signal), it's extracted as a UserAction.

    Returns deduplicated list of UserAction objects.
    """
    if not text:
        return []

    actions: list[UserAction] = []
    seen_flags: set[str] = set()

    for line in text.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        for pattern in _FLAG_PATTERNS:
            match = pattern.search(line_stripped)
            if match:
                flag = match.group(1).strip()
                if flag in seen_flags:
                    continue

                # Accept if: line has a build phrase, OR the flag is a linker/openmp
                # flag (strong signal on its own), OR the line explicitly says
                # "add"/"use"/"enable" near the flag.
                has_build_phrase = bool(_BUILD_PHRASES.search(line_stripped))
                has_action_verb = bool(re.search(
                    r"\b(?:add|use|enable|pass|include|set|switch to|compile with)\b",
                    line_stripped, re.IGNORECASE,
                ))

                if has_build_phrase or has_action_verb:
                    # Use the full line as the suggestion, trimmed
                    suggestion = line_stripped.lstrip("- *>•")
                    suggestion = suggestion.strip()
                    if len(suggestion) > 200:
                        suggestion = suggestion[:200] + "..."

                    seen_flags.add(flag)
                    actions.append(UserAction(
                        suggestion=suggestion,
                        flag=flag,
                        iteration=iteration,
                        source="llm_reasoning",
                    ))

    return actions
