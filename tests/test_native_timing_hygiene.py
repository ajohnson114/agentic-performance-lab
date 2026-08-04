"""Regression tests for two native-benchmark timing defects across
perflab/demo_tasks.

BUG 1 -- ``std::chrono::high_resolution_clock``
    Every native benchmark timed its work with ``high_resolution_clock``. That
    name promises nothing: on libstdc++ it is a plain ``typedef`` for
    ``system_clock``, i.e. the wall clock, which is NOT monotonic and is
    adjustable by NTP. A clock step mid-benchmark yields a wrong -- possibly
    negative -- interval. Verified empirically on gcc 13 / libstdc++:

        high_resolution_clock is system_clock : YES
        high_resolution_clock::is_steady      : FALSE

    (On Apple libc++ it happens to alias ``steady_clock``, which is exactly why
    the bug survived: it is invisible on the macOS dev box and only bites on
    the Linux/CUDA hosts where these tasks actually run.) ``steady_clock`` is
    guaranteed monotonic and is the correct clock for measuring a duration.

BUG 2 -- no compiler barrier in the timed region
    Nothing stopped the optimizer from hoisting, sinking, or deleting the work
    being timed. At -O2/-O3 a computation whose result is never read is dead
    code and may legally be removed outright, and a loop-invariant computation
    may be hoisted out of the repeat loop so that an empty loop gets timed.
    Measured on both clang and gcc 13 at -O3: an unobserved sum-of-squares loop
    compiles to ZERO floating-point instructions and the harness reports
    0.000000 ms. With the barrier it compiles to real vector FMAs and reports
    ~2-4 ms.

    The fix is a pair of file-local helpers (no Google Benchmark dependency)::

        template <class T>
        inline void do_not_optimize(const T& value) {
            asm volatile("" : : "r,m"(value) : "memory");
        }
        inline void clobber_memory() { asm volatile("" : : : "memory"); }

    The ``"r,m"`` multiple-alternative constraint is load-bearing: it lets the
    operand stay in a REGISTER when it fits in one and falls back to MEMORY
    only when it does not. A bare ``"m"`` would force a spill to the stack on
    every call and inflate the very measurement the barrier exists to protect.

These tests assert both fixes stay fixed. They are pure source-level checks, so
they cover the CUDA files too -- which cannot be compiled on a box without
nvcc, and which are otherwise only verified by inspection.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_TASKS = REPO_ROOT / "perflab" / "demo_tasks"

NATIVE_SUFFIXES = {".cpp", ".cu", ".cc", ".c", ".h", ".hpp", ".cuh"}

# The barrier helpers every native benchmark defines locally.
BARRIER_NAMES = ("do_not_optimize", "clobber_memory")

# Files that contain a timed region and therefore must carry a barrier. Derived
# dynamically below, but pinned here so that DELETING a benchmark's timing does
# not silently make these tests vacuous.
EXPECTED_TIMED_FILES = {
    "matmul/cpp/matmul.cpp",
    "matmul/cpp_parallel/matmul.cpp",
    "matmul/cuda/sgemm.cu",
    "matmul/cuda_h100/sgemm.cu",
    "matmul/cuda_tensorcore/hgemm.cu",
    "reduction/cpp_cuda/reduce.cu",
}


def _native_sources() -> list[Path]:
    """Every git-tracked native source under demo_tasks.

    ``out/`` holds run artifacts (logs, bench JSON, and potentially snapshots of
    candidate sources the optimizer produced). It is gitignored and is not
    source we own, so it is excluded.
    """
    return sorted(
        p
        for p in DEMO_TASKS.rglob("*")
        if p.suffix in NATIVE_SUFFIXES
        and p.is_file()
        and "out" not in p.relative_to(DEMO_TASKS).parts
    )


def _rel(path: Path) -> str:
    return path.relative_to(DEMO_TASKS).as_posix()


def test_native_sources_are_discovered():
    """Guard against the glob silently matching nothing."""
    found = _native_sources()
    assert found, f"no native sources found under {DEMO_TASKS}"
    assert {_rel(p) for p in found} >= EXPECTED_TIMED_FILES, (
        "a known timed benchmark disappeared from the glob; if a task was "
        "removed on purpose, update EXPECTED_TIMED_FILES"
    )


@pytest.mark.parametrize("src", _native_sources(), ids=_rel)
def test_no_high_resolution_clock(src: Path):
    """BUG 1: high_resolution_clock must never come back.

    On libstdc++ it is system_clock -- the non-monotonic wall clock.
    """
    text = src.read_text(encoding="utf-8")
    # Ignore the explanatory comments that name the banned clock on purpose.
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("//")
    )
    assert "high_resolution_clock" not in code, (
        f"{_rel(src)} uses std::chrono::high_resolution_clock. On libstdc++ "
        "that is a typedef for system_clock (wall clock, non-monotonic, "
        "NTP-adjustable). Use std::chrono::steady_clock to measure durations."
    )


@pytest.mark.parametrize("src", _native_sources(), ids=_rel)
def test_timed_regions_use_a_monotonic_clock(src: Path):
    """Any chrono-based timing must be steady_clock."""
    text = src.read_text(encoding="utf-8")
    clocks = set(re.findall(r"std::chrono::(\w+_clock)::now", text))
    disallowed = clocks - {"steady_clock"}
    assert not disallowed, (
        f"{_rel(src)} reads time from {sorted(disallowed)}; duration "
        "measurement must use std::chrono::steady_clock (monotonic)."
    )


@pytest.mark.parametrize(
    "rel", sorted(EXPECTED_TIMED_FILES), ids=lambda r: r
)
def test_timed_region_defines_barriers(rel: str):
    """BUG 2: each benchmark defines both barrier helpers locally.

    They are defined per-file rather than in a shared header on purpose: the
    wheel's package-data globs (pyproject.toml) ship only *.yaml/*.py/*.cpp/*.cu
    from demo_tasks, so a new .h would be dropped from the wheel and break these
    tasks for anyone who pip-installed perflab.
    """
    text = (DEMO_TASKS / rel).read_text(encoding="utf-8")
    for name in BARRIER_NAMES:
        assert f"inline void {name}" in text, (
            f"{rel} does not define {name}(). Without it the optimizer may "
            "delete the timed work entirely (measured: 0.000000 ms at -O3)."
        )


@pytest.mark.parametrize(
    "rel", sorted(EXPECTED_TIMED_FILES), ids=lambda r: r
)
def test_barrier_uses_register_or_memory_constraint(rel: str):
    """The "r,m" constraint must not degrade to a bare "m".

    A memory-only constraint forces the operand to be spilled to the stack on
    every call, adding per-iteration cost to the measurement it is meant to
    protect. Register-first is what keeps the barrier free.
    """
    text = (DEMO_TASKS / rel).read_text(encoding="utf-8")
    assert 'asm volatile("" : : "r,m"(value) : "memory")' in text, (
        f'{rel} must use the register-or-memory constraint "r,m" in '
        "do_not_optimize; a bare \"m\" would force a spill and inflate timings."
    )
    assert 'asm volatile("" : : : "memory")' in text, (
        f"{rel} must define clobber_memory() with a full memory clobber so "
        "stores are committed before the clock is read."
    )


@pytest.mark.parametrize(
    "rel", sorted(EXPECTED_TIMED_FILES), ids=lambda r: r
)
def test_timed_region_is_bracketed_by_barriers(rel: str):
    """The barriers must actually be invoked inside each timed region.

    Defining the helpers is not enough -- they have to be called between the two
    clock reads, otherwise the timed work is still eligible for elimination.
    """
    text = (DEMO_TASKS / rel).read_text(encoding="utf-8")
    lines = text.splitlines()

    # Locate the timed region: chrono benchmarks bracket it with steady_clock
    # reads; reduce.cu uses clock_gettime(CLOCK_MONOTONIC, ...).
    marks = [
        i
        for i, ln in enumerate(lines)
        if "steady_clock::now()" in ln or "clock_gettime(CLOCK_MONOTONIC" in ln
    ]
    assert len(marks) >= 2, f"{rel}: could not locate a timed region"

    start, end = marks[0], marks[-1]
    body = "\n".join(lines[start : end + 1])
    for name in BARRIER_NAMES:
        assert f"{name}(" in body, (
            f"{rel}: {name}() is defined but never called between the clock "
            f"reads (lines {start + 1}-{end + 1}). The timed work is still "
            "eligible for dead-code elimination."
        )


@pytest.mark.parametrize(
    "rel",
    sorted(r for r in EXPECTED_TIMED_FILES if r.endswith(".cu")),
    ids=lambda r: r,
)
def test_cuda_kernel_launch_is_synchronized_inside_timed_region(rel: str):
    """A host clock around an async kernel launch measures launch overhead only.

    ``kernel<<<...>>>`` returns immediately, so without a device sync BEFORE the
    second clock read the interval is meaningless. All four CUDA benchmarks
    already did this correctly; this test keeps it that way.
    """
    text = (DEMO_TASKS / rel).read_text(encoding="utf-8")
    lines = text.splitlines()

    marks = [
        i
        for i, ln in enumerate(lines)
        if "steady_clock::now()" in ln or "clock_gettime(CLOCK_MONOTONIC" in ln
    ]
    start, end = marks[0], marks[-1]
    body_lines = lines[start : end + 1]
    body = "\n".join(body_lines)

    launch_idx = [i for i, ln in enumerate(body_lines) if "<<<" in ln]
    if not launch_idx:
        pytest.skip(f"{rel}: no kernel launch inside the timed region")

    assert "cudaDeviceSynchronize()" in body, (
        f"{rel}: a kernel is launched inside the timed region but the device "
        "is never synchronized before the closing clock read -- the measured "
        "interval would be launch overhead, not kernel time."
    )

    sync_idx = [i for i, ln in enumerate(body_lines) if "cudaDeviceSynchronize()" in ln]
    assert max(sync_idx) > min(launch_idx), (
        f"{rel}: cudaDeviceSynchronize() must come AFTER the kernel launch "
        "and before the closing clock read."
    )
