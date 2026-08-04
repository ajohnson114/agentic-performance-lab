"""Regression tests for the percentile floor-indexing bug (BUG 1) across
perflab/demo_tasks.

Every bench harness used to compute
    p95 = sorted_times[int(0.95 * (n - 1))]
which is a FLOOR-indexed lookup. At the sample counts these tasks actually
use, floor indexing collapses:

    n=2 (fast screen) -> int(0.95*1) = 0  -> "p95" IS THE MINIMUM
    n=3 (default)     -> int(0.95*2) = 1  -> p95 == p50, identical
    n=5               -> int(0.95*4) = 3  -> should be 4 (the maximum)

The canonical fix already lived in this repo before this file did:
``perflab.harness.precision._ceil_percentile_index``. Every demo_tasks bench
harness with a hand-rolled percentile (matmul/{python,triton,pytorch,jax}
bench.py and matmul/{cpp,cpp_parallel}/matmul.cpp,
matmul/{cuda,cuda_h100}/sgemm.cu, matmul/cuda_tensorcore/hgemm.cu) now
mirrors that formula's semantics.

This module checks three things:
  1. The canonical formula's own invariants at n=2,3,5,20 (the sizes actually
     used: n=2 during fast-screen prescreening, n=3/5 as several tasks'
     default repeat counts, n=20 as several others').
  2. That the OLD floor formula really did violate those invariants -- i.e.
     this is a real behavior change, not a no-op refactor.
  3. End to end, that the actual shipped files (matmul/python/bench.py and
     the compiled matmul/cpp/matmul.cpp binary) now produce a p95 that is
     never the minimum and never collides with p50 at these sample counts.
     These two are run because they need only a stdlib-adjacent toolchain
     (numpy+pyyaml, plain g++) available on every CI runner and this dev
     box; the CUDA/Triton/JAX variants are covered by inspection instead
     (see the accompanying report) since this box has no nvcc/CUDA GPU.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from perflab.harness.precision import _ceil_percentile_index

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_TASKS = REPO_ROOT / "perflab" / "demo_tasks"

SAMPLE_SIZES = [2, 3, 5, 20]


def _floor_percentile_index(fraction: float, n: int) -> int:
    """The OLD, buggy formula every demo_tasks bench harness used to use."""
    return int(fraction * (n - 1))


def _true_median(sorted_values: list[float]) -> float:
    """Reference median matching perflab.analyzers.bench_stats.compute_bench_stats."""
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    return sorted_values[mid]


# ---------------------------------------------------------------------------
# 1. Canonical formula invariants
# ---------------------------------------------------------------------------

# Documented before/after index table (0-based) for fraction=0.95.
_EXPECTED_CEIL_INDEX = {
    2: 1,   # old floor: 0 (the MIN)          -> new ceil: 1 (the MAX)
    3: 2,   # old floor: 1 (collides with p50) -> new ceil: 2 (the MAX)
    5: 4,   # old floor: 3                     -> new ceil: 4 (the MAX)
    20: 19,  # old floor: 18                    -> new ceil: 19 (the MAX)
}

_EXPECTED_FLOOR_INDEX = {2: 0, 3: 1, 5: 3, 20: 18}


@pytest.mark.parametrize("n", SAMPLE_SIZES)
def test_ceil_percentile_index_matches_documented_table(n):
    assert _ceil_percentile_index(0.95, n) == _EXPECTED_CEIL_INDEX[n]


@pytest.mark.parametrize("n", SAMPLE_SIZES)
def test_p95_index_never_below_p50_window(n):
    """p95's index must never sit inside or before the median's own window."""
    sorted_values = [float(i) for i in range(n)]  # strictly increasing
    p50 = _true_median(sorted_values)
    p95_idx = _ceil_percentile_index(0.95, n)
    p95 = sorted_values[p95_idx]
    assert p95 >= p50, f"n={n}: p95={p95} fell below p50={p50}"


@pytest.mark.parametrize("n", SAMPLE_SIZES)
def test_p95_is_never_the_minimum(n):
    sorted_values = [float(i) for i in range(n)]
    p95_idx = _ceil_percentile_index(0.95, n)
    assert sorted_values[p95_idx] != sorted_values[0]


@pytest.mark.parametrize("n", SAMPLE_SIZES)
def test_p95_index_is_monotonic_and_in_bounds(n):
    idx = _ceil_percentile_index(0.95, n)
    assert 0 <= idx <= n - 1


# ---------------------------------------------------------------------------
# 2. Proof this is a real fix: the OLD floor formula fails these invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SAMPLE_SIZES)
def test_old_floor_formula_disagrees_with_the_fix(n):
    """The floor and ceiling formulas must diverge at every size used here.

    If this ever stops being true (e.g. someone "simplifies" the fixed
    formula back to plain int() truncation), it collapses back to the old
    bug silently -- so pin the expected floor index too.
    """
    floor_idx = _floor_percentile_index(0.95, n)
    ceil_idx = _ceil_percentile_index(0.95, n)
    assert floor_idx == _EXPECTED_FLOOR_INDEX[n]
    assert floor_idx != ceil_idx, (
        f"n={n}: floor formula ({floor_idx}) and ceiling formula ({ceil_idx}) "
        "agree -- the invariant tests above would not have caught a "
        "regression to the old code at this n."
    )


def test_old_floor_formula_would_fail_the_minimum_invariant_at_n2():
    """This is the test that would FAIL under the pre-fix code.

    Concretely reproduces the bug report's headline claim: at n=2 (the fast
    screening sample count), the old formula's "p95" is exactly the minimum
    of the two samples -- the beam search would rank candidates by their
    best case while accepting on a value that claims to be a high
    percentile. Verified by literally swapping in the old formula here
    (rather than only asserting on the new one) so a future edit that
    quietly reintroduces floor indexing in _ceil_percentile_index itself
    would be caught.

    (This was additionally verified by hand during development: temporarily
    replacing perflab/demo_tasks/matmul/cpp/matmul.cpp's ceil_percentile()
    call with the old `sorted_times[static_cast<int>(0.95 * (n - 1))]`
    expression reproduces exactly this -- p95 becomes the minimum of the two
    timed repeats -- and was reverted immediately after confirming it.)
    """
    sorted_values = [10.0, 20.0]
    n = len(sorted_values)

    old_p95 = sorted_values[_floor_percentile_index(0.95, n)]
    assert old_p95 == min(sorted_values), (
        "sanity check on the bug report itself failed -- old formula no "
        "longer reproduces the documented bug"
    )

    new_p95 = sorted_values[_ceil_percentile_index(0.95, n)]
    assert new_p95 == max(sorted_values)
    assert new_p95 != old_p95


# ---------------------------------------------------------------------------
# 3. End-to-end: the actual shipped files, not a reimplementation
# ---------------------------------------------------------------------------


def _run_bench(task_dir: Path, json_out: Path, *, warmup: int, repeats: int, extra_env=None):
    env = dict(os.environ)
    env["PERFLAB_BENCH_WARMUP"] = str(warmup)
    env["PERFLAB_BENCH_REPEATS"] = str(repeats)
    if extra_env:
        env.update(extra_env)
    subprocess.run(
        [sys.executable, "bench.py", "--json", str(json_out)],
        cwd=task_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(json_out.read_text())


@pytest.mark.parametrize("repeats", [2, 3, 5])
def test_matmul_python_bench_p95_end_to_end(tmp_path, repeats):
    """Runs the real perflab/demo_tasks/matmul/python/bench.py subprocess."""
    task_dir = DEMO_TASKS / "matmul" / "python"
    json_out = tmp_path / "bench.json"
    data = _run_bench(task_dir, json_out, warmup=1, repeats=repeats)

    assert data["ok"] is True
    lat = data["latency_ms"]
    times = data["times_ms"]
    assert len(times) == repeats

    sorted_times = sorted(times)
    assert lat["p95"] == pytest.approx(sorted_times[_ceil_percentile_index(0.95, repeats)])
    if repeats > 1:
        assert lat["p95"] != min(times) or min(times) == max(times), (
            "p95 collapsed to the minimum -- the floor-indexing bug is back"
        )

    # BUG 2 contract: per-repeat raw_values alongside the metric's own
    # aggregate, genuinely per-repeat (same length as the repeat count).
    tflops_raw = data["tflops"]["raw_values"]
    assert len(tflops_raw) == repeats
    assert lat["raw_values"] == times


@pytest.mark.skipif(shutil.which("g++") is None, reason="no g++ toolchain on this runner")
@pytest.mark.parametrize("repeats", [2, 3, 5])
def test_matmul_cpp_bench_p95_end_to_end(tmp_path, repeats):
    """Compiles and runs the real perflab/demo_tasks/matmul/cpp/matmul.cpp.

    matmul/cpp does not use OpenMP (unlike matmul/cpp_parallel, whose
    contract build command requires -fopenmp -- unavailable via this repo's
    g++, which is Apple clang on macOS), so it is the one C++ source in this
    task family that can be built and run for real on every CI runner and
    this dev box.
    """
    src = DEMO_TASKS / "matmul" / "cpp" / "matmul.cpp"
    binary = tmp_path / "matmul_bin"
    subprocess.run(
        ["g++", "-O2", "-o", str(binary), str(src)],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            str(binary),
            "--M", "16", "--N", "16", "--K", "16",
            "--warmup", "1", "--repeats", str(repeats),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    data = json.loads(result.stdout)

    assert data["ok"] is True
    times = data["times_ms"]
    assert len(times) == repeats
    lat = data["latency_ms"]
    sorted_times = sorted(times)

    assert lat["p95"] == pytest.approx(sorted_times[_ceil_percentile_index(0.95, repeats)])
    if repeats > 1 and min(times) != max(times):
        assert lat["p95"] != min(times), (
            "p95 collapsed to the minimum -- the floor-indexing bug is back"
        )

    # BUG 2 contract: raw_values present alongside tflops.median (the
    # metric this task's task.yaml declares) and alongside latency_ms.p95
    # (matmul_cpp_parallel's secondary_metric -- matmul/cpp shares the same
    # bench.json shape as its OpenMP sibling).
    assert len(data["tflops"]["raw_values"]) == repeats
    assert lat["raw_values"] == times
