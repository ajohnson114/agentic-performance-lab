"""Correctness + anti-gaming gate for the pure-Python matmul.

Protected file: tests.py is NOT in task.yaml's edit_policy.allowed_paths, so a
candidate cannot weaken these checks, and the workspace is hash-verified after
every iteration.

Dependencies are numpy only — no torch. This is the CPU-only laptop demo, so it
must run under `pip install perflab[tasks-python]` with nothing else installed,
which is also what CI's real-task job uses. The shipped `perflab.harness`
helpers are backend-neutral: they dispatch on the value's own type (see
perflab/harness/_array.py), so they check numpy arrays and plain nested lists
exactly as they check torch tensors, and a type they cannot inspect raises
instead of quietly passing. So the anti-gaming gates below are the shipped
library, not a local re-implementation of it.

The gates, in order:

  0. random_matrix() honors the requested shape — hand-rolled. It lives in the
     editable matmul.py, so the input generator is part of the attack surface,
     and the expected shape is task knowledge no library helper has.
  1. Accuracy against an fp64 numpy reference — hand-rolled, for the bound
     (see ATOL). The harness's assert_ulp_close does reject this task's float32
     cheat (measured: p99 of 0 ULP for the fp64 kernels, far past its hard
     ceiling for float32), but it is not the check this task means. It states a
     p99 ULP bound over a random ~1000-element sample, whereas ATOL is an
     absolute bound over every element that has to equal bench.py's
     MAX_ABS_ERR, so the timed path and the correctness path agree on what
     "correct" is. It also rounds *both* operands to float32 before measuring
     (harness/_array.py: flat_float32_list, which is what lets a legitimately
     fp32 kernel be scored against an fp64 reference at all), so on an all-fp64
     task it is really asking "do these agree to within half a float32 ULP" —
     a differently shaped bound that happens to land on the right side of this
     particular cheat. The output does go through the harness's
     assert_real_array first, which rejects a lazy proxy standing in for a
     materialized array.
  2. Determinism: the same inputs must produce the same answer — harness
     assert_deterministic, its Phase 1.
  3. No-op guard: different inputs must produce a different answer — the same
     call's Phase 2. It compares two separate input_factory() draws and
     correctly declines to flag anything when those draws are identical, so
     make_inputs() below must vary on every call or this gate is inert.
  4. Memoization guard: overwrite the input contents in place and re-run —
     harness assert_no_memoization, which poisons every mutable input (numpy
     buffer or nested list) while preserving object identity.

Every gate passes for the optimization this task exists to demonstrate —
replacing the triple loop with numpy — and fails for the usual ways of gaming
a benchmark: caching across calls, returning a stale buffer, no-op kernels,
shrinking the problem, and silently dropping to float32.

Deliberately absent: a thread guard. numpy dispatches into a multi-threaded
BLAS, so it would reject the intended solution — and thread injection is not a
viable cheat here anyway, since a correct result is required by the time
matmul() returns.

The harness helpers are silent on success and raise on failure, so the per-gate
log line — the most legible artifact of a run — is printed here by gate_ok().
Checks raise AssertionError explicitly rather than using `assert`, which
`python -O` strips.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
from matmul import matmul, random_matrix

from perflab.harness import (
    assert_deterministic,
    assert_no_memoization,
    assert_real_array,
)

# Small enough that the naive triple loop's passes land well inside the 60 s
# correctness timeout; large enough (4096 output elements) to be a real check
# rather than a smoke test.
M = N = K = 64

# Slack for the fp64 accumulation-order difference between a Python
# left-to-right sum and BLAS's blocked/SIMD reassociation (~1e-15 at K=64),
# while staying orders of magnitude tighter than the ~5e-7 error a float32
# kernel shows at this size. Mirrors bench.py's MAX_ABS_ERR. Passed explicitly
# to every harness helper (with rtol=0) so the bound stays absolute and
# task-chosen rather than inheriting a default.
ATOL = 1e-9

# The framework re-runs correctness with PERFLAB_DETERMINISM_SEED set
# (anti_gaming.determinism_rerun). Fold it into every seed so the second run
# genuinely uses different data — otherwise that check is inert.
SEED_OFFSET = int(os.environ.get("PERFLAB_DETERMINISM_SEED", "0"))

# Advanced on every make_inputs() call; see the docstring there.
_draw = itertools.count()


def as_array(mat) -> np.ndarray:
    """Kernel output -> ndarray.

    Accepts list-of-lists or ndarray so a numpy rewrite of matmul() may return
    either. Copies, so a kernel that hands back a buffer it later overwrites
    cannot make two captured results silently alias.
    """
    return np.array(mat)


def reference(A, B) -> np.ndarray:
    """fp64 numpy reference. Also the reference_fn for both harness gates."""
    return np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64)


def make_inputs() -> tuple:
    """A fresh (A, B) pair — genuinely different data on EVERY call.

    This is the input_factory for both harness gates, and its variation is
    load-bearing: assert_deterministic's no-op detector (gate 3) compares the
    outputs of two separate input_factory() calls and, correctly, refuses to
    flag identical outputs when the two draws were themselves identical. A
    fixed seed here would silently disarm it. PERFLAB_DETERMINISM_SEED is
    folded in so the framework's determinism re-run uses different data too.
    """
    draw = next(_draw)
    return (
        random_matrix(M, K, seed=SEED_OFFSET + 1000 * draw + 42),
        random_matrix(K, N, seed=SEED_OFFSET + 1000 * draw + 123),
    )


def gate_ok(label: str, detail: str = "") -> None:
    """Print the one line this gate contributes to the correctness log."""
    suffix = f"   ({detail})" if detail else ""
    print(f"[gate] {label:<28}ok{suffix}")


def main():
    A, B = make_inputs()

    # --- Gate 0: the input generator must honor the requested shape ---------
    # bench.py reports the shapes it actually multiplied and
    # contract.fixed_params rejects a shrunk problem, but failing here first
    # gives a much clearer message than a contract violation.
    if as_array(A).shape != (M, K) or as_array(B).shape != (K, N):
        raise AssertionError(
            f"random_matrix ignored the requested shape: got "
            f"{as_array(A).shape} and {as_array(B).shape}, expected "
            f"{(M, K)} and {(K, N)}"
        )
    gate_ok("input shapes honored")

    # --- Gate 1: accuracy against an fp64 numpy reference -------------------
    C = matmul(A, B)
    # Materialized output, not a proxy that defers the work until compared.
    assert_real_array(C, "matmul output")
    out = as_array(C)
    ref = reference(A, B)
    if out.shape != ref.shape:
        raise AssertionError(f"matmul returned shape {out.shape}, expected {ref.shape}")
    if not np.isfinite(out).all():
        raise AssertionError("matmul returned non-finite values")
    err = float(np.abs(out.astype(np.float64) - ref).max())
    if not err < ATOL:
        raise AssertionError(
            f"accuracy: max_abs={err:.3e} >= {ATOL:g}. Reassociation by a "
            f"vectorized fp64 kernel lands ~1e-15; a float32 kernel lands ~5e-7."
        )
    gate_ok("accuracy vs fp64 numpy", f"max_abs={err:.3e}")

    # --- Gates 2 and 3: one harness call, two guarantees --------------------
    # Phase 1 re-runs the kernel on identical inputs and requires an identical
    # answer; Phase 2 draws two different input sets and requires the answers
    # to differ; Phase 3 re-checks against the reference. Phase 2 stands down
    # when the two draws are identical, so prove make_inputs() actually varies
    # first — a random_matrix() that ignored its seed would otherwise disarm it.
    if np.array_equal(as_array(make_inputs()[0]), as_array(make_inputs()[0])):
        raise AssertionError(
            "random_matrix ignores its seed: two draws with different seeds "
            "returned identical data. That would disarm the no-op guard below, "
            "which needs two genuinely different input sets to compare."
        )
    assert_deterministic(
        fn=matmul,
        input_factory=make_inputs,
        reference_fn=reference,
        n_runs=3,
        atol=ATOL,
        rtol=0.0,
    )
    gate_ok("determinism (same inputs)", "harness, 3 runs")
    gate_ok("no-op guard (new inputs)", "harness, 2 distinct draws")

    # --- Gate 4: memoization guard -----------------------------------------
    # bench.py builds A and B once and passes those same objects to matmul()
    # on every repeat, so a cache keyed on id(A)/id(B) — or a "compute once,
    # return the stored C" shortcut — would make repeats 2..N free and collapse
    # the measured time. The harness overwrites the input contents in place
    # (same objects, same ids, same numpy buffer addresses) and requires the
    # answer to track the new data.
    assert_no_memoization(
        fn=matmul,
        input_factory=make_inputs,
        reference_fn=reference,
        atol=ATOL,
        rtol=0.0,
        n_rounds=2,
    )
    gate_ok("memoization guard", "harness, 2 poison rounds")

    print("ok", {"M": M, "N": N, "K": K})


if __name__ == "__main__":
    main()
