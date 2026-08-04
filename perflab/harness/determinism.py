"""Mitigation #4: Output Determinism Check (No-Op / Buffer Reuse Guard).

Prevents LLM-generated code from:
  - Launching a no-op kernel that returns stale buffer contents from prior runs
  - Exploiting shared memory overflow to read uninitialized but "lucky" garbage
  - Relying on uninitialized output buffers that happen to contain correct values

The fix: run the kernel multiple times with identical inputs and verify outputs
match exactly. Also zero the output buffer before each run to catch buffer
reuse. Run with different inputs to catch no-ops.

Backend-neutral: outputs may be torch tensors, numpy arrays, JAX arrays,
nested Python lists/tuples, or plain numbers. All three phases run for every
one of them. An output type the harness cannot inspect raises -- earlier
versions guarded each phase with ``isinstance(out, torch.Tensor)``, so a numpy
or list output made Phases 2 and 3 *pass while checking nothing*, which is the
one failure mode an anti-gaming check must never have.

Vacuity is a failure, not a pass. Every way this check can end up comparing
nothing is either an error or (where an error would be too strict) a warning:

  * a kernel output with zero elements, or one containing NaN -- NaN is not
    equal to itself, so it makes every comparison here meaningless;
  * ``n_runs < 1`` (the kernel never runs) is an error, ``n_runs == 1`` (Phase
    1 has a single output and therefore nothing to compare it against) warns;
  * an ``input_factory`` that hands back the same data on consecutive calls,
    which disarms the no-op phase -- see Phase 2 below;
  * a kernel that returns the same buffer object every call, which would make
    two results compare equal to each other because they *are* each other.
    Every captured result is snapshotted the instant it is produced, before
    anything else gets a chance to overwrite it.

Tolerance split: the same-inputs reproducibility check (Phase 1) runs the SAME
binary twice and asks whether it produces the SAME bits -- that is not an
accuracy question, so its default tolerance is a fixed strict 1e-5, never the
task-declared PERFLAB_ACCURACY_TOLERANCE. Otherwise a task that legitimately
loosens tolerance for e.g. fp16 math (say 1e-2) would let run-to-run garbage
from uninitialized buffers slip past the very check built to catch it. The
optional reference_fn comparison (Phase 3) is a genuine accuracy question --
kernel output vs a reference implementation -- so its default DOES track the
task tolerance. An explicit atol/rtol passed by the caller (a task author's
deliberate choice in tests.py) always wins for both.

Usage in tests.py:
    from perflab.harness.determinism import assert_deterministic

    assert_deterministic(
        fn=lambda A, B: kernel(A, B),
        input_factory=lambda: (torch.randn(M, K, device=dev), torch.randn(K, N, device=dev)),
        reference_fn=lambda A, B: A @ B,
        n_runs=3,
        atol=1e-5,
    )
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

from perflab.harness import _array

#: Extra independent draws before concluding a kernel is a no-op. A real no-op
#: collides on every draw, so retries cost a true positive nothing; an honest
#: low-cardinality kernel (booleans, reductions) escapes an accidental collision.
_NOOP_RETRIES = 4


def _inputs_differ(first: tuple, second: tuple) -> bool:
    """True if any positional input differs between two draws."""
    return any(
        not _array.values_equal(a, b)
        for a, b in zip(first, second, strict=True)
    )


def assert_deterministic(
    fn: Callable[..., Any],
    input_factory: Callable[[], tuple],
    reference_fn: Callable[..., Any] | None = None,
    n_runs: int = 3,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    """Verify kernel produces deterministic, correct output across multiple runs.

    Args:
        fn: The kernel/function to test. Called as fn(*inputs).
        input_factory: Callable that returns a tuple of fresh input arrays. It
            must return *varying* data: the no-op phase compares two calls'
            outputs and can only conclude anything if the two inputs differ.
        reference_fn: Optional reference implementation for correctness check.
        n_runs: Number of times to run with identical inputs (default 3).
        atol: Absolute tolerance for comparison. When passed explicitly it wins
            for every comparison. When left as None the default is split by
            purpose: the same-inputs reproducibility check (Phase 1) uses a
            fixed strict 1e-5, while the reference comparison (Phase 3) uses the
            task's declared accuracy tolerance (PERFLAB_ACCURACY_TOLERANCE, or
            1e-5). See the module docstring for the rationale.
        rtol: Relative tolerance for comparison (same default resolution).

    Raises:
        AssertionError if outputs differ across runs, diverge from reference,
        have a type the harness cannot compare, or are so degenerate (empty,
        NaN, or produced from non-varying inputs) that the check would pass
        without having verified anything.
    """
    from perflab.harness.tolerance import env_accuracy_tolerance

    if n_runs < 1:
        raise ValueError(
            f"assert_deterministic: n_runs={n_runs} never runs the kernel, so "
            f"the reproducibility check would pass without executing anything. "
            f"Use n_runs >= 2."
        )
    if n_runs == 1:
        warnings.warn(
            "assert_deterministic: n_runs=1 produces a single output, so the "
            "same-inputs reproducibility check (Phase 1) has nothing to compare "
            "it against and cannot detect run-to-run garbage. Use n_runs >= 2.",
            stacklevel=2,
        )

    # Reproducibility (same binary, same inputs): a fixed strict default that
    # never inherits the loose task tolerance. Explicit caller args still win.
    det_atol = atol if atol is not None else 1e-5
    det_rtol = rtol if rtol is not None else 1e-5
    # Reference correctness (kernel vs reference impl): a genuine accuracy
    # question, so its default tracks the task-declared tolerance.
    ref_atol = atol if atol is not None else env_accuracy_tolerance(1e-5)
    ref_rtol = rtol if rtol is not None else env_accuracy_tolerance(1e-5)

    # --- Phase 1: Same inputs, multiple runs → outputs must be identical ---
    inputs = _require_inputs(input_factory)
    outputs = []
    for _run_idx in range(n_runs):
        result = fn(*inputs)
        # Raises for a type we cannot compare, rather than collecting it and
        # quietly comparing nothing.
        _array.require_backend(result, "kernel output")
        # Detach and copy to prevent aliasing: a kernel handing back a buffer
        # it overwrites on the next call must not make two runs look equal.
        outputs.append(_array.detached_copy(result, "kernel output"))

    # One output is enough to reject the degenerate shapes: every later run has
    # to match this one within tolerance, and neither an empty array nor a NaN
    # can satisfy that silently.
    _reject_vacuous_output(outputs[0], "kernel output")

    # Compare all runs against the first
    for i in range(1, len(outputs)):
        ok, max_diff = _array.allclose(
            outputs[0],
            outputs[i],
            atol=det_atol,
            rtol=det_rtol,
            name_a="run 0 output",
            name_b=f"run {i} output",
        )
        if not ok:
            raise AssertionError(
                f"Non-deterministic output: run 0 vs run {i} differ "
                f"(max_diff={max_diff:.2e}, atol={det_atol}, rtol={det_rtol}). "
                f"Kernel may be reading uninitialized memory or shared "
                f"memory overflow."
            )

    # --- Phase 2: Different inputs → outputs must change (catches no-ops) ---
    inputs_a = _require_inputs(input_factory)
    inputs_b = _require_inputs(input_factory)
    if len(inputs_a) != len(inputs_b):
        raise AssertionError(
            f"input_factory returned {len(inputs_a)} inputs on one call and "
            f"{len(inputs_b)} on the next; the no-op check cannot pair them up."
        )

    # Compare the inputs BEFORE running the kernel. Doing it afterwards (as an
    # earlier version did) let a kernel that overwrites its own inputs in place
    # -- e.g. `def fn(a, b): a[:] = 0; b[:] = 0; return CONST` -- leave the two
    # input sets looking identical, which silently disarmed this entire phase
    # and passed the no-op kernel it exists to catch.
    inputs_differ = _inputs_differ(inputs_a, inputs_b)

    out_a = fn(*inputs_a)
    _array.require_backend(out_a, "kernel output")
    # Snapshot before the second call, for the same reason Phase 1 does: an
    # honest kernel writing into a preallocated output buffer returns the same
    # object twice, and comparing that object against itself would report every
    # such kernel as a no-op.
    out_a = _array.detached_copy(out_a, "kernel output")
    out_b = fn(*inputs_b)
    _array.require_backend(out_b, "kernel output")
    outputs_identical = _array.exact_equal(out_a, out_b, "output A", "output B")

    if not inputs_differ:
        # The factory handed back the same data twice -- a constant factory, or
        # one that reseeds its RNG identically on every call. Nothing here
        # distinguishes a real kernel from one that ignores its arguments.
        if outputs_identical and reference_fn is None:
            raise AssertionError(
                "No-op check cannot run: input_factory returned identical "
                "inputs on consecutive calls, so 'different inputs must change "
                "the output' has nothing to compare. A factory that reseeds its "
                "RNG with a fixed seed on every call looks random but is not. "
                "Make input_factory return fresh varying data (or pass a "
                "reference_fn, which checks correctness independently)."
            )
        # Otherwise something else in this call still carries evidence: either
        # the outputs demonstrably changed (so the kernel is not returning one
        # fixed constant) or Phase 3 will compare against a reference, which a
        # constant-returning kernel cannot satisfy. Weaker than the real check,
        # so say so rather than passing quietly.
        warnings.warn(
            "assert_deterministic: input_factory returned identical inputs on "
            "consecutive calls, so the no-op phase could not vary them and is "
            "not protecting this task. Make input_factory return fresh varying "
            "data.",
            stacklevel=2,
        )
    elif outputs_identical:
        # "Different inputs collided on the same output" is not proof of a
        # no-op: at low cardinality an honest kernel does that legitimately. An
        # 8-element boolean output has only 256 possible values, so two random
        # draws land on the same one often -- and a reduction over booleans
        # (any/all) collides far more often still. Failing on the first
        # collision made this check reject correct kernels ~2.5% of the time,
        # which is the worst kind of anti-gaming bug: it accuses honest work.
        #
        # A genuine no-op is identical on EVERY draw, so retrying costs a true
        # positive nothing while making an accidental collision vanishingly
        # unlikely. Only conclude no-op if every attempt collides.
        for _ in range(_NOOP_RETRIES):
            retry_inputs = _require_inputs(input_factory)
            if not _inputs_differ(inputs_a, retry_inputs):
                continue
            retry_out = _array.detached_copy(fn(*retry_inputs), "kernel output")
            if not _array.exact_equal(out_a, retry_out, "output A", "output B"):
                break
        else:
            raise AssertionError(
                "No-op kernel detected: different inputs produced identical "
                f"outputs on {_NOOP_RETRIES + 1} independent draws. The kernel "
                "may be copying inputs to output without computation, or "
                "returning a stale buffer."
            )

    # --- Phase 3: Reference check (if provided) ---
    if reference_fn is not None:
        inputs_check = _require_inputs(input_factory)
        actual = fn(*inputs_check)
        _array.require_backend(actual, "kernel output")
        # Snapshot before the reference runs: if kernel and reference share one
        # scratch buffer, comparing them afterwards compares that buffer with
        # itself and passes no matter what the kernel computed.
        actual = _array.detached_copy(actual, "kernel output")
        expected = reference_fn(*inputs_check)
        _array.require_backend(expected, "reference output")
        _reject_vacuous_output(actual, "kernel output")
        ok, max_diff = _array.allclose(
            actual,
            expected,
            atol=ref_atol,
            rtol=ref_rtol,
            name_a="kernel output",
            name_b="reference output",
        )
        if not ok:
            raise AssertionError(
                f"Correctness failure against reference: max_diff={max_diff:.2e} "
                f"(atol={ref_atol}, rtol={ref_rtol})"
            )


def _require_inputs(
    input_factory: Callable[[], tuple], check_name: str = "assert_deterministic"
) -> tuple:
    """Call ``input_factory`` and validate what it returned.

    Shared with `perflab.harness.pointer_poison`: both checks drive the kernel
    through the same ``fn(*input_factory())`` contract and need the same guard.

    A factory that returns a bare array rather than a tuple of them gets
    splatted element-wise into the kernel (``fn(*array)``), which fails later
    with an unrelated "truth value of an array is ambiguous"; catching it here
    names the actual mistake.
    """
    from perflab.harness import _array

    inputs = input_factory()
    if isinstance(inputs, list):
        inputs = tuple(inputs)
    if not isinstance(inputs, tuple):
        raise _array.UnsupportedValueError(
            f"{check_name}: input_factory must return a tuple of inputs (one "
            f"per kernel argument), but returned a {type(inputs).__name__}. "
            f"Wrap a single input in a 1-tuple: `lambda: (x,)`."
        )
    if not inputs:
        raise ValueError(
            f"{check_name}: input_factory returned no inputs, so the "
            f"no-op check cannot tell a real kernel from one that ignores its "
            f"arguments. Give input_factory at least one varying input."
        )
    return inputs


def _reject_vacuous_output(value: Any, name: str) -> None:
    """Reject outputs that make every comparison in this check meaningless.

    Two shapes of vacuity, both of which otherwise read as success:

    * **Zero elements.** ``allclose`` over an empty array is vacuously true on
      every backend, so an empty output passes each phase without a single
      value ever being compared.
    * **NaN.** NaN is the one value that is not equal to itself, so it turns
      every tolerance and equality comparison here into a coin flip: Phase 1
      reports a spurious "non-deterministic", while the equality-based no-op
      probe in Phase 2 sees two NaN outputs as *different* and clears a kernel
      that returned nothing but NaN.
    """
    from perflab.harness import _array

    shape = _array._shape_or_none(value)
    if shape is not None and math.prod(shape) == 0:
        raise AssertionError(
            f"{name} has shape {shape}, i.e. zero elements, so every comparison "
            f"in this check is vacuously true and nothing is actually verified. "
            f"Benchmark a size with at least one element."
        )
    # Self-comparison as a NaN probe: torch.equal, numpy.array_equal and Python
    # `==` all report False for NaN and True for everything else (including
    # infinities), so "not equal to itself" is exactly "contains NaN" on every
    # backend -- and costs one native pass rather than a Python-level scan.
    if not _array.exact_equal(value, value, name, name):
        raise AssertionError(
            f"{name} contains NaN. NaN never compares equal to itself, so it "
            f"cannot be checked for reproducibility or correctness: this check "
            f"would report a spurious difference in one phase and a spurious "
            f"pass in another. Fix the kernel (or the inputs it is given) so it "
            f"produces finite values."
        )
