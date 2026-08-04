"""Mitigation #6: Pointer Poisoning (Memoization / Caching Guard).

Prevents LLM-generated code from using a static cache (e.g., C++
std::unordered_map or Python dict) keyed by tensor data pointers, exploiting
PyTorch's deterministic memory allocator that reuses the same addresses
across benchmark iterations.

The fix: after the initial correctness check passes, overwrite input tensors
in-place with new random data and re-run the kernel. If the kernel is
memoizing based on pointer addresses, it will return stale (incorrect) results
for the new data.

Backend-neutral: torch tensors (``.data.copy_``), numpy arrays (``arr[...] =``)
and Python lists (recursive element assignment) are all overwritten in place,
which is what makes the poison work -- the object identity and, for
torch/numpy, the buffer address are unchanged, so a cache keyed on ``id()`` or
``data_ptr()`` still scores a hit. Immutable inputs (JAX arrays, tuples, plain
numbers) cannot be poisoned; if *no* input can be, the check raises rather
than running vacuously.

Poisoning that does not poison. Overwriting an input is only the *start* of the
check -- what matters is whether the overwrite actually moved the expected
answer, because that is the only thing a stale cached result can disagree with.
Three ways it silently does not, all of which this module now treats as
failures rather than passes:

  * the new random data lands on the old data (a one-element bool input has a
    50% chance of this per round, which was letting a memoizing kernel through
    ~40% of the time);
  * the expected answer is unchanged even though the inputs are, so a stale
    cached result still matches the reference;
  * there is nothing to compare in the first place -- a zero-element input or
    output makes every comparison vacuously true.

The kernel's result is also snapshotted the instant it is produced, before the
reference runs. A kernel and reference that share one scratch buffer otherwise
end up being compared against themselves, which passes for any kernel at all.

Usage in tests.py:
    from perflab.harness.pointer_poison import assert_no_memoization

    assert_no_memoization(
        fn=kernel,
        input_factory=lambda: (torch.randn(M, K, device=dev), torch.randn(K, N, device=dev)),
        reference_fn=lambda A, B: A @ B,
        atol=1e-5,
    )
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

#: How many times to re-roll the poison data when a round comes out vacuous
#: (unchanged inputs, or an unchanged expected answer). Small integer and bool
#: inputs collide often enough to need retries; float arrays essentially never
#: reach the second attempt.
_POISON_ATTEMPTS = 8


def assert_no_memoization(
    fn: Callable[..., Any],
    input_factory: Callable[[], tuple],
    reference_fn: Callable[..., Any],
    atol: float | None = None,
    rtol: float | None = None,
    n_rounds: int = 2,
) -> None:
    """Detect memoization by mutating inputs in-place after initial correctness pass.

    Strategy:
      1. Create inputs, run kernel, verify correctness (establishes cache entry)
      2. Overwrite the SAME storage with new random data (same pointers), and
         confirm the overwrite actually changed both the data and the answer
         the reference computes from it
      3. Re-run kernel on the mutated inputs
      4. If the kernel returns the OLD result (from cache), correctness fails

    This catches pointer-keyed caches because the pointer hasn't changed but
    the data has.

    Args:
        fn: The kernel function to test.
        input_factory: Creates a fresh tuple of input arrays.
        reference_fn: Reference implementation for correctness.
        atol: Absolute tolerance. Defaults to the task's declared accuracy
            tolerance (PERFLAB_ACCURACY_TOLERANCE), or 1e-5.
        rtol: Relative tolerance (same default resolution).
        n_rounds: Number of poison rounds (default 2).

    Raises:
        AssertionError if memoization is detected, if the initial correctness
        check fails, or if the check would be vacuous -- no input can be
        mutated in place, an input or output has zero elements, an output is
        NaN, or poisoning cannot be made to change the expected answer.
    """
    from perflab.harness import _array
    from perflab.harness.determinism import _reject_vacuous_output, _require_inputs
    from perflab.harness.tolerance import env_accuracy_tolerance

    if n_rounds < 1:
        raise ValueError(
            f"assert_no_memoization: n_rounds={n_rounds} never poisons anything, "
            f"so the check would pass on a fully memoizing kernel. Use "
            f"n_rounds >= 1."
        )
    if atol is None:
        atol = env_accuracy_tolerance(1e-5)
    if rtol is None:
        rtol = env_accuracy_tolerance(1e-5)

    # Phase 1: Initial run to populate any cache
    inputs = _require_inputs(input_factory, "assert_no_memoization")
    result_initial = fn(*inputs)
    _array.require_backend(result_initial, "kernel output")
    # Snapshot before the reference runs. Two reasons: the kernel may hand back
    # a buffer it reuses, and kernel + reference may share one scratch buffer --
    # in which case comparing them below would compare that buffer against
    # itself and clear any kernel at all.
    baseline = _array.detached_copy(result_initial, "kernel output")
    ref_initial = reference_fn(*inputs)
    _array.require_backend(ref_initial, "reference output")
    ref_baseline = _array.detached_copy(ref_initial, "reference output")

    _reject_vacuous_output(baseline, "kernel output")

    ok, max_diff = _array.allclose(
        baseline,
        ref_baseline,
        atol=atol,
        rtol=rtol,
        name_a="kernel output",
        name_b="reference output",
    )
    if not ok:
        raise AssertionError(
            f"Initial correctness check failed before poison test: "
            f"max_diff={max_diff:.2e}"
        )

    fillable = [inp for inp in inputs if _is_poisonable(inp)]
    if not fillable:
        raise AssertionError(
            "Pointer-poison check cannot run: none of the inputs can be "
            "overwritten in place with data (JAX arrays, tuples and plain "
            "numbers are immutable, and a zero-element array has nothing to "
            "overwrite), so re-running would prove nothing. Pass at least one "
            "mutable, non-empty input -- a torch tensor, a numpy array or a "
            "list."
        )

    # Phase 2: Poison inputs in-place and re-run
    for round_idx in range(n_rounds):
        # Overwrite the SAME objects with new random data. This preserves the
        # memory address but changes the content -- and is retried until the
        # content really did change and really did move the expected answer,
        # since a poison that does neither cannot expose any cache.
        ref_poisoned = _poison_until_effective(
            inputs, fillable, reference_fn, ref_baseline, round_idx
        )

        # Re-run kernel on poisoned inputs
        result_poisoned = fn(*inputs)
        _array.require_backend(result_poisoned, "kernel output")
        result_poisoned = _array.detached_copy(result_poisoned, "kernel output")

        ok, max_diff = _array.allclose(
            result_poisoned,
            ref_poisoned,
            atol=atol,
            rtol=rtol,
            name_a="kernel output",
            name_b="reference output",
        )
        if ok:
            continue

        # Check if the result is suspiciously close to the INITIAL result
        # (i.e., the cache returned the old answer). A comparison that cannot
        # even be made (e.g. the kernel changed the output shape) just means
        # "not the stale answer" -- we are already inside a failure branch, so
        # this only chooses which message to raise.
        try:
            stale, _ = _array.allclose(
                result_poisoned,
                baseline,
                atol=atol,
                rtol=rtol,
                name_a="kernel output",
                name_b="pre-poison output",
            )
        except AssertionError:
            stale = False
        if stale:
            raise AssertionError(
                f"Memoization detected (round {round_idx + 1}): after in-place "
                f"input mutation, kernel returned the ORIGINAL result instead "
                f"of computing with new data. This indicates a static cache "
                f"keyed by input identity or buffer address. "
                f"max_diff_vs_ref={max_diff:.2e}"
            )
        raise AssertionError(
            f"Correctness failure after input poisoning (round {round_idx + 1}): "
            f"max_diff={max_diff:.2e}. Kernel may be reading stale data "
            f"from a cache or buffer."
        )


def _is_poisonable(value: Any) -> bool:
    """True if overwriting ``value`` in place can actually change anything.

    ``_array.is_fillable`` answers "can this be written through", which a
    zero-element torch or numpy array passes -- writing to it succeeds and
    changes nothing, so the poison round is a no-op and the check passes on a
    fully memoizing kernel. (An empty *list* is already rejected there, so this
    also brings the three backends back into agreement.)
    """
    from perflab.harness import _array

    if not _array.is_fillable(value):
        return False
    shape = _array._shape_or_none(value)
    return shape is None or math.prod(shape) != 0


def _poison_until_effective(
    inputs: tuple,
    fillable: list,
    reference_fn: Callable[..., Any],
    ref_baseline: Any,
    round_idx: int,
) -> Any:
    """Overwrite ``fillable`` in place until the poison actually bites.

    Returns the reference output computed from the poisoned inputs -- computed
    here, before the kernel runs, so that a kernel which scribbles on its own
    inputs cannot corrupt the answer it is about to be judged against.

    A round only proves something if the poisoned inputs differ from the old
    ones AND the reference answer moved with them: a stale cached result is
    detected precisely by disagreeing with the new expected answer, so if that
    answer did not change there is nothing for it to disagree with.
    """
    from perflab.harness import _array

    for _attempt in range(_POISON_ATTEMPTS):
        changed = False
        for inp in fillable:
            before = _array.detached_copy(inp, "input")
            _array.fill_from(inp, _array.random_like(inp))
            if not _array.exact_equal(before, inp, "input", "poisoned input"):
                changed = True
        if not changed:
            # Random data landed on the old data (common for bool and small
            # integer inputs); re-roll rather than run a round that proves
            # nothing.
            continue

        ref_poisoned = reference_fn(*inputs)
        _array.require_backend(ref_poisoned, "reference output")
        ref_poisoned = _array.detached_copy(ref_poisoned, "reference output")
        if not _array.exact_equal(ref_poisoned, ref_poisoned, "reference output", "itself"):
            raise AssertionError(
                f"Reference output is NaN on poisoned inputs (round "
                f"{round_idx + 1}). The poison fills inputs with standard "
                f"normal / small integer data, which has left the reference "
                f"outside its domain, so no comparison in this round means "
                f"anything. Give the kernel inputs whose whole range is valid, "
                f"or check this task with assert_deterministic instead."
            )
        if not _array.exact_equal(
            ref_poisoned, ref_baseline, "reference output", "pre-poison reference output"
        ):
            return ref_poisoned
        # The inputs changed but the expected answer did not, so a cached stale
        # result would still look correct. Re-roll.

    raise AssertionError(
        f"Pointer-poison round {round_idx + 1} is vacuous: after "
        f"{_POISON_ATTEMPTS} attempts, overwriting the inputs with fresh random "
        f"data never changed the value the reference computes from them. A "
        f"memoizing kernel returning its cached answer would be indistinguishable "
        f"from an honest one, so this check cannot pass. Use inputs the kernel's "
        f"output actually depends on (a larger or wider-ranged array)."
    )
