"""Mitigation #6: Pointer Poisoning (Memoization / Caching Guard).

Prevents LLM-generated code from using a static cache (e.g., C++
std::unordered_map or Python dict) keyed by tensor data pointers, exploiting
PyTorch's deterministic memory allocator that reuses the same addresses
across benchmark iterations.

The fix: after the initial correctness check passes, overwrite input tensors
in-place with new random data and re-run the kernel. If the kernel is
memoizing based on pointer addresses, it will return stale (incorrect) results
for the new data.

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

from collections.abc import Callable
from typing import Any


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
      2. Overwrite the SAME tensor storage with new random data (same pointers)
      3. Re-run kernel on the mutated tensors
      4. If the kernel returns the OLD result (from cache), correctness fails

    This catches pointer-keyed caches because the pointer hasn't changed but
    the data has.

    Args:
        fn: The kernel function to test.
        input_factory: Creates a fresh tuple of input tensors.
        reference_fn: Reference implementation for correctness.
        atol: Absolute tolerance. Defaults to the task's declared accuracy
            tolerance (PERFLAB_ACCURACY_TOLERANCE), or 1e-5.
        rtol: Relative tolerance (same default resolution).
        n_rounds: Number of poison rounds (default 2).

    Raises:
        AssertionError if memoization is detected.
    """
    import torch

    from perflab.harness.tolerance import env_accuracy_tolerance
    if atol is None:
        atol = env_accuracy_tolerance(1e-5)
    if rtol is None:
        rtol = env_accuracy_tolerance(1e-5)

    # Phase 1: Initial run to populate any cache
    inputs = input_factory()
    result_initial = fn(*inputs)
    ref_initial = reference_fn(*inputs)

    if isinstance(result_initial, torch.Tensor):
        result_initial = result_initial.detach().float().cpu()
        ref_initial_f = ref_initial.detach().float().cpu() if isinstance(ref_initial, torch.Tensor) else ref_initial
        if not torch.allclose(result_initial, ref_initial_f, atol=atol, rtol=rtol):
            max_diff = (result_initial - ref_initial_f).abs().max().item()
            raise AssertionError(
                f"Initial correctness check failed before poison test: "
                f"max_diff={max_diff:.2e}"
            )

    # Phase 2: Poison inputs in-place and re-run
    for round_idx in range(n_rounds):
        # Overwrite SAME tensor objects with new random data
        # This preserves the memory address but changes the content
        for inp in inputs:
            if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                inp.data.copy_(torch.randn_like(inp))
            elif isinstance(inp, torch.Tensor) and inp.dtype in (torch.int32, torch.int64):
                inp.data.copy_(torch.randint_like(inp, low=-100, high=100))

        # Re-run kernel on poisoned inputs
        result_poisoned = fn(*inputs)
        ref_poisoned = reference_fn(*inputs)

        if isinstance(result_poisoned, torch.Tensor) and isinstance(ref_poisoned, torch.Tensor):
            result_f = result_poisoned.detach().float().cpu()
            ref_f = ref_poisoned.detach().float().cpu()

            if not torch.allclose(result_f, ref_f, atol=atol, rtol=rtol):
                max_diff = (result_f - ref_f).abs().max().item()

                # Check if the result is suspiciously close to the INITIAL result
                # (i.e., the cache returned the old answer)
                if isinstance(result_initial, torch.Tensor) and torch.allclose(
                    result_f, result_initial, atol=atol, rtol=rtol
                ):
                    raise AssertionError(
                        f"Memoization detected (round {round_idx + 1}): after in-place "
                        f"input mutation, kernel returned the ORIGINAL result instead "
                        f"of computing with new data. This indicates a static cache "
                        f"keyed by tensor pointer address. max_diff_vs_ref={max_diff:.2e}"
                    )
                else:
                    raise AssertionError(
                        f"Correctness failure after input poisoning (round {round_idx + 1}): "
                        f"max_diff={max_diff:.2e}. Kernel may be reading stale data "
                        f"from a cache or buffer."
                    )
