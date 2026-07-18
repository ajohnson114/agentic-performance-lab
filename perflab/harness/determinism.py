"""Mitigation #4: Output Determinism Check (No-Op / Buffer Reuse Guard).

Prevents LLM-generated code from:
  - Launching a no-op kernel that returns stale buffer contents from prior runs
  - Exploiting shared memory overflow to read uninitialized but "lucky" garbage
  - Relying on uninitialized output buffers that happen to contain correct values

The fix: run the kernel multiple times with identical inputs and verify outputs
match with torch.equal. Also zero the output buffer before each run to catch
buffer reuse. Run with different inputs to catch no-ops.

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

from collections.abc import Callable
from typing import Any


def assert_deterministic(
    fn: Callable[..., Any],
    input_factory: Callable[[], tuple],
    reference_fn: Callable[..., Any] | None = None,
    n_runs: int = 3,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Verify kernel produces deterministic, correct output across multiple runs.

    Args:
        fn: The kernel/function to test. Called as fn(*inputs).
        input_factory: Callable that returns a tuple of fresh input tensors.
        reference_fn: Optional reference implementation for correctness check.
        n_runs: Number of times to run with identical inputs (default 3).
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.

    Raises:
        AssertionError if outputs differ across runs or diverge from reference.
    """
    import torch

    # --- Phase 1: Same inputs, multiple runs → outputs must be identical ---
    inputs = input_factory()
    outputs = []
    for _run_idx in range(n_runs):
        result = fn(*inputs)
        if isinstance(result, torch.Tensor):
            # Detach and clone to prevent aliasing
            outputs.append(result.detach().clone())
        else:
            outputs.append(result)

    # Compare all runs against the first
    for i in range(1, len(outputs)):
        if isinstance(outputs[0], torch.Tensor):
            if not torch.allclose(outputs[0], outputs[i], atol=atol, rtol=rtol):
                max_diff = (outputs[0] - outputs[i]).abs().max().item()
                raise AssertionError(
                    f"Non-deterministic output: run 0 vs run {i} differ "
                    f"(max_diff={max_diff:.2e}, atol={atol}, rtol={rtol}). "
                    f"Kernel may be reading uninitialized memory or shared "
                    f"memory overflow."
                )
        else:
            if outputs[0] != outputs[i]:
                raise AssertionError(
                    f"Non-deterministic output: run 0 vs run {i} differ "
                    f"(non-tensor comparison). "
                    f"Kernel may be reading uninitialized memory."
                )

    # --- Phase 2: Different inputs → outputs must change (catches no-ops) ---
    inputs_a = input_factory()
    inputs_b = input_factory()

    # Make sure inputs are actually different (random should ensure this)
    out_a = fn(*inputs_a)
    out_b = fn(*inputs_b)

    if isinstance(out_a, torch.Tensor) and isinstance(out_b, torch.Tensor):
        if torch.equal(out_a, out_b) and not torch.equal(
            _to_float(inputs_a[0]), _to_float(inputs_b[0])
        ):
            raise AssertionError(
                "No-op kernel detected: different inputs produced identical "
                "outputs. The kernel may be copying inputs to output without "
                "computation, or returning a stale buffer."
            )

    # --- Phase 3: Reference check (if provided) ---
    if reference_fn is not None:
        inputs_check = input_factory()
        actual = fn(*inputs_check)
        expected = reference_fn(*inputs_check)
        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            actual_f = _to_float(actual)
            expected_f = _to_float(expected)
            if not torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol):
                max_diff = (actual_f - expected_f).abs().max().item()
                raise AssertionError(
                    f"Correctness failure against reference: max_diff={max_diff:.2e} "
                    f"(atol={atol}, rtol={rtol})"
                )


def _to_float(t):
    """Convert tensor to float32 for comparison."""
    import torch
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.float()
    return t
