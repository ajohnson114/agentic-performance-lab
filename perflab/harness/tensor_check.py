"""Mitigation #3: Tensor Type Validation (Lazy Evaluation Guard).

Prevents LLM-generated code from returning a torch.Tensor subclass that
stores inputs but defers computation until correctness checks invoke __eq__
or other comparison operators.

The fix: validate that the output is a plain torch.Tensor (not a subclass),
has allocated storage, and a non-null data pointer.

Usage in tests.py:
    from perflab.harness.tensor_check import assert_real_tensor

    output = kernel(A, B)
    assert_real_tensor(output)
    # ... then proceed with correctness checks
"""
from __future__ import annotations


def assert_real_tensor(tensor, name: str = "output") -> None:
    """Validate that tensor is a genuine, materialized torch.Tensor.

    Checks:
      1. It is an instance of torch.Tensor
      2. Its exact type is torch.Tensor (not a subclass)
      3. It has allocated storage (not a view of nothing)
      4. Its data pointer is non-null (storage actually exists)
      5. It is not a nested tensor (which can mask lazy eval)
      6. It has a concrete shape (not symbolic/lazy)

    Raises AssertionError with a descriptive message on failure.

    Check 5 (nested) intentionally runs before check 6 (shape/stride): the
    legacy "strided" nested-tensor layout has, across torch versions,
    sometimes raised an internal RuntimeError from .shape/.stride() access
    itself (e.g. torch 2.13 raises "NestedTensorImpl doesn't support sizes")
    rather than returning symbolic values. is_nested is always safe to read
    on a nested tensor, so checking it first gives a precise, stable
    diagnostic instead of a generic "non-concrete shape" one that depends on
    which torch version happens to be installed.
    """
    import torch

    # Check 1: isinstance
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError(
            f"Lazy evaluation detected: {name} is {type(tensor).__name__}, "
            f"not torch.Tensor"
        )

    # Check 2: exact type (reject subclasses)
    if type(tensor) is not torch.Tensor:
        raise AssertionError(
            f"Lazy evaluation detected: {name} is a {type(tensor).__name__} "
            f"subclass of torch.Tensor. Custom subclasses can defer computation "
            f"until comparison operators are called."
        )

    # Check 3: has storage
    if not tensor.storage().size():
        # Zero-element tensors are OK for shape-only checks, but suspicious
        # for actual computation outputs
        if tensor.numel() > 0:
            raise AssertionError(
                f"Lazy evaluation detected: {name} claims {tensor.numel()} elements "
                f"but has empty storage"
            )

    # Check 4: non-null data pointer
    if tensor.numel() > 0 and tensor.data_ptr() == 0:
        raise AssertionError(
            f"Lazy evaluation detected: {name} has null data pointer "
            f"(storage not allocated)"
        )

    # Check 5: not a nested tensor. Deliberately before the shape/stride
    # check below -- see the docstring note on check ordering.
    if tensor.is_nested:
        raise AssertionError(
            f"Suspicious tensor: {name} is a nested tensor, which can mask "
            f"lazy evaluation patterns"
        )

    # Check 6: concrete shape (not symbolic)
    try:
        _ = tensor.shape
        _ = tensor.stride()
    except Exception as e:  # noqa: BLE001 -- any failure accessing shape/stride indicates a non-concrete tensor
        raise AssertionError(
            f"Lazy evaluation detected: {name} has non-concrete shape/stride: {e}"
        ) from e
