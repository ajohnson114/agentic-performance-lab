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
      5. It has a concrete shape (not symbolic/lazy)
      6. It is not a nested tensor (which can mask lazy eval)

    Raises AssertionError with a descriptive message on failure.
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

    # Check 5: concrete shape (not symbolic)
    try:
        _ = tensor.shape
        _ = tensor.stride()
    except Exception as e:
        raise AssertionError(
            f"Lazy evaluation detected: {name} has non-concrete shape/stride: {e}"
        )

    # Check 6: not a nested tensor
    if tensor.is_nested:
        raise AssertionError(
            f"Suspicious tensor: {name} is a nested tensor, which can mask "
            f"lazy evaluation patterns"
        )
