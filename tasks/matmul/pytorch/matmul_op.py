"""Editable matmul operation for PyTorch.

The agent can modify this file to optimize the matrix multiplication.
For example: torch.compile, nn.Linear, custom kernels, AMP wrappers, etc.
"""
import torch


def matmul_op(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Perform matrix multiplication A @ B."""
    return A @ B
