"""Editable matmul operation for JAX.

The agent can modify this file to optimize the matrix multiplication.
For example: @jax.jit, dtype selection, custom sharding, etc.
"""
import jax.numpy as jnp


def matmul_op(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Perform matrix multiplication A @ B."""
    return jnp.matmul(A, B)
