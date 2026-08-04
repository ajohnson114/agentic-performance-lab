"""Correctness test: verify JAX matmul against numpy reference."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def main():
    M, N, K = 128, 128, 128

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (M, K), dtype=jnp.float32)
    B = jax.random.normal(k2, (K, N), dtype=jnp.float32)

    C_jax = jnp.matmul(A, B)

    # Numpy reference
    A_np = np.array(A)
    B_np = np.array(B)
    C_ref = A_np @ B_np
    C_test = np.array(C_jax)

    max_abs = np.abs(C_test - C_ref).max()
    assert max_abs < 1e-3, f"max_abs too large: {max_abs}"
    print("ok", {"M": M, "N": N, "K": K, "max_abs": float(max_abs)})


if __name__ == "__main__":
    main()
