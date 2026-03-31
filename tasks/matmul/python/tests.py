"""Correctness test: verify pure-Python matmul against numpy reference."""
from __future__ import annotations

import numpy as np

from matmul import matmul, random_matrix


def main():
    M, N, K = 32, 32, 32

    A = random_matrix(M, K, seed=42)
    B = random_matrix(K, N, seed=123)

    C = matmul(A, B)

    # Convert to numpy for reference
    A_np = np.array(A, dtype=np.float64)
    B_np = np.array(B, dtype=np.float64)
    C_np = np.array(C, dtype=np.float64)
    C_ref = A_np @ B_np

    max_abs = np.abs(C_np - C_ref).max()
    # Pure float64 should be very accurate
    assert max_abs < 1e-8, f"max_abs too large: {max_abs}"
    print("ok", {"M": M, "N": N, "K": K, "max_abs": float(max_abs)})


if __name__ == "__main__":
    main()
