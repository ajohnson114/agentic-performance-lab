"""Intentionally slow pure-Python matrix multiplication.

Uses triple-nested loops with no vectorization.  An optimizing agent should
replace this with numpy or other vectorized operations.
"""
from __future__ import annotations


def zeros(rows: int, cols: int) -> list[list[float]]:
    """Create a rows x cols matrix of zeros."""
    return [[0.0] * cols for _ in range(rows)]


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply matrices A (MxK) and B (KxN) using triple-nested loops."""
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    C = zeros(M, N)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def random_matrix(rows: int, cols: int, seed: int = 0) -> list[list[float]]:
    """Simple LCG-based deterministic random matrix (values in [-0.5, 0.5])."""
    state = seed
    mat = zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            # Linear congruential generator
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            mat[i][j] = (state / 0x7FFFFFFF) - 0.5
    return mat
