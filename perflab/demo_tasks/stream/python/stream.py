"""Memory-bound array streaming with deliberate cache-unfriendly access.

Anti-patterns:
  1. Column-major traversal of row-major arrays (stride-N access)
  2. No chunking — poor cache locality for large arrays
  3. Scalar Python loops — no vectorization
  4. Random access pattern in gather operation
"""
import numpy as np

N = 4096  # Array dimension


def stream_copy(A: np.ndarray, B: np.ndarray) -> None:
    """Copy A to B element-by-element in column-major order (cache-unfriendly)."""
    rows, cols = A.shape
    for j in range(cols):
        for i in range(rows):
            B[i, j] = A[i, j]


def stream_scale(A: np.ndarray, B: np.ndarray, scalar: float) -> None:
    """Scale: B = scalar * A, element-by-element."""
    rows, cols = A.shape
    for j in range(cols):
        for i in range(rows):
            B[i, j] = scalar * A[i, j]


def stream_add(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    """Add: C = A + B, element-by-element."""
    rows, cols = A.shape
    for j in range(cols):
        for i in range(rows):
            C[i, j] = A[i, j] + B[i, j]


def stream_triad(A: np.ndarray, B: np.ndarray, C: np.ndarray, scalar: float) -> None:
    """Triad: A = B + scalar * C, element-by-element."""
    rows, cols = A.shape
    for j in range(cols):
        for i in range(rows):
            A[i, j] = B[i, j] + scalar * C[i, j]


def run_stream(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> int:
    """Run all stream operations on caller-provided buffers.

    Buffers are allocated and reset by bench.py so allocation and RNG stay out
    of the timed region -- see bench.py's module docstring for why.
    """
    scalar = 3.0

    stream_copy(B, A)
    stream_scale(A, B, scalar)
    stream_add(A, B, C)
    stream_triad(A, B, C, scalar)

    # Total bytes: copy(2N^2) + scale(2N^2) + add(3N^2) + triad(3N^2) = 10*N^2 elements
    # Each element is 8 bytes (float64)
    return 10 * N * N * 8


if __name__ == "__main__":
    A = np.zeros((N, N), dtype=np.float64)
    B = np.random.randn(N, N).astype(np.float64)
    C = np.random.randn(N, N).astype(np.float64)
    total_bytes = run_stream(A, B, C)
    print(f"Processed {total_bytes / 1e9:.2f} GB")
    print(f"Checksum: {A.sum():.6f}")
