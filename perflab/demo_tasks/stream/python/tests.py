"""Correctness tests for stream operations."""
import numpy as np
from stream import stream_add, stream_copy, stream_scale, stream_triad


def test_copy():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.zeros_like(A)
    stream_copy(A, B)
    assert np.allclose(B, A), f"Copy failed: {B} != {A}"


def test_scale():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.zeros_like(A)
    stream_scale(A, B, 2.0)
    assert np.allclose(B, 2.0 * A), f"Scale failed: {B}"


def test_add():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    C = np.zeros_like(A)
    stream_add(A, B, C)
    assert np.allclose(C, A + B), f"Add failed: {C}"


def test_triad():
    B = np.array([[1.0, 2.0], [3.0, 4.0]])
    C = np.array([[5.0, 6.0], [7.0, 8.0]])
    A = np.zeros_like(B)
    stream_triad(A, B, C, 3.0)
    expected = B + 3.0 * C
    assert np.allclose(A, expected), f"Triad failed: {A} != {expected}"


if __name__ == "__main__":
    test_copy()
    test_scale()
    test_add()
    test_triad()
    print("All tests passed!")
