"""Sample task: sum of squares computed with a naive Python loop.

This is deliberately slow. An optimizing agent should replace the loop
with numpy vectorization or a math formula.
"""
from __future__ import annotations


def sum_of_squares(n: int) -> float:
    """Return 1^2 + 2^2 + ... + n^2 using a naive loop."""
    total = 0.0
    for i in range(1, n + 1):
        total += i * i
    return total
