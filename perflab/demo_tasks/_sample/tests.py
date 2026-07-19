"""Correctness test for the sample task.

Must exit 0 on success. Any non-zero exit code causes the agent to reject
the candidate that produced this failure.
"""
from sample import sum_of_squares


def main():
    # Known values: sum of squares formula is n*(n+1)*(2n+1)/6
    cases = [
        (1, 1.0),
        (10, 385.0),
        (100, 338350.0),
        (1000, 333833500.0),
    ]
    for n, expected in cases:
        result = sum_of_squares(n)
        assert abs(result - expected) < 1e-6, (
            f"sum_of_squares({n}) = {result}, expected {expected}"
        )

    print("ok")


if __name__ == "__main__":
    main()
