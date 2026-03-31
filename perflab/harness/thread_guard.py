"""Mitigation #2: Thread Injection Guard.

Prevents LLM-generated code from spawning background CPU threads that perform
GPU work asynchronously while the kernel returns immediately, making timing
appear faster than actual computation.

The fix: record threading.active_count() before and after kernel execution.
If new threads appear, the candidate is rejected.

Usage in bench.py:
    from perflab.harness.thread_guard import ThreadGuard

    guard = ThreadGuard()
    for _ in range(repeats):
        guard.snapshot()
        result = kernel(A, B)
        guard.check()  # raises if new threads appeared
"""
from __future__ import annotations

import threading


class ThreadGuard:
    """Monitors thread count across kernel execution boundaries.

    Captures a thread-count snapshot before kernel execution, then checks
    after execution that no new threads were spawned.
    """

    def __init__(self, tolerance: int = 0):
        """Initialize the guard.

        Args:
            tolerance: Number of new threads allowed (default 0).
                       Some frameworks lazily start thread pools on first use,
                       so set tolerance=1 or higher during warmup.
        """
        self._tolerance = tolerance
        self._baseline: int = 0
        self._baseline_names: set[str] = set()

    def snapshot(self) -> int:
        """Capture current thread count. Returns the count."""
        self._baseline = threading.active_count()
        self._baseline_names = {t.name for t in threading.enumerate()}
        return self._baseline

    def check(self) -> None:
        """Assert no new threads were created since snapshot().

        Raises AssertionError with details about new threads if violated.
        """
        current = threading.active_count()
        new_count = current - self._baseline
        if new_count > self._tolerance:
            current_names = {t.name for t in threading.enumerate()}
            new_threads = current_names - self._baseline_names
            raise AssertionError(
                f"Thread injection detected: {new_count} new thread(s) appeared "
                f"during kernel execution (tolerance={self._tolerance}). "
                f"New threads: {new_threads or '(unnamed)'}"
            )

    @property
    def thread_delta(self) -> int:
        """Return the number of new threads since last snapshot."""
        return threading.active_count() - self._baseline


def assert_no_new_threads(fn, *args, tolerance: int = 0, **kwargs):
    """Run fn(*args, **kwargs) and assert no new threads are created.

    Returns the function result.
    Raises AssertionError if new threads appear.
    """
    guard = ThreadGuard(tolerance=tolerance)
    guard.snapshot()
    result = fn(*args, **kwargs)
    guard.check()
    return result
