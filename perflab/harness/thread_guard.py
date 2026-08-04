"""Mitigation #2: Thread Injection Guard.

Prevents LLM-generated code from spawning background CPU threads that perform
GPU work asynchronously while the kernel returns immediately, making timing
appear faster than actual computation.

The fix: record which threads are alive before kernel execution, then check
after execution that no *new* thread appeared. Identity, not population count:
a candidate that starts a worker while an unrelated pre-existing thread exits
leaves ``threading.active_count()`` unchanged, so a count-only guard reports a
clean run while the smuggled worker is still going.

Scope, stated plainly: this sees Python threads -- anything registered with
the ``threading`` module. Threads created below Python (OpenMP/BLAS pools,
``std::thread`` in a C++ or CUDA extension, Grand Central Dispatch queues) are
invisible to ``threading.enumerate()`` and are NOT caught here. That is a
deliberate trade rather than an oversight: BLAS spawns its worker pool lazily
on the first matmul, so counting OS-level threads would fire on the first
honest numpy kernel in every task. Native-thread offload has to be caught by
the device-side barrier instead -- see `perflab.harness.gpu_sync.SyncTimer`,
which drains outstanding work inside the measured window.

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
    """Monitors which threads are alive across kernel execution boundaries.

    Captures the set of live threads before kernel execution, then checks
    after execution that no thread present now was absent then.
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
        # Thread objects, not names or idents: names collide freely (nothing
        # stops a candidate naming its worker "MainThread") and idents are
        # recycled by the OS once a thread exits, so a new thread can inherit
        # a dead one's ident and look like it was there all along. Holding the
        # objects also pins them, which keeps id() reuse off the table.
        self._baseline_threads: set[threading.Thread] | None = None

    def snapshot(self) -> int:
        """Capture the set of live threads. Returns the count."""
        alive = threading.enumerate()
        self._baseline_threads = set(alive)
        self._baseline = len(alive)
        return self._baseline

    def check(self) -> None:
        """Assert no new threads were created since snapshot().

        Raises AssertionError with details about new threads if violated.
        """
        if self._baseline_threads is None:
            raise AssertionError(
                "ThreadGuard.check() called before snapshot(); there is no "
                "baseline to compare against, so thread injection cannot be "
                "ruled out."
            )
        new_threads = [
            thread
            for thread in threading.enumerate()
            if thread not in self._baseline_threads
        ]
        if len(new_threads) > self._tolerance:
            names = sorted(f"{t.name}{' (daemon)' if t.daemon else ''}" for t in new_threads)
            raise AssertionError(
                f"Thread injection detected: {len(new_threads)} new thread(s) "
                f"appeared during kernel execution (tolerance={self._tolerance}). "
                f"New threads: {names}"
            )

    @property
    def thread_delta(self) -> int:
        """Return the number of new threads since last snapshot."""
        if self._baseline_threads is None:
            return threading.active_count() - self._baseline
        return sum(
            1
            for thread in threading.enumerate()
            if thread not in self._baseline_threads
        )


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
