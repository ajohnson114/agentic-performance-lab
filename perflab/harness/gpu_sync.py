"""Mitigation #1: Stream Injection Guard.

Prevents LLM-generated code from executing computation on a side CUDA stream
while the benchmark timing records only the default stream.

The fix: force a full device synchronization (not just stream-level event
recording) before starting and after finishing the timed region.  This ensures
ALL streams are drained and the wall-clock time reflects true execution.

Usage in bench.py:
    from perflab.harness.gpu_sync import SyncTimer

    timer = SyncTimer(device)
    for _ in range(repeats):
        timer.start()
        result = kernel(A, B)
        elapsed_s = timer.stop()
        times.append(elapsed_s)
"""
from __future__ import annotations

import contextlib
import time


def _sync_device(device) -> None:
    """Full synchronization for any device type."""
    if device is None:
        return
    dev_type = getattr(device, "type", str(device))
    if dev_type == "cuda":
        import torch
        torch.cuda.synchronize()
    elif dev_type == "mps":
        import torch
        torch.mps.synchronize()
    # CPU, TPU: no sync needed (blocking by default)


class SyncTimer:
    """Timer that enforces full device synchronization around measurements.

    Hybrid timing: uses both perf_counter AND device sync to ensure
    side-stream work is captured in the timing window.
    """

    def __init__(self, device=None):
        self._device = device
        self._t0: float = 0.0

    def start(self) -> None:
        """Synchronize all device streams, then start the clock."""
        _sync_device(self._device)
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        """Synchronize all device streams, then stop the clock.

        Returns elapsed time in seconds.
        """
        _sync_device(self._device)
        return time.perf_counter() - self._t0


@contextlib.contextmanager
def cuda_sync_guard(device=None):
    """Context manager that synchronizes before and after a timed block.

    Usage:
        with cuda_sync_guard(device):
            t0 = time.perf_counter()
            result = kernel(A, B)
        # device is fully synced here
    """
    _sync_device(device)
    yield
    _sync_device(device)
