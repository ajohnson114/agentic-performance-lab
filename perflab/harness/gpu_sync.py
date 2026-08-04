"""Mitigation #1: Stream Injection Guard.

Prevents LLM-generated code from executing computation on a side CUDA stream
while the benchmark timing records only the default stream.

The fix: force a full device synchronization (not just stream-level event
recording) before starting and after finishing the timed region.  This ensures
ALL streams are drained and the wall-clock time reflects true execution.

Backend-neutral. `SyncTimer` drains whichever backend is actually live:
torch CUDA, torch MPS, and JAX (whose dispatch is asynchronous on every
platform, CPU included). numpy/CPU work is synchronous, so there is nothing to
drain. Crucially, a timer built with no device -- ``SyncTimer()``, the natural
call for a task that never touches a torch device object -- no longer skips
synchronization entirely: it syncs every framework already imported in the
process. On Apple Silicon that was a real mis-timing, since MPS work launched
inside the timed region could complete after the clock stopped.

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
    """Full synchronization for any device type.

    An explicit torch device keeps its exact original handling. Everything
    else -- ``None``, a CPU device, a JAX device, a "tpu" string -- falls
    through to the live-backend drain, which is a no-op when nothing
    asynchronous is loaded and a real barrier when something is.
    """
    from perflab.harness import _array

    if device is None:
        _array.sync()
        return
    dev_type = getattr(device, "type", str(device))
    if dev_type == "cuda":
        import torch
        # Pass the device explicitly -- a bare synchronize() drains the
        # ambient current device, not necessarily the one being timed, so
        # side-stream/side-device work on another CUDA device would escape
        # the timing window.
        torch.cuda.synchronize(device)
    elif dev_type == "mps":
        import torch
        torch.mps.synchronize()
    else:
        # CPU/TPU/JAX devices and anything unrecognized: drain what is live.
        _array.sync()


class SyncTimer:
    """Timer that enforces full device synchronization around measurements.

    Hybrid timing: uses both perf_counter AND device sync to ensure
    side-stream work is captured in the timing window.
    """

    def __init__(self, device=None):
        self._device = device
        self._t0: float | None = None

    def start(self) -> None:
        """Synchronize all device streams, then start the clock."""
        _sync_device(self._device)
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        """Synchronize all device streams, then stop the clock.

        Returns elapsed time in seconds. Raises if `start` was never called:
        the start timestamp has no zero value that is safe to fall back to.
        ``perf_counter()`` counts from an arbitrary epoch (process start or
        boot, platform-dependent), so subtracting a 0.0 default returned a
        plausible-looking float in the hundred-thousand-second range -- a
        garbage measurement that no downstream check could recognize as
        garbage. A timing primitive that cannot measure must say so.
        """
        _sync_device(self._device)
        if self._t0 is None:
            raise RuntimeError(
                "SyncTimer.stop() called without a matching start(); there is "
                "no measurement window to report."
            )
        return time.perf_counter() - self._t0


@contextlib.contextmanager
def cuda_sync_guard(device=None):
    """Context manager that synchronizes before and after a timed block.

    Usage:
        with cuda_sync_guard(device):
            t0 = time.perf_counter()
            result = kernel(A, B)
        # device is fully synced here

    The trailing barrier runs in a ``finally``: a kernel that raises partway
    through can still have launched asynchronous work, and leaving that work in
    flight lets it land during -- and be charged to -- whatever the harness
    measures next.
    """
    _sync_device(device)
    try:
        yield
    finally:
        _sync_device(device)
