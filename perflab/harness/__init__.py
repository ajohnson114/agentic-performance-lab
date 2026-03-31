"""PerfLab harness helpers for reward-hack mitigation.

These utilities are designed for use in protected bench.py and tests.py files
to defend against LLM-generated code that games benchmarks rather than
genuinely optimizing performance.

Mitigations provided:
  1. gpu_sync      — Force full CUDA/MPS synchronization around timed regions
  2. thread_guard  — Detect background thread injection
  3. tensor_check  — Validate output tensor type, storage, and data pointer
  4. determinism   — Verify output reproducibility across repeated runs
  5. precision     — ULP-accurate precision checking against fp64 reference
  6. pointer_poison — Defeat static memoization via input mutation and re-run
"""

from perflab.harness.gpu_sync import cuda_sync_guard, SyncTimer
from perflab.harness.thread_guard import assert_no_new_threads, ThreadGuard
from perflab.harness.tensor_check import assert_real_tensor
from perflab.harness.determinism import assert_deterministic
from perflab.harness.precision import assert_ulp_close
from perflab.harness.pointer_poison import assert_no_memoization

__all__ = [
    "cuda_sync_guard",
    "SyncTimer",
    "assert_no_new_threads",
    "ThreadGuard",
    "assert_real_tensor",
    "assert_deterministic",
    "assert_ulp_close",
    "assert_no_memoization",
]
