"""PerfLab harness helpers for reward-hack mitigation.

These utilities are designed for use in protected bench.py and tests.py files
to defend against LLM-generated code that games benchmarks rather than
genuinely optimizing performance.

Mitigations provided:
  1. gpu_sync      — Force full CUDA/MPS/JAX synchronization around timed regions
  2. thread_guard  — Detect background thread injection
  3. tensor_check  — Validate output type, storage, and data pointer
  4. determinism   — Verify output reproducibility across repeated runs
  5. precision     — ULP-accurate precision checking against fp64 reference
  6. pointer_poison — Defeat static memoization via input mutation and re-run

Backend-neutral: values may be torch tensors, numpy arrays, JAX arrays, nested
Python lists/tuples, or plain numbers, so every PerfLab backend (Python, C++,
CUDA, PyTorch, JAX, Triton) can use these helpers. Backends are detected from
the value's own type, so importing this package pulls in no optional
dependency; a value whose type cannot be inspected makes the check raise
rather than pass. See perflab/harness/_array.py for the adapter.
"""

from perflab.harness.determinism import assert_deterministic
from perflab.harness.gpu_sync import SyncTimer, cuda_sync_guard
from perflab.harness.pointer_poison import assert_no_memoization
from perflab.harness.precision import assert_ulp_close
from perflab.harness.tensor_check import assert_real_array, assert_real_tensor
from perflab.harness.thread_guard import ThreadGuard, assert_no_new_threads
from perflab.harness.tolerance import env_accuracy_tolerance

__all__ = [
    "cuda_sync_guard",
    "SyncTimer",
    "assert_no_new_threads",
    "ThreadGuard",
    "assert_real_tensor",
    "assert_real_array",
    "assert_deterministic",
    "assert_ulp_close",
    "assert_no_memoization",
    "env_accuracy_tolerance",
]
