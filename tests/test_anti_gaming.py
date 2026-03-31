"""Tests for reward-hack mitigations.

Covers:
  1. perflab.harness.gpu_sync — SyncTimer, cuda_sync_guard
  2. perflab.harness.thread_guard — ThreadGuard, assert_no_new_threads
  3. perflab.harness.tensor_check — assert_real_tensor
  4. perflab.harness.determinism — assert_deterministic
  5. perflab.harness.precision — assert_ulp_close, _ulp_distance
  6. perflab.harness.pointer_poison — assert_no_memoization
  7. perflab.runners.benchmark — validate_bench_variance
  8. perflab.runners.correctness — run_correctness_twice
  9. perflab.task_spec — AntiGamingSpec parsing
 10. perflab.optimizers.event_log — anti_gaming_warning event
"""
from __future__ import annotations

import json
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. GPU Sync (SyncTimer, cuda_sync_guard)
# ---------------------------------------------------------------------------

class TestSyncTimer:
    def test_start_stop_returns_positive_duration(self):
        from perflab.harness.gpu_sync import SyncTimer
        timer = SyncTimer(device=None)  # CPU mode
        timer.start()
        time.sleep(0.01)
        elapsed = timer.stop()
        assert elapsed > 0
        assert elapsed < 1.0  # sanity

    def test_sync_called_for_cuda_device(self):
        from perflab.harness.gpu_sync import SyncTimer
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_device = MagicMock()
        mock_device.type = "cuda"
        with patch.dict("sys.modules", {"torch": mock_torch}):
            timer = SyncTimer(device=mock_device)
            timer.start()
            timer.stop()
        assert mock_torch.cuda.synchronize.call_count == 2  # start + stop

    def test_sync_called_for_mps_device(self):
        from perflab.harness.gpu_sync import SyncTimer
        mock_torch = MagicMock()
        mock_device = MagicMock()
        mock_device.type = "mps"
        with patch.dict("sys.modules", {"torch": mock_torch}):
            timer = SyncTimer(device=mock_device)
            timer.start()
            timer.stop()
        assert mock_torch.mps.synchronize.call_count == 2

    def test_cuda_sync_guard_context_manager(self):
        from perflab.harness.gpu_sync import cuda_sync_guard
        # CPU: no sync needed, just verify it doesn't crash
        with cuda_sync_guard(device=None):
            x = 1 + 1
        assert x == 2


# ---------------------------------------------------------------------------
# 2. Thread Guard
# ---------------------------------------------------------------------------

class TestThreadGuard:
    def test_no_new_threads_passes(self):
        from perflab.harness.thread_guard import ThreadGuard
        guard = ThreadGuard()
        guard.snapshot()
        _ = sum(range(100))
        guard.check()  # should not raise

    def test_new_thread_detected(self):
        from perflab.harness.thread_guard import ThreadGuard
        guard = ThreadGuard()
        guard.snapshot()

        barrier = threading.Event()
        def worker():
            barrier.wait()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        try:
            with pytest.raises(AssertionError, match="Thread injection detected"):
                guard.check()
        finally:
            barrier.set()
            t.join(timeout=2)

    def test_tolerance_allows_threads(self):
        from perflab.harness.thread_guard import ThreadGuard
        guard = ThreadGuard(tolerance=1)
        guard.snapshot()

        barrier = threading.Event()
        t = threading.Thread(target=lambda: barrier.wait(), daemon=True)
        t.start()
        try:
            guard.check()  # should NOT raise with tolerance=1
        finally:
            barrier.set()
            t.join(timeout=2)

    def test_assert_no_new_threads_functional(self):
        from perflab.harness.thread_guard import assert_no_new_threads
        result = assert_no_new_threads(lambda x: x * 2, 21)
        assert result == 42

    def test_thread_delta_property(self):
        from perflab.harness.thread_guard import ThreadGuard
        guard = ThreadGuard()
        guard.snapshot()
        assert guard.thread_delta == 0


# ---------------------------------------------------------------------------
# 3. Tensor Check
# ---------------------------------------------------------------------------

class TestTensorCheck:
    def test_real_tensor_passes(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.tensor_check import assert_real_tensor
        t = torch.randn(3, 4)
        assert_real_tensor(t)  # should not raise

    def test_non_tensor_fails(self):
        pytest.importorskip("torch")
        from perflab.harness.tensor_check import assert_real_tensor
        with pytest.raises(AssertionError, match="not torch.Tensor"):
            assert_real_tensor([1, 2, 3])

    def test_tensor_subclass_fails(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.tensor_check import assert_real_tensor

        class LazyTensor(torch.Tensor):
            pass

        t = LazyTensor(torch.randn(3, 4))
        with pytest.raises(AssertionError, match="subclass"):
            assert_real_tensor(t)

    def test_null_data_ptr_fails(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.tensor_check import assert_real_tensor
        # Empty tensor with zero elements is fine
        t = torch.empty(0)
        assert_real_tensor(t)  # should not raise for zero-element

    def test_nested_tensor_fails(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.tensor_check import assert_real_tensor
        if not hasattr(torch, "nested"):
            pytest.skip("torch.nested not available")
        try:
            nt = torch.nested.nested_tensor([torch.randn(2), torch.randn(3)])
            with pytest.raises(AssertionError, match="nested"):
                assert_real_tensor(nt)
        except (AttributeError, RuntimeError):
            pytest.skip("nested tensor creation not supported")


# ---------------------------------------------------------------------------
# 4. Determinism Check
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_deterministic_fn_passes(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.determinism import assert_deterministic

        def matmul(A, B):
            return A @ B

        assert_deterministic(
            fn=matmul,
            input_factory=lambda: (torch.randn(8, 8), torch.randn(8, 8)),
            reference_fn=matmul,
            n_runs=3,
            atol=1e-5,
        )

    def test_noop_kernel_detected(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.determinism import assert_deterministic

        fixed_output = torch.randn(8, 8)

        def noop_kernel(A, B):
            return fixed_output.clone()

        with pytest.raises(AssertionError, match="No-op kernel detected"):
            assert_deterministic(
                fn=noop_kernel,
                input_factory=lambda: (torch.randn(8, 8), torch.randn(8, 8)),
            )

    def test_nondeterministic_kernel_detected(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.determinism import assert_deterministic

        call_count = [0]

        def unstable_kernel(A, B):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return A @ B + 1000.0  # wrong on even calls
            return A @ B

        with pytest.raises(AssertionError, match="Non-deterministic"):
            assert_deterministic(
                fn=unstable_kernel,
                input_factory=lambda: (torch.ones(4, 4), torch.ones(4, 4)),
                n_runs=4,
                atol=1e-5,
            )


# ---------------------------------------------------------------------------
# 5. Precision (ULP) Check
# ---------------------------------------------------------------------------

class TestPrecision:
    def test_ulp_distance_same_value(self):
        from perflab.harness.precision import _ulp_distance
        assert _ulp_distance(1.0, 1.0) == 0.0

    def test_ulp_distance_nan(self):
        from perflab.harness.precision import _ulp_distance
        import math
        assert _ulp_distance(float("nan"), 1.0) == float("inf")

    def test_ulp_distance_close_values(self):
        from perflab.harness.precision import _ulp_distance
        # Adjacent floats should be 1 ULP apart
        import struct
        bits = struct.pack('d', 1.0)
        int_val = struct.unpack('Q', bits)[0]
        next_float = struct.unpack('d', struct.pack('Q', int_val + 1))[0]
        dist = _ulp_distance(1.0, next_float)
        assert dist == pytest.approx(1.0, abs=0.1)

    def test_exact_fp32_passes(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.precision import assert_ulp_close
        a = torch.randn(100, 100)
        stats = assert_ulp_close(a, a, max_ulp=0)
        assert stats["max_ulp_observed"] == 0

    def test_fp16_cast_detected(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.precision import assert_ulp_close

        # Create fp32 data, downcast to fp16, upcast back — precision loss
        ref = torch.randn(1000) * 100  # large values amplify fp16 error
        degraded = ref.half().float()

        with pytest.raises(AssertionError, match="Precision downgrade"):
            assert_ulp_close(degraded, ref, max_ulp=2)

    def test_dtype_mismatch_detected(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.precision import assert_ulp_close
        actual = torch.randn(10).half()
        ref = torch.randn(10)
        with pytest.raises(AssertionError, match="Precision downgrade.*dtype"):
            assert_ulp_close(actual, ref, expected_dtype=torch.float32)


# ---------------------------------------------------------------------------
# 6. Pointer Poisoning
# ---------------------------------------------------------------------------

class TestPointerPoison:
    def test_honest_kernel_passes(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.pointer_poison import assert_no_memoization

        assert_no_memoization(
            fn=lambda A, B: A @ B,
            input_factory=lambda: (torch.randn(8, 8), torch.randn(8, 8)),
            reference_fn=lambda A, B: A @ B,
            atol=1e-4,
        )

    def test_memoizing_kernel_detected(self):
        torch = pytest.importorskip("torch")
        from perflab.harness.pointer_poison import assert_no_memoization

        cache = {}

        def caching_kernel(A, B):
            key = A.data_ptr()
            if key not in cache:
                cache[key] = (A @ B).clone()
            return cache[key]

        with pytest.raises(AssertionError, match="[Mm]emoization|[Cc]orrectness"):
            assert_no_memoization(
                fn=caching_kernel,
                input_factory=lambda: (torch.randn(8, 8), torch.randn(8, 8)),
                reference_fn=lambda A, B: A @ B,
                atol=1e-4,
            )


# ---------------------------------------------------------------------------
# 7. Benchmark Variance Check
# ---------------------------------------------------------------------------

class TestBenchVariance:
    def test_normal_variance_passes(self):
        from perflab.runners.benchmark import validate_bench_variance
        bench = {
            "times_ms": [10.1, 10.3, 9.8, 10.5, 10.0],
            "ok": True,
        }
        warnings = validate_bench_variance(bench)
        assert warnings == []

    def test_zero_variance_detected(self):
        from perflab.runners.benchmark import validate_bench_variance
        bench = {
            "throughput": {
                "all": [100.0, 100.0, 100.0, 100.0, 100.0],
                "median": 100.0,
            },
            "ok": True,
        }
        warnings = validate_bench_variance(bench)
        assert len(warnings) == 1
        assert "Zero variance" in warnings[0]
        assert "memoization" in warnings[0].lower() or "caching" in warnings[0].lower()

    def test_short_arrays_skipped(self):
        from perflab.runners.benchmark import validate_bench_variance
        bench = {
            "values": [1.0, 1.0],  # only 2 values, not checked
            "ok": True,
        }
        warnings = validate_bench_variance(bench)
        assert warnings == []

    def test_nested_zero_variance(self):
        from perflab.runners.benchmark import validate_bench_variance
        bench = {
            "latency": {
                "all": [5.0, 5.0, 5.0, 5.0],
            },
            "ok": True,
        }
        warnings = validate_bench_variance(bench)
        assert len(warnings) == 1
        assert "latency.all" in warnings[0]

    def test_non_numeric_arrays_ignored(self):
        from perflab.runners.benchmark import validate_bench_variance
        bench = {
            "labels": ["a", "b", "c", "d"],
            "ok": True,
        }
        warnings = validate_bench_variance(bench)
        assert warnings == []


# ---------------------------------------------------------------------------
# 8. Correctness Twice (determinism re-run)
# ---------------------------------------------------------------------------

class TestCorrectnessTwice:
    def test_both_pass(self, tmp_workspace: Path):
        from perflab.runners.correctness import run_correctness_twice
        test_script = tmp_workspace / "tests.py"
        test_script.write_text("print('ok')\n")
        res, warnings = run_correctness_twice(
            f"python3 {test_script}", cwd=tmp_workspace,
        )
        assert res.returncode == 0
        assert warnings == []

    def test_first_pass_second_fail(self, tmp_workspace: Path):
        from perflab.runners.correctness import run_correctness_twice
        import os
        # Script that fails when PERFLAB_DETERMINISM_SEED is set
        test_script = tmp_workspace / "tests.py"
        test_script.write_text(textwrap.dedent("""\
            import os, sys
            if os.environ.get("PERFLAB_DETERMINISM_SEED"):
                print("fail", file=sys.stderr)
                sys.exit(1)
            print("ok")
        """))
        res, warnings = run_correctness_twice(
            f"python3 {test_script}", cwd=tmp_workspace,
        )
        assert res.returncode == 0  # first run passed
        assert len(warnings) == 1
        assert "Determinism check failed" in warnings[0]

    def test_first_fail_skips_second(self, tmp_workspace: Path):
        from perflab.runners.correctness import run_correctness_twice
        test_script = tmp_workspace / "tests.py"
        test_script.write_text("import sys; sys.exit(1)\n")
        res, warnings = run_correctness_twice(
            f"python3 {test_script}", cwd=tmp_workspace,
        )
        assert res.returncode == 1
        assert warnings == []  # no second run attempted


# ---------------------------------------------------------------------------
# 9. AntiGamingSpec in task_spec.py
# ---------------------------------------------------------------------------

class TestAntiGamingSpec:
    def test_defaults(self, sample_task_yaml: Path):
        from perflab.task_spec import TaskSpec
        task = TaskSpec.load(sample_task_yaml)
        assert task.anti_gaming.bench_variance_check is True
        assert task.anti_gaming.determinism_rerun is True
        assert task.anti_gaming.gaming_speedup_threshold == 100.0
        assert task.anti_gaming.thread_count_check is False
        assert task.anti_gaming.max_thread_delta == 0

    def test_custom_values(self, tmp_workspace: Path):
        from perflab.task_spec import TaskSpec
        task_file = tmp_workspace / "task.yaml"
        task_file.write_text(textwrap.dedent("""\
            name: anti-gaming-test
            program_type: cuda
            correctness:
              cmd: "python tests.py"
            benchmark:
              cmd: "python bench.py --json out/bench.json"
              metric:
                name: tflops.median
                mode: maximize
            anti_gaming:
              bench_variance_check: false
              determinism_rerun: false
              gaming_speedup_threshold: 5.0
              thread_count_check: true
              max_thread_delta: 2
        """))
        task = TaskSpec.load(task_file)
        assert task.anti_gaming.bench_variance_check is False
        assert task.anti_gaming.determinism_rerun is False
        assert task.anti_gaming.gaming_speedup_threshold == 5.0
        assert task.anti_gaming.thread_count_check is True
        assert task.anti_gaming.max_thread_delta == 2

    def test_missing_section_uses_defaults(self, tmp_workspace: Path):
        from perflab.task_spec import TaskSpec
        task_file = tmp_workspace / "task.yaml"
        task_file.write_text(textwrap.dedent("""\
            name: no-anti-gaming
            program_type: python
            correctness:
              cmd: "python tests.py"
            benchmark:
              cmd: "python bench.py --json out/bench.json"
              metric:
                name: throughput.median
        """))
        task = TaskSpec.load(task_file)
        assert task.anti_gaming.bench_variance_check is True
        assert task.anti_gaming.determinism_rerun is True


# ---------------------------------------------------------------------------
# 10. Event Log
# ---------------------------------------------------------------------------

class TestAntiGamingEventLog:
    def test_anti_gaming_warning_written(self, tmp_path: Path):
        from perflab.optimizers.event_log import AgentEventLog
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        log = AgentEventLog(run_dir=run_dir)
        log.anti_gaming_warning(
            iteration=2,
            check_type="bench_variance",
            details="Zero variance in times_ms",
            candidate_index=1,
        )
        events_path = run_dir / "agent_events.jsonl"
        assert events_path.exists()
        event = json.loads(events_path.read_text().strip())
        assert event["event_type"] == "anti_gaming_warning"
        assert event["iteration"] == 2
        assert event["check_type"] == "bench_variance"
        assert event["candidate_index"] == 1

    def test_replay_includes_anti_gaming(self, tmp_path: Path):
        from perflab.optimizers.event_log import AgentEventLog, replay_events
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        log = AgentEventLog(run_dir=run_dir)
        log.anti_gaming_warning(1, "suspicious_speedup", "15x speedup on iter 1")
        text = replay_events(run_dir)
        assert "ANTI-GAMING" in text
        assert "suspicious_speedup" in text


# ---------------------------------------------------------------------------
# 11. Integration: harness __init__ imports
# ---------------------------------------------------------------------------

class TestHarnessImports:
    def test_all_exports_importable(self):
        from perflab.harness import (
            cuda_sync_guard,
            SyncTimer,
            assert_no_new_threads,
            ThreadGuard,
            assert_real_tensor,
            assert_deterministic,
            assert_ulp_close,
            assert_no_memoization,
        )
        assert callable(cuda_sync_guard)
        assert callable(SyncTimer)
        assert callable(assert_no_new_threads)
        assert callable(ThreadGuard)
        assert callable(assert_real_tensor)
        assert callable(assert_deterministic)
        assert callable(assert_ulp_close)
        assert callable(assert_no_memoization)
