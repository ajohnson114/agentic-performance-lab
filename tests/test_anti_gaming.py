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
        # Must sync the device being timed, not whatever the ambient current
        # device happens to be -- a bare synchronize() call drains the wrong
        # device when timed code runs on a non-default CUDA device/stream.
        mock_torch.cuda.synchronize.assert_called_with(mock_device)

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
        assert _ulp_distance(float("nan"), 1.0) == float("inf")

    def test_ulp_distance_close_values(self):
        # Adjacent floats should be 1 ULP apart
        import struct

        from perflab.harness.precision import _ulp_distance
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

    def test_single_outlier_missed_by_floor_indexing_now_caught(self, monkeypatch):
        """n=4 with ulp dists [1, 2, 3, 5000]: floor-indexed p99
        (int(0.99 * 3) == 2) picked 3 and let the 5000-ULP outlier through
        entirely. Ceiling indexing must pick index 3 and catch it."""
        torch = pytest.importorskip("torch")
        from perflab.harness import precision as precision_mod

        dists = iter([1.0, 2.0, 3.0, 5000.0])
        monkeypatch.setattr(precision_mod, "_ulp_distance", lambda a, b: next(dists))

        a = torch.zeros(4)
        b = torch.zeros(4)
        with pytest.raises(AssertionError, match="Precision downgrade"):
            precision_mod.assert_ulp_close(a, b, max_ulp=10.0)

    def test_hard_ulp_ceiling_catches_outlier_percentile_alone_misses(self, monkeypatch):
        """A single catastrophic outlier can still slip past a ceiling-indexed
        p99 once the sample is large enough that the top 1% covers more than
        one element. HARD_MAX_ULP_FACTOR is the backstop for that case."""
        torch = pytest.importorskip("torch")
        from perflab.harness import precision as precision_mod

        n = 200
        values = [1.0] * (n - 1) + [10000.0]
        dists = iter(values)
        monkeypatch.setattr(precision_mod, "_ulp_distance", lambda a, b: next(dists))

        a = torch.zeros(n)
        b = torch.zeros(n)
        with pytest.raises(AssertionError, match="hard ceiling"):
            precision_mod.assert_ulp_close(a, b, max_ulp=10.0)


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
        assert task.anti_gaming.gaming_speedup_threshold == 10.0
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
            SyncTimer,
            ThreadGuard,
            assert_deterministic,
            assert_no_memoization,
            assert_no_new_threads,
            assert_real_tensor,
            assert_ulp_close,
            cuda_sync_guard,
        )
        assert callable(cuda_sync_guard)
        assert callable(SyncTimer)
        assert callable(assert_no_new_threads)
        assert callable(ThreadGuard)
        assert callable(assert_real_tensor)
        assert callable(assert_deterministic)
        assert callable(assert_ulp_close)
        assert callable(assert_no_memoization)


# ---------------------------------------------------------------------------
# 11. Anti-gaming wiring into the evaluate/accept flow
# ---------------------------------------------------------------------------

class _RecordingEventLog:
    """Event log stub recording every (method, args, kwargs) call."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return _record

    def named(self, method: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == method]


def _eval_ctx(tmp_workspace, sample_task_yaml, tmp_path, event_log=None, **cfg):
    from types import SimpleNamespace

    from perflab.optimizers.agent import AgentConfig, AgentContext
    from perflab.optimizers.progress import PrintProgress
    from perflab.task_spec import TaskSpec

    (tmp_workspace / "algo.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    task = TaskSpec.load(sample_task_yaml)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = AgentContext(
        task=task,
        config=AgentConfig(**cfg),
        llm_config=None,
        provider=None,
        progress=PrintProgress(),
        ws=task.workspace,
        rp=SimpleNamespace(run_dir=run_dir, artifacts_dir=run_dir / "artifacts"),
        event_log=event_log if event_log is not None else _RecordingEventLog(),
    )
    ctx.iteration = 1
    return ctx


def _cmd_ok(returncode: int = 0):
    from perflab.tools.shell import CmdResult
    return CmdResult(cmd=[], returncode=returncode, stdout="", stderr="", duration_s=0.01)


def _bench(value: float = 2.0, extra: dict | None = None) -> dict:
    b = {
        "ok": True,
        "throughput": {"median": value},
        "meta": {"warmup": 1, "repeats": 5},
    }
    if extra:
        b.update(extra)
    return b


_BLOCK_ARGS = ("algo.py", "return 1", "return 2")


class TestDeterminismRerunWiring:
    def test_determinism_failure_rejects_candidate(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        from perflab.optimizers.patch import SearchReplaceBlock
        from perflab.optimizers.phases import evaluate as evaluate_mod

        log = _RecordingEventLog()
        ctx = _eval_ctx(tmp_workspace, sample_task_yaml, tmp_path, event_log=log)
        assert ctx.task.anti_gaming.determinism_rerun is True  # default on

        monkeypatch.setattr(
            evaluate_mod, "run_correctness_twice",
            lambda *a, **k: (_cmd_ok(), ["Determinism check failed: rc mismatch"]),
        )
        bench_called = []
        monkeypatch.setattr(
            evaluate_mod, "run_benchmark",
            lambda *a, **k: bench_called.append(1) or (_cmd_ok(), _bench()),
        )

        cand, errors = evaluate_mod.evaluate_single_candidate(
            ctx, 0, [SearchReplaceBlock(*_BLOCK_ARGS)], reasoning="", use_fast=False,
        )

        assert cand.value is None
        assert "determinism re-run failed" in cand.description
        assert errors and errors[0]["type"] == "anti_gaming"
        assert bench_called == []  # rejected before benchmarking
        assert log.named("anti_gaming_warning")

    def test_determinism_rerun_disabled_uses_single_run(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        from perflab.optimizers.patch import SearchReplaceBlock
        from perflab.optimizers.phases import evaluate as evaluate_mod

        ctx = _eval_ctx(tmp_workspace, sample_task_yaml, tmp_path)
        ctx.task.anti_gaming.determinism_rerun = False

        calls = {"single": 0, "twice": 0}

        def _single(*a, **k):
            calls["single"] += 1
            return _cmd_ok()

        def _twice(*a, **k):
            calls["twice"] += 1
            return _cmd_ok(), []

        monkeypatch.setattr(evaluate_mod, "run_correctness", _single)
        monkeypatch.setattr(evaluate_mod, "run_correctness_twice", _twice)
        monkeypatch.setattr(
            evaluate_mod, "run_benchmark", lambda *a, **k: (_cmd_ok(), _bench()),
        )

        cand, errors = evaluate_mod.evaluate_single_candidate(
            ctx, 0, [SearchReplaceBlock(*_BLOCK_ARGS)], reasoning="", use_fast=False,
        )

        assert cand.value == 2.0
        assert calls == {"single": 1, "twice": 0}


class TestThreadCountCheckWiring:
    def _run(self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch, bench):
        from perflab.optimizers.patch import SearchReplaceBlock
        from perflab.optimizers.phases import evaluate as evaluate_mod

        log = _RecordingEventLog()
        ctx = _eval_ctx(tmp_workspace, sample_task_yaml, tmp_path, event_log=log)
        ctx.task.anti_gaming.determinism_rerun = False
        ctx.task.anti_gaming.thread_count_check = True
        ctx.task.anti_gaming.max_thread_delta = 0

        monkeypatch.setattr(evaluate_mod, "run_correctness", lambda *a, **k: _cmd_ok())
        monkeypatch.setattr(
            evaluate_mod, "run_benchmark", lambda *a, **k: (_cmd_ok(), bench),
        )
        cand, errors = evaluate_mod.evaluate_single_candidate(
            ctx, 0, [SearchReplaceBlock(*_BLOCK_ARGS)], reasoning="", use_fast=False,
        )
        return cand, errors, log

    def test_thread_delta_above_max_rejects(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        cand, errors, log = self._run(
            tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
            _bench(extra={"thread_delta": 3}),
        )
        assert cand.value is None
        assert "thread check failed" in cand.description
        assert errors[0]["type"] == "anti_gaming"
        assert log.named("anti_gaming_warning")

    def test_thread_delta_within_max_passes(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        cand, errors, log = self._run(
            tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
            _bench(extra={"thread_delta": 0}),
        )
        assert cand.value == 2.0

    def test_missing_thread_delta_warns_but_passes(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        cand, errors, log = self._run(
            tmp_workspace, sample_task_yaml, tmp_path, monkeypatch, _bench(),
        )
        assert cand.value == 2.0
        warnings = log.named("anti_gaming_warning")
        assert warnings and warnings[0][1][1] == "thread_count"

    def test_unparseable_thread_delta_rejects_without_raising(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        # A candidate-controlled non-numeric thread_delta must reject the
        # candidate, not crash the run.
        cand, errors, log = self._run(
            tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
            _bench(extra={"thread_delta": "lots"}),
        )
        assert cand.value is None
        assert "thread check failed" in cand.description
        assert errors[0]["type"] == "anti_gaming"
        # Prompt-injection guard: raw value goes in "output", not "description".
        assert "lots" not in errors[0]["description"]
        assert "lots" in errors[0]["output"]
        assert log.named("anti_gaming_warning")

    def test_nonfinite_thread_delta_rejects_without_raising(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        cand, errors, log = self._run(
            tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
            _bench(extra={"thread_delta": float("nan")}),
        )
        assert cand.value is None
        assert errors[0]["type"] == "anti_gaming"

    def test_null_meta_does_not_crash(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        # bench.json with "meta": null previously crashed on meta.get(...).
        cand, errors, log = self._run(
            tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
            _bench(extra={"meta": None}),
        )
        assert cand.value == 2.0  # no thread_delta anywhere -> warn and pass
        warnings = log.named("anti_gaming_warning")
        assert warnings and warnings[0][1][1] == "thread_count"

    def test_thread_delta_from_null_meta_with_top_level_value(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        # meta is null but a top-level thread_delta is present and over the max.
        cand, errors, log = self._run(
            tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
            _bench(extra={"meta": None, "thread_delta": 3}),
        )
        assert cand.value is None
        assert "thread check failed" in cand.description
        assert errors[0]["type"] == "anti_gaming"


class TestBenchVarianceWiring:
    def test_zero_variance_is_advisory_not_rejecting(
        self, tmp_workspace, sample_task_yaml, tmp_path, monkeypatch,
    ):
        from perflab.optimizers.patch import SearchReplaceBlock
        from perflab.optimizers.phases import evaluate as evaluate_mod

        log = _RecordingEventLog()
        ctx = _eval_ctx(tmp_workspace, sample_task_yaml, tmp_path, event_log=log)
        ctx.task.anti_gaming.determinism_rerun = False

        cached_bench = _bench()
        cached_bench["throughput"]["all"] = [5.0, 5.0, 5.0, 5.0]

        monkeypatch.setattr(evaluate_mod, "run_correctness", lambda *a, **k: _cmd_ok())
        monkeypatch.setattr(
            evaluate_mod, "run_benchmark", lambda *a, **k: (_cmd_ok(), cached_bench),
        )

        cand, errors = evaluate_mod.evaluate_single_candidate(
            ctx, 0, [SearchReplaceBlock(*_BLOCK_ARGS)], reasoning="", use_fast=False,
        )

        assert cand.value == 2.0  # advisory: still scored
        warnings = log.named("anti_gaming_warning")
        assert warnings and warnings[0][1][1] == "bench_variance"


class TestGamingSpeedupDetector:
    def _accept(self, tmp_path, mode, baseline, cand_value, threshold):
        from types import SimpleNamespace

        from perflab.optimizers.phases import evaluate as evaluate_mod

        log = _RecordingEventLog()
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        ctx = SimpleNamespace(
            task=SimpleNamespace(
                benchmark=SimpleNamespace(
                    metric=SimpleNamespace(name="m", mode=mode),
                ),
                constraints=SimpleNamespace(regression_tolerance=0.02),
                anti_gaming=SimpleNamespace(gaming_speedup_threshold=threshold),
            ),
            ws=ws,
            rp=SimpleNamespace(run_dir=tmp_path),
            iteration=1,
            progress=SimpleNamespace(on_message=lambda m: None),
            event_log=log,
            history=[],
            baseline_val=baseline,
            best_value=baseline,
            best_iter=0,
            accepted_patches=[],
            accepted_count=0,
            sec_metric=None,
            config=SimpleNamespace(isolation=None, top_k=3),
        )
        cand = __import__(
            "perflab.optimizers.phases.evaluate", fromlist=["BeamCandidate"],
        ).BeamCandidate(
            iteration=1, index=0, blocks=[], description="candidate 1", value=cand_value,
        )
        with patch.object(evaluate_mod, "snapshot_workspace", lambda *a, **k: None):
            accepted, _, _ = evaluate_mod.accept_best(ctx, [cand], use_fast=False)
        return accepted, log

    def test_fires_for_minimize_mode_metrics(self, tmp_path):
        # 100ms -> 0.5ms latency = 200x single-step gain; threshold 100x.
        # The old value/baseline ratio would be 0.005 and could never fire.
        accepted, log = self._accept(
            tmp_path, mode="minimize", baseline=100.0, cand_value=0.5, threshold=100.0,
        )
        assert accepted is True
        warnings = log.named("anti_gaming_warning")
        assert warnings and warnings[0][1][1] == "speedup_threshold"

    def test_respects_configured_threshold(self, tmp_path):
        # 4x gain under a 5x threshold: no warning...
        accepted, log = self._accept(
            tmp_path, mode="maximize", baseline=1.0, cand_value=4.0, threshold=5.0,
        )
        assert accepted is True
        assert log.named("anti_gaming_warning") == []

        # ...but the same gain under a 3x threshold warns.
        accepted, log = self._accept(
            tmp_path, mode="maximize", baseline=1.0, cand_value=4.0, threshold=3.0,
        )
        assert accepted is True
        assert log.named("anti_gaming_warning")

    def test_zero_metric_minimize_flagged(self, tmp_path):
        # A minimize-mode candidate reporting exactly 0.0 (stubbed/no-op kernel)
        # is accepted, but must trigger the zero_metric anti-gaming warning:
        # improvement_factor stays neutral (1.0) for a zero value, so the
        # gain>threshold check never fires for it.
        accepted, log = self._accept(
            tmp_path, mode="minimize", baseline=100.0, cand_value=0.0, threshold=100.0,
        )
        assert accepted is True
        kinds = [w[1][1] for w in log.named("anti_gaming_warning")]
        assert "zero_metric" in kinds

    def test_zero_metric_maximize_not_flagged(self, tmp_path):
        # A real 0.0 under maximize can't beat a positive baseline, so it's not
        # accepted and zero_metric never fires (the guard is minimize-only).
        accepted, log = self._accept(
            tmp_path, mode="maximize", baseline=100.0, cand_value=0.0, threshold=100.0,
        )
        assert accepted is False
        kinds = [w[1][1] for w in log.named("anti_gaming_warning")]
        assert "zero_metric" not in kinds


class TestImprovementFactor:
    def test_maximize(self):
        from perflab.analyzers.metrics_rollup import improvement_factor
        assert improvement_factor(10.0, 1.0, "maximize") == 10.0
        assert improvement_factor(1.0, 10.0, "maximize") == 0.1

    def test_minimize(self):
        from perflab.analyzers.metrics_rollup import improvement_factor
        assert improvement_factor(1.0, 10.0, "minimize") == 10.0
        assert improvement_factor(10.0, 1.0, "minimize") == 0.1

    def test_zero_values_return_neutral(self):
        from perflab.analyzers.metrics_rollup import improvement_factor
        assert improvement_factor(0.0, 10.0, "minimize") == 1.0
        assert improvement_factor(10.0, 0.0, "maximize") == 1.0


# ---------------------------------------------------------------------------
# 12. Protected-file tamper check must fire before autotune's sweep trials
# ---------------------------------------------------------------------------

class _SilentProgress:
    def on_message(self, msg: str) -> None:
        pass


class TestProtectedFileCheckBeforeAutotune:
    """Accepted candidate code executes in the *real* workspace during
    autotune's up-to-15 correctness+bench trials (agent.py's iteration loop,
    ~L360-380). Previously verify_protected_files() only ran once at the end
    of the iteration, so a candidate that rewrites tests.py at runtime could
    poison every sweep decision until then. The guard must also run right
    after accept_best() reports accepted_any=True, before autotune.run().
    """

    def test_restored_before_autotune_runs(
        self, tmp_workspace, sample_task_yaml, monkeypatch,
    ):
        from types import SimpleNamespace

        from perflab.llm.config import LLMConfig
        from perflab.optimizers.agent import AgentConfig, run_agent
        from perflab.optimizers.patch import SearchReplaceBlock
        from perflab.optimizers.phases import autotune as autotune_mod
        from perflab.optimizers.phases import baseline as baseline_mod
        from perflab.optimizers.phases import evaluate as evaluate_mod
        from perflab.optimizers.phases import finalize as finalize_mod
        from perflab.optimizers.phases import generate as generate_mod
        from perflab.optimizers.phases import prescreen as prescreen_mod
        from perflab.task_spec import TaskSpec

        (tmp_workspace / "tests.py").write_text("print('original tests')\n", encoding="utf-8")
        (tmp_workspace / "bench.py").write_text("print('original bench')\n", encoding="utf-8")
        (tmp_workspace / "algo.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        task = TaskSpec.load(sample_task_yaml)
        ws = task.workspace
        original_tests = (ws / "tests.py").read_text(encoding="utf-8")

        def fake_baseline(ctx):
            ctx.baseline_val = 10.0
            ctx.best_value = 10.0

        block = SearchReplaceBlock("algo.py", "return 1", "return 2")

        def fake_generate(ctx):
            return SimpleNamespace(
                llm_failed=False, candidate_blocks=[[block]],
                candidate_reasoning=[""], generation_errors=[],
            )

        def fake_prescreen(ctx, candidate_blocks, candidate_reasoning):
            return [{"ci": 0, "blocks": candidate_blocks[0], "reasoning": "", "passed": True}]

        def fake_eval_single(ctx, ci, blocks, reasoning, use_fast):
            return evaluate_mod.BeamCandidate(
                iteration=ctx.iteration, index=ci, blocks=blocks,
                description="candidate 1", reasoning=reasoning, value=20.0,
            ), []

        def fake_accept_best(ctx, candidates, use_fast):
            # Simulate a hostile accepted candidate rewriting tests.py at
            # runtime -- exactly the scenario the mid-iteration guard exists
            # to catch -- before the tamper check gets a chance to run.
            (ws / "tests.py").write_text("POISONED\n", encoding="utf-8")
            return True, 0.5, 20.0

        autotune_saw_tests_content: list[str] = []

        def fake_autotune_run(ctx):
            # By the time autotune's sweep trials would start, the mid-
            # iteration verify call must already have restored tests.py.
            autotune_saw_tests_content.append((ws / "tests.py").read_text(encoding="utf-8"))

        def fake_reprofile(ctx, accepted_value):
            pass

        finalize_calls: list[str] = []

        monkeypatch.setattr(baseline_mod, "run", fake_baseline)
        monkeypatch.setattr(generate_mod, "run", fake_generate)
        monkeypatch.setattr(prescreen_mod, "run", fake_prescreen)
        monkeypatch.setattr(evaluate_mod, "evaluate_single_candidate", fake_eval_single)
        monkeypatch.setattr(evaluate_mod, "accept_best", fake_accept_best)
        monkeypatch.setattr(autotune_mod, "run", fake_autotune_run)
        monkeypatch.setattr(evaluate_mod, "reprofile_after_accept", fake_reprofile)
        monkeypatch.setattr(
            finalize_mod, "run",
            lambda ctx, status="completed": finalize_calls.append(status),
        )

        run_agent(
            task, sample_task_yaml,
            AgentConfig(max_iters=1, early_stop=False),
            LLMConfig(provider="ollama", model="test"),
            progress=_SilentProgress(),
            provider=SimpleNamespace(),
        )

        assert autotune_saw_tests_content == [original_tests]
        assert (ws / "tests.py").read_text(encoding="utf-8") == original_tests
        assert finalize_calls == ["completed"]
