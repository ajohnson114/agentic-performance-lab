"""Adversarial regression tests for the harness runtime/validation helpers.

Every test here corresponds to an attack that the helper *did not* catch (or
caught with the wrong exception type) before the July 2026 audit of
tensor_check / gpu_sync / thread_guard / tolerance. The theme throughout is
that a check which silently passes is worse than no check: each case below is
a value or a situation that used to earn a green light it had not proved.
"""
from __future__ import annotations

import concurrent.futures
import sys
import threading
import time
import types
from collections import namedtuple
from unittest.mock import patch

import pytest

from perflab.harness.gpu_sync import SyncTimer, cuda_sync_guard
from perflab.harness.tensor_check import assert_real_array, assert_real_tensor
from perflab.harness.thread_guard import ThreadGuard
from perflab.harness.tolerance import env_accuracy_tolerance

# ---------------------------------------------------------------------------
# A. assert_real_array -- torch
# ---------------------------------------------------------------------------

class TestTorchMaterialization:
    def test_sparse_tensor_raises_assertion_not_notimplemented(self):
        """Sparse layouts cannot be inspected: untyped_storage() raises
        NotImplementedError from inside torch, which used to escape the check
        entirely rather than failing it."""
        torch = pytest.importorskip("torch")
        dense = torch.zeros(4, 4)
        dense[0, 0] = 1.0
        for sparse in (dense.to_sparse(), dense.to_sparse_csr()):
            with pytest.raises(AssertionError, match="not a dense torch.strided"):
                assert_real_tensor(sparse)

    def test_dense_strided_tensor_still_passes(self):
        torch = pytest.importorskip("torch")
        assert_real_tensor(torch.randn(3, 4))
        assert_real_array(torch.randn(3, 4))
        assert_real_array(torch.randn(8, 8)[::2, ::2])   # non-contiguous view
        assert_real_array(torch.empty(0))                # zero-element
        assert_real_array(torch.tensor(3.0))             # 0-d
        assert_real_array(torch.randn(3, requires_grad=True) * 2)

    def test_meta_tensor_rejected(self):
        """A meta tensor is shape-only and free to allocate -- the cheapest
        possible way to return a "result" of the right shape."""
        torch = pytest.importorskip("torch")
        with pytest.raises(AssertionError, match="null data pointer"):
            assert_real_array(torch.empty(1024, 1024, device="meta"))

    def test_nested_tensor_keeps_its_own_diagnostic(self):
        torch = pytest.importorskip("torch")
        nested = torch.nested.nested_tensor([torch.randn(2), torch.randn(3)])
        with pytest.raises(AssertionError, match="nested"):
            assert_real_tensor(nested)


# ---------------------------------------------------------------------------
# A. assert_real_array -- numpy dtypes
# ---------------------------------------------------------------------------

class TestNumpyDtypeGuards:
    def test_structured_dtype_with_object_field_rejected(self):
        """`dtype == object` is False for a structured dtype that still stores
        Python objects per field, so this array full of unevaluated proxies
        used to sail through."""
        np = pytest.importorskip("numpy")
        holder = np.zeros(4, dtype=[("x", object)])
        with pytest.raises(AssertionError, match="dtype=object"):
            assert_real_array(holder)

    def test_plain_object_dtype_still_rejected(self):
        np = pytest.importorskip("numpy")
        with pytest.raises(AssertionError, match="dtype=object"):
            assert_real_array(np.empty(2, dtype=object))

    # timedelta64 is deliberately absent: numpy classes it under
    # np.signedinteger, so it reads as numeric here. It is raw int64 storage
    # with no way to smuggle an unevaluated object, so it is harmless.
    @pytest.mark.parametrize("dtype", ["<U4", "V8", "datetime64[s]"])
    def test_non_numeric_dtypes_rejected(self, dtype):
        np = pytest.importorskip("numpy")
        with pytest.raises(AssertionError, match="non-numeric dtype"):
            assert_real_array(np.zeros(3, dtype=dtype))

    def test_non_numeric_numpy_scalar_rejected(self):
        np = pytest.importorskip("numpy")
        with pytest.raises(AssertionError, match="non-numeric dtype"):
            assert_real_array(np.str_("not a number"))

    @pytest.mark.parametrize(
        "dtype", ["float16", "float32", "float64", "int8", "int64", "uint8",
                  "bool", "complex64"]
    )
    def test_numeric_dtypes_still_pass(self, dtype):
        np = pytest.importorskip("numpy")
        assert_real_array(np.zeros(3, dtype=dtype))

    def test_numpy_scalars_and_views_still_pass(self):
        np = pytest.importorskip("numpy")
        assert_real_array(np.float64(1.0))
        assert_real_array(np.array(3.0))                       # 0-d
        assert_real_array(np.zeros(0))                         # empty
        assert_real_array(np.arange(16.0).reshape(4, 4)[::2])  # non-contiguous
        assert_real_array(np.broadcast_to(np.zeros(1), (64, 64)))


# ---------------------------------------------------------------------------
# A. assert_real_array -- sequences
# ---------------------------------------------------------------------------

class TestSequenceGuards:
    def test_lazy_list_subclass_rejected(self):
        """The sequence-shaped lazy proxy: a list subclass that computes in
        __eq__. Left empty it also had nothing for the element loop to
        inspect, so the check passed having verified precisely nothing."""
        class LazyList(list):
            def __eq__(self, other):  # the real work would happen here
                return True

            __hash__ = None

        for value in (LazyList(), LazyList([1.0, 2.0])):
            with pytest.raises(AssertionError, match="overrides __eq__"):
                assert_real_array(value)

    def test_lazy_tuple_subclass_rejected(self):
        class LazyTuple(tuple):
            def __iter__(self):
                return iter([1.0])

        with pytest.raises(AssertionError, match="overrides __iter__"):
            assert_real_array(LazyTuple((1.0,)))

    def test_namedtuple_of_values_still_passes(self):
        """A namedtuple leaves every comparison operator inherited from tuple,
        so it is a labelled tuple and not a proxy. torch's return_types struct
        sequences have the same shape."""
        pair = namedtuple("pair", "values indices")
        assert_real_array(pair(1.0, 2.0))
        assert_real_array(pair([1.0, 2.0], [3.0, 4.0]))

    def test_plain_sequences_still_pass(self):
        assert_real_array([1.0, 2.0])
        assert_real_array((1.0, 2))
        assert_real_array([[[1.0, 2.0]], [[3.0, 4.0]]])

    def test_self_referential_sequence_fails_as_assertion(self):
        """Used to blow the stack with RecursionError, which is not an
        AssertionError and so escaped every caller's check-failure handler."""
        cycle: list = []
        cycle.append(cycle)
        with pytest.raises(AssertionError, match="nests more than"):
            assert_real_array(cycle)


# ---------------------------------------------------------------------------
# A. assert_real_array -- JAX (structural; jax need not be installed)
# ---------------------------------------------------------------------------

def _jax_stand_in():
    """A module whose class graph mirrors real JAX.

    Real shape: ``jax.Array`` is the abstract base; ``jax._src.core.Tracer``
    derives from it (class __name__ "Tracer"), and every concrete tracer
    (DynamicJaxprTracer, ...) derives from Tracer; ``ArrayImpl`` is the
    materialized array.
    """
    module = types.ModuleType("jax")

    class Array:
        pass

    class Tracer(Array):
        __module__ = "jax"

    module.Array = Array
    module.Tracer = Tracer
    return module


class TestJaxStructuralGuards:
    def _classes(self):
        jax = _jax_stand_in()

        class DynamicJaxprTracer(jax.Tracer):
            __module__ = "jax"
            shape = (128, 128)

            def block_until_ready(self):
                return self

        class ArrayImpl(jax.Array):
            __module__ = "jaxlib"
            shape = (128, 128)
            deleted = False

            def is_deleted(self):
                return self.deleted

            def block_until_ready(self):
                return self

        class ShapeDtypeStruct:  # jax.eval_shape output: shape, no data
            __module__ = "jax"
            shape = (128, 128)

        return jax, DynamicJaxprTracer, ArrayImpl, ShapeDtypeStruct

    def test_tracer_rejected(self):
        jax, tracer_cls, _, _ = self._classes()
        with patch.dict(sys.modules, {"jax": jax}):
            with pytest.raises(AssertionError, match="traced abstract value"):
                assert_real_array(tracer_cls())

    def test_tracer_rejected_even_if_jax_module_absent(self):
        _, tracer_cls, _, _ = self._classes()
        with patch.dict(sys.modules, {}):
            sys.modules.pop("jax", None)
            with pytest.raises(AssertionError, match="traced abstract value"):
                assert_real_array(tracer_cls())

    def test_shape_dtype_struct_rejected(self):
        jax, _, _, sds_cls = self._classes()
        with patch.dict(sys.modules, {"jax": jax}):
            with pytest.raises(AssertionError, match="not a jax.Array"):
                assert_real_array(sds_cls())

    def test_deleted_buffer_rejected(self):
        jax, _, array_cls, _ = self._classes()
        deleted = array_cls()
        deleted.deleted = True
        with patch.dict(sys.modules, {"jax": jax}):
            with pytest.raises(AssertionError, match="deleted JAX buffer"):
                assert_real_array(deleted)

    def test_concrete_array_passes_and_is_blocked_on(self):
        jax, _, array_cls, _ = self._classes()
        blocked = []

        class Tracked(array_cls):
            def block_until_ready(self):
                blocked.append(True)
                return self

        with patch.dict(sys.modules, {"jax": jax}):
            assert_real_array(Tracked())
        assert blocked == [True]


# ---------------------------------------------------------------------------
# B. SyncTimer / cuda_sync_guard
# ---------------------------------------------------------------------------

class TestSyncTimerState:
    def test_stop_without_start_raises(self):
        """perf_counter counts from an arbitrary epoch, so the old 0.0 default
        made an unstarted timer report ~10^5 seconds as a measurement."""
        with pytest.raises(RuntimeError, match="without a matching start"):
            SyncTimer().stop()

    def test_elapsed_is_never_negative_and_is_monotonic(self):
        timer = SyncTimer()
        for _ in range(200):
            timer.start()
            assert timer.stop() >= 0.0

        timer.start()
        first = timer.stop()
        time.sleep(0.002)
        second = timer.stop()      # same window, later end
        assert second > first

    def test_reuse_after_exception_in_timed_region(self):
        timer = SyncTimer()
        timer.start()
        with pytest.raises(ValueError):
            raise ValueError("kernel blew up")
        # State must be usable again, and must not carry the aborted window.
        timer.start()
        time.sleep(0.01)
        assert 0.005 < timer.stop() < 2.0

    def test_nested_timers_measure_their_own_windows(self):
        outer, inner = SyncTimer(), SyncTimer()
        outer.start()
        time.sleep(0.01)
        inner.start()
        time.sleep(0.01)
        inner_elapsed = inner.stop()
        outer_elapsed = outer.stop()
        assert inner_elapsed >= 0.005
        assert outer_elapsed > inner_elapsed

    def test_cuda_sync_guard_syncs_on_exception_path(self):
        """A kernel that raises can still have launched async work; leaving it
        in flight lets it land inside the next measurement."""
        import perflab.harness.gpu_sync as gpu_sync

        calls = []
        with patch.object(gpu_sync, "_sync_device", calls.append):
            with pytest.raises(RuntimeError):
                with cuda_sync_guard("mps"):
                    raise RuntimeError("boom")
        assert calls == ["mps", "mps"]

    def test_cuda_sync_guard_syncs_around_clean_block(self):
        import perflab.harness.gpu_sync as gpu_sync

        calls = []
        with patch.object(gpu_sync, "_sync_device", calls.append):
            with cuda_sync_guard(None):
                pass
        assert calls == [None, None]


@pytest.mark.skipif(
    "not __import__('importlib').util.find_spec('torch')",
    reason="torch not installed",
)
class TestSyncTimerOnRealAccelerator:
    def test_device_none_drains_live_mps(self):
        """The Apple-Silicon mis-timing, on real hardware: MPS dispatch is
        asynchronous, so an unsynced clock stops before the work does."""
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("no MPS device")

        a = torch.randn(1024, 1024, device="mps")
        b = torch.randn(1024, 1024, device="mps")
        for _ in range(3):
            a @ b
        torch.mps.synchronize()

        unsynced, synced = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            a @ b
            unsynced.append(time.perf_counter() - t0)
            torch.mps.synchronize()

            timer = SyncTimer()          # no device: the natural call
            timer.start()
            a @ b
            synced.append(timer.stop())

        # The un-synced clock reports a fraction of the real cost; the timer
        # with no device argument must not.
        assert min(synced) > 2 * max(unsynced)

    def test_cpu_only_task_is_not_penalized(self):
        """stop() syncs inside the measured window, so the liveness probes
        must not drag an accelerator the task never touched into it."""
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available():
            pytest.skip("live CUDA context makes this measurement device-bound")

        overheads = []
        for _ in range(200):
            timer = SyncTimer()
            timer.start()
            overheads.append(timer.stop())
        overheads.sort()
        median = overheads[len(overheads) // 2]
        # Measured ~0.8 us on an M4 with MPS available but unused. The bound is
        # loose enough for a loaded CI box while still failing if the probes
        # start issuing a real device barrier per call.
        assert median < 1e-3, f"empty SyncTimer window cost {median * 1e6:.1f} us"


# ---------------------------------------------------------------------------
# C. ThreadGuard
# ---------------------------------------------------------------------------

class TestThreadGuardIdentity:
    def test_net_zero_churn_is_caught(self):
        """Kill one pre-existing thread while starting a worker and the
        population count is unchanged -- a count-only guard reports a clean
        run while the smuggled worker keeps computing."""
        stop_victim, stop_worker = threading.Event(), threading.Event()
        victim = threading.Thread(
            target=stop_victim.wait, daemon=True, name="victim")
        victim.start()
        try:
            guard = ThreadGuard()
            baseline = guard.snapshot()

            worker = threading.Thread(
                target=stop_worker.wait, daemon=True, name="smuggled")
            worker.start()
            stop_victim.set()
            victim.join(timeout=5)
            assert not victim.is_alive()
            # The scenario is only meaningful if the count really is unchanged.
            assert threading.active_count() == baseline

            try:
                with pytest.raises(AssertionError, match="smuggled"):
                    guard.check()
            finally:
                stop_worker.set()
                worker.join(timeout=5)
        finally:
            stop_victim.set()
            victim.join(timeout=5)

    def test_thread_named_like_an_existing_one_is_caught(self):
        """Names are not identities: nothing stops a candidate naming its
        worker MainThread to empty out the reported set."""
        stop = threading.Event()
        guard = ThreadGuard()
        guard.snapshot()
        worker = threading.Thread(target=stop.wait, daemon=True, name="MainThread")
        worker.start()
        try:
            with pytest.raises(AssertionError, match="Thread injection detected"):
                guard.check()
        finally:
            stop.set()
            worker.join(timeout=5)

    def test_check_before_snapshot_says_so(self):
        with pytest.raises(AssertionError, match="before snapshot"):
            ThreadGuard().check()

    def test_daemon_offload_is_caught_and_named(self):
        stop = threading.Event()
        guard = ThreadGuard()
        guard.snapshot()
        worker = threading.Thread(target=stop.wait, daemon=True, name="offload")
        worker.start()
        try:
            with pytest.raises(AssertionError, match="offload.*daemon"):
                guard.check()
        finally:
            stop.set()
            worker.join(timeout=5)

    def test_tolerance_still_allows_declared_thread_count(self):
        stop = threading.Event()
        guard = ThreadGuard(tolerance=1)
        guard.snapshot()
        worker = threading.Thread(target=stop.wait, daemon=True)
        worker.start()
        try:
            guard.check()
        finally:
            stop.set()
            worker.join(timeout=5)


class TestThreadGuardNoFalsePositives:
    def test_pool_created_and_shut_down_inside_the_window(self):
        guard = ThreadGuard()
        guard.snapshot()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(lambda i: i * 2, range(8))) == [
                i * 2 for i in range(8)
            ]
        guard.check()

    def test_threads_created_and_joined_inside_the_window(self):
        guard = ThreadGuard()
        guard.snapshot()
        workers = [threading.Thread(target=lambda: sum(range(1000)))
                   for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        guard.check()

    def test_blas_worker_threads_on_first_matmul(self):
        """BLAS spawns its pool lazily on the first matmul. Those threads live
        below Python, so this must not read as injection."""
        np = pytest.importorskip("numpy")
        guard = ThreadGuard()
        guard.snapshot()
        matrix = np.random.rand(512, 512)
        assert float((matrix @ matrix).sum()) != 0.0
        guard.check()

    def test_thread_delta_is_zero_right_after_snapshot(self):
        guard = ThreadGuard()
        guard.snapshot()
        assert guard.thread_delta == 0


# ---------------------------------------------------------------------------
# D. env_accuracy_tolerance
# ---------------------------------------------------------------------------

class TestAccuracyToleranceParsing:
    @pytest.mark.parametrize(
        "raw", ["inf", "-inf", "Infinity", "INF", "1e400", "nan", "NaN", "-1",
                "-1e-3"]
    )
    def test_unusable_tolerances_fall_back_to_default(self, raw, monkeypatch):
        """atol=inf makes allclose True for every pair of arrays, i.e. accuracy
        checking switches itself off. float() reaches inf from more spellings
        than it looks -- including any overflowing literal like 1e400."""
        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", raw)
        with pytest.warns(RuntimeWarning, match="not a usable tolerance"):
            assert env_accuracy_tolerance(1e-5) == 1e-5

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1e-3", 1e-3),
            (" 1e-3 ", 1e-3),
            ("1e-3\n", 1e-3),
            ("0", 0.0),
            ("0.0", 0.0),
            ("+1e-3", 1e-3),
            (".5", 0.5),
            ("exact", 0.0),
            ("EXACT", 0.0),
            (" Exact ", 0.0),
            ("garbage", 1e-5),
            ("", 1e-5),
            ("   ", 1e-5),
            ("1,5", 1e-5),
            ("1e-3 # comment", 1e-5),
        ],
    )
    def test_accepted_and_ignored_spellings(self, raw, expected, monkeypatch):
        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", raw)
        assert env_accuracy_tolerance(1e-5) == expected

    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("PERFLAB_ACCURACY_TOLERANCE", raising=False)
        assert env_accuracy_tolerance(1e-5) == 1e-5
        assert env_accuracy_tolerance(1e-2) == 1e-2

    def test_zero_tolerance_is_honored_not_treated_as_missing(self, monkeypatch):
        """0.0 is falsy; it must still mean "bit-exact", not "unset"."""
        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "0")
        assert env_accuracy_tolerance(1e-5) == 0.0
