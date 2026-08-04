"""Backend-agnostic harness tests.

perflab.harness used to run every real check behind an
``isinstance(x, torch.Tensor)`` guard, so a numpy, JAX or list-of-floats task
could call `assert_deterministic` and have Phases 2 and 3 *pass without
checking anything*. These tests pin the fix:

  * the same gaming patterns (stale buffer, no-op kernel, wrong answer,
    identity-keyed cache, precision downgrade) are caught for numpy, nested
    Python lists and torch alike;
  * an output type the harness cannot inspect RAISES rather than passing;
  * importing perflab.harness needs no optional dependency, and the pure
    Python paths keep working with numpy absent.

numpy and torch are optional (CI's test job installs neither), so both are
importorskip-guarded. JAX is never installed here, so its paths are exercised
with stand-in objects whose ``__module__`` puts them on the JAX branch.
"""
from __future__ import annotations

import importlib
import math
import random
import struct
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from perflab.harness import _array

# ---------------------------------------------------------------------------
# Backend kits: one per backend, exposing the same tiny vocabulary so a single
# set of gaming scenarios can be run against all of them.
# ---------------------------------------------------------------------------

N = 16
BIG = 600


class ListKit:
    """Nested Python sequences -- the backend with no dependencies at all."""

    name = "list"

    def vector(self, n=N, scale=1.0):
        return [random.gauss(0.0, 1.0) * scale for _ in range(n)]

    def inputs(self):
        return (self.vector(), self.vector())

    def honest(self, a, b):
        return [x * y for x, y in zip(a, b, strict=True)]

    def offset(self, out, delta):
        return [x + delta for x in out]

    def copy(self, out):
        return list(out)

    def const(self):
        return [float(i) for i in range(N)]

    def identity_key(self, a):
        return id(a)

    def to_fp16(self, values):
        return [struct.unpack("<e", struct.pack("<e", v))[0] for v in values]


class NumpyKit:
    name = "numpy"

    def __init__(self):
        self.np = pytest.importorskip("numpy")
        self.rng = self.np.random.default_rng(20260802)

    def vector(self, n=N, scale=1.0):
        return self.rng.standard_normal(n) * scale

    def inputs(self):
        return (self.vector(), self.vector())

    def honest(self, a, b):
        return a * b

    def offset(self, out, delta):
        return out + delta

    def copy(self, out):
        return out.copy()

    def const(self):
        return self.np.arange(N, dtype=self.np.float64)

    def identity_key(self, a):
        return a.ctypes.data

    def to_fp16(self, values):
        return values.astype(self.np.float16).astype(self.np.float32)


class TorchKit:
    name = "torch"

    def __init__(self):
        self.torch = pytest.importorskip("torch")

    def vector(self, n=N, scale=1.0):
        return self.torch.randn(n) * scale

    def inputs(self):
        return (self.vector(), self.vector())

    def honest(self, a, b):
        return a * b

    def offset(self, out, delta):
        return out + delta

    def copy(self, out):
        return out.detach().clone()

    def const(self):
        return self.torch.arange(N, dtype=self.torch.float32)

    def identity_key(self, a):
        return a.data_ptr()

    def to_fp16(self, values):
        return values.half().float()


@pytest.fixture(params=["list", "numpy", "torch"])
def kit(request):
    return {"list": ListKit, "numpy": NumpyKit, "torch": TorchKit}[request.param]()


# ---------------------------------------------------------------------------
# 1. Backend detection
# ---------------------------------------------------------------------------

class TestBackendDetection:
    def test_python_values(self):
        assert _array.backend_of([1.0, 2.0]) == _array.SEQUENCE
        assert _array.backend_of((1.0, 2.0)) == _array.SEQUENCE
        assert _array.backend_of(1.5) == _array.SCALAR
        assert _array.backend_of(3) == _array.SCALAR
        assert _array.backend_of(True) == _array.SCALAR

    def test_numpy_by_module_name(self):
        np = pytest.importorskip("numpy")
        assert _array.backend_of(np.zeros(3)) == _array.NUMPY
        assert _array.backend_of(np.float64(1.0)) == _array.NUMPY

    def test_torch_by_module_name(self):
        torch = pytest.importorskip("torch")
        assert _array.backend_of(torch.zeros(3)) == _array.TORCH

    def test_torch_subclass_detected_via_loaded_module(self):
        # A subclass reports its own defining module, so the module-name test
        # misses it; the isinstance fallback against the loaded torch catches it.
        torch = pytest.importorskip("torch")

        class Sneaky(torch.Tensor):
            pass

        assert Sneaky.__module__ != "torch"
        assert _array.backend_of(Sneaky(torch.zeros(3))) == _array.TORCH

    def test_unknown_types(self):
        class Weird:
            pass

        assert _array.backend_of(Weird()) == _array.UNKNOWN
        assert _array.backend_of({"a": 1}) == _array.UNKNOWN
        assert _array.backend_of("text") == _array.UNKNOWN
        assert _array.backend_of(None) == _array.UNKNOWN

    def test_require_backend_raises_with_type_name(self):
        class Weird:
            pass

        with pytest.raises(_array.UnsupportedValueError, match="Weird"):
            _array.require_backend(Weird(), "output")

    def test_unsupported_error_is_also_an_assertion_error(self):
        # Harness callers catch AssertionError; an unsupported type must reach
        # them as a check failure rather than an unrelated crash.
        assert issubclass(_array.UnsupportedValueError, AssertionError)
        assert issubclass(_array.UnsupportedValueError, TypeError)

    def test_is_array(self):
        assert _array.is_array([1.0])
        assert not _array.is_array(1.0)
        assert not _array.is_array(object())


# ---------------------------------------------------------------------------
# 2. Determinism -- all three phases, every backend
# ---------------------------------------------------------------------------

class TestDeterminismAcrossBackends:
    def test_honest_implementation_passes(self, kit):
        from perflab.harness import assert_deterministic

        assert_deterministic(
            fn=kit.honest,
            input_factory=kit.inputs,
            reference_fn=kit.honest,
            n_runs=3,
        )

    def test_nondeterministic_implementation_caught(self, kit):
        """Phase 1: the same inputs must give the same answer."""
        from perflab.harness import assert_deterministic

        calls = {"n": 0}

        def unstable(a, b):
            calls["n"] += 1
            return kit.offset(kit.honest(a, b), 1000.0 * (calls["n"] % 2))

        with pytest.raises(AssertionError, match="Non-deterministic"):
            assert_deterministic(fn=unstable, input_factory=kit.inputs, n_runs=4)

    def test_stale_shared_buffer_caught(self, kit):
        """Phase 1 again, via the specific hazard: a kernel that hands back the
        same buffer every call. Without the snapshot in detached_copy, all runs
        would alias one object and compare equal to themselves."""
        from perflab.harness import assert_deterministic

        buffer = kit.const()
        calls = {"n": 0}

        def reuses_buffer(a, b):
            calls["n"] += 1
            fresh = kit.offset(kit.honest(a, b), float(calls["n"]))
            _array.fill_from(buffer, fresh)
            return buffer

        with pytest.raises(AssertionError, match="Non-deterministic"):
            assert_deterministic(fn=reuses_buffer, input_factory=kit.inputs, n_runs=3)

    def test_noop_kernel_caught(self, kit):
        """Phase 2: this is the phase that silently passed for non-torch."""
        from perflab.harness import assert_deterministic

        fixed = kit.const()

        def noop(a, b):
            return kit.copy(fixed)

        with pytest.raises(AssertionError, match="No-op kernel detected"):
            assert_deterministic(fn=noop, input_factory=kit.inputs)

    def test_wrong_answer_vs_reference_caught(self, kit):
        """Phase 3: also silently passed for non-torch outputs."""
        from perflab.harness import assert_deterministic

        with pytest.raises(AssertionError, match="Correctness failure against reference"):
            assert_deterministic(
                fn=lambda a, b: kit.offset(kit.honest(a, b), 0.5),
                input_factory=kit.inputs,
                reference_fn=kit.honest,
            )

    def test_unsupported_output_raises_instead_of_passing(self, kit):
        """The core regression: an output the harness cannot read must fail."""
        from perflab.harness import assert_deterministic

        class Opaque:
            """Looks correct to any check that only compares with ==."""

            def __eq__(self, other):
                return True

            def __hash__(self):
                return 0

        with pytest.raises(AssertionError, match="cannot inspect"):
            assert_deterministic(fn=lambda a, b: Opaque(), input_factory=kit.inputs)

    def test_tolerance_split_holds_for_every_backend(self, kit, monkeypatch):
        # A loose task tolerance must not loosen the Phase-1 reproducibility
        # check; it does apply to the Phase-3 reference comparison.
        from perflab.harness import assert_deterministic

        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "1e-1")
        calls = {"n": 0}

        def drifting(a, b):
            calls["n"] += 1
            return kit.offset(kit.honest(a, b), 1e-3 * (calls["n"] % 2))

        with pytest.raises(AssertionError, match="Non-deterministic"):
            assert_deterministic(fn=drifting, input_factory=kit.inputs, n_runs=2)

        # ... while a 1e-3 divergence from the reference is inside 1e-1.
        assert_deterministic(
            fn=lambda a, b: kit.offset(kit.honest(a, b), 1e-3),
            input_factory=kit.inputs,
            reference_fn=kit.honest,
            n_runs=1,
        )

    def test_empty_input_factory_rejected(self, kit):
        from perflab.harness import assert_deterministic

        with pytest.raises(ValueError, match="no inputs"):
            assert_deterministic(fn=lambda: kit.const(), input_factory=tuple)


# ---------------------------------------------------------------------------
# 3. Pointer poisoning -- every backend
# ---------------------------------------------------------------------------

class TestPointerPoisonAcrossBackends:
    def test_honest_implementation_passes(self, kit):
        from perflab.harness import assert_no_memoization

        assert_no_memoization(
            fn=kit.honest,
            input_factory=kit.inputs,
            reference_fn=kit.honest,
            atol=1e-6,
        )

    def test_identity_keyed_cache_caught(self, kit):
        from perflab.harness import assert_no_memoization

        cache: dict = {}

        def memoizing(a, b):
            key = kit.identity_key(a)
            if key not in cache:
                cache[key] = kit.copy(kit.honest(a, b))
            return cache[key]

        with pytest.raises(AssertionError, match="Memoization detected"):
            assert_no_memoization(
                fn=memoizing,
                input_factory=kit.inputs,
                reference_fn=kit.honest,
                atol=1e-6,
            )

    def test_fill_from_preserves_object_identity(self, kit):
        original = kit.inputs()[0]
        key_before = kit.identity_key(original)
        snapshot = kit.copy(original)

        _array.fill_from(original, _array.random_like(original))

        assert kit.identity_key(original) == key_before
        assert not _array.exact_equal(original, snapshot)

    def test_initial_correctness_failure_reported(self, kit):
        from perflab.harness import assert_no_memoization

        with pytest.raises(AssertionError, match="Initial correctness check failed"):
            assert_no_memoization(
                fn=lambda a, b: kit.offset(kit.honest(a, b), 1.0),
                input_factory=kit.inputs,
                reference_fn=kit.honest,
                atol=1e-6,
            )

    def test_unsupported_output_raises(self, kit):
        from perflab.harness import assert_no_memoization

        with pytest.raises(AssertionError, match="cannot inspect"):
            assert_no_memoization(
                fn=lambda a, b: object(),
                input_factory=kit.inputs,
                reference_fn=kit.honest,
            )

    def test_no_mutable_input_is_an_error_not_a_pass(self):
        """A poison check with nothing to poison proves nothing, so it fails."""
        from perflab.harness import assert_no_memoization

        with pytest.raises(AssertionError, match="cannot run"):
            assert_no_memoization(
                fn=lambda a, b: a * b,
                input_factory=lambda: (2.0, 3.0),  # immutable scalars
                reference_fn=lambda a, b: a * b,
            )


# ---------------------------------------------------------------------------
# 4. ULP precision -- every backend
# ---------------------------------------------------------------------------

class TestPrecisionAcrossBackends:
    def test_identical_values_pass(self, kit):
        from perflab.harness import assert_ulp_close

        values = kit.vector(BIG)
        stats = assert_ulp_close(values, kit.copy(values), max_ulp=0)
        assert stats["max_ulp_observed"] == 0
        assert stats["n_samples"] == BIG

    def test_fp16_downgrade_caught(self, kit):
        from perflab.harness import assert_ulp_close

        reference = kit.vector(BIG, scale=100.0)
        degraded = kit.to_fp16(reference)

        with pytest.raises(AssertionError, match="Precision downgrade"):
            assert_ulp_close(degraded, reference, max_ulp=2)

    def test_shape_mismatch_caught(self, kit):
        from perflab.harness import assert_ulp_close

        with pytest.raises(AssertionError, match="Shape mismatch"):
            assert_ulp_close(kit.vector(8), kit.vector(9))

    def test_sampling_path_used_for_large_inputs(self, kit):
        from perflab.harness import assert_ulp_close

        values = kit.vector(3000)
        stats = assert_ulp_close(values, kit.copy(values), max_ulp=0,
                                 sample_fraction=0.01, min_samples=100)
        assert stats["n_samples"] == 100
        assert stats["max_ulp_observed"] == 0

    def test_dtype_name_expectation(self, kit):
        from perflab.harness import assert_ulp_close

        values = kit.vector(32)
        # dtype names work as the expectation on every backend.
        assert_ulp_close(values, kit.copy(values), max_ulp=0,
                         expected_dtype=_array.dtype_name(values))
        with pytest.raises(AssertionError, match="Precision downgrade.*dtype"):
            assert_ulp_close(values, kit.copy(values), expected_dtype="float8")

    def test_unsupported_input_raises(self, kit):
        from perflab.harness import assert_ulp_close

        with pytest.raises(AssertionError, match="cannot inspect"):
            assert_ulp_close(object(), kit.vector(8))

    def test_torch_dtype_object_still_accepted(self):
        """Back-compat: expected_dtype=torch.float32 keeps working."""
        torch = pytest.importorskip("torch")
        from perflab.harness import assert_ulp_close

        values = torch.randn(64)
        assert_ulp_close(values, values.clone(), max_ulp=0,
                         expected_dtype=torch.float32)
        with pytest.raises(AssertionError, match="Precision downgrade.*dtype"):
            assert_ulp_close(values.half(), values, expected_dtype=torch.float32)


# ---------------------------------------------------------------------------
# 5. assert_real_array
# ---------------------------------------------------------------------------

class TestAssertRealArray:
    def test_plain_values_pass(self, kit):
        from perflab.harness import assert_real_array

        assert_real_array(kit.vector(8))

    def test_nested_sequences_pass(self):
        from perflab.harness import assert_real_array

        assert_real_array([[1.0, 2.0], [3.0, 4.0]])
        assert_real_array((1.0, 2))

    def test_unknown_type_rejected(self):
        from perflab.harness import assert_real_array

        with pytest.raises(AssertionError, match="cannot inspect"):
            assert_real_array(object())

    def test_lazy_float_subclass_rejected(self):
        from perflab.harness import assert_real_array

        class LazyFloat(float):
            def __eq__(self, other):  # would compute on demand
                return True

            def __hash__(self):
                return 0

        with pytest.raises(AssertionError, match="Lazy evaluation detected"):
            assert_real_array(LazyFloat(1.0))

    def test_lazy_element_inside_sequence_rejected(self):
        from perflab.harness import assert_real_array

        class LazyFloat(float):
            pass

        with pytest.raises(AssertionError, match=r"output\[1\]"):
            assert_real_array([1.0, LazyFloat(2.0)])

    def test_numpy_array_passes(self):
        np = pytest.importorskip("numpy")
        from perflab.harness import assert_real_array

        assert_real_array(np.zeros((2, 3)))
        assert_real_array(np.float64(1.0))
        assert_real_array(np.zeros(0))

    def test_numpy_subclass_rejected(self):
        np = pytest.importorskip("numpy")
        from perflab.harness import assert_real_array

        class LazyArray(np.ndarray):
            pass

        lazy = np.zeros(4).view(LazyArray)
        with pytest.raises(AssertionError, match="subclass or proxy"):
            assert_real_array(lazy)

    def test_numpy_object_dtype_rejected(self):
        np = pytest.importorskip("numpy")
        from perflab.harness import assert_real_array

        holder = np.empty(2, dtype=object)
        holder[0] = object()
        with pytest.raises(AssertionError, match="dtype=object"):
            assert_real_array(holder)

    def test_torch_path_unchanged(self):
        torch = pytest.importorskip("torch")
        from perflab.harness import assert_real_array, assert_real_tensor

        assert_real_array(torch.randn(3))
        with pytest.raises(AssertionError, match="subclass"):
            class LazyTensor(torch.Tensor):
                pass

            assert_real_array(LazyTensor(torch.randn(3)))
        # The torch-only entry point is untouched and still rejects non-tensors.
        with pytest.raises(AssertionError, match="not torch.Tensor"):
            assert_real_tensor([1, 2, 3])


# ---------------------------------------------------------------------------
# 6. JAX paths (jax is not installed; stand-ins carry jax module names)
# ---------------------------------------------------------------------------

class FakeJaxArray:
    """Stand-in for jax.Array: its module puts it on the JAX branch."""

    __module__ = "jaxlib.xla_extension"

    def __init__(self, data):
        np = pytest.importorskip("numpy")
        self._data = np.asarray(data, dtype=np.float64)
        self.blocked = 0

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def block_until_ready(self):
        self.blocked += 1
        return self

    def __array__(self, dtype=None, copy=None):
        return self._data if dtype is None else self._data.astype(dtype)


class Tracer:
    """Stands in for jax.core.Tracer, the base every real tracer inherits."""

    __module__ = "jax._src.core"


class FakeTracer(Tracer):
    """Stand-in for a jax tracer: abstract, carries no data."""

    __module__ = "jax._src.interpreters.partial_eval"
    shape = (2, 2)
    dtype = "float32"


class ForeignJaxish(FakeJaxArray):
    """Lives in a jax module but is not an instance of jax.Array."""

    __module__ = "jax._src.array"


def _fake_jax_module(**overrides):
    module = SimpleNamespace(
        Array=FakeJaxArray,
        effects_barrier=MagicMock(),
        block_until_ready=MagicMock(),
        numpy=SimpleNamespace(zeros=lambda *a, **k: FakeJaxArray([0.0])),
    )
    for key, value in overrides.items():
        setattr(module, key, value)
    return module


class TestJaxPaths:
    def test_backend_detection(self):
        pytest.importorskip("numpy")
        assert _array.backend_of(FakeJaxArray([1.0])) == _array.JAX
        assert _array.backend_of(FakeTracer()) == _array.JAX

    def test_tracer_rejected(self):
        from perflab.harness import assert_real_array

        with pytest.raises(AssertionError, match="traced abstract value"):
            assert_real_array(FakeTracer(), name="logits")

    def test_real_array_forces_concreteness(self):
        from perflab.harness import assert_real_array

        array = FakeJaxArray([1.0, 2.0])
        with patch.dict("sys.modules", {"jax": _fake_jax_module()}):
            assert_real_array(array)
        assert array.blocked == 1, "block_until_ready must be called"

    def test_non_jax_array_rejected(self):
        from perflab.harness import assert_real_array

        impostor = ForeignJaxish([1.0])
        foreign_array_cls = type("Array", (), {})
        with patch.dict("sys.modules", {"jax": _fake_jax_module(Array=foreign_array_cls)}):
            with pytest.raises(AssertionError, match="not a jax.Array"):
                assert_real_array(impostor)

    def test_deleted_buffer_rejected(self):
        from perflab.harness import assert_real_array

        array = FakeJaxArray([1.0])
        array.is_deleted = lambda: True
        with pytest.raises(AssertionError, match="deleted JAX buffer"):
            assert_real_array(array)

    def test_determinism_runs_on_jax_values(self):
        from perflab.harness import assert_deterministic

        def honest(a, b):
            return FakeJaxArray(a._data * b._data)

        def factory():
            return (FakeJaxArray([1.0, 2.0, 3.0]), FakeJaxArray([0.5, 0.25, 0.125]))

        assert_deterministic(fn=honest, input_factory=factory, reference_fn=honest)

        constant = FakeJaxArray([9.0, 9.0, 9.0])
        with pytest.raises(AssertionError, match="No-op kernel detected"):
            assert_deterministic(
                fn=lambda a, b: constant,
                input_factory=lambda: (
                    FakeJaxArray([random.random() for _ in range(3)]),
                    FakeJaxArray([random.random() for _ in range(3)]),
                ),
            )

    def test_immutable_inputs_reported_clearly(self):
        from perflab.harness import assert_no_memoization

        array = FakeJaxArray([1.0, 2.0])
        assert not _array.is_fillable(array)
        with pytest.raises(_array.UnsupportedValueError, match="immutable"):
            _array.fill_from(array, FakeJaxArray([3.0, 4.0]))
        with pytest.raises(AssertionError, match="cannot run"):
            assert_no_memoization(
                fn=lambda a: a,
                input_factory=lambda: (FakeJaxArray([1.0]),),
                reference_fn=lambda a: a,
            )

    def test_sync_blocks_jax_array(self):
        array = FakeJaxArray([1.0])
        _array.sync(array)
        assert array.blocked == 1

    def test_sync_drains_live_jax(self):
        module = _fake_jax_module()
        with patch.dict("sys.modules", {"jax": module}):
            _array.sync()
        module.effects_barrier.assert_called_once()

    def test_sync_falls_back_to_block_until_ready(self):
        module = _fake_jax_module(effects_barrier=None)
        with patch.dict("sys.modules", {"jax": module}):
            _array.sync()
        module.block_until_ready.assert_called_once()


# ---------------------------------------------------------------------------
# 7. SyncTimer / device synchronization
# ---------------------------------------------------------------------------

class TestSyncTimerBackends:
    def _mock_torch(self, *, cuda=False, mps=False):
        torch = MagicMock()
        torch.cuda.is_initialized.return_value = cuda
        torch.backends.mps.is_available.return_value = mps
        torch.mps.current_allocated_memory.return_value = 4096 if mps else 0
        return torch

    def test_device_none_syncs_live_mps(self):
        """The Apple-Silicon mis-timing: SyncTimer() used to sync nothing, so
        asynchronous MPS work landed after the clock stopped."""
        from perflab.harness.gpu_sync import SyncTimer

        torch = self._mock_torch(mps=True)
        with patch.dict("sys.modules", {"torch": torch}):
            timer = SyncTimer()
            timer.start()
            timer.stop()
        assert torch.mps.synchronize.call_count == 2
        torch.cuda.synchronize.assert_not_called()

    def test_device_none_syncs_live_cuda(self):
        from perflab.harness.gpu_sync import SyncTimer

        torch = self._mock_torch(cuda=True)
        with patch.dict("sys.modules", {"torch": torch}):
            timer = SyncTimer()
            timer.start()
            timer.stop()
        assert torch.cuda.synchronize.call_count == 2
        torch.mps.synchronize.assert_not_called()

    def test_device_none_is_noop_without_accelerators(self):
        from perflab.harness.gpu_sync import SyncTimer

        torch = self._mock_torch()
        with patch.dict("sys.modules", {"torch": torch}):
            timer = SyncTimer()
            timer.start()
            elapsed = timer.stop()
        torch.cuda.synchronize.assert_not_called()
        torch.mps.synchronize.assert_not_called()
        assert elapsed >= 0

    def test_explicit_cuda_device_unchanged(self):
        from perflab.harness.gpu_sync import SyncTimer

        torch = MagicMock()
        device = MagicMock()
        device.type = "cuda"
        with patch.dict("sys.modules", {"torch": torch}):
            timer = SyncTimer(device=device)
            timer.start()
            timer.stop()
        assert torch.cuda.synchronize.call_count == 2
        torch.cuda.synchronize.assert_called_with(device)

    def test_cpu_device_string_drains_live_backends(self):
        """A JAX-on-CPU task still needs a barrier: dispatch is async there."""
        from perflab.harness.gpu_sync import SyncTimer

        module = _fake_jax_module()
        with patch.dict("sys.modules", {"jax": module}):
            timer = SyncTimer(device="cpu")
            timer.start()
            timer.stop()
        assert module.effects_barrier.call_count == 2

    def test_unused_mps_is_not_synced(self):
        """SyncTimer.stop syncs inside the measured window, so an accelerator
        the task never touched must not add ~9 us to every measurement."""
        from perflab.harness.gpu_sync import SyncTimer

        torch = MagicMock()
        torch.cuda.is_initialized.return_value = False
        torch.backends.mps.is_available.return_value = True
        torch.mps.current_allocated_memory.return_value = 0
        with patch.dict("sys.modules", {"torch": torch}):
            SyncTimer().start()
        torch.mps.synchronize.assert_not_called()

    def test_sync_never_raises_on_backend_errors(self):
        torch = MagicMock()
        torch.cuda.is_initialized.side_effect = RuntimeError("driver blew up")
        torch.backends.mps.is_available.side_effect = RuntimeError("no metal")
        with patch.dict("sys.modules", {"torch": torch}):
            _array.sync()  # must not raise: timing is not a check

    def test_sync_on_plain_values_is_noop(self):
        _array.sync(1.0)
        _array.sync([1.0, 2.0])

    def test_cuda_sync_guard_still_works(self):
        from perflab.harness.gpu_sync import cuda_sync_guard

        with cuda_sync_guard(device=None):
            value = 1 + 1
        assert value == 2


# ---------------------------------------------------------------------------
# 8. Optional-dependency discipline
# ---------------------------------------------------------------------------

class TestNoOptionalDependencies:
    def test_import_works_without_numpy_or_torch(self):
        """perflab.harness must import with zero optional deps installed."""
        with patch.dict("sys.modules", {"numpy": None, "torch": None}):
            for name in [m for m in list(sys.modules) if m.startswith("perflab.harness")]:
                del sys.modules[name]
            harness = importlib.import_module("perflab.harness")
            assert callable(harness.assert_deterministic)
            assert callable(harness.assert_real_array)
            with pytest.raises(ImportError):
                importlib.import_module("numpy")

    def test_python_checks_run_without_numpy(self):
        """A list-of-floats task is fully checkable on a numpy-free install."""
        with patch.dict("sys.modules", {"numpy": None, "torch": None}):
            for name in [m for m in list(sys.modules) if m.startswith("perflab.harness")]:
                del sys.modules[name]
            harness = importlib.import_module("perflab.harness")
            kit = ListKit()

            harness.assert_deterministic(
                fn=kit.honest, input_factory=kit.inputs, reference_fn=kit.honest
            )
            harness.assert_no_memoization(
                fn=kit.honest,
                input_factory=kit.inputs,
                reference_fn=kit.honest,
                atol=1e-9,
            )
            harness.assert_ulp_close(kit.vector(64), kit.vector(64), max_ulp=math.inf)
            harness.assert_real_array([1.0, 2.0])
            with pytest.raises(AssertionError, match="No-op kernel detected"):
                harness.assert_deterministic(
                    fn=lambda a, b: kit.const(), input_factory=kit.inputs
                )

    def test_missing_numpy_error_names_the_extra(self):
        pytest.importorskip("numpy")
        array = _array.to_numpy_f64([1.0, 2.0])  # a real ndarray to feed back in
        with patch.dict("sys.modules", {"numpy": None}):
            with pytest.raises(_array.UnsupportedValueError, match=r"perflab\[tasks-python\]"):
                _array.to_numpy_f64(array)


# ---------------------------------------------------------------------------
# 9. _array primitives
# ---------------------------------------------------------------------------

class TestArrayPrimitives:
    def test_shape_of(self, kit):
        assert _array.shape_of(kit.vector(5)) == (5,)
        assert _array.shape_of([[1.0, 2.0], [3.0, 4.0]]) == (2, 2)
        assert _array.shape_of(1.0) == ()

    def test_ragged_sequence_raises(self):
        with pytest.raises(_array.UnsupportedValueError, match="ragged"):
            _array.shape_of([[1.0, 2.0], [3.0]])

    def test_allclose_reports_max_diff_on_failure(self, kit):
        a = kit.const()
        b = kit.offset(a, 0.25)
        ok, diff = _array.allclose(a, b, atol=1e-6, rtol=0.0)
        assert not ok
        assert diff == pytest.approx(0.25, abs=1e-6)
        ok, diff = _array.allclose(a, kit.copy(a), atol=1e-6, rtol=0.0)
        assert ok and diff == 0.0

    def test_allclose_refuses_to_broadcast(self):
        np = pytest.importorskip("numpy")
        # np.allclose would broadcast these and report success.
        assert np.allclose(np.zeros((4, 1)), np.zeros((4, 4)))
        with pytest.raises(AssertionError, match="Shape mismatch"):
            _array.allclose(np.zeros((4, 1)), np.zeros((4, 4)))

    def test_allclose_across_backends(self):
        np = pytest.importorskip("numpy")
        ok, _ = _array.allclose([1.0, 2.0], np.array([1.0, 2.0]))
        assert ok
        ok, _ = _array.allclose([1.0, 2.0], np.array([1.0, 2.5]))
        assert not ok

    def test_allclose_nan_never_passes(self):
        ok, diff = _array.allclose([float("nan")], [float("nan")])
        assert not ok
        assert math.isinf(diff)

    def test_exact_equal(self, kit):
        a = kit.const()
        assert _array.exact_equal(a, kit.copy(a))
        assert not _array.exact_equal(a, kit.offset(a, 1e-12))
        assert not _array.exact_equal(kit.vector(4), kit.vector(5))

    def test_values_equal_tolerates_exotic_inputs(self):
        # Config-ish inputs are compared, not skipped: equal stays equal...
        assert _array.values_equal("mode", "mode")
        assert not _array.values_equal("mode", "other")

        class Exploding:
            def __eq__(self, other):
                raise RuntimeError("nope")

        # ...and a value that cannot be compared is reported as *changed*,
        # which keeps the no-op detector armed rather than disarming it.
        assert not _array.values_equal(Exploding(), Exploding())

    def test_detached_copy_breaks_aliasing(self, kit):
        original = kit.vector(4)
        snapshot = _array.detached_copy(original)
        _array.fill_from(original, _array.random_like(original))
        assert not _array.exact_equal(original, snapshot)

    def test_fill_from_nested_lists_keeps_row_identity(self):
        rows = [[1.0, 2.0], [3.0, 4.0]]
        row0 = rows[0]
        _array.fill_from(rows, [[9.0, 8.0], [7.0, 6.0]])
        assert rows == [[9.0, 8.0], [7.0, 6.0]]
        assert rows[0] is row0, "inner row objects must keep their identity"

    def test_fill_from_rejects_immutable_targets(self):
        with pytest.raises(_array.UnsupportedValueError, match="immutable"):
            _array.fill_from((1.0, 2.0), (3.0, 4.0))
        with pytest.raises(_array.UnsupportedValueError, match="immutable"):
            _array.fill_from(1.0, 2.0)

    def test_dtype_name(self):
        assert _array.dtype_name([1.0, 2.0]) == "float64"
        assert _array.dtype_name([1, 2]) == "int64"
        assert _array.dtype_name(True) == "bool"

    def test_dtype_name_numpy_and_torch(self):
        np = pytest.importorskip("numpy")
        assert _array.dtype_name(np.zeros(2, dtype=np.float32)) == "float32"
        torch = pytest.importorskip("torch")
        assert _array.dtype_name(torch.zeros(2, dtype=torch.float16)) == "float16"

    def test_to_numpy_f64_normalizes_every_backend(self):
        np = pytest.importorskip("numpy")
        assert _array.to_numpy_f64([[1, 2], [3, 4]]).dtype == np.float64
        assert _array.to_numpy_f64(FakeJaxArray([1.0, 2.0])).tolist() == [1.0, 2.0]
        torch = pytest.importorskip("torch")
        converted = _array.to_numpy_f64(torch.ones(2, dtype=torch.bfloat16))
        assert converted.dtype == np.float64 and converted.tolist() == [1.0, 1.0]

    def test_max_abs_diff(self):
        assert _array.max_abs_diff([1.0, 5.0], [1.0, 3.0]) == pytest.approx(2.0)
        assert _array.max_abs_diff([1.0], [1.0]) == 0.0

    def test_flat_float32_list_rounds_like_torch(self):
        torch = pytest.importorskip("torch")
        value = 1.0 + 2.0**-30  # not representable in float32
        assert _array.flat_float32_list([value]) == [1.0]
        assert _array.flat_float32_list(torch.tensor([value], dtype=torch.float64)) == [1.0]

    def test_random_like_matches_dtype(self, kit):
        original = kit.vector(8)
        replacement = _array.random_like(original)
        assert _array.shape_of(replacement) == _array.shape_of(original)
        assert _array.dtype_name(replacement) == _array.dtype_name(original)

    def test_random_like_integer_arrays(self):
        np = pytest.importorskip("numpy")
        ints = np.zeros(8, dtype=np.int32)
        replacement = _array.random_like(ints)
        assert replacement.dtype == np.int32
        torch = pytest.importorskip("torch")
        t_ints = torch.zeros(8, dtype=torch.int64)
        assert _array.random_like(t_ints).dtype == torch.int64
        t_bool = torch.zeros(8, dtype=torch.bool)
        assert _array.random_like(t_bool).dtype == torch.bool
