"""Adversarial audit of assert_deterministic / assert_no_memoization.

These two checks exist to prove an LLM-generated kernel did not cheat, so the
failure mode that matters is not "does it run" but "can it report success
having verified nothing". This file is the regression suite for the vacuous
passes found by differential testing across backends:

  * Phase 2 compared the two input sets *after* running the kernel, so a kernel
    that overwrote its own inputs left them looking identical and switched the
    no-op detector off;
  * an ``input_factory`` that does not vary (a constant, or an RNG reseeded
    with a fixed seed on every call) disarmed the same phase silently;
  * Phase 2 compared the kernel's second output against a first output that was
    the *same object*, so an honest kernel writing into a preallocated buffer
    was reported as a no-op while a kernel sharing a scratch buffer with the
    reference was compared against itself and passed;
  * zero-element and NaN outputs made every comparison vacuously true (or
    vacuously false, in Phase 1's case);
  * ``n_runs=0`` / ``n_rounds=0`` skipped whole phases without a word;
  * a poison round that failed to change the data -- or changed it without
    moving the expected answer -- proved nothing, which let a memoizing kernel
    through ~40% of the time on one-element bool inputs.

Every scenario below is run through torch, numpy and nested Python lists and
the verdicts are required to agree: a check that catches a cheat on torch but
waves it through on numpy is the same bug in a different costume.
"""
from __future__ import annotations

import math
import random
import warnings

import pytest

from perflab.harness import _array
from perflab.harness.determinism import assert_deterministic
from perflab.harness.pointer_poison import assert_no_memoization

N = 8


# ---------------------------------------------------------------------------
# Backend kits: one vocabulary, three backends.
# ---------------------------------------------------------------------------

class ListKit:
    name = "list"

    def vec(self, n=N):
        return [random.gauss(0.0, 1.0) for _ in range(n)]

    def const(self, n=N):
        return [float(i) for i in range(n)]

    def zeros(self, n=N):
        return [0.0] * n

    def empty(self):
        return []

    def mul(self, a, b):
        return [x * y for x, y in zip(a, b, strict=True)]

    def offset(self, out, delta):
        return [x + delta for x in out]

    def copy(self, out):
        return list(out)


class NumpyKit:
    name = "numpy"

    def __init__(self):
        self.np = pytest.importorskip("numpy")

    def vec(self, n=N):
        return self.np.random.default_rng().standard_normal(n)

    def const(self, n=N):
        return self.np.arange(n, dtype=self.np.float64)

    def zeros(self, n=N):
        return self.np.zeros(n)

    def empty(self):
        return self.np.zeros(0)

    def mul(self, a, b):
        return a * b

    def offset(self, out, delta):
        return out + delta

    def copy(self, out):
        return out.copy()


class TorchKit:
    name = "torch"

    def __init__(self):
        self.torch = pytest.importorskip("torch")

    def vec(self, n=N):
        return self.torch.randn(n)

    def const(self, n=N):
        return self.torch.arange(n, dtype=self.torch.float32)

    def zeros(self, n=N):
        return self.torch.zeros(n)

    def empty(self):
        return self.torch.zeros(0)

    def mul(self, a, b):
        return a * b

    def offset(self, out, delta):
        return out + delta

    def copy(self, out):
        return out.detach().clone()


KIT_TYPES = {"list": ListKit, "numpy": NumpyKit, "torch": TorchKit}


@pytest.fixture(params=list(KIT_TYPES))
def kit(request):
    return KIT_TYPES[request.param]()


def _available_kits():
    """Every backend importable here -- numpy and torch are optional installs."""
    kits = []
    for factory in KIT_TYPES.values():
        try:
            kits.append(factory())
        except BaseException:  # noqa: BLE001 - importorskip raises Skipped
            continue
    return kits


# ---------------------------------------------------------------------------
# 1. The no-op phase cannot be disarmed
# ---------------------------------------------------------------------------

class TestNoOpPhaseCannotBeDisarmed:
    def test_input_wiping_kernel_is_still_caught(self, kit):
        """The exploitable one: comparing inputs AFTER running the kernel let a
        kernel that zeroed its own inputs make them look identical, so the no-op
        phase concluded nothing and passed a kernel that ignored its arguments
        entirely."""
        fixed = kit.const()

        def wiping_noop(a, b):
            _array.fill_from(a, kit.zeros())
            _array.fill_from(b, kit.zeros())
            return kit.copy(fixed)

        with pytest.raises(AssertionError, match="No-op kernel detected"):
            assert_deterministic(fn=wiping_noop, input_factory=lambda: (kit.vec(), kit.vec()))

    def test_identically_seeded_factory_is_rejected(self, kit):
        """A factory that reseeds its RNG the same way on every call looks
        random but hands back identical data, which leaves 'different inputs
        must change the output' with nothing to compare."""
        def seeded():
            rng = random.Random(1234)
            return (
                [rng.gauss(0.0, 1.0) for _ in range(N)],
                [rng.gauss(0.0, 1.0) for _ in range(N)],
            )

        fixed = kit.const()
        with pytest.raises(AssertionError, match="No-op check cannot run"):
            assert_deterministic(
                fn=lambda a, b: kit.copy(fixed), input_factory=seeded
            )

    def test_factory_returning_the_same_objects_is_rejected(self, kit):
        reused = (kit.vec(), kit.vec())
        fixed = kit.const()

        with pytest.raises(AssertionError, match="No-op check cannot run"):
            assert_deterministic(
                fn=lambda a, b: kit.copy(fixed), input_factory=lambda: reused
            )

    def test_constant_scalar_factory_is_rejected(self):
        with pytest.raises(AssertionError, match="No-op check cannot run"):
            assert_deterministic(
                fn=lambda a, b: a * b, input_factory=lambda: (2.0, 3.0)
            )

    def test_non_varying_factory_warns_when_a_reference_covers_it(self, kit):
        """With a reference_fn, Phase 3 independently rules out a constant
        kernel, so a non-varying factory is a warning rather than an error --
        but it is never silent."""
        reused = (kit.vec(), kit.vec())

        with pytest.warns(UserWarning, match="identical inputs"):
            assert_deterministic(
                fn=kit.mul, input_factory=lambda: reused, reference_fn=kit.mul
            )


# ---------------------------------------------------------------------------
# 2. Output aliasing: snapshots, not object identity
# ---------------------------------------------------------------------------

class TestOutputAliasing:
    def test_honest_kernel_with_preallocated_buffer_passes(self, kit):
        """False positive: the two Phase-2 results were the same object, so an
        honest kernel that writes into a reused output buffer was accused of
        being a no-op."""
        buffer = kit.zeros()

        def honest_reuses_buffer(a, b):
            _array.fill_from(buffer, kit.mul(a, b))
            return buffer

        assert_deterministic(
            fn=honest_reuses_buffer, input_factory=lambda: (kit.vec(), kit.vec())
        )

    def test_kernel_sharing_a_buffer_with_the_reference_is_still_checked(self, kit):
        """Vacuous pass: when kernel and reference write into one scratch
        buffer, the Phase-3 comparison compared that buffer with itself."""
        shared = kit.zeros()

        def wrong_kernel(a, b):
            _array.fill_from(shared, kit.offset(kit.mul(a, b), 99.0))
            return shared

        def reference(a, b):
            _array.fill_from(shared, kit.mul(a, b))
            return shared

        with pytest.raises(AssertionError, match="Correctness failure against reference"):
            assert_deterministic(
                fn=wrong_kernel,
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=reference,
                n_runs=2,
            )

    def test_poison_check_sharing_a_buffer_with_the_reference(self, kit):
        shared = kit.zeros()

        def wrong_kernel(a, b):
            _array.fill_from(shared, kit.offset(kit.mul(a, b), 99.0))
            return shared

        def reference(a, b):
            _array.fill_from(shared, kit.mul(a, b))
            return shared

        with pytest.raises(AssertionError, match="Initial correctness check failed"):
            assert_no_memoization(
                fn=wrong_kernel,
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=reference,
                atol=1e-9,
            )


# ---------------------------------------------------------------------------
# 3. Degenerate outputs: empty and NaN
# ---------------------------------------------------------------------------

class TestDegenerateOutputs:
    def test_zero_element_output_is_rejected(self, kit):
        with pytest.raises(AssertionError, match="zero elements"):
            assert_deterministic(
                fn=lambda a, b: kit.empty(),
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=lambda a, b: kit.empty(),
            )

    def test_zero_element_output_is_rejected_by_poison_check(self, kit):
        with pytest.raises(AssertionError, match="zero elements"):
            assert_no_memoization(
                fn=lambda a, b: kit.empty(),
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=lambda a, b: kit.empty(),
            )

    def test_zero_element_inputs_cannot_be_poisoned(self, kit):
        """A zero-element torch/numpy array reports itself as writeable, so the
        poison "succeeded" while changing nothing -- an empty *list* was already
        rejected, so the three backends disagreed too."""
        with pytest.raises(AssertionError, match="zero elements|cannot run"):
            assert_no_memoization(
                fn=lambda a, b: kit.mul(a, b),
                input_factory=lambda: (kit.empty(), kit.empty()),
                reference_fn=kit.mul,
            )

    def test_nan_output_is_rejected(self, kit):
        with pytest.raises(AssertionError, match="contains NaN"):
            assert_deterministic(
                fn=lambda a, b: kit.offset(kit.mul(a, b), float("nan")),
                input_factory=lambda: (kit.vec(), kit.vec()),
            )

    def test_nan_output_is_rejected_even_with_a_single_run(self, kit):
        """n_runs=1 skips the Phase-1 comparison, which used to be the one place
        an all-NaN output was noticed (by accident, as 'non-deterministic')."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(AssertionError, match="contains NaN"):
                assert_deterministic(
                    fn=lambda a, b: kit.offset(kit.mul(a, b), float("nan")),
                    input_factory=lambda: (kit.vec(), kit.vec()),
                    n_runs=1,
                )

    def test_nan_output_is_rejected_by_poison_check(self, kit):
        with pytest.raises(AssertionError, match="contains NaN"):
            assert_no_memoization(
                fn=lambda a, b: kit.offset(kit.mul(a, b), float("nan")),
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=lambda a, b: kit.offset(kit.mul(a, b), float("nan")),
            )


# ---------------------------------------------------------------------------
# 4. Run/round counts that skip whole phases
# ---------------------------------------------------------------------------

class TestPhaseCountsCannotSkipSilently:
    def test_zero_runs_is_an_error(self, kit):
        with pytest.raises(ValueError, match="never runs the kernel"):
            assert_deterministic(
                fn=kit.mul, input_factory=lambda: (kit.vec(), kit.vec()), n_runs=0
            )

    def test_single_run_warns_that_phase_one_compares_nothing(self, kit):
        with pytest.warns(UserWarning, match="n_runs=1"):
            assert_deterministic(
                fn=kit.mul,
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=kit.mul,
                n_runs=1,
            )

    def test_zero_poison_rounds_is_an_error(self, kit):
        """n_rounds=0 ran the initial correctness check and stopped, so a fully
        memoizing kernel passed the memoization check."""
        cache: dict = {}

        def memoizing(a, b):
            cache.setdefault(id(a), kit.copy(kit.mul(a, b)))
            return cache[id(a)]

        with pytest.raises(ValueError, match="never poisons anything"):
            assert_no_memoization(
                fn=memoizing,
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=kit.mul,
                n_rounds=0,
            )


# ---------------------------------------------------------------------------
# 5. input_factory contract
# ---------------------------------------------------------------------------

class TestInputFactoryContract:
    def test_bare_array_return_names_the_mistake(self):
        """`lambda: arr` (instead of `lambda: (arr,)`) gets splatted row-wise
        into the kernel and used to surface as numpy's 'truth value of an array
        is ambiguous'."""
        np = pytest.importorskip("numpy")

        with pytest.raises(_array.UnsupportedValueError, match="must return a tuple"):
            assert_deterministic(
                fn=lambda *a: a[0], input_factory=lambda: np.zeros(4)
            )

    def test_bare_array_return_named_for_the_poison_check_too(self):
        np = pytest.importorskip("numpy")

        with pytest.raises(_array.UnsupportedValueError, match="assert_no_memoization"):
            assert_no_memoization(
                fn=lambda *a: a[0],
                input_factory=lambda: np.zeros(4),
                reference_fn=lambda *a: a[0],
            )

    def test_empty_factory_still_rejected_before_any_phase(self, kit):
        with pytest.raises(ValueError, match="no inputs"):
            assert_deterministic(fn=lambda: kit.const(), input_factory=tuple)

    def test_varying_arity_is_reported(self, kit):
        counter = {"n": 0}

        def wobbly():
            counter["n"] += 1
            return (kit.vec(),) if counter["n"] % 2 else (kit.vec(), kit.vec())

        with pytest.raises(AssertionError, match="inputs on one call"):
            assert_deterministic(fn=lambda *a: kit.copy(a[0]), input_factory=wobbly, n_runs=2)


# ---------------------------------------------------------------------------
# 6. Poison rounds that poison nothing
# ---------------------------------------------------------------------------

class TestPoisonMustActuallyBite:
    def test_colliding_random_data_is_re_rolled(self, kit, monkeypatch):
        """`random_like` can hand back the data already there -- a one-element
        bool input hits this half the time -- and the round then proves nothing.
        Here the first three rolls are forced to collide."""
        real_random_like = _array.random_like
        rolls = {"n": 0}

        def colliding(value, name="input"):
            rolls["n"] += 1
            if rolls["n"] <= 3:
                return _array.detached_copy(value, name)  # no change at all
            return real_random_like(value, name)

        monkeypatch.setattr(_array, "random_like", colliding)

        cache: dict = {}

        def memoizing(a, b):
            cache.setdefault(id(a), kit.copy(kit.mul(a, b)))
            return cache[id(a)]

        with pytest.raises(AssertionError, match="Memoization detected"):
            assert_no_memoization(
                fn=memoizing,
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=kit.mul,
                atol=1e-9,
                n_rounds=1,
            )
        assert rolls["n"] > 3, "the colliding rolls must have been retried"

    def test_poison_that_never_changes_anything_fails_rather_than_passes(
        self, kit, monkeypatch
    ):
        monkeypatch.setattr(
            _array, "random_like", lambda value, name="input": _array.detached_copy(value)
        )

        cache: dict = {}

        def memoizing(a, b):
            cache.setdefault(id(a), kit.copy(kit.mul(a, b)))
            return cache[id(a)]

        with pytest.raises(AssertionError, match="vacuous"):
            assert_no_memoization(
                fn=memoizing,
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=kit.mul,
                atol=1e-9,
            )

    def test_reference_answer_that_ignores_the_inputs_is_vacuous(self, kit):
        """If poisoning does not move the expected answer, a stale cached result
        still matches it and no cache can ever be detected."""
        constant = kit.const()

        with pytest.raises(AssertionError, match="vacuous"):
            assert_no_memoization(
                fn=lambda a, b: kit.copy(constant),
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=lambda a, b: kit.copy(constant),
                atol=1e-9,
            )

    def test_one_element_bool_cache_never_slips_through(self):
        """The statistical version: a one-element bool input collided with its
        replacement ~50% of the time, so a memoizing kernel used to pass ~40% of
        runs. It must now never pass -- 'caught' or 'vacuous', never silence."""
        np = pytest.importorskip("numpy")
        verdicts = []
        for _ in range(30):
            cache: dict = {}

            def memoizing(a, cache=cache):
                cache.setdefault(id(a), np.array([bool(a[0])]))
                return cache[id(a)]

            try:
                assert_no_memoization(
                    fn=memoizing,
                    input_factory=lambda: (np.array([True]),),
                    reference_fn=lambda a: np.array([bool(a[0])]),
                    atol=0.0,
                    n_rounds=1,
                )
                verdicts.append("PASSED")
            except AssertionError as exc:
                verdicts.append("caught" if "Memoization" in str(exc) else "vacuous")

        assert "PASSED" not in verdicts, f"memoizing kernel slipped through: {verdicts}"
        assert verdicts.count("caught") >= 25, verdicts

    def test_honest_bool_kernel_still_passes(self):
        np = pytest.importorskip("numpy")

        assert_no_memoization(
            fn=lambda a: a.astype(np.float64) * 2.0,
            input_factory=lambda: (np.random.default_rng().integers(0, 2, N).astype(bool),),
            reference_fn=lambda a: a.astype(np.float64) * 2.0,
            atol=0.0,
        )


# ---------------------------------------------------------------------------
# 7. Dtype coverage
# ---------------------------------------------------------------------------

TORCH_DTYPES = ["float16", "bfloat16", "float32", "float64", "int32", "int64", "bool"]
NUMPY_DTYPES = ["float16", "float32", "float64", "int32", "int64", "bool"]


def _torch_inputs(torch, name):
    dtype = getattr(torch, name)
    if name == "bool":
        return torch.randint(0, 2, (N,), dtype=dtype)
    if name.startswith("int"):
        return torch.randint(-50, 50, (N,), dtype=dtype)
    return torch.randn(N).to(dtype)


def _numpy_inputs(np, name):
    rng = np.random.default_rng()
    if name == "bool":
        return rng.integers(0, 2, N).astype(bool)
    if name.startswith("int"):
        return rng.integers(-50, 50, N).astype(name)
    return rng.standard_normal(N).astype(name)


def _combine(a, b):
    """A dtype-preserving binary op that works for floats, ints and bools."""
    return (a & b) if str(getattr(a, "dtype", "")) in ("torch.bool", "bool") else a * b


class TestDtypeCoverage:
    @pytest.mark.parametrize("dtype_name", TORCH_DTYPES)
    def test_torch_dtypes_honest_pass_and_noop_caught(self, dtype_name):
        torch = pytest.importorskip("torch")

        def factory():
            return (_torch_inputs(torch, dtype_name), _torch_inputs(torch, dtype_name))

        assert_deterministic(fn=_combine, input_factory=factory, reference_fn=_combine)

        fixed = _torch_inputs(torch, dtype_name)
        with pytest.raises(AssertionError, match="No-op kernel detected"):
            assert_deterministic(fn=lambda a, b: fixed.clone(), input_factory=factory)

    @pytest.mark.parametrize("dtype_name", NUMPY_DTYPES)
    def test_numpy_dtypes_honest_pass_and_noop_caught(self, dtype_name):
        np = pytest.importorskip("numpy")

        def factory():
            return (_numpy_inputs(np, dtype_name), _numpy_inputs(np, dtype_name))

        assert_deterministic(fn=_combine, input_factory=factory, reference_fn=_combine)

        fixed = _numpy_inputs(np, dtype_name)
        with pytest.raises(AssertionError, match="No-op kernel detected"):
            assert_deterministic(fn=lambda a, b: fixed.copy(), input_factory=factory)

    @pytest.mark.parametrize("dtype_name", TORCH_DTYPES)
    def test_torch_dtypes_survive_the_poison_check(self, dtype_name):
        torch = pytest.importorskip("torch")

        def factory():
            return (_torch_inputs(torch, dtype_name), _torch_inputs(torch, dtype_name))

        assert_no_memoization(
            fn=_combine, input_factory=factory, reference_fn=_combine, atol=0.0
        )


# ---------------------------------------------------------------------------
# 8. Tolerance semantics on every backend
# ---------------------------------------------------------------------------

class TestToleranceSemantics:
    def test_env_tolerance_cannot_loosen_phase_one(self, kit, monkeypatch):
        """PERFLAB_ACCURACY_TOLERANCE is an accuracy bound, not a licence for
        run-to-run garbage: a 1e-3 drift must still fail under a 1e-1 task
        tolerance, on every backend."""
        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "1e-1")
        calls = {"n": 0}

        def drifting(a, b):
            calls["n"] += 1
            return kit.offset(kit.mul(a, b), 1e-3 * (calls["n"] % 2))

        with pytest.raises(AssertionError, match="Non-deterministic"):
            assert_deterministic(
                fn=drifting, input_factory=lambda: (kit.vec(), kit.vec()), n_runs=3
            )

    def test_env_tolerance_still_governs_the_reference_check(self, kit, monkeypatch):
        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "1e-1")

        assert_deterministic(
            fn=lambda a, b: kit.offset(kit.mul(a, b), 1e-3),
            input_factory=lambda: (kit.vec(), kit.vec()),
            reference_fn=kit.mul,
            n_runs=3,
        )

        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "1e-6")
        with pytest.raises(AssertionError, match="Correctness failure"):
            assert_deterministic(
                fn=lambda a, b: kit.offset(kit.mul(a, b), 1e-3),
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=kit.mul,
                n_runs=3,
            )

    def test_explicit_atol_wins_for_both_phases(self, kit, monkeypatch):
        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "1e-9")
        calls = {"n": 0}

        def drifting(a, b):
            calls["n"] += 1
            return kit.offset(kit.mul(a, b), 1e-3 * (calls["n"] % 2))

        # Explicit 1e-1 covers both the drift (Phase 1) and the reference gap.
        assert_deterministic(
            fn=drifting,
            input_factory=lambda: (kit.vec(), kit.vec()),
            reference_fn=kit.mul,
            n_runs=3,
            atol=1e-1,
            rtol=1e-1,
        )

    def test_poison_check_tracks_the_env_tolerance(self, kit, monkeypatch):
        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "1e-1")

        assert_no_memoization(
            fn=lambda a, b: kit.offset(kit.mul(a, b), 1e-3),
            input_factory=lambda: (kit.vec(), kit.vec()),
            reference_fn=kit.mul,
        )

        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "1e-9")
        with pytest.raises(AssertionError, match="Initial correctness check failed"):
            assert_no_memoization(
                fn=lambda a, b: kit.offset(kit.mul(a, b), 1e-3),
                input_factory=lambda: (kit.vec(), kit.vec()),
                reference_fn=kit.mul,
            )


# ---------------------------------------------------------------------------
# 9. Cross-backend differential: same scenario, same verdict everywhere
# ---------------------------------------------------------------------------

def _verdict(thunk) -> str:
    """Classify an outcome into a backend-independent tag."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thunk()
        return "pass"
    except ValueError as exc:
        detail = str(exc).partition(":")[2] or str(exc)
        return f"value-error:{detail.strip()[:24]}"
    except AssertionError as exc:
        message = str(exc)
        for marker in (
            "No-op kernel detected",
            "No-op check cannot run",
            "Non-deterministic",
            "Correctness failure against reference",
            "Initial correctness check failed",
            "Memoization detected",
            "zero elements",
            "contains NaN",
            "cannot run",
            "vacuous",
            "must return a tuple",
            "cannot inspect",
        ):
            if marker in message:
                return marker
        return f"other:{message[:40]}"


def _determinism_scenarios(kit):
    """name -> thunk, for one backend kit."""
    fixed = kit.const()
    reused_inputs = (kit.vec(), kit.vec())
    buffer = kit.zeros()
    shared = kit.zeros()

    def honest_buffered(a, b):
        _array.fill_from(buffer, kit.mul(a, b))
        return buffer

    def wrong_shared(a, b):
        _array.fill_from(shared, kit.offset(kit.mul(a, b), 99.0))
        return shared

    def reference_shared(a, b):
        _array.fill_from(shared, kit.mul(a, b))
        return shared

    def wiping(a, b):
        _array.fill_from(a, kit.zeros())
        _array.fill_from(b, kit.zeros())
        return kit.copy(fixed)

    def unstable(a, b, state={"n": 0}):  # noqa: B006 - per-scenario counter
        state["n"] += 1
        return kit.offset(kit.mul(a, b), 1000.0 * (state["n"] % 2))

    varying = lambda: (kit.vec(), kit.vec())  # noqa: E731
    return {
        "honest": lambda: assert_deterministic(
            fn=kit.mul, input_factory=varying, reference_fn=kit.mul),
        "nondeterministic": lambda: assert_deterministic(
            fn=unstable, input_factory=varying, n_runs=4),
        "noop": lambda: assert_deterministic(
            fn=lambda a, b: kit.copy(fixed), input_factory=varying),
        "wrong-answer": lambda: assert_deterministic(
            fn=lambda a, b: kit.offset(kit.mul(a, b), 0.5),
            input_factory=varying, reference_fn=kit.mul),
        "input-wiping-noop": lambda: assert_deterministic(
            fn=wiping, input_factory=varying),
        "non-varying-factory": lambda: assert_deterministic(
            fn=lambda a, b: kit.copy(fixed), input_factory=lambda: reused_inputs),
        "empty-output": lambda: assert_deterministic(
            fn=lambda a, b: kit.empty(), input_factory=varying),
        "nan-output": lambda: assert_deterministic(
            fn=lambda a, b: kit.offset(kit.mul(a, b), float("nan")),
            input_factory=varying),
        "nan-output-single-run": lambda: assert_deterministic(
            fn=lambda a, b: kit.offset(kit.mul(a, b), float("nan")),
            input_factory=varying, n_runs=1),
        "honest-reused-buffer": lambda: assert_deterministic(
            fn=honest_buffered, input_factory=varying),
        "shared-buffer-vs-reference": lambda: assert_deterministic(
            fn=wrong_shared, input_factory=varying, reference_fn=reference_shared),
        "zero-runs": lambda: assert_deterministic(
            fn=kit.mul, input_factory=varying, n_runs=0),
        "uninspectable-output": lambda: assert_deterministic(
            fn=lambda a, b: object(), input_factory=varying),
    }


def _poison_scenarios(kit):
    cache: dict = {}
    constant = kit.const()
    shared = kit.zeros()

    def memoizing(a, b):
        cache.setdefault(id(a), kit.copy(kit.mul(a, b)))
        return cache[id(a)]

    def wrong_shared(a, b):
        _array.fill_from(shared, kit.offset(kit.mul(a, b), 99.0))
        return shared

    def reference_shared(a, b):
        _array.fill_from(shared, kit.mul(a, b))
        return shared

    varying = lambda: (kit.vec(), kit.vec())  # noqa: E731
    return {
        "honest": lambda: assert_no_memoization(
            fn=kit.mul, input_factory=varying, reference_fn=kit.mul, atol=1e-9),
        "id-keyed-cache": lambda: assert_no_memoization(
            fn=memoizing, input_factory=varying, reference_fn=kit.mul, atol=1e-9),
        "zero-rounds": lambda: assert_no_memoization(
            fn=memoizing, input_factory=varying, reference_fn=kit.mul,
            atol=1e-9, n_rounds=0),
        "empty-inputs": lambda: assert_no_memoization(
            fn=kit.mul, input_factory=lambda: (kit.empty(), kit.empty()),
            reference_fn=kit.mul, atol=1e-9),
        "empty-output": lambda: assert_no_memoization(
            fn=lambda a, b: kit.empty(), input_factory=varying,
            reference_fn=lambda a, b: kit.empty(), atol=1e-9),
        "shared-buffer-vs-reference": lambda: assert_no_memoization(
            fn=wrong_shared, input_factory=varying,
            reference_fn=reference_shared, atol=1e-9),
        "input-independent-reference": lambda: assert_no_memoization(
            fn=lambda a, b: kit.copy(constant), input_factory=varying,
            reference_fn=lambda a, b: kit.copy(constant), atol=1e-9),
        "immutable-inputs": lambda: assert_no_memoization(
            fn=lambda a, b: a * b, input_factory=lambda: (2.0, 3.0),
            reference_fn=lambda a, b: a * b),
        "uninspectable-output": lambda: assert_no_memoization(
            fn=lambda a, b: object(), input_factory=varying, reference_fn=kit.mul),
    }


@pytest.mark.parametrize("scenario", list(_determinism_scenarios(ListKit())))
def test_determinism_verdicts_agree_across_backends(scenario):
    kits = _available_kits()
    if len(kits) < 2:
        pytest.skip("need at least two backends to diff")
    verdicts = {k.name: _verdict(_determinism_scenarios(k)[scenario]) for k in kits}
    assert len(set(verdicts.values())) == 1, f"{scenario}: backends disagree: {verdicts}"


@pytest.mark.parametrize("scenario", list(_poison_scenarios(ListKit())))
def test_poison_verdicts_agree_across_backends(scenario):
    kits = _available_kits()
    if len(kits) < 2:
        pytest.skip("need at least two backends to diff")
    verdicts = {k.name: _verdict(_poison_scenarios(k)[scenario]) for k in kits}
    assert len(set(verdicts.values())) == 1, f"{scenario}: backends disagree: {verdicts}"


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("honest", "pass"),
        ("nondeterministic", "Non-deterministic"),
        ("noop", "No-op kernel detected"),
        ("wrong-answer", "Correctness failure against reference"),
        ("input-wiping-noop", "No-op kernel detected"),
        ("non-varying-factory", "No-op check cannot run"),
        ("empty-output", "zero elements"),
        ("nan-output", "contains NaN"),
        ("nan-output-single-run", "contains NaN"),
        ("honest-reused-buffer", "pass"),
        ("shared-buffer-vs-reference", "Correctness failure against reference"),
        ("uninspectable-output", "cannot inspect"),
    ],
)
def test_determinism_verdicts_are_the_intended_ones(kit, scenario, expected):
    assert _verdict(_determinism_scenarios(kit)[scenario]) == expected


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("honest", "pass"),
        ("id-keyed-cache", "Memoization detected"),
        ("empty-inputs", "zero elements"),
        ("empty-output", "zero elements"),
        ("shared-buffer-vs-reference", "Initial correctness check failed"),
        ("input-independent-reference", "vacuous"),
        ("immutable-inputs", "cannot run"),
        ("uninspectable-output", "cannot inspect"),
    ],
)
def test_poison_verdicts_are_the_intended_ones(kit, scenario, expected):
    assert _verdict(_poison_scenarios(kit)[scenario]) == expected


# ---------------------------------------------------------------------------
# 10. Helper-level pins
# ---------------------------------------------------------------------------

class TestVacuityHelpers:
    def test_nan_probe_is_exactly_nan_not_inf(self, kit):
        """`_reject_vacuous_output` detects NaN as 'not equal to itself'. That
        must stay true for NaN only -- infinities compare equal to themselves on
        all three backends and must not be swept up."""
        from perflab.harness.determinism import _reject_vacuous_output

        finite = kit.zeros()
        _array.fill_from(finite, [float("inf")] * N)
        _reject_vacuous_output(finite, "kernel output")  # must not raise

        nan_values = kit.zeros()
        _array.fill_from(nan_values, [float("nan")] * N)
        with pytest.raises(AssertionError, match="contains NaN"):
            _reject_vacuous_output(nan_values, "kernel output")

    def test_zero_element_detection_uses_the_element_count(self, kit):
        from perflab.harness.determinism import _reject_vacuous_output

        with pytest.raises(AssertionError, match="zero elements"):
            _reject_vacuous_output(kit.empty(), "kernel output")
        _reject_vacuous_output(kit.zeros(), "kernel output")  # must not raise
        assert math.prod(_array.shape_of(kit.zeros())) == N

    def test_scalar_output_is_not_treated_as_empty(self):
        from perflab.harness.determinism import _reject_vacuous_output

        _reject_vacuous_output(1.5, "kernel output")  # shape () == one element
