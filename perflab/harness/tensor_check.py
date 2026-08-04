"""Mitigation #3: Output Type Validation (Lazy Evaluation Guard).

Prevents LLM-generated code from returning a proxy object that stores its
inputs but defers computation until correctness checks invoke __eq__ or other
comparison operators -- the benchmark then times an object construction while
the test pays for the real work.

The fix: validate that the output is a plain, materialized array of its
backend -- not a subclass, with allocated storage and a concrete shape.

Two entry points:
  * ``assert_real_tensor`` — torch-specific, the original check, unchanged.
  * ``assert_real_array``  — backend dispatch (torch, numpy, JAX, Python
    sequences and numbers). Every branch either validates or raises; there is
    no branch that accepts a value it did not inspect.

Usage in tests.py:
    from perflab.harness.tensor_check import assert_real_array

    output = kernel(A, B)
    assert_real_array(output)
    # ... then proceed with correctness checks
"""
from __future__ import annotations

import sys


def assert_real_tensor(tensor, name: str = "output") -> None:
    """Validate that tensor is a genuine, materialized torch.Tensor.

    Checks, in order:
      1. It is an instance of torch.Tensor
      2. Its exact type is torch.Tensor (not a subclass)
      3. It is not a nested tensor (which can mask lazy eval)
      4. It is a dense strided tensor (not sparse/quantized-adjacent layouts,
         whose storage cannot be inspected at all)
      5. It has allocated storage (not a view of nothing)
      6. Its data pointer is non-null (storage actually exists)
      7. It has a concrete shape (not symbolic/lazy)

    Raises AssertionError with a descriptive message on failure.

    Ordering is load-bearing. ``is_nested`` and ``layout`` are the only two
    properties that are *always* safe to read; every deeper probe below them
    can raise from inside torch for some layout:

      * the legacy "strided" nested-tensor layout has, across torch versions,
        sometimes raised an internal RuntimeError from .shape/.stride() access
        itself (e.g. torch 2.13 raises "NestedTensorImpl doesn't support
        sizes") rather than returning symbolic values;
      * every sparse layout raises NotImplementedError("Cannot access storage
        of SparseTensorImpl") from ``untyped_storage()``.

    Reading the two safe properties first turns both cases into a precise
    AssertionError that callers catching AssertionError actually see, instead
    of a torch-internal exception type that escapes the check.

    A quantized tensor passes deliberately: it is strided, has a real data
    pointer, and its values are genuinely computed, so it is not the hazard
    this function exists to catch. It is still not comparable by the rest of
    the harness -- a task producing one should dequantize before the accuracy
    check.
    """
    import torch

    # Check 1: isinstance
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError(
            f"Lazy evaluation detected: {name} is {type(tensor).__name__}, "
            f"not torch.Tensor"
        )

    # Check 2: exact type (reject subclasses)
    if type(tensor) is not torch.Tensor:
        raise AssertionError(
            f"Lazy evaluation detected: {name} is a {type(tensor).__name__} "
            f"subclass of torch.Tensor. Custom subclasses can defer computation "
            f"until comparison operators are called."
        )

    # Check 3: not a nested tensor -- see the docstring note on ordering.
    if tensor.is_nested:
        raise AssertionError(
            f"Suspicious tensor: {name} is a nested tensor, which can mask "
            f"lazy evaluation patterns"
        )

    # Check 4: dense strided layout. A sparse tensor reports a full dense
    # shape while storing only its non-zeros, and its storage cannot be
    # inspected at all -- untyped_storage() raises NotImplementedError from
    # inside torch, which would escape this function as a non-AssertionError.
    if tensor.layout is not torch.strided:
        raise AssertionError(
            f"Suspicious tensor: {name} has layout {tensor.layout}, not a dense "
            f"torch.strided tensor. Its storage cannot be inspected, so this "
            f"check cannot prove the values were materialized. Return a dense "
            f"tensor (e.g. .to_dense()) from the kernel."
        )

    # Check 5: has storage. untyped_storage() rather than the deprecated
    # storage(), which emits a TypedStorage UserWarning into every correctness
    # log. Sizes differ (untyped is in bytes, typed in elements) but only
    # emptiness is tested here, so the check is unchanged.
    if not tensor.untyped_storage().size():
        # Zero-element tensors are OK for shape-only checks, but suspicious
        # for actual computation outputs
        if tensor.numel() > 0:
            raise AssertionError(
                f"Lazy evaluation detected: {name} claims {tensor.numel()} elements "
                f"but has empty storage"
            )

    # Check 6: non-null data pointer
    if tensor.numel() > 0 and tensor.data_ptr() == 0:
        raise AssertionError(
            f"Lazy evaluation detected: {name} has null data pointer "
            f"(storage not allocated)"
        )

    # Check 7: concrete shape (not symbolic)
    try:
        _ = tensor.shape
        _ = tensor.stride()
    except Exception as e:  # noqa: BLE001 -- any failure accessing shape/stride indicates a non-concrete tensor
        raise AssertionError(
            f"Lazy evaluation detected: {name} has non-concrete shape/stride: {e}"
        ) from e


def assert_real_array(value, name: str = "output") -> None:
    """Validate that ``value`` is a genuine, materialized array on any backend.

    Dispatches to the per-backend check:
      * torch  — `assert_real_tensor` (exact type, storage, data pointer, ...)
      * numpy  — exact ``ndarray``/scalar type, numeric dtype, real buffer
      * JAX    — not a ``Tracer``, not deleted, forced concrete
      * list/tuple — no proxy subclass, no cycles, every element recursively
      * number — exact ``int``/``float``/``bool``, not a subclass

    Raises AssertionError with a descriptive message on failure, including for
    a type the harness does not model: a validator that cannot inspect its
    argument must never report success.
    """
    _assert_real_array(value, name, depth=0)


def _assert_real_array(value, name: str, depth: int) -> None:
    from perflab.harness import _array

    backend = _array.require_backend(value, name)
    if backend == _array.TORCH:
        assert_real_tensor(value, name)
    elif backend == _array.NUMPY:
        _assert_real_numpy(value, name)
    elif backend == _array.JAX:
        _assert_real_jax(value, name)
    elif backend == _array.SEQUENCE:
        _assert_real_sequence(value, name, depth)
    else:
        _assert_real_scalar(value, name)


#: Nesting depth beyond which a sequence is treated as unreadable. A real
#: task output is a handful of dimensions deep; anything past this is either a
#: cycle (``a = []; a.append(a)``) or a structure the harness cannot compare,
#: and recursing further would raise RecursionError -- which is not an
#: AssertionError, so it would escape every caller's check-failure handler.
_MAX_SEQUENCE_DEPTH = 64

#: Operators a lazy proxy would override to do its real work at comparison
#: time. A list/tuple subclass that leaves all of these inherited (namedtuple,
#: torch's ``return_types`` struct sequences) is just a labelled tuple and is
#: accepted; one that overrides any of them is the sequence-shaped version of
#: the lazy tensor and is rejected.
_SEQUENCE_DUNDERS = ("__eq__", "__ne__", "__iter__", "__getitem__", "__len__")


def _assert_real_sequence(value, name: str, depth: int) -> None:
    """Validate a list/tuple: no lazy subclass, no cycles, real elements."""
    if depth >= _MAX_SEQUENCE_DEPTH:
        raise AssertionError(
            f"Lazy evaluation detected: {name} nests more than "
            f"{_MAX_SEQUENCE_DEPTH} levels deep (or contains a cycle), so its "
            f"elements cannot be inspected."
        )

    kind = tuple if isinstance(value, tuple) else list
    if type(value) is not kind:
        overridden = [
            dunder
            for dunder in _SEQUENCE_DUNDERS
            if getattr(type(value), dunder, None) is not getattr(kind, dunder, None)
        ]
        if overridden:
            raise AssertionError(
                f"Lazy evaluation detected: {name} is a {type(value).__name__}, a "
                f"{kind.__name__} subclass that overrides {', '.join(overridden)}. "
                f"Overriding comparison or iteration lets the object defer "
                f"computation until the correctness check runs."
            )

    for index, element in enumerate(value):
        _assert_real_array(element, f"{name}[{index}]", depth + 1)


def _assert_real_numpy(array, name: str) -> None:
    """Reject numpy proxies, subclasses and non-numeric (object) arrays.

    A dtype=object array is the numpy-shaped version of the lazy-evaluation
    hazard: its elements can be arbitrary objects that only compute when
    compared, so it is rejected even though the array itself is materialized.
    """
    from perflab.harness import _array

    np = _array.require_numpy(name)

    if isinstance(array, np.generic):
        # A numpy scalar (np.float64 etc.) is fully materialized by definition.
        _assert_numeric_dtype(np, array.dtype, name, "numpy scalar")
        return

    if type(array) is not np.ndarray:
        raise AssertionError(
            f"Lazy evaluation detected: {name} is a {type(array).__name__}, a "
            f"numpy.ndarray subclass or proxy. Custom subclasses can defer "
            f"computation until comparison operators are called."
        )

    _assert_numeric_dtype(np, array.dtype, name, "array")

    if array.size > 0 and array.ctypes.data == 0:
        raise AssertionError(
            f"Lazy evaluation detected: {name} claims {array.size} elements "
            f"but has a null data pointer (storage not allocated)"
        )

    try:
        _ = tuple(int(dim) for dim in array.shape)
        _ = array.strides
    except Exception as e:  # noqa: BLE001 -- any failure here means a non-concrete array
        raise AssertionError(
            f"Lazy evaluation detected: {name} has non-concrete shape/strides: {e}"
        ) from e


def _assert_numeric_dtype(np, dtype, name: str, what: str) -> None:
    """Require a dtype whose elements are numbers stored as raw bytes.

    ``dtype == object`` is not sufficient on its own: a *structured* dtype such
    as ``[("x", object)]`` compares unequal to ``object`` while still storing
    arbitrary Python objects per field, so ``np.zeros(4, dtype=[("x", object)])``
    is a fully-fledged lazy-evaluation vehicle that the equality test misses.
    ``dtype.hasobject`` is the flag that is true for both.

    Anything else non-numeric (str, bytes, void, datetime64) is rejected too:
    the rest of the harness cannot compare those values at all, so passing them
    here would hand back a green light for a check that can never run.
    """
    if dtype.hasobject:
        raise AssertionError(
            f"Lazy evaluation detected: {name} has dtype=object ({dtype!r}), so its "
            f"elements may be unevaluated proxy objects rather than numbers."
        )
    if not (np.issubdtype(dtype, np.number) or dtype == np.bool_):
        raise AssertionError(
            f"Suspicious {what}: {name} has non-numeric dtype {dtype!r}. The "
            f"harness compares numbers; a value it cannot compare must not pass "
            f"as a materialized result."
        )


def _assert_real_jax(array, name: str) -> None:
    """Reject JAX tracers and force the array to be concrete.

    A ``Tracer`` is exactly the JAX-shaped version of what `assert_real_tensor`
    exists to catch: an abstract stand-in recorded during tracing that carries
    no data, so any timing taken around it measures graph construction rather
    than computation. JAX dispatch is also asynchronous, so a returned array
    may still be in flight; ``block_until_ready`` forces the work to actually
    happen before any measurement is trusted.
    """
    if any(base.__name__ == "Tracer" for base in type(array).__mro__):
        raise AssertionError(
            f"Lazy evaluation detected: {name} is a JAX {type(array).__name__}, "
            f"a traced abstract value rather than a materialized array. It "
            f"carries no data, so nothing has actually been computed."
        )

    jax = sys.modules.get("jax")
    array_cls = getattr(jax, "Array", None) if jax is not None else None
    if isinstance(array_cls, type) and not isinstance(array, array_cls):
        raise AssertionError(
            f"Lazy evaluation detected: {name} is a {type(array).__name__}, "
            f"not a jax.Array"
        )

    is_deleted = getattr(array, "is_deleted", None)
    if callable(is_deleted) and is_deleted():
        raise AssertionError(
            f"Lazy evaluation detected: {name} refers to a deleted JAX buffer"
        )

    blocker = getattr(array, "block_until_ready", None)
    if not callable(blocker):
        raise AssertionError(
            f"Lazy evaluation detected: {name} looks like a JAX array but has no "
            f"block_until_ready(), so it cannot be forced to a concrete value"
        )
    try:
        blocker()
    except Exception as e:  # noqa: BLE001 -- any failure to materialize is the failure we test for
        raise AssertionError(
            f"Lazy evaluation detected: {name} could not be materialized: {e}"
        ) from e

    try:
        _ = tuple(int(dim) for dim in array.shape)
    except Exception as e:  # noqa: BLE001 -- symbolic/abstract shape
        raise AssertionError(
            f"Lazy evaluation detected: {name} has non-concrete shape: {e}"
        ) from e


def _assert_real_scalar(value, name: str) -> None:
    """Reject int/float/bool subclasses.

    A ``class Lazy(float)`` that overrides ``__eq__`` to compute on demand is
    the scalar-shaped version of the lazy tensor, and passes any
    ``isinstance(x, float)`` test.
    """
    if type(value) not in (bool, int, float):
        raise AssertionError(
            f"Lazy evaluation detected: {name} is a {type(value).__name__}, a "
            f"subclass of a Python number rather than a plain int/float/bool. "
            f"Subclasses can override comparison operators to defer computation."
        )
