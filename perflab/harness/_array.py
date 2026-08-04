"""Backend adapter for the harness checks (internal).

Why this module exists
----------------------
The reward-hack mitigations in this package were written against torch, but
PerfLab tasks are Python, C++, CUDA, PyTorch, JAX and Triton. A numpy or
list-of-lists task could import the same helpers and have them *silently pass
without checking anything*, because every real check sat behind an
``isinstance(x, torch.Tensor)`` guard. This module is the dispatch layer that
lets each mitigation run its actual check on whatever a task hands it.

Two rules govern everything here.

1. **Nothing is imported at module scope.** numpy is not a core PerfLab
   dependency (it lives in the ``tasks-*`` extras), and torch/jax are heavy
   optional installs, so ``import perflab.harness`` must work with zero
   optional dependencies present. Backends are therefore identified from
   ``type(x).__module__``, which needs no import at all: a torch tensor is
   recognized on a machine where numpy was never installed, and vice versa.
   Frameworks are only touched through ``sys.modules`` (never an ``import``)
   unless the value itself proves the framework is already loaded.

2. **An unrecognized value raises; it never falls through to a pass.** Every
   entry point routes unknown types into `UnsupportedValueError`. A check that
   cannot inspect its input is a check that is not running, and a silently
   non-running anti-gaming check is worse than no check at all -- it reads as a
   green light.

Supported backends: ``torch.Tensor``, ``numpy.ndarray`` (and numpy scalars),
JAX arrays, nested ``list``/``tuple`` of numbers, and plain Python numbers.
"""
from __future__ import annotations

import math
import random
import struct
import sys
from typing import Any

# Backend tags. Plain strings rather than an enum: they are compared, formatted
# into messages, and never persisted.
TORCH = "torch"
NUMPY = "numpy"
JAX = "jax"
SEQUENCE = "sequence"
SCALAR = "scalar"
UNKNOWN = "unknown"

#: Backends backed by a third-party framework object.
FRAMEWORKS = (TORCH, NUMPY, JAX)
#: Backends that carry more than one value.
ARRAY_BACKENDS = (TORCH, NUMPY, JAX, SEQUENCE)
#: Backends built out of plain Python objects (no framework needed to compare).
PYTHON_BACKENDS = (SEQUENCE, SCALAR)


class UnsupportedValueError(TypeError, AssertionError):
    """A value whose backend the harness cannot determine.

    Deliberately both a TypeError (it *is* a type problem) and an
    AssertionError (harness callers -- tests.py files and pytest -- treat
    AssertionError as "this check failed"). Subclassing both means an
    unsupported type surfaces as a check failure through existing
    ``pytest.raises(AssertionError)`` call sites instead of escaping as an
    unrelated crash, while still reading as a type error to anyone debugging.
    """


_NUMPY_HINT = (
    'numpy is required for this comparison but is not installed. '
    'Install it with: pip install "perflab[tasks-python]"'
)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def backend_of(value: Any) -> str:
    """Return the backend tag for ``value`` without importing anything.

    Detection is by defining module (``type(value).__module__``) first, so no
    framework needs to be installed -- let alone imported -- to classify a
    value from a different one.
    """
    root = type(value).__module__.split(".", 1)[0]
    if root in FRAMEWORKS:
        return root
    if root == "jaxlib":
        return JAX
    # Subclasses and test doubles are declared in *their own* module, so the
    # module-name test above misses them (a `class Lazy(torch.Tensor)` in
    # tests.py reports "tests.py"'s module). Fall back to isinstance against
    # frameworks that are ALREADY loaded -- a sys.modules lookup, never an
    # import, so this stays free for backends that are not in play.
    guessed = _backend_by_isinstance(value)
    if guessed is not None:
        return guessed
    if isinstance(value, (bool, int, float)):
        return SCALAR
    if isinstance(value, (list, tuple)):
        return SEQUENCE
    return UNKNOWN


def _backend_by_isinstance(value: Any) -> str | None:
    """Classify ``value`` against already-loaded frameworks, or None."""
    for module_name, attr, tag in (
        ("torch", "Tensor", TORCH),
        ("numpy", "ndarray", NUMPY),
        ("jax", "Array", JAX),
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        cls = getattr(module, attr, None)
        # `isinstance(cls, type)` filters out MagicMock stand-ins, whose
        # attributes are mocks rather than classes.
        if isinstance(cls, type) and isinstance(value, cls):
            return tag
    return None


def require_backend(value: Any, name: str = "value") -> str:
    """Return the backend tag for ``value``, raising if it is unknown.

    This is the choke point for rule 2: a harness check calls it before
    inspecting anything, so an unhandled type fails loudly instead of
    skipping the check.
    """
    backend = backend_of(value)
    if backend == UNKNOWN:
        raise UnsupportedValueError(
            f"{name} is a {_type_label(value)}, which the PerfLab harness cannot "
            f"inspect. Supported: torch.Tensor, numpy.ndarray, JAX arrays, nested "
            f"list/tuple of numbers, and plain Python numbers. This check refuses "
            f"to pass on a value it cannot compare -- convert {name} to one of "
            f"those types (e.g. numpy.asarray(...) or list(...)) before checking."
        )
    return backend


def is_array(value: Any) -> bool:
    """True if ``value`` holds a collection of numbers this module can read."""
    return backend_of(value) in ARRAY_BACKENDS


def _type_label(value: Any) -> str:
    cls = type(value)
    module = cls.__module__
    return cls.__qualname__ if module == "builtins" else f"{module}.{cls.__qualname__}"


def require_numpy(name: str = "value"):
    """Import numpy on demand, with an actionable error when it is missing."""
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules patch
        raise UnsupportedValueError(f"cannot compare {name}: {_NUMPY_HINT}") from exc
    return numpy


def _live(module_name: str):
    """Return an already-imported module, or None. Never imports."""
    return sys.modules.get(module_name)


def _torch():
    """The torch module, which a torch-backed value proves is already loaded."""
    module = _live("torch")
    if module is None:  # pragma: no cover - unreachable via backend_of(TORCH)
        raise UnsupportedValueError(
            "a torch value was passed but the torch module is not loaded"
        )
    return module


# ---------------------------------------------------------------------------
# Shape / dtype
# ---------------------------------------------------------------------------

def shape_of(value: Any, name: str = "value") -> tuple[int, ...]:
    """Return the shape of ``value`` as a tuple of ints.

    Nested sequences are walked and validated: a ragged nesting raises rather
    than reporting the first row's shape for the whole structure.
    """
    backend = require_backend(value, name)
    if backend in FRAMEWORKS:
        return tuple(int(dim) for dim in value.shape)
    if backend == SCALAR:
        return ()
    if len(value) == 0:
        return (0,)
    first = shape_of(value[0], name)
    for index, element in enumerate(value[1:], start=1):
        other = shape_of(element, name)
        if other != first:
            raise UnsupportedValueError(
                f"{name} is ragged: element 0 has shape {first} but element "
                f"{index} has shape {other}. The harness cannot compare ragged "
                f"structures element-wise."
            )
    return (len(value), *first)


def _shape_or_none(value: Any) -> tuple[int, ...] | None:
    """Best-effort shape, or None when the value does not expose one.

    Framework stand-ins used in tests carry no ``.shape``; returning None lets
    the shape guard stand down for them rather than crash, while real arrays
    are always checked.
    """
    backend = backend_of(value)
    if backend in FRAMEWORKS:
        raw = getattr(value, "shape", None)
        if raw is None:
            return None
        try:
            return tuple(int(dim) for dim in raw)
        except (TypeError, ValueError):
            return None
    if backend in PYTHON_BACKENDS:
        try:
            return shape_of(value)
        except UnsupportedValueError:
            # A heterogeneous container -- e.g. a multi-output kernel returning
            # (out, lse) for attention, (values, indices) for top-k, or
            # (out, scalar_loss). shape_of calls that "ragged" because it is
            # trying to derive ONE shape, but as a structure it is perfectly
            # comparable: pair the elements by position and compare each. Wiring
            # None here stands the single-shape guard down so the element-wise
            # recursion below can do exactly that; genuinely mismatched leaves
            # still fail on their own shapes.
            return None
    return None


def dtype_of(value: Any) -> Any:
    """Native dtype object when the backend has one, else None."""
    return getattr(value, "dtype", None)


def dtype_name(value: Any, name: str = "value") -> str:
    """Normalized dtype name, e.g. "float32", for any supported backend."""
    backend = require_backend(value, name)
    if backend == TORCH:
        return str(dtype_of(value)).replace("torch.", "")
    if backend in (NUMPY, JAX):
        return str(dtype_of(value))
    if backend == SCALAR:
        if isinstance(value, bool):
            return "bool"
        return "int64" if isinstance(value, int) else "float64"
    if len(value) == 0:
        return "empty"
    return dtype_name(value[0], name)


def dtype_repr(value: Any) -> Any:
    """Native dtype if present, else the normalized name (for messages)."""
    native = dtype_of(value)
    return native if native is not None else dtype_name(value)


def dtype_matches(value: Any, expected: Any) -> bool:
    """True if ``value``'s dtype is ``expected``.

    Tries the backend's own equality first so torch keeps its exact historical
    semantics (``actual.dtype != expected_dtype``), then falls back to
    comparing normalized names so ``"float32"``, ``numpy.float32`` and
    ``torch.float32`` can all be used as the expectation.
    """
    native = dtype_of(value)
    if native is not None:
        try:
            if bool(native == expected):
                return True
        except (TypeError, ValueError):
            pass
    return dtype_name(value) == _dtype_label(expected)


def _dtype_label(expected: Any) -> str:
    if isinstance(expected, str):
        return expected
    name = getattr(expected, "__name__", None)  # numpy scalar types
    if isinstance(name, str):
        return name
    return str(expected).replace("torch.", "").rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def to_numpy(value: Any, name: str = "value"):
    """Materialize ``value`` as a numpy array, preserving its dtype.

    Used for exact comparisons, where coercing to float64 would collapse
    distinct int64 values.
    """
    backend = require_backend(value, name)
    numpy = require_numpy(name)
    if backend == TORCH:
        return value.detach().cpu().numpy()
    if backend == JAX:
        return numpy.asarray(_jax_ready(value))
    try:
        return numpy.asarray(value)
    except (TypeError, ValueError) as exc:
        raise UnsupportedValueError(
            f"{name} could not be converted to a numpy array: {exc}"
        ) from exc


def to_numpy_f64(value: Any, name: str = "value"):
    """Materialize ``value`` as a float64 numpy array -- the canonical form.

    Every cross-backend comparison funnels through this so a torch tensor, a
    JAX array and a list of floats are compared on identical terms.
    """
    backend = require_backend(value, name)
    numpy = require_numpy(name)
    if backend == TORCH:
        return value.detach().cpu().double().numpy()
    if backend == JAX:
        return numpy.asarray(_jax_ready(value), dtype=numpy.float64)
    try:
        return numpy.asarray(value, dtype=numpy.float64)
    except (TypeError, ValueError) as exc:
        raise UnsupportedValueError(
            f"{name} could not be converted to a float64 numpy array: {exc}"
        ) from exc


def flat_float32_list(
    value: Any, indices: list[int] | None = None, name: str = "value"
) -> list[float]:
    """Flatten to Python floats, rounded to float32, optionally sampled.

    The float32 rounding is deliberate and load-bearing for the ULP check: it
    reproduces the ``.detach().float().cpu()`` step the torch-only version
    applied to *both* operands, so an fp64 reference is rounded to the working
    precision before ULP distances are taken. Without it a correct fp32 kernel
    would sit ~2**29 float64-ULPs from its own fp64 reference and every task
    would fail.
    """
    return flat_float_list(value, "float32", indices, name)


def flat_float64_list(
    value: Any, indices: list[int] | None = None, name: str = "value"
) -> list[float]:
    """Flatten to Python floats at full float64, no rounding, optionally sampled.

    Every backend is widened rather than narrowed here so that the caller --
    not the backend -- decides the working precision. Rounding then happens in
    one place (``_round_to_precision``) for all backends, which makes
    cross-backend agreement structural instead of a coincidence of three
    separate cast implementations.
    """
    backend = require_backend(value, name)
    if backend == TORCH:
        torch = _torch()
        flat = value.detach().reshape(-1).double().cpu()
        if indices is not None:
            flat = flat[torch.as_tensor(indices, dtype=torch.int64)]
        return [float(v) for v in flat.tolist()]
    if backend in (NUMPY, JAX):
        numpy = require_numpy(name)
        source = _jax_ready(value) if backend == JAX else value
        flat = numpy.asarray(source).astype(numpy.float64).reshape(-1)
        if indices is not None:
            flat = flat[numpy.asarray(indices, dtype=numpy.int64)]
        return [float(v) for v in flat.tolist()]
    flat = [float(v) for v in _flatten_python(value, name)]
    if indices is not None:
        flat = [flat[i] for i in indices]
    return flat


def flat_float_list(
    value: Any,
    precision: str = "float32",
    indices: list[int] | None = None,
    name: str = "value",
) -> list[float]:
    """Flatten to Python floats rounded to ``precision``, optionally sampled."""
    return [
        _round_to_precision(v, precision)
        for v in flat_float64_list(value, indices, name)
    ]


def _flatten_python(value: Any, name: str) -> list[float]:
    backend = require_backend(value, name)
    if backend == SCALAR:
        return [float(value)]
    if backend == SEQUENCE:
        out: list[float] = []
        for element in value:
            out.extend(_flatten_python(element, name))
        return out
    raise UnsupportedValueError(
        f"{name} mixes a {backend} array into a Python sequence; flatten it "
        f"into a single backend before checking."
    )


def _to_float32(value: float) -> float:
    """Round a Python float to float32, matching a torch ``.float()`` cast."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


#: struct formats per precision. bfloat16 has no struct format -- it is the
#: top 16 bits of a float32 -- so it is derived rather than packed.
FLOAT_PRECISIONS: dict[str, tuple[int, str | None, str | None]] = {
    "float16": (16, "<e", "<H"),
    "bfloat16": (16, None, None),
    "float32": (32, "<f", "<I"),
    "float64": (64, "<d", "<Q"),
}


def _round_to_precision(value: float, precision: str) -> float:
    """Round a Python float to ``precision``, round-to-nearest-even."""
    if precision == "float64":
        return value
    if precision not in FLOAT_PRECISIONS:
        raise UnsupportedValueError(
            f"unsupported precision {precision!r}; expected one of "
            f"{sorted(FLOAT_PRECISIONS)}"
        )
    if precision == "bfloat16":
        return struct.unpack("<f", struct.pack("<I", bfloat16_bits(value) << 16))[0]
    _bits, ffmt, _ifmt = FLOAT_PRECISIONS[precision]
    try:
        return struct.unpack(ffmt, struct.pack(ffmt, value))[0]  # type: ignore[arg-type]
    except (OverflowError, ValueError):
        return math.copysign(math.inf, value)


def bfloat16_bits(value: float) -> int:
    """Bit pattern of ``value`` rounded to bfloat16 (round-to-nearest-even).

    bfloat16 is float32 truncated to its top 16 bits, so the rounding has to be
    applied by hand. inf/NaN are truncated rather than rounded: adding the
    rounding bias to a saturated exponent would carry into a different class
    (a large finite value becoming inf, or inf becoming NaN).
    """
    bits = struct.unpack("<I", struct.pack("<f", _to_float32(value)))[0]
    if (bits >> 23) & 0xFF == 0xFF:
        return (bits >> 16) & 0xFFFF
    bits += 0x7FFF + ((bits >> 16) & 1)
    return (bits >> 16) & 0xFFFF


def float_bits(value: float, precision: str) -> int:
    """Raw bit pattern of ``value`` in ``precision``."""
    if precision == "bfloat16":
        return bfloat16_bits(value)
    nbits, ffmt, ifmt = FLOAT_PRECISIONS[precision]
    try:
        return struct.unpack(ifmt, struct.pack(ffmt, value))[0]  # type: ignore[arg-type]
    except (OverflowError, ValueError):
        sign = 1 << (nbits - 1)
        exponent = {"float16": 0x7C00, "float32": 0x7F800000,
                    "float64": 0x7FF0000000000000}[precision]
        return exponent | sign if math.copysign(1.0, value) < 0 else exponent


def detached_copy(value: Any, name: str = "value"):
    """Snapshot ``value`` so later mutation of the original cannot alias it.

    Checks that capture "what the kernel returned last time" depend on this:
    a kernel handing back a buffer it later overwrites must not be able to
    make two captured results silently identical.
    """
    backend = require_backend(value, name)
    if backend == TORCH:
        return value.detach().clone()
    if backend == NUMPY:
        return value.copy()
    if backend == JAX:
        # JAX arrays are immutable, so the value itself is already a snapshot;
        # block first so a pending async computation cannot land afterwards.
        sync(value)
        return value
    if backend == SEQUENCE:
        copied = [detached_copy(element, name) for element in value]
        return copied if isinstance(value, list) else tuple(copied)
    return value


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _join_backend(a: Any, b: Any, name_a: str, name_b: str) -> str:
    """Backend to compare a pair in: Python values promote to the framework."""
    backend_a = require_backend(a, name_a)
    backend_b = require_backend(b, name_b)
    if backend_a == backend_b:
        return backend_a
    if backend_a in PYTHON_BACKENDS:
        return backend_b
    if backend_b in PYTHON_BACKENDS:
        return backend_a
    # Two different frameworks (e.g. torch vs jax): numpy is the meeting point.
    return NUMPY


def _as_torch(value: Any, torch: Any) -> Any:
    return value if backend_of(value) == TORCH else torch.as_tensor(value)


def allclose(
    a: Any,
    b: Any,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    name_a: str = "actual",
    name_b: str = "expected",
) -> tuple[bool, float]:
    """Compare two values within tolerance on any backend.

    Returns ``(ok, max_abs_diff)``. ``max_abs_diff`` is only computed when the
    comparison fails -- it exists for the failure message -- and is 0.0 on
    success. Computing it eagerly would add passes over every tensor on the
    happy path and would break outright for dtypes that do not support
    subtraction (torch bool tensors), which is why the torch-only original
    also computed it exclusively inside its failure branch.

    Mismatched shapes raise instead of broadcasting: both ``torch.allclose``
    and ``numpy.allclose`` happily broadcast a (N, 1) against a (N, N) and
    report success, which would let a kernel returning the wrong shape pass.
    """
    backend = _join_backend(a, b, name_a, name_b)
    shape_a, shape_b = _shape_or_none(a), _shape_or_none(b)
    if shape_a is not None and shape_b is not None and shape_a != shape_b:
        raise AssertionError(
            f"Shape mismatch: {name_a} has shape {shape_a}, {name_b} has shape "
            f"{shape_b}. These cannot be compared element-wise."
        )

    if backend == TORCH:
        torch = _torch()
        tensor_a, tensor_b = _as_torch(a, torch), _as_torch(b, torch)
        tensor_a, tensor_b = _promote_torch_pair(tensor_a, tensor_b, torch)
        if bool(torch.allclose(tensor_a, tensor_b, atol=atol, rtol=rtol)):
            return True, 0.0
        return False, _torch_max_abs_diff(tensor_a, tensor_b)
    if backend in (NUMPY, JAX):
        numpy = require_numpy(name_a)
        array_a = to_numpy_f64(a, name_a)
        array_b = to_numpy_f64(b, name_b)
        if bool(numpy.allclose(array_a, array_b, atol=atol, rtol=rtol)):
            return True, 0.0
        if array_a.size == 0:
            return False, 0.0
        return False, float(numpy.max(numpy.abs(array_a - array_b)))
    ok, worst = _python_allclose(a, b, atol, rtol, name_a, name_b)
    return (True, 0.0) if ok else (False, worst)


def _promote_torch_pair(a: Any, b: Any, torch: Any) -> tuple[Any, Any]:
    """Bring two tensors to a common dtype before comparing them.

    ``torch.allclose``/``torch.equal`` raise RuntimeError on a dtype mismatch,
    and RuntimeError is not an AssertionError -- so it escapes a task's
    correctness handling as an unrelated crash rather than a check failure.
    That matters because mixed precision is the *intended* optimization for
    several tasks: a bf16 kernel output compared against an fp32 reference is
    the normal case here, not an error. The numpy path already normalizes both
    sides via to_numpy_f64, so this only restores parity.

    Promotion never loses information -- it widens to the common type (bf16 vs
    fp32 -> fp32), so the comparison is made at the finer of the two.
    """
    if a.dtype == b.dtype:
        return a, b
    try:
        common = torch.promote_types(a.dtype, b.dtype)
    except Exception:  # noqa: BLE001 - exotic dtype pair; fall back to fp64
        return a.double(), b.double()
    return a.to(common), b.to(common)


def _torch_max_abs_diff(a: Any, b: Any) -> float:
    try:
        return float((a - b).abs().max().item())
    except Exception:  # noqa: BLE001 - diagnostic only; dtypes such as bool
        return float("nan")  # cannot subtract, and the failure still stands.


def _python_allclose(
    a: Any, b: Any, atol: float, rtol: float, name_a: str, name_b: str
) -> tuple[bool, float]:
    """Recursive tolerance compare over plain Python values.

    Pure Python on purpose: a list-of-floats task must be checkable on an
    install with no numpy at all.
    """
    backend_a = require_backend(a, name_a)
    backend_b = require_backend(b, name_b)
    if SEQUENCE in (backend_a, backend_b):
        if backend_a != SEQUENCE or backend_b != SEQUENCE or len(a) != len(b):
            raise AssertionError(
                f"Shape mismatch: {name_a} has shape {shape_of(a, name_a)}, "
                f"{name_b} has shape {shape_of(b, name_b)}. These cannot be "
                f"compared element-wise."
            )
        ok_all, worst = True, 0.0
        for element_a, element_b in zip(a, b, strict=True):
            # Full dispatch, so a tuple of tensors compares element-wise.
            ok, diff = allclose(element_a, element_b, atol, rtol, name_a, name_b)
            if not ok:
                ok_all = False
                worst = float("inf") if math.isnan(diff) else max(worst, diff)
        return ok_all, worst
    value_a, value_b = float(a), float(b)
    if math.isnan(value_a) or math.isnan(value_b):
        # NaN compares unequal to everything, including itself; report inf so
        # the failure message shows a number.
        return False, float("inf")
    # Non-finite values compare by equality, matching numpy.isclose. Falling
    # through to the tolerance test inverts BOTH cases and is not a rounding
    # nicety -- it is a correctness hole:
    #   +inf vs -inf -> |a-b| is inf, and `inf <= atol + rtol*inf` is True,
    #                   so a kernel returning -inf "matches" a +inf reference.
    #   +inf vs +inf -> |a-b| is nan, and `nan <= anything` is False,
    #                   so a correct kernel is rejected.
    # torch and numpy both get this right; only the pure-Python leaf did not.
    if math.isinf(value_a) or math.isinf(value_b):
        equal = value_a == value_b
        return equal, 0.0 if equal else float("inf")
    diff = abs(value_a - value_b)
    return diff <= atol + rtol * abs(value_b), diff


def max_abs_diff(a: Any, b: Any, name_a: str = "actual", name_b: str = "expected") -> float:
    """Largest absolute element-wise difference between two values.

    Zero tolerance means `allclose` only reports success for identical values,
    where the maximum difference is 0.0 anyway -- so the second element of its
    result is the true maximum in both branches.
    """
    _, diff = allclose(a, b, atol=0.0, rtol=0.0, name_a=name_a, name_b=name_b)
    return diff


def exact_equal(a: Any, b: Any, name_a: str = "actual", name_b: str = "expected") -> bool:
    """Bit-for-bit equality, mirroring ``torch.equal`` semantics.

    Differing shapes give False rather than raising -- "these are not the same
    value" is the correct answer to an equality question.
    """
    backend = _join_backend(a, b, name_a, name_b)
    shape_a, shape_b = _shape_or_none(a), _shape_or_none(b)
    if shape_a is not None and shape_b is not None and shape_a != shape_b:
        return False
    if backend == TORCH:
        torch = _torch()
        tensor_a, tensor_b = _as_torch(a, torch), _as_torch(b, torch)
        return bool(torch.equal(*_promote_torch_pair(tensor_a, tensor_b, torch)))
    if backend in (NUMPY, JAX):
        numpy = require_numpy(name_a)
        return bool(numpy.array_equal(to_numpy(a, name_a), to_numpy(b, name_b)))
    if backend == SEQUENCE:
        if len(a) != len(b):
            return False
        return all(
            exact_equal(element_a, element_b, name_a, name_b)
            for element_a, element_b in zip(a, b, strict=True)
        )
    return bool(a == b)


def values_equal(a: Any, b: Any) -> bool:
    """Lenient equality used to ask "did these inputs actually change?".

    Unlike `exact_equal` this tolerates types the harness does not model --
    a task may legitimately pass a string mode flag or a config object
    alongside its arrays. Falling back to Python ``==`` is still a real
    comparison, not a skipped check; a value that cannot be compared at all is
    reported as *different*, which is the conservative direction (it keeps the
    no-op detector armed rather than disarming it).
    """
    try:
        return exact_equal(a, b)
    except UnsupportedValueError:
        try:
            return bool(a == b)
        except Exception:  # noqa: BLE001 - exotic __eq__; treat as "changed"
            return False


# ---------------------------------------------------------------------------
# In-place mutation (pointer poisoning)
# ---------------------------------------------------------------------------

def is_fillable(value: Any) -> bool:
    """True if `fill_from` can overwrite ``value`` in place."""
    backend = backend_of(value)
    if backend == TORCH:
        return True
    if backend == NUMPY:
        flags = getattr(value, "flags", None)
        return bool(getattr(flags, "writeable", False))
    if backend == SEQUENCE:
        return isinstance(value, list) and len(value) > 0
    return False


def fill_from(dst: Any, src: Any, name: str = "input") -> None:
    """Overwrite ``dst``'s contents from ``src``, preserving object identity.

    This is what the pointer-poison check is built on: the object -- and, for
    torch/numpy, the underlying buffer address -- must stay exactly the same
    while the data changes, so a cache keyed on ``id()`` or ``data_ptr()``
    scores a hit and returns its stale answer.
    """
    backend = require_backend(dst, name)
    if backend == TORCH:
        torch = _torch()
        # .data rather than the tensor itself: copy_ into a leaf that requires
        # grad raises otherwise. Same address either way.
        target = getattr(dst, "data", dst)
        target.copy_(_as_torch(src, torch))
        return
    if backend == NUMPY:
        dst[...] = src
        return
    if backend == JAX:
        raise UnsupportedValueError(
            f"{name} is a JAX array, which is immutable, so its contents cannot "
            f"be overwritten in place. Pass the poison-test inputs as numpy "
            f"arrays (JAX accepts them directly) so the buffer can be mutated."
        )
    if backend == SEQUENCE:
        if not isinstance(dst, list):
            raise UnsupportedValueError(
                f"{name} is a tuple, which is immutable; use a list so its "
                f"contents can be overwritten in place."
            )
        if len(dst) != len(src):
            raise UnsupportedValueError(
                f"cannot overwrite {name}: length {len(dst)} vs source {len(src)}"
            )
        for index, element in enumerate(src):
            # Recurse into nested lists so inner row objects keep their
            # identity too -- a cache keyed on id(row) must still hit.
            if isinstance(dst[index], list) and isinstance(element, (list, tuple)):
                fill_from(dst[index], element, name)
            else:
                dst[index] = element
        return
    raise UnsupportedValueError(
        f"{name} is a {_type_label(dst)}, which is immutable, so it cannot be "
        f"overwritten in place."
    )


def random_like(value: Any, name: str = "input"):
    """Fresh random data with the same shape and dtype as ``value``.

    Guaranteed to differ from ``value`` element-wise wherever the dtype allows
    it. That guarantee matters because this is poison data for the memoization
    check: an element that happens to be redrawn to its original value is an
    element the poisoning did not poison, so a cached result still matches
    there and the check weakens silently.

    Drawing at random and hoping is not good enough at low cardinality. A bool
    element collides 50% of the time, so a small bool input let a memoizing
    kernel through outright; retrying only pushes the probability down
    (8 retries still leaves ~0.4% per call, which showed up as a ~2.5% flake
    across a suite). Forcing the difference removes the failure mode instead
    of shrinking it, and makes the check deterministic rather than lucky.
    """
    backend = require_backend(value, name)
    if backend == TORCH:
        return _force_differs_torch(_random_like_torch(value), value)
    if backend == NUMPY:
        return _force_differs_numpy(_random_like_numpy(value, name), value, name)
    if backend == SEQUENCE:
        return [random_like(element, name) for element in value]
    if backend == SCALAR:
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1
        candidate = random.gauss(0.0, 1.0)
        return candidate if candidate != value else value + 1.0
    raise UnsupportedValueError(
        f"cannot generate replacement data for {name} ({_type_label(value)})"
    )


def _force_differs_numpy(new: Any, old: Any, name: str):
    """Perturb any element of ``new`` that still equals ``old``."""
    numpy = require_numpy(name)
    if old.size == 0:
        return new
    same = new == old
    if not bool(numpy.any(same)):
        return new
    dtype = old.dtype
    if dtype == numpy.bool_:
        new[same] = ~old[same]
    elif numpy.issubdtype(dtype, numpy.integer):
        # +1 with wraparound is still a change for every width, including the
        # dtype maximum, where it wraps to the minimum.
        new[same] = (old[same] + numpy.ones((), dtype=dtype)).astype(dtype)
    else:
        new[same] = old[same] + numpy.ones((), dtype=dtype)
    return new


def _force_differs_torch(new: Any, old: Any):
    """Perturb any element of ``new`` that still equals ``old``."""
    torch = _torch()
    if old.numel() == 0:
        return new
    same = new == old
    if not bool(same.any()):
        return new
    if old.dtype == torch.bool:
        new[same] = ~old[same]
    else:
        new[same] = old[same] + torch.ones((), dtype=old.dtype)
    return new


def _random_like_torch(value: Any):
    torch = _torch()
    if value.is_floating_point() or value.is_complex():
        return torch.randn_like(value)
    if value.dtype == torch.bool:
        return torch.randint_like(value, low=0, high=2)
    low = 0 if str(value.dtype).startswith("torch.uint") else -100
    return torch.randint_like(value, low=low, high=100)


def _random_like_numpy(value: Any, name: str):
    numpy = require_numpy(name)
    rng = numpy.random.default_rng()
    dtype = value.dtype
    if numpy.issubdtype(dtype, numpy.floating):
        return rng.standard_normal(value.shape).astype(dtype)
    if numpy.issubdtype(dtype, numpy.complexfloating):
        real = rng.standard_normal(value.shape)
        imag = rng.standard_normal(value.shape)
        return (real + 1j * imag).astype(dtype)
    if dtype == numpy.bool_:
        return rng.integers(0, 2, size=value.shape).astype(bool)
    if numpy.issubdtype(dtype, numpy.unsignedinteger):
        return rng.integers(0, 100, size=value.shape).astype(dtype)
    if numpy.issubdtype(dtype, numpy.integer):
        return rng.integers(-100, 100, size=value.shape).astype(dtype)
    raise UnsupportedValueError(
        f"cannot generate replacement data for {name}: dtype {dtype} is not numeric"
    )


# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------

def _jax_ready(value: Any) -> Any:
    """Force a JAX array to be materialized, returning it."""
    blocker = getattr(value, "block_until_ready", None)
    if callable(blocker):
        result = blocker()
        return value if result is None else result
    jax = _live("jax")
    if jax is not None and hasattr(jax, "block_until_ready"):
        jax.block_until_ready(value)
    return value


def sync(value: Any = None) -> None:
    """Drain outstanding device work for whichever backend is live.

    With no argument this synchronizes every framework that is *already
    imported* -- torch CUDA/MPS and JAX. Restricting it to loaded modules
    keeps it free (and import-free) for a task that never touches them, while
    ensuring an asynchronous backend cannot leave work in flight past the end
    of a timed region.

    An unrecognized object syncs everything live rather than raising: this is
    a timing primitive, not a check, and over-syncing can only cost time,
    whereas under-syncing silently under-reports it.
    """
    if value is None:
        _sync_live_backends()
        return
    backend = backend_of(value)
    if backend == TORCH:
        _sync_torch_object(value)
        return
    if backend == JAX:
        _jax_ready(value)
        return
    if backend in (NUMPY, SEQUENCE, SCALAR):
        return
    _sync_live_backends()


def _mps_in_use(torch: Any) -> bool:
    """True if anything is allocated on MPS -- i.e. work could be in flight.

    Falls back to True when the allocator probe is unavailable (older torch),
    so an unknown state syncs rather than skips.
    """
    probe = getattr(getattr(torch, "mps", None), "current_allocated_memory", None)
    if not callable(probe):
        return True
    return bool(probe())


def _sync_torch_object(value: Any) -> None:
    """Sync the device a torch tensor (or torch.device) belongs to."""
    torch = _live("torch")
    if torch is None:  # pragma: no cover - a torch value implies torch is loaded
        return
    device = getattr(value, "device", value)
    device_type = getattr(device, "type", None)
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "mps":
        torch.mps.synchronize()


def _sync_live_backends() -> None:
    """Barrier every asynchronous backend that is actually in use.

    "In use" is deliberately narrower than "available": an idle
    ``torch.mps.synchronize()`` costs ~9 us, and `SyncTimer.stop` runs its
    barrier *inside* the measured window, so draining an accelerator a CPU-only
    task never touched would inflate every measurement it takes. Both probes
    below (a live CUDA context, a nonzero MPS allocation) cost well under a
    microsecond and are true whenever there can be work to wait for.
    """
    torch = _live("torch")
    if torch is not None:
        try:
            # False until a CUDA context exists in this process.
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - best-effort drain; never fail timing
            pass
        try:
            if torch.backends.mps.is_available() and _mps_in_use(torch):
                torch.mps.synchronize()
        except Exception:  # noqa: BLE001 - best-effort drain; never fail timing
            pass
    jax = _live("jax")
    if jax is not None:
        try:
            barrier = getattr(jax, "effects_barrier", None)
            if callable(barrier):
                barrier()
            else:
                jax.block_until_ready(jax.numpy.zeros(()))
        except Exception:  # noqa: BLE001 - best-effort drain; never fail timing
            pass
