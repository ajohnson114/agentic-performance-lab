"""Determinism guard tolerance-split tests.

The Phase-1 same-inputs reproducibility check runs the SAME binary twice and
asks whether it produces the SAME bits, so its default tolerance must be a
fixed strict 1e-5 -- never the task-declared PERFLAB_ACCURACY_TOLERANCE. A task
that legitimately loosens tolerance (e.g. 1e-2 for fp16 math) must not thereby
let run-to-run garbage from uninitialized buffers slip past the exact check
built to catch it. The Phase-3 reference comparison IS a genuine accuracy
question, so its default keeps tracking the env tolerance. An explicit caller
atol/rtol wins for both.

torch is faked (patch.dict sys.modules, per the repo mocking pattern) so the
tolerance-resolution logic is exercised deterministically without a real torch
install. The fake records the atol/rtol handed to every ``allclose`` call so
each phase's resolved tolerance can be asserted directly.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perflab.harness.determinism import assert_deterministic


class FakeTensor:
    """Minimal float-vector tensor stand-in for the determinism code path."""

    def __init__(self, values):
        self.values = [float(v) for v in values]

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.values)

    def float(self):
        return self

    def cpu(self):
        return self

    @property
    def dtype(self):
        return "float32"

    def __sub__(self, other):
        return FakeTensor(
            [a - b for a, b in zip(self.values, other.values, strict=True)]
        )

    def abs(self):
        return FakeTensor([abs(v) for v in self.values])

    def max(self):
        return FakeTensor([max(self.values)])

    def item(self):
        return self.values[0]


def _make_fake_torch(record: list[dict]):
    def allclose(a, b, atol=1e-8, rtol=1e-5):
        record.append({"atol": atol, "rtol": rtol})
        return all(
            abs(x - y) <= atol + rtol * abs(y)
            for x, y in zip(a.values, b.values, strict=True)
        )

    def equal(a, b):
        return a.values == b.values

    return SimpleNamespace(
        Tensor=FakeTensor,
        allclose=allclose,
        equal=equal,
        float16="float16",
        bfloat16="bfloat16",
    )


def test_phase1_strict_default_ignores_loose_env(monkeypatch):
    # A task declaring a loose 1e-2 accuracy tolerance must NOT loosen the
    # same-inputs reproducibility check: a 5e-3 run-to-run drift still fails.
    monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "1e-2")
    record: list[dict] = []
    fake = _make_fake_torch(record)

    seq = iter([1.0, 1.005, 1.010])  # run 0 vs run 1 differ by 5e-3

    def fn(*_):
        return FakeTensor([next(seq)])

    with patch.dict("sys.modules", {"torch": fake}):
        with pytest.raises(AssertionError, match="Non-deterministic"):
            assert_deterministic(
                fn=fn,
                input_factory=lambda: (FakeTensor([1.0]),),
                n_runs=2,
            )

    # Phase 1 resolved the strict 1e-5 default, not the 1e-2 env tolerance.
    assert record and record[0] == {"atol": 1e-5, "rtol": 1e-5}


def test_phase1_honors_explicit_caller_atol(monkeypatch):
    # An explicit atol passed by the task author (a deliberate choice in
    # tests.py) still wins for the reproducibility check.
    monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "1e-2")
    record: list[dict] = []
    fake = _make_fake_torch(record)

    seq = iter([1.0, 1.005, 1.010, 1.015])

    def fn(*_):
        return FakeTensor([next(seq)])

    with patch.dict("sys.modules", {"torch": fake}):
        # 5e-3 phase-1 drift is within the explicitly-passed 1e-2 tolerance.
        assert_deterministic(
            fn=fn,
            input_factory=lambda: (FakeTensor([1.0]),),
            n_runs=2,
            atol=1e-2,
        )

    assert record and record[0]["atol"] == 1e-2


def test_reference_default_tracks_env_tolerance(monkeypatch):
    # Phase 1 stays strict while the Phase-3 reference comparison inherits the
    # task-declared tolerance -- both defaults verified in a single run.
    monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "1e-2")
    record: list[dict] = []
    fake = _make_fake_torch(record)

    # Distinct value per call: Phase 1 is deterministic per input, Phase 2 sees
    # differing inputs/outputs (no false no-op), Phase 3 gets its own input.
    counter = iter([1.0, 2.0, 3.0, 4.0])

    def input_factory():
        return (FakeTensor([next(counter)]),)

    def fn(x):
        return FakeTensor([x.values[0] * 2.0])          # deterministic

    def reference_fn(x):
        return FakeTensor([x.values[0] * 2.0 + 0.005])  # 5e-3 off reference

    with patch.dict("sys.modules", {"torch": fake}):
        assert_deterministic(
            fn=fn,
            input_factory=input_factory,
            reference_fn=reference_fn,
            n_runs=3,
        )

    # Phase-1 comparisons used the strict default...
    assert record[:-1] and all(r == {"atol": 1e-5, "rtol": 1e-5} for r in record[:-1])
    # ...and the reference comparison tracked the 1e-2 env tolerance.
    assert record[-1] == {"atol": 1e-2, "rtol": 1e-2}
