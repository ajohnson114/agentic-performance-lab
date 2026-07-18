"""Tests for the grid search knob sweep in perflab optimize."""
from __future__ import annotations

from perflab.optimizers.propose_params import (
    KnobPatch,
    generate_sweep_candidates,
    propose_knob_sweep,
    sample_candidates,
)


class TestGenerateSweepCandidates:
    def test_single_knob(self):
        knobs = {"M": 512, "block_size": 16, "sweep": {"block_size": [16, 32, 64]}}
        candidates = generate_sweep_candidates(knobs)
        assert len(candidates) == 3
        assert candidates[0].new_knobs["block_size"] == 16
        assert candidates[1].new_knobs["block_size"] == 32
        assert candidates[2].new_knobs["block_size"] == 64

    def test_multi_knob_cartesian_product(self):
        knobs = {
            "M": 1024,
            "sweep": {
                "block_size": [16, 32],
                "num_workers": [0, 4],
            },
        }
        candidates = generate_sweep_candidates(knobs)
        assert len(candidates) == 4  # 2 x 2
        combos = [(c.new_knobs["block_size"], c.new_knobs["num_workers"]) for c in candidates]
        assert (16, 0) in combos
        assert (16, 4) in combos
        assert (32, 0) in combos
        assert (32, 4) in combos

    def test_sweep_key_stripped(self):
        knobs = {"M": 512, "sweep": {"block_size": [16, 32]}}
        candidates = generate_sweep_candidates(knobs)
        for c in candidates:
            assert "sweep" not in c.new_knobs

    def test_preserves_non_sweep_knobs(self):
        knobs = {"M": 512, "N": 512, "block_size": 16, "sweep": {"block_size": [32, 64]}}
        candidates = generate_sweep_candidates(knobs)
        for c in candidates:
            assert c.new_knobs["M"] == 512
            assert c.new_knobs["N"] == 512

    def test_no_sweep_section(self):
        knobs = {"M": 512, "block_size": 16}
        assert generate_sweep_candidates(knobs) == []

    def test_empty_sweep(self):
        knobs = {"M": 512, "sweep": {}}
        assert generate_sweep_candidates(knobs) == []

    def test_sweep_not_a_dict(self):
        knobs = {"M": 512, "sweep": "invalid"}
        assert generate_sweep_candidates(knobs) == []

    def test_empty_value_list(self):
        knobs = {"sweep": {"block_size": []}}
        assert generate_sweep_candidates(knobs) == []

    def test_description_format(self):
        knobs = {"sweep": {"block_size": [64], "dtype": ["fp16"]}}
        candidates = generate_sweep_candidates(knobs)
        assert len(candidates) == 1
        assert "block_size=64" in candidates[0].description
        assert "dtype=fp16" in candidates[0].description

    def test_large_grid(self):
        knobs = {"sweep": {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}}
        candidates = generate_sweep_candidates(knobs)
        assert len(candidates) == 27  # 3^3


class TestSampleCandidates:
    def test_under_limit_returns_all(self):
        candidates = [KnobPatch(description=str(i), new_knobs={"x": i}) for i in range(5)]
        result = sample_candidates(candidates, max_trials=10)
        assert len(result) == 5

    def test_over_limit_subsamples(self):
        candidates = [KnobPatch(description=str(i), new_knobs={"x": i}) for i in range(50)]
        result = sample_candidates(candidates, max_trials=10)
        assert len(result) == 10

    def test_deterministic(self):
        candidates = [KnobPatch(description=str(i), new_knobs={"x": i}) for i in range(50)]
        a = sample_candidates(candidates, max_trials=10, seed=42)
        b = sample_candidates(candidates, max_trials=10, seed=42)
        assert [c.description for c in a] == [c.description for c in b]

    def test_different_seeds_differ(self):
        candidates = [KnobPatch(description=str(i), new_knobs={"x": i}) for i in range(50)]
        a = sample_candidates(candidates, max_trials=10, seed=1)
        b = sample_candidates(candidates, max_trials=10, seed=2)
        assert [c.description for c in a] != [c.description for c in b]

    def test_exact_limit(self):
        candidates = [KnobPatch(description=str(i), new_knobs={"x": i}) for i in range(10)]
        result = sample_candidates(candidates, max_trials=10)
        assert len(result) == 10


class TestProposeKnobSweep:
    def test_sweep_takes_priority(self):
        knobs = {"torch_compile": False, "batch": 1, "sweep": {"batch": [1, 4, 16]}}
        candidates = propose_knob_sweep(knobs)
        # Should return sweep candidates, not legacy ones
        assert len(candidates) == 3
        assert all("sweep" not in c.new_knobs for c in candidates)

    def test_legacy_fallback_without_sweep(self):
        knobs = {"torch_compile": False, "batch": 1}
        candidates = propose_knob_sweep(knobs)
        # Legacy: torch_compile=True + batch=4, batch=16
        assert len(candidates) > 0
        descs = [c.description for c in candidates]
        assert any("torch_compile" in d for d in descs)

    def test_legacy_no_duplicates(self):
        knobs = {"torch_compile": True, "batch": 4}
        candidates = propose_knob_sweep(knobs)
        # Should not propose the current value
        for c in candidates:
            if "torch_compile" in c.description:
                assert c.new_knobs["torch_compile"] != True  # noqa: E712
            if "batch" in c.description:
                assert c.new_knobs["batch"] != 4
