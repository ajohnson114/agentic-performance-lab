"""Tests for HLO op attribution engine."""
from __future__ import annotations

from perflab.analyzers.hlo_attribution import (
    _categorize_op,
    _diagnose_op,
    compute_hlo_attribution,
)


class TestCategorizeOp:
    def test_compute_ops(self):
        assert _categorize_op("dot") == "compute"
        assert _categorize_op("convolution") == "compute"
        assert _categorize_op("reduce") == "compute"

    def test_memory_ops(self):
        assert _categorize_op("copy") == "memory"
        assert _categorize_op("pad") == "memory"
        assert _categorize_op("transpose") == "memory"
        assert _categorize_op("reshape") == "memory"

    def test_communication_ops(self):
        assert _categorize_op("all-reduce") == "communication"
        assert _categorize_op("all-gather") == "communication"
        assert _categorize_op("collective-permute") == "communication"

    def test_control_ops(self):
        assert _categorize_op("while") == "control"
        assert _categorize_op("conditional") == "control"
        assert _categorize_op("fusion") == "control"

    def test_elementwise_defaults_to_compute(self):
        assert _categorize_op("add") == "compute"
        assert _categorize_op("multiply") == "compute"
        assert _categorize_op("tanh") == "compute"


class TestDiagnoseOp:
    def test_dominant_dot(self):
        diagnosis, suggestions = _diagnose_op("dot", 100, 40.0, 250)
        assert "Matrix multiplications" in diagnosis
        assert any("bf16" in s.lower() for s in suggestions)

    def test_padding_waste(self):
        diagnosis, suggestions = _diagnose_op("pad", 50, 20.0, 250)
        assert "padding" in diagnosis.lower()
        assert len(suggestions) > 0

    def test_copy_overhead(self):
        diagnosis, suggestions = _diagnose_op("copy", 30, 15.0, 200)
        assert "copies" in diagnosis.lower() or "copy" in diagnosis.lower()

    def test_collective_comm(self):
        diagnosis, suggestions = _diagnose_op("all-reduce", 10, 5.0, 200)
        assert "communication" in diagnosis.lower() or "collective" in diagnosis.lower()

    def test_infeed(self):
        diagnosis, suggestions = _diagnose_op("infeed", 5, 2.0, 250)
        assert "transfer" in diagnosis.lower() or "infeed" in diagnosis.lower()

    def test_generic_dominant_op(self):
        diagnosis, suggestions = _diagnose_op("some_unusual_op", 100, 25.0, 400)
        assert "25%" in diagnosis

    def test_small_op_no_diagnosis(self):
        diagnosis, suggestions = _diagnose_op("add", 5, 2.0, 250)
        assert diagnosis == ""
        assert suggestions == []


class TestComputeHloAttribution:
    def test_basic_attribution(self):
        jax_summary = {
            "hlo_ops": [
                {"op": "dot", "count": 100},
                {"op": "add", "count": 50},
                {"op": "pad", "count": 20},
                {"op": "reshape", "count": 30},
            ],
            "hlo_module_count": 3,
        }
        result = compute_hlo_attribution(jax_summary)
        assert result is not None
        assert result.total_ops == 200
        assert result.total_modules == 3
        assert len(result.entries) == 4
        # dot should be ranked first (highest weighted cost)
        assert result.entries[0].op == "dot"
        assert result.entries[0].category == "compute"
        assert result.entries[0].estimated_device_pct > 0

    def test_empty_ops(self):
        result = compute_hlo_attribution({"hlo_ops": []})
        assert result is None

    def test_no_ops_key(self):
        result = compute_hlo_attribution({})
        assert result is None

    def test_with_trace_metrics(self):
        jax_summary = {
            "hlo_ops": [{"op": "dot", "count": 50}],
            "host_time_us": 1000.0,
            "device_time_us": 9000.0,
            "device_fraction": 0.9,
        }
        result = compute_hlo_attribution(jax_summary, trace_metrics=jax_summary)
        assert result is not None
        assert result.host_time_us == 1000.0
        assert result.device_time_us == 9000.0
        assert result.device_fraction == 0.9

    def test_entries_sorted_by_device_pct(self):
        jax_summary = {
            "hlo_ops": [
                {"op": "reshape", "count": 100},  # many but cheap (0.1 weight = 10)
                {"op": "dot", "count": 10},       # few but expensive (10.0 weight = 100)
            ],
        }
        result = compute_hlo_attribution(jax_summary)
        assert result is not None
        # dot should be ranked first despite fewer count (higher weighted cost)
        assert result.entries[0].op == "dot"

    def test_pct_of_ops_correct(self):
        jax_summary = {
            "hlo_ops": [
                {"op": "dot", "count": 25},
                {"op": "add", "count": 75},
            ],
        }
        result = compute_hlo_attribution(jax_summary)
        assert result is not None
        dot_entry = next(e for e in result.entries if e.op == "dot")
        add_entry = next(e for e in result.entries if e.op == "add")
        assert dot_entry.pct_of_ops == 25.0
        assert add_entry.pct_of_ops == 75.0

    def test_communication_ops_categorized(self):
        jax_summary = {
            "hlo_ops": [
                {"op": "dot", "count": 50},
                {"op": "all-reduce", "count": 10},
            ],
        }
        result = compute_hlo_attribution(jax_summary)
        assert result is not None
        ar_entry = next(e for e in result.entries if e.op == "all-reduce")
        assert ar_entry.category == "communication"

    def test_diagnoses_populated(self):
        jax_summary = {
            "hlo_ops": [
                {"op": "dot", "count": 200},
                {"op": "pad", "count": 100},
                {"op": "add", "count": 50},
            ],
        }
        result = compute_hlo_attribution(jax_summary)
        assert result is not None
        dot_entry = next(e for e in result.entries if e.op == "dot")
        assert dot_entry.diagnosis != ""
        assert len(dot_entry.suggestions) > 0
