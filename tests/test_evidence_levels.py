"""Tests for evidence-level provenance on BottleneckDiagnosis.

Covers the ``Evidence`` value object (perflab/analyzers/bottleneck_types.py),
its population across the bottleneck_{gpu,system,cpu}.py rules, and its
rendering in the LLM prompt and JSON emitters (reporting/generate.py,
server/analysis.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from perflab.analyzers.bottleneck_analyzer import compute_source_hints, diagnose_bottlenecks
from perflab.analyzers.bottleneck_types import BottleneckDiagnosis, Evidence

_VALID_LEVELS = {"observed", "derived", "inferred"}


class TestEvidenceDefaults:
    def test_defaults_from_original_five_fields(self):
        """A diagnosis built with only the original 5 fields gets derived defaults."""
        d = BottleneckDiagnosis(
            rank=1,
            bottleneck="Low SM utilization (23%)",
            root_cause="Insufficient parallelism",
            confidence="high",
            suggested_actions=["Increase batch size"],
        )
        assert d.evidence.level == "derived"
        assert d.evidence.rule_id == ""
        assert d.evidence.metrics == {}

    def test_evidence_default_factory_is_independent_per_instance(self):
        # default_factory=Evidence must not share mutable state across instances.
        a = BottleneckDiagnosis(rank=1, bottleneck="a", root_cause="", confidence="low")
        b = BottleneckDiagnosis(rank=2, bottleneck="b", root_cause="", confidence="low")
        assert a.evidence is not b.evidence
        assert a.evidence.metrics is not b.evidence.metrics
        a.evidence.metrics["x"] = 1.0
        assert b.evidence.metrics == {}

    def test_to_dict_shape(self):
        ev = Evidence(level="inferred", rule_id="io_hotspot_pct_high", metrics={"pct": 45.0})
        assert ev.to_dict() == {"level": "inferred", "rule_id": "io_hotspot_pct_high", "metrics": {"pct": 45.0}}

    def test_evidence_is_unhashable(self):
        # frozen=True auto-generates __hash__, but metrics is a plain dict, so
        # hashing must fail loudly rather than silently succeed on identity.
        ev = Evidence()
        try:
            hash(ev)
        except TypeError:
            pass
        else:
            raise AssertionError("Evidence with a dict field should be unhashable")


def _all_fixture_diagnoses() -> list[BottleneckDiagnosis]:
    """Run diagnose_bottlenecks over a broad set of profiler-summary fixtures.

    Reuses the shapes already exercised in test_bottleneck_analyzer.py /
    test_bottleneck_new.py, spanning GPU (ncu/nsys), CPU (linux_perf +
    source hints), JAX/TPU, torch trace (incl. MPS cross-profiler), and the
    memray/eBPF/lock/thread/power profilers.
    """
    diags: list[BottleneckDiagnosis] = []

    # -- GPU: ncu + nsys --
    diags += diagnose_bottlenecks(
        {
            "ncu": {
                "sm_utilization_pct": 10,
                "memory_throughput_pct": 95,
                "achieved_occupancy_pct": 10,
                "dominant_kernel": {
                    "name": "kern", "registers_per_thread": 200,
                    "memory_throughput_pct": 85, "compute_throughput_pct": 20,
                },
            },
            "nsys": {
                "cuda_kernel_time_ms": 50, "duration_s": 10,
                "avg_kernel_gap_us": 300, "api_overhead_ms": 500,
            },
        },
        "cuda", top_n=50,
    )

    # -- Host/device cross-analysis (nsys + linux_perf together, cpp/cuda only) --
    diags += diagnose_bottlenecks(
        {
            "nsys": {"cuda_kernel_time_ms": 100, "duration_s": 10, "gpu_active_pct": 10},
            "linux_perf": {"hotspots": [{"function": "cpu_fn", "pct": 50}]},
        },
        "cuda", top_n=50,
    )

    # -- CPU: perf rules + SIMD source-hint rule (inferred) --
    hints = compute_source_hints({"main.cpp": "int main(){return 0;}"}, "cpp")
    diags += diagnose_bottlenecks(
        {
            "linux_perf": {
                "ipc": 0.3, "cache_miss_rate": 0.2, "branch_miss_rate": 0.2,
                "hotspots": [{"function": "f", "pct": 60, "module": "m"}],
                "cpus_utilized": 1.0,
            },
        },
        "cpp", top_n=50, source_hints=hints, system_info={"cpu_count": 8},
    )

    # -- JAX / TPU --
    diags += diagnose_bottlenecks(
        {"jax": {"xla_recompilations": 3, "hlo_module_count": 20}},
        "jax", top_n=50,
    )

    # -- Torch trace: real GPU low ratio + excessive sync --
    diags += diagnose_bottlenecks(
        {
            "torch_profiler": {
                "cpu_vs_gpu": {"total_cpu_op_us": 100000.0, "total_gpu_kernel_us": 10000.0, "ratio": 0.1},
                "sync_count": 20, "total_sync_time_us": 5000,
                "top_gpu_kernels": [{"name": "sgemm", "total_us": 10000.0, "count": 5, "pct": 100.0}],
            },
        },
        "pytorch", top_n=50, device="cuda",
    )

    # -- MPS cross-profiler synthesis (explicit "inferred" example from the spec) --
    diags += diagnose_bottlenecks(
        {
            "torch_profiler": {"top_ops": [{"total_us": 50000.0}]},
            "metal_trace": {"gpu_time_total_ms": 1.0, "duration_s": 10},
        },
        "pytorch", top_n=50, device="mps",
    )

    # -- memray / eBPF / lock contention / thread sched / power --
    diags += diagnose_bottlenecks(
        {
            "memray": {"peak_memory_mb": 8000},
            "ebpf": {
                "read_latency": {"p99_ns": 50_000_000}, "write_latency": {},
                "read_syscalls": 0, "write_syscalls": 0,
            },
            "lock_contention": {
                "lock_stats": {"locks": [{"acquired": 100}], "total_contended": 50, "total_wait_ns": 0},
            },
            "thread_sched": {
                "latency": [{"task": "main", "avg_delay_ms": 10.0}],
                "timehist": {"migrations": 200},
            },
            "power": {
                "gpu_power": {
                    "power_samples": (
                        [{"watts": 300}, {"watts": 300}]
                        + [{"watts": 250}, {"watts": 200}]
                        + [{"watts": 150}, {"watts": 150}, {"watts": 150}, {"watts": 150}]
                    ),
                },
            },
        },
        "python", top_n=50,
    )

    return diags


class TestEvidenceAcrossFixtures:
    def test_at_least_one_of_each_kind_of_rule_fired(self):
        # Sanity check the fixtures actually exercise the analyzer (i.e. this
        # test isn't accidentally checking an empty list).
        diags = _all_fixture_diagnoses()
        assert len(diags) >= 10

    def test_evidence_level_always_one_of_three_literals(self):
        diags = _all_fixture_diagnoses()
        assert diags
        for d in diags:
            assert d.evidence.level in _VALID_LEVELS, (d.bottleneck, d.evidence.level)

    def test_every_diagnosis_has_a_rule_id(self):
        diags = _all_fixture_diagnoses()
        assert diags
        for d in diags:
            assert d.evidence.rule_id, (d.bottleneck, "missing rule_id")

    def test_source_hint_rule_is_inferred(self):
        """Rules that consume compute_source_hints (the cpp no-SIMD rule) are inferred."""
        diags = _all_fixture_diagnoses()
        simd_diags = [d for d in diags if "simd" in d.bottleneck.lower()]
        assert simd_diags, "expected the no-SIMD source-hint rule to have fired"
        for d in simd_diags:
            assert d.evidence.level == "inferred"

    def test_mixed_evidence_levels_present(self):
        # Not every rule should collapse to the same level -- both derived
        # (the common case) and inferred (cross-profiler/source-hint guesses)
        # must show up across this fixture set.
        diags = _all_fixture_diagnoses()
        levels = {d.evidence.level for d in diags}
        assert "derived" in levels
        assert "inferred" in levels


class TestPromptRendersEvidence:
    def _ctx(self, **kwargs):
        from perflab.optimizers.prompt import PromptContext
        defaults: dict = dict(source_files={}, profiler_summaries={}, bench_results={})
        defaults.update(kwargs)
        return PromptContext(**defaults)

    def _render(self, diag: dict) -> str:
        from perflab.optimizers.prompt import build_prompt
        ctx = self._ctx(bottleneck_diagnoses=[diag])
        msgs = build_prompt(ctx)
        return "\n".join(m.content for m in msgs)

    def test_inferred_marker_present(self):
        content = self._render({
            "rank": 1,
            "bottleneck": "I/O-bound (45% of samples in data loading)",
            "root_cause": "Data loading and preprocessing dominate execution time",
            "confidence": "medium",
            "suggested_actions": [],
            "evidence": {
                "level": "inferred", "rule_id": "io_hotspot_pct_high",
                "metrics": {"io_hotspot_pct": 45.0, "threshold": 20.0},
            },
        })
        assert "[inferred: io_hotspot_pct_high]" in content
        assert "io_hotspot_pct=45" in content

    def test_derived_diagnosis_has_no_inferred_marker(self):
        content = self._render({
            "rank": 1,
            "bottleneck": "Low SM utilization (23%)",
            "root_cause": "Insufficient parallelism or small kernel launches",
            "confidence": "high",
            "suggested_actions": [],
            "evidence": {
                "level": "derived", "rule_id": "ncu_sm_util_low",
                "metrics": {"sm_util_pct": 23.0, "threshold": 50.0},
            },
        })
        assert "[derived: ncu_sm_util_low]" in content
        assert "[inferred:" not in content

    def test_missing_evidence_key_degrades_gracefully(self):
        # Diagnoses from before this feature existed have no "evidence" key at
        # all -- must render without raising, defaulting to no marker shown.
        content = self._render({
            "rank": 1, "bottleneck": "Old-format diagnosis",
            "root_cause": "x", "confidence": "low", "suggested_actions": [],
        })
        assert "Old-format diagnosis" in content


class TestRoundTripJson:
    def test_generate_reports_emits_evidence(self, tmp_path: Path):
        from perflab.reporting.generate import ReportParams, generate_reports

        run_dir = tmp_path / "run-001"
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "ncu_summary.json").write_text(
            json.dumps({"sm_utilization_pct": 15}), encoding="utf-8"
        )
        params = ReportParams(
            run_dir=run_dir,
            run_id="run-001",
            task_name="matmul",
            metric_name="gflops",
            metric_mode="maximize",
            program_type="cuda",
            history=[{"iteration": 0, "value": 100.0, "accepted": True, "notes": "baseline"}],
            baseline_val=100.0,
            best_value=100.0,
            best_iter=0,
        )
        result = generate_reports(params)
        diags = result["bottleneck_diagnoses"]
        assert diags
        for d in diags:
            assert "evidence" in d
            assert set(d["evidence"]) == {"level", "rule_id", "metrics"}
            assert d["evidence"]["level"] in _VALID_LEVELS
            assert d["evidence"]["rule_id"]

        # report.json on disk carries the same shape (real round-trip, not just in-memory).
        report_json = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert report_json["bottleneck_diagnoses"][0]["evidence"]["level"] in _VALID_LEVELS

    def test_get_bottlenecks_emits_evidence(self, tmp_path: Path):
        from perflab.memory.run_store import RunStore
        from perflab.server.mcp_server import get_bottlenecks

        store = RunStore(tmp_path)
        rp = store.new_run("demo-task", program_type="cuda")
        (rp.artifacts_dir / "ncu_summary.json").write_text(
            json.dumps({"sm_utilization_pct": 5.0}), encoding="utf-8"
        )
        result = get_bottlenecks(rp.run_id, out_dir=str(tmp_path))
        assert result
        for d in result:
            assert "evidence" in d
            assert set(d["evidence"]) == {"level", "rule_id", "metrics"}
            assert d["evidence"]["level"] in _VALID_LEVELS
            assert d["evidence"]["rule_id"]
