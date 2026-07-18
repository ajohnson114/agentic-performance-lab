"""Tests for perflab/orchestrator.py public entrypoints (profile_only, optimize).

Runs are hermetic and fast: the fast_env fixture empties the profiler list
(so the orchestrator's do_profiles=True pipelines never shell out to
perf/nsys/py-spy even where those tools are installed), stubs system-info
collection, and disables roofline hardware auto-detection. Benchmarks are
stdlib+yaml scripts whose reported metric is a deterministic function of the
knob value in tuning.yaml.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

import perflab.orchestrator
import perflab.profilers
import perflab.tools.sysinfo
from perflab.orchestrator import optimize, profile_only
from perflab.task_spec import TaskSpec

TASK_YAML = textwrap.dedent("""\
    name: orch-test
    program_type: python
    build: null
    correctness:
      cmd: "python tests.py"
      expected_exit: 0
    benchmark:
      cmd: "python bench.py --json out/bench.json"
      metric:
        name: throughput.median
        mode: maximize
      warmup: 1
      repeats: 2
    edit_policy:
      allowed_paths:
        - "*.py"
    constraints:
      max_iters: 3
      regression_tolerance: 0.02
      rlimit_as_gb: 0
    contract:
      fixed_params: {}
      min_repeats: 1
      required_bench_fields:
        - ok
        - throughput.median
""")

# All bench scripts include a nonce so bench.json content changes every run
# (the anti-tamper hash check in perflab.runners.benchmark requires this).
_BENCH_COMMON = textwrap.dedent("""\
    import json, os, sys
    import yaml

    knobs = {}
    if os.path.exists("tuning.yaml"):
        knobs = yaml.safe_load(open("tuning.yaml", encoding="utf-8")) or {}
""")

_BENCH_WRITE = textwrap.dedent("""\
    payload = {
        "ok": True,
        "throughput": {"median": value},
        "meta": {"device": "cpu"},
        "nonce": os.urandom(8).hex(),
    }
    os.makedirs("out", exist_ok=True)
    with open("out/bench.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
""")

# Constant metric, no knob dependence.
CONSTANT_BENCH = _BENCH_COMMON + "value = 100.0\n" + _BENCH_WRITE

# Metric scales with the `scale` knob: sweep over [1, 2, 4] must pick 4.
SWEEP_BENCH = (
    _BENCH_COMMON
    + "value = 100.0 * float(knobs.get('scale', 1))\n"
    + _BENCH_WRITE
)

# Legacy-mode bench: crashes (without touching bench.json) when the
# torch_compile candidate is applied; constant metric otherwise.
CRASH_LEGACY_BENCH = (
    _BENCH_COMMON
    + "if knobs.get('torch_compile'):\n"
    + "    sys.exit(2)\n"
    + "value = 100.0\n"
    + _BENCH_WRITE
)

# Sweep-mode bench: crashes on scale=2, works for scale=1.
CRASH_SWEEP_BENCH = (
    _BENCH_COMMON
    + "if int(knobs.get('scale', 1)) == 2:\n"
    + "    sys.exit(2)\n"
    + "value = 100.0 * float(knobs.get('scale', 1))\n"
    + _BENCH_WRITE
)


def make_workspace(
    tmp_path: Path, bench_body: str, tuning: str | None = None
) -> TaskSpec:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "task.yaml").write_text(TASK_YAML, encoding="utf-8")
    (ws / "tests.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (ws / "bench.py").write_text(bench_body, encoding="utf-8")
    if tuning is not None:
        (ws / "tuning.yaml").write_text(tuning, encoding="utf-8")
    return TaskSpec.load(ws / "task.yaml")


@pytest.fixture
def fast_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make orchestrator runs hermetic: no profilers, no hardware probing.

    The orchestrator's baseline and final run_pipeline calls use
    do_profiles=True; an empty profiler list keeps them fast everywhere
    (on Linux CI, linux_perf.is_available() would otherwise be True and
    re-run the bench under perf).
    """
    monkeypatch.setattr(perflab.profilers, "select_profilers", lambda task: [])
    monkeypatch.setattr(
        perflab.tools.sysinfo,
        "collect_system_info",
        lambda: {"platform": "test-platform", "cpu_count": 1},
    )
    monkeypatch.setattr(perflab.orchestrator, "resolve_roofline", lambda task: None)


def _report(run_dir: Path) -> dict:
    return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


class TestProfileOnly:
    def test_profile_only_produces_reports_and_meta(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        task = make_workspace(tmp_path, CONSTANT_BENCH)
        run_dir = profile_only(task)

        assert (run_dir / "bench.json").exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "dashboard.html").exists()

        report = _report(run_dir)
        assert report["best_value"] == 100.0
        assert report["baseline_value"] == 100.0

        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "profiled"
        assert meta["best_value"] == 100.0

    def test_invalid_contract_rejected_before_running(self, tmp_path: Path) -> None:
        task = make_workspace(tmp_path, CONSTANT_BENCH)
        task.contract.required_bench_fields = ["bad..path"]
        with pytest.raises(ValueError, match="Invalid contract"):
            profile_only(task)
        with pytest.raises(ValueError, match="Invalid contract"):
            optimize(task)
        # Validation fires before any run directory is created
        assert not (task.workspace / "out" / "runs").exists()


class TestOptimizeNoTuning:
    def test_stops_after_baseline_without_tuning_yaml(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        task = make_workspace(tmp_path, CONSTANT_BENCH)  # no tuning.yaml
        run_dir = optimize(task)

        report = _report(run_dir)
        rows = report["rows"]
        assert rows[0]["notes"] == "baseline"
        assert rows[0]["accepted"] is True

        stop_rows = [r for r in rows if r["notes"] == "no tuning.yaml; stopping"]
        assert len(stop_rows) == 1
        assert stop_rows[0]["accepted"] is False
        assert stop_rows[0]["value"] == 100.0

        assert report["best_value"] == 100.0
        assert report["best_iter"] == 0
        assert (run_dir / "dashboard.html").exists()


class TestOptimizeSweep:
    def test_sweep_picks_best_candidate(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        tuning = "scale: 1\nsweep:\n  scale: [1, 2, 4]\n"
        task = make_workspace(tmp_path, SWEEP_BENCH, tuning=tuning)
        run_dir = optimize(task)

        report = _report(run_dir)
        rows = report["rows"]
        by_notes = {r["notes"]: r for r in rows}

        # scale=1 equals baseline -> below the 2% improvement bar
        assert by_notes["scale=1"]["accepted"] is False
        assert by_notes["scale=2"]["accepted"] is True
        assert by_notes["scale=2"]["value"] == 200.0
        assert by_notes["scale=4"]["accepted"] is True
        assert by_notes["scale=4"]["value"] == 400.0

        # Winner confirmed by a full re-benchmark
        assert by_notes["confirmed re-benchmark"]["value"] == 400.0
        assert by_notes["confirmed re-benchmark"]["accepted"] is True
        assert report["best_value"] == 400.0
        assert report["best_iter"] == 3

        # Winning knobs written back to the workspace, sweep section kept so
        # a second `perflab optimize` still runs in sweep mode
        final_knobs = yaml.safe_load(
            (task.workspace / "tuning.yaml").read_text(encoding="utf-8")
        )
        assert final_knobs["scale"] == 4
        assert final_knobs["sweep"] == {"scale": [1, 2, 4]}

        # Snapshots for baseline and each improving trial
        assert (run_dir / "knobs_iter0.yaml").exists()
        assert (run_dir / "knobs_trial2.yaml").exists()
        assert (run_dir / "knobs_trial3.yaml").exists()

    def test_sweep_crashing_candidate_recorded_and_run_continues(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        tuning = "scale: 1\nsweep:\n  scale: [1, 2]\n"
        task = make_workspace(tmp_path, CRASH_SWEEP_BENCH, tuning=tuning)
        run_dir = optimize(task)  # must not raise

        report = _report(run_dir)
        rows = report["rows"]
        error_rows = [r for r in rows if "(error:" in r["notes"]]
        assert len(error_rows) == 1
        assert "scale=2" in error_rows[0]["notes"]
        assert error_rows[0]["accepted"] is False
        # Failed trial carries the previous row's value forward
        assert error_rows[0]["value"] == 100.0
        assert report["best_value"] == 100.0
        assert report["best_iter"] == 0


class TestOptimizeLegacyLoop:
    def test_legacy_crashing_candidate_recorded_and_loop_continues(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        # No sweep: section -> legacy hardcoded knob sweep. First proposed
        # candidate (torch_compile=True) crashes the bench; the loop must
        # record it as a failed row and keep evaluating the batch candidates.
        tuning = "torch_compile: false\nbatch: 1\n"
        task = make_workspace(tmp_path, CRASH_LEGACY_BENCH, tuning=tuning)
        run_dir = optimize(task)  # must not raise

        report = _report(run_dir)
        rows = report["rows"]

        error_rows = [r for r in rows if "(error:" in r["notes"]]
        assert len(error_rows) == 1
        assert "torch_compile=True" in error_rows[0]["notes"]
        assert error_rows[0]["accepted"] is False
        assert error_rows[0]["value"] == 100.0

        # The loop went on after the crash and terminated normally
        assert rows[-1]["notes"] == "no improvement"
        assert report["best_value"] == 100.0
        assert report["best_iter"] == 0

        # Knobs restored to the original configuration after the crash
        final_knobs = yaml.safe_load(
            (task.workspace / "tuning.yaml").read_text(encoding="utf-8")
        )
        assert final_knobs == {"torch_compile": False, "batch": 1}
        assert (run_dir / "dashboard.html").exists()
