"""`perflab optimize` must not promote a knob configuration it cannot measure.

This is the behavior change: the knob-search path in orchestrator.py called the
bare ratio test directly (three separate sites), so any trial whose measured
value landed above `regression_tolerance` was written back into the user's
tuning.yaml as the new default — including pure run-to-run jitter. It now asks
perflab.analyzers.decision, the same module the agent beam search and the
ci-check regression path use.

Every test here fails on the old code except the two that pin what must NOT
change: a genuine win is still accepted, and a benchmark that publishes no
per-repeat samples still gets exactly the historical ratio test.

Runs are hermetic and fast in the same way tests/test_orchestrator.py's are:
no profilers, no hardware probing, stdlib+yaml bench scripts.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

import perflab.orchestrator
import perflab.profilers
import perflab.tools.sysinfo
from perflab.orchestrator import optimize
from perflab.task_spec import TaskSpec

TASK_YAML = textwrap.dedent("""\
    name: orch-noise-test
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
      repeats: 5
    edit_policy:
      allowed_paths:
        - "*.py"
    constraints:
      max_iters: 3
      regression_tolerance: 0.02
      rlimit_as_gb: 0
    {extra_constraints}contract:
      fixed_params: {}
      min_repeats: 1
      required_bench_fields:
        - ok
        - throughput.median
""")

# 5 samples spread +/-8% around the metric: CV = 6.3%, so the 95% interval
# half-width is 5.7% of the value and nothing below a ~12% win is resolvable.
# A 5% "win" therefore clears the 2% tolerance and dies on variance — the exact
# case the old bare-ratio knob search accepted.
_SAMPLED_BENCH = textwrap.dedent("""\
    import json, os
    import yaml

    knobs = {}
    if os.path.exists("tuning.yaml"):
        knobs = yaml.safe_load(open("tuning.yaml", encoding="utf-8")) or {}

    value = 100.0 * float(knobs.get("scale", 1))
    raw = [value * (1 + 0.08 * r) for r in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    payload = {
        "ok": True,
        "throughput": {"median": value, "raw_values": raw},
        "meta": {"device": "cpu"},
        "nonce": os.urandom(8).hex(),
    }
    os.makedirs("out", exist_ok=True)
    with open("out/bench.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
""")

# Same metric values, no per-repeat array: the decision has nothing to test
# with, so it must stay exactly the historical ratio.
_UNSAMPLED_BENCH = _SAMPLED_BENCH.replace(
    '"throughput": {"median": value, "raw_values": raw},',
    '"throughput": {"median": value},',
)

# Legacy (no sweep: section) mode: the hardcoded torch_compile candidate buys a
# 5% "win", the batch candidates buy nothing.
_LEGACY_SAMPLED_BENCH = textwrap.dedent("""\
    import json, os
    import yaml

    knobs = {}
    if os.path.exists("tuning.yaml"):
        knobs = yaml.safe_load(open("tuning.yaml", encoding="utf-8")) or {}

    value = 105.0 if knobs.get("torch_compile") else 100.0
    raw = [value * (1 + 0.08 * r) for r in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    payload = {
        "ok": True,
        "throughput": {"median": value, "raw_values": raw},
        "meta": {"device": "cpu"},
        "nonce": os.urandom(8).hex(),
    }
    os.makedirs("out", exist_ok=True)
    with open("out/bench.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
""")


def make_workspace(
    tmp_path: Path, bench_body: str, tuning: str | None = None,
    extra_constraints: str = "",
) -> TaskSpec:
    ws = tmp_path / "ws"
    ws.mkdir()
    # str.replace, not str.format: the YAML contains a literal `{}`.
    (ws / "task.yaml").write_text(
        TASK_YAML.replace("{extra_constraints}", extra_constraints), encoding="utf-8",
    )
    (ws / "tests.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (ws / "bench.py").write_text(bench_body, encoding="utf-8")
    if tuning is not None:
        (ws / "tuning.yaml").write_text(tuning, encoding="utf-8")
    return TaskSpec.load(ws / "task.yaml")


@pytest.fixture
def fast_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(perflab.profilers, "select_profilers", lambda task: [])
    monkeypatch.setattr(
        perflab.tools.sysinfo,
        "collect_system_info",
        lambda: {"platform": "test-platform", "cpu_count": 1},
    )
    monkeypatch.setattr(perflab.orchestrator, "resolve_roofline", lambda task: None)


def _report(run_dir: Path) -> dict:
    return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


class TestSweepRejectsWithinNoise:
    def test_five_percent_win_inside_the_noise_is_not_promoted(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        """The headline change. Old code: scale=1.05 accepted, best_value=105."""
        tuning = "scale: 1\nsweep:\n  scale: [1, 1.05]\n"
        task = make_workspace(tmp_path, _SAMPLED_BENCH, tuning=tuning)
        run_dir = optimize(task)

        report = _report(run_dir)
        rows = {r["notes"].split(":")[0]: r for r in report["rows"]}
        assert rows["scale=1.05"]["accepted"] is False
        assert rows["scale=1.05"]["value"] == pytest.approx(105.0)
        assert report["best_value"] == pytest.approx(100.0)
        assert report["best_iter"] == 0

    def test_the_rejection_says_why_in_the_report(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        """"Nothing won" and "won, but unmeasurably" must stay distinguishable
        after the run — otherwise the change just looks like a broken sweep."""
        tuning = "scale: 1\nsweep:\n  scale: [1, 1.05]\n"
        task = make_workspace(tmp_path, _SAMPLED_BENCH, tuning=tuning)
        run_dir = optimize(task)

        notes = {r["notes"] for r in _report(run_dir)["rows"]}
        noise_note = next(n for n in notes if n.startswith("scale=1.05"))
        assert "within noise" in noise_note
        assert "CV=6.3%" in noise_note
        assert "n=5" in noise_note
        # The trial that simply failed the tolerance keeps its bare description.
        assert "scale=1" in notes

    def test_rejection_is_logged(
        self, fast_env: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        tuning = "scale: 1\nsweep:\n  scale: [1, 1.05]\n"
        task = make_workspace(tmp_path, _SAMPLED_BENCH, tuning=tuning)
        with caplog.at_level("INFO", logger="perflab.orchestrator"):
            optimize(task)
        assert any(
            "rejected" in r.getMessage() and "within noise" in r.getMessage()
            for r in caplog.records
        )


class TestSweepStillAcceptsRealWins:
    def test_three_x_win_survives_the_same_noisy_measurement(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        """The gate must cost nothing on a genuine win — same 8% spread, 3x."""
        tuning = "scale: 1\nsweep:\n  scale: [1, 3]\n"
        task = make_workspace(tmp_path, _SAMPLED_BENCH, tuning=tuning)
        run_dir = optimize(task)

        report = _report(run_dir)
        rows = {r["notes"]: r for r in report["rows"]}
        assert rows["scale=3"]["accepted"] is True
        assert rows["scale=3"]["value"] == pytest.approx(300.0)
        # ...and the confirmation re-benchmark against the baseline holds.
        assert rows["confirmed re-benchmark"]["accepted"] is True
        assert report["best_value"] == pytest.approx(300.0)
        assert report["best_iter"] == 2

    def test_tolerance_only_rule_restores_the_old_behavior(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        """The escape hatch for a deterministic metric or an unpinnable box."""
        tuning = "scale: 1\nsweep:\n  scale: [1, 1.05]\n"
        task = make_workspace(
            tmp_path, _SAMPLED_BENCH, tuning=tuning,
            extra_constraints="  decision_rule: tolerance_only\n",
        )
        run_dir = optimize(task)

        report = _report(run_dir)
        rows = {r["notes"]: r for r in report["rows"]}
        assert rows["scale=1.05"]["accepted"] is True
        assert report["best_value"] == pytest.approx(105.0)

    def test_noise_gate_false_restores_the_old_behavior(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        tuning = "scale: 1\nsweep:\n  scale: [1, 1.05]\n"
        task = make_workspace(
            tmp_path, _SAMPLED_BENCH, tuning=tuning,
            extra_constraints="  noise_gate: false\n",
        )
        run_dir = optimize(task)
        assert _report(run_dir)["best_value"] == pytest.approx(105.0)


class TestNoSamplesIsUnchanged:
    def test_bare_ratio_when_the_harness_publishes_no_samples(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        """No per-repeat array: nothing to test with, so nothing changes."""
        tuning = "scale: 1\nsweep:\n  scale: [1, 1.05]\n"
        task = make_workspace(tmp_path, _UNSAMPLED_BENCH, tuning=tuning)
        run_dir = optimize(task)

        report = _report(run_dir)
        rows = {r["notes"]: r for r in report["rows"]}
        assert rows["scale=1.05"]["accepted"] is True
        assert report["best_value"] == pytest.approx(105.0)

    def test_unverified_acceptance_is_logged_not_silent(
        self, fast_env: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        tuning = "scale: 1\nsweep:\n  scale: [1, 1.05]\n"
        task = make_workspace(tmp_path, _UNSAMPLED_BENCH, tuning=tuning)
        with caplog.at_level("INFO", logger="perflab.orchestrator"):
            optimize(task)
        assert any(
            "without variance verification" in r.getMessage()
            for r in caplog.records
        )


class TestLegacyLoopRejectsWithinNoise:
    def test_within_noise_candidate_does_not_advance_the_loop(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        """Old code accepted torch_compile=True (105 vs 100) and iterated."""
        tuning = "torch_compile: false\nbatch: 1\n"
        task = make_workspace(tmp_path, _LEGACY_SAMPLED_BENCH, tuning=tuning)
        run_dir = optimize(task)

        report = _report(run_dir)
        assert report["best_value"] == pytest.approx(100.0)
        assert report["best_iter"] == 0
        last = report["rows"][-1]
        assert last["accepted"] is False
        # The summary row explains the stop rather than just saying "nothing".
        assert last["notes"].startswith("no improvement")
        assert "within noise" in last["notes"]
        assert "torch_compile" in last["notes"]

    def test_knobs_left_at_the_original_configuration(
        self, fast_env: None, tmp_path: Path
    ) -> None:
        import yaml

        tuning = "torch_compile: false\nbatch: 1\n"
        task = make_workspace(tmp_path, _LEGACY_SAMPLED_BENCH, tuning=tuning)
        optimize(task)

        final = yaml.safe_load(
            (task.workspace / "tuning.yaml").read_text(encoding="utf-8")
        )
        assert final == {"torch_compile": False, "batch": 1}
