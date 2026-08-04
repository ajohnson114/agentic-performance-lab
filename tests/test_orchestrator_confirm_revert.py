"""A sweep whose winner fails confirmation must not leave that winner behind.

`perflab optimize` writes the winning knobs to tuning.yaml *before* running the
confirmation re-benchmark. If the confirmation then fails, the file has already
been overwritten -- so without an explicit revert the sweep hands the user a
configuration whose improvement it just rejected. That is the gate computing the
right answer and nothing acting on it.

The window has always been there. It was near-unreachable while confirmation was
a bare ratio, because anything that won a sweep also cleared a 2% threshold; a
gate that can answer "not distinguishable from noise" makes it the common path.
Observed for real on a dataloader sweep: a 25.9% apparent win at CV=18.2%,
rejected as noise, and written to tuning.yaml anyway.
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
from perflab.orchestrator import optimize
from perflab.task_spec import TaskSpec

TASK_YAML = textwrap.dedent("""\
    name: "revert_probe"
    program_type: "python"
    correctness: { cmd: "python tests.py", expected_exit: 0 }
    benchmark:
      cmd: "python bench.py"
      metric: { name: "throughput.median", mode: "maximize" }
      warmup: 0
      repeats: 5
    constraints:
      max_iters: 4
      regression_tolerance: 0.02
""")

ORIGINAL_TUNING = "flag: false\nsweep:\n  flag: [false, true]\n"

# A fluke: the knob looks like a 3x win while the sweep measures it, then the
# effect evaporates by the time the confirmation re-benchmark runs. Call 4 is
# the confirmation (1 baseline + 2 trials + 1 confirm).
FLUKE_BENCH = textwrap.dedent("""\
    import json, os, yaml
    counter = "calls.txt"
    n = int(open(counter).read()) + 1 if os.path.exists(counter) else 1
    open(counter, "w").write(str(n))
    knobs = yaml.safe_load(open("tuning.yaml", encoding="utf-8")) or {}
    value = 300.0 if (knobs.get("flag") and n <= 3) else 100.0
    os.makedirs("out", exist_ok=True)
    json.dump(
        {"ok": True,
         "throughput": {"median": value, "raw_values": [value] * 5},
         "meta": {"device": "cpu"},
         "nonce": os.urandom(8).hex()},
        open("out/bench.json", "w"),
    )
""")


@pytest.fixture
def fast_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(perflab.profilers, "select_profilers", lambda task: [])
    monkeypatch.setattr(
        perflab.tools.sysinfo, "collect_system_info",
        lambda: {"platform": "test", "cpu_count": 1},
    )
    monkeypatch.setattr(perflab.orchestrator, "resolve_roofline", lambda task: None)


@pytest.fixture
def workspace(tmp_path: Path) -> TaskSpec:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "task.yaml").write_text(TASK_YAML, encoding="utf-8")
    (ws / "tests.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (ws / "bench.py").write_text(FLUKE_BENCH, encoding="utf-8")
    (ws / "tuning.yaml").write_text(ORIGINAL_TUNING, encoding="utf-8")
    return TaskSpec.load(ws / "task.yaml")


def _tuning(task: TaskSpec) -> dict:
    return yaml.safe_load((task.workspace / "tuning.yaml").read_text(encoding="utf-8"))


class TestUnconfirmedWinnerIsNotPersisted:
    def test_tuning_yaml_is_reverted_when_confirmation_fails(
        self, fast_env: None, workspace: TaskSpec, tmp_path: Path
    ):
        optimize(workspace, max_trials=4)
        knobs = _tuning(workspace)
        assert knobs["flag"] is False, (
            "the unconfirmed winner was left in tuning.yaml — the sweep handed "
            "back a configuration whose improvement it rejected"
        )

    def test_the_sweep_section_survives_the_revert(
        self, fast_env: None, workspace: TaskSpec, tmp_path: Path
    ):
        """Reverting must not destroy the user's sweep definition."""
        optimize(workspace, max_trials=4)
        assert _tuning(workspace)["sweep"] == {"flag": [False, True]}

    def test_the_report_says_it_reverted(
        self, fast_env: None, workspace: TaskSpec, tmp_path: Path
    ):
        """Silently undoing a change is its own kind of confusing."""
        run_dir = optimize(workspace, max_trials=4)
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        notes = " ".join(str(r.get("notes", "")) for r in report.get("rows", []))
        assert "did not hold" in notes
        assert "reverted" in notes


class TestConfirmedWinnerIsStillKept:
    """The revert must not fire when the win is real."""

    def test_a_durable_win_stays_in_tuning_yaml(
        self, fast_env: None, tmp_path: Path
    ):
        ws = tmp_path / "ws2"
        ws.mkdir()
        (ws / "task.yaml").write_text(TASK_YAML, encoding="utf-8")
        (ws / "tests.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
        # Same 3x win, but it holds on every call including the confirmation.
        (ws / "bench.py").write_text(textwrap.dedent("""\
            import json, os, yaml
            knobs = yaml.safe_load(open("tuning.yaml", encoding="utf-8")) or {}
            value = 300.0 if knobs.get("flag") else 100.0
            os.makedirs("out", exist_ok=True)
            json.dump(
                {"ok": True,
                 "throughput": {"median": value, "raw_values": [value] * 5},
                 "meta": {"device": "cpu"},
                 "nonce": os.urandom(8).hex()},
                open("out/bench.json", "w"),
            )
        """), encoding="utf-8")
        (ws / "tuning.yaml").write_text(ORIGINAL_TUNING, encoding="utf-8")
        task = TaskSpec.load(ws / "task.yaml")

        optimize(task, max_trials=4)
        assert _tuning(task)["flag"] is True
