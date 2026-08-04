"""Wiring: phases/evaluate.py must interleave only where it is worth paying for.

Three properties are pinned here.

* Selecting ``decision_rule: paired_difference`` switches the *authoritative*
  re-benchmark to the interleaved runner, and the accept decision is then made
  against the incumbent measured in that same interleaved run -- not against
  ``ctx.best_value``, which is the stale number pairing exists to stop using.
* The fast screen stays cheap. It is a ranking pass; doubling its cost to make
  it rigorous would defeat the reason it exists.
* Nothing changes for a task that does not opt in.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

from perflab.optimizers.phases import evaluate as evaluate_mod
from perflab.optimizers.phases.evaluate import BeamCandidate
from perflab.runners.paired import (
    CANDIDATE,
    INCUMBENT,
    BlockPlan,
    PairedRun,
    Spawn,
    SpawnMeasurement,
    interleave_order,
)
from perflab.task_spec import ContractSpec


class _RecordingLog:
    def __init__(self):
        self.events: list[tuple] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.events.append((name, args, kwargs))
        return record


def _ctx(tmp_path: Path, decision_rule: str | None, best_value: float = 100.0):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / "kernel.py").write_text("x = 1\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    return SimpleNamespace(
        task=SimpleNamespace(
            benchmark=SimpleNamespace(
                metric=SimpleNamespace(name="latency_ms.p50", mode="minimize"),
                cmd="python bench.py", warmup=1, repeats=12,
            ),
            build=None,
            program_type="python",
            out_dir=ws / "out",
            contract=ContractSpec(min_repeats=3, required_bench_fields=["ok"]),
            constraints=SimpleNamespace(
                regression_tolerance=0.02,
                decision_rule=decision_rule,
                noise_gate=True,
                cv_threshold=None,
                rlimit_as_gb=None,
                env_passthrough=None,
            ),
            anti_gaming=SimpleNamespace(gaming_speedup_threshold=10.0),
        ),
        ws=ws,
        rp=SimpleNamespace(run_dir=run_dir),
        iteration=3,
        progress=SimpleNamespace(on_message=lambda m: None),
        event_log=_RecordingLog(),
        config=SimpleNamespace(isolation=None, top_k=2),
        best_value=best_value,
        best_iter=0,
        baseline_val=200.0,
        accepted_count=0,
        history=[],
        accepted_patches=[],
        sec_metric=None,
        total_estimated_cost_usd=None,
    )


def _paired_run(diffs: list[float], incumbent: float = 100.0) -> PairedRun:
    """A PairedRun whose per-pair differences are exactly ``diffs`` (minimize)."""
    pairs = [(incumbent - d, incumbent) for d in diffs]
    # BlockPlan directly, not plan_blocks(): the helper fabricates runs with
    # pair counts plan_blocks refuses, so the rule's own defence can be tested.
    plan = BlockPlan(
        blocks=len(diffs), repeats_per_block=2, total_repeats=12,
        lead_in_spawns=0, counterbalanced=True,
    )
    order = interleave_order(len(diffs))
    spawns: list[Spawn] = []
    for i, (cand, inc) in enumerate(pairs):
        for arm, value in ((INCUMBENT, inc), (CANDIDATE, cand)):
            spawns.append(Spawn(
                arm=arm, pair=i, position=len(spawns),
                measurement=SpawnMeasurement(
                    value=value, samples=[value, value],
                    bench={"ok": True, "latency_ms": {"p50": value}},
                ),
                wall_s=0.1,
            ))
    return PairedRun(
        pairs=pairs,
        candidate_value=statistics.median([p[0] for p in pairs]),
        incumbent_value=statistics.median([p[1] for p in pairs]),
        candidate_blocks=[p[0] for p in pairs],
        incumbent_blocks=[p[1] for p in pairs],
        candidate_samples=[p[0] for p in pairs],
        incumbent_samples=[p[1] for p in pairs],
        candidate_bench={"ok": True}, incumbent_bench={"ok": True},
        order=order, spawns=spawns, lead_in=[], plan=plan, wall_s=9.9,
    )


def _candidate(value: float) -> BeamCandidate:
    return BeamCandidate(
        iteration=3, index=0, blocks=[], description="candidate 1",
        value=value, samples=[value] * 4,
    )


class _Harness:
    """Runs accept_best with both benchmark entry points stubbed out."""

    def __init__(self, monkeypatch, tmp_path, decision_rule, paired_run=None,
                 paired_error: Exception | None = None, full_value: float = 94.0,
                 best_value: float = 100.0):
        self.ctx = _ctx(tmp_path, decision_rule, best_value=best_value)
        self.messages: list[str] = []
        self.ctx.progress.on_message = self.messages.append
        self.paired_calls: list[dict] = []
        self.full_calls: list[dict] = []

        def fake_paired(cmd, **kwargs):
            # The two arm workspaces are torn down when the context manager
            # exits, so anything to be asserted about them has to be captured
            # while they still exist.
            self.paired_calls.append({
                "cmd": cmd,
                "incumbent_contents": sorted(
                    p.name for p in kwargs["incumbent_cwd"].iterdir()
                ),
                "candidate_contents": sorted(
                    p.name for p in kwargs["candidate_cwd"].iterdir()
                ),
                **kwargs,
            })
            if paired_error is not None:
                raise paired_error
            return paired_run

        def fake_full(cmd, cwd, **kwargs):
            self.full_calls.append({"cmd": cmd, "cwd": cwd, **kwargs})
            return None, {
                "ok": True,
                "meta": {"repeats": 12, "warmup": 1},
                "latency_ms": {"p50": full_value, "raw_values": [full_value] * 4},
            }

        monkeypatch.setattr(evaluate_mod, "run_paired_benchmark", fake_paired)
        monkeypatch.setattr(evaluate_mod, "run_benchmark", fake_full)
        monkeypatch.setattr(
            evaluate_mod, "snapshot_workspace", lambda *a, **k: None,
        )

    def run(self, candidate_value: float = 94.0, use_fast: bool = True):
        return evaluate_mod.accept_best(
            self.ctx, [_candidate(candidate_value)], use_fast,
        )

    def said(self, fragment: str) -> bool:
        return any(fragment in m for m in self.messages)


class TestOptIn:
    def test_default_rule_never_pays_for_interleaving(self, monkeypatch, tmp_path):
        h = _Harness(monkeypatch, tmp_path, decision_rule=None)
        accepted, _, _ = h.run()
        assert accepted is True
        assert h.paired_calls == []
        assert len(h.full_calls) == 1          # the ordinary full re-bench

    def test_paired_rule_switches_the_authoritative_measurement(
        self, monkeypatch, tmp_path,
    ):
        h = _Harness(
            monkeypatch, tmp_path, decision_rule="paired_difference",
            paired_run=_paired_run([6.0, 5.0, 7.0, 4.0, 6.5, 5.5]),
        )
        accepted, _, value = h.run()
        assert accepted is True
        assert len(h.paired_calls) == 1
        assert h.full_calls == []              # and does not also re-bench plainly
        # median of the six candidate block values, not any single block.
        assert value == pytest.approx(94.25)

    def test_the_paired_runner_gets_the_task_benchmark_settings(
        self, monkeypatch, tmp_path,
    ):
        h = _Harness(
            monkeypatch, tmp_path, decision_rule="paired_difference",
            paired_run=_paired_run([6.0, 5.0, 7.0, 4.0, 6.5, 5.5]),
        )
        h.run()
        call = h.paired_calls[0]
        assert call["cmd"] == "python bench.py"
        assert call["metric_name"] == "latency_ms.p50"
        assert call["total_repeats"] == 12
        assert call["warmup"] == 1
        assert call["program_type"] == "python"

    def test_the_two_arms_are_separate_disposable_copies_of_the_workspace(
        self, monkeypatch, tmp_path,
    ):
        h = _Harness(
            monkeypatch, tmp_path, decision_rule="paired_difference",
            paired_run=_paired_run([6.0, 5.0, 7.0, 4.0, 6.5, 5.5]),
        )
        h.run()
        call = h.paired_calls[0]
        a, b = call["incumbent_cwd"], call["candidate_cwd"]
        assert a != b
        assert a != h.ctx.ws and b != h.ctx.ws       # never the real workspace
        assert a.name == "ws" and b.name == "ws"     # same construction
        assert call["incumbent_contents"] == call["candidate_contents"] == ["kernel.py"]
        assert not a.exists() and not b.exists()     # and disposed of afterwards


class TestDecisionUsesTheInterleavedIncumbent:
    def test_stale_best_value_does_not_reach_the_decision(
        self, monkeypatch, tmp_path,
    ):
        """The candidate must be judged against the freshly-measured incumbent.

        ``ctx.best_value`` here is 130 -- a number recorded when the machine was
        much slower. Judged against it, a 94 ms candidate looks like a 28% win.
        Judged against the incumbent measured moments earlier in the same
        interleaved run (100 ms), it is a 6% win that still has to clear the
        paired test. Only the latter is a real comparison.
        """
        h = _Harness(
            monkeypatch, tmp_path, decision_rule="paired_difference",
            paired_run=_paired_run([6.0, 5.0, 7.0, 4.0, 6.5, 5.5]),
            best_value=130.0,
        )
        accepted, rel, _ = h.run()
        assert accepted is True
        assert h.said("machine drift since it was last benchmarked")

    def test_an_inconsistent_win_is_rejected_even_when_the_median_looks_good(
        self, monkeypatch, tmp_path,
    ):
        h = _Harness(
            monkeypatch, tmp_path, decision_rule="paired_difference",
            paired_run=_paired_run([20.0, -19.0, 21.0, -18.0, 22.0, -1.0]),
        )
        accepted, _, _ = h.run()
        assert accepted is False
        assert h.said("not significant under pairing")
        # The paired-specific noise advice, not the unpaired repeats formula.
        assert h.said("more repeats inside a block would not")
        assert not h.said("raise benchmark.repeats")


class TestFailsClosedInTheWiring:
    def test_a_broken_interleaved_run_falls_back_and_says_so(
        self, monkeypatch, tmp_path,
    ):
        h = _Harness(
            monkeypatch, tmp_path, decision_rule="paired_difference",
            paired_error=RuntimeError("benchmark exited with code 1"),
        )
        accepted, _, _ = h.run()
        assert len(h.paired_calls) == 1
        assert len(h.full_calls) == 1          # fell back to the plain re-bench
        assert h.said("Paired measurement unavailable")
        assert accepted is True
        # ... and the accept is labelled as having skipped the paired test.
        assert h.said("accepted WITHOUT the paired test")

    def test_a_contract_violation_in_any_block_falls_back(
        self, monkeypatch, tmp_path,
    ):
        run = _paired_run([6.0, 5.0, 7.0, 4.0, 6.5, 5.5])
        broken = list(run.spawns)
        broken[3] = Spawn(
            arm=broken[3].arm, pair=broken[3].pair, position=broken[3].position,
            measurement=SpawnMeasurement(value=94.0, samples=[], bench={}),
            wall_s=0.1,
        )
        object.__setattr__(run, "spawns", broken)
        h = _Harness(
            monkeypatch, tmp_path, decision_rule="paired_difference",
            paired_run=run,
        )
        h.run()
        assert h.said("contract violation in interleaved run")
        assert len(h.full_calls) == 1

    def test_too_few_blocks_to_run_the_test_does_not_accept_on_the_paired_path(
        self, monkeypatch, tmp_path,
    ):
        """A 4-pair run reaches the rule, which refuses to pretend it tested."""
        h = _Harness(
            monkeypatch, tmp_path, decision_rule="paired_difference",
            paired_run=_paired_run([6.0, 5.0, 7.0, 4.0]),
        )
        accepted, _, _ = h.run()
        assert accepted is True                # the unpaired fallback allows it
        assert h.said("accepted WITHOUT the paired test")


class TestScreenStaysCheap:
    def test_a_candidate_that_fails_the_screen_never_reaches_the_runner(
        self, monkeypatch, tmp_path,
    ):
        h = _Harness(
            monkeypatch, tmp_path, decision_rule="paired_difference",
            paired_run=_paired_run([6.0] * 6),
        )
        accepted, _, _ = h.run(candidate_value=99.9)   # a 0.1% "win"
        assert accepted is False
        assert h.paired_calls == []
        assert h.full_calls == []

    def test_the_rebench_budget_still_caps_interleaved_runs(
        self, monkeypatch, tmp_path,
    ):
        h = _Harness(
            monkeypatch, tmp_path, decision_rule="paired_difference",
            paired_run=_paired_run([0.1] * 6),        # never accepted
        )
        h.ctx.config.top_k = 1
        evaluate_mod.accept_best(
            h.ctx, [_candidate(94.0), _candidate(93.0), _candidate(92.0)], True,
        )
        assert len(h.paired_calls) == 1
        assert h.said("Re-bench budget")

    def test_non_fast_mode_with_the_default_rule_is_untouched(
        self, monkeypatch, tmp_path,
    ):
        """No screen, no re-bench: exactly the pre-existing behavior."""
        h = _Harness(monkeypatch, tmp_path, decision_rule=None)
        accepted, _, _ = h.run(use_fast=False)
        assert accepted is True
        assert h.paired_calls == []
        assert h.full_calls == []

    def test_non_fast_mode_with_the_paired_rule_still_interleaves(
        self, monkeypatch, tmp_path,
    ):
        h = _Harness(
            monkeypatch, tmp_path, decision_rule="paired_difference",
            paired_run=_paired_run([6.0, 5.0, 7.0, 4.0, 6.5, 5.5]),
        )
        accepted, _, _ = h.run(use_fast=False)
        assert accepted is True
        assert len(h.paired_calls) == 1


class TestContractEnforcement:
    def test_the_min_repeats_floor_moves_to_the_aggregate_not_away(
        self, monkeypatch, tmp_path,
    ):
        """Blocks are shorter than a full run by design; the total must still
        clear contract.min_repeats."""
        ctx = _ctx(tmp_path, "paired_difference")
        ctx.task.contract = ContractSpec(min_repeats=100, required_bench_fields=["ok"])
        run = _paired_run([6.0, 5.0, 7.0, 4.0, 6.5, 5.5])
        errors = evaluate_mod._validate_paired_contract(ctx, run)
        assert errors
        assert "below contract min_repeats=100" in errors[0]

    def test_fixed_params_are_checked_on_every_single_spawn(
        self, monkeypatch, tmp_path,
    ):
        ctx = _ctx(tmp_path, "paired_difference")
        ctx.task.contract = ContractSpec(
            fixed_params={"M": 256}, min_repeats=1, required_bench_fields=["ok"],
        )
        run = _paired_run([6.0, 5.0, 7.0, 4.0, 6.5, 5.5])
        errors = evaluate_mod._validate_paired_contract(ctx, run)
        assert len(errors) == 5                       # capped, but every spawn flagged
        assert all("fixed_param 'M'" in e for e in errors)
        assert errors[0].startswith("A block 0")

    def test_a_clean_run_validates(self, monkeypatch, tmp_path):
        ctx = _ctx(tmp_path, "paired_difference")
        run = _paired_run([6.0, 5.0, 7.0, 4.0, 6.5, 5.5])
        assert evaluate_mod._validate_paired_contract(ctx, run) == []
