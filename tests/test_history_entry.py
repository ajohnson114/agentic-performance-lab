"""Tests for perflab.optimizers.history.make_history_entry."""
from __future__ import annotations

from perflab.optimizers.history import make_history_entry


class TestBaseFields:
    def test_base_keys_and_values(self):
        entry = make_history_entry(3, "swap loops", 12.0, 10.0, accepted=True)
        assert entry == {
            "iteration": 3,
            "description": "swap loops",
            "value": 12.0,
            "accepted": True,
            "delta": 2.0,
            "speedup": 1.2,
        }

    def test_key_order_matches_hand_rolled_dicts(self):
        entry = make_history_entry(1, "d", 2.0, 4.0, accepted=False)
        assert list(entry) == [
            "iteration", "description", "value", "accepted", "delta", "speedup",
        ]


class TestDeltaSpeedupMath:
    def test_regression_value_below_baseline(self):
        entry = make_history_entry(1, "slower", 2.0, 4.0, accepted=False)
        assert entry["delta"] == -2.0
        assert entry["speedup"] == 0.5

    def test_baseline_zero_speedup_is_one(self):
        entry = make_history_entry(0, "baseline", 0.0, 0.0, accepted=True)
        assert entry["delta"] == 0.0
        assert entry["speedup"] == 1.0

    def test_value_equals_baseline(self):
        entry = make_history_entry(0, "baseline", 7.5, 7.5, accepted=True)
        assert entry["delta"] == 0.0
        assert entry["speedup"] == 1.0


class TestMinimizeMode:
    def test_improvement_speedup_above_one(self):
        # Latency 10ms -> 5ms is a 2x speedup, not 0.5x
        entry = make_history_entry(1, "halved latency", 5.0, 10.0, accepted=True, mode="minimize")
        assert entry["speedup"] == 2.0
        assert entry["delta"] == -5.0

    def test_regression_speedup_below_one(self):
        entry = make_history_entry(1, "slower", 20.0, 10.0, accepted=False, mode="minimize")
        assert entry["speedup"] == 0.5

    def test_zero_value_speedup_is_one(self):
        entry = make_history_entry(1, "d", 0.0, 10.0, accepted=False, mode="minimize")
        assert entry["speedup"] == 1.0

    def test_default_mode_is_maximize(self):
        assert make_history_entry(1, "d", 5.0, 10.0, accepted=False)["speedup"] == 0.5


class TestExtras:
    def test_none_extras_dropped(self):
        entry = make_history_entry(
            1, "d", 1.0, 1.0, accepted=True,
            secondary_value=None, bench_wall_time_s=None, reasoning=None,
        )
        assert "secondary_value" not in entry
        assert "bench_wall_time_s" not in entry
        assert "reasoning" not in entry

    def test_truthy_extras_kept(self):
        entry = make_history_entry(
            1, "d", 1.0, 1.0, accepted=True,
            reasoning="unrolled the loop", secondary_value=0.25,
        )
        assert entry["reasoning"] == "unrolled the loop"
        assert entry["secondary_value"] == 0.25

    def test_falsy_but_not_none_extras_kept(self):
        # Only None is dropped; 0.0 and False are real values.
        entry = make_history_entry(
            1, "d", 1.0, 1.0, accepted=True,
            secondary_value=0.0, profiling_overhead_pct=0.0,
        )
        assert entry["secondary_value"] == 0.0
        assert entry["profiling_overhead_pct"] == 0.0

    def test_extras_appended_after_base_fields_in_order(self):
        entry = make_history_entry(
            1, "d", 1.0, 1.0, accepted=True,
            reasoning="r", secondary_value=2.0,
        )
        assert list(entry)[-2:] == ["reasoning", "secondary_value"]


class TestAcceptedFlag:
    def test_accepted_true(self):
        assert make_history_entry(1, "d", 1.0, 1.0, accepted=True)["accepted"] is True

    def test_accepted_false(self):
        assert make_history_entry(1, "d", 1.0, 1.0, accepted=False)["accepted"] is False


class TestAutotuneHistoryMode:
    """FIX 7: autotune.run must pass the metric mode so minimize-mode sweeps
    record a mode-aware speedup (a 2x latency win, not 0.5x)."""

    def test_minimize_sweep_records_mode_aware_speedup(self, monkeypatch):
        from types import SimpleNamespace

        from perflab.optimizers.phases import autotune as autotune_mod

        ctx = SimpleNamespace(
            task=SimpleNamespace(
                benchmark=SimpleNamespace(metric=SimpleNamespace(mode="minimize")),
            ),
            iteration=3,
            best_value=10.0,
            baseline_val=10.0,
            history=[],
        )
        # Sweep finds a better (lower) latency of 5.0.
        monkeypatch.setattr(
            autotune_mod, "_auto_tune_sweep", lambda ctx, max_trials=15: 5.0,
        )
        autotune_mod.run(ctx)

        assert len(ctx.history) == 1
        entry = ctx.history[0]
        assert entry["value"] == 5.0
        assert entry["speedup"] == 2.0  # 10 -> 5 latency is 2x, not 0.5x

    def test_maximize_sweep_speedup_unchanged(self, monkeypatch):
        from types import SimpleNamespace

        from perflab.optimizers.phases import autotune as autotune_mod

        ctx = SimpleNamespace(
            task=SimpleNamespace(
                benchmark=SimpleNamespace(metric=SimpleNamespace(mode="maximize")),
            ),
            iteration=1,
            best_value=10.0,
            baseline_val=10.0,
            history=[],
        )
        monkeypatch.setattr(
            autotune_mod, "_auto_tune_sweep", lambda ctx, max_trials=15: 20.0,
        )
        autotune_mod.run(ctx)
        assert ctx.history[0]["speedup"] == 2.0  # 10 -> 20 throughput is 2x
