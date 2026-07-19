"""Drift check behavior: env_passthrough forwarding and baseline re-measurement.

The periodic drift check must run the benchmark with the same task-declared
env vars as every other benchmark invocation, and detected drift (>5%) must
trigger a re-measure of the baseline snapshot instead of only warning.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

from perflab.optimizers.phases import evaluate as evaluate_mod


class _RecordingLog:
    def __init__(self):
        self.events: list[tuple] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.events.append((name, args, kwargs))
        return record


def _ctx(ws: Path, run_dir: Path, accepted_count: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(
            benchmark=SimpleNamespace(
                metric=SimpleNamespace(name="throughput.median", mode="maximize"),
                cmd="python bench.py", warmup=1, repeats=5,
            ),
            build=None,
            program_type="python",
            constraints=SimpleNamespace(
                rlimit_as_gb=None, env_passthrough=["OMP_NUM_THREADS"],
            ),
            out_dir=ws / "out",
        ),
        ws=ws,
        rp=SimpleNamespace(run_dir=run_dir),
        iteration=4,
        progress=SimpleNamespace(on_message=lambda m: None),
        event_log=_RecordingLog(),
        config=SimpleNamespace(isolation=None),
        accepted_count=accepted_count,
        baseline_val=10.0,
        latest_diagnostics=None,
    )


class TestDriftCheck:
    def _reprofile(self, tmp_path, monkeypatch, drift_value: float, accepted_count: int = 3):
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        ctx = _ctx(ws, tmp_path / "run", accepted_count=accepted_count)
        monkeypatch.setattr(
            evaluate_mod, "run_pipeline_for_ctx",
            lambda *a, **k: (None, None, None, None),
        )
        bench_calls: list[dict] = []

        def fake_benchmark(cmd, cwd, **kwargs):
            bench_calls.append(kwargs)
            return (None, {"ok": True, "throughput": {"median": drift_value}})

        monkeypatch.setattr(evaluate_mod, "run_benchmark", fake_benchmark)
        remeasure_calls: list[tuple] = []
        monkeypatch.setattr(
            evaluate_mod, "remeasure_baseline",
            lambda c, current_value=None: remeasure_calls.append((c, current_value)),
        )
        evaluate_mod.reprofile_after_accept(ctx, accepted_value=10.0)
        return ctx, bench_calls, remeasure_calls

    def test_drift_bench_forwards_env_passthrough(self, tmp_path, monkeypatch):
        _, bench_calls, _ = self._reprofile(tmp_path, monkeypatch, drift_value=10.2)
        assert len(bench_calls) == 1
        assert bench_calls[0]["env_passthrough"] == ["OMP_NUM_THREADS"]

    def test_large_drift_triggers_baseline_remeasure(self, tmp_path, monkeypatch):
        ctx, _, remeasure_calls = self._reprofile(tmp_path, monkeypatch, drift_value=20.0)
        # The drift benchmark's measurement (20.0) is passed through as
        # current_value so best_value can be re-anchored under the same
        # conditions as the baseline re-measure.
        assert remeasure_calls == [(ctx, 20.0)]

    def test_small_drift_does_not_remeasure(self, tmp_path, monkeypatch):
        _, _, remeasure_calls = self._reprofile(tmp_path, monkeypatch, drift_value=10.2)
        assert remeasure_calls == []

    def test_no_drift_check_off_cycle(self, tmp_path, monkeypatch):
        _, bench_calls, _ = self._reprofile(
            tmp_path, monkeypatch, drift_value=20.0, accepted_count=2,
        )
        assert bench_calls == []


class TestRemeasureBaseline:
    def _setup(self, tmp_path, with_zip: bool = True) -> SimpleNamespace:
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "main.py").write_text("optimized", encoding="utf-8")
        run_dir = tmp_path / "run"
        (run_dir / "snapshots").mkdir(parents=True)
        if with_zip:
            with zipfile.ZipFile(run_dir / "snapshots" / "baseline.zip", "w") as zf:
                zf.writestr("main.py", "baseline")
        return _ctx(ws, run_dir)

    def test_restores_baseline_sources_in_temp_copy(self, tmp_path, monkeypatch):
        ctx = self._setup(tmp_path)
        seen: dict = {}

        def fake_benchmark(cmd, cwd, **kwargs):
            seen["main_py"] = (Path(cwd) / "main.py").read_text(encoding="utf-8")
            seen["cwd"] = Path(cwd)
            seen["kwargs"] = kwargs
            return (None, {"ok": True, "throughput": {"median": 42.0}})

        monkeypatch.setattr(evaluate_mod, "run_benchmark", fake_benchmark)
        evaluate_mod.remeasure_baseline(ctx)

        assert seen["main_py"] == "baseline"
        assert seen["cwd"] != ctx.ws
        assert seen["kwargs"]["env_passthrough"] == ["OMP_NUM_THREADS"]
        assert ctx.baseline_val == 42.0
        assert ("baseline_remeasured", (4, 10.0, 42.0), {}) in ctx.event_log.events
        # Real workspace keeps the accepted (optimized) code
        assert (ctx.ws / "main.py").read_text(encoding="utf-8") == "optimized"

    def test_missing_snapshot_keeps_baseline(self, tmp_path, monkeypatch):
        ctx = self._setup(tmp_path, with_zip=False)
        bench_calls: list[int] = []
        monkeypatch.setattr(
            evaluate_mod, "run_benchmark",
            lambda *a, **k: bench_calls.append(1),
        )
        evaluate_mod.remeasure_baseline(ctx)
        assert ctx.baseline_val == 10.0
        assert bench_calls == []

    def test_benchmark_failure_keeps_baseline(self, tmp_path, monkeypatch):
        ctx = self._setup(tmp_path)

        def broken_benchmark(*a, **k):
            raise RuntimeError("bench exploded")

        monkeypatch.setattr(evaluate_mod, "run_benchmark", broken_benchmark)
        evaluate_mod.remeasure_baseline(ctx)
        assert ctx.baseline_val == 10.0
        assert all(name != "baseline_remeasured" for name, _, _ in ctx.event_log.events)

    def test_current_value_reanchors_best(self, tmp_path, monkeypatch):
        # When a current-conditions measurement is supplied, best_value is
        # re-anchored to it so final speedup = baseline/best compares both sides
        # under the same conditions.
        ctx = self._setup(tmp_path)
        ctx.best_value = 15.0  # measured under old conditions

        monkeypatch.setattr(
            evaluate_mod, "run_benchmark",
            lambda cmd, cwd, **k: (None, {"ok": True, "throughput": {"median": 42.0}}),
        )
        evaluate_mod.remeasure_baseline(ctx, current_value=99.0)

        assert ctx.baseline_val == 42.0
        assert ctx.best_value == 99.0
        ev = [e for e in ctx.event_log.events if e[0] == "baseline_remeasured"]
        assert ev and ev[0][2] == {"best_old": 15.0, "best_new": 99.0}

    def test_none_current_value_leaves_best(self, tmp_path, monkeypatch):
        # Without a current-conditions measurement, only the baseline moves.
        ctx = self._setup(tmp_path)
        ctx.best_value = 15.0

        monkeypatch.setattr(
            evaluate_mod, "run_benchmark",
            lambda cmd, cwd, **k: (None, {"ok": True, "throughput": {"median": 42.0}}),
        )
        evaluate_mod.remeasure_baseline(ctx)

        assert ctx.baseline_val == 42.0
        assert ctx.best_value == 15.0
        ev = [e for e in ctx.event_log.events if e[0] == "baseline_remeasured"]
        assert ev and ev[0] == ("baseline_remeasured", (4, 10.0, 42.0), {})
