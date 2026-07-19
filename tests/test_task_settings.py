"""Documented task.yaml settings must actually be honored at runtime.

Covers the settings that were previously parsed (or documented) but ignored:
allow_fast_math / accuracy_tolerance parsing, per-task benchmark warmup and
repeats, contract min_repeats/min_warmup enforcement, agent top_k, and the
profiler allow_sudo config knob.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

from perflab.task_spec import ContractSpec, TaskSpec
from perflab.tools.shell import CmdResult


def _write_task(tmp_path: Path, extra_constraints: str = "") -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / "out").mkdir(exist_ok=True)
    task_file = ws / "task.yaml"
    base = textwrap.dedent("""\
        name: settings-task
        program_type: python
        build: null
        correctness:
          cmd: "python tests.py"
        benchmark:
          cmd: "python bench.py --json out/bench.json"
          metric:
            name: throughput.median
            mode: maximize
          warmup: 7
          repeats: 9
        constraints:
          max_iters: 5
    """)
    task_file.write_text(base + extra_constraints, encoding="utf-8")
    return task_file


class TestConstraintsParsing:
    def test_allow_fast_math_and_accuracy_tolerance_parsed(self, tmp_path):
        task_file = _write_task(tmp_path, extra_constraints=(
            "  allow_fast_math: true\n"
            "  accuracy_tolerance: \"1e-3\"\n"
        ))
        task = TaskSpec.load(task_file)
        assert task.constraints.allow_fast_math is True
        assert task.constraints.accuracy_tolerance == "1e-3"

    def test_defaults_when_absent(self, tmp_path):
        task = TaskSpec.load(_write_task(tmp_path))
        assert task.constraints.allow_fast_math is False
        assert task.constraints.accuracy_tolerance is None


class TestMinSamplingEnforcement:
    def _contract(self, min_repeats=1, min_warmup=0):
        return ContractSpec(min_repeats=min_repeats, min_warmup=min_warmup)

    def test_reported_below_min_is_violation(self):
        from perflab.runners.benchmark import validate_contract
        bench = {"ok": True, "meta": {"repeats": 3, "warmup": 0}}
        errors = validate_contract(bench, self._contract(min_repeats=10, min_warmup=2))
        assert any("min_repeats=10" in e for e in errors)
        assert any("min_warmup=2" in e for e in errors)

    def test_reported_at_min_passes(self):
        from perflab.runners.benchmark import validate_contract
        bench = {"ok": True, "meta": {"repeats": 10, "warmup": 2}}
        assert validate_contract(bench, self._contract(min_repeats=10, min_warmup=2)) == []

    def test_missing_meta_with_explicit_min_warns_not_errors(self, caplog):
        # A legacy harness that never reports meta.repeats must not fail the
        # candidate: bench.py is tamper-protected, so a missing field means
        # legacy harness, not gaming. Enforcement is skipped with a warning.
        import logging

        from perflab.runners.benchmark import validate_contract
        bench = {"ok": True, "meta": {}}
        with caplog.at_level(logging.WARNING, logger="perflab.runners.benchmark"):
            errors = validate_contract(bench, self._contract(min_repeats=10))
        assert errors == []
        assert any(
            "min_repeats" in r.getMessage() and "not enforced" in r.getMessage()
            for r in caplog.records
        )

    def test_reported_below_min_still_errors_when_meta_present(self):
        # Present-but-below-minimum stays a hard error (only *missing* is soft).
        from perflab.runners.benchmark import validate_contract
        bench = {"ok": True, "meta": {"repeats": 3}}
        errors = validate_contract(bench, self._contract(min_repeats=10))
        assert any("min_repeats=10" in e for e in errors)

    def test_missing_meta_with_default_min_passes(self):
        # Legacy harnesses that don't report meta.repeats/warmup must not
        # start failing under the dataclass defaults (min_repeats=1).
        from perflab.runners.benchmark import validate_contract
        assert validate_contract({"ok": True}, self._contract()) == []

    def test_fast_screen_skips_min_sampling(self):
        from perflab.runners.benchmark import validate_contract
        bench = {"ok": True, "meta": {"repeats": 2, "warmup": 0}}
        contract = self._contract(min_repeats=10, min_warmup=3)
        assert validate_contract(bench, contract, enforce_min_sampling=False) == []
        assert validate_contract(bench, contract) != []


class _BenchRunCmdFake:
    """Fake run_cmd for run_benchmark: records env, writes bench.json."""

    def __init__(self, cwd: Path):
        self.env: dict | None = None
        self._cwd = cwd

    def __call__(self, cmd, cwd=None, env=None, **kwargs):
        self.env = dict(env or {})
        out = Path(cwd) / "out"
        out.mkdir(exist_ok=True)
        (out / "bench.json").write_text(
            json.dumps({"ok": True, "throughput": {"median": 1.0}}), encoding="utf-8",
        )
        return CmdResult(cmd=list(cmd), returncode=0, stdout="", stderr="", duration_s=0.01)


class TestPerTaskWarmupRepeats:
    def test_task_values_reach_bench_env(self, tmp_path, monkeypatch):
        import perflab.runners.benchmark as benchmark_mod

        monkeypatch.delenv("PERFLAB_BENCH_WARMUP", raising=False)
        monkeypatch.delenv("PERFLAB_BENCH_REPEATS", raising=False)
        fake = _BenchRunCmdFake(tmp_path)
        monkeypatch.setattr(benchmark_mod, "run_cmd", fake)

        benchmark_mod.run_benchmark(
            "python bench.py", cwd=tmp_path, warmup=7, repeats=9,
        )

        assert fake.env["PERFLAB_BENCH_WARMUP"] == "7"
        assert fake.env["PERFLAB_BENCH_REPEATS"] == "9"

    def test_session_env_var_outranks_task_value(self, tmp_path, monkeypatch):
        import perflab.runners.benchmark as benchmark_mod

        monkeypatch.setenv("PERFLAB_BENCH_WARMUP", "11")
        monkeypatch.delenv("PERFLAB_BENCH_REPEATS", raising=False)
        fake = _BenchRunCmdFake(tmp_path)
        monkeypatch.setattr(benchmark_mod, "run_cmd", fake)

        benchmark_mod.run_benchmark(
            "python bench.py", cwd=tmp_path, warmup=7, repeats=9,
        )

        assert fake.env["PERFLAB_BENCH_WARMUP"] == "11"
        assert fake.env["PERFLAB_BENCH_REPEATS"] == "9"

    def test_fast_mode_overrides_task_values(self, tmp_path, monkeypatch):
        import perflab.runners.benchmark as benchmark_mod

        fake = _BenchRunCmdFake(tmp_path)
        monkeypatch.setattr(benchmark_mod, "run_cmd", fake)

        benchmark_mod.run_benchmark(
            "python bench.py", cwd=tmp_path, fast_mode=True, warmup=7, repeats=9,
        )

        assert fake.env["PERFLAB_BENCH_WARMUP"] == "0"
        assert fake.env["PERFLAB_BENCH_REPEATS"] == "2"


class TestTopKCap:
    def _ctx(self, ws: Path, run_dir: Path, top_k: int):
        return SimpleNamespace(
            task=SimpleNamespace(
                benchmark=SimpleNamespace(
                    metric=SimpleNamespace(name="throughput.median", mode="maximize"),
                    cmd="python bench.py", warmup=1, repeats=5,
                ),
                build=None,
                program_type="python",
                constraints=SimpleNamespace(
                    regression_tolerance=0.02, rlimit_as_gb=None,
                    env_passthrough=[],
                ),
                contract=ContractSpec(),
                anti_gaming=SimpleNamespace(gaming_speedup_threshold=100.0),
                out_dir=ws / "out",
            ),
            ws=ws,
            rp=SimpleNamespace(run_dir=run_dir),
            iteration=1,
            progress=SimpleNamespace(on_message=lambda m: None),
            event_log=SimpleNamespace(__getattr__=None),
            history=[],
            baseline_val=10.0,
            best_value=10.0,
            best_iter=0,
            accepted_patches=[],
            accepted_count=0,
            sec_metric=None,
            config=SimpleNamespace(isolation=None, top_k=top_k),
        )

    def _run(self, tmp_path, monkeypatch, top_k: int) -> int:
        from perflab.optimizers.phases import evaluate as evaluate_mod

        ws = tmp_path / f"ws{top_k}"
        ws.mkdir()
        run_dir = tmp_path / f"run{top_k}"
        run_dir.mkdir()

        class _Log:
            def __getattr__(self, name):
                return lambda *a, **k: None

        ctx = self._ctx(ws, run_dir, top_k)
        ctx.event_log = _Log()

        rebench_calls = []

        def fake_benchmark(cmd, cwd, **kwargs):
            rebench_calls.append(1)
            # Full re-bench never confirms the fast-screen improvement
            return (
                CmdResult(cmd=[], returncode=0, stdout="", stderr="", duration_s=0.01),
                {"ok": True, "throughput": {"median": 5.0}},
            )

        monkeypatch.setattr(evaluate_mod, "run_benchmark", fake_benchmark)

        candidates = [
            evaluate_mod.BeamCandidate(
                iteration=1, index=i, blocks=[], description=f"candidate {i + 1}",
                value=v,
            )
            for i, v in enumerate([20.0, 15.0, 12.0])
        ]
        accepted, _, _ = evaluate_mod.accept_best(ctx, candidates, use_fast=True)
        assert accepted is False
        return len(rebench_calls)

    def test_top_k_limits_full_rebench_candidates(self, tmp_path, monkeypatch):
        assert self._run(tmp_path, monkeypatch, top_k=1) == 1
        assert self._run(tmp_path, monkeypatch, top_k=3) == 3


class TestRebenchBudget:
    """FIX 8: top_k is a per-iteration full-re-bench budget, not a pre-loop
    candidate slice. Non-improving candidates are skipped for free; only
    candidates that enter the full re-bench spend budget; exhaustion logs how
    many improving candidates went unexamined and breaks.
    """

    def _ctx(self, ws: Path, run_dir: Path, top_k: int, messages: list):
        return SimpleNamespace(
            task=SimpleNamespace(
                benchmark=SimpleNamespace(
                    metric=SimpleNamespace(name="throughput.median", mode="maximize"),
                    cmd="python bench.py", warmup=1, repeats=5,
                ),
                build=None,
                program_type="python",
                constraints=SimpleNamespace(
                    regression_tolerance=0.02, rlimit_as_gb=None, env_passthrough=[],
                ),
                contract=ContractSpec(),
                anti_gaming=SimpleNamespace(gaming_speedup_threshold=100.0),
                out_dir=ws / "out",
            ),
            ws=ws,
            rp=SimpleNamespace(run_dir=run_dir),
            iteration=1,
            progress=SimpleNamespace(on_message=messages.append),
            event_log=self._Log(),
            history=[],
            baseline_val=10.0,
            best_value=10.0,
            best_iter=0,
            accepted_patches=[],
            accepted_count=0,
            sec_metric=None,
            config=SimpleNamespace(isolation=None, top_k=top_k),
        )

    class _Log:
        def __getattr__(self, name):
            return lambda *a, **k: None

    def _run(self, tmp_path, monkeypatch, top_k, values, rebench_value, tag):
        from unittest.mock import patch

        from perflab.optimizers.phases import evaluate as evaluate_mod

        ws = tmp_path / f"ws_{tag}"
        ws.mkdir()
        run_dir = tmp_path / f"run_{tag}"
        run_dir.mkdir()
        messages: list[str] = []
        ctx = self._ctx(ws, run_dir, top_k, messages)

        rebench_calls: list[int] = []

        def fake_benchmark(cmd, cwd, **kwargs):
            rebench_calls.append(1)
            return (
                CmdResult(cmd=[], returncode=0, stdout="", stderr="", duration_s=0.01),
                {"ok": True, "throughput": {"median": rebench_value}},
            )

        monkeypatch.setattr(evaluate_mod, "run_benchmark", fake_benchmark)

        candidates = [
            evaluate_mod.BeamCandidate(
                iteration=1, index=i, blocks=[], description=f"candidate {i + 1}", value=v,
            )
            for i, v in enumerate(values)
        ]
        with patch.object(evaluate_mod, "snapshot_workspace", lambda *a, **k: None):
            accepted, _, accepted_value = evaluate_mod.accept_best(ctx, candidates, use_fast=True)
        return candidates, accepted, accepted_value, len(rebench_calls), messages

    def test_improving_candidate_accepted_when_earlier_did_not_spend_budget(
        self, tmp_path, monkeypatch,
    ):
        # Input order [9 (non-improving), 8 (non-improving), 30 (improving)]:
        # the 3rd candidate improves and is accepted; the earlier two are
        # skipped for free (never re-benched), so the top_k=2 budget is intact.
        candidates, accepted, accepted_value, rebenches, messages = self._run(
            tmp_path, monkeypatch, top_k=2,
            values=[9.0, 8.0, 30.0], rebench_value=40.0, tag="accept",
        )
        assert accepted is True
        assert accepted_value == 40.0
        assert candidates[2].accepted is True  # the 3rd (input order) candidate
        assert candidates[0].accepted is False and candidates[1].accepted is False
        assert rebenches == 1  # only the improving candidate spent budget
        assert not any("unexamined" in m for m in messages)

    def test_nonimproving_candidates_do_not_consume_budget(self, tmp_path, monkeypatch):
        # top_k=2, but only the top candidate improves and its re-bench fails.
        # The two non-improving candidates are skipped without a re-bench, so
        # exactly one re-bench runs and the budget is never exhausted.
        _, accepted, _, rebenches, messages = self._run(
            tmp_path, monkeypatch, top_k=2,
            values=[30.0, 9.0, 8.0], rebench_value=5.0, tag="freeskip",
        )
        assert accepted is False
        assert rebenches == 1
        assert not any("unexamined" in m for m in messages)

    def test_budget_exhausted_logs_and_breaks(self, tmp_path, monkeypatch):
        # Three improving candidates, top_k=2, every re-bench fails to confirm:
        # the first two exhaust the budget, the third is left unexamined with a
        # progress message, and the loop breaks (only two re-benches run).
        _, accepted, _, rebenches, messages = self._run(
            tmp_path, monkeypatch, top_k=2,
            values=[30.0, 25.0, 20.0], rebench_value=5.0, tag="exhaust",
        )
        assert accepted is False
        assert rebenches == 2  # budget capped the re-benches
        exhausted = [m for m in messages if "unexamined" in m]
        assert exhausted and "1 improving candidate" in exhausted[0]


class TestAccuracyToleranceEnv:
    def test_forwarded_to_correctness_subprocess(self, tmp_path, monkeypatch):
        import perflab.runners.correctness as correctness_mod

        captured = {}

        def fake_run_cmd(cmd, cwd=None, env=None, **kwargs):
            captured["env"] = dict(env or {})
            return CmdResult(cmd=list(cmd), returncode=0, stdout="", stderr="", duration_s=0.01)

        monkeypatch.setattr(correctness_mod, "run_cmd", fake_run_cmd)

        correctness_mod.run_correctness(
            "python tests.py", cwd=tmp_path, accuracy_tolerance="1e-3",
        )
        assert captured["env"]["PERFLAB_ACCURACY_TOLERANCE"] == "1e-3"

    def test_absent_by_default(self, tmp_path, monkeypatch):
        import perflab.runners.correctness as correctness_mod

        captured = {}

        def fake_run_cmd(cmd, cwd=None, env=None, **kwargs):
            captured["env"] = env
            return CmdResult(cmd=list(cmd), returncode=0, stdout="", stderr="", duration_s=0.01)

        monkeypatch.setattr(correctness_mod, "run_cmd", fake_run_cmd)

        correctness_mod.run_correctness("python tests.py", cwd=tmp_path)
        assert captured["env"] is None

    def test_env_accuracy_tolerance_helper(self, monkeypatch):
        from perflab.harness.tolerance import env_accuracy_tolerance

        monkeypatch.delenv("PERFLAB_ACCURACY_TOLERANCE", raising=False)
        assert env_accuracy_tolerance(1e-5) == 1e-5

        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "1e-3")
        assert env_accuracy_tolerance(1e-5) == 1e-3

        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "exact")
        assert env_accuracy_tolerance(1e-5) == 0.0

        monkeypatch.setenv("PERFLAB_ACCURACY_TOLERANCE", "garbage")
        assert env_accuracy_tolerance(1e-5) == 1e-5


class TestBuildTimeout:
    _TAIL = (
        "correctness:\n  cmd: \"python tests.py\"\n"
        "benchmark:\n  cmd: \"python bench.py --json out/bench.json\"\n"
        "  metric:\n    name: throughput.median\n    mode: maximize\n"
    )

    def _load(self, tmp_path: Path, build_block: str) -> TaskSpec:
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        (ws / "out").mkdir(exist_ok=True)
        task_file = ws / "task.yaml"
        task_file.write_text(
            "name: build-timeout-task\nprogram_type: cpp\n" + build_block + self._TAIL,
            encoding="utf-8",
        )
        return TaskSpec.load(task_file)

    def test_timeout_parsed_from_yaml(self, tmp_path):
        task = self._load(tmp_path, "build:\n  cmd: \"make\"\n  timeout_s: 900\n")
        assert task.build is not None and task.build.timeout_s == 900

    def test_timeout_defaults_none_when_unset(self, tmp_path):
        task = self._load(tmp_path, "build:\n  cmd: \"make\"\n")
        assert task.build is not None and task.build.timeout_s is None

    def test_nonpositive_timeout_rejected(self, tmp_path):
        import pytest
        with pytest.raises(ValueError, match="build.timeout_s"):
            self._load(tmp_path, "build:\n  cmd: \"make\"\n  timeout_s: 0\n")

    def _run_prescreen_build(self, tmp_path, monkeypatch, build_block: str) -> int:
        import perflab.tools.shell as shell_mod
        from perflab.optimizers.patch import SearchReplaceBlock
        from perflab.optimizers.phases import prescreen as prescreen_mod

        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        (ws / "out").mkdir(exist_ok=True)
        (ws / "algo.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        task_file = ws / "task.yaml"
        task_file.write_text(
            "name: build-timeout-task\nprogram_type: cpp\n" + build_block + self._TAIL
            + "edit_policy:\n  allowed_paths:\n    - \"*.py\"\n",
            encoding="utf-8",
        )
        task = TaskSpec.load(task_file)

        captured: dict = {}

        def fake_run_cmd(cmd, cwd=None, timeout_s=None, **kwargs):
            captured["timeout_s"] = timeout_s
            return CmdResult(cmd=list(cmd), returncode=0, stdout="", stderr="", duration_s=0.01)

        monkeypatch.setattr(shell_mod, "run_cmd", fake_run_cmd)
        monkeypatch.setattr(
            prescreen_mod, "run_correctness",
            lambda *a, **k: CmdResult(cmd=[], returncode=0, stdout="", stderr="", duration_s=0.01),
        )

        result = prescreen_mod._prescreen_candidate(
            0, [SearchReplaceBlock("algo.py", "return 1", "return 2")], "", task, ws,
        )
        assert result["passed"] is True
        return captured["timeout_s"]

    def test_build_site_forwards_task_timeout(self, tmp_path, monkeypatch):
        assert self._run_prescreen_build(
            tmp_path, monkeypatch, "build:\n  cmd: \"make\"\n  timeout_s: 900\n",
        ) == 900

    def test_build_site_defaults_to_600_when_unset(self, tmp_path, monkeypatch):
        from perflab.task_spec import DEFAULT_BUILD_TIMEOUT_S
        assert self._run_prescreen_build(
            tmp_path, monkeypatch, "build:\n  cmd: \"make\"\n",
        ) == DEFAULT_BUILD_TIMEOUT_S == 600


class TestProfilerAllowSudoConfig:
    def test_yaml_overlay(self):
        from perflab.config import PerfLabConfig, _overlay_yaml

        cfg = PerfLabConfig()
        assert cfg.profiler.allow_sudo is False
        _overlay_yaml(cfg, {"profiler": {"allow_sudo": True}})
        assert cfg.profiler.allow_sudo is True

    def test_env_overlay(self, monkeypatch):
        from perflab.config import PerfLabConfig, _overlay_env

        monkeypatch.setenv("PERFLAB_PROFILER_ALLOW_SUDO", "1")
        cfg = PerfLabConfig()
        _overlay_env(cfg)
        assert cfg.profiler.allow_sudo is True
