"""End-to-end tests for perflab/runners/pipeline.py with profiling disabled.

Each test builds a tiny synthetic python task in tmp_path: a task.yaml
(imitating tests/conftest.py's sample), a stdlib-only bench.py that writes a
valid bench.json, and a trivial correctness script. Profiling stays off
(do_profiles defaults to False), so nothing shells out to perf/nsys/etc.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from perflab.runners.benchmark import metric_value
from perflab.runners.pipeline import run_pipeline
from perflab.task_spec import TaskSpec

# bench.json includes a nonce so its content hash changes every run,
# satisfying the anti-tamper check in perflab.runners.benchmark.
BENCH_OK = textwrap.dedent("""\
    import argparse, json, os, time

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="out/bench.json")
    args = parser.parse_args()

    payload = {
        "ok": True,
        "throughput": {"median": 123.5},
        "meta": {"N": 64, "device": "cpu"},
        "nonce": os.urandom(8).hex(),
        "ts": time.time(),
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(payload, f)
""")

BENCH_NO_JSON = "raise SystemExit(0)\n"

BENCH_CRASH = "raise SystemExit(3)\n"

BENCH_INVALID_JSON = textwrap.dedent("""\
    import os
    os.makedirs("out", exist_ok=True)
    with open("out/bench.json", "w", encoding="utf-8") as f:
        f.write("this is { not valid json")
""")

BENCH_MISSING_OK_FIELD = textwrap.dedent("""\
    import json, os
    os.makedirs("out", exist_ok=True)
    payload = {"throughput": {"median": 1.0}, "nonce": os.urandom(8).hex()}
    with open("out/bench.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
""")


def make_task(
    tmp_path: Path,
    *,
    bench_body: str = BENCH_OK,
    correctness_body: str = "raise SystemExit(0)\n",
    correctness_expected_exit: int = 0,
    build_yaml: str = "null",
    fixed_params_yaml: str = "{}",
) -> TaskSpec:
    """Write a minimal python task workspace and load its TaskSpec."""
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    (ws / "task.yaml").write_text(textwrap.dedent(f"""\
        name: pipeline-test
        program_type: python
        build: {build_yaml}
        correctness:
          cmd: "python tests.py"
          expected_exit: {correctness_expected_exit}
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
          fixed_params: {fixed_params_yaml}
          min_repeats: 1
          required_bench_fields:
            - ok
            - throughput.median
    """), encoding="utf-8")
    (ws / "bench.py").write_text(bench_body, encoding="utf-8")
    (ws / "tests.py").write_text(correctness_body, encoding="utf-8")
    return TaskSpec.load(ws / "task.yaml")


@pytest.fixture
def run_dirs(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return run_dir, run_dir / "artifacts"


class TestPipelineSuccess:
    def test_returns_bench_with_extractable_metric(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        task = make_task(tmp_path)
        run_dir, artifacts_dir = run_dirs

        result = run_pipeline(task, run_dir, artifacts_dir, save_logs=True)

        assert result.bench["ok"] is True
        assert metric_value(result.bench, "throughput.median") == 123.5
        assert result.bench_wall_s is not None and result.bench_wall_s > 0
        # Profiling off: no profiler wall time, no diagnostics, no artifacts
        assert result.profile_wall_s is None
        assert result.diagnostics is None
        assert result.artifacts == {}

    def test_bench_json_copied_and_logs_written(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        task = make_task(tmp_path)
        run_dir, artifacts_dir = run_dirs

        run_pipeline(task, run_dir, artifacts_dir, save_logs=True)

        copied = json.loads((run_dir / "bench.json").read_text(encoding="utf-8"))
        assert copied["ok"] is True
        assert copied["throughput"]["median"] == 123.5
        for log_name in (
            "correctness.stdout.txt",
            "correctness.stderr.txt",
            "bench.stdout.txt",
            "bench.stderr.txt",
        ):
            assert (run_dir / "logs" / log_name).exists(), log_name
        assert artifacts_dir.exists()

    def test_correctness_honors_expected_exit(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        task = make_task(
            tmp_path,
            correctness_body="raise SystemExit(3)\n",
            correctness_expected_exit=3,
        )
        run_dir, artifacts_dir = run_dirs
        result = run_pipeline(task, run_dir, artifacts_dir)
        assert result.bench["ok"] is True

    def test_build_step_runs_before_bench(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        build_yaml = '{cmd: "python build.py", expected_exit: 0}'
        task = make_task(tmp_path, build_yaml=build_yaml)
        ws = task.workspace
        (ws / "build.py").write_text(
            "open('built.txt', 'w').write('yes')\n", encoding="utf-8"
        )
        run_dir, artifacts_dir = run_dirs

        result = run_pipeline(task, run_dir, artifacts_dir, save_logs=True)

        assert (ws / "built.txt").exists()
        assert result.bench["ok"] is True
        assert (run_dir / "logs" / "build.stdout.txt").exists()


class TestPipelineFailures:
    def test_build_failure_raises(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        build_yaml = '{cmd: "python build.py", expected_exit: 0}'
        task = make_task(tmp_path, build_yaml=build_yaml)
        (task.workspace / "build.py").write_text(
            "raise SystemExit(2)\n", encoding="utf-8"
        )
        run_dir, artifacts_dir = run_dirs
        with pytest.raises(RuntimeError, match="Build failed with code 2"):
            run_pipeline(task, run_dir, artifacts_dir)

    def test_correctness_failure_aborts_before_bench(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        task = make_task(tmp_path, correctness_body="raise SystemExit(1)\n")
        run_dir, artifacts_dir = run_dirs
        with pytest.raises(RuntimeError, match="Correctness failed with code 1"):
            run_pipeline(task, run_dir, artifacts_dir)
        # Benchmark never ran
        assert not (task.workspace / "out" / "bench.json").exists()

    def test_bench_never_writes_json_raises(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        task = make_task(tmp_path, bench_body=BENCH_NO_JSON)
        run_dir, artifacts_dir = run_dirs
        with pytest.raises(FileNotFoundError, match="did not create"):
            run_pipeline(task, run_dir, artifacts_dir)

    def test_bench_nonzero_exit_raises(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        """A crashing bench must fail on its exit code, even when a previous
        good run left a bench.json behind."""
        task = make_task(tmp_path)
        run_dir, artifacts_dir = run_dirs
        run_pipeline(task, run_dir, artifacts_dir)  # good run writes bench.json

        (task.workspace / "bench.py").write_text(BENCH_CRASH, encoding="utf-8")
        with pytest.raises(RuntimeError, match="exited with code 3"):
            run_pipeline(task, run_dir, artifacts_dir)

    def test_bench_stale_json_raises_anti_tamper(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        """A bench that exits 0 without rewriting bench.json (e.g. candidate
        code faking success against a pre-written result) must be rejected by
        the anti-tamper hash check."""
        task = make_task(tmp_path)
        run_dir, artifacts_dir = run_dirs
        run_pipeline(task, run_dir, artifacts_dir)  # good run writes bench.json

        (task.workspace / "bench.py").write_text(BENCH_NO_JSON, encoding="utf-8")
        with pytest.raises(RuntimeError, match="was not modified"):
            run_pipeline(task, run_dir, artifacts_dir)

    def test_bench_invalid_json_raises(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        task = make_task(tmp_path, bench_body=BENCH_INVALID_JSON)
        run_dir, artifacts_dir = run_dirs
        with pytest.raises(json.JSONDecodeError):
            run_pipeline(task, run_dir, artifacts_dir)

    def test_contract_violation_missing_required_field(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        task = make_task(tmp_path, bench_body=BENCH_MISSING_OK_FIELD)
        run_dir, artifacts_dir = run_dirs
        # Without contract validation the pipeline is permissive
        result = run_pipeline(task, run_dir, artifacts_dir)
        assert "ok" not in result.bench
        # With validation it must raise
        with pytest.raises(RuntimeError, match="Contract violation"):
            run_pipeline(task, run_dir, artifacts_dir, validate_contract_spec=True)

    def test_contract_violation_fixed_params_mismatch(
        self, tmp_path: Path, run_dirs: tuple[Path, Path]
    ) -> None:
        # bench.py reports meta.N=64 but the contract pins N=128
        task = make_task(tmp_path, fixed_params_yaml="{N: 128}")
        run_dir, artifacts_dir = run_dirs
        with pytest.raises(RuntimeError, match=r"meta\.N=64"):
            run_pipeline(task, run_dir, artifacts_dir, validate_contract_spec=True)
