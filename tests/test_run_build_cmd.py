"""Regression coverage for run_build_cmd, the single funnel every build call
site in the codebase now goes through (perflab/ci.py, optimizers/phases/
{autotune,evaluate,pipeline}.py).

Found via five separate real-money debugging sessions on real H100/RTX
hardware, one call site at a time: every one of these places used to call
run_cmd directly and never resolved a memory limit at all, silently falling
back to run_cmd's bare 4GB CPU default regardless of program_type. nvcc
-arch=native (unlike a hardcoded -arch=sm_90) queries the GPU driver at
compile time to auto-detect compute capability, creating a CUDA context
that exceeds 4GB -- an opaque build failure (exit 1, no useful stderr).
run_build_cmd is the fix applied once, at the source, instead of five times
at each call site.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from perflab.runners.benchmark import run_build_cmd
from perflab.task_spec import TaskSpec
from perflab.tools.shell import DEFAULT_GPU_RLIMIT_AS_BYTES, DEFAULT_RLIMIT_AS_BYTES, CmdResult


def _make_task(tmp_path: Path, *, program_type: str, rlimit_as_gb_yaml: str = "null") -> TaskSpec:
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    task_file = ws / "task.yaml"
    task_file.write_text(textwrap.dedent(f"""\
        name: test-task
        program_type: {program_type}
        build: {{cmd: "true", expected_exit: 0}}
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
          rlimit_as_gb: {rlimit_as_gb_yaml}
        contract:
          fixed_params: {{}}
          min_repeats: 1
          required_bench_fields:
            - ok
    """), encoding="utf-8")
    return TaskSpec.load(task_file)


class TestRunBuildCmdRlimit:
    def test_gpu_program_type_gets_gpu_rlimit_by_default(self, tmp_path, monkeypatch):
        task = _make_task(tmp_path, program_type="cuda")
        captured = {}

        def fake_run_cmd(argv, **kwargs):
            captured["rlimit_as_bytes"] = kwargs.get("rlimit_as_bytes")
            return CmdResult(cmd=argv, returncode=0, stdout="", stderr="", duration_s=0.0)

        monkeypatch.setattr("perflab.runners.benchmark.run_cmd", fake_run_cmd)
        run_build_cmd(task, task.workspace)

        assert captured["rlimit_as_bytes"] == DEFAULT_GPU_RLIMIT_AS_BYTES

    def test_cpu_program_type_gets_cpu_rlimit_by_default(self, tmp_path, monkeypatch):
        task = _make_task(tmp_path, program_type="cpp")
        captured = {}

        def fake_run_cmd(argv, **kwargs):
            captured["rlimit_as_bytes"] = kwargs.get("rlimit_as_bytes")
            return CmdResult(cmd=argv, returncode=0, stdout="", stderr="", duration_s=0.0)

        monkeypatch.setattr("perflab.runners.benchmark.run_cmd", fake_run_cmd)
        run_build_cmd(task, task.workspace)

        assert captured["rlimit_as_bytes"] == DEFAULT_RLIMIT_AS_BYTES

    def test_explicit_task_yaml_override_wins_over_gpu_default(self, tmp_path, monkeypatch):
        task = _make_task(tmp_path, program_type="cuda", rlimit_as_gb_yaml="8")
        captured = {}

        def fake_run_cmd(argv, **kwargs):
            captured["rlimit_as_bytes"] = kwargs.get("rlimit_as_bytes")
            return CmdResult(cmd=argv, returncode=0, stdout="", stderr="", duration_s=0.0)

        monkeypatch.setattr("perflab.runners.benchmark.run_cmd", fake_run_cmd)
        run_build_cmd(task, task.workspace)

        assert captured["rlimit_as_bytes"] == 8 * 1024**3

    def test_explicit_zero_disables_rlimit_entirely(self, tmp_path, monkeypatch):
        task = _make_task(tmp_path, program_type="cuda", rlimit_as_gb_yaml="0")
        captured = {}

        def fake_run_cmd(argv, **kwargs):
            captured["rlimit_as_bytes"] = kwargs.get("rlimit_as_bytes")
            return CmdResult(cmd=argv, returncode=0, stdout="", stderr="", duration_s=0.0)

        monkeypatch.setattr("perflab.runners.benchmark.run_cmd", fake_run_cmd)
        run_build_cmd(task, task.workspace)

        assert captured["rlimit_as_bytes"] is None
