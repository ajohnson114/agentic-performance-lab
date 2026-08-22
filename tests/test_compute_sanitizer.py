"""compute-sanitizer accept-path gate.

Two layers, tested separately:

* perflab.tools.compute_sanitizer -- detection, availability, and parsing of
  compute-sanitizer's text output, against a mocked run_cmd (no real GPU or
  CUDA toolkit needed).
* optimizers.phases.evaluate._cuda_sanitizer_gate -- the wiring into
  accept_best(): applicability/availability gating, event logging, and that
  a dirty result actually rejects the candidate while a clean one does not.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perflab.optimizers.phases import evaluate as evaluate_mod
from perflab.optimizers.phases.evaluate import BeamCandidate
from perflab.task_spec import ContractSpec, TaskSpec
from perflab.tools import compute_sanitizer as cs
from perflab.tools.shell import CmdResult

# --- uses_cuda_build ---------------------------------------------------------


class TestUsesCudaBuild:
    def test_none(self):
        assert cs.uses_cuda_build(None) is False

    def test_empty(self):
        assert cs.uses_cuda_build("") is False

    def test_gpp_build_is_not_cuda(self):
        assert cs.uses_cuda_build("g++ -O3 -march=native -o prog main.cpp") is False

    def test_plain_nvcc_build(self):
        assert cs.uses_cuda_build("nvcc -O2 -arch=native -o sgemm_bin sgemm.cu") is True

    def test_nvcc_with_path_prefix(self):
        assert cs.uses_cuda_build("/usr/local/cuda/bin/nvcc -O2 -o bin k.cu") is True

    def test_does_not_match_substring_of_another_word(self):
        # "nvcc" must be a whole word, not a substring of e.g. a directory name.
        assert cs.uses_cuda_build("g++ -Iallnvccflags -o prog main.cpp") is False

    def test_program_type_is_irrelevant(self):
        # reduction/cpp_cuda is program_type "cpp" but builds with nvcc --
        # this function must say True regardless of what program_type is.
        assert cs.uses_cuda_build("nvcc -O2 -arch=native -o reduce_bin reduce.cu") is True


# --- compute_sanitizer_available ---------------------------------------------


class TestAvailability:
    def test_missing_binary(self, monkeypatch):
        monkeypatch.setattr(cs.shutil, "which", lambda name: None)
        assert cs.compute_sanitizer_available() is False

    def test_present_binary(self, monkeypatch):
        monkeypatch.setattr(cs.shutil, "which", lambda name: "/usr/local/cuda/bin/compute-sanitizer")
        assert cs.compute_sanitizer_available() is True


# --- SanitizerToolResult / SanitizerReport -----------------------------------


class TestReportShape:
    def test_tool_result_clean_requires_ran_and_zero_errors(self):
        assert cs.SanitizerToolResult("memcheck", True, 0, 0, "", 0.1).clean is True
        assert cs.SanitizerToolResult("memcheck", True, 1, 1, "", 0.1).clean is False
        assert cs.SanitizerToolResult("memcheck", False, -1, 1, "", 0.1).clean is False

    def test_report_clean_iff_all_tools_clean(self):
        clean = cs.SanitizerToolResult("memcheck", True, 0, 0, "", 0.1)
        dirty = cs.SanitizerToolResult("racecheck", True, 3, 1, "", 0.1)
        assert cs.SanitizerReport(True, True, [clean]).clean is True
        assert cs.SanitizerReport(True, True, [clean, dirty]).clean is False

    def test_reason_empty_when_clean(self):
        clean = cs.SanitizerToolResult("memcheck", True, 0, 0, "", 0.1)
        assert cs.SanitizerReport(True, True, [clean]).reason == ""

    def test_reason_names_the_failing_tool_and_count(self):
        dirty = cs.SanitizerToolResult("racecheck", True, 3, 1, "", 0.1)
        reason = cs.SanitizerReport(True, True, [dirty]).reason
        assert "racecheck" in reason
        assert "3 error" in reason

    def test_reason_distinguishes_inconclusive_from_a_clean_count(self):
        # ran=False (no ERROR SUMMARY parsed) must not be describable as "0 errors".
        inconclusive = cs.SanitizerToolResult("memcheck", False, -1, 124, "", 60.0)
        reason = cs.SanitizerReport(True, True, [inconclusive]).reason
        assert "inconclusive" in reason
        assert "0 error" not in reason


# --- run_compute_sanitizer: parsing against a mocked run_cmd ----------------


def _cmd_result(stdout: str, returncode: int = 0) -> CmdResult:
    return CmdResult(cmd=[], returncode=returncode, stdout=stdout, stderr="", duration_s=1.0)


class TestRunComputeSanitizer:
    def test_clean_output_parses_zero_errors(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            cs, "run_cmd",
            lambda *a, **k: _cmd_result("========= ERROR SUMMARY: 0 errors\n"),
        )
        report = cs.run_compute_sanitizer("python tests.py", tmp_path, tools=["memcheck"])
        assert report.clean is True
        assert report.results[0].errors == 0
        assert report.results[0].ran is True

    def test_dirty_output_parses_error_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            cs, "run_cmd",
            lambda *a, **k: _cmd_result(
                "========= Invalid __global__ write of size 4 bytes\n"
                "========= ERROR SUMMARY: 2 errors\n"
            ),
        )
        report = cs.run_compute_sanitizer("python tests.py", tmp_path, tools=["memcheck"])
        assert report.clean is False
        assert report.results[0].errors == 2

    def test_missing_summary_line_is_not_clean(self, tmp_path, monkeypatch):
        # Simulates a crash or timeout before compute-sanitizer could report:
        # must fail closed, not be treated as "0 errors".
        monkeypatch.setattr(cs, "run_cmd", lambda *a, **k: _cmd_result("Segmentation fault\n", 139))
        report = cs.run_compute_sanitizer("python tests.py", tmp_path, tools=["memcheck"])
        assert report.clean is False
        assert report.results[0].ran is False
        assert report.results[0].errors == -1

    def test_runs_every_configured_tool(self, tmp_path, monkeypatch):
        calls = []

        def _fake_run_cmd(argv, **kwargs):
            calls.append(argv)
            return _cmd_result("========= ERROR SUMMARY: 0 errors\n")

        monkeypatch.setattr(cs, "run_cmd", _fake_run_cmd)
        report = cs.run_compute_sanitizer(
            "python tests.py", tmp_path, tools=["memcheck", "racecheck"],
        )
        assert [r.tool for r in report.results] == ["memcheck", "racecheck"]
        assert len(calls) == 2
        for argv, tool in zip(calls, ["memcheck", "racecheck"], strict=True):
            assert argv[0] == "compute-sanitizer"
            assert "--tool" in argv and tool in argv
            assert "--target-processes" in argv and "all" in argv
            # The wrapped correctness command itself must still be present.
            assert argv[-2:] == ["python", "tests.py"]

    def test_wraps_target_processes_all_so_child_processes_are_covered(self, tmp_path, monkeypatch):
        # tests.py subprocess.run()s the compiled binary rather than exec-ing
        # it directly -- compute-sanitizer must be told to follow children.
        captured = {}

        def _fake_run_cmd(argv, **kwargs):
            captured["argv"] = argv
            return _cmd_result("========= ERROR SUMMARY: 0 errors\n")

        monkeypatch.setattr(cs, "run_cmd", _fake_run_cmd)
        cs.run_compute_sanitizer("python tests.py", tmp_path, tools=["memcheck"])
        i = captured["argv"].index("--target-processes")
        assert captured["argv"][i + 1] == "all"


# --- The accept-path gate (optimizers.phases.evaluate) ----------------------


class _RecordingLog:
    def __init__(self):
        self.events: list[tuple] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.events.append((name, args, kwargs))
        return record


def _ctx(tmp_path: Path, *, build_cmd: str | None = "nvcc -O2 -o sgemm_bin sgemm.cu",
          sanitizer_enabled: bool = True, best_value: float = 100.0):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "sgemm.cu").write_text("// naive kernel\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    messages: list[str] = []
    event_log = _RecordingLog()
    ctx = SimpleNamespace(
        task=SimpleNamespace(
            benchmark=SimpleNamespace(
                metric=SimpleNamespace(name="tflops.median", mode="maximize"),
                cmd="python bench.py", warmup=1, repeats=5,
            ),
            build=SimpleNamespace(cmd=build_cmd, expected_exit=0, timeout_s=None) if build_cmd else None,
            correctness=SimpleNamespace(cmd="python tests.py", expected_exit=0),
            program_type="cuda",
            out_dir=ws / "out",
            contract=ContractSpec(),
            constraints=SimpleNamespace(
                regression_tolerance=0.02, noise_gate=True, decision_rule=None,
                cv_threshold=None, rlimit_as_gb=None, env_passthrough=[],
                compute_sanitizer=sanitizer_enabled,
                compute_sanitizer_tools=None, compute_sanitizer_timeout_s=60,
            ),
            anti_gaming=SimpleNamespace(gaming_speedup_threshold=1000.0),
        ),
        ws=ws,
        rp=SimpleNamespace(run_dir=run_dir),
        iteration=1,
        progress=SimpleNamespace(on_message=messages.append),
        event_log=event_log,
        history=[],
        baseline_val=best_value,
        best_value=best_value,
        best_iter=0,
        accepted_patches=[],
        accepted_count=0,
        sec_metric=None,
        config=SimpleNamespace(isolation=None, top_k=3),
        sanitizer_unavailable_warned=False,
    )
    return ctx, messages, event_log


def _accept(ctx, value: float = 110.0):
    cand = BeamCandidate(
        iteration=1, index=0, blocks=[], description="candidate 1",
        value=value, samples=[],
    )
    with patch.object(evaluate_mod, "snapshot_workspace", lambda *a, **k: None):
        return evaluate_mod.accept_best(ctx, [cand], use_fast=False)


class TestSanitizerGateWiring:
    def test_non_cuda_build_never_invokes_sanitizer(self, tmp_path, monkeypatch):
        ctx, _, log = _ctx(tmp_path, build_cmd="g++ -O3 -o prog main.cpp")
        monkeypatch.setattr(cs, "compute_sanitizer_available", lambda: True)

        def _boom(*a, **k):
            raise AssertionError("compute-sanitizer must not run for a non-CUDA build")

        monkeypatch.setattr(cs, "run_compute_sanitizer", _boom)

        accepted, _, _ = _accept(ctx)
        assert accepted is True
        assert not any(e[0] == "compute_sanitizer_check" for e in log.events)

    def test_no_build_step_never_invokes_sanitizer(self, tmp_path, monkeypatch):
        ctx, _, _ = _ctx(tmp_path, build_cmd=None)
        monkeypatch.setattr(cs, "compute_sanitizer_available", lambda: True)
        monkeypatch.setattr(
            cs, "run_compute_sanitizer",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")),
        )
        accepted, _, _ = _accept(ctx)
        assert accepted is True

    def test_disabled_via_constraints_skips_the_gate(self, tmp_path, monkeypatch):
        ctx, _, _ = _ctx(tmp_path, sanitizer_enabled=False)
        monkeypatch.setattr(cs, "compute_sanitizer_available", lambda: True)
        monkeypatch.setattr(
            cs, "run_compute_sanitizer",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run when disabled")),
        )
        accepted, _, _ = _accept(ctx)
        assert accepted is True

    def test_unavailable_binary_warns_once_and_still_accepts(self, tmp_path, monkeypatch):
        ctx, messages, log = _ctx(tmp_path)
        monkeypatch.setattr(cs, "compute_sanitizer_available", lambda: False)

        accepted, _, _ = _accept(ctx)

        assert accepted is True
        assert ctx.sanitizer_unavailable_warned is True
        assert any("compute-sanitizer not found" in m for m in messages)
        assert any(e[0] == "anti_gaming_warning" for e in log.events)

    def test_clean_result_accepts_and_logs_event(self, tmp_path, monkeypatch):
        ctx, messages, log = _ctx(tmp_path)
        monkeypatch.setattr(cs, "compute_sanitizer_available", lambda: True)
        monkeypatch.setattr(evaluate_mod, "_build_arm", lambda ctx, cwd: "")
        clean_report = cs.SanitizerReport(True, True, [
            cs.SanitizerToolResult("memcheck", True, 0, 0, "", 0.1),
            cs.SanitizerToolResult("racecheck", True, 0, 0, "", 0.1),
        ])
        monkeypatch.setattr(cs, "run_compute_sanitizer", lambda *a, **k: clean_report)

        accepted, _, _ = _accept(ctx)

        assert accepted is True
        checks = [e for e in log.events if e[0] == "compute_sanitizer_check"]
        assert len(checks) == 1
        assert checks[0][2]["clean"] is True
        assert any("compute-sanitizer clean" in m for m in messages)

    def test_dirty_result_rejects_the_candidate(self, tmp_path, monkeypatch):
        ctx, messages, log = _ctx(tmp_path)
        monkeypatch.setattr(cs, "compute_sanitizer_available", lambda: True)
        monkeypatch.setattr(evaluate_mod, "_build_arm", lambda ctx, cwd: "")
        dirty_report = cs.SanitizerReport(True, True, [
            cs.SanitizerToolResult("memcheck", True, 2, 1, "oob write", 0.1),
            cs.SanitizerToolResult("racecheck", True, 0, 0, "", 0.1),
        ])
        monkeypatch.setattr(cs, "run_compute_sanitizer", lambda *a, **k: dirty_report)

        accepted, rel_improvement, accepted_value = _accept(ctx)

        assert accepted is False
        assert rel_improvement is None
        assert accepted_value is None
        # best_value must be unchanged -- the candidate was never applied.
        assert ctx.best_value == 100.0
        checks = [e for e in log.events if e[0] == "compute_sanitizer_check"]
        assert checks[0][2]["clean"] is False
        assert any("memcheck: 2 error" in m for m in messages)

    def test_build_failure_under_sanitizer_rejects(self, tmp_path, monkeypatch):
        ctx, messages, _ = _ctx(tmp_path)
        monkeypatch.setattr(cs, "compute_sanitizer_available", lambda: True)
        monkeypatch.setattr(evaluate_mod, "_build_arm", lambda ctx, cwd: "build failed (rc=1)")
        monkeypatch.setattr(
            cs, "run_compute_sanitizer",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run after a failed build")),
        )

        accepted, _, _ = _accept(ctx)

        assert accepted is False

    @pytest.mark.parametrize("program_type", ["cpp"])
    def test_program_type_cpp_with_nvcc_build_still_checked(self, tmp_path, monkeypatch, program_type):
        # reduction/cpp_cuda is program_type "cpp" with an nvcc build step --
        # the gate must not key off program_type.
        ctx, _, log = _ctx(tmp_path, build_cmd="nvcc -O2 -o reduce_bin reduce.cu")
        ctx.task.program_type = program_type
        monkeypatch.setattr(cs, "compute_sanitizer_available", lambda: True)
        monkeypatch.setattr(evaluate_mod, "_build_arm", lambda ctx, cwd: "")
        clean_report = cs.SanitizerReport(True, True, [
            cs.SanitizerToolResult("memcheck", True, 0, 0, "", 0.1),
            cs.SanitizerToolResult("racecheck", True, 0, 0, "", 0.1),
        ])
        monkeypatch.setattr(cs, "run_compute_sanitizer", lambda *a, **k: clean_report)

        accepted, _, _ = _accept(ctx)

        assert accepted is True
        assert any(e[0] == "compute_sanitizer_check" for e in log.events)


# --- task.yaml loading --------------------------------------------------


def _write_task_yaml(tmp_path: Path, extra_constraints: str = "") -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "sgemm.cu").write_text("// naive\n", encoding="utf-8")
    task_file = tmp_path / "task.yaml"
    task_file.write_text(
        f"""\
name: test-cuda-task
workspace: ws
program_type: cuda
build:
  cmd: "nvcc -O2 -arch=native -o sgemm_bin sgemm.cu"
correctness:
  cmd: "python tests.py"
benchmark:
  cmd: "python bench.py --json out/bench.json"
  metric:
    name: tflops.median
    mode: maximize
edit_policy:
  allowed_paths:
    - "sgemm.cu"
constraints:
  rlimit_as_gb: 0
{extra_constraints}
contract:
  fixed_params: {{}}
  min_repeats: 1
  required_bench_fields:
    - ok
""",
        encoding="utf-8",
    )
    return task_file


class TestTaskSpecLoading:
    def test_defaults_when_unset(self, tmp_path):
        task = TaskSpec.load(_write_task_yaml(tmp_path))
        assert task.constraints.compute_sanitizer is True
        assert task.constraints.compute_sanitizer_tools == ["memcheck", "racecheck"]
        assert task.constraints.compute_sanitizer_timeout_s == 180

    def test_explicit_overrides(self, tmp_path):
        task_file = _write_task_yaml(
            tmp_path,
            extra_constraints=(
                "  compute_sanitizer: false\n"
                "  compute_sanitizer_tools: [memcheck]\n"
                "  compute_sanitizer_timeout_s: 30\n"
            ),
        )
        task = TaskSpec.load(task_file)
        assert task.constraints.compute_sanitizer is False
        assert task.constraints.compute_sanitizer_tools == ["memcheck"]
        assert task.constraints.compute_sanitizer_timeout_s == 30
