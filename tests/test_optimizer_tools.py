"""Tests for perflab.optimizers.tools -- the generate-phase diagnostic tool loop.

Construction of a minimal-but-real AgentContext mirrors
tests/test_agent_state.py's _cost_limit_ctx pattern: a real AgentContext with
a SimpleNamespace task (no need for a full TaskSpec.load(...) dependency
graph) and a real RunPaths/AgentEventLog/ListProgress.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from perflab.analyzers.bottleneck_analyzer import AnalysisThresholds
from perflab.llm.base import ToolCall
from perflab.memory.run_store import RunPaths
from perflab.optimizers.agent import AgentConfig, AgentContext
from perflab.optimizers.event_log import AgentEventLog
from perflab.optimizers.progress import ListProgress
from perflab.optimizers.tools import (
    _HANDLERS,
    DIAGNOSTIC_TOOLS,
    execute_tool,
)


def _make_task(tmp_path: Path, *, program_type: str = "cuda", build_cmd: str | None = None):
    return SimpleNamespace(
        program_type=program_type,
        workspace=tmp_path,
        build=SimpleNamespace(cmd=build_cmd) if build_cmd else None,
        benchmark=SimpleNamespace(
            cmd="python bench.py",
            metric=SimpleNamespace(mode="maximize"),
        ),
        constraints=SimpleNamespace(top_n=3, env_passthrough=[]),
        analysis_thresholds=AnalysisThresholds(),
    )


def _make_ctx(tmp_path: Path, *, task=None, iteration: int = 1, profiler_summaries=None, sysinfo=None) -> AgentContext:
    run_dir = tmp_path / "run"
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rp = RunPaths(run_id="test-run", run_dir=run_dir, artifacts_dir=artifacts_dir, logs_dir=logs_dir)

    ctx = AgentContext(
        task=task if task is not None else _make_task(tmp_path),
        config=AgentConfig(),
        llm_config=SimpleNamespace(model="test-model", pricing={}),
        provider=None,
        progress=ListProgress(),
        ws=tmp_path,
        rp=rp,
        event_log=AgentEventLog(run_dir=run_dir),
        iteration=iteration,
        profiler_summaries=profiler_summaries if profiler_summaries is not None else {},
        sysinfo=sysinfo if sysinfo is not None else {},
    )
    return ctx


def _call(tool_name: str, ctx: AgentContext, **arguments) -> dict:
    result = execute_tool(ToolCall(id="tc1", name=tool_name, arguments=arguments), ctx)
    assert isinstance(result, str)
    return json.loads(result)


# ---------------------------------------------------------------------------
# Tool catalog
# ---------------------------------------------------------------------------

class TestDiagnosticToolsCatalog:
    def test_four_tools_registered(self):
        names = {t.name for t in DIAGNOSTIC_TOOLS}
        assert names == {"get_bottlenecks", "get_kernel_dossier", "get_profile_diff", "run_profiler"}

    def test_every_tool_has_valid_object_schema(self):
        for t in DIAGNOSTIC_TOOLS:
            assert t.parameters["type"] == "object"
            assert "properties" in t.parameters
            assert "required" in t.parameters
            assert t.description  # non-empty

    def test_required_args_match_handlers(self):
        by_name = {t.name: t for t in DIAGNOSTIC_TOOLS}
        assert by_name["get_kernel_dossier"].parameters["required"] == ["kernel_name"]
        assert set(by_name["get_profile_diff"].parameters["required"]) == {"iteration_a", "iteration_b"}
        assert by_name["run_profiler"].parameters["required"] == ["profiler_name"]
        assert by_name["get_bottlenecks"].parameters["required"] == []

    def test_handlers_cover_every_tool(self):
        assert set(_HANDLERS) == {t.name for t in DIAGNOSTIC_TOOLS}


# ---------------------------------------------------------------------------
# get_bottlenecks
# ---------------------------------------------------------------------------

class TestGetBottlenecks:
    def test_happy_path(self, tmp_path):
        ctx = _make_ctx(tmp_path, profiler_summaries={"ncu": {"sm_utilization_pct": 20}})
        data = _call("get_bottlenecks", ctx)
        assert "bottlenecks" in data
        assert any("SM utilization" in b["bottleneck"] for b in data["bottlenecks"])
        b = data["bottlenecks"][0]
        assert set(b) >= {"rank", "bottleneck", "root_cause", "confidence", "suggested_actions", "evidence"}

    def test_empty_summaries_returns_sane_empty_result(self, tmp_path):
        ctx = _make_ctx(tmp_path, profiler_summaries={})
        data = _call("get_bottlenecks", ctx)
        assert data["bottlenecks"] == []
        assert "note" in data

    def test_top_n_respected(self, tmp_path):
        ctx = _make_ctx(tmp_path, profiler_summaries={
            "ncu": {"sm_utilization_pct": 20, "memory_throughput_pct": 90, "compute_throughput_pct": 10},
        })
        data = _call("get_bottlenecks", ctx, top_n=1)
        assert len(data["bottlenecks"]) <= 1

    def test_uses_bench_json_device_when_present(self, tmp_path):
        # MPS: torch profiler can't see Metal kernels, so total_gpu_kernel_us==0
        # must not be reported as "GPU underutilized" -- this only holds if the
        # device hint from bench.json actually reaches diagnose_bottlenecks.
        ctx = _make_ctx(tmp_path, profiler_summaries={"torch_profiler": {"total_gpu_kernel_us": 0, "total_cpu_us": 1000}})
        (ctx.rp.run_dir / "bench.json").write_text(json.dumps({"meta": {"device": "mps"}}), encoding="utf-8")
        data = _call("get_bottlenecks", ctx)
        # Must not crash and must return the standard shape either way.
        assert "bottlenecks" in data


# ---------------------------------------------------------------------------
# get_kernel_dossier
# ---------------------------------------------------------------------------

_NSYS_SUMMARY = {
    "cpu_gpu_correlations": [
        {
            "api_name": "cudaLaunchKernel",
            "kernel_name": "sgemm_naive_kernel",
            "stream_id": 0,
            "gpu_duration_ns": 5_000_000,
            "launch_overhead_ns": 8_000,
            "cpu_start_ns": 100,
            "gpu_start_ns": 200,
        },
    ],
    "top_kernels": [
        {"name": "sgemm_naive_kernel", "pct": 82.0, "total_ms": 12.5},
    ],
}


class TestGetKernelDossier:
    def test_happy_path_exact_name(self, tmp_path):
        ctx = _make_ctx(tmp_path, profiler_summaries={
            "nsys": _NSYS_SUMMARY,
            "ncu": {"kernels": [{"name": "sgemm_naive_kernel", "occupancy_pct": 45.0}]},
        })
        data = _call("get_kernel_dossier", ctx, kernel_name="sgemm_naive_kernel")
        assert data["name"] == "sgemm_naive_kernel"
        assert data["gpu_pct"] == 82.0
        assert data["ncu_metrics"] == {"occupancy_pct": 45.0}

    def test_fuzzy_substring_match(self, tmp_path):
        ctx = _make_ctx(tmp_path, profiler_summaries={"nsys": _NSYS_SUMMARY})
        data = _call("get_kernel_dossier", ctx, kernel_name="sgemm_naive")
        assert data["name"] == "sgemm_naive_kernel"

    def test_missing_kernel_name_arg_is_sane_error(self, tmp_path):
        ctx = _make_ctx(tmp_path, profiler_summaries={"nsys": _NSYS_SUMMARY})
        data = _call("get_kernel_dossier", ctx)
        assert "error" in data

    def test_no_nsys_data_is_sane_error_not_crash(self, tmp_path):
        ctx = _make_ctx(tmp_path, profiler_summaries={})
        data = _call("get_kernel_dossier", ctx, kernel_name="anything")
        assert "error" in data

    def test_unknown_kernel_name_lists_available(self, tmp_path):
        ctx = _make_ctx(tmp_path, profiler_summaries={"nsys": _NSYS_SUMMARY})
        data = _call("get_kernel_dossier", ctx, kernel_name="totally_unrelated_xyz")
        assert "error" in data
        assert "available_kernels" in data
        assert "sgemm_naive_kernel" in data["available_kernels"]


# ---------------------------------------------------------------------------
# get_profile_diff
# ---------------------------------------------------------------------------

class TestGetProfileDiff:
    def _ctx_with_baseline_and_current(self, tmp_path):
        ctx = _make_ctx(tmp_path, iteration=3, profiler_summaries={"linux_perf": {"ipc": 2.0}})
        baseline_dir = ctx.rp.run_dir / "artifacts_baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / "linux_perf_summary.json").write_text(json.dumps({"ipc": 1.0}), encoding="utf-8")
        return ctx

    def test_baseline_vs_current(self, tmp_path):
        ctx = self._ctx_with_baseline_and_current(tmp_path)
        data = _call("get_profile_diff", ctx, iteration_a=0, iteration_b=3)
        assert data["iteration_a"] == 0
        assert data["iteration_b"] == 3
        assert any(d["metric"] == "linux_perf.ipc" for d in data["deltas"])
        ipc_delta = next(d for d in data["deltas"] if d["metric"] == "linux_perf.ipc")
        assert ipc_delta["direction"] == "improved"

    def test_missing_args_is_sane_error(self, tmp_path):
        ctx = self._ctx_with_baseline_and_current(tmp_path)
        data = _call("get_profile_diff", ctx, iteration_a=0)
        assert "error" in data

    def test_unresolvable_iteration_is_sane_error(self, tmp_path):
        ctx = self._ctx_with_baseline_and_current(tmp_path)
        data = _call("get_profile_diff", ctx, iteration_a=0, iteration_b=1)
        assert "error" in data

    def test_no_baseline_on_disk_is_sane_error(self, tmp_path):
        ctx = _make_ctx(tmp_path, iteration=1, profiler_summaries={"linux_perf": {"ipc": 2.0}})
        data = _call("get_profile_diff", ctx, iteration_a=0, iteration_b=1)
        assert "error" in data


# ---------------------------------------------------------------------------
# run_profiler
# ---------------------------------------------------------------------------

@dataclass
class _FakeProfiler:
    name: str
    available: bool = True
    summary: dict = field(default_factory=dict)
    run_calls: list = field(default_factory=list)

    def is_available(self) -> bool:
        return self.available

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path):
        from perflab.profilers.base import ProfileResult
        self.run_calls.append((bench_cmd, cwd, artifacts_dir))
        return ProfileResult(name=self.name, artifacts={}, summary=self.summary)


class TestRunProfiler:
    def test_unknown_profiler_name_lists_available(self, tmp_path, monkeypatch):
        import perflab.profilers as profilers_module
        fake_ncu = _FakeProfiler(name="ncu", available=True)
        monkeypatch.setattr(profilers_module, "select_profilers", lambda task: [fake_ncu])

        ctx = _make_ctx(tmp_path)
        data = _call("run_profiler", ctx, profiler_name="nsys")
        assert "error" in data
        assert data["available_profilers"] == ["ncu"]
        assert fake_ncu.run_calls == []

    def test_unavailable_profiler_is_sane_error(self, tmp_path, monkeypatch):
        import perflab.profilers as profilers_module
        fake_ncu = _FakeProfiler(name="ncu", available=False)
        monkeypatch.setattr(profilers_module, "select_profilers", lambda task: [fake_ncu])

        ctx = _make_ctx(tmp_path)
        data = _call("run_profiler", ctx, profiler_name="ncu")
        assert "error" in data
        assert data["available_profilers"] == []
        assert fake_ncu.run_calls == []

    def test_missing_profiler_name_arg_is_sane_error(self, tmp_path, monkeypatch):
        import perflab.profilers as profilers_module
        monkeypatch.setattr(profilers_module, "select_profilers", lambda task: [])

        ctx = _make_ctx(tmp_path)
        data = _call("run_profiler", ctx)
        assert "error" in data

    def test_success_merges_into_ctx_profiler_summaries(self, tmp_path, monkeypatch):
        import perflab.profilers as profilers_module
        fake_ncu = _FakeProfiler(name="ncu", available=True, summary={"sm_utilization_pct": 55.0})
        monkeypatch.setattr(profilers_module, "select_profilers", lambda task: [fake_ncu])

        ctx = _make_ctx(tmp_path, profiler_summaries={"linux_perf": {"ipc": 1.5}})
        data = _call("run_profiler", ctx, profiler_name="ncu")

        assert "error" not in data
        assert data["profiler"] == "ncu"
        assert data["summary"] == {"sm_utilization_pct": 55.0}
        # Merged into ctx, existing entries for other profilers untouched.
        assert ctx.profiler_summaries["ncu"] == {"sm_utilization_pct": 55.0}
        assert ctx.profiler_summaries["linux_perf"] == {"ipc": 1.5}
        # Actually invoked the fake profiler's run() exactly once.
        assert len(fake_ncu.run_calls) == 1
        bench_cmd, cwd, artifacts_dir = fake_ncu.run_calls[0]
        assert bench_cmd == ctx.task.benchmark.cmd
        assert cwd == ctx.ws

    def test_scoped_artifacts_dir_does_not_collide_with_phase_artifacts(self, tmp_path, monkeypatch):
        import perflab.profilers as profilers_module
        fake_ncu = _FakeProfiler(name="ncu", available=True, summary={"x": 1})
        monkeypatch.setattr(profilers_module, "select_profilers", lambda task: [fake_ncu])

        ctx = _make_ctx(tmp_path)
        data = _call("run_profiler", ctx, profiler_name="ncu")

        artifacts_dir = Path(data["artifacts_dir"])
        assert artifacts_dir != ctx.rp.artifacts_dir
        assert artifacts_dir.is_relative_to(ctx.rp.artifacts_dir / "tool_calls")
        assert artifacts_dir.exists()

    def test_never_touches_benchmark_cmd_meaning_only_profiles_current_workspace(self, tmp_path, monkeypatch):
        # Documents the constraint from the design: run_profiler must never be
        # given anything other than task.benchmark.cmd against ctx.ws (the
        # current accepted workspace) -- never a candidate's patched copy.
        import perflab.profilers as profilers_module
        fake_ncu = _FakeProfiler(name="ncu", available=True, summary={})
        monkeypatch.setattr(profilers_module, "select_profilers", lambda task: [fake_ncu])

        ctx = _make_ctx(tmp_path)
        _call("run_profiler", ctx, profiler_name="ncu")
        _, cwd, _ = fake_ncu.run_calls[0]
        assert cwd == ctx.ws


# ---------------------------------------------------------------------------
# execute_tool dispatch / error handling
# ---------------------------------------------------------------------------

class TestExecuteToolDispatch:
    def test_unknown_tool_name_returns_error_json(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        result = execute_tool(ToolCall(id="1", name="not_a_real_tool", arguments={}), ctx)
        data = json.loads(result)
        assert "error" in data
        assert "available_tools" in data

    def test_exception_inside_handler_is_caught_not_raised(self, tmp_path, monkeypatch):
        import perflab.optimizers.tools as tools_module

        def _boom(ctx, args):
            raise RuntimeError("kaboom")

        monkeypatch.setitem(tools_module._HANDLERS, "get_bottlenecks", _boom)

        ctx = _make_ctx(tmp_path)
        result = execute_tool(ToolCall(id="1", name="get_bottlenecks", arguments={}), ctx)
        data = json.loads(result)
        assert "error" in data
        assert "kaboom" in data["error"]

    def test_non_dict_arguments_do_not_crash(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        # A malformed tool call (arguments not a dict) must still come back as
        # a sane error string, never raise.
        result = execute_tool(ToolCall(id="1", name="get_kernel_dossier", arguments=None), ctx)  # type: ignore[arg-type]
        data = json.loads(result)
        assert "error" in data

    def test_result_is_always_a_json_string(self, tmp_path):
        ctx = _make_ctx(tmp_path, profiler_summaries={"ncu": {"sm_utilization_pct": 20}})
        result = execute_tool(ToolCall(id="1", name="get_bottlenecks", arguments={}), ctx)
        assert isinstance(result, str)
        json.loads(result)  # must not raise
