"""Tests for MCP server tools: helpers, run management, analysis, and agent jobs."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

_has_fastmcp = True
try:
    import fastmcp  # noqa: F401
except ImportError:
    _has_fastmcp = False

needs_fastmcp = pytest.mark.skipif(not _has_fastmcp, reason="fastmcp not installed")


# ---------------------------------------------------------------------------
# Fake run construction
# ---------------------------------------------------------------------------

def _make_run(
    out_dir: Path,
    *,
    task: str = "demo-task",
    program_type: str = "python",
    report: dict | None = None,
    bench: dict | None = None,
    summaries: dict[str, dict] | None = None,
    system_info: dict | None = None,
    events: list[dict] | None = None,
    meta_extra: dict | None = None,
) -> str:
    """Create a real run directory via RunStore and return its run_id."""
    from perflab.memory.run_store import RunStore

    store = RunStore(out_dir)
    rp = store.new_run(task, program_type=program_type)
    if meta_extra:
        store.update_meta(rp.run_id, meta_extra)
    if report is not None:
        (rp.run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    if bench is not None:
        (rp.run_dir / "bench.json").write_text(json.dumps(bench), encoding="utf-8")
    for name, summary in (summaries or {}).items():
        (rp.artifacts_dir / f"{name}_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    if system_info is not None:
        (rp.run_dir / "system_info.json").write_text(json.dumps(system_info), encoding="utf-8")
    if events is not None:
        with (rp.run_dir / "agent_events.jsonl").open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
    return rp.run_id


# ---------------------------------------------------------------------------
# Helpers: _guard_output_size and _to_dicts
# ---------------------------------------------------------------------------

@needs_fastmcp
class TestGuardOutputSize:
    def test_small_dict_passes_through(self):
        from perflab.server.mcp_server import _guard_output_size

        obj = {"a": 1, "b": [1, 2, 3]}
        assert _guard_output_size(obj) is obj

    def test_small_list_passes_through(self):
        from perflab.server.mcp_server import _guard_output_size

        obj = [{"a": 1}, {"b": 2}]
        assert _guard_output_size(obj) is obj

    def test_oversized_object_truncated(self):
        from perflab.server.mcp_server import _MAX_OUTPUT_BYTES, _guard_output_size

        obj = {"data": "x" * (_MAX_OUTPUT_BYTES * 2)}
        result = _guard_output_size(obj)
        assert result is not obj
        assert result["_truncated"] is True
        assert "get_run_section" in result["_notice"]
        assert result["_original_size_bytes"] > _MAX_OUTPUT_BYTES
        assert len(result["_partial_data"]) <= 50_000 + 3  # payload + "..."


@needs_fastmcp
class TestToDicts:
    def test_dataclass_items(self):
        from perflab.server.mcp_server import _to_dicts

        @dataclasses.dataclass
        class Point:
            x: int
            y: int

        assert _to_dicts([Point(1, 2)]) == [{"x": 1, "y": 2}]

    def test_namedtuple_items(self):
        from perflab.server.mcp_server import _to_dicts

        NT = namedtuple("NT", ["a", "b"])
        assert _to_dicts([NT(1, 2)]) == [{"a": 1, "b": 2}]

    def test_dict_items_pass_through(self):
        from perflab.server.mcp_server import _to_dicts

        assert _to_dicts([{"k": "v"}]) == [{"k": "v"}]

    def test_other_items_stringified(self):
        from perflab.server.mcp_server import _to_dicts

        assert _to_dicts([42]) == [{"value": "42"}]


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

@needs_fastmcp
class TestListRuns:
    def test_empty_out_dir(self, tmp_path):
        from perflab.server.mcp_server import list_runs

        assert list_runs(out_dir=str(tmp_path)) == []

    def test_lists_runs_newest_first(self, tmp_path):
        from perflab.server.mcp_server import list_runs

        rid_a = _make_run(tmp_path, task="alpha")
        rid_b = _make_run(tmp_path, task="beta")
        result = list_runs(out_dir=str(tmp_path))
        assert [r["run_id"] for r in result] == [rid_b, rid_a]

    def test_filter_by_task(self, tmp_path):
        from perflab.server.mcp_server import list_runs

        rid_a = _make_run(tmp_path, task="alpha")
        _make_run(tmp_path, task="beta")
        result = list_runs(task="alpha", out_dir=str(tmp_path))
        assert [r["run_id"] for r in result] == [rid_a]

    def test_limit(self, tmp_path):
        from perflab.server.mcp_server import list_runs

        for _ in range(3):
            _make_run(tmp_path)
        assert len(list_runs(limit=2, out_dir=str(tmp_path))) == 2

    def test_enriched_from_meta(self, tmp_path):
        from perflab.server.mcp_server import list_runs

        rid = _make_run(tmp_path, meta_extra={"best_value": 3.5, "status": "completed"})
        result = list_runs(out_dir=str(tmp_path))
        entry = next(r for r in result if r["run_id"] == rid)
        assert entry["best_value"] == 3.5
        assert entry["status"] == "completed"


@needs_fastmcp
class TestGetRun:
    def test_full_run_data(self, tmp_path):
        from perflab.server.mcp_server import get_run

        rid = _make_run(
            tmp_path,
            report={"best_value": 2.0},
            bench={"ok": True},
            summaries={"pyspy": {"top_functions": []}},
        )
        result = get_run(rid, out_dir=str(tmp_path))
        assert result["run_id"] == rid
        assert result["meta"]["task"] == "demo-task"
        assert result["report"] == {"best_value": 2.0}
        assert result["bench"] == {"ok": True}
        assert result["profiler_summaries"] == {"pyspy": {"top_functions": []}}

    def test_traversal_run_id_rejected(self, tmp_path):
        from perflab.server.mcp_server import get_run

        with pytest.raises(ValueError, match="Invalid run_id"):
            get_run("../escape", out_dir=str(tmp_path))

    def test_missing_run_raises(self, tmp_path):
        from perflab.server.mcp_server import get_run

        with pytest.raises(FileNotFoundError):
            get_run("nonexistent-run", out_dir=str(tmp_path))


@needs_fastmcp
class TestGetRunSection:
    def _run_with_everything(self, tmp_path) -> str:
        return _make_run(
            tmp_path,
            report={"best_value": 2.0},
            bench={"ok": True, "latency_ms": {"p50": 1.0}},
            summaries={"nsys": {"gpu_active_pct": 80.0}, "torch_profiler": {"sync_count": 3}},
            system_info={"cpu_count": 8},
            events=[{"event_type": "baseline_complete", "value": 1.0}],
        )

    def test_meta_section(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = self._run_with_everything(tmp_path)
        result = get_run_section(rid, "meta", out_dir=str(tmp_path))
        assert result["task"] == "demo-task"

    def test_report_section(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = self._run_with_everything(tmp_path)
        assert get_run_section(rid, "report", out_dir=str(tmp_path)) == {"best_value": 2.0}

    def test_bench_section(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = self._run_with_everything(tmp_path)
        result = get_run_section(rid, "bench", out_dir=str(tmp_path))
        assert result["ok"] is True

    def test_missing_optional_section_errors(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = _make_run(tmp_path)  # no report.json
        result = get_run_section(rid, "report", out_dir=str(tmp_path))
        assert "not found" in result["error"]

    def test_system_info_section(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = self._run_with_everything(tmp_path)
        assert get_run_section(rid, "system_info", out_dir=str(tmp_path)) == {"cpu_count": 8}

    def test_system_info_missing(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = _make_run(tmp_path)
        result = get_run_section(rid, "system_info", out_dir=str(tmp_path))
        assert "error" in result

    def test_event_log_section(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = self._run_with_everything(tmp_path)
        result = get_run_section(rid, "event_log", out_dir=str(tmp_path))
        assert result["events"] == [{"event_type": "baseline_complete", "value": 1.0}]

    def test_event_log_missing(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = _make_run(tmp_path)
        result = get_run_section(rid, "event_log", out_dir=str(tmp_path))
        assert "error" in result

    def test_profiler_summaries_section(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = self._run_with_everything(tmp_path)
        result = get_run_section(rid, "profiler_summaries", out_dir=str(tmp_path))
        assert set(result) == {"nsys", "torch_profiler"}

    def test_profiler_name_exact(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = self._run_with_everything(tmp_path)
        result = get_run_section(rid, "nsys", out_dir=str(tmp_path))
        assert result == {"gpu_active_pct": 80.0}

    def test_profiler_name_substring_fallback(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = self._run_with_everything(tmp_path)
        # "torch" is not a key but matches "torch_profiler"
        result = get_run_section(rid, "torch", out_dir=str(tmp_path))
        assert result == {"sync_count": 3}

    def test_unknown_section_lists_available(self, tmp_path):
        from perflab.server.mcp_server import get_run_section

        rid = self._run_with_everything(tmp_path)
        result = get_run_section(rid, "bogus", out_dir=str(tmp_path))
        assert "Unknown section" in result["error"]
        assert "meta" in result["available_sections"]
        assert "nsys" in result["available_sections"]


@needs_fastmcp
class TestCompareRuns:
    def test_delta_ratio_and_bottleneck_diff(self, tmp_path):
        from perflab.server.mcp_server import compare_runs

        rid_a = _make_run(
            tmp_path,
            report={"best_value": 1.0, "bottleneck_diagnoses": [{"bottleneck": "gpu idle"}]},
        )
        rid_b = _make_run(
            tmp_path,
            report={"best_value": 2.0, "bottleneck_diagnoses": [{"bottleneck": "memory bound"}]},
        )
        result = compare_runs(rid_a, rid_b, out_dir=str(tmp_path))
        assert result["value_a"] == 1.0
        assert result["value_b"] == 2.0
        assert result["delta"] == 1.0
        assert result["ratio"] == 2.0
        assert result["resolved_bottlenecks"] == ["gpu idle"]
        assert result["new_bottlenecks"] == ["memory bound"]


@needs_fastmcp
class TestReplayRun:
    def test_traversal_run_id_rejected(self, tmp_path):
        from perflab.server.mcp_server import replay_run

        result = replay_run("../escape", out_dir=str(tmp_path))
        assert "Invalid run_id" in result["error"]

    def test_missing_run_dir(self, tmp_path):
        from perflab.server.mcp_server import replay_run

        result = replay_run("no-such-run", out_dir=str(tmp_path))
        assert "not found" in result["error"]

    def test_replays_events(self, tmp_path):
        from perflab.server.mcp_server import replay_run

        rid = _make_run(tmp_path, events=[{"event_type": "baseline_complete", "value": 1.5}])
        result = replay_run(rid, out_dir=str(tmp_path))
        assert result["run_id"] == rid
        assert "Agent Run Replay" in result["summary"]
        assert "Baseline" in result["summary"]


# ---------------------------------------------------------------------------
# Analysis tools
# ---------------------------------------------------------------------------

@needs_fastmcp
class TestGetBottlenecks:
    def test_no_summaries_returns_empty(self, tmp_path):
        from perflab.server.mcp_server import get_bottlenecks

        rid = _make_run(tmp_path)
        assert get_bottlenecks(rid, out_dir=str(tmp_path)) == []

    def test_diagnoses_from_summaries(self, tmp_path):
        from perflab.server.mcp_server import get_bottlenecks

        # sm_utilization_pct far below default ncu_sm_util_low threshold
        rid = _make_run(
            tmp_path,
            program_type="cuda",
            summaries={"ncu": {"sm_utilization_pct": 5.0}},
        )
        result = get_bottlenecks(rid, out_dir=str(tmp_path))
        assert len(result) >= 1
        for diag in result:
            assert set(diag) == {"rank", "bottleneck", "root_cause", "confidence", "suggested_actions"}
        assert result[0]["rank"] == 1


@needs_fastmcp
class TestGetRooflineAnalysis:
    def test_no_bench_data(self, tmp_path):
        from perflab.server.mcp_server import get_roofline_analysis

        rid = _make_run(tmp_path)
        result = get_roofline_analysis(rid, out_dir=str(tmp_path))
        assert "No benchmark data" in result["error"]

    def test_matmul_roofline_with_peaks(self, tmp_path, monkeypatch):
        import perflab.roofline_peaks
        from perflab.server.mcp_server import get_roofline_analysis

        fake_peaks = SimpleNamespace(
            device="Fake GPU", source="test", peak_tflops=100.0,
            peak_mem_bw_gbs=1000.0, dtype_peaks={"fp32": 100.0},
        )
        monkeypatch.setattr(perflab.roofline_peaks, "infer_peaks", lambda *a, **k: fake_peaks)

        rid = _make_run(
            tmp_path,
            bench={"meta": {"M": 1024, "N": 1024, "K": 1024}, "latency_ms": {"p50": 10.0}},
        )
        result = get_roofline_analysis(rid, out_dir=str(tmp_path))
        assert result["run_id"] == rid
        assert result["roofline_point"]["tflops"] > 0
        assert result["peaks"]["device"] == "Fake GPU"
        assert result["peaks"]["dtype_peaks"] == {"fp32": 100.0}
        assert result["pct_of_peak"] > 0

    def test_no_peaks_detected(self, tmp_path, monkeypatch):
        import perflab.roofline_peaks
        from perflab.server.mcp_server import get_roofline_analysis

        monkeypatch.setattr(perflab.roofline_peaks, "infer_peaks", lambda *a, **k: None)
        rid = _make_run(
            tmp_path,
            bench={"meta": {"M": 64, "N": 64, "K": 64}, "latency_ms": {"p50": 5.0}},
        )
        result = get_roofline_analysis(rid, out_dir=str(tmp_path))
        assert "roofline_point" in result
        assert "peaks" not in result


@needs_fastmcp
class TestGetProfileDiff:
    def test_requires_summaries_in_both_runs(self, tmp_path):
        from perflab.server.mcp_server import get_profile_diff

        rid_a = _make_run(tmp_path, summaries={"linux_perf": {"ipc": 1.0}})
        rid_b = _make_run(tmp_path)  # no summaries
        result = get_profile_diff(rid_a, rid_b, out_dir=str(tmp_path))
        assert "error" in result

    def test_diffs_metrics(self, tmp_path):
        from perflab.server.mcp_server import get_profile_diff

        rid_a = _make_run(tmp_path, summaries={"linux_perf": {"ipc": 1.0}})
        rid_b = _make_run(tmp_path, summaries={"linux_perf": {"ipc": 2.0}})
        result = get_profile_diff(rid_a, rid_b, out_dir=str(tmp_path))
        assert result["run_a"] == rid_a
        assert result["run_b"] == rid_b
        ipc = next(d for d in result["deltas"] if d["metric"] == "linux_perf.ipc")
        assert ipc["before"] == 1.0
        assert ipc["after"] == 2.0
        assert isinstance(result["hotspot_shifts"], list)


@needs_fastmcp
class TestGetThresholds:
    def test_defaults(self):
        from perflab.analyzers.bottleneck_analyzer import AnalysisThresholds
        from perflab.server.mcp_server import get_thresholds

        result = get_thresholds()
        assert result["total_fields"] == len(dataclasses.fields(AnalysisThresholds))
        assert result["overridden_count"] == 0
        assert result["thresholds"]["ncu_sm_util_low"]["value"] == 50.0

    def test_task_overrides_marked(self, tmp_path, monkeypatch):
        import perflab.config
        from perflab.config import PerfLabConfig
        from perflab.server.mcp_server import get_thresholds

        monkeypatch.setattr(perflab.config, "load_config", lambda: PerfLabConfig())
        task_file = tmp_path / "task.yaml"
        task_file.write_text(
            "name: t\n"
            "program_type: python\n"
            "correctness:\n  cmd: 'true'\n"
            "benchmark:\n  cmd: 'true'\n  metric:\n    name: x\n    mode: maximize\n"
            "analysis_thresholds:\n  ncu_sm_util_low: 42.0\n",
            encoding="utf-8",
        )
        result = get_thresholds(str(task_file))
        entry = result["thresholds"]["ncu_sm_util_low"]
        assert entry["value"] == 42.0
        assert entry["overridden"] is True
        assert entry["default"] == 50.0
        assert result["overridden_count"] == 1


@needs_fastmcp
class TestGetBuildRecommendations:
    def test_no_build_cmd_errors(self, sample_task_yaml):
        from perflab.server.mcp_server import get_build_recommendations

        result = get_build_recommendations(str(sample_task_yaml))
        assert "no build command" in result["error"]

    def test_traversal_run_id_rejected(self, sample_cpp_task_yaml):
        from perflab.server.mcp_server import get_build_recommendations

        result = get_build_recommendations(str(sample_cpp_task_yaml), run_id="../escape")
        assert "Invalid run_id" in result["error"]

    def test_static_recommendations(self, sample_cpp_task_yaml):
        from perflab.server.mcp_server import get_build_recommendations

        result = get_build_recommendations(str(sample_cpp_task_yaml))
        assert result["build_cmd"].startswith("g++")
        assert result["program_type"] == "cpp"
        assert isinstance(result["isa_recommendations"], list)
        assert result["profiler_recommendations"] == []


# ---------------------------------------------------------------------------
# Environment tools (internal probes mocked at their import sites)
# ---------------------------------------------------------------------------

@needs_fastmcp
class TestDoctorCheck:
    def test_aggregates_check_results(self, monkeypatch):
        import perflab.doctor
        from perflab.server.mcp_server import doctor_check

        fake_results = [
            SimpleNamespace(name="python", status="pass", message="3.12"),
            SimpleNamespace(name="torch", status="warn", message="not installed"),
            SimpleNamespace(name="llm", status="fail", message="no key"),
        ]
        monkeypatch.setattr(perflab.doctor, "run_doctor", lambda **kw: fake_results)
        result = doctor_check()
        assert result["summary"] == {"passed": 1, "warnings": 1, "failures": 1}
        assert result["ready"] is False
        assert result["checks"][0] == {"name": "python", "status": "pass", "message": "3.12"}

    def test_ready_when_no_failures(self, monkeypatch):
        import perflab.doctor
        from perflab.server.mcp_server import doctor_check

        fake_results = [SimpleNamespace(name="python", status="pass", message="ok")]
        monkeypatch.setattr(perflab.doctor, "run_doctor", lambda **kw: fake_results)
        assert doctor_check()["ready"] is True


@needs_fastmcp
class TestGetPeaks:
    def _patch_probes(self, monkeypatch, peaks):
        import perflab.roofline_peaks as rp

        monkeypatch.setattr(rp, "infer_peaks", lambda *a, **k: peaks)
        monkeypatch.setattr(rp, "list_cuda_gpus", lambda: [])
        monkeypatch.setattr(rp, "list_metal_gpus", lambda: [{"name": "Fake Metal"}])
        monkeypatch.setattr(rp, "selection_hints", lambda: {"auto": "use auto"})

    def test_peaks_detected(self, monkeypatch):
        from perflab.server.mcp_server import get_peaks

        fake_peaks = SimpleNamespace(
            device="Fake GPU", source="table", peak_tflops=10.0,
            peak_mem_bw_gbs=200.0, dtype_peaks={"fp16": 20.0},
        )
        self._patch_probes(monkeypatch, fake_peaks)
        result = get_peaks()
        assert result["peaks"]["device"] == "Fake GPU"
        assert result["peaks"]["dtype_peaks"] == {"fp16": 20.0}
        assert result["metal_gpus"] == [{"name": "Fake Metal"}]
        assert result["hints"] == {"auto": "use auto"}

    def test_no_peaks(self, monkeypatch):
        from perflab.server.mcp_server import get_peaks

        self._patch_probes(monkeypatch, None)
        result = get_peaks()
        assert result["peaks"] is None


# ---------------------------------------------------------------------------
# Profiling and agent tools
# ---------------------------------------------------------------------------

@needs_fastmcp
class TestProfileTask:
    def test_collects_summaries(self, sample_task_yaml, tmp_path, monkeypatch):
        import perflab.orchestrator
        from perflab.server.mcp_server import profile_task

        run_dir = tmp_path / "fake_run"
        (run_dir / "artifacts").mkdir(parents=True)
        (run_dir / "artifacts" / "pyspy_summary.json").write_text(
            json.dumps({"top_functions": ["f"]}), encoding="utf-8"
        )
        (run_dir / "artifacts" / "broken_summary.json").write_text("{not json", encoding="utf-8")

        monkeypatch.setattr(perflab.orchestrator, "profile_only", lambda task: run_dir)
        result = profile_task(str(sample_task_yaml))
        assert result["run_dir"] == str(run_dir)
        assert result["profiler_summaries"] == {"pyspy": {"top_functions": ["f"]}}


def _wait_for_terminal(job_id: str, deadline_s: float = 20.0) -> dict:
    from perflab.server.mcp_server import get_agent_progress

    info: dict = {}
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        info = get_agent_progress(job_id)
        if info.get("status") in ("completed", "failed"):
            return info
        time.sleep(0.05)
    pytest.fail(f"job {job_id} did not reach a terminal status: {info}")


@pytest.fixture
def stub_agent_env(tmp_path, monkeypatch):
    """Task yaml + stubbed run_agent/LLM/config for agent-tool tests."""
    import perflab.config
    import perflab.optimizers.agent
    from perflab.config import PerfLabConfig
    from perflab.llm.config import LLMConfig
    from perflab.server.mcp_server import create_task

    created = create_task(name="agent_stub", program_type="python", tasks_root=str(tmp_path))
    assert "error" not in created

    monkeypatch.setattr(perflab.config, "load_config", lambda: PerfLabConfig())
    monkeypatch.setattr(
        LLMConfig, "load", staticmethod(lambda path=None: LLMConfig(provider="ollama", model="stub"))
    )

    def fake_run_agent(task, task_file, config, llm_config, *, expert_suggestion=None,
                       progress=None, provider=None):
        if progress is not None:
            progress.on_message("stub iteration")
        return SimpleNamespace(
            best_value=2.5, best_iter=3, baseline_value=1.0, run_dir=tmp_path / "agent_run",
        )

    monkeypatch.setattr(perflab.optimizers.agent, "run_agent", fake_run_agent)
    return created["task_yaml"]


@needs_fastmcp
class TestStartAgent:
    def test_background_run_completes(self, stub_agent_env, monkeypatch):
        from perflab.server.mcp_server import start_agent

        # The shipped default is "auto", which resolves via bwrap availability —
        # pin it so isolation == "none" holds on sandbox-capable Linux hosts too.
        monkeypatch.setattr("perflab.tools.isolation._bwrap_usable", lambda: False)
        started = start_agent(stub_agent_env, iters=2, candidates=1)
        assert "job_id" in started
        assert started["status"] in ("starting", "running")

        info = _wait_for_terminal(started["job_id"])
        assert info["status"] == "completed", info
        assert info["result"]["best_value"] == 2.5
        assert info["result"]["best_iter"] == 3
        assert info["result"]["baseline_value"] == 1.0
        assert info["result"]["run_dir"].endswith("agent_run")
        assert info["result"]["isolation"] == "none"
        assert "stub iteration" in info["recent_messages"]

    def test_rejected_while_agent_lock_held(self, stub_agent_env):
        from perflab.server import mcp_server

        assert mcp_server._agent_lock.acquire(timeout=1)
        try:
            started = mcp_server.start_agent(stub_agent_env)
            # Rejection is immediate — no worker thread is spawned
            assert started["status"] == "failed"
            assert "already in progress" in started["error"]
            # get_agent_progress stays consistent with the returned job_id
            info = mcp_server.get_agent_progress(started["job_id"])
            assert info["status"] == "failed"
            assert "already in progress" in info["error"]
        finally:
            mcp_server._agent_lock.release()

    def test_lock_released_after_rejection(self, stub_agent_env):
        """A rejected start must not leak the agent lock or executor slot."""
        from perflab.server import mcp_server

        assert mcp_server._agent_lock.acquire(timeout=1)
        try:
            mcp_server.start_agent(stub_agent_env)
        finally:
            mcp_server._agent_lock.release()

        started = mcp_server.start_agent(stub_agent_env, iters=1, candidates=1)
        info = _wait_for_terminal(started["job_id"])
        assert info["status"] == "completed"


@needs_fastmcp
class TestActiveRunEviction:
    def _agent_module(self):
        import importlib

        from perflab.server.mcp_server import start_agent

        return importlib.import_module(start_agent.__module__)

    def test_eviction_keeps_in_flight_and_recent(self, monkeypatch):
        mod = self._agent_module()
        runs = {
            "old-done": {"status": "completed"},
            "mid-fail": {"status": "failed"},
            "running": {"status": "running"},
            "new-done": {"status": "completed"},
        }
        monkeypatch.setattr(mod, "_active_runs", runs)
        monkeypatch.setattr(mod, "_MAX_TERMINAL_RUNS", 2)

        mod._register_job("fresh", {"status": "starting", "progress": None})
        assert "old-done" not in runs  # oldest terminal evicted
        assert set(runs) == {"mid-fail", "running", "new-done", "fresh"}

    def test_no_eviction_below_cap(self, monkeypatch):
        mod = self._agent_module()
        runs = {"done": {"status": "completed"}}
        monkeypatch.setattr(mod, "_active_runs", runs)
        monkeypatch.setattr(mod, "_MAX_TERMINAL_RUNS", 2)

        mod._register_job("fresh", {"status": "starting", "progress": None})
        assert set(runs) == {"done", "fresh"}

    def test_start_agent_registers_through_eviction(self, stub_agent_env, monkeypatch):
        mod = self._agent_module()
        runs = {
            "a": {"status": "completed"},
            "b": {"status": "completed"},
            "c": {"status": "completed"},
        }
        monkeypatch.setattr(mod, "_active_runs", runs)
        monkeypatch.setattr(mod, "_MAX_TERMINAL_RUNS", 2)

        started = mod.start_agent(stub_agent_env, iters=1, candidates=1)
        info = _wait_for_terminal(started["job_id"])
        assert info["status"] == "completed"
        assert "a" not in runs  # oldest terminal evicted at registration
        assert started["job_id"] in runs


@needs_fastmcp
class TestProgramTypesConstant:
    def test_single_source_of_truth(self):
        from perflab.server.mcp_server import _PROGRAM_TYPES

        assert _PROGRAM_TYPES == ("python", "pytorch", "jax", "triton", "cpp", "cuda")

    def test_invalid_type_error_lists_all_types(self, tmp_path):
        from perflab.server.mcp_server import _PROGRAM_TYPES, create_task, suggest_profilers

        for result in (
            create_task(name="x", program_type="rust", tasks_root=str(tmp_path)),
            suggest_profilers("rust"),
        ):
            for pt in _PROGRAM_TYPES:
                assert pt in result["error"]


@needs_fastmcp
class TestGetAgentProgress:
    def test_unknown_job_id(self):
        from perflab.server.mcp_server import get_agent_progress

        result = get_agent_progress("no-such-job")
        assert "Unknown job_id" in result["error"]


@needs_fastmcp
class TestOptimizeTask:
    def test_no_context_errors(self):
        from perflab.server.mcp_server import optimize_task

        result = asyncio.run(optimize_task("task.yaml", ctx=None))
        assert "No MCP context" in result["error"]
