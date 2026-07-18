"""Tests for TMA Level 2/3, profiler-driven flag recommendations, and flag tracking."""
from __future__ import annotations

from perflab.analyzers.bottleneck_analyzer import (
    diagnose_bottlenecks,
)
from perflab.analyzers.build_flags import (
    recommend_flags_from_profiling,
)
from perflab.analyzers.tma import (
    TMALevel2Result,
    TMAResult,
    _parse_amd_tma,
    _parse_toplev_output,
    format_tma_summary,
)

# ---------------------------------------------------------------------------
# TMA Level 2 parsing (toplev)
# ---------------------------------------------------------------------------

class TestParseToplev:
    def test_csv_format(self):
        text = """\
Area,Value,Unit
Frontend_Bound,15.2,%
Frontend_Bound.Fetch_Latency,10.1,%
Frontend_Bound.Fetch_Bandwidth,5.1,%
Backend_Bound,52.3,%
Backend_Bound.Memory_Bound,38.5,%
Backend_Bound.Core_Bound,13.8,%
Backend_Bound.Memory_Bound.L1_Bound,5.2,%
Backend_Bound.Memory_Bound.L2_Bound,8.3,%
Backend_Bound.Memory_Bound.L3_Bound,3.0,%
Backend_Bound.Memory_Bound.DRAM_Bound,18.5,%
Backend_Bound.Memory_Bound.Store_Bound,3.5,%
"""
        result = _parse_toplev_output(text)
        assert result is not None
        assert result.memory_bound_pct == 38.5
        assert result.core_bound_pct == 13.8
        assert result.fetch_latency_pct == 10.1
        assert result.l1_bound_pct == 5.2
        assert result.l2_bound_pct == 8.3
        assert result.dram_bound_pct == 18.5
        assert result.store_bound_pct == 3.5
        assert result.source == "toplev"

    def test_space_format(self):
        text = """\
backend_bound.memory_bound 42.0%
backend_bound.core_bound 15.0%
backend_bound.memory_bound.dram_bound 30.0%
"""
        result = _parse_toplev_output(text)
        assert result is not None
        assert result.memory_bound_pct == 42.0
        assert result.dram_bound_pct == 30.0

    def test_empty_input(self):
        assert _parse_toplev_output("") is None

    def test_no_relevant_metrics(self):
        assert _parse_toplev_output("random text\nno metrics here\n") is None

    def test_to_dict_dominant_level(self):
        result = TMALevel2Result(
            memory_bound_pct=40.0,
            l1_bound_pct=5.0,
            l2_bound_pct=10.0,
            dram_bound_pct=25.0,
        )
        d = result.to_dict()
        assert d["dominant_memory_level"] == "DRAM"
        assert d["memory_bound_pct"] == 40.0

    def test_to_dict_l1_dominant(self):
        result = TMALevel2Result(
            memory_bound_pct=30.0,
            l1_bound_pct=20.0,
            l2_bound_pct=5.0,
            dram_bound_pct=5.0,
        )
        d = result.to_dict()
        assert d["dominant_memory_level"] == "L1"


# ---------------------------------------------------------------------------
# AMD TMA parsing
# ---------------------------------------------------------------------------

class TestParseAmdTma:
    def test_basic_parsing(self):
        text = """\
 Performance counter stats:
      1,000,000      L1-dcache-loads
        100,000      L1-dcache-load-misses
         50,000      LLC-loads
         25,000      LLC-load-misses
"""
        result = _parse_amd_tma(text)
        assert result is not None
        assert result.source == "amd-perf"
        # L1 miss rate = 100k/1M = 10%, LLC miss rate = 25k/50k = 50%
        assert result.dram_bound_pct is not None
        assert result.dram_bound_pct > 0

    def test_no_loads_returns_none(self):
        assert _parse_amd_tma("random text") is None


# ---------------------------------------------------------------------------
# TMA bottleneck rules
# ---------------------------------------------------------------------------

class TestTmaBottleneckRules:
    def test_dram_bound_detected(self):
        summaries = {
            "linux_perf": {
                "ipc": 0.8,
                "tma_level2": {
                    "memory_bound_pct": 45.0,
                    "dram_bound_pct": 35.0,
                    "dominant_memory_level": "DRAM",
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cpp")
        tma_diags = [d for d in diags if "DRAM Bound" in d.bottleneck]
        assert len(tma_diags) >= 1

    def test_l1_bound_detected(self):
        summaries = {
            "linux_perf": {
                "ipc": 0.8,
                "tma_level2": {
                    "memory_bound_pct": 40.0,
                    "l1_bound_pct": 30.0,
                    "dominant_memory_level": "L1",
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cpp")
        tma_diags = [d for d in diags if "L1 Bound" in d.bottleneck]
        assert len(tma_diags) >= 1

    def test_core_bound_detected(self):
        summaries = {
            "linux_perf": {
                "ipc": 0.8,
                "tma_level2": {
                    "core_bound_pct": 35.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cpp")
        core_diags = [d for d in diags if "Core Bound" in d.bottleneck]
        assert len(core_diags) >= 1

    def test_low_memory_bound_no_finding(self):
        summaries = {
            "linux_perf": {
                "ipc": 2.0,
                "tma_level2": {
                    "memory_bound_pct": 10.0,
                    "dram_bound_pct": 5.0,
                    "dominant_memory_level": "DRAM",
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cpp")
        tma_diags = [d for d in diags if "TMA Level" in d.bottleneck]
        assert len(tma_diags) == 0


# ---------------------------------------------------------------------------
# Profiler-driven flag recommendations
# ---------------------------------------------------------------------------

class TestProfilerDrivenFlags:
    def test_high_cache_miss_suggests_prefetch(self):
        recs = recommend_flags_from_profiling(
            "g++ -O3 -o bin main.cpp",
            {"linux_perf": {"cache_miss_rate": 0.12}},
            "cpp",
        )
        flags = [r.flag for r in recs]
        assert "-fprefetch-loop-arrays" in flags

    def test_frontend_bound_suggests_alignment(self):
        recs = recommend_flags_from_profiling(
            "g++ -O3 -o bin main.cpp",
            {"linux_perf": {"tma": {"frontend_bound_pct": 35.0}}},
            "cpp",
        )
        flags = [r.flag for r in recs]
        assert "-falign-functions=32" in flags

    def test_bad_speculation_suggests_pgo(self):
        recs = recommend_flags_from_profiling(
            "g++ -O3 -o bin main.cpp",
            {"linux_perf": {
                "tma": {"bad_speculation_pct": 30.0},
                "branch_miss_rate": 0.08,
            }},
            "cpp",
        )
        flags = [r.flag for r in recs]
        assert any("profile" in f.lower() for f in flags)

    def test_low_ipc_suggests_unroll(self):
        recs = recommend_flags_from_profiling(
            "g++ -O3 -o bin main.cpp",
            {"linux_perf": {"ipc": 0.3}},
            "cpp",
        )
        flags = [r.flag for r in recs]
        assert "-funroll-loops" in flags

    def test_no_issues_no_flags(self):
        recs = recommend_flags_from_profiling(
            "g++ -O3 -march=native -o bin main.cpp",
            {"linux_perf": {"ipc": 2.5, "cache_miss_rate": 0.01}},
            "cpp",
        )
        assert len(recs) == 0

    def test_python_returns_empty(self):
        recs = recommend_flags_from_profiling(
            "python main.py",
            {"linux_perf": {"cache_miss_rate": 0.5}},
            "python",
        )
        assert len(recs) == 0

    def test_dram_bound_suggests_unroll(self):
        recs = recommend_flags_from_profiling(
            "g++ -O3 -o bin main.cpp",
            {"linux_perf": {"tma_level2": {"dominant_memory_level": "DRAM"}}},
            "cpp",
        )
        flags = [r.flag for r in recs]
        assert "-funroll-loops" in flags


# ---------------------------------------------------------------------------
# Format TMA with Level 2
# ---------------------------------------------------------------------------

class TestFormatTmaWithLevel2:
    def test_format_with_level2(self):
        tma = TMAResult(
            frontend_bound_pct=15.0,
            backend_bound_pct=55.0,
            bad_speculation_pct=5.0,
            retiring_pct=25.0,
        )
        l2 = TMALevel2Result(
            memory_bound_pct=40.0,
            core_bound_pct=15.0,
            l1_bound_pct=5.0,
            dram_bound_pct=30.0,
        )
        text = format_tma_summary(tma, l2)
        assert "Level 2 breakdown" in text
        assert "Memory Bound: 40.0%" in text
        assert "Level 3 memory hierarchy" in text
        assert "DRAM Bound: 30.0%" in text

    def test_format_without_level2(self):
        tma = TMAResult(
            frontend_bound_pct=15.0,
            backend_bound_pct=55.0,
            bad_speculation_pct=5.0,
            retiring_pct=25.0,
        )
        text = format_tma_summary(tma)
        assert "Level 2" not in text
        assert "Backend Bound:  55.0%" in text


# ---------------------------------------------------------------------------
# Event log build flags
# ---------------------------------------------------------------------------

class TestEventLogBuildFlags:
    def test_build_flags_state_event(self, tmp_path):
        from perflab.optimizers.event_log import AgentEventLog
        log = AgentEventLog(run_dir=tmp_path)
        log.build_flags_state(
            iteration=1,
            build_cmd="g++ -O3 -march=native -o bin main.cpp",
            flags_recommended=[{"flag": "-flto"}, {"flag": "-funroll-loops"}],
        )

        import json
        events_path = tmp_path / "agent_events.jsonl"
        assert events_path.exists()
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        assert len(events) == 1
        assert events[0]["event_type"] == "build_flags_state"
        assert events[0]["build_cmd"] == "g++ -O3 -march=native -o bin main.cpp"
        assert len(events[0]["flags_recommended"]) == 2
