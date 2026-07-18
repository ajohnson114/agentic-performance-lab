"""Tests for temporal CPU-GPU attribution linking.

Tests the attribution strategies:
1. Call chain walking (caller_function propagation)
2. Temporal NVTX matching (kernel within NVTX time window)
3. Py-spy temporal join (Python function during kernel launch)

Torch-trace timestamp cross-referencing was removed (separate profiling run
and clock domain than nsys — no valid alignment), so tests also verify that
torch summaries do NOT produce temporal framework_op matches.
"""
from __future__ import annotations

from perflab.analyzers.gpu_attribution import (
    CpuGpuEdge,
    _build_nvtx_temporal_index,
    _build_pyspy_temporal_index,
    _find_best_enclosing,
    _match_kernel_to_nvtx_temporal,
    _match_kernel_to_pyspy_temporal,
    _most_common_caller,
    build_cpu_gpu_call_graph,
    build_kernel_dossiers,
    compute_attribution_ranking,
    enrich_with_framework_context,
)

# ---------------------------------------------------------------------------
# 1. Call chain walking — caller_function propagation
# ---------------------------------------------------------------------------

class TestCallChainPropagation:
    def test_caller_propagated_to_edge(self):
        """Call chain walking enriches correlations with caller_function."""
        correlations = [
            {"api_name": "cudaLaunchKernel", "kernel_name": "sgemm",
             "stream_id": 1, "gpu_duration_ns": 5_000_000,
             "launch_overhead_ns": 10_000,
             "caller_function": "forward"},
            {"api_name": "cudaLaunchKernel", "kernel_name": "sgemm",
             "stream_id": 1, "gpu_duration_ns": 4_000_000,
             "launch_overhead_ns": 12_000,
             "caller_function": "forward"},
        ]
        edges = build_cpu_gpu_call_graph(correlations)
        assert len(edges) == 1
        assert edges[0].caller_function == "forward"

    def test_most_common_caller_wins(self):
        """When multiple callers exist, the most common one is used."""
        correlations = [
            {"api_name": "cudaLaunchKernel", "kernel_name": "relu",
             "stream_id": 1, "gpu_duration_ns": 1_000_000,
             "launch_overhead_ns": 5_000,
             "caller_function": "train_step"},
            {"api_name": "cudaLaunchKernel", "kernel_name": "relu",
             "stream_id": 1, "gpu_duration_ns": 1_000_000,
             "launch_overhead_ns": 5_000,
             "caller_function": "train_step"},
            {"api_name": "cudaLaunchKernel", "kernel_name": "relu",
             "stream_id": 1, "gpu_duration_ns": 1_000_000,
             "launch_overhead_ns": 5_000,
             "caller_function": "val_step"},
        ]
        edges = build_cpu_gpu_call_graph(correlations)
        assert edges[0].caller_function == "train_step"

    def test_no_caller_gives_none(self):
        """Without caller_function in correlations, edge.caller is None."""
        correlations = [
            {"api_name": "cudaLaunchKernel", "kernel_name": "sgemm",
             "stream_id": 1, "gpu_duration_ns": 5_000_000,
             "launch_overhead_ns": 10_000},
        ]
        edges = build_cpu_gpu_call_graph(correlations)
        assert edges[0].caller_function is None

    def test_most_common_caller_empty(self):
        assert _most_common_caller([]) is None
        assert _most_common_caller([{"other": "data"}]) is None

    def test_caller_used_for_cpu_matching(self):
        """caller_function enables CPU hotspot matching when fuzzy name fails."""
        nsys = {
            "top_kernels": [
                {"name": "volta_sgemm_128x128_nn", "pct": 85.0, "total_ms": 100},
            ],
            "cpu_gpu_correlations": [
                {"api_name": "cudaLaunchKernel",
                 "kernel_name": "volta_sgemm_128x128_nn",
                 "stream_id": 1, "gpu_duration_ns": 100_000_000,
                 "launch_overhead_ns": 10_000,
                 "cpu_start_ns": 1_000_000, "gpu_start_ns": 1_100_000,
                 "caller_function": "train_step"},
            ],
        }
        perf = {
            "hotspots": [
                {"function": "train_step", "pct": 45.0},
            ],
        }
        ranking = compute_attribution_ranking(nsys, perf)
        assert len(ranking) > 0
        # Should match via caller_function even though kernel name != CPU function
        assert ranking[0].cpu_pct == 45.0
        assert ranking[0].caller_function == "train_step"


# ---------------------------------------------------------------------------
# 2. Temporal NVTX matching
# ---------------------------------------------------------------------------

class TestTemporalNvtxMatching:
    def test_build_nvtx_temporal_index(self):
        nvtx = [
            {"name": "aten::mm", "duration_ms": 5.0, "pct": 50.0,
             "start_ns": 1000, "end_ns": 6_000_000},
            {"name": "aten::relu", "duration_ms": 2.0, "pct": 20.0,
             "start_ns": 7_000_000, "end_ns": 9_000_000},
        ]
        index = _build_nvtx_temporal_index(nvtx)
        assert len(index) == 2
        assert index[0][2] == "aten::mm"  # sorted by start
        assert index[1][2] == "aten::relu"

    def test_build_index_skips_unnamed(self):
        nvtx = [
            {"name": "(unnamed)", "start_ns": 100, "end_ns": 200},
            {"name": "", "start_ns": 300, "end_ns": 400},
        ]
        assert _build_nvtx_temporal_index(nvtx) == []

    def test_build_index_skips_no_timestamps(self):
        nvtx = [{"name": "aten::mm", "duration_ms": 5.0}]
        assert _build_nvtx_temporal_index(nvtx) == []

    def test_kernel_matched_to_enclosing_nvtx(self):
        """Kernel launch that falls within an NVTX range is attributed to it."""
        nvtx_index = [(1000, 10_000_000, "aten::mm"), (10_000_000, 20_000_000, "aten::relu")]
        # cpu_start_ns=5_000_000 falls within aten::mm range
        timestamps = [(5_000_000, 5_500_000)]
        result = _match_kernel_to_nvtx_temporal(timestamps, nvtx_index)
        assert result == "aten::mm"

    def test_kernel_matched_to_most_specific_nvtx(self):
        """When nested NVTX ranges exist, the most specific (shortest) wins."""
        nvtx_index = [
            (1000, 50_000_000, "forward"),      # broad range
            (2_000_000, 8_000_000, "aten::mm"),  # narrow range nested inside
        ]
        timestamps = [(3_000_000, 3_500_000)]
        result = _match_kernel_to_nvtx_temporal(timestamps, nvtx_index)
        assert result == "aten::mm"  # most specific

    def test_no_match_outside_ranges(self):
        nvtx_index = [(1000, 5_000_000, "aten::mm")]
        timestamps = [(10_000_000, 10_500_000)]
        result = _match_kernel_to_nvtx_temporal(timestamps, nvtx_index)
        assert result is None

    def test_empty_inputs(self):
        assert _match_kernel_to_nvtx_temporal([], [(1, 2, "x")]) is None
        assert _match_kernel_to_nvtx_temporal([(1, 2)], []) is None

    def test_temporal_nvtx_in_ranking(self):
        """Temporal NVTX matching works in the full ranking pipeline."""
        nsys = {
            "top_kernels": [
                {"name": "volta_sgemm_128x128_nn", "pct": 85.0, "total_ms": 100},
            ],
            "cpu_gpu_correlations": [
                {"api_name": "cudaLaunchKernel",
                 "kernel_name": "volta_sgemm_128x128_nn",
                 "stream_id": 1, "gpu_duration_ns": 100_000_000,
                 "launch_overhead_ns": 10_000,
                 "cpu_start_ns": 5_000_000, "gpu_start_ns": 5_100_000},
            ],
            "nvtx_ranges": [
                {"name": "aten::mm", "duration_ms": 10.0, "pct": 80.0,
                 "start_ns": 1_000_000, "end_ns": 15_000_000},
            ],
        }
        ranking = compute_attribution_ranking(nsys)
        assert ranking[0].framework_op == "aten::mm"


# ---------------------------------------------------------------------------
# 3. Torch trace cross-reference is disabled (cross-clock matching removed)
# ---------------------------------------------------------------------------

class TestTorchTraceCrossRefDisabled:
    def test_torch_summary_produces_no_temporal_match(self):
        """Torch trace timestamps come from a separate run/clock domain than
        nsys, so no framework_op may be derived from timestamp matching —
        even when the (meaningless) numbers happen to overlap."""
        nsys = {
            "top_kernels": [
                {"name": "volta_sgemm_128x128_nn", "pct": 85.0, "total_ms": 100},
            ],
            "cpu_gpu_correlations": [
                {"api_name": "cudaLaunchKernel",
                 "kernel_name": "volta_sgemm_128x128_nn",
                 "stream_id": 1, "gpu_duration_ns": 100_000_000,
                 "launch_overhead_ns": 10_000,
                 "cpu_start_ns": 3_000_000, "gpu_start_ns": 3_100_000},
            ],
        }
        torch_summary = {
            "_raw_cpu_ops": [
                # Would "enclose" cpu_start_ns after us→ns conversion, but the
                # clocks are unrelated so this must not produce a match.
                {"name": "aten::mm", "ts": 1000, "dur": 5000},
            ],
        }
        ranking = compute_attribution_ranking(nsys, torch_summary=torch_summary)
        assert ranking[0].framework_op is None

    def test_nvtx_still_matches_with_torch_summary_present(self):
        """NVTX ranges (recorded in the same nsys run) still drive
        framework_op attribution when a torch summary is also passed."""
        nsys = {
            "top_kernels": [
                {"name": "volta_sgemm_128x128_nn", "pct": 85.0, "total_ms": 100},
            ],
            "cpu_gpu_correlations": [
                {"api_name": "cudaLaunchKernel",
                 "kernel_name": "volta_sgemm_128x128_nn",
                 "stream_id": 1, "gpu_duration_ns": 100_000_000,
                 "launch_overhead_ns": 10_000,
                 "cpu_start_ns": 5_000_000, "gpu_start_ns": 5_100_000},
            ],
            "nvtx_ranges": [
                {"name": "aten::mm", "duration_ms": 10.0, "pct": 80.0,
                 "start_ns": 1_000_000, "end_ns": 15_000_000},
            ],
        }
        torch_summary = {
            "_raw_cpu_ops": [
                {"name": "torch_only_op", "ts": 1000, "dur": 5000},
            ],
        }
        ranking = compute_attribution_ranking(nsys, torch_summary=torch_summary)
        assert ranking[0].framework_op == "aten::mm"


# ---------------------------------------------------------------------------
# 4. Py-spy temporal join
# ---------------------------------------------------------------------------

class TestPyspyTemporalJoin:
    def test_build_pyspy_index(self):
        pyspy_summary = {
            "timed_samples": [
                {"function": "train_step", "ts_ns": 1_000_000, "dur_ns": 10_000_000},
                {"function": "forward", "ts_ns": 11_000_000, "dur_ns": 10_000_000},
            ],
        }
        index = _build_pyspy_temporal_index(pyspy_summary)
        assert len(index) == 2
        assert index[0][2] == "train_step"

    def test_default_duration_when_missing(self):
        pyspy_summary = {
            "timed_samples": [
                {"function": "train_step", "ts_ns": 1_000_000, "dur_ns": 0},
            ],
        }
        index = _build_pyspy_temporal_index(pyspy_summary)
        assert len(index) == 1
        # Default 10ms duration
        assert index[0][1] == 1_000_000 + 10_000_000

    def test_empty_pyspy(self):
        assert _build_pyspy_temporal_index(None) == []
        assert _build_pyspy_temporal_index({}) == []
        assert _build_pyspy_temporal_index({"timed_samples": []}) == []

    def test_kernel_matched_to_pyspy_sample(self):
        """Kernel launch during a py-spy sample is attributed to that function."""
        pyspy_index = [
            (1_000_000, 11_000_000, "train_step"),
            (11_000_000, 21_000_000, "val_step"),
        ]
        timestamps = [(5_000_000, 5_500_000)]
        result = _match_kernel_to_pyspy_temporal(timestamps, pyspy_index)
        assert result == "train_step"

    def test_no_match_outside_samples(self):
        pyspy_index = [(1_000_000, 5_000_000, "train_step")]
        timestamps = [(10_000_000, 10_500_000)]
        result = _match_kernel_to_pyspy_temporal(timestamps, pyspy_index)
        assert result is None

    def test_pyspy_in_ranking(self):
        """Py-spy temporal join provides caller when call chain is unavailable."""
        nsys = {
            "top_kernels": [
                {"name": "volta_sgemm_128x128_nn", "pct": 85.0, "total_ms": 100},
            ],
            "cpu_gpu_correlations": [
                {"api_name": "cudaLaunchKernel",
                 "kernel_name": "volta_sgemm_128x128_nn",
                 "stream_id": 1, "gpu_duration_ns": 100_000_000,
                 "launch_overhead_ns": 10_000,
                 "cpu_start_ns": 5_000_000, "gpu_start_ns": 5_100_000},
            ],
        }
        pyspy = {
            "timed_samples": [
                {"function": "train_step", "ts_ns": 1_000_000, "dur_ns": 20_000_000},
            ],
        }
        ranking = compute_attribution_ranking(nsys, pyspy_summary=pyspy)
        assert ranking[0].caller_function == "train_step"


# ---------------------------------------------------------------------------
# Framework enrichment with temporal NVTX
# ---------------------------------------------------------------------------

class TestEnrichWithTemporalNvtx:
    def test_temporal_nvtx_preferred_over_name(self):
        """Temporal NVTX matching takes priority over name-based matching."""
        edges = [CpuGpuEdge(
            "cudaLaunchKernel", "volta_sgemm_128x128_nn", 1,
            10, 100.0, 10.0, 80.0,
        )]
        nvtx_ranges = [
            {"name": "aten::linear", "duration_ms": 5.0, "pct": 50.0,
             "start_ns": 1_000_000, "end_ns": 10_000_000},
        ]
        correlations = [
            {"kernel_name": "volta_sgemm_128x128_nn",
             "cpu_start_ns": 3_000_000, "gpu_start_ns": 3_100_000},
        ]
        enriched = enrich_with_framework_context(
            edges, nvtx_ranges=nvtx_ranges,
            correlations=correlations, program_type="pytorch",
        )
        # Should use temporal match (aten::linear) not name heuristic (aten::mm)
        assert enriched[0].framework_op == "aten::linear"

    def test_fallback_to_name_heuristic(self):
        """Without timestamps, falls back to name-based heuristics."""
        edges = [CpuGpuEdge(
            "cudaLaunchKernel", "volta_sgemm_128x128_nn", 1,
            10, 100.0, 10.0, 80.0,
        )]
        enriched = enrich_with_framework_context(
            edges, nvtx_ranges=None, program_type="pytorch",
        )
        assert enriched[0].framework_op == "aten::mm"  # from name heuristic


# ---------------------------------------------------------------------------
# Attribution priority and scoring
# ---------------------------------------------------------------------------

class TestAttributionPriority:
    def test_strategy_priority_order(self):
        """Verify strategies are applied in priority: callchain > NVTX > pyspy
        (torch temporal matching removed — torch_summary must not win)."""
        nsys = {
            "top_kernels": [
                {"name": "my_kernel", "pct": 90.0, "total_ms": 200},
            ],
            "cpu_gpu_correlations": [
                {"api_name": "cudaLaunchKernel",
                 "kernel_name": "my_kernel",
                 "stream_id": 1, "gpu_duration_ns": 200_000_000,
                 "launch_overhead_ns": 10_000,
                 "cpu_start_ns": 5_000_000, "gpu_start_ns": 5_100_000,
                 "caller_function": "from_callchain"},
            ],
            "nvtx_ranges": [
                {"name": "nvtx_op", "start_ns": 1_000_000, "end_ns": 20_000_000,
                 "duration_ms": 19.0, "pct": 90.0},
            ],
        }
        torch_summary = {
            "_raw_cpu_ops": [
                {"name": "torch_op", "ts": 1000, "dur": 20000},
            ],
        }
        pyspy = {
            "timed_samples": [
                {"function": "pyspy_func", "ts_ns": 1_000_000, "dur_ns": 20_000_000},
            ],
        }
        ranking = compute_attribution_ranking(
            nsys, torch_summary=torch_summary, pyspy_summary=pyspy,
        )
        # caller_function from call chain (highest priority for caller)
        assert ranking[0].caller_function == "from_callchain"
        # framework_op from temporal NVTX (highest priority for framework op)
        assert ranking[0].framework_op == "nvtx_op"

    def test_attribution_bonus_in_scoring(self):
        """Entries with richer attribution get a scoring bonus."""
        nsys = {
            "top_kernels": [
                {"name": "kernel_a", "pct": 50.0, "total_ms": 50},
                {"name": "kernel_b", "pct": 50.0, "total_ms": 50},
            ],
            "cpu_gpu_correlations": [
                {"api_name": "cudaLaunchKernel",
                 "kernel_name": "kernel_a",
                 "stream_id": 1, "gpu_duration_ns": 50_000_000,
                 "launch_overhead_ns": 10_000,
                 "cpu_start_ns": 5_000_000, "gpu_start_ns": 5_100_000,
                 "caller_function": "train_step"},
                {"api_name": "cudaLaunchKernel",
                 "kernel_name": "kernel_b",
                 "stream_id": 1, "gpu_duration_ns": 50_000_000,
                 "launch_overhead_ns": 10_000,
                 "cpu_start_ns": 15_000_000, "gpu_start_ns": 15_100_000},
            ],
        }
        ranking = compute_attribution_ranking(nsys)
        # kernel_a should rank higher due to attribution bonus (has caller)
        assert ranking[0].name == "kernel_a"
        assert ranking[0].caller_function == "train_step"


# ---------------------------------------------------------------------------
# Py-spy speedscope parsing
# ---------------------------------------------------------------------------

class TestSpeedscopeParsing:
    def test_parse_sampled_profile(self):
        import json
        import tempfile
        from pathlib import Path

        from perflab.profilers.python_pyspy import _parse_speedscope_json

        data = {
            "$schema": "https://www.speedscope.app/file-format-schema.json",
            "shared": {
                "frames": [
                    {"name": "main", "file": "main.py"},
                    {"name": "train_step", "file": "train.py"},
                    {"name": "forward", "file": "model.py"},
                ],
            },
            "profiles": [{
                "type": "sampled",
                "name": "thread 0",
                "unit": "nanoseconds",
                "startValue": 1000000,
                "endValue": 50000000,
                "samples": [
                    [0, 1],     # main → train_step
                    [0, 1, 2],  # main → train_step → forward
                    [0, 1, 2],  # main → train_step → forward
                ],
                "weights": [
                    10000000,  # 10ms
                    10000000,
                    10000000,
                ],
            }],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = _parse_speedscope_json(Path(f.name))

        assert len(result) == 3
        # Leaf frames: train_step, forward, forward
        assert result[0]["function"] == "train_step"
        assert result[1]["function"] == "forward"
        assert result[0]["ts_ns"] == 1000000
        assert result[0]["dur_ns"] == 10000000

    def test_parse_evented_profile(self):
        import json
        import tempfile
        from pathlib import Path

        from perflab.profilers.python_pyspy import _parse_speedscope_json

        data = {
            "shared": {
                "frames": [
                    {"name": "main", "file": "main.py"},
                    {"name": "forward", "file": "model.py"},
                ],
            },
            "profiles": [{
                "type": "evented",
                "name": "thread 0",
                "unit": "milliseconds",
                "startValue": 0,
                "endValue": 100,
                "events": [
                    {"type": "O", "at": 0, "frame": 0},
                    {"type": "O", "at": 10, "frame": 1},
                    {"type": "C", "at": 50, "frame": 1},
                    {"type": "C", "at": 100, "frame": 0},
                ],
            }],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = _parse_speedscope_json(Path(f.name))

        assert len(result) == 2
        forward = [s for s in result if s["function"] == "forward"][0]
        assert forward["ts_ns"] == 10_000_000  # 10ms → ns
        assert forward["dur_ns"] == 40_000_000  # 40ms duration

    def test_parse_empty_file(self):
        from pathlib import Path

        from perflab.profilers.python_pyspy import _parse_speedscope_json
        assert _parse_speedscope_json(Path("/nonexistent/file.json")) == []


# ---------------------------------------------------------------------------
# Torch trace raw CPU op extraction
# ---------------------------------------------------------------------------

class TestTorchTraceRawOps:
    def test_raw_cpu_ops_extracted(self):
        import json
        import tempfile
        from pathlib import Path

        from perflab.profilers.pytorch_profiler import _parse_torch_trace

        trace = {
            "traceEvents": [
                {"ph": "X", "name": "aten::mm", "cat": "cpu_op",
                 "ts": 1000, "dur": 5000, "args": {}},
                {"ph": "X", "name": "aten::relu", "cat": "cpu_op",
                 "ts": 7000, "dur": 200, "args": {}},  # dur=200 > 100 threshold
                {"ph": "X", "name": "aten::tiny", "cat": "cpu_op",
                 "ts": 8000, "dur": 50, "args": {}},   # dur=50 < 100 threshold
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(trace, f)
            f.flush()
            result = _parse_torch_trace(Path(f.name))

        raw_ops = result.get("_raw_cpu_ops", [])
        assert len(raw_ops) == 2  # tiny op filtered out
        names = {op["name"] for op in raw_ops}
        assert "aten::mm" in names
        assert "aten::relu" in names
        assert "aten::tiny" not in names


# ---------------------------------------------------------------------------
# Bisect-based interval search
# ---------------------------------------------------------------------------

class TestFindBestEnclosing:
    def test_finds_most_specific(self):
        """Returns shortest enclosing interval among nested ranges."""
        index = [
            (1000, 50_000_000, "forward"),
            (2_000_000, 8_000_000, "aten::mm"),
        ]
        assert _find_best_enclosing(3_000_000, index) == "aten::mm"

    def test_non_overlapping(self):
        """Finds the correct interval among non-overlapping ranges."""
        index = [
            (1_000_000, 5_000_000, "a"),
            (6_000_000, 10_000_000, "b"),
            (11_000_000, 15_000_000, "c"),
        ]
        assert _find_best_enclosing(7_000_000, index) == "b"

    def test_no_match(self):
        index = [(1000, 5000, "a")]
        assert _find_best_enclosing(10_000, index) is None

    def test_empty_index(self):
        assert _find_best_enclosing(100, []) is None

    def test_boundary_inclusive(self):
        index = [(1000, 5000, "a")]
        assert _find_best_enclosing(1000, index) == "a"
        assert _find_best_enclosing(5000, index) == "a"

    def test_many_intervals_early_exit(self):
        """With many non-overlapping intervals, bisect + early exit is efficient."""
        index = [(i * 1000, i * 1000 + 500, f"f{i}") for i in range(10_000)]
        assert _find_best_enclosing(5000 * 1000 + 100, index) == "f5000"


# ---------------------------------------------------------------------------
# Speedscope → hotspots extraction
# ---------------------------------------------------------------------------

class TestExtractHotspotsFromSpeedscope:
    def test_aggregates_by_function(self):
        from perflab.profilers.python_pyspy import _extract_hotspots_from_speedscope
        samples = [
            {"function": "forward", "file": "model.py", "ts_ns": 0, "dur_ns": 5_000_000},
            {"function": "forward", "file": "model.py", "ts_ns": 5_000_000, "dur_ns": 5_000_000},
            {"function": "loss", "file": "loss.py", "ts_ns": 10_000_000, "dur_ns": 20_000_000},
        ]
        hotspots = _extract_hotspots_from_speedscope(samples)
        assert len(hotspots) == 2
        # loss: 20ms / 30ms total ≈ 66.7%; forward: 10ms / 30ms ≈ 33.3%
        assert hotspots[0]["function"] == "loss"
        assert hotspots[0]["pct"] == 66.7
        assert hotspots[1]["function"] == "forward"
        assert hotspots[1]["location"] == "model.py"

    def test_empty_samples(self):
        from perflab.profilers.python_pyspy import _extract_hotspots_from_speedscope
        assert _extract_hotspots_from_speedscope([]) == []

    def test_respects_top_n(self):
        from perflab.profilers.python_pyspy import _extract_hotspots_from_speedscope
        samples = [
            {"function": f"func_{i}", "file": "f.py", "ts_ns": i * 1000, "dur_ns": 1000}
            for i in range(20)
        ]
        hotspots = _extract_hotspots_from_speedscope(samples, top_n=5)
        assert len(hotspots) == 5


# ---------------------------------------------------------------------------
# Kernel dossier carries temporal attribution fields
# ---------------------------------------------------------------------------

class TestDossierTemporalFields:
    def test_dossier_has_caller_and_framework_op(self):
        """build_kernel_dossiers propagates caller_function and framework_op."""
        attrib = [
            {
                "name": "volta_sgemm_128x128_nn",
                "gpu_pct": 85.0,
                "gpu_time_ms": 100.0,
                "launch_overhead_us": 15.0,
                "caller_function": "train_step",
                "framework_op": "aten::mm",
                "diagnosis": "Kernel consumes 85% of GPU time",
                "suggestions": ["Use Tensor Cores"],
            },
        ]
        dossiers = build_kernel_dossiers(attrib, ncu_summary=None, sass_entries=None)
        assert len(dossiers) == 1
        d = dossiers[0]
        assert d.caller_function == "train_step"
        assert d.framework_op == "aten::mm"
        assert d.name == "volta_sgemm_128x128_nn"
        assert d.gpu_pct == 85.0

    def test_dossier_none_when_no_attribution_fields(self):
        """Dossier gracefully handles missing caller/framework fields."""
        attrib = [
            {
                "name": "my_kernel",
                "gpu_pct": 50.0,
                "gpu_time_ms": 50.0,
                "diagnosis": "50% GPU time",
                "suggestions": [],
            },
        ]
        dossiers = build_kernel_dossiers(attrib, ncu_summary=None, sass_entries=None)
        assert len(dossiers) == 1
        assert dossiers[0].caller_function is None
        assert dossiers[0].framework_op is None
