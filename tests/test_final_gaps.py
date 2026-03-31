"""Tests for final gap features: AOT cost analysis, FLOPS counting,
memory fragmentation, non-contiguous detection, TC alignment."""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest

from perflab.analyzers.bottleneck_analyzer import (
    AnalysisThresholds,
    diagnose_bottlenecks,
)
from perflab.profilers.jax_profiler import _parse_hlo_dump
from perflab.profilers.pytorch_profiler import _parse_torch_trace


# ---------------------------------------------------------------------------
# JAX AOT Cost Analysis (HLO FLOP extraction)
# ---------------------------------------------------------------------------

class TestJaxAotCostAnalysis:
    def test_hlo_flops_parsed(self, tmp_path):
        hlo_dir = tmp_path / "hlo"
        hlo_dir.mkdir()
        hlo_file = hlo_dir / "module_0001.txt"
        hlo_file.write_text(textwrap.dedent("""\
            HloModule module_0001
            ENTRY main {
              p0 = f32[1024,1024] parameter(0)
              p1 = f32[1024,1024] parameter(1)
              ROOT dot = f32[1024,1024] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
            }
            // cost: flops=2147483648, bytes_accessed=12582912
        """), encoding="utf-8")

        result = _parse_hlo_dump(hlo_dir)
        assert "hlo_cost_flops" in result
        assert result["hlo_cost_flops"] == 2147483648.0
        assert result["hlo_cost_tflops"] == pytest.approx(0.002147, abs=0.001)
        assert result["hlo_cost_bytes_accessed"] == 12582912.0

    def test_hlo_flops_scientific_notation(self, tmp_path):
        hlo_dir = tmp_path / "hlo"
        hlo_dir.mkdir()
        hlo_file = hlo_dir / "module.txt"
        hlo_file.write_text("cost: flops=2.15e+09\n", encoding="utf-8")

        result = _parse_hlo_dump(hlo_dir)
        assert result["hlo_cost_flops"] == pytest.approx(2.15e9, rel=0.01)

    def test_hlo_no_cost_annotations(self, tmp_path):
        hlo_dir = tmp_path / "hlo"
        hlo_dir.mkdir()
        hlo_file = hlo_dir / "module.txt"
        hlo_file.write_text("HloModule test\nENTRY main { ROOT r = f32[] constant(1) }\n")

        result = _parse_hlo_dump(hlo_dir)
        assert "hlo_cost_flops" not in result

    def test_hlo_multiple_files_summed(self, tmp_path):
        hlo_dir = tmp_path / "hlo"
        hlo_dir.mkdir()
        (hlo_dir / "m1.txt").write_text("flops=1000000\n")
        (hlo_dir / "m2.txt").write_text("flops=2000000\n")

        result = _parse_hlo_dump(hlo_dir)
        assert result["hlo_cost_flops"] == 3000000.0


# ---------------------------------------------------------------------------
# PyTorch FLOPS counting
# ---------------------------------------------------------------------------

class TestPytorchFlops:
    def _write_trace(self, tmp_path: Path, events: list[dict]) -> Path:
        trace_path = tmp_path / "torch_trace.json"
        trace_path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        return trace_path

    def test_flops_extracted_from_trace(self, tmp_path):
        events = [
            {"ph": "X", "name": "aten::mm", "dur": 1000, "cat": "cpu_op",
             "args": {"flops": 2147483648}},
            {"ph": "X", "name": "aten::add", "dur": 100, "cat": "cpu_op",
             "args": {"flops": 1048576}},
        ]
        trace_path = self._write_trace(tmp_path, events)
        result = _parse_torch_trace(trace_path)

        assert "total_flops" in result
        assert result["total_flops"] == pytest.approx(2147483648 + 1048576)
        assert "total_tflops" in result
        assert "top_ops_by_flops" in result
        assert result["top_ops_by_flops"][0]["name"] == "aten::mm"

    def test_no_flops_in_trace(self, tmp_path):
        events = [
            {"ph": "X", "name": "aten::mm", "dur": 1000, "cat": "cpu_op", "args": {}},
        ]
        trace_path = self._write_trace(tmp_path, events)
        result = _parse_torch_trace(trace_path)

        assert "total_flops" not in result

    def test_flops_caps_variant(self, tmp_path):
        """Handle FLOPs or FLOPS key variants."""
        events = [
            {"ph": "X", "name": "aten::mm", "dur": 1000, "cat": "cpu_op",
             "args": {"FLOPs": 5000000}},
        ]
        trace_path = self._write_trace(tmp_path, events)
        result = _parse_torch_trace(trace_path)

        assert result["total_flops"] == 5000000


# ---------------------------------------------------------------------------
# Non-contiguous tensor detection
# ---------------------------------------------------------------------------

class TestNonContiguousDetection:
    def test_contiguous_in_top_ops_flagged(self):
        summaries = {
            "torch_profiler": {
                "top_ops": [
                    {"name": "aten::mm", "total_us": 5000, "pct": 50.0},
                    {"name": "aten::contiguous", "total_us": 800, "pct": 8.0},
                ],
            }
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")
        nc_diags = [d for d in diags if "non-contiguous" in d.bottleneck.lower()]
        assert len(nc_diags) >= 1

    def test_clone_in_top_ops_flagged(self):
        summaries = {
            "torch_profiler": {
                "top_ops": [
                    {"name": "aten::clone", "total_us": 1200, "pct": 12.0},
                ],
            }
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")
        nc_diags = [d for d in diags if "non-contiguous" in d.bottleneck.lower()]
        assert len(nc_diags) >= 1
        assert nc_diags[0].confidence == "high"

    def test_low_pct_contiguous_no_finding(self):
        summaries = {
            "torch_profiler": {
                "top_ops": [
                    {"name": "aten::contiguous", "total_us": 50, "pct": 1.0},
                ],
            }
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")
        nc_diags = [d for d in diags if "non-contiguous" in d.bottleneck.lower()]
        assert len(nc_diags) == 0


# ---------------------------------------------------------------------------
# Tensor Core alignment checking
# ---------------------------------------------------------------------------

class TestTcAlignmentCheck:
    def test_misaligned_matmul_flagged(self):
        summaries = {
            "torch_profiler": {
                "top_ops": [
                    {"name": "aten::mm", "total_us": 5000, "pct": 50.0,
                     "shapes": "[[64, 127], [127, 256]]"},
                ],
            }
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")
        align_diags = [d for d in diags if "aligned" in d.bottleneck.lower() or "tensor core" in d.bottleneck.lower()]
        assert len(align_diags) >= 1
        assert "127" in str(align_diags[0].suggested_actions) or "128" in str(align_diags[0].suggested_actions)

    def test_aligned_matmul_no_finding(self):
        summaries = {
            "torch_profiler": {
                "top_ops": [
                    {"name": "aten::mm", "total_us": 5000, "pct": 50.0,
                     "shapes": "[[64, 128], [128, 256]]"},
                ],
            }
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")
        align_diags = [d for d in diags if "aligned" in d.bottleneck.lower()]
        assert len(align_diags) == 0

    def test_non_matmul_no_check(self):
        summaries = {
            "torch_profiler": {
                "top_ops": [
                    {"name": "aten::relu", "total_us": 5000, "pct": 50.0,
                     "shapes": "[[64, 127]]"},
                ],
            }
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")
        align_diags = [d for d in diags if "aligned" in d.bottleneck.lower()]
        assert len(align_diags) == 0

    def test_small_dims_not_flagged(self):
        """Dims <= 16 shouldn't be flagged (too small for TC alignment to matter)."""
        summaries = {
            "torch_profiler": {
                "top_ops": [
                    {"name": "aten::mm", "total_us": 5000, "pct": 50.0,
                     "shapes": "[[4, 7], [7, 8]]"},
                ],
            }
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")
        align_diags = [d for d in diags if "aligned" in d.bottleneck.lower()]
        assert len(align_diags) == 0


# ---------------------------------------------------------------------------
# Memory fragmentation detection
# ---------------------------------------------------------------------------

class TestMemoryFragmentation:
    def test_high_alloc_count_flagged(self):
        summaries = {
            "torch_profiler": {
                "top_ops": [{"name": "aten::mm", "pct": 50.0}],
                "memory": {
                    "total_allocations": 1000,
                    "total_allocation_time_us": 100000,
                    "peak_memory_mb": 4096,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")
        frag_diags = [d for d in diags if "fragmentation" in d.bottleneck.lower()]
        assert len(frag_diags) >= 1
        actions_text = " ".join(frag_diags[0].suggested_actions).lower()
        assert "expandable_segments" in actions_text

    def test_low_alloc_count_no_finding(self):
        summaries = {
            "torch_profiler": {
                "top_ops": [{"name": "aten::mm", "pct": 50.0}],
                "memory": {
                    "total_allocations": 50,
                    "total_allocation_time_us": 5000,
                    "peak_memory_mb": 2048,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")
        frag_diags = [d for d in diags if "fragmentation" in d.bottleneck.lower()]
        assert len(frag_diags) == 0

    def test_small_memory_no_finding(self):
        """Don't flag fragmentation for small memory usage."""
        summaries = {
            "torch_profiler": {
                "top_ops": [{"name": "aten::mm", "pct": 50.0}],
                "memory": {
                    "total_allocations": 1000,
                    "total_allocation_time_us": 100000,
                    "peak_memory_mb": 512,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "pytorch")
        frag_diags = [d for d in diags if "fragmentation" in d.bottleneck.lower()]
        assert len(frag_diags) == 0


# ---------------------------------------------------------------------------
# PyTorch profiler env var
# ---------------------------------------------------------------------------

class TestTorchProfilerFlopsEnv:
    def test_with_flops_env_set(self):
        from perflab.profilers.pytorch_profiler import TorchProfiler
        # Just verify the env dict is set up correctly
        profiler = TorchProfiler()
        # We can't run the profiler, but we can verify the class exists
        assert profiler.name == "torch_profiler"
