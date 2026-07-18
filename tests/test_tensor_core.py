"""Tests for Tensor Core support: NCU metrics parsing, bottleneck rules, and task spec."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from perflab.analyzers.bottleneck_analyzer import (
    AnalysisThresholds,
    diagnose_bottlenecks,
)
from perflab.profilers.ncu_profiler import _parse_ncu_csv

# ---------------------------------------------------------------------------
# NCU Tensor Core metric parsing
# ---------------------------------------------------------------------------

class TestNcuTensorCoreMetrics:
    def _write_csv(self, tmp_path: Path, header: str, rows: list[str]) -> Path:
        csv_path = tmp_path / "ncu_metrics.csv"
        content = header + "\n" + "\n".join(rows) + "\n"
        csv_path.write_text(content, encoding="utf-8")
        return csv_path

    def test_tensor_core_utilization_parsed(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),Tensor Active (%),Tensor Throughput (%)"
        rows = [
            "hgemm_wmma_naive,85.0,42.5,38.0",
            "hgemm_wmma_naive,90.0,45.0,40.0",
        ]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert "tensor_core_utilization_pct" in result
        assert result["tensor_core_utilization_pct"] == pytest.approx(43.8, abs=0.1)
        assert "tensor_core_throughput_pct" in result
        assert result["tensor_core_throughput_pct"] == pytest.approx(39.0, abs=0.1)

    def test_tensor_core_per_kernel(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),Tensor Active (%)"
        rows = ["my_kernel,80.0,55.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert result["kernels"][0]["tensor_core_utilization_pct"] == 55.0

    def test_no_tensor_columns_graceful(self, tmp_path):
        header = "Kernel Name,SM Throughput (%)"
        rows = ["sgemm_naive,70.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert "tensor_core_utilization_pct" not in result
        assert result["kernels"][0].get("tensor_core_utilization_pct") is None

    def test_pipe_tensor_column_name(self, tmp_path):
        """ncu sometimes uses 'pipe_tensor_cycles_active' as column name."""
        header = "Kernel Name,SM Throughput (%),pipe_tensor_cycles_active (%)"
        rows = ["wmma_kernel,90.0,60.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert result["kernels"][0]["tensor_core_utilization_pct"] == 60.0

    def test_tensor_utilization_column_variant(self, tmp_path):
        """ncu sometimes uses 'Tensor Utilization' as column name."""
        header = "Kernel Name,SM Throughput (%),Tensor Utilization (%)"
        rows = ["wmma_kernel,90.0,72.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert result["kernels"][0]["tensor_core_utilization_pct"] == 72.0

    def test_tensor_throughput_column_variant(self, tmp_path):
        """ncu sometimes uses 'pipe_tensor_throughput' as column name."""
        header = "Kernel Name,SM Throughput (%),pipe_tensor_throughput (%)"
        rows = ["wmma_kernel,90.0,55.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert result["kernels"][0]["tensor_core_throughput_pct"] == 55.0

    def test_weighted_average_tensor_metrics(self, tmp_path):
        """Tensor Core metrics should be included in weighted-average aggregates."""
        header = "Kernel Name,SM Throughput (%),Tensor Active (%),Tensor Throughput (%)"
        rows = [
            "kernel_a,80.0,50.0,45.0",
            "kernel_a,85.0,55.0,50.0",
            "kernel_b,90.0,60.0,55.0",
        ]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        # kernel_a has 2 invocations, kernel_b has 1
        # weighted avg TC util = (52.5*2 + 60.0*1) / 3 = 55.0
        assert "tensor_core_utilization_pct" in result
        assert result["tensor_core_utilization_pct"] == pytest.approx(55.0, abs=0.1)


# ---------------------------------------------------------------------------
# Bottleneck analyzer Tensor Core rules
# ---------------------------------------------------------------------------

class TestTensorCoreBottleneckRules:
    def test_low_tc_utilization_detected(self):
        summaries = {
            "ncu": {
                "tensor_core_utilization_pct": 5.0,
                "dominant_kernel": {
                    "name": "hgemm_wmma",
                    "tensor_core_utilization_pct": 5.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        tc_diags = [d for d in diags if "tensor core" in d.bottleneck.lower()]
        assert len(tc_diags) >= 1
        assert tc_diags[0].confidence == "high"  # < 10% → high confidence

    def test_moderate_tc_utilization_medium_confidence(self):
        summaries = {
            "ncu": {
                "tensor_core_utilization_pct": 20.0,
                "dominant_kernel": {
                    "name": "hgemm_wmma",
                    "tensor_core_utilization_pct": 20.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        tc_diags = [d for d in diags if "tensor core" in d.bottleneck.lower()]
        assert len(tc_diags) >= 1
        assert tc_diags[0].confidence == "medium"

    def test_good_tc_utilization_no_finding(self):
        summaries = {
            "ncu": {
                "tensor_core_utilization_pct": 65.0,
                "dominant_kernel": {
                    "name": "hgemm_wmma",
                    "tensor_core_utilization_pct": 65.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        tc_diags = [d for d in diags if "tensor core" in d.bottleneck.lower()]
        assert len(tc_diags) == 0

    def test_compute_bound_no_tc_suggests_tensor_cores(self):
        """Compute-bound on CUDA cores with no TC metrics should suggest TC."""
        summaries = {
            "ncu": {
                "dominant_kernel": {
                    "name": "sgemm_naive",
                    "memory_throughput_pct": 20.0,
                    "compute_throughput_pct": 85.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        tc_diags = [d for d in diags if "tensor core" in d.bottleneck.lower()]
        assert len(tc_diags) >= 1
        assert any("cuda core" in d.bottleneck.lower() for d in tc_diags)

    def test_custom_tc_threshold(self):
        thresholds = AnalysisThresholds(ncu_tc_util_low=50.0)
        summaries = {
            "ncu": {
                "tensor_core_utilization_pct": 40.0,
                "dominant_kernel": {
                    "name": "hgemm",
                    "tensor_core_utilization_pct": 40.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda", thresholds=thresholds)
        tc_diags = [d for d in diags if "tensor core" in d.bottleneck.lower()]
        assert len(tc_diags) >= 1

    def test_tc_actions_include_wmma(self):
        summaries = {
            "ncu": {
                "tensor_core_utilization_pct": 5.0,
                "dominant_kernel": {
                    "name": "hgemm",
                    "tensor_core_utilization_pct": 5.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        tc_diags = [d for d in diags if "tensor core" in d.bottleneck.lower()]
        assert len(tc_diags) >= 1
        all_actions = " ".join(tc_diags[0].suggested_actions).lower()
        assert "wmma" in all_actions or "mma" in all_actions


# ---------------------------------------------------------------------------
# Task spec dtype_peaks
# ---------------------------------------------------------------------------

class TestTaskSpecDtypePeaks:
    def test_roofline_dtype_peaks_loaded(self, tmp_path):
        from perflab.task_spec import TaskSpec

        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text(textwrap.dedent("""\
            name: "test_tc"
            workspace: "."
            program_type: "cuda"
            correctness:
              cmd: "echo ok"
            benchmark:
              cmd: "echo bench"
              metric:
                name: "tflops.median"
            roofline:
              peak_tflops: 989.0
              peak_mem_bw_gbs: 3350.0
              peak_fp16_tflops: 1979.0
              dtype_peaks:
                peak_tflops_fp32: 67.0
                peak_tflops_tf32: 989.0
                peak_tflops_fp16: 1979.0
                peak_tflops_bf16: 1979.0
        """), encoding="utf-8")

        spec = TaskSpec.load(task_yaml)
        assert spec.roofline is not None
        assert spec.roofline.dtype_peaks is not None
        assert spec.roofline.dtype_peaks["peak_tflops_fp16"] == 1979.0
        assert spec.roofline.dtype_peaks["peak_tflops_tf32"] == 989.0
        assert spec.roofline.dtype_peaks["peak_tflops_fp32"] == 67.0
        assert spec.roofline.peak_fp16_tflops == 1979.0

    def test_roofline_no_dtype_peaks(self, tmp_path):
        from perflab.task_spec import TaskSpec

        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text(textwrap.dedent("""\
            name: "test_no_dtype"
            workspace: "."
            program_type: "cuda"
            correctness:
              cmd: "echo ok"
            benchmark:
              cmd: "echo bench"
              metric:
                name: "tflops.median"
            roofline:
              peak_tflops: 989.0
              peak_mem_bw_gbs: 3350.0
        """), encoding="utf-8")

        spec = TaskSpec.load(task_yaml)
        assert spec.roofline is not None
        assert spec.roofline.dtype_peaks is None

    def test_ncu_tc_threshold_in_analysis_thresholds(self, tmp_path):
        from perflab.task_spec import TaskSpec

        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text(textwrap.dedent("""\
            name: "test_thresh"
            workspace: "."
            program_type: "cuda"
            correctness:
              cmd: "echo ok"
            benchmark:
              cmd: "echo bench"
              metric:
                name: "tflops.median"
            analysis_thresholds:
              ncu_tc_util_low: 50.0
        """), encoding="utf-8")

        spec = TaskSpec.load(task_yaml)
        assert spec.analysis_thresholds.ncu_tc_util_low == 50.0
