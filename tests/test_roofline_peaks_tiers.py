"""Tests for the GPU peak table's three-tier resolution (table / computed / measured)."""
from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from perflab import roofline_peaks
from perflab.reporting.dashboard_html import GlanceData, write_dashboard_html
from perflab.reporting.report_md import write_report_md


def _gpu_row(**overrides) -> dict:
    row = {
        "name": "NVIDIA A100-SXM4-40GB",
        "compute_cap": "8.0",
        "clocks.max.sm": "1410",
        "memory.clock": "1215",
        "memory.bus_width": "5120",
        "multiprocessor_count": "108",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Tier 1: exact match on the full nvidia-smi name
# ---------------------------------------------------------------------------

class TestTierOneExactMatch:
    def test_exact_match_returns_table_source(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with patch.object(roofline_peaks, "_nvidia_smi_query", return_value=[_gpu_row()]):
            result = roofline_peaks.infer_cuda_peaks()
        assert result is not None
        assert result.source == "table"
        assert result.peak_tflops == 312.0  # bf16 representative value
        assert result.peak_mem_bw_gbs == 1555.0
        assert result.dtype_peaks is not None
        assert result.dtype_peaks["peak_tflops_fp32"] == 19.5

    def test_exact_match_distinguishes_variants(self):
        sxm = roofline_peaks._lookup_dtype_peaks_exact("NVIDIA H100 80GB HBM3")
        pcie = roofline_peaks._lookup_dtype_peaks_exact("NVIDIA H100 PCIe")
        assert sxm is not None and pcie is not None
        assert (
            roofline_peaks._KNOWN_GPU_SPECS[sxm[0]].mem_bw_gbs
            != roofline_peaks._KNOWN_GPU_SPECS[pcie[0]].mem_bw_gbs
        )

    def test_no_exact_match_for_name_only_containing_a_short_key(self):
        # These only *contain* a legacy short key as a substring; they must
        # not exact-match it (that would silently claim tier "table").
        assert roofline_peaks._lookup_dtype_peaks_exact("NVIDIA A100 Ultra Turbo") is None
        assert roofline_peaks._lookup_dtype_peaks_exact("NVIDIA H100 Ultra Turbo") is None

    def test_lookup_dtype_peaks_still_substring_matches(self):
        # agent.py/pipeline.py/prompt.py rely on this for free-text
        # task.target_hardware values like "A100" -- must keep working.
        peaks = roofline_peaks._lookup_dtype_peaks("A100")
        assert peaks is not None
        assert peaks["peak_tflops_fp16"] == 312.0


# ---------------------------------------------------------------------------
# Tier 2: computed bandwidth/TFLOPS from bus_width/clock, for unknown cards
# ---------------------------------------------------------------------------

class TestTierTwoComputed:
    def test_unknown_gpu_uses_computed_tier(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        row = _gpu_row(
            name="NVIDIA RTX 6000 Fictional",
            compute_cap="9.0",
            **{
                "clocks.max.sm": "2000",
                "memory.clock": "10000",
                "memory.bus_width": "384",
                "multiprocessor_count": "160",
            },
        )
        with patch.object(roofline_peaks, "_nvidia_smi_query", return_value=[row]):
            result = roofline_peaks.infer_cuda_peaks()
        assert result is not None
        assert result.source == "computed"
        expected_bw = (384 / 8.0) * (10000 * 1e6) * 2.0 / 1e9
        assert result.peak_mem_bw_gbs == pytest.approx(expected_bw)
        csm = roofline_peaks._cores_per_sm("9.0")
        expected_tflops = (160 * csm * 2.0 * 2.0) / 1000.0
        assert result.peak_tflops == pytest.approx(expected_tflops)
        assert result.dtype_peaks is None  # no table entry at all, not even a substring guess

    def test_computed_gpu_bandwidth_formula(self):
        bw = roofline_peaks._computed_gpu_bandwidth(
            {"memory.bus_width": "5120", "memory.clock": "1215"}
        )
        assert bw == pytest.approx((5120 / 8.0) * (1215 * 1e6) * 2.0 / 1e9)

    def test_computed_gpu_bandwidth_missing_fields(self):
        assert roofline_peaks._computed_gpu_bandwidth({}) is None

    def test_substring_match_does_not_claim_table_source(self, monkeypatch, caplog):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        row = _gpu_row(
            name="NVIDIA H100 Custom Variant XYZ",  # substring "H100" but no exact entry
            compute_cap="9.0",
            **{
                "clocks.max.sm": "1830",
                "memory.clock": "2619",
                "memory.bus_width": "5120",
                "multiprocessor_count": "132",
            },
        )
        with patch.object(roofline_peaks, "_nvidia_smi_query", return_value=[row]):
            with caplog.at_level(logging.WARNING, logger="perflab.roofline_peaks"):
                result = roofline_peaks.infer_cuda_peaks()
        assert result is not None
        assert result.source == "computed"  # never "table" for a mere substring hit
        assert result.dtype_peaks is not None  # still attached for context
        assert any("substring" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Tier 3: measured torch-calibration fallback, cached per GPU name
# ---------------------------------------------------------------------------

class TestTierThreeMeasured:
    def test_measured_tier_used_when_computed_unavailable(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        row = _gpu_row(
            name="NVIDIA Mystery GPU",
            **{"memory.bus_width": "", "memory.clock": "", "clocks.max.sm": ""},
        )
        with patch.object(roofline_peaks, "_nvidia_smi_query", return_value=[row]), \
             patch.object(roofline_peaks, "_measured_cuda_peaks") as mock_measured:
            mock_measured.return_value = roofline_peaks.Peaks(
                123.0, 456.0, "measured", "NVIDIA Mystery GPU",
            )
            result = roofline_peaks.infer_cuda_peaks()
        assert result is not None
        assert result.source == "measured"
        assert result.peak_tflops == 123.0
        assert result.peak_mem_bw_gbs == 456.0
        mock_measured.assert_called_once()

    def test_measured_cuda_peaks_returns_none_without_torch(self):
        # torch is not installed in this environment, so a plain `import torch`
        # already raises ModuleNotFoundError; the ImportError guard should
        # make this a graceful None rather than an uncaught exception.
        with patch.dict(sys.modules, {"torch": None}):
            result = roofline_peaks._measured_cuda_peaks("Some GPU", 0)
        assert result is None

    def test_measured_cuda_peaks_probes_and_caches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(roofline_peaks, "_CACHE_PATH", tmp_path / "peaks.json")
        monkeypatch.delenv("PERFLAB_PEAKS_NO_CACHE", raising=False)
        fake_torch = MagicMock()
        fake_torch.device = MagicMock(side_effect=lambda s: s)

        with patch.dict(sys.modules, {"torch": fake_torch}), \
             patch.object(roofline_peaks, "_gpu_matmul_tflops_probe", return_value=100.0) as mock_tflops, \
             patch.object(roofline_peaks, "_gpu_bandwidth_copy_probe", return_value=500.0) as mock_bw:
            result = roofline_peaks._measured_cuda_peaks("Unknown GPU 9000", 0)
            assert result is not None
            assert result.source == "measured"
            assert result.peak_tflops == 100.0
            assert result.peak_mem_bw_gbs == 500.0
            mock_tflops.assert_called_once()
            mock_bw.assert_called_once()

            cache_file = roofline_peaks.gpu_measured_cache_path("Unknown GPU 9000")
            assert cache_file.exists()

            # Second call should hit the cache rather than re-running probes.
            result2 = roofline_peaks._measured_cuda_peaks("Unknown GPU 9000", 0)
            assert result2 is not None
            assert result2.peak_tflops == 100.0
            mock_tflops.assert_called_once()
            mock_bw.assert_called_once()

    def test_gpu_measured_cache_path_is_per_gpu_name(self, monkeypatch, tmp_path):
        monkeypatch.setattr(roofline_peaks, "_CACHE_PATH", tmp_path / "peaks.json")
        p1 = roofline_peaks.gpu_measured_cache_path("NVIDIA A100-SXM4-40GB")
        p2 = roofline_peaks.gpu_measured_cache_path("NVIDIA H100 PCIe")
        assert p1 != p2
        assert p1.parent == tmp_path
        assert p1.name.startswith("peaks-")


# ---------------------------------------------------------------------------
# Dashboard / report display of the source tier
# ---------------------------------------------------------------------------

class TestSourceTierDisplay:
    def test_dashboard_shows_source_tier(self, tmp_path):
        glance = GlanceData(
            metric_name="tflops",
            baseline_value=1.0,
            best_value=2.0,
            best_iter=1,
            total_iterations=1,
            speedup=2.0,
            accepted_count=1,
            peak_tflops=312.0,
            roofline_source="measured",
            roofline_device="NVIDIA Mystery GPU",
        )
        out = tmp_path / "dashboard.html"
        write_dashboard_html(
            path=out, title="t", metric_png_rel=None, glance=glance,
        )
        html = out.read_text(encoding="utf-8")
        assert "peaks: measured" in html
        assert "NVIDIA Mystery GPU" in html

    def test_report_md_shows_source_tier(self, tmp_path):
        data = {
            "task_name": "t", "run_id": "r1", "metric_name": "tflops",
            "metric_mode": "maximize", "best_value": 2.0, "best_iter": 1,
            "baseline_value": 1.0, "rows": [], "latest_artifacts": {},
            "roofline_peaks": {
                "peak_tflops": 312.0, "peak_mem_bw_gbs": 1555.0,
                "source": "table", "device": "NVIDIA A100-SXM4-40GB",
            },
        }
        out = tmp_path / "report.md"
        write_report_md(out, data)
        text = out.read_text(encoding="utf-8")
        assert "## Roofline" in text
        assert "Source: table" in text
        assert "312.000 TFLOPS" in text
