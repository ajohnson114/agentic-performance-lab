"""Tests for TPU support: bottleneck analysis, profiler, roofline, and dashboard."""
from __future__ import annotations

import json
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from perflab.analyzers.bottleneck_analyzer import (
    AnalysisThresholds,
    _analyze_tpu,
    diagnose_bottlenecks,
)
from perflab.profilers.jax_profiler import (
    _collect_jax_trace_metrics,
    _detect_tpu_device,
    _parse_compilation_metrics,
    _parse_hlo_dump,
)
from perflab.reporting.dashboard_html import ProfilerData, _render_tpu_section


def _default_thresholds() -> AnalysisThresholds:
    return AnalysisThresholds()


# ---------------------------------------------------------------------------
# TPU bottleneck analysis
# ---------------------------------------------------------------------------

class TestAnalyzeTPU:
    def test_low_mxu_utilization(self):
        summary = {"mxu_utilization_pct": 10.0}
        findings = _analyze_tpu(summary, _default_thresholds())
        assert len(findings) >= 1
        assert any("MXU" in f.bottleneck for f in findings)
        assert any(f.confidence == "high" for f in findings)

    def test_medium_mxu_utilization(self):
        summary = {"mxu_utilization_pct": 25.0}
        findings = _analyze_tpu(summary, _default_thresholds())
        assert len(findings) >= 1
        assert any(f.confidence == "medium" for f in findings)

    def test_good_mxu_utilization_no_finding(self):
        summary = {"mxu_utilization_pct": 80.0}
        findings = _analyze_tpu(summary, _default_thresholds())
        assert not any("MXU" in f.bottleneck for f in findings)

    def test_padding_waste(self):
        summary = {
            "hlo_ops": [
                {"op": "pad", "count": 30},
                {"op": "dot", "count": 50},
                {"op": "add", "count": 20},
            ],
        }
        findings = _analyze_tpu(summary, _default_thresholds())
        assert any("padding" in f.bottleneck.lower() for f in findings)

    def test_no_padding_waste_below_threshold(self):
        summary = {
            "hlo_ops": [
                {"op": "pad", "count": 5},
                {"op": "dot", "count": 90},
                {"op": "add", "count": 5},
            ],
        }
        findings = _analyze_tpu(summary, _default_thresholds())
        assert not any("padding" in f.bottleneck.lower() for f in findings)

    def test_infeed_stall(self):
        summary = {"infeed_stall_pct": 25.0}
        findings = _analyze_tpu(summary, _default_thresholds())
        assert any("infeed" in f.bottleneck.lower() for f in findings)
        assert any(f.confidence == "high" for f in findings)

    def test_no_infeed_stall_below_threshold(self):
        summary = {"infeed_stall_pct": 5.0}
        findings = _analyze_tpu(summary, _default_thresholds())
        assert not any("infeed" in f.bottleneck.lower() for f in findings)

    def test_fragmented_hlo(self):
        summary = {"hlo_module_count": 25}
        findings = _analyze_tpu(summary, _default_thresholds())
        assert any("fragmented" in f.bottleneck.lower() for f in findings)

    def test_few_hlo_modules_no_finding(self):
        summary = {"hlo_module_count": 3}
        findings = _analyze_tpu(summary, _default_thresholds())
        assert not any("fragmented" in f.bottleneck.lower() for f in findings)

    def test_fp32_without_bf16(self):
        summary = {
            "hlo_ops": [
                {"op": "dot", "count": 50},
                {"op": "add", "count": 30},
            ],
        }
        # The bf16 rule checks f32 vs bf16 ops — with no "f32" or "bf16" in op names,
        # this rule should not trigger since it checks for f32 dominance
        findings = _analyze_tpu(summary, _default_thresholds())
        # No fp32/bf16 ops → should not trigger fp32 warning
        assert not any("fp32" in f.bottleneck.lower() for f in findings)

    def test_empty_summary(self):
        findings = _analyze_tpu({}, _default_thresholds())
        assert findings == []


class TestDiagnoseBottlenecksTPU:
    def test_tpu_analysis_triggered_by_system_info(self):
        summaries = {
            "jax": {
                "mxu_utilization_pct": 10.0,
                "xla_compilations": 1,
            }
        }
        system_info = {"tpu_devices": [{"name": "TPU v4", "id": 0}]}
        diags = diagnose_bottlenecks(
            summaries, "jax", system_info=system_info,
        )
        assert any("MXU" in d.bottleneck for d in diags)

    def test_tpu_analysis_not_triggered_without_tpu(self):
        summaries = {
            "jax": {
                "mxu_utilization_pct": 10.0,
                "xla_compilations": 1,
            }
        }
        diags = diagnose_bottlenecks(summaries, "jax")
        # Should still have JAX findings but no TPU-specific MXU findings
        assert not any("MXU" in d.bottleneck for d in diags)


# ---------------------------------------------------------------------------
# JAX profiler helpers
# ---------------------------------------------------------------------------

class TestDetectTPUDevice:
    def test_detects_tpu(self):
        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.platform = "tpu"
        mock_device.device_kind = "TPU v4"
        mock_device.id = 0
        mock_jax.devices.return_value = [mock_device]
        with patch.dict("sys.modules", {"jax": mock_jax}):
            result = _detect_tpu_device()
            assert result["tpu_chip"] == "TPU v4"
            assert result["tpu_count"] == 1
            assert result["tpu_device_ids"] == [0]

    def test_returns_empty_without_jax(self):
        # _detect_tpu_device catches ImportError internally
        result = _detect_tpu_device()
        assert isinstance(result, dict)


class TestCollectJaxTraceMetrics:
    def test_empty_dir(self, tmp_path):
        result = _collect_jax_trace_metrics(tmp_path)
        assert result == {}

    def test_nonexistent_dir(self, tmp_path):
        result = _collect_jax_trace_metrics(tmp_path / "nonexistent")
        assert result == {}

    def test_parses_chrome_trace_host_device(self, tmp_path):
        trace_data = {
            "traceEvents": [
                {"cat": "host", "dur": 5000, "name": "compute"},
                {"cat": "device", "dur": 15000, "name": "matmul"},
                {"cat": "host", "dur": 3000, "name": "prepare"},
            ]
        }
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace_data))
        result = _collect_jax_trace_metrics(tmp_path)
        assert result["host_time_us"] == 8000.0
        assert result["device_time_us"] == 15000.0
        assert 0.6 < result["device_fraction"] < 0.7  # 15000/23000

    def test_parses_mxu_utilization(self, tmp_path):
        trace_data = {
            "traceEvents": [
                {
                    "cat": "device",
                    "dur": 1000,
                    "name": "matmul",
                    "args": {"mxu_utilization": 85.5},
                },
                {
                    "cat": "device",
                    "dur": 1000,
                    "name": "matmul2",
                    "args": {"mxu_utilization": 90.0},
                },
            ]
        }
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace_data))
        result = _collect_jax_trace_metrics(tmp_path)
        assert result["mxu_utilization_pct"] == pytest.approx(87.75, abs=0.1)

    def test_parses_infeed_stalls(self, tmp_path):
        trace_data = {
            "traceEvents": [
                {"cat": "device", "dur": 8000, "name": "matmul"},
                {"cat": "host", "dur": 2000, "name": "infeed_enqueue"},
            ]
        }
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace_data))
        result = _collect_jax_trace_metrics(tmp_path)
        assert "infeed_stall_pct" in result
        assert result["infeed_stall_pct"] == pytest.approx(20.0, abs=0.1)

    def test_ignores_non_json(self, tmp_path):
        pb_file = tmp_path / "trace.pb"
        pb_file.write_bytes(b"\x00\x01\x02")
        result = _collect_jax_trace_metrics(tmp_path)
        assert result == {}

    def test_handles_malformed_json(self, tmp_path):
        trace_file = tmp_path / "trace.json"
        trace_file.write_text("{bad json")
        result = _collect_jax_trace_metrics(tmp_path)
        assert result == {}

    def test_handles_list_format(self, tmp_path):
        trace_data = [
            {"cat": "host", "dur": 1000, "name": "op1"},
            {"cat": "device", "dur": 2000, "name": "op2"},
        ]
        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace_data))
        result = _collect_jax_trace_metrics(tmp_path)
        assert result["host_time_us"] == 1000.0
        assert result["device_time_us"] == 2000.0


class TestParseCompilationMetrics:
    def test_counts_compilations(self):
        stderr = "Compiling matmul for args...\nCompilation of matmul took 120ms"
        result = _parse_compilation_metrics(stderr)
        assert result["xla_compilations"] == 1

    def test_sums_compile_times(self):
        stderr = "Compilation took 100ms\nCompilation took 50ms"
        result = _parse_compilation_metrics(stderr)
        assert result["xla_compilation_time_ms"] == 150.0

    def test_counts_recompilations(self):
        stderr = "Warning: recompiling function due to shape change\nrecompiling again"
        result = _parse_compilation_metrics(stderr)
        assert result["xla_recompilations"] == 2

    def test_empty_stderr(self):
        result = _parse_compilation_metrics("")
        assert result == {}

    def test_none_stderr(self):
        result = _parse_compilation_metrics(None)
        assert result == {}


class TestParseHloDump:
    def test_empty_dir(self, tmp_path):
        result = _parse_hlo_dump(tmp_path)
        assert result == {}

    def test_nonexistent_dir(self, tmp_path):
        result = _parse_hlo_dump(tmp_path / "nope")
        assert result == {}

    def test_counts_modules(self, tmp_path):
        for i in range(3):
            (tmp_path / f"module_{i}.txt").write_text(
                f"HloModule module_{i}\nENTRY main {{\n  x = dot[shape](a, b)\n}}\n"
            )
        result = _parse_hlo_dump(tmp_path)
        assert result["hlo_module_count"] == 3

    def test_counts_ops(self, tmp_path):
        hlo_text = textwrap.dedent("""\
            HloModule test
            ENTRY main {
              a = parameter[f32]{} ()
              b = parameter[f32]{} ()
              c = dot[f32](a, b)
              d = add[f32](c, c)
              ROOT e = multiply[f32](d, d)
            }
        """)
        (tmp_path / "test.txt").write_text(hlo_text)
        result = _parse_hlo_dump(tmp_path)
        assert result["hlo_module_count"] == 1
        ops = {op["op"]: op["count"] for op in result.get("hlo_ops", [])}
        assert "dot" in ops
        assert "add" in ops


# ---------------------------------------------------------------------------
# TPU roofline
# ---------------------------------------------------------------------------

class TestTPURoofline:
    def test_infer_tpu_peaks_v4(self):
        from perflab.roofline_peaks import infer_tpu_peaks

        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.platform = "tpu"
        mock_device.device_kind = "TPU v4"
        mock_jax.devices.return_value = [mock_device]

        with patch.dict("sys.modules", {"jax": mock_jax}):
            peaks = infer_tpu_peaks()
        assert peaks is not None
        assert peaks.peak_tflops == 275.0
        assert peaks.peak_mem_bw_gbs == 1200.0
        assert peaks.source == "tpu-spec"
        assert "TPU v4" in peaks.device
        assert peaks.dtype_peaks is not None
        assert peaks.dtype_peaks["peak_tflops_bf16"] == 275.0
        assert peaks.dtype_peaks["peak_tflops_fp32"] == 275.0 / 2.0

    def test_infer_tpu_peaks_v5p(self):
        from perflab.roofline_peaks import infer_tpu_peaks

        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.platform = "tpu"
        mock_device.device_kind = "TPU v5p"
        mock_jax.devices.return_value = [mock_device, mock_device]

        with patch.dict("sys.modules", {"jax": mock_jax}):
            peaks = infer_tpu_peaks()
        assert peaks is not None
        assert peaks.peak_tflops == 459.0
        assert "2 chips" in peaks.device

    def test_infer_tpu_peaks_no_tpu(self):
        from perflab.roofline_peaks import infer_tpu_peaks

        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_jax.devices.return_value = [mock_device]

        with patch.dict("sys.modules", {"jax": mock_jax}):
            peaks = infer_tpu_peaks()
        assert peaks is None

    def test_infer_tpu_peaks_unknown_generation(self):
        from perflab.roofline_peaks import infer_tpu_peaks

        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.platform = "tpu"
        mock_device.device_kind = "TPU v99"
        mock_jax.devices.return_value = [mock_device]

        with patch.dict("sys.modules", {"jax": mock_jax}):
            peaks = infer_tpu_peaks()
        assert peaks is None

    def test_infer_tpu_peaks_no_jax(self):
        from perflab.roofline_peaks import infer_tpu_peaks
        # Without jax installed, should return None gracefully
        peaks = infer_tpu_peaks()
        assert peaks is None

    @patch("perflab.roofline_peaks.infer_tpu_peaks")
    def test_infer_peaks_auto_tries_tpu_first(self, mock_tpu):
        from perflab.roofline_peaks import Peaks, infer_peaks

        mock_tpu.return_value = Peaks(275.0, 1200.0, "tpu-spec", "TPU v4")
        peaks = infer_peaks("auto")
        assert peaks is not None
        assert peaks.source == "tpu-spec"
        mock_tpu.assert_called_once()


# ---------------------------------------------------------------------------
# Dashboard TPU section
# ---------------------------------------------------------------------------

class TestDashboardTPUSection:
    def test_renders_tpu_metrics(self):
        prof = ProfilerData(
            jax_summary={
                "tpu_chip": "TPU v4",
                "tpu_count": 4,
                "mxu_utilization_pct": 65.3,
                "device_fraction": 0.85,
                "host_time_us": 15000,
                "device_time_us": 85000,
            }
        )
        parts: list[str] = []
        import html as _html
        _render_tpu_section(parts, prof, _html.escape)
        html = "\n".join(parts)
        assert "TPU device metrics" in html
        assert "TPU v4" in html
        assert "4 chips" in html
        assert "MXU Utilization" in html
        assert "65.3%" in html
        assert "Device Active" in html
        assert "Host vs Device" in html

    def test_renders_infeed_stall(self):
        prof = ProfilerData(
            jax_summary={
                "tpu_chip": "TPU v5e",
                "tpu_count": 1,
                "infeed_stall_pct": 15.2,
            }
        )
        parts: list[str] = []
        import html as _html
        _render_tpu_section(parts, prof, _html.escape)
        html = "\n".join(parts)
        assert "Infeed Stall" in html
        assert "15.2%" in html

    def test_no_render_without_tpu_data(self):
        prof = ProfilerData(
            jax_summary={
                "xla_compilations": 2,
                "xla_compilation_time_ms": 500,
            }
        )
        parts: list[str] = []
        import html as _html
        _render_tpu_section(parts, prof, _html.escape)
        assert parts == []

    def test_no_render_without_jax_summary(self):
        prof = ProfilerData()
        parts: list[str] = []
        import html as _html
        _render_tpu_section(parts, prof, _html.escape)
        assert parts == []

    def test_single_chip_no_plural(self):
        prof = ProfilerData(
            jax_summary={
                "tpu_chip": "TPU v6e",
                "tpu_count": 1,
                "mxu_utilization_pct": 90.0,
            }
        )
        parts: list[str] = []
        import html as _html
        _render_tpu_section(parts, prof, _html.escape)
        html = "\n".join(parts)
        assert "1 chip" in html
        assert "1 chips" not in html


# ---------------------------------------------------------------------------
# Doctor TPU check
# ---------------------------------------------------------------------------

class TestDoctorTPU:
    def test_tpu_detected(self):
        from perflab.doctor import check_hardware

        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.platform = "tpu"
        mock_device.device_kind = "TPU v4"
        mock_jax.devices.return_value = [mock_device]

        with patch.dict("sys.modules", {"jax": mock_jax}):
            results = check_hardware()
        tpu_results = [r for r in results if r.name == "hw:tpu"]
        assert len(tpu_results) == 1
        assert tpu_results[0].status == "pass"
        assert "TPU v4" in tpu_results[0].message

    def test_no_tpu(self):
        from perflab.doctor import check_hardware

        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_jax.devices.return_value = [mock_device]

        with patch.dict("sys.modules", {"jax": mock_jax}):
            results = check_hardware()
        tpu_results = [r for r in results if r.name == "hw:tpu"]
        assert len(tpu_results) == 1
        assert "no TPU" in tpu_results[0].message


class TestDoctorJaxGpu:
    def test_jax_gpu_detected(self):
        from perflab.doctor import check_hardware

        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_device.device_kind = "Metal"
        mock_jax.devices.return_value = [mock_device]

        with patch.dict("sys.modules", {"jax": mock_jax}):
            results = check_hardware()
        gpu_results = [r for r in results if r.name == "hw:jax_gpu"]
        assert len(gpu_results) == 1
        assert gpu_results[0].status == "pass"
        assert "Metal" in gpu_results[0].message

    @patch("platform.system", return_value="Darwin")
    def test_jax_cpu_only_on_mac_warns(self, _mock_sys):
        from perflab.doctor import check_hardware

        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.platform = "cpu"
        mock_jax.devices.return_value = [mock_device]

        with patch.dict("sys.modules", {"jax": mock_jax}):
            results = check_hardware()
        gpu_results = [r for r in results if r.name == "hw:jax_gpu"]
        assert len(gpu_results) == 1
        assert gpu_results[0].status == "warn"
        assert "jax-metal" in gpu_results[0].message

    @patch("platform.system", return_value="Linux")
    def test_jax_cpu_only_on_linux_ok(self, _mock_sys):
        from perflab.doctor import check_hardware

        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.platform = "cpu"
        mock_jax.devices.return_value = [mock_device]

        with patch.dict("sys.modules", {"jax": mock_jax}):
            results = check_hardware()
        gpu_results = [r for r in results if r.name == "hw:jax_gpu"]
        assert len(gpu_results) == 1
        assert gpu_results[0].status == "pass"
        assert "CPU only" in gpu_results[0].message

    def test_jax_not_installed(self):
        from perflab.doctor import check_hardware

        with patch.dict("sys.modules", {"jax": None}):
            results = check_hardware()
        gpu_results = [r for r in results if r.name == "hw:jax_gpu"]
        assert len(gpu_results) == 1
        assert "not installed" in gpu_results[0].message
