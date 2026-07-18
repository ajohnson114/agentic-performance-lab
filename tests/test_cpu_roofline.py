"""Tests for CPU roofline spec-based estimation."""
from unittest.mock import MagicMock, patch

from perflab.roofline_peaks import (
    _estimate_cpu_peaks,
    _physical_cpu_count,
    infer_cpu_peaks,
)


def test_estimate_cpu_peaks_returns_peaks_or_none():
    """Should return Peaks or None, never raise."""
    result = _estimate_cpu_peaks()
    # On any machine this should either return valid peaks or None
    if result is not None:
        assert result.peak_tflops > 0
        assert result.peak_mem_bw_gbs > 0
        assert result.source == "cpu-spec"
        assert result.device != ""


def test_estimate_cpu_peaks_apple_silicon():
    """Test Apple Silicon detection path."""
    with patch("perflab.roofline_peaks.platform") as mock_plat, \
         patch("perflab.roofline_peaks._run") as mock_run:
        mock_plat.system.return_value = "Darwin"

        def fake_run(cmd):
            cmd_str = " ".join(cmd)
            if "machdep.cpu.brand_string" in cmd_str:
                return "Apple M2 Pro"
            if "hw.perflevel0.logicalcpu" in cmd_str:
                return "10"
            if "hw.cpufrequency_max" in cmd_str:
                return "3500000000"
            return None

        mock_run.side_effect = fake_run
        result = _estimate_cpu_peaks()
        assert result is not None
        assert "M2 Pro" in result.device
        assert result.peak_tflops > 0
        assert result.peak_mem_bw_gbs == 200.0  # known M2 Pro bandwidth
        assert result.source == "cpu-spec"


def test_estimate_cpu_peaks_linux_avx512():
    """Test Linux with AVX-512 detection."""
    with patch("perflab.roofline_peaks.platform") as mock_plat, \
         patch("perflab.roofline_peaks._run") as mock_run:
        mock_plat.system.return_value = "Linux"

        def fake_run(cmd):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "model name" in cmd_str:
                return "Intel Xeon w9-3495X"
            if "CPU max MHz" in cmd_str:
                return "4800.0"
            if "flags" in cmd_str:
                return "fpu sse sse2 avx avx2 avx512f avx512bw fma"
            if "dmidecode" in cmd_str:
                return None
            return None

        mock_run.side_effect = fake_run

        # Physical cores (56), not logical (112): SMT must not inflate the roof
        with patch("perflab.roofline_peaks._physical_cpu_count", return_value=56):
            result = _estimate_cpu_peaks()
        assert result is not None
        assert result.peak_tflops > 0
        # 56 cores x 32 (avx512 FLOP/cycle) x 4.8 GHz / 1000
        expected_tflops = (56 * 32 * 4.8) / 1000.0
        assert abs(result.peak_tflops - expected_tflops) < 0.01
        assert result.source == "cpu-spec"


def test_estimate_cpu_peaks_linux_falls_back_to_logical():
    """When physical core detection fails, logical count is the last resort."""
    with patch("perflab.roofline_peaks.platform") as mock_plat, \
         patch("perflab.roofline_peaks._run") as mock_run, \
         patch("perflab.roofline_peaks._physical_cpu_count", return_value=None), \
         patch("multiprocessing.cpu_count", return_value=16):
        mock_plat.system.return_value = "Linux"

        def fake_run(cmd):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "model name" in cmd_str:
                return "Some CPU"
            if "CPU max MHz" in cmd_str:
                return "3000.0"
            if "flags" in cmd_str:
                return "fpu sse sse2 avx2 fma"
            return None

        mock_run.side_effect = fake_run
        result = _estimate_cpu_peaks()
        assert result is not None
        expected_tflops = (16 * 16 * 3.0) / 1000.0
        assert abs(result.peak_tflops - expected_tflops) < 0.01


class TestPhysicalCpuCount:
    def test_prefers_psutil_physical_count(self):
        """psutil.cpu_count(logical=False) wins when psutil is importable."""
        mock_psutil = MagicMock()
        mock_psutil.cpu_count.return_value = 8
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            assert _physical_cpu_count() == 8
        mock_psutil.cpu_count.assert_called_with(logical=False)

    def test_psutil_none_falls_through_to_platform(self):
        """psutil returning None (unknown) falls through to platform probes."""
        mock_psutil = MagicMock()
        mock_psutil.cpu_count.return_value = None
        with patch.dict("sys.modules", {"psutil": mock_psutil}), \
             patch("perflab.roofline_peaks.platform") as mock_plat, \
             patch("perflab.roofline_peaks._run") as mock_run:
            mock_plat.system.return_value = "Darwin"
            mock_run.return_value = "10"
            assert _physical_cpu_count() == 10

    def test_macos_sysctl_fallback_without_psutil(self):
        with patch.dict("sys.modules", {"psutil": None}), \
             patch("perflab.roofline_peaks.platform") as mock_plat, \
             patch("perflab.roofline_peaks._run") as mock_run:
            mock_plat.system.return_value = "Darwin"

            def fake_run(cmd):
                if "hw.physicalcpu" in " ".join(cmd):
                    return "12"
                return None

            mock_run.side_effect = fake_run
            assert _physical_cpu_count() == 12

    def test_linux_lscpu_fallback_without_psutil(self):
        with patch.dict("sys.modules", {"psutil": None}), \
             patch("perflab.roofline_peaks.platform") as mock_plat, \
             patch("perflab.roofline_peaks._run") as mock_run:
            mock_plat.system.return_value = "Linux"

            def fake_run(cmd):
                cmd_str = " ".join(cmd)
                if "lscpu" in cmd_str:
                    return "24"
                return None

            mock_run.side_effect = fake_run
            assert _physical_cpu_count() == 24

    def test_returns_none_when_everything_fails(self):
        with patch.dict("sys.modules", {"psutil": None}), \
             patch("perflab.roofline_peaks.platform") as mock_plat, \
             patch("perflab.roofline_peaks._run", return_value=None):
            mock_plat.system.return_value = "Linux"
            assert _physical_cpu_count() is None


def test_infer_cpu_peaks_prefers_spec():
    """infer_cpu_peaks should use spec-based first, then torch fallback."""
    with patch("perflab.roofline_peaks._estimate_cpu_peaks") as mock_spec:
        from perflab.roofline_peaks import Peaks
        mock_spec.return_value = Peaks(1.0, 50.0, "cpu-spec", "Test CPU")
        result = infer_cpu_peaks()
        assert result is not None
        assert result.source == "cpu-spec"
        mock_spec.assert_called_once()


def test_infer_cpu_peaks_falls_back_to_torch():
    """When spec estimation fails, should try torch calibration."""
    with patch("perflab.roofline_peaks._estimate_cpu_peaks", return_value=None), \
         patch("perflab.roofline_peaks.infer_torch_calibration") as mock_torch:
        mock_torch.return_value = None
        result = infer_cpu_peaks()
        assert result is None
        mock_torch.assert_called_once_with(device="cpu")
