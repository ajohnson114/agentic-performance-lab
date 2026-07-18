"""Tests for hardware mismatch detection in perflab.optimizers.agent."""
from __future__ import annotations

from perflab.optimizers.phases.baseline import _check_hardware_mismatch


class TestCheckHardwareMismatch:
    def test_mismatch_detected(self):
        """H100 target but A100 detected should produce a mismatch message."""
        sysinfo = {"nvidia_gpus": [{"name": "NVIDIA A100-SXM4-80GB"}]}
        result = _check_hardware_mismatch("H100", sysinfo)
        assert result is not None
        assert "mismatch" in result.lower()
        assert "H100" in result
        assert "A100" in result

    def test_match_substring(self):
        """H100 target with H100 SXM detected should match (substring)."""
        sysinfo = {"nvidia_gpus": [{"name": "NVIDIA H100 SXM5 80GB"}]}
        result = _check_hardware_mismatch("H100", sysinfo)
        assert result is None

    def test_match_case_insensitive(self):
        """Case-insensitive matching."""
        sysinfo = {"nvidia_gpus": [{"name": "NVIDIA h100 SXM"}]}
        result = _check_hardware_mismatch("H100", sysinfo)
        assert result is None

    def test_no_target_hardware(self):
        """No target hardware should return None."""
        sysinfo = {"nvidia_gpus": [{"name": "NVIDIA A100"}]}
        result = _check_hardware_mismatch(None, sysinfo)
        assert result is None

    def test_empty_target_hardware(self):
        """Empty string target hardware should return None."""
        sysinfo = {"nvidia_gpus": [{"name": "NVIDIA A100"}]}
        result = _check_hardware_mismatch("", sysinfo)
        assert result is None

    def test_no_gpus_detected(self):
        """No GPUs in system info should return None."""
        sysinfo = {"nvidia_gpus": []}
        result = _check_hardware_mismatch("H100", sysinfo)
        assert result is None

    def test_no_gpu_key(self):
        """Missing nvidia_gpus key should return None."""
        sysinfo = {}
        result = _check_hardware_mismatch("H100", sysinfo)
        assert result is None

    def test_multiple_gpus_one_matches(self):
        """If any GPU matches, no mismatch."""
        sysinfo = {
            "nvidia_gpus": [
                {"name": "NVIDIA A100-SXM4-80GB"},
                {"name": "NVIDIA H100 SXM5 80GB"},
            ]
        }
        result = _check_hardware_mismatch("H100", sysinfo)
        assert result is None

    def test_gpu_name_contains_target(self):
        """GPU name that contains target string should match."""
        sysinfo = {"nvidia_gpus": [{"name": "NVIDIA RTX 4090"}]}
        result = _check_hardware_mismatch("4090", sysinfo)
        assert result is None
