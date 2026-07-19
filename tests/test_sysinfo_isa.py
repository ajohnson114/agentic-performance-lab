"""Tests for CPU ISA feature detection and C++ compiler detection in
perflab.tools.sysinfo."""
from __future__ import annotations

import subprocess

from perflab.tools import sysinfo
from perflab.tools.sysinfo import detect_cpp_compiler, detect_cpu_isa_features


class TestDetectCPUISAFeatures:
    def test_returns_valid_structure(self):
        features = detect_cpu_isa_features()
        assert isinstance(features, dict)
        assert "max_simd_width_bits" in features
        assert isinstance(features["max_simd_width_bits"], int)
        # All boolean flags present
        for key in ("sse", "sse2", "avx", "avx2", "avx512f", "fma", "neon"):
            assert key in features
            assert isinstance(features[key], bool)

    def test_max_width_at_least_128_on_modern_cpu(self):
        features = detect_cpu_isa_features()
        # Any modern x86 or ARM CPU should have at least 128-bit SIMD
        assert features["max_simd_width_bits"] >= 128

    def test_neon_on_arm(self):
        import platform
        features = detect_cpu_isa_features()
        if platform.machine() in ("arm64", "aarch64"):
            assert features["neon"] is True
            assert features["max_simd_width_bits"] >= 128

    def test_avx_implies_sse(self):
        features = detect_cpu_isa_features()
        if features["avx2"]:
            # AVX2 implies at least 256-bit SIMD
            assert features["max_simd_width_bits"] >= 256


class TestDetectCppCompiler:
    """Regression tests for the g++ -> c++ fallback.

    Bug: the c++ fallback used to be nested inside the *same* try block as
    the g++ probe, so a missing g++ raised FileNotFoundError straight past
    the fallback (which only ran when g++ existed but exited nonzero). The
    fix gives each candidate its own try/except.
    """

    def test_prefers_gxx_when_available(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            assert cmd[0] == "g++"
            return subprocess.CompletedProcess(cmd, 0, stdout="g++ (GCC) 13.2.0\n", stderr="")

        monkeypatch.setattr(sysinfo.subprocess, "run", fake_run)
        assert detect_cpp_compiler() == "g++ (GCC) 13.2.0"

    def test_falls_back_to_cxx_when_gxx_missing(self, monkeypatch):
        """The bug case: g++ absent entirely (FileNotFoundError)."""
        def fake_run(cmd, **kwargs):
            if cmd[0] == "g++":
                raise FileNotFoundError("g++ not found")
            assert cmd[0] == "c++"
            return subprocess.CompletedProcess(cmd, 0, stdout="Apple clang version 15.0.0\n", stderr="")

        monkeypatch.setattr(sysinfo.subprocess, "run", fake_run)
        assert detect_cpp_compiler() == "Apple clang version 15.0.0"

    def test_falls_back_to_cxx_when_gxx_exits_nonzero(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            if cmd[0] == "g++":
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error")
            return subprocess.CompletedProcess(cmd, 0, stdout="c++ version X\n", stderr="")

        monkeypatch.setattr(sysinfo.subprocess, "run", fake_run)
        assert detect_cpp_compiler() == "c++ version X"

    def test_returns_none_when_neither_available(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise FileNotFoundError(f"{cmd[0]} not found")

        monkeypatch.setattr(sysinfo.subprocess, "run", fake_run)
        assert detect_cpp_compiler() is None

    def test_returns_none_when_both_exit_nonzero(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error")

        monkeypatch.setattr(sysinfo.subprocess, "run", fake_run)
        assert detect_cpp_compiler() is None
