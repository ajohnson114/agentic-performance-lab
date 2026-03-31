"""Tests for CPU ISA feature detection in perflab.tools.sysinfo."""
from __future__ import annotations

from perflab.tools.sysinfo import detect_cpu_isa_features


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
