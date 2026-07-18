"""Tests for perflab.tools.symbols (kernel base names + C++ demangling)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from perflab.tools.symbols import (
    cxxfilt_available,
    demangle,
    kernel_base_name,
    mangled_base_name,
)


class TestKernelBaseName:
    def test_plain_name_unchanged(self):
        assert kernel_base_name("sgemm_naive") == "sgemm_naive"

    def test_strips_namespaces(self):
        assert kernel_base_name("at::native::add_kernel") == "add_kernel"

    def test_strips_template_args(self):
        assert kernel_base_name("gemm_kernel<float, 128>") == "gemm_kernel"

    def test_strips_params(self):
        assert kernel_base_name("matmul(float const*, int)") == "matmul"

    def test_keeps_params_when_disabled(self):
        name = "matmul(float const*, int)"
        assert kernel_base_name(name, strip_params=False) == name

    def test_combined(self):
        assert kernel_base_name("ns::foo<double>(int, int)") == "foo"

    def test_empty_string(self):
        assert kernel_base_name("") == ""


class TestMangledBaseName:
    def test_extracts_identifier(self):
        assert mangled_base_name("_Z12sgemm_naivePfS_S_iii") == "sgemm_naivePfS_S_iii"

    def test_non_mangled_returns_none(self):
        assert mangled_base_name("plain_name") is None

    def test_prefix_without_length_returns_none(self):
        assert mangled_base_name("_Zfoo") is None


class TestDemangle:
    def setup_method(self):
        # Clear caches between tests so mocks take effect
        demangle.cache_clear()
        cxxfilt_available.cache_clear()

    def teardown_method(self):
        demangle.cache_clear()
        cxxfilt_available.cache_clear()

    def test_non_mangled_short_circuits(self):
        with patch("perflab.tools.symbols.shutil.which") as mock_which:
            assert demangle("plain_name") == "plain_name"
            mock_which.assert_not_called()

    def test_missing_cxxfilt_returns_unchanged(self):
        with patch("perflab.tools.symbols.shutil.which", return_value=None):
            assert demangle("_Z6matmulPKfS0_Pfiii") == "_Z6matmulPKfS0_Pfiii"

    def test_missing_cxxfilt_base_name_fallback(self):
        with patch("perflab.tools.symbols.shutil.which", return_value=None):
            result = demangle("_Z6matmulPKfS0_Pfiii", base_name_fallback=True)
            assert result == "matmulPKfS0_Pfiii"

    def test_cxxfilt_demangles(self):
        mock_result = MagicMock()
        mock_result.stdout = "matmul(float const*, int)\n"
        with patch("perflab.tools.symbols.shutil.which", return_value="/usr/bin/c++filt"), \
             patch("perflab.tools.symbols.subprocess.run", return_value=mock_result) as mock_run:
            assert demangle("_Z6matmulPKfi") == "matmul(float const*, int)"
            mock_run.assert_called_once_with(
                ["c++filt", "_Z6matmulPKfi"],
                capture_output=True,
                text=True,
                timeout=5,
            )

    def test_cxxfilt_failure_falls_back_to_base_name(self):
        with patch("perflab.tools.symbols.shutil.which", return_value="/usr/bin/c++filt"), \
             patch("perflab.tools.symbols.subprocess.run", side_effect=OSError("boom")):
            result = demangle("_Z6matmulPKfi", base_name_fallback=True)
            assert result == "matmulPKfi"

    def test_cxxfilt_failure_without_fallback_returns_unchanged(self):
        with patch("perflab.tools.symbols.shutil.which", return_value="/usr/bin/c++filt"), \
             patch("perflab.tools.symbols.subprocess.run", side_effect=OSError("boom")):
            assert demangle("_Z6matmulPKfi") == "_Z6matmulPKfi"

    def test_cxxfilt_empty_output_returns_unchanged(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        with patch("perflab.tools.symbols.shutil.which", return_value="/usr/bin/c++filt"), \
             patch("perflab.tools.symbols.subprocess.run", return_value=mock_result):
            assert demangle("_Z6matmulPKfi") == "_Z6matmulPKfi"

    def test_result_is_cached(self):
        mock_result = MagicMock()
        mock_result.stdout = "matmul(int)\n"
        with patch("perflab.tools.symbols.shutil.which", return_value="/usr/bin/c++filt"), \
             patch("perflab.tools.symbols.subprocess.run", return_value=mock_result) as mock_run:
            assert demangle("_Z6matmuli") == "matmul(int)"
            assert demangle("_Z6matmuli") == "matmul(int)"
            mock_run.assert_called_once()
