"""Tests for perflab.analyzers.tma."""
from __future__ import annotations

from perflab.analyzers.tma import (
    TMAResult,
    _parse_tma_fallback,
    _parse_tma_output,
    format_tma_summary,
)


class TestTMAResult:
    def test_dominant_bottleneck_backend(self):
        tma = TMAResult(
            frontend_bound_pct=10.0,
            backend_bound_pct=55.0,
            bad_speculation_pct=5.0,
            retiring_pct=30.0,
        )
        assert tma.dominant_bottleneck == "backend_bound"

    def test_dominant_bottleneck_frontend(self):
        tma = TMAResult(
            frontend_bound_pct=50.0,
            backend_bound_pct=20.0,
            bad_speculation_pct=10.0,
            retiring_pct=20.0,
        )
        assert tma.dominant_bottleneck == "frontend_bound"

    def test_dominant_bottleneck_retiring(self):
        tma = TMAResult(
            frontend_bound_pct=5.0,
            backend_bound_pct=10.0,
            bad_speculation_pct=5.0,
            retiring_pct=80.0,
        )
        assert tma.dominant_bottleneck == "retiring"

    def test_to_dict(self):
        tma = TMAResult(
            frontend_bound_pct=23.4,
            backend_bound_pct=45.2,
            bad_speculation_pct=8.1,
            retiring_pct=23.3,
        )
        d = tma.to_dict()
        assert d["frontend_bound_pct"] == 23.4
        assert d["backend_bound_pct"] == 45.2
        assert d["dominant_bottleneck"] == "backend_bound"


class TestParseTmaOutput:
    def test_parse_standard_format(self):
        text = """
 Performance counter stats for 'bench':

    23.4%  frontend bound
    45.2%  backend bound
     8.1%  bad speculation
    23.3%  retiring
"""
        result = _parse_tma_output(text)
        assert result is not None
        assert abs(result.frontend_bound_pct - 23.4) < 0.1
        assert abs(result.backend_bound_pct - 45.2) < 0.1
        assert abs(result.bad_speculation_pct - 8.1) < 0.1
        assert abs(result.retiring_pct - 23.3) < 0.1

    def test_parse_reversed_format(self):
        text = """
frontend bound:   15.2%
backend bound:    60.1%
bad speculation:  4.5%
retiring:         20.2%
"""
        result = _parse_tma_output(text)
        assert result is not None
        assert abs(result.backend_bound_pct - 60.1) < 0.1

    def test_parse_incomplete_returns_none(self):
        text = "23.4%  frontend bound\n45.2%  backend bound\n"
        result = _parse_tma_output(text)
        assert result is None

    def test_parse_empty_returns_none(self):
        result = _parse_tma_output("")
        assert result is None

    def test_parse_with_underscores(self):
        text = """
23.4% frontend_bound
45.2% backend_bound
8.1% bad_speculation
23.3% retiring
"""
        result = _parse_tma_output(text)
        assert result is not None


class TestParseTmaFallback:
    def test_parse_raw_counters(self):
        text = """
 Performance counter stats for 'bench':

        1,000,000      topdown-fetch-bubbles
          200,000      topdown-recovery-bubbles
        3,000,000      topdown-slots-issued
        2,500,000      topdown-slots-retired
       10,000,000      topdown-total-slots
"""
        result = _parse_tma_fallback(text)
        assert result is not None
        assert abs(result.frontend_bound_pct - 10.0) < 0.1  # 1M/10M
        assert abs(result.retiring_pct - 25.0) < 0.1  # 2.5M/10M
        # bad_spec = (3M - 2.5M + 0.2M) / 10M = 7%
        assert abs(result.bad_speculation_pct - 7.0) < 0.1

    def test_empty_returns_none(self):
        result = _parse_tma_fallback("")
        assert result is None

    def test_zero_total_returns_none(self):
        text = "        0      topdown-total-slots\n"
        result = _parse_tma_fallback(text)
        assert result is None


class TestFormatTmaSummary:
    def test_format_output(self):
        tma = TMAResult(
            frontend_bound_pct=15.0,
            backend_bound_pct=50.0,
            bad_speculation_pct=5.0,
            retiring_pct=30.0,
        )
        text = format_tma_summary(tma)
        assert "Frontend Bound: 15.0%" in text
        assert "Backend Bound:  50.0%" in text
        assert "Backend Bound" in text  # dominant bottleneck
