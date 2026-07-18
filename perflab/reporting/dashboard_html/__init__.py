"""HTML dashboard generator for PerfLab runs.

Package split from the former single-module ``dashboard_html.py``; the public
import path is unchanged. Private ``_``-prefixed helpers are re-exported for
tests that exercise individual sections.
"""
from __future__ import annotations

from .accelerator_sections import (
    _render_jax_section,
    _render_metal_section,
    _render_ncu_section,
    _render_nsys_section,
    _render_tpu_section,
)
from .data import AnalysisData, GlanceData, ProfilerData
from .diagnostics import (
    _render_diagnostics,
    _render_environment,
    _render_outcome_analysis,
    _render_user_actions,
)
from .page import write_dashboard_html
from .profiler_sections import (
    _filter_torch_ops,
    _has_jax_data,
    _has_memray_data,
    _has_metal_data,
    _has_ncu_data,
    _has_nsys_data,
    _has_pyspy_data,
    _has_torch_data,
    _render_memray_section,
    _render_profiler,
    _render_pyspy_hotspots,
    _render_pyspy_section,
    _render_torch_section,
)
from .widgets import (
    _fmt_ms,
    _fmt_pct,
    _metric_pill,
    _render_bar_chart,
    _render_speedscope_link,
    _summary_ok,
)

__all__ = [
    "AnalysisData",
    "GlanceData",
    "ProfilerData",
    "write_dashboard_html",
    # Private helpers re-exported for tests
    "_filter_torch_ops",
    "_fmt_ms",
    "_fmt_pct",
    "_has_jax_data",
    "_has_memray_data",
    "_has_metal_data",
    "_has_ncu_data",
    "_has_nsys_data",
    "_has_pyspy_data",
    "_has_torch_data",
    "_metric_pill",
    "_render_bar_chart",
    "_render_diagnostics",
    "_render_environment",
    "_render_jax_section",
    "_render_memray_section",
    "_render_metal_section",
    "_render_ncu_section",
    "_render_nsys_section",
    "_render_outcome_analysis",
    "_render_profiler",
    "_render_pyspy_hotspots",
    "_render_pyspy_section",
    "_render_speedscope_link",
    "_render_torch_section",
    "_render_tpu_section",
    "_render_user_actions",
    "_summary_ok",
]
