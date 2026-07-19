"""Dataclasses describing the payloads rendered into the dashboard."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GlanceData:
    """Data for the at-a-glance section of the dashboard."""
    metric_name: str = ""
    baseline_value: float = 0.0
    best_value: float = 0.0
    best_iter: int = 0
    total_iterations: int = 0
    speedup: float = 1.0
    accepted_count: int = 0
    early_stop_reason: str | None = None
    rows: list[dict] = field(default_factory=list)
    accepted_patches: list[dict] = field(default_factory=list)
    achieved_tflops: float | None = None
    peak_tflops: float | None = None
    achieved_bw_gbs: float | None = None
    peak_mem_bw_gbs: float | None = None
    roofline_source: str | None = None  # e.g. "cpu-spec", "nvidia-smi", "mps-heuristic"
    roofline_device: str | None = None  # e.g. "Apple M4", "NVIDIA A100"
    llm_model: str = ""
    llm_provider: str = ""
    llm_total_calls: int = 0
    llm_total_input_tokens: int = 0
    llm_total_output_tokens: int = 0
    llm_total_latency_s: float = 0.0
    llm_estimated_cost_usd: float | None = None


@dataclass
class ProfilerData:
    """Profiler artifacts to embed in the dashboard."""
    torch_summary: dict | None = None
    pyspy_summary: dict | None = None
    speedscope_json_path: Path | None = None  # pyspy_speedscope.json for interactive viewer
    torch_trace_path: Path | None = None  # torch_trace.json path
    metal_summary: dict | None = None
    nsys_summary: dict | None = None
    ncu_summary: dict | None = None
    roofline_png_path: Path | None = None
    jax_summary: dict | None = None
    memray_summary: dict | None = None
    perfetto_trace_path: Path | None = None
    # Baseline (for comparison)
    baseline_torch_summary: dict | None = None
    baseline_pyspy_summary: dict | None = None
    baseline_speedscope_json_path: Path | None = None
    baseline_metal_summary: dict | None = None
    baseline_nsys_summary: dict | None = None
    baseline_ncu_summary: dict | None = None
    baseline_jax_summary: dict | None = None
    diff_flame_svg_path: Path | None = None


@dataclass
class AnalysisData:
    """Analyzer/diagnostic payloads to render in the dashboard."""
    bottleneck_diagnoses: list[dict] | None = None
    gpu_attribution: list[dict] | None = None
    profile_diff: list[dict] | None = None
    build_flag_recs: list[dict] | None = None
    hotspot_diff: list[dict] | None = None
    history: list[dict] | None = None
    tma_data: dict | None = None
    tma_level2_data: dict | None = None
    power_data: dict | None = None
    vectorization: list[dict] | None = None
    gpu_memory: dict | None = None
    thread_sched: dict | None = None
    ebpf_data: dict | None = None
    lock_contention_data: dict | None = None
    hlo_attribution: list[dict] | None = None
    user_actions: list[dict] | None = None
    microarch_summary: dict | None = None
    torch_flops: dict | None = None
