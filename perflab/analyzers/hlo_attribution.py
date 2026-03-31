"""HLO op attribution engine: host→device call graph for JAX/TPU workloads.

Analogous to gpu_attribution.py for CUDA/NSys, this module builds an
attribution ranking from XLA HLO dumps and JAX profiler trace data.
It identifies which HLO operations dominate device time and maps them
to optimization opportunities.

Works for both TPU and GPU JAX workloads — any backend that produces
XLA HLO dumps.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HloOpEntry:
    """A single HLO operation type with its compute attribution."""
    op: str                     # e.g. "dot", "convolution", "reduce"
    count: int                  # number of instances across all modules
    pct_of_ops: float           # % of total HLO ops
    category: str               # "compute", "memory", "control", "communication"
    estimated_device_pct: float | None = None  # % of device time (from trace)
    diagnosis: str = ""
    suggestions: list[str] = field(default_factory=list)


@dataclass
class HloAttribution:
    """Full HLO attribution result."""
    entries: list[HloOpEntry]
    total_ops: int
    total_modules: int
    host_time_us: float | None = None
    device_time_us: float | None = None
    device_fraction: float | None = None
    dtype_distribution: dict[str, int] = field(default_factory=dict)


# HLO op categories for grouping
_COMPUTE_OPS = {
    "dot", "convolution", "reduce", "reduce-window", "scatter",
    "gather", "select-and-scatter", "fft", "triangular-solve",
    "cholesky", "custom-call",
}
_MEMORY_OPS = {
    "copy", "pad", "reshape", "transpose", "slice", "concatenate",
    "broadcast", "bitcast", "dynamic-slice", "dynamic-update-slice",
    "reverse", "sort",
}
_CONTROL_OPS = {
    "while", "conditional", "call", "map", "fusion",
    "all-reduce", "all-gather", "all-to-all", "collective-permute",
    "infeed", "outfeed",
}
_COMMUNICATION_OPS = {
    "all-reduce", "all-gather", "all-to-all", "collective-permute",
    "infeed", "outfeed",
}

# Rough cost weights for estimating device time from op counts.
# dot/convolution dominate MXU time; memory ops are cheaper.
_OP_COST_WEIGHTS: dict[str, float] = {
    "dot": 10.0,
    "convolution": 10.0,
    "reduce": 3.0,
    "custom-call": 5.0,
    "pad": 1.0,
    "copy": 0.5,
    "reshape": 0.1,
    "transpose": 0.5,
    "broadcast": 0.3,
    "concatenate": 0.5,
    "slice": 0.3,
    "add": 0.5,
    "multiply": 0.5,
    "subtract": 0.5,
    "divide": 0.5,
    "compare": 0.3,
    "select": 0.3,
    "maximum": 0.3,
    "minimum": 0.3,
    "exponential": 1.0,
    "log": 1.0,
    "tanh": 1.0,
    "sqrt": 0.5,
    "negate": 0.3,
    "clamp": 0.3,
    "convert": 0.5,
}


def _categorize_op(op: str) -> str:
    """Categorize an HLO op into compute/memory/control/communication."""
    op_lower = op.lower()
    if op_lower in _COMMUNICATION_OPS:
        return "communication"
    if op_lower in _CONTROL_OPS:
        return "control"
    if op_lower in _MEMORY_OPS:
        return "memory"
    if op_lower in _COMPUTE_OPS:
        return "compute"
    # Elementwise ops (add, multiply, etc.) are light compute
    return "compute"


def _diagnose_op(op: str, count: int, pct: float, total_ops: int) -> tuple[str, list[str]]:
    """Generate diagnosis and suggestions for a dominant HLO op."""
    suggestions: list[str] = []
    diagnosis = ""

    op_lower = op.lower()

    if op_lower == "dot" and pct > 30:
        diagnosis = f"Matrix multiplications dominate ({pct:.0f}% of ops) — this is expected for matmul/attention workloads"
        suggestions = [
            "Ensure bf16 dtype for 2x MXU throughput on TPU",
            "Pad matrix dimensions to multiples of 128 for TPU tile alignment",
            "Use jax.lax.dot_general with preferred_element_type=jnp.bfloat16",
        ]
    elif op_lower == "pad" and pct > 15:
        diagnosis = f"XLA-inserted padding is {pct:.0f}% of ops — wasting compute on alignment"
        suggestions = [
            "Pre-pad inputs to multiples of 128 to avoid XLA padding",
            "Use batch sizes that are powers of 2",
        ]
    elif op_lower == "copy" and pct > 10:
        diagnosis = f"Memory copies are {pct:.0f}% of ops — unnecessary data movement"
        suggestions = [
            "Use in-place operations where possible",
            "Check for redundant reshapes/transposes that trigger copies",
            "Use jax.lax.with_sharding_constraint for explicit placement",
        ]
    elif op_lower in ("all-reduce", "all-gather", "collective-permute") and count > 0:
        diagnosis = f"Collective communication ({op}, {count} calls) — inter-chip overhead"
        suggestions = [
            "Overlap communication with computation using pipelining",
            "Check sharding strategy — reduce unnecessary all-reduces",
        ]
    elif op_lower == "convolution" and pct > 20:
        diagnosis = f"Convolutions are {pct:.0f}% of ops"
        suggestions = [
            "Use bf16 for convolutions on TPU",
            "Check spatial dimensions align to TPU tile boundaries",
        ]
    elif op_lower in ("infeed", "outfeed") and count > 0:
        diagnosis = f"Host↔device data transfer ({op}, {count} calls)"
        suggestions = [
            "Use prefetching to overlap data loading with computation",
            "Batch more data per infeed to amortize transfer overhead",
        ]

    if not diagnosis and pct > 20:
        diagnosis = f"'{op}' accounts for {pct:.0f}% of HLO operations"

    return diagnosis, suggestions


def compute_hlo_attribution(
    jax_summary: dict,
    trace_metrics: dict | None = None,
) -> HloAttribution | None:
    """Compute HLO op attribution from JAX profiler summary data.

    Args:
        jax_summary: The JAX profiler summary dict (contains hlo_ops, hlo_module_count)
        trace_metrics: Optional trace metrics dict (contains host_time_us, device_time_us)

    Returns:
        HloAttribution with ranked ops and diagnostics, or None if no data.
    """
    hlo_ops = jax_summary.get("hlo_ops", [])
    if not hlo_ops:
        return None

    total_ops = sum(op.get("count", 0) for op in hlo_ops)
    if total_ops == 0:
        return None

    total_modules = jax_summary.get("hlo_module_count", 0)

    # Compute weighted cost estimate for device time attribution
    weighted_costs: dict[str, float] = {}
    for op_entry in hlo_ops:
        op_name = op_entry.get("op", "")
        count = op_entry.get("count", 0)
        weight = _OP_COST_WEIGHTS.get(op_name.lower(), 1.0)
        weighted_costs[op_name] = count * weight

    total_weighted = sum(weighted_costs.values()) or 1.0

    entries: list[HloOpEntry] = []
    for op_entry in hlo_ops:
        op_name = op_entry.get("op", "")
        count = op_entry.get("count", 0)
        pct = (count / total_ops * 100) if total_ops > 0 else 0

        estimated_device_pct = (
            weighted_costs.get(op_name, 0) / total_weighted * 100
        )

        category = _categorize_op(op_name)
        diagnosis, suggestions = _diagnose_op(op_name, count, pct, total_ops)

        entries.append(HloOpEntry(
            op=op_name,
            count=count,
            pct_of_ops=round(pct, 1),
            category=category,
            estimated_device_pct=round(estimated_device_pct, 1),
            diagnosis=diagnosis,
            suggestions=suggestions,
        ))

    # Sort by estimated device time contribution
    entries.sort(
        key=lambda e: e.estimated_device_pct or 0,
        reverse=True,
    )

    result = HloAttribution(
        entries=entries,
        total_ops=total_ops,
        total_modules=total_modules,
    )

    # Incorporate trace metrics if available
    if trace_metrics:
        result.host_time_us = trace_metrics.get("host_time_us")
        result.device_time_us = trace_metrics.get("device_time_us")
        result.device_fraction = trace_metrics.get("device_fraction")

    return result
