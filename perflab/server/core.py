"""Shared FastMCP instance and output helpers for the PerfLab MCP server.

Every tool module registers on the single ``mcp`` instance defined here;
importing ``perflab.server.mcp_server`` pulls all of them in.
"""
from __future__ import annotations

import dataclasses
import json
from typing import TypeVar

from fastmcp import FastMCP

mcp = FastMCP("perflab", instructions="PerfLab agentic profiling & optimization server")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_MAX_OUTPUT_BYTES = 100_000  # ~100 KB

_PROGRAM_TYPES = ("python", "pytorch", "jax", "triton", "cpp", "cuda")


_JSONContainer = TypeVar("_JSONContainer", bound="dict | list")


def _guard_output_size(obj: _JSONContainer) -> _JSONContainer | dict:
    """If the JSON-serialized output exceeds _MAX_OUTPUT_BYTES, truncate it."""
    encoded = json.dumps(obj, default=str)
    if len(encoded.encode("utf-8", errors="replace")) <= _MAX_OUTPUT_BYTES:
        return obj
    truncated = encoded[:_MAX_OUTPUT_BYTES].rsplit(",", 1)[0]
    return {
        "_truncated": True,
        "_notice": (
            f"Output exceeded {_MAX_OUTPUT_BYTES // 1000} KB limit and was truncated. "
            "Use get_run_section for granular access to specific profiler data."
        ),
        "_partial_data": truncated[:50_000] + "...",
        "_original_size_bytes": len(encoded.encode("utf-8", errors="replace")),
    }


def _to_dicts(items: list) -> list[dict]:
    """Serialize a list of dataclasses/namedtuples to list of dicts."""
    result = []
    for item in items:
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            result.append(dataclasses.asdict(item))
        elif hasattr(item, "_asdict"):
            result.append(item._asdict())
        elif isinstance(item, dict):
            result.append(item)
        else:
            result.append({"value": str(item)})
    return result
