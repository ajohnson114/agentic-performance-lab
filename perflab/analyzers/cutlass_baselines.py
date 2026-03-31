"""CUTLASS-derived baseline tile configurations per GPU architecture.

Provides known-optimal GEMM tile sizes, pipeline stages, and warp arrangements
for each GPU generation, derived from CUTLASS profiling data. These serve as:
1. Starting points for auto-tuning (sweep center)
2. Expert hints in the LLM prompt
3. Validation references ("your tile size is 4x smaller than CUTLASS optimal")

Values are from CUTLASS 3.x default kernel configurations for standard GEMM
operations. Actual optimal values depend on the specific kernel (epilogue,
activation, fusion), but these are strong starting points.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TileConfig:
    """Optimal tile configuration for a specific GPU + dtype combination."""
    tile_m: int
    tile_n: int
    tile_k: int
    stages: int       # software pipeline stages
    warps: int         # warps per CTA
    cluster_m: int = 1  # thread block cluster (Hopper only)
    cluster_n: int = 1


# Known-optimal GEMM configurations derived from CUTLASS profiling.
# Keys: (compute_capability, dtype)
# These represent the configurations CUTLASS found fastest for square GEMMs
# on each GPU generation.
_CUTLASS_CONFIGS: dict[tuple[str, str], TileConfig] = {
    # --- Volta (sm_70) ---
    ("sm_70", "fp32"): TileConfig(tile_m=128, tile_n=128, tile_k=8, stages=2, warps=4),
    ("sm_70", "fp16"): TileConfig(tile_m=128, tile_n=256, tile_k=32, stages=2, warps=8),

    # --- Turing (sm_75) ---
    ("sm_75", "fp32"): TileConfig(tile_m=128, tile_n=128, tile_k=8, stages=2, warps=4),
    ("sm_75", "fp16"): TileConfig(tile_m=128, tile_n=256, tile_k=32, stages=2, warps=8),

    # --- Ampere (sm_80) — A100 ---
    ("sm_80", "fp32"): TileConfig(tile_m=128, tile_n=128, tile_k=32, stages=3, warps=8),
    ("sm_80", "tf32"): TileConfig(tile_m=128, tile_n=128, tile_k=32, stages=3, warps=8),
    ("sm_80", "fp16"): TileConfig(tile_m=128, tile_n=256, tile_k=32, stages=4, warps=8),
    ("sm_80", "bf16"): TileConfig(tile_m=128, tile_n=256, tile_k=32, stages=4, warps=8),

    # --- Ada Lovelace (sm_89) — RTX 4090 ---
    ("sm_89", "fp32"): TileConfig(tile_m=128, tile_n=128, tile_k=32, stages=3, warps=8),
    ("sm_89", "tf32"): TileConfig(tile_m=128, tile_n=128, tile_k=32, stages=3, warps=8),
    ("sm_89", "fp16"): TileConfig(tile_m=128, tile_n=256, tile_k=64, stages=3, warps=8),
    ("sm_89", "bf16"): TileConfig(tile_m=128, tile_n=256, tile_k=64, stages=3, warps=8),

    # --- Hopper (sm_90) — H100 ---
    ("sm_90", "fp32"): TileConfig(tile_m=128, tile_n=256, tile_k=64, stages=4, warps=8),
    ("sm_90", "tf32"): TileConfig(tile_m=128, tile_n=256, tile_k=64, stages=4, warps=8, cluster_m=2, cluster_n=1),
    ("sm_90", "fp16"): TileConfig(tile_m=128, tile_n=256, tile_k=64, stages=5, warps=8, cluster_m=2, cluster_n=1),
    ("sm_90", "bf16"): TileConfig(tile_m=128, tile_n=256, tile_k=64, stages=5, warps=8, cluster_m=2, cluster_n=1),
    ("sm_90", "fp8"):  TileConfig(tile_m=128, tile_n=256, tile_k=128, stages=5, warps=8, cluster_m=2, cluster_n=1),
}

# Map GPU names to compute capabilities
_GPU_TO_SM: dict[str, str] = {
    "V100": "sm_70",
    "RTX 2080": "sm_75",
    "A100": "sm_80",
    "A100-SXM": "sm_80",
    "A10": "sm_80",
    "A30": "sm_80",
    "RTX 3090": "sm_86",
    "RTX 3080": "sm_86",
    "RTX 4090": "sm_89",
    "RTX 4080": "sm_89",
    "L40": "sm_89",
    "L4": "sm_89",
    "H100": "sm_90",
    "H100-SXM": "sm_90",
    "H200": "sm_90",
}


def lookup_cutlass_config(
    device_name: str | None = None,
    compute_capability: str | None = None,
    dtype: str = "fp32",
) -> TileConfig | None:
    """Look up the CUTLASS-optimal tile configuration for a GPU + dtype.

    Args:
        device_name: GPU name (e.g., "NVIDIA H100 SXM"). Fuzzy-matched against known GPUs.
        compute_capability: SM version (e.g., "sm_90"). Takes priority over device_name.
        dtype: Data type (fp32, fp16, bf16, tf32, fp8).

    Returns:
        TileConfig or None if no matching configuration is found.
    """
    sm = compute_capability

    # Resolve device name to SM version
    if sm is None and device_name:
        name_upper = (device_name or "").upper()
        for gpu_key, sm_val in _GPU_TO_SM.items():
            if gpu_key.upper() in name_upper:
                sm = sm_val
                break

    if sm is None:
        return None

    # Normalize dtype
    d = dtype.lower()
    if d in ("float32", "float"):
        d = "fp32"
    elif d in ("float16", "half"):
        d = "fp16"
    elif d in ("bfloat16",):
        d = "bf16"
    elif d in ("tensorfloat32",):
        d = "tf32"

    # Direct lookup
    config = _CUTLASS_CONFIGS.get((sm, d))
    if config:
        return config

    # Fallback: sm_86 (Ampere consumer) uses sm_80 configs
    if sm in ("sm_86", "sm_87"):
        return _CUTLASS_CONFIGS.get(("sm_80", d))

    return None


def generate_sweep_around_baseline(
    baseline: TileConfig,
    *,
    include_half: bool = True,
    include_double: bool = True,
) -> dict[str, list[int]]:
    """Generate a sweep space centered on the CUTLASS baseline.

    Explores 0.5x, 1x, and 2x of each tile dimension. This ensures
    the sweep is centered on the known-optimal configuration rather
    than searching blindly.

    Returns a dict suitable for the tuning.yaml sweep section.
    """
    def _range(val: int) -> list[int]:
        candidates = []
        if include_half:
            half = max(8, val // 2)
            if half not in candidates:
                candidates.append(half)
        candidates.append(val)
        if include_double:
            double = val * 2
            candidates.append(double)
        return sorted(set(candidates))

    sweep: dict[str, list[int]] = {
        "TILE_M": _range(baseline.tile_m),
        "TILE_N": _range(baseline.tile_n),
        "TILE_K": _range(baseline.tile_k),
    }

    # Stages: explore ±1
    stage_vals = sorted(set([
        max(1, baseline.stages - 1),
        baseline.stages,
        baseline.stages + 1,
    ]))
    sweep["NUM_STAGES"] = stage_vals

    return sweep


def format_cutlass_hint(config: TileConfig, dtype: str, gpu_name: str) -> str:
    """Format CUTLASS baseline as a hint for the LLM prompt."""
    lines = [
        f"CUTLASS optimal for {gpu_name} ({dtype}):",
        f"  Tile: {config.tile_m}×{config.tile_n}×{config.tile_k}",
        f"  Pipeline stages: {config.stages}",
        f"  Warps per CTA: {config.warps}",
    ]
    if config.cluster_m > 1 or config.cluster_n > 1:
        lines.append(f"  Thread Block Cluster: {config.cluster_m}×{config.cluster_n} (Hopper)")
    lines.append(
        "  Use these as starting values — fine-tune ±50% based on "
        "your kernel's specific register/shared memory usage."
    )
    return "\n".join(lines)
