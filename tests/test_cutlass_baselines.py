"""Tests for CUTLASS baseline configurations and auto-tuning integration."""
from __future__ import annotations

import pytest

from perflab.analyzers.cutlass_baselines import (
    TileConfig,
    lookup_cutlass_config,
    generate_sweep_around_baseline,
    format_cutlass_hint,
)


# ---------------------------------------------------------------------------
# CUTLASS config lookup
# ---------------------------------------------------------------------------

class TestLookupCutlassConfig:
    def test_h100_fp16(self):
        cfg = lookup_cutlass_config(device_name="NVIDIA H100 SXM", dtype="fp16")
        assert cfg is not None
        assert cfg.tile_m == 128
        assert cfg.tile_n == 256
        assert cfg.stages == 5
        assert cfg.cluster_m == 2  # Hopper clusters

    def test_a100_fp32(self):
        cfg = lookup_cutlass_config(device_name="NVIDIA A100-SXM4-80GB", dtype="fp32")
        assert cfg is not None
        assert cfg.tile_m == 128
        assert cfg.stages == 3

    def test_rtx_4090_fp16(self):
        cfg = lookup_cutlass_config(device_name="NVIDIA GeForce RTX 4090", dtype="fp16")
        assert cfg is not None
        assert cfg.tile_k == 64

    def test_v100(self):
        cfg = lookup_cutlass_config(device_name="Tesla V100-SXM2", dtype="fp16")
        assert cfg is not None

    def test_sm_override(self):
        cfg = lookup_cutlass_config(compute_capability="sm_90", dtype="bf16")
        assert cfg is not None
        assert cfg.tile_m == 128
        assert cfg.stages == 5

    def test_unknown_gpu_returns_none(self):
        assert lookup_cutlass_config(device_name="Unknown GPU XYZ") is None

    def test_empty_returns_none(self):
        assert lookup_cutlass_config() is None

    def test_dtype_normalization(self):
        cfg1 = lookup_cutlass_config(compute_capability="sm_80", dtype="float16")
        cfg2 = lookup_cutlass_config(compute_capability="sm_80", dtype="fp16")
        assert cfg1 is not None
        assert cfg2 is not None
        assert cfg1.tile_m == cfg2.tile_m

    def test_rtx_3090_falls_back_to_sm80(self):
        """RTX 3090 (sm_86) should fall back to sm_80 configs."""
        cfg = lookup_cutlass_config(device_name="NVIDIA GeForce RTX 3090", dtype="fp32")
        assert cfg is not None
        assert cfg.tile_m == 128

    def test_h100_fp8(self):
        cfg = lookup_cutlass_config(device_name="NVIDIA H100", dtype="fp8")
        assert cfg is not None
        assert cfg.tile_k == 128  # FP8 uses larger K tiles


# ---------------------------------------------------------------------------
# Sweep generation around baseline
# ---------------------------------------------------------------------------

class TestSweepGeneration:
    def test_basic_sweep(self):
        baseline = TileConfig(tile_m=128, tile_n=256, tile_k=32, stages=3, warps=8)
        sweep = generate_sweep_around_baseline(baseline)
        assert "TILE_M" in sweep
        assert "TILE_N" in sweep
        assert "TILE_K" in sweep
        assert "NUM_STAGES" in sweep
        # Should include baseline value
        assert 128 in sweep["TILE_M"]
        assert 256 in sweep["TILE_N"]
        assert 32 in sweep["TILE_K"]
        assert 3 in sweep["NUM_STAGES"]

    def test_includes_half_and_double(self):
        baseline = TileConfig(tile_m=128, tile_n=128, tile_k=32, stages=3, warps=8)
        sweep = generate_sweep_around_baseline(baseline)
        assert 64 in sweep["TILE_M"]   # half
        assert 256 in sweep["TILE_M"]  # double

    def test_stages_range(self):
        baseline = TileConfig(tile_m=128, tile_n=128, tile_k=32, stages=3, warps=8)
        sweep = generate_sweep_around_baseline(baseline)
        assert 2 in sweep["NUM_STAGES"]  # stages - 1
        assert 3 in sweep["NUM_STAGES"]  # baseline
        assert 4 in sweep["NUM_STAGES"]  # stages + 1

    def test_no_double(self):
        baseline = TileConfig(tile_m=128, tile_n=128, tile_k=32, stages=3, warps=8)
        sweep = generate_sweep_around_baseline(baseline, include_double=False)
        assert 256 not in sweep["TILE_M"]

    def test_minimum_tile_size(self):
        """Tile sizes should not go below 8."""
        baseline = TileConfig(tile_m=16, tile_n=16, tile_k=8, stages=2, warps=4)
        sweep = generate_sweep_around_baseline(baseline)
        for key in ("TILE_M", "TILE_N", "TILE_K"):
            assert all(v >= 8 for v in sweep[key])


# ---------------------------------------------------------------------------
# Format hint
# ---------------------------------------------------------------------------

class TestFormatHint:
    def test_basic_hint(self):
        cfg = TileConfig(tile_m=128, tile_n=256, tile_k=64, stages=5, warps=8)
        hint = format_cutlass_hint(cfg, "fp16", "H100")
        assert "128×256×64" in hint
        assert "stages: 5" in hint.lower()
        assert "H100" in hint

    def test_hopper_cluster(self):
        cfg = TileConfig(tile_m=128, tile_n=256, tile_k=64, stages=5, warps=8,
                         cluster_m=2, cluster_n=1)
        hint = format_cutlass_hint(cfg, "fp16", "H100")
        assert "Cluster" in hint
        assert "2×1" in hint

    def test_no_cluster(self):
        cfg = TileConfig(tile_m=128, tile_n=128, tile_k=32, stages=3, warps=8)
        hint = format_cutlass_hint(cfg, "fp32", "A100")
        assert "Cluster" not in hint
