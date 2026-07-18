"""Tests for GPU-side multi-cache-level diagnosis (L1 → L2 → DRAM)."""
from __future__ import annotations

from perflab.analyzers.bottleneck_analyzer import diagnose_bottlenecks


class TestGpuCacheHierarchy:
    def test_l1_bottleneck_detected(self):
        summaries = {
            "ncu": {
                "dominant_kernel": {
                    "name": "naive_gemm",
                    "l1_hit_rate": 30.0,
                    "l2_hit_rate": 60.0,
                    "memory_throughput_pct": 50.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        l1_diags = [d for d in diags if "L1 cache bottleneck" in d.bottleneck]
        assert len(l1_diags) >= 1
        assert "L1 hit: 30%" in l1_diags[0].bottleneck
        assert "shared memory" in " ".join(l1_diags[0].suggested_actions).lower()

    def test_l2_bottleneck_detected(self):
        summaries = {
            "ncu": {
                "dominant_kernel": {
                    "name": "tiled_gemm",
                    "l1_hit_rate": 85.0,
                    "l2_hit_rate": 35.0,
                    "memory_throughput_pct": 60.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        l2_diags = [d for d in diags if "L2 cache bottleneck" in d.bottleneck]
        assert len(l2_diags) >= 1
        assert "L2 hit: 35%" in l2_diags[0].bottleneck
        assert any("swizzl" in a.lower() or "l2" in a.lower() for a in l2_diags[0].suggested_actions)

    def test_dram_saturated_detected(self):
        summaries = {
            "ncu": {
                "dominant_kernel": {
                    "name": "optimized_gemm",
                    "l1_hit_rate": 92.0,
                    "l2_hit_rate": 78.0,
                    "memory_throughput_pct": 85.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        dram_diags = [d for d in diags if "DRAM bandwidth saturated" in d.bottleneck]
        assert len(dram_diags) >= 1
        assert any("precision" in a.lower() or "fuse" in a.lower() for a in dram_diags[0].suggested_actions)

    def test_healthy_cache_no_finding(self):
        summaries = {
            "ncu": {
                "dominant_kernel": {
                    "name": "good_gemm",
                    "l1_hit_rate": 90.0,
                    "l2_hit_rate": 85.0,
                    "memory_throughput_pct": 30.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        cache_diags = [d for d in diags if "cache bottleneck" in d.bottleneck.lower() or "DRAM" in d.bottleneck]
        assert len(cache_diags) == 0

    def test_missing_data_no_crash(self):
        summaries = {
            "ncu": {
                "dominant_kernel": {
                    "name": "kern",
                    "memory_throughput_pct": 50.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        cache_diags = [d for d in diags if "cache" in d.bottleneck.lower()]
        assert len(cache_diags) == 0  # No L1/L2 data → no cache diagnosis


class TestGpuCacheInDossier:
    def test_l1_bottleneck_in_dossier_prompt(self):
        from perflab.analyzers.gpu_attribution import KernelDossier
        from perflab.optimizers.prompt import PromptContext, build_prompt

        dossier = KernelDossier(
            name="naive_gemm",
            gpu_pct=85.0,
            gpu_time_ms=120.0,
            ncu_metrics={
                "sm_utilization_pct": 50.0,
                "memory_throughput_pct": 60.0,
                "compute_throughput_pct": 30.0,
                "l1_hit_rate": 25.0,
                "l2_hit_rate": 55.0,
            },
        )
        ctx = PromptContext(
            source_files={"k.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["k.cu"],
            program_type="cuda",
            kernel_dossiers=[dossier],
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "L1 bottleneck" in full_text
        assert "25%" in full_text

    def test_healthy_cache_shows_healthy(self):
        from perflab.analyzers.gpu_attribution import KernelDossier
        from perflab.optimizers.prompt import PromptContext, build_prompt

        dossier = KernelDossier(
            name="good_gemm",
            gpu_pct=85.0,
            gpu_time_ms=120.0,
            ncu_metrics={
                "sm_utilization_pct": 80.0,
                "l1_hit_rate": 92.0,
                "l2_hit_rate": 88.0,
                "memory_throughput_pct": 30.0,
            },
        )
        ctx = PromptContext(
            source_files={"k.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 5.0}},
            roofline=None,
            history=[],
            allowed_paths=["k.cu"],
            program_type="cuda",
            kernel_dossiers=[dossier],
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "healthy" in full_text.lower()
