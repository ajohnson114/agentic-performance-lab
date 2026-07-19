"""Tests for advanced NCU metrics: warp stalls, bank conflicts, coalescing, occupancy limiters, instruction mix."""
from __future__ import annotations

from pathlib import Path

from perflab.analyzers.bottleneck_analyzer import (
    AnalysisThresholds,
    diagnose_bottlenecks,
)
from perflab.profilers.ncu_profiler import _ncu_summary_usable, _parse_ncu_csv

# ---------------------------------------------------------------------------
# NCU Warp Stall Reason parsing
# ---------------------------------------------------------------------------

class TestNcuWarpStalls:
    def _write_csv(self, tmp_path: Path, header: str, rows: list[str]) -> Path:
        csv_path = tmp_path / "ncu_metrics.csv"
        content = header + "\n" + "\n".join(rows) + "\n"
        csv_path.write_text(content, encoding="utf-8")
        return csv_path

    def test_stall_reasons_parsed(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),stalled_long_scoreboard (%),stalled_barrier (%),stalled_not_selected (%)"
        rows = ["sgemm_naive,80.0,45.0,15.0,10.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert "warp_stall_reasons" in result
        assert result["warp_stall_reasons"]["long_scoreboard"] == 45.0
        assert result["warp_stall_reasons"]["barrier"] == 15.0
        assert result["dominant_stall_reason"] == "long_scoreboard"
        assert result["dominant_stall_pct"] == 45.0

    def test_stall_reasons_per_kernel(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),stalled_long_scoreboard (%),stalled_math_pipe_throttle (%)"
        rows = ["kern_a,80.0,20.0,50.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        kern = result["kernels"][0]
        assert kern["dominant_stall_reason"] == "math_pipe_throttle"
        assert kern["dominant_stall_pct"] == 50.0

    def test_no_stall_columns_graceful(self, tmp_path):
        header = "Kernel Name,SM Throughput (%)"
        rows = ["sgemm,80.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert "warp_stall_reasons" not in result
        assert "dominant_stall_reason" not in result

    def test_stall_column_variant(self, tmp_path):
        """ncu may use 'stall_long_scoreboard' instead of 'stalled_long_scoreboard'."""
        header = "Kernel Name,SM Throughput (%),stall_long_scoreboard (%)"
        rows = ["kern,80.0,35.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert "warp_stall_reasons" in result
        assert result["warp_stall_reasons"]["long_scoreboard"] == 35.0


# ---------------------------------------------------------------------------
# NCU Bank Conflicts
# ---------------------------------------------------------------------------

class TestNcuBankConflicts:
    def _write_csv(self, tmp_path: Path, header: str, rows: list[str]) -> Path:
        csv_path = tmp_path / "ncu_metrics.csv"
        content = header + "\n" + "\n".join(rows) + "\n"
        csv_path.write_text(content, encoding="utf-8")
        return csv_path

    def test_bank_conflicts_parsed(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),bank_conflicts"
        rows = ["tiled_gemm,85.0,512"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert result["kernels"][0]["bank_conflicts"] == 512
        assert result["bank_conflicts"] == 512

    def test_no_bank_conflict_column(self, tmp_path):
        header = "Kernel Name,SM Throughput (%)"
        rows = ["kern,80.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert "bank_conflicts" not in result


# ---------------------------------------------------------------------------
# NCU Memory Coalescing (sectors per request)
# ---------------------------------------------------------------------------

class TestNcuCoalescing:
    def _write_csv(self, tmp_path: Path, header: str, rows: list[str]) -> Path:
        csv_path = tmp_path / "ncu_metrics.csv"
        content = header + "\n" + "\n".join(rows) + "\n"
        csv_path.write_text(content, encoding="utf-8")
        return csv_path

    def test_sectors_per_request_parsed(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),average_sectors_per_request"
        rows = ["kern,80.0,8.5"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert result["kernels"][0]["sectors_per_request"] == 8.5
        assert result["sectors_per_request"] == 8.5

    def test_sectors_per_request_variant(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),sectors_per_request"
        rows = ["kern,80.0,2.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert result["kernels"][0]["sectors_per_request"] == 2.0

    def test_coalesced_access(self, tmp_path):
        """Sectors per request close to 1.0 means well-coalesced."""
        header = "Kernel Name,SM Throughput (%),sectors_per_request"
        rows = ["kern,80.0,1.2"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert result["kernels"][0]["sectors_per_request"] == 1.2


# ---------------------------------------------------------------------------
# NCU Occupancy Limiters
# ---------------------------------------------------------------------------

class TestNcuOccupancyLimiters:
    def _write_csv(self, tmp_path: Path, header: str, rows: list[str]) -> Path:
        csv_path = tmp_path / "ncu_metrics.csv"
        content = header + "\n" + "\n".join(rows) + "\n"
        csv_path.write_text(content, encoding="utf-8")
        return csv_path

    def test_occupancy_limiters_parsed(self, tmp_path):
        header = (
            "Kernel Name,SM Throughput (%),"
            "occupancy_limit_registers (%),occupancy_limit_shared_mem (%),"
            "occupancy_limit_block (%),theoretical_occupancy (%)"
        )
        rows = ["kern,80.0,50.0,75.0,100.0,50.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        kern = result["kernels"][0]
        assert kern["occupancy_limit_registers_pct"] == 50.0
        assert kern["occupancy_limit_shared_mem_pct"] == 75.0
        assert kern["occupancy_limit_block_pct"] == 100.0
        assert kern["theoretical_occupancy_pct"] == 50.0
        # Should be propagated to result level
        assert result["occupancy_limit_registers_pct"] == 50.0
        assert result["theoretical_occupancy_pct"] == 50.0

    def test_no_limiter_columns(self, tmp_path):
        header = "Kernel Name,SM Throughput (%)"
        rows = ["kern,80.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert "occupancy_limit_registers_pct" not in result


# ---------------------------------------------------------------------------
# NCU Instruction Mix
# ---------------------------------------------------------------------------

class TestNcuInstructionMix:
    def _write_csv(self, tmp_path: Path, header: str, rows: list[str]) -> Path:
        csv_path = tmp_path / "ncu_metrics.csv"
        content = header + "\n" + "\n".join(rows) + "\n"
        csv_path.write_text(content, encoding="utf-8")
        return csv_path

    def test_instruction_mix_parsed(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),pipe_fma_active (%),pipe_fp64 (%),pipe_alu (%),pipe_xu (%)"
        rows = ["kern,80.0,60.0,5.0,20.0,3.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        mix = result["kernels"][0]["instruction_mix"]
        assert mix["fp32_fma"] == 60.0
        assert mix["fp64"] == 5.0
        assert mix["int_alu"] == 20.0
        assert mix["sfu"] == 3.0
        assert result["instruction_mix"] == mix

    def test_partial_instruction_mix(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),pipe_fma_active (%)"
        rows = ["kern,80.0,70.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        mix = result["kernels"][0]["instruction_mix"]
        assert mix["fp32_fma"] == 70.0
        assert "fp64" not in mix

    def test_no_instruction_mix(self, tmp_path):
        header = "Kernel Name,SM Throughput (%)"
        rows = ["kern,80.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert "instruction_mix" not in result.get("kernels", [{}])[0]


# ---------------------------------------------------------------------------
# _ncu_summary_usable predicate (gates the live-run fallback in run())
# ---------------------------------------------------------------------------

class TestNcuSummaryUsable:
    def test_empty_summary_is_unusable(self):
        assert _ncu_summary_usable({}) is False

    def test_no_kernels_is_unusable(self):
        assert _ncu_summary_usable({"kernel_count": 3}) is False

    def test_name_and_invocations_only_is_unusable(self):
        # An unrecognized CSV layout parses into rows but extracts no metric.
        summary = {"kernels": [{"name": "(unknown)", "invocations": 1}]}
        assert _ncu_summary_usable(summary) is False

    def test_kernel_with_a_metric_is_usable(self):
        summary = {"kernels": [{"name": "k1", "invocations": 1, "sm_utilization_pct": 50.0}]}
        assert _ncu_summary_usable(summary) is True

    def test_usable_if_any_kernel_has_a_metric(self):
        summary = {"kernels": [
            {"name": "bare", "invocations": 2},
            {"name": "rich", "invocations": 1, "memory_throughput_pct": 30.0},
        ]}
        assert _ncu_summary_usable(summary) is True


# ---------------------------------------------------------------------------
# Bottleneck rules for new NCU metrics
# ---------------------------------------------------------------------------

class TestWarpStallBottleneck:
    def test_dominant_stall_detected(self):
        summaries = {
            "ncu": {
                "dominant_stall_reason": "long_scoreboard",
                "dominant_stall_pct": 55.0,
                "dominant_kernel": {"name": "sgemm"},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        stall_diags = [d for d in diags if "warp stall" in d.bottleneck.lower()]
        assert len(stall_diags) >= 1
        assert "long_scoreboard" in stall_diags[0].bottleneck
        assert stall_diags[0].confidence == "high"

    def test_moderate_stall_medium_confidence(self):
        summaries = {
            "ncu": {
                "dominant_stall_reason": "barrier",
                "dominant_stall_pct": 35.0,
                "dominant_kernel": {"name": "kern"},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        stall_diags = [d for d in diags if "warp stall" in d.bottleneck.lower()]
        assert len(stall_diags) >= 1
        assert stall_diags[0].confidence == "medium"

    def test_low_stall_no_finding(self):
        summaries = {
            "ncu": {
                "dominant_stall_reason": "not_selected",
                "dominant_stall_pct": 15.0,
                "dominant_kernel": {"name": "kern"},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        stall_diags = [d for d in diags if "warp stall" in d.bottleneck.lower()]
        assert len(stall_diags) == 0

    def test_stall_actions_include_relevant_suggestions(self):
        for reason, keyword in [
            ("long_scoreboard", "global memory"),
            ("barrier", "syncthreads"),
            ("math_pipe_throttle", "tensor core"),
            ("short_scoreboard", "bank conflict"),
        ]:
            summaries = {
                "ncu": {
                    "dominant_stall_reason": reason,
                    "dominant_stall_pct": 50.0,
                    "dominant_kernel": {"name": "kern"},
                }
            }
            diags = diagnose_bottlenecks(summaries, "cuda")
            stall_diags = [d for d in diags if "warp stall" in d.bottleneck.lower()]
            assert len(stall_diags) >= 1
            all_text = (stall_diags[0].root_cause + " " + " ".join(stall_diags[0].suggested_actions)).lower()
            assert keyword in all_text, f"Expected '{keyword}' in actions for stall '{reason}'"


class TestBankConflictBottleneck:
    def test_high_bank_conflicts_detected(self):
        summaries = {
            "ncu": {
                "bank_conflicts": 5000,
                "dominant_kernel": {"name": "tiled_gemm", "bank_conflicts": 5000},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        bc_diags = [d for d in diags if "bank conflict" in d.bottleneck.lower()]
        assert len(bc_diags) >= 1
        assert bc_diags[0].confidence == "high"

    def test_moderate_bank_conflicts(self):
        summaries = {
            "ncu": {
                "bank_conflicts": 200,
                "dominant_kernel": {"name": "kern", "bank_conflicts": 200},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        bc_diags = [d for d in diags if "bank conflict" in d.bottleneck.lower()]
        assert len(bc_diags) >= 1
        assert bc_diags[0].confidence == "medium"

    def test_low_bank_conflicts_no_finding(self):
        summaries = {
            "ncu": {
                "bank_conflicts": 50,
                "dominant_kernel": {"name": "kern", "bank_conflicts": 50},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        bc_diags = [d for d in diags if "bank conflict" in d.bottleneck.lower()]
        assert len(bc_diags) == 0


class TestUncoalescedAccessBottleneck:
    def test_high_sectors_per_request(self):
        summaries = {
            "ncu": {
                "sectors_per_request": 12.0,
                "dominant_kernel": {"name": "naive_kern", "sectors_per_request": 12.0},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        coal_diags = [d for d in diags if "uncoalesced" in d.bottleneck.lower()]
        assert len(coal_diags) >= 1
        assert coal_diags[0].confidence == "high"

    def test_moderate_sectors_per_request(self):
        summaries = {
            "ncu": {
                "sectors_per_request": 5.0,
                "dominant_kernel": {"name": "kern", "sectors_per_request": 5.0},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        coal_diags = [d for d in diags if "uncoalesced" in d.bottleneck.lower()]
        assert len(coal_diags) >= 1
        assert coal_diags[0].confidence == "medium"

    def test_coalesced_no_finding(self):
        summaries = {
            "ncu": {
                "sectors_per_request": 1.5,
                "dominant_kernel": {"name": "kern", "sectors_per_request": 1.5},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        coal_diags = [d for d in diags if "uncoalesced" in d.bottleneck.lower()]
        assert len(coal_diags) == 0


class TestOccupancyLimiterBottleneck:
    def test_register_limited_occupancy(self):
        summaries = {
            "ncu": {
                "achieved_occupancy_pct": 30.0,
                "dominant_kernel": {
                    "name": "kern",
                    "occupancy_limit_registers_pct": 35.0,
                    "occupancy_limit_shared_mem_pct": 100.0,
                    "occupancy_limit_block_pct": 100.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        occ_diags = [d for d in diags if "limited by registers" in d.bottleneck.lower()]
        assert len(occ_diags) >= 1
        assert "__launch_bounds__" in " ".join(occ_diags[0].suggested_actions)

    def test_shared_mem_limited_occupancy(self):
        summaries = {
            "ncu": {
                "achieved_occupancy_pct": 25.0,
                "dominant_kernel": {
                    "name": "kern",
                    "occupancy_limit_registers_pct": 100.0,
                    "occupancy_limit_shared_mem_pct": 25.0,
                    "occupancy_limit_block_pct": 100.0,
                },
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        occ_diags = [d for d in diags if "limited by shared memory" in d.bottleneck.lower()]
        assert len(occ_diags) >= 1

    def test_no_limiter_data_no_extra_finding(self):
        """Low occupancy without limiter data should not produce limiter diagnosis."""
        summaries = {
            "ncu": {
                "achieved_occupancy_pct": 30.0,
                "dominant_kernel": {"name": "kern"},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        limiter_diags = [d for d in diags if "limited by" in d.bottleneck.lower()]
        assert len(limiter_diags) == 0


class TestFP64Bottleneck:
    def test_fp64_usage_detected(self):
        summaries = {
            "ncu": {
                "instruction_mix": {"fp64": 40.0, "fp32_fma": 10.0},
                "dominant_kernel": {"name": "double_kern"},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        fp64_diags = [d for d in diags if "fp64" in d.bottleneck.lower()]
        assert len(fp64_diags) >= 1
        assert fp64_diags[0].confidence == "high"

    def test_moderate_fp64(self):
        summaries = {
            "ncu": {
                "instruction_mix": {"fp64": 15.0, "fp32_fma": 50.0},
                "dominant_kernel": {"name": "kern"},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        fp64_diags = [d for d in diags if "fp64" in d.bottleneck.lower()]
        assert len(fp64_diags) >= 1
        assert fp64_diags[0].confidence == "medium"

    def test_no_fp64_no_finding(self):
        summaries = {
            "ncu": {
                "instruction_mix": {"fp32_fma": 60.0},
                "dominant_kernel": {"name": "kern"},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        fp64_diags = [d for d in diags if "fp64" in d.bottleneck.lower()]
        assert len(fp64_diags) == 0

    def test_low_fp64_no_finding(self):
        summaries = {
            "ncu": {
                "instruction_mix": {"fp64": 5.0, "fp32_fma": 60.0},
                "dominant_kernel": {"name": "kern"},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        fp64_diags = [d for d in diags if "fp64" in d.bottleneck.lower()]
        assert len(fp64_diags) == 0


# ---------------------------------------------------------------------------
# Threshold configuration tests
# ---------------------------------------------------------------------------

class TestNewThresholds:
    def test_bank_conflict_threshold(self):
        t = AnalysisThresholds(ncu_bank_conflicts_high=500.0)
        assert t.ncu_bank_conflicts_high == 500.0

    def test_sectors_per_request_threshold(self):
        t = AnalysisThresholds(ncu_sectors_per_request_high=2.0)
        assert t.ncu_sectors_per_request_high == 2.0

    def test_stall_pct_threshold(self):
        t = AnalysisThresholds(ncu_stall_pct_high=20.0)
        assert t.ncu_stall_pct_high == 20.0

    def test_custom_thresholds_affect_rules(self):
        # With low threshold, even small stalls should be flagged
        thresholds = AnalysisThresholds(ncu_stall_pct_high=10.0)
        summaries = {
            "ncu": {
                "dominant_stall_reason": "long_scoreboard",
                "dominant_stall_pct": 15.0,
                "dominant_kernel": {"name": "kern"},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda", thresholds=thresholds)
        stall_diags = [d for d in diags if "warp stall" in d.bottleneck.lower()]
        assert len(stall_diags) >= 1


# ---------------------------------------------------------------------------
# TMA pipe utilization parsing
# ---------------------------------------------------------------------------

class TestNcuTmaPipe:
    def _write_csv(self, tmp_path: Path, header: str, rows: list[str]) -> Path:
        csv_path = tmp_path / "ncu_metrics.csv"
        content = header + "\n" + "\n".join(rows) + "\n"
        csv_path.write_text(content, encoding="utf-8")
        return csv_path

    def test_tma_pipe_parsed(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),pipe_tma_active (%)"
        rows = ["hopper_kern,90.0,65.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert result["kernels"][0]["tma_pipe_utilization_pct"] == 65.0
        assert result["tma_pipe_utilization_pct"] == 65.0

    def test_no_tma_column(self, tmp_path):
        header = "Kernel Name,SM Throughput (%)"
        rows = ["kern,80.0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert "tma_pipe_utilization_pct" not in result


# ---------------------------------------------------------------------------
# Register spill detection
# ---------------------------------------------------------------------------

class TestNcuRegisterSpills:
    def _write_csv(self, tmp_path: Path, header: str, rows: list[str]) -> Path:
        csv_path = tmp_path / "ncu_metrics.csv"
        content = header + "\n" + "\n".join(rows) + "\n"
        csv_path.write_text(content, encoding="utf-8")
        return csv_path

    def test_local_memory_parsed(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),local_memory_bytes"
        rows = ["kern,80.0,4096"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert result["kernels"][0]["local_memory_bytes"] == 4096
        assert result["local_memory_bytes"] == 4096

    def test_spill_bytes_column(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),spill_bytes"
        rows = ["kern,80.0,2048"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert result["kernels"][0]["local_memory_bytes"] == 2048

    def test_no_spill(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),local_memory_bytes"
        rows = ["kern,80.0,0"]
        csv_path = self._write_csv(tmp_path, header, rows)
        result = _parse_ncu_csv(csv_path)

        assert "local_memory_bytes" not in result.get("kernels", [{}])[0]


class TestRegisterSpillBottleneck:
    def test_spill_detected(self):
        summaries = {
            "ncu": {
                "local_memory_bytes": 2048,
                "dominant_kernel": {"name": "kern", "local_memory_bytes": 2048},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        spill_diags = [d for d in diags if "register spill" in d.bottleneck.lower()]
        assert len(spill_diags) >= 1
        assert spill_diags[0].confidence == "high"

    def test_small_spill_medium_confidence(self):
        summaries = {
            "ncu": {
                "local_memory_bytes": 512,
                "dominant_kernel": {"name": "kern", "local_memory_bytes": 512},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        spill_diags = [d for d in diags if "register spill" in d.bottleneck.lower()]
        assert len(spill_diags) >= 1
        assert spill_diags[0].confidence == "medium"

    def test_no_spill_no_finding(self):
        summaries = {
            "ncu": {
                "dominant_kernel": {"name": "kern"},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        spill_diags = [d for d in diags if "register spill" in d.bottleneck.lower()]
        assert len(spill_diags) == 0


# ---------------------------------------------------------------------------
# Stall GMMA (Hopper warp stall)
# ---------------------------------------------------------------------------

class TestStallGmma:
    def test_gmma_stall_diagnosed(self):
        summaries = {
            "ncu": {
                "dominant_stall_reason": "gmma",
                "dominant_stall_pct": 40.0,
                "dominant_kernel": {"name": "wgmma_kern"},
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        stall_diags = [d for d in diags if "warp stall" in d.bottleneck.lower()]
        assert len(stall_diags) >= 1
        assert "gmma" in stall_diags[0].bottleneck
        all_text = " ".join(stall_diags[0].suggested_actions).lower()
        assert "wgmma" in all_text or "hopper" in all_text

    def test_gmma_stall_parsed_from_csv(self, tmp_path):
        header = "Kernel Name,SM Throughput (%),stalled_gmma (%)"
        rows = ["wgmma_kern,90.0,42.0"]
        csv_path = tmp_path / "ncu_metrics.csv"
        csv_path.write_text(header + "\n" + rows[0] + "\n", encoding="utf-8")
        result = _parse_ncu_csv(csv_path)

        assert result["dominant_stall_reason"] == "gmma"
        assert result["dominant_stall_pct"] == 42.0


# ---------------------------------------------------------------------------
# cudaMalloc hot-path detection
# ---------------------------------------------------------------------------

class TestCudaMallocBottleneck:
    def test_malloc_in_hot_path_detected(self):
        summaries = {
            "nsys": {
                "top_api_calls": [
                    {"name": "cudaMalloc", "pct": 25.0},
                    {"name": "cudaLaunchKernel", "pct": 60.0},
                ],
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        malloc_diags = [d for d in diags if "allocation" in d.bottleneck.lower()]
        assert len(malloc_diags) >= 1
        assert malloc_diags[0].confidence == "high"

    def test_cuda_free_detected(self):
        summaries = {
            "nsys": {
                "top_api_calls": [
                    {"name": "cudaFree", "pct": 8.0},
                ],
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        malloc_diags = [d for d in diags if "allocation" in d.bottleneck.lower()]
        assert len(malloc_diags) >= 1

    def test_low_malloc_no_finding(self):
        summaries = {
            "nsys": {
                "top_api_calls": [
                    {"name": "cudaMalloc", "pct": 2.0},
                ],
            }
        }
        diags = diagnose_bottlenecks(summaries, "cuda")
        malloc_diags = [d for d in diags if "allocation" in d.bottleneck.lower()]
        assert len(malloc_diags) == 0


# ---------------------------------------------------------------------------
# Optimization detection tests
# ---------------------------------------------------------------------------

class TestOptimizationDetection:
    def test_flex_attention_detected(self):
        from perflab.optimizers.prompt import _detect_existing_optimizations
        files = {"model.py": "output = flex_attention(q, k, v)"}
        result = _detect_existing_optimizations(files, "pytorch")
        assert any("FlexAttention" in r for r in result)

    def test_torchao_detected(self):
        from perflab.optimizers.prompt import _detect_existing_optimizations
        files = {"model.py": "from torchao.quantization import quantize_"}
        result = _detect_existing_optimizations(files, "pytorch")
        assert any("torchao" in r for r in result)

    def test_nested_tensors_detected(self):
        from perflab.optimizers.prompt import _detect_existing_optimizations
        files = {"model.py": "x = torch.nested.nested_tensor(tensors)"}
        result = _detect_existing_optimizations(files, "pytorch")
        assert any("Nested tensors" in r for r in result)

    def test_semi_structured_sparsity_detected(self):
        from perflab.optimizers.prompt import _detect_existing_optimizations
        files = {"model.py": "w = to_sparse_semi_structured(weight)"}
        result = _detect_existing_optimizations(files, "pytorch")
        assert any("sparsity" in r.lower() for r in result)

    def test_shard_map_detected(self):
        from perflab.optimizers.prompt import _detect_existing_optimizations
        files = {"model.py": "out = shard_map(f, mesh, in_specs, out_specs)(x)"}
        result = _detect_existing_optimizations(files, "jax")
        assert any("shard_map" in r for r in result)

    def test_pallas_detected(self):
        from perflab.optimizers.prompt import _detect_existing_optimizations
        files = {"kernel.py": "from jax.experimental.pallas import pallas_call"}
        result = _detect_existing_optimizations(files, "jax")
        assert any("Pallas" in r for r in result)

    def test_wmma_detected(self):
        from perflab.optimizers.prompt import _detect_existing_optimizations
        files = {"kernel.cu": "wmma::mma_sync(acc, a, b, acc);"}
        result = _detect_existing_optimizations(files, "cuda")
        assert any("WMMA" in r or "Tensor Core" in r for r in result)

    def test_cp_async_detected(self):
        from perflab.optimizers.prompt import _detect_existing_optimizations
        files = {"kernel.cu": "cp.async.cg.shared.global [dst], [src], 16;"}
        result = _detect_existing_optimizations(files, "cuda")
        assert any("cp.async" in r or "async" in r.lower() for r in result)

    def test_cub_detected(self):
        from perflab.optimizers.prompt import _detect_existing_optimizations
        files = {"kernel.cu": "cub::DeviceReduce::Sum(...)"}
        result = _detect_existing_optimizations(files, "cuda")
        assert any("CUB" in r for r in result)
