"""Tests for perflab.profilers.power_profiler."""
from __future__ import annotations

from pathlib import Path

from perflab.profilers.power_profiler import (
    _compute_gpu_memory_stats,
    _compute_gpu_power_stats,
    _parse_rapl_output,
)


class TestParseRaplOutput:
    def test_parse_all_energy_fields(self, tmp_path: Path):
        text = """
 Performance counter stats for 'bench':

     12.34 Joules power/energy-pkg/
      8.56 Joules power/energy-cores/
      2.10 Joules power/energy-ram/

       5.000000000 seconds time elapsed
"""
        path = tmp_path / "rapl.txt"
        path.write_text(text)
        result = _parse_rapl_output(path)
        assert abs(result["package_joules"] - 12.34) < 0.01
        assert abs(result["cores_joules"] - 8.56) < 0.01
        assert abs(result["ram_joules"] - 2.10) < 0.01
        assert abs(result["elapsed_seconds"] - 5.0) < 0.01
        assert abs(result["avg_package_watts"] - 2.468) < 0.01

    def test_parse_package_only(self, tmp_path: Path):
        text = """
     25.00 Joules power/energy-pkg/
      10.000000000 seconds time elapsed
"""
        path = tmp_path / "rapl.txt"
        path.write_text(text)
        result = _parse_rapl_output(path)
        assert abs(result["package_joules"] - 25.0) < 0.01
        assert abs(result["avg_package_watts"] - 2.5) < 0.01

    def test_parse_empty_file(self, tmp_path: Path):
        path = tmp_path / "rapl.txt"
        path.write_text("")
        result = _parse_rapl_output(path)
        assert result == {}

    def test_parse_nonexistent_file(self, tmp_path: Path):
        path = tmp_path / "nonexistent.txt"
        result = _parse_rapl_output(path)
        assert result == {}


class TestComputeGpuPowerStats:
    def test_basic_stats(self):
        samples = [100.0, 150.0, 200.0, 250.0, 300.0]
        result = _compute_gpu_power_stats(samples)
        assert result["sample_count"] == 5
        assert abs(result["avg_watts"] - 200.0) < 0.1
        assert result["min_watts"] == 100.0
        assert result["max_watts"] == 300.0
        assert result["p50_watts"] == 200.0

    def test_single_sample(self):
        result = _compute_gpu_power_stats([150.0])
        assert result["sample_count"] == 1
        assert result["avg_watts"] == 150.0
        assert result["min_watts"] == 150.0
        assert result["max_watts"] == 150.0

    def test_empty_samples(self):
        result = _compute_gpu_power_stats([])
        assert result == {}

    def test_p95_calculation(self):
        samples = list(range(1, 101))  # 1 to 100
        result = _compute_gpu_power_stats([float(x) for x in samples])
        assert result["p95_watts"] >= 95.0


class TestComputeGpuMemoryStats:
    def test_basic_stats(self):
        samples = [
            {"used_mib": 4000.0, "total_mib": 8192.0},
            {"used_mib": 5000.0, "total_mib": 8192.0},
            {"used_mib": 6000.0, "total_mib": 8192.0},
        ]
        result = _compute_gpu_memory_stats(samples)
        assert result["sample_count"] == 3
        assert result["total_mib"] == 8192.0
        assert abs(result["avg_used_mib"] - 5000.0) < 0.1
        assert result["max_used_mib"] == 6000.0
        assert abs(result["utilization_pct"] - 73.2) < 0.2

    def test_single_sample(self):
        result = _compute_gpu_memory_stats([{"used_mib": 2048.0, "total_mib": 8192.0}])
        assert result["max_used_mib"] == 2048.0
        assert result["utilization_pct"] == 25.0

    def test_empty_samples(self):
        result = _compute_gpu_memory_stats([])
        assert result == {}

    def test_full_utilization(self):
        samples = [{"used_mib": 8192.0, "total_mib": 8192.0}]
        result = _compute_gpu_memory_stats(samples)
        assert abs(result["utilization_pct"] - 100.0) < 0.1
