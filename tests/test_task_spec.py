"""Tests for perflab.task_spec.TaskSpec loading."""
from __future__ import annotations

from pathlib import Path

import pytest

from perflab.task_spec import TaskSpec


class TestTaskSpecLoad:
    def test_load_basic_fields(self, sample_task_yaml: Path):
        """Load a minimal task and verify core fields."""
        task = TaskSpec.load(sample_task_yaml)
        assert task.name == "test-task"
        assert task.program_type == "python"
        assert task.build is None
        assert task.correctness.cmd == "python tests.py"
        assert task.correctness.expected_exit == 0
        assert task.benchmark.metric.name == "throughput.median"
        assert task.benchmark.metric.mode == "maximize"
        assert task.benchmark.warmup == 1
        assert task.benchmark.repeats == 5

    def test_workspace_is_parent_dir(self, sample_task_yaml: Path):
        """Workspace should be the parent directory of task.yaml."""
        task = TaskSpec.load(sample_task_yaml)
        assert task.workspace == sample_task_yaml.parent.resolve()

    def test_edit_policy(self, sample_task_yaml: Path):
        task = TaskSpec.load(sample_task_yaml)
        assert "*.py" in task.edit_policy.allowed_paths

    def test_constraints(self, sample_task_yaml: Path):
        task = TaskSpec.load(sample_task_yaml)
        assert task.constraints.max_iters == 5
        assert task.constraints.regression_tolerance == 0.02

    def test_rlimit_as_gb_zero_means_none(self, sample_task_yaml: Path):
        """rlimit_as_gb=0 in YAML should produce a 0.0 float (disabled via _resolve_rlimit)."""
        task = TaskSpec.load(sample_task_yaml)
        # When rlimit_as_gb is 0, the TaskSpec stores 0.0.
        # The benchmark runner interprets rlimit_as_gb <= 0 as disabled.
        assert task.constraints.rlimit_as_gb == 0.0

    def test_contract_defaults(self, sample_task_yaml: Path):
        task = TaskSpec.load(sample_task_yaml)
        assert task.contract.min_repeats == 1
        assert "ok" in task.contract.required_bench_fields

    def test_load_cpp_task(self, sample_cpp_task_yaml: Path):
        """Load a C++ task with build step and roofline."""
        task = TaskSpec.load(sample_cpp_task_yaml)
        assert task.name == "test-cpp-task"
        assert task.program_type == "cpp"
        assert task.build is not None
        assert "g++" in task.build.cmd
        assert task.roofline is not None
        assert task.roofline.peak_tflops == 1.0
        assert task.roofline.peak_mem_bw_gbs == 50.0
        assert task.target_hardware == "Intel Xeon"

    def test_contract_fixed_params(self, sample_cpp_task_yaml: Path):
        task = TaskSpec.load(sample_cpp_task_yaml)
        assert task.contract.fixed_params == {"M": 512, "N": 512, "K": 512}
        assert task.contract.min_repeats == 5

    def test_analysis_thresholds_defaults(self, sample_task_yaml: Path):
        """Default analysis thresholds should be populated."""
        task = TaskSpec.load(sample_task_yaml)
        # Spot-check a few default values
        assert task.analysis_thresholds.ncu_sm_util_low > 0
        assert task.analysis_thresholds.perf_ipc_low > 0

    def test_missing_optional_fields_use_defaults(self, tmp_workspace: Path):
        """A minimal task.yaml without optional fields should use defaults."""
        task_file = tmp_workspace / "task.yaml"
        task_file.write_text(
            "name: minimal\n"
            "program_type: python\n"
            "correctness:\n"
            "  cmd: python tests.py\n"
            "benchmark:\n"
            "  cmd: python bench.py --json out/bench.json\n"
            "  metric:\n"
            "    name: value\n",
            encoding="utf-8",
        )
        task = TaskSpec.load(task_file)
        assert task.name == "minimal"
        assert task.build is None
        assert task.roofline is None
        assert task.target_hardware is None
        assert task.constraints.max_iters == 10
        assert task.constraints.regression_tolerance == 0.02
        assert task.edit_policy.allowed_paths == []
        assert task.benchmark.secondary_metric is None

    def test_secondary_metric_parsed(self, tmp_workspace: Path):
        """A task with secondary_metric should parse it correctly."""
        task_file = tmp_workspace / "task.yaml"
        task_file.write_text(
            "name: pareto-test\n"
            "program_type: cpp\n"
            "correctness:\n"
            "  cmd: python tests.py\n"
            "benchmark:\n"
            "  cmd: python bench.py --json out/bench.json\n"
            "  metric:\n"
            "    name: tflops.median\n"
            "    mode: maximize\n"
            "  secondary_metric:\n"
            "    name: latency_ms.p95\n"
            "    mode: minimize\n",
            encoding="utf-8",
        )
        task = TaskSpec.load(task_file)
        assert task.benchmark.secondary_metric is not None
        assert task.benchmark.secondary_metric.name == "latency_ms.p95"
        assert task.benchmark.secondary_metric.mode == "minimize"
