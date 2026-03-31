"""Shared fixtures for perflab tests."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory with a minimal task layout."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "out").mkdir()
    return ws


@pytest.fixture
def sample_task_yaml(tmp_workspace: Path) -> Path:
    """Write a minimal task.yaml and return its path."""
    task_file = tmp_workspace / "task.yaml"
    task_file.write_text(textwrap.dedent("""\
        name: test-task
        program_type: python
        build: null
        correctness:
          cmd: "python tests.py"
          expected_exit: 0
        benchmark:
          cmd: "python bench.py --json out/bench.json"
          metric:
            name: throughput.median
            mode: maximize
          warmup: 1
          repeats: 5
        edit_policy:
          allowed_paths:
            - "*.py"
        constraints:
          max_iters: 5
          regression_tolerance: 0.02
          rlimit_as_gb: 0
        contract:
          fixed_params: {}
          min_repeats: 1
          required_bench_fields:
            - ok
    """), encoding="utf-8")
    return task_file


@pytest.fixture
def sample_cpp_task_yaml(tmp_workspace: Path) -> Path:
    """Write a minimal C++ task.yaml and return its path."""
    task_file = tmp_workspace / "task.yaml"
    task_file.write_text(textwrap.dedent("""\
        name: test-cpp-task
        program_type: cpp
        target_hardware: "Intel Xeon"
        build:
          cmd: "g++ -O2 -o matmul_bin matmul.cpp"
          expected_exit: 0
        correctness:
          cmd: "python tests.py"
          expected_exit: 0
        benchmark:
          cmd: "python bench.py --json out/bench.json"
          metric:
            name: tflops.median
            mode: maximize
          warmup: 3
          repeats: 20
        roofline:
          peak_tflops: 1.0
          peak_mem_bw_gbs: 50.0
        edit_policy:
          allowed_paths:
            - "matmul.cpp"
        constraints:
          max_iters: 10
          regression_tolerance: 0.02
        contract:
          fixed_params:
            M: 512
            N: 512
            K: 512
          min_repeats: 5
          required_bench_fields:
            - ok
            - tflops.median
    """), encoding="utf-8")
    return task_file
