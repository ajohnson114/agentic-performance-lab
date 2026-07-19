"""Tests for MCP task-authoring tools and template generation."""
from __future__ import annotations

import textwrap

import pytest
import yaml

_has_fastmcp = True
try:
    import fastmcp  # noqa: F401
except ImportError:
    _has_fastmcp = False

needs_fastmcp = pytest.mark.skipif(not _has_fastmcp, reason="fastmcp not installed")


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------

class TestGenerateTaskFiles:
    """Tests for perflab.server.task_templates.generate_task_files."""

    def test_python_generates_all_files(self, tmp_path):
        from perflab.server.task_templates import generate_task_files

        files = generate_task_files(
            name="my_task",
            program_type="python",
            workspace="tasks/custom/my_task",
            description="Sum of squares",
        )
        assert "my_task.py" in files
        assert "bench.py" in files
        assert "tests.py" in files
        assert "tuning.yaml" in files
        assert "task.yaml" in files

    @pytest.mark.parametrize("program_type", ["python", "pytorch", "jax", "triton", "cpp", "cuda"])
    def test_all_program_types_generate(self, program_type):
        from perflab.server.task_templates import generate_task_files

        files = generate_task_files(
            name="test_task",
            program_type=program_type,
            workspace="tasks/test/test_task",
        )
        assert len(files) == 5
        assert "bench.py" in files
        assert "tests.py" in files
        assert "tuning.yaml" in files
        assert "task.yaml" in files

    def test_cpp_generates_cpp_source(self):
        from perflab.server.task_templates import generate_task_files

        files = generate_task_files(name="matmul", program_type="cpp", workspace="tasks/custom/matmul")
        assert "matmul.cpp" in files
        assert "nvcc" not in files["task.yaml"]
        assert "g++" in files["task.yaml"]

    def test_cuda_generates_cu_source(self):
        from perflab.server.task_templates import generate_task_files

        files = generate_task_files(name="kernel", program_type="cuda", workspace="tasks/custom/kernel")
        assert "kernel.cu" in files
        assert "nvcc" in files["task.yaml"]

    def test_pytorch_bench_has_cuda_sync(self):
        from perflab.server.task_templates import generate_task_files

        files = generate_task_files(name="model", program_type="pytorch", workspace="tasks/custom/model")
        assert "synchronize" in files["bench.py"]

    def test_jax_bench_has_block_until_ready(self):
        from perflab.server.task_templates import generate_task_files

        files = generate_task_files(name="jax_op", program_type="jax", workspace="tasks/custom/jax_op")
        assert "block_until_ready" in files["bench.py"]

    def test_task_yaml_parses(self, tmp_path):
        from perflab.server.task_templates import generate_task_files

        files = generate_task_files(
            name="my_task",
            program_type="python",
            workspace=str(tmp_path),
            metric_name="throughput.median",
            metric_mode="maximize",
        )
        # Write all files
        for filename, content in files.items():
            (tmp_path / filename).write_text(content, encoding="utf-8")

        # Parse task.yaml
        data = yaml.safe_load((tmp_path / "task.yaml").read_text(encoding="utf-8"))
        assert data["name"] == "my_task"
        assert data["program_type"] == "python"
        assert data["benchmark"]["metric"]["name"] == "throughput.median"
        assert data["benchmark"]["metric"]["mode"] == "maximize"

    def test_task_yaml_loads_as_taskspec(self, tmp_path):
        from perflab.server.task_templates import generate_task_files
        from perflab.task_spec import TaskSpec

        files = generate_task_files(
            name="loadtest",
            program_type="python",
            workspace=str(tmp_path),
        )
        for filename, content in files.items():
            (tmp_path / filename).write_text(content, encoding="utf-8")

        task = TaskSpec.load(tmp_path / "task.yaml")
        assert task.name == "loadtest"
        assert task.program_type == "python"

    def test_fixed_params_in_contract(self, tmp_path):
        from perflab.server.task_templates import generate_task_files

        files = generate_task_files(
            name="matmul",
            program_type="python",
            workspace=str(tmp_path),
            fixed_params={"M": 4096, "N": 4096},
        )
        for filename, content in files.items():
            (tmp_path / filename).write_text(content, encoding="utf-8")

        data = yaml.safe_load((tmp_path / "task.yaml").read_text(encoding="utf-8"))
        assert data["contract"]["fixed_params"]["M"] == 4096
        assert data["contract"]["fixed_params"]["N"] == 4096

    def test_target_hardware_in_task_yaml(self):
        from perflab.server.task_templates import generate_task_files

        files = generate_task_files(
            name="hw_test",
            program_type="pytorch",
            workspace="tasks/custom/hw_test",
            target_hardware="NVIDIA A100",
        )
        data = yaml.safe_load(files["task.yaml"])
        assert data["target_hardware"] == "NVIDIA A100"

    @pytest.mark.parametrize("program_type", ["python", "pytorch", "jax", "triton", "cpp", "cuda"])
    def test_scaffold_omits_dead_profile_plan(self, program_type):
        """profile_plan is parsed by TaskSpec but never consumed by profiler
        selection (perflab.profilers.select_profilers branches on
        program_type only) -- new task.yaml scaffolds must not suggest it
        does anything by including it."""
        from perflab.server.task_templates import generate_task_files

        files = generate_task_files(
            name="scaffold_test",
            program_type=program_type,
            workspace="tasks/custom/scaffold_test",
        )
        assert "profile_plan" not in files["task.yaml"]
        data = yaml.safe_load(files["task.yaml"])
        assert "profile_plan" not in data


# ---------------------------------------------------------------------------
# Profiler suggestions
# ---------------------------------------------------------------------------

class TestSuggestProfilers:
    def test_python_gets_cpu_flame(self):
        from perflab.server.task_templates import suggest_profilers

        result = suggest_profilers("python")
        assert "cpu_flame" in result["always"]

    def test_pytorch_gets_torch_trace(self):
        from perflab.server.task_templates import suggest_profilers

        result = suggest_profilers("pytorch")
        assert "torch_trace" in result["always"]

    def test_cuda_gets_ncu(self):
        from perflab.server.task_templates import suggest_profilers

        result = suggest_profilers("cuda")
        assert "ncu" in result["always"]

    def test_apple_silicon_gets_metal(self):
        from perflab.server.task_templates import suggest_profilers

        result = suggest_profilers("pytorch", target_hardware="Apple M2 Pro")
        assert "metal_trace" in result["optional"]
        assert "ncu" not in result["always"]
        assert "nsys" not in result["always"]

    def test_tpu_gets_jax_profiler(self):
        from perflab.server.task_templates import suggest_profilers

        result = suggest_profilers("jax", target_hardware="TPU v5e")
        assert "jax" in result["always"]

    def test_rationale_populated(self):
        from perflab.server.task_templates import suggest_profilers

        result = suggest_profilers("pytorch")
        assert len(result["rationale"]) > 0
        assert "torch_trace" in result["rationale"]


# ---------------------------------------------------------------------------
# Threshold suggestions
# ---------------------------------------------------------------------------

class TestSuggestThresholds:
    def test_python_thresholds(self):
        from perflab.server.task_templates import suggest_thresholds

        result = suggest_thresholds("python")
        thresh = result["suggested_thresholds"]
        assert "perf_ipc_low" in thresh
        assert thresh["perf_ipc_low"]["value"] == 0.5

    def test_cuda_thresholds(self):
        from perflab.server.task_templates import suggest_thresholds

        result = suggest_thresholds("cuda")
        thresh = result["suggested_thresholds"]
        assert "ncu_sm_util_low" in thresh

    def test_tpu_hardware_adds_tpu_thresholds(self):
        from perflab.server.task_templates import suggest_thresholds

        result = suggest_thresholds("jax", target_hardware="TPU v5e")
        thresh = result["suggested_thresholds"]
        assert "tpu_mxu_util_low" in thresh
        assert "tpu_padding_waste_pct_high" in thresh

    def test_descriptions_present(self):
        from perflab.server.task_templates import suggest_thresholds

        result = suggest_thresholds("cpp")
        for _key, entry in result["suggested_thresholds"].items():
            assert "description" in entry
            assert len(entry["description"]) > 0


# ---------------------------------------------------------------------------
# Bench.py linting
# ---------------------------------------------------------------------------

class TestLintBenchScript:
    def test_valid_bench_passes(self):
        from perflab.server.task_templates import lint_bench_script

        content = textwrap.dedent("""\
            import json, os, time
            from statistics import median
            parser.add_argument("--json", required=True)
            warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
            repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 20))
            json.dumps(out)
            "ok": True,
        """)
        result = lint_bench_script(content)
        assert result["passed"] is True
        assert len(result["errors"]) == 0

    def test_missing_json_flag(self):
        from perflab.server.task_templates import lint_bench_script

        result = lint_bench_script("import time\nprint('hello')")
        assert result["passed"] is False
        assert any("--json" in e for e in result["errors"])

    def test_missing_json_dumps(self):
        from perflab.server.task_templates import lint_bench_script

        result = lint_bench_script("--json\nprint('hello')")
        assert result["passed"] is False
        assert any("json.dump" in e for e in result["errors"])

    def test_missing_env_vars_warns(self):
        from perflab.server.task_templates import lint_bench_script

        content = '--json\njson.dumps({"ok": True})\nmedian(times)\n'
        result = lint_bench_script(content)
        assert result["passed"] is True
        assert any("PERFLAB_BENCH_WARMUP" in w for w in result["warnings"])
        assert any("PERFLAB_BENCH_REPEATS" in w for w in result["warnings"])

    def test_cuda_without_sync_warns(self):
        from perflab.server.task_templates import lint_bench_script

        content = '--json\njson.dumps({"ok": True})\ntorch.cuda.is_available()\nmedian(times)\n'
        result = lint_bench_script(content)
        assert any("synchronize" in w for w in result["warnings"])

    def test_jax_without_block_warns(self):
        from perflab.server.task_templates import lint_bench_script

        content = '--json\njson.dumps({"ok": True})\nimport jax\nmedian(times)\n'
        result = lint_bench_script(content)
        assert any("block_until_ready" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Contract suggestion
# ---------------------------------------------------------------------------

class TestSuggestContract:
    def test_detects_dimension_params(self):
        from perflab.server.task_templates import suggest_contract_from_bench

        content = textwrap.dedent("""\
            knobs = yaml.safe_load(Path("tuning.yaml").read_text())
            M = int(knobs.get("M", 4096))
            N = int(knobs.get("N", 4096))
            K = int(knobs.get("K", 4096))
        """)
        result = suggest_contract_from_bench(content, "cuda")
        assert "M" in result["fixed_params"]
        assert "N" in result["fixed_params"]
        assert "K" in result["fixed_params"]

    def test_skips_tunable_params(self):
        from perflab.server.task_templates import suggest_contract_from_bench

        content = textwrap.dedent("""\
            knobs = yaml.safe_load(Path("tuning.yaml").read_text())
            BLOCK_SIZE = int(knobs.get("BLOCK_SIZE", 64))
            TILE_M = int(knobs.get("TILE_M", 128))
        """)
        result = suggest_contract_from_bench(content, "cuda")
        assert "BLOCK_SIZE" not in result["fixed_params"]
        assert "TILE_M" not in result["fixed_params"]

    def test_detects_output_fields(self):
        from perflab.server.task_templates import suggest_contract_from_bench

        content = textwrap.dedent("""\
            out = {
                "tflops": {"median": tflops},
                "latency_ms": {"median": med},
                "meta": {"N": N},
                "ok": True,
            }
        """)
        result = suggest_contract_from_bench(content, "pytorch")
        assert "tflops" in result["required_bench_fields"]
        assert "latency_ms" in result["required_bench_fields"]
        # 'meta' should be excluded
        assert "meta" not in result["required_bench_fields"]

    def test_gpu_min_repeats(self):
        from perflab.server.task_templates import suggest_contract_from_bench

        result = suggest_contract_from_bench("", "cuda")
        assert result["min_repeats"] >= 5

    def test_cpp_min_repeats(self):
        from perflab.server.task_templates import suggest_contract_from_bench

        result = suggest_contract_from_bench("", "cpp")
        assert result["min_repeats"] >= 10


# ---------------------------------------------------------------------------
# MCP tool integration (create_task, validate_task)
# ---------------------------------------------------------------------------

@needs_fastmcp
class TestCreateTaskTool:
    """Test the create_task MCP tool function directly."""

    def test_creates_directory_and_files(self, tmp_path):
        from perflab.server.mcp_server import create_task

        result = create_task(
            name="my_bench",
            program_type="python",
            tasks_root=str(tmp_path),
        )
        assert "error" not in result
        assert len(result["files_created"]) == 5
        assert (tmp_path / "custom" / "my_bench" / "task.yaml").exists()
        assert (tmp_path / "custom" / "my_bench" / "bench.py").exists()
        assert (tmp_path / "custom" / "my_bench" / "tests.py").exists()

    def test_rejects_invalid_program_type(self, tmp_path):
        from perflab.server.mcp_server import create_task

        result = create_task(name="bad", program_type="rust", tasks_root=str(tmp_path))
        assert "error" in result

    def test_rejects_existing_directory(self, tmp_path):
        from perflab.server.mcp_server import create_task

        (tmp_path / "custom" / "dup").mkdir(parents=True)
        result = create_task(name="dup", program_type="python", tasks_root=str(tmp_path))
        assert "error" in result
        assert "already exists" in result["error"]

    def test_rejects_path_traversal_names(self, tmp_path):
        from perflab.server.mcp_server import create_task

        for bad in ("../evil", "a/b", "..", ".", ".hidden", "/abs/path", ""):
            result = create_task(name=bad, program_type="python", tasks_root=str(tmp_path))
            assert "error" in result, f"name={bad!r} should be rejected"

        result = create_task(
            name="ok", program_type="python", category="../outside",
            tasks_root=str(tmp_path),
        )
        assert "error" in result
        assert not (tmp_path.parent / "outside").exists()

    def test_custom_category(self, tmp_path):
        from perflab.server.mcp_server import create_task

        result = create_task(
            name="attn", program_type="pytorch", category="attention",
            tasks_root=str(tmp_path),
        )
        assert "error" not in result
        assert (tmp_path / "attention" / "attn" / "task.yaml").exists()

    def test_created_task_validates(self, tmp_path):
        from perflab.server.mcp_server import create_task, validate_task

        create_task(
            name="val_test",
            program_type="python",
            tasks_root=str(tmp_path),
        )
        result = validate_task(str(tmp_path / "custom" / "val_test" / "task.yaml"))
        assert result["valid"] is True, f"Validation errors: {result['errors']}"


@needs_fastmcp
class TestValidateTaskTool:
    def test_missing_file(self, tmp_path):
        from perflab.server.mcp_server import validate_task

        result = validate_task(str(tmp_path / "nonexistent.yaml"))
        assert result["valid"] is False
        assert any("not found" in e.lower() for e in result["errors"])

    def test_invalid_yaml(self, tmp_path):
        from perflab.server.mcp_server import validate_task

        bad_yaml = tmp_path / "task.yaml"
        bad_yaml.write_text("{{invalid: yaml: [}", encoding="utf-8")
        result = validate_task(str(bad_yaml))
        assert result["valid"] is False

    def test_missing_required_fields(self, tmp_path):
        from perflab.server.mcp_server import validate_task

        (tmp_path / "task.yaml").write_text("name: test\n", encoding="utf-8")
        result = validate_task(str(tmp_path / "task.yaml"))
        assert result["valid"] is False
        assert any("program_type" in e for e in result["errors"])

    def test_invalid_program_type(self, tmp_path):
        from perflab.server.mcp_server import validate_task

        content = yaml.dump({
            "name": "test",
            "program_type": "fortran",
            "correctness": {"cmd": "true"},
            "benchmark": {"cmd": "true", "metric": {"name": "x", "mode": "maximize"}},
        })
        (tmp_path / "task.yaml").write_text(content, encoding="utf-8")
        result = validate_task(str(tmp_path / "task.yaml"))
        assert result["valid"] is False
        assert any("fortran" in e for e in result["errors"])

    def test_blocklisted_edit_policy(self, tmp_path):
        from perflab.server.mcp_server import validate_task

        content = yaml.dump({
            "name": "test",
            "program_type": "python",
            "correctness": {"cmd": "true"},
            "benchmark": {"cmd": "true", "metric": {"name": "x", "mode": "maximize"}},
            "edit_policy": {"allowed_paths": ["bench.py"]},
        })
        (tmp_path / "task.yaml").write_text(content, encoding="utf-8")
        (tmp_path / "bench.py").write_text("pass", encoding="utf-8")
        (tmp_path / "tests.py").write_text("pass", encoding="utf-8")
        result = validate_task(str(tmp_path / "task.yaml"))
        assert result["valid"] is False
        assert any("blocklisted" in e for e in result["errors"])

    def test_cpp_without_build_errors(self, tmp_path):
        from perflab.server.mcp_server import validate_task

        content = yaml.dump({
            "name": "test",
            "program_type": "cpp",
            "correctness": {"cmd": "true"},
            "benchmark": {"cmd": "true", "metric": {"name": "x", "mode": "maximize"}},
        })
        (tmp_path / "task.yaml").write_text(content, encoding="utf-8")
        result = validate_task(str(tmp_path / "task.yaml"))
        assert any("build" in e.lower() for e in result["errors"])

    def test_valid_sample_task(self):
        """Validate the existing _sample task."""
        from perflab.server.mcp_server import validate_task

        result = validate_task("tasks/_sample/task.yaml")
        assert result["valid"] is True, f"Errors: {result['errors']}"

    def test_warns_missing_bench_py(self, tmp_path):
        from perflab.server.mcp_server import validate_task

        content = yaml.dump({
            "name": "test",
            "program_type": "python",
            "correctness": {"cmd": "true"},
            "benchmark": {"cmd": "true", "metric": {"name": "x", "mode": "maximize"}},
        })
        (tmp_path / "task.yaml").write_text(content, encoding="utf-8")
        result = validate_task(str(tmp_path / "task.yaml"))
        assert any("bench.py" in w for w in result["warnings"])

    def test_contract_validation_errors(self, tmp_path):
        from perflab.server.mcp_server import validate_task

        content = yaml.dump({
            "name": "test",
            "program_type": "python",
            "correctness": {"cmd": "true"},
            "benchmark": {"cmd": "true", "metric": {"name": "x", "mode": "maximize"}},
            "contract": {"required_bench_fields": ["..bad"]},
        })
        (tmp_path / "task.yaml").write_text(content, encoding="utf-8")
        (tmp_path / "bench.py").write_text("pass", encoding="utf-8")
        (tmp_path / "tests.py").write_text("pass", encoding="utf-8")
        result = validate_task(str(tmp_path / "task.yaml"))
        assert any("Contract" in e for e in result["errors"])


@needs_fastmcp
class TestLintBenchScriptTool:
    """Test the lint_bench_script MCP tool wrapper."""

    def test_lints_sample_bench(self):
        from perflab.server.mcp_server import lint_bench_script

        result = lint_bench_script("tasks/_sample/task.yaml")
        assert "error" not in result
        assert result["passed"] is True

    def test_missing_task_yaml(self, tmp_path):
        from perflab.server.mcp_server import lint_bench_script

        result = lint_bench_script(str(tmp_path / "nonexistent.yaml"))
        assert "error" in result

    def test_missing_bench_py(self, tmp_path):
        from perflab.server.mcp_server import lint_bench_script

        (tmp_path / "task.yaml").write_text("name: test\n", encoding="utf-8")
        result = lint_bench_script(str(tmp_path / "task.yaml"))
        assert "error" in result
        assert "bench.py" in result["error"]


@needs_fastmcp
class TestSuggestContractTool:
    """Test the suggest_contract MCP tool wrapper."""

    def test_analyzes_sample_bench(self):
        from perflab.server.mcp_server import suggest_contract

        result = suggest_contract("tasks/_sample/task.yaml")
        assert "error" not in result
        assert "fixed_params" in result
        assert "required_bench_fields" in result

    def test_missing_bench_py(self, tmp_path):
        from perflab.server.mcp_server import suggest_contract

        (tmp_path / "task.yaml").write_text("name: test\n", encoding="utf-8")
        result = suggest_contract(str(tmp_path / "task.yaml"))
        assert "error" in result


@needs_fastmcp
class TestSuggestProfilersTool:
    def test_valid_type(self):
        from perflab.server.mcp_server import suggest_profilers

        result = suggest_profilers("pytorch")
        assert "always" in result
        assert "optional" in result

    def test_invalid_type(self):
        from perflab.server.mcp_server import suggest_profilers

        result = suggest_profilers("fortran")
        assert "error" in result


@needs_fastmcp
class TestSuggestThresholdsTool:
    def test_valid_type(self):
        from perflab.server.mcp_server import suggest_thresholds

        result = suggest_thresholds("cuda")
        assert "suggested_thresholds" in result

    def test_invalid_type(self):
        from perflab.server.mcp_server import suggest_thresholds

        result = suggest_thresholds("ruby")
        assert "error" in result


@needs_fastmcp
class TestAgentIsolationResolution:
    """The MCP agent tools must honor task.yaml/config isolation (the CLI
    already did; the server previously built AgentConfig without it)."""

    def test_task_yaml_level_reaches_policy(self, tmp_path, monkeypatch):
        import perflab.config
        from perflab.config import PerfLabConfig
        from perflab.server.mcp_server import _resolve_isolation

        monkeypatch.setattr(perflab.config, "load_config", lambda: PerfLabConfig())
        task_file = tmp_path / "task.yaml"
        task_file.write_text("isolation:\n  level: restricted\n", encoding="utf-8")

        policy = _resolve_isolation(task_file)
        assert policy is not None
        assert policy.level == "restricted"

    def test_defaults_to_no_policy(self, tmp_path, monkeypatch):
        import perflab.config
        from perflab.config import PerfLabConfig
        from perflab.server.mcp_server import _resolve_isolation

        monkeypatch.setattr(perflab.config, "load_config", lambda: PerfLabConfig())
        task_file = tmp_path / "task.yaml"
        task_file.write_text("name: t\n", encoding="utf-8")

        assert _resolve_isolation(task_file) is None
