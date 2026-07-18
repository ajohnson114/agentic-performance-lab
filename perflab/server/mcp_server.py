"""PerfLab MCP server — exposes profiling and optimization tools via FastMCP."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, TypeVar

from fastmcp import Context, FastMCP

mcp = FastMCP("perflab", instructions="PerfLab agentic profiling & optimization server")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_MAX_OUTPUT_BYTES = 100_000  # ~100 KB


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


# ---------------------------------------------------------------------------
# Background agent run management
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=1)
_active_runs: dict[str, dict] = {}
_lock = threading.Lock()
_agent_lock = threading.Lock()


# ===========================================================================
# Task inspection
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
def list_tasks(tasks_root: str = "tasks") -> list[dict]:
    """List available task definitions by globbing for task.yaml files."""
    import yaml

    root = Path(tasks_root)
    results: list[dict] = []
    for p in sorted(root.rglob("task.yaml")):
        try:
            with open(p) as f:
                data = yaml.safe_load(f)
            results.append({
                "path": str(p),
                "name": data.get("name", p.parent.name),
                "program_type": data.get("program_type", "python"),
            })
        except Exception:  # noqa: BLE001 -- best-effort listing, a single malformed task.yaml must not abort the scan
            results.append({"path": str(p), "name": p.parent.name, "error": "failed to parse"})
    return results


@mcp.tool(annotations={"readOnlyHint": True})
def show_task(task_yaml: str) -> dict:
    """Show effective configuration for a task: benchmark, constraints, contract, edit policy, data hints, and tuning knobs."""
    from perflab.task_spec import TaskSpec

    task = TaskSpec.load(Path(task_yaml))

    result: dict = {
        "name": task.name,
        "workspace": str(task.workspace),
        "program_type": task.program_type,
        "target_hardware": task.target_hardware,
        "benchmark": {
            "cmd": task.benchmark.cmd,
            "metric": task.benchmark.metric.name,
            "mode": task.benchmark.metric.mode,
            "warmup": task.benchmark.warmup,
            "repeats": task.benchmark.repeats,
        },
        "constraints": dataclasses.asdict(task.constraints),
        "contract": dataclasses.asdict(task.contract),
        "agent": dataclasses.asdict(task.agent),
        "edit_policy": {"allowed_paths": task.edit_policy.allowed_paths},
        "correctness": {
            "cmd": task.correctness.cmd,
            "expected_exit": task.correctness.expected_exit,
        },
    }

    if task.build:
        result["build"] = {"cmd": task.build.cmd}

    if task.roofline:
        result["roofline"] = dataclasses.asdict(task.roofline)

    # Data hints — only non-None fields
    dh_fields = {
        f.name: getattr(task.data_hints, f.name)
        for f in dataclasses.fields(task.data_hints)
        if getattr(task.data_hints, f.name) is not None
    }
    if dh_fields:
        result["data_hints"] = dh_fields

    # Tuning.yaml info
    tuning_path = task.workspace / "tuning.yaml"
    if tuning_path.exists():
        try:
            import yaml
            knobs = yaml.safe_load(tuning_path.read_text(encoding="utf-8"))
            if knobs:
                sweep = knobs.get("sweep", {})
                fixed = task.contract.fixed_params if task.contract else {}
                tunable = {k: v for k, v in knobs.items() if k != "sweep" and k not in fixed}
                result["tuning"] = {
                    "fixed_params": fixed,
                    "tunable_params": tunable,
                    "sweep": sweep,
                }
        except Exception:  # noqa: BLE001 -- best-effort tuning.yaml display, skip on any parse issue
            pass

    return _guard_output_size(result)


@mcp.tool(annotations={"readOnlyHint": True})
def show_task_schema() -> dict:
    """Show the task.yaml schema: all fields, types, and descriptions."""
    return {
        "schema": [
            {"section": "REQUIRED", "fields": [
                {"name": "name", "type": "str", "desc": "Task name (used in reports and file paths)"},
                {"name": "program_type", "type": "str", "desc": "python | pytorch | jax | triton | cpp | cuda"},
                {"name": "correctness.cmd", "type": "str", "desc": "Command to run correctness test (must exit 0)"},
                {"name": "benchmark.cmd", "type": "str", "desc": "Command to run benchmark (must write --json)"},
                {"name": "benchmark.metric.name", "type": "str", "desc": "Dotted path into bench.json (e.g., tflops.median)"},
                {"name": "benchmark.metric.mode", "type": "str", "desc": "maximize | minimize"},
                {"name": "edit_policy.allowed_paths", "type": "list[str]", "desc": "Files the agent can edit"},
            ]},
            {"section": "BENCHMARK", "fields": [
                {"name": "benchmark.warmup", "type": "int", "desc": "Warmup iterations before timing (default: 3)"},
                {"name": "benchmark.repeats", "type": "int", "desc": "Timed iterations (default: 20)"},
                {"name": "benchmark.secondary_metric", "type": "dict|null", "desc": "Secondary metric for Pareto analysis"},
            ]},
            {"section": "CONSTRAINTS", "fields": [
                {"name": "constraints.max_iters", "type": "int", "desc": "Max agent iterations (default: 10)"},
                {"name": "constraints.regression_tolerance", "type": "float", "desc": "Min improvement fraction (default: 0.02)"},
                {"name": "constraints.rlimit_as_gb", "type": "float|null", "desc": "Memory limit in GB (null=auto)"},
                {"name": "constraints.prompt_token_budget", "type": "int", "desc": "Max prompt tokens (0=unlimited)"},
                {"name": "constraints.allow_fast_math", "type": "bool", "desc": "Permit -ffast-math (default: false)"},
            ]},
            {"section": "CONTRACT", "fields": [
                {"name": "contract.fixed_params", "type": "dict", "desc": "Values enforced in bench.json meta (e.g., {M: 4096})"},
                {"name": "contract.min_repeats", "type": "int", "desc": "Minimum benchmark repeats (default: 1)"},
                {"name": "contract.required_bench_fields", "type": "list[str]", "desc": "Fields that must exist in bench.json"},
            ]},
            {"section": "ROOFLINE", "fields": [
                {"name": "roofline.peak_tflops", "type": "float", "desc": "Hardware peak TFLOPS"},
                {"name": "roofline.peak_mem_bw_gbs", "type": "float", "desc": "Peak memory bandwidth in GB/s"},
                {"name": "roofline.dtype_peaks", "type": "dict|null", "desc": "Per-dtype peaks (fp32, tf32, fp16, bf16)"},
            ]},
            {"section": "DATA_HINTS", "fields": [
                {"name": "data_hints.sparsity", "type": "float|null", "desc": "Fraction of zeros (>0.9 suggests sparse formats)"},
                {"name": "data_hints.dtype_safety", "type": "str|null", "desc": "fp16_safe | bf16_safe | int8_safe"},
                {"name": "data_hints.access_pattern", "type": "str|null", "desc": "sequential | random | strided | blocked"},
                {"name": "data_hints.batch_size_range", "type": "list|null", "desc": "[min, max] production batch sizes"},
                {"name": "data_hints.custom", "type": "list|null", "desc": "Free-form hints: ['data is symmetric']"},
            ]},
        ],
    }


@mcp.tool(annotations={"readOnlyHint": True})
def show_task_authoring_guide() -> dict:
    """Show a step-by-step guide for creating a new PerfLab task from scratch.

    Returns an onboarding-friendly walkthrough covering directory structure,
    required files, common pitfalls, and which authoring tools to use at each
    step. Start here if you are new to PerfLab.
    """
    return {
        "overview": (
            "A PerfLab task is a directory containing 4-5 files that define a "
            "performance optimization challenge. The agent reads task.yaml, "
            "runs your benchmark, profiles your code, and iteratively improves it."
        ),
        "steps": [
            {
                "step": 1,
                "title": "Scaffold the task directory",
                "description": (
                    "Use the create_task tool to generate a complete starter "
                    "directory with all required files pre-filled for your "
                    "program type (python, pytorch, jax, triton, cpp, cuda)."
                ),
                "tool": "create_task",
            },
            {
                "step": 2,
                "title": "Replace the placeholder source code",
                "description": (
                    "Edit the generated source file (e.g., my_task.py) with "
                    "your actual workload. Keep the function signature so that "
                    "bench.py and tests.py can import it."
                ),
            },
            {
                "step": 3,
                "title": "Write correctness tests",
                "description": (
                    "Edit tests.py with known-good expected values. The agent "
                    "rejects any optimization that makes tests.py fail."
                ),
            },
            {
                "step": 4,
                "title": "Adapt the benchmark harness",
                "description": (
                    "Edit bench.py to time your actual workload. Make sure the "
                    "JSON output includes the metric path referenced in "
                    "task.yaml. Use lint_bench_script to check compliance."
                ),
                "tool": "lint_bench_script",
            },
            {
                "step": 5,
                "title": "Lock down the contract",
                "description": (
                    "Use suggest_contract to analyze bench.py and identify "
                    "which parameters should be fixed (problem dimensions the "
                    "agent must not shrink). Add these to task.yaml's contract."
                ),
                "tool": "suggest_contract",
            },
            {
                "step": 6,
                "title": "Pick profilers and thresholds",
                "description": (
                    "Use suggest_profilers and suggest_thresholds to get "
                    "recommended settings for your program type and hardware. "
                    "Copy the suggestions into task.yaml."
                ),
                "tools": ["suggest_profilers", "suggest_thresholds"],
            },
            {
                "step": 7,
                "title": "Validate everything",
                "description": (
                    "Run validate_task on your task.yaml to catch schema "
                    "errors, missing files, and contract issues before your "
                    "first profiling run."
                ),
                "tool": "validate_task",
            },
            {
                "step": 8,
                "title": "Profile and optimize",
                "description": (
                    "Use profile_task for a baseline, then start_agent or "
                    "optimize_task to begin the optimization loop."
                ),
                "tools": ["profile_task", "start_agent", "optimize_task"],
            },
        ],
        "required_files": {
            "task.yaml": "Task configuration — benchmark command, metrics, constraints, edit policy",
            "bench.py": "Benchmark harness — times the workload, writes JSON metrics",
            "tests.py": "Correctness test — must exit 0 for valid code",
            "source file": "The code to optimize (e.g., matmul.py, kernel.cu)",
            "tuning.yaml": "Optional — numeric knobs for auto-tuning sweeps",
        },
        "common_pitfalls": [
            "Forgetting --json flag in bench.py (metrics won't be captured)",
            "Not calling torch.cuda.synchronize() or block_until_ready() (GPU timing is wrong)",
            "Leaving fixed_params empty (agent may shrink the problem to 'optimize')",
            "Including bench.py or tests.py in edit_policy (they're blocklisted for safety)",
            "Using python instead of python3 on macOS",
        ],
    }


@mcp.tool(annotations={"readOnlyHint": True})
def show_tuning_schema() -> dict:
    """Show tuning.yaml schema: fixed vs tunable params, sweep syntax."""
    return {
        "fixed_params": {
            "description": "Protected by contract.fixed_params — cannot be changed by agent or optimizer",
            "examples": ["M", "N", "K", "batch_size", "seq_len", "num_images"],
        },
        "tunable_params": {
            "description": "Fair game for optimization — agent can modify these",
            "examples": ["block_size", "TILE_M", "TILE_N", "TILE_K", "num_warps", "NUM_STAGES", "num_workers", "lr"],
        },
        "sweep": {
            "description": "Auto-tuning search space. Cartesian product of all lists, benchmarked automatically after accepted code edits (max 15 trials).",
            "syntax": {
                "TILE_M": [64, 128, 256],
                "TILE_N": [64, 128, 256],
                "NUM_STAGES": [2, 3, 4],
            },
            "note": "For CUDA, sweep ranges are centered on CUTLASS-optimal baselines. Each config is contract-validated.",
        },
    }


# ===========================================================================
# Task authoring — tools for creating and validating new tasks
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def create_task(
    name: str,
    program_type: str,
    category: str = "custom",
    description: str = "Performance optimization task",
    target_hardware: str | None = None,
    metric_name: str = "latency_ms.median",
    metric_mode: str = "minimize",
    fixed_params: dict | None = None,
    tasks_root: str = "tasks",
) -> dict:
    """Scaffold a complete task directory with all required files.

    Creates tasks/<category>/<name>/ with: task.yaml, bench.py, tests.py,
    a source file template, and tuning.yaml — all pre-configured for the
    chosen program_type (python, pytorch, jax, triton, cpp, cuda).

    The generated files are working starting points. Edit the source file with
    your actual workload, then adapt bench.py and tests.py accordingly.

    Args:
        name: Task name (used for the directory and source file).
        program_type: python | pytorch | jax | triton | cpp | cuda.
        category: Subdirectory under tasks/ (e.g. "matmul", "attention").
        description: Brief description of the optimization workload.
        target_hardware: Optional hardware hint (e.g. "NVIDIA A100", "Apple M2 Pro").
        metric_name: Dotted path into bench.json (e.g. "latency_ms.median", "tflops.median").
        metric_mode: "maximize" or "minimize".
        fixed_params: Contract fixed parameters (e.g. {"M": 4096, "N": 4096}).
        tasks_root: Root directory for tasks (default: "tasks").
    """
    from perflab.server.task_templates import generate_task_files

    valid_types = ("python", "pytorch", "jax", "triton", "cpp", "cuda")
    if program_type not in valid_types:
        return {"error": f"Invalid program_type: {program_type!r}. Must be one of: {valid_types}"}
    if metric_mode not in ("maximize", "minimize"):
        return {"error": f"Invalid metric_mode: {metric_mode!r}. Must be 'maximize' or 'minimize'"}

    task_dir = Path(tasks_root) / category / name
    if task_dir.exists():
        return {"error": f"Directory already exists: {task_dir}. Choose a different name or remove it first."}

    workspace = str(task_dir)
    typed_params: dict[str, int | float] | None = None
    if fixed_params:
        typed_params = {}
        for k, v in fixed_params.items():
            try:
                typed_params[str(k)] = int(v) if isinstance(v, int) or (isinstance(v, float) and v == int(v)) else float(v)
            except (ValueError, TypeError):
                return {"error": f"fixed_params[{k!r}] must be numeric, got {type(v).__name__}"}

    files = generate_task_files(
        name=name,
        program_type=program_type,
        workspace=workspace,
        description=description,
        target_hardware=target_hardware,
        metric_name=metric_name,
        metric_mode=metric_mode,
        fixed_params=typed_params,
    )

    task_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for filename, content in files.items():
        filepath = task_dir / filename
        filepath.write_text(content, encoding="utf-8")
        created.append(str(filepath))

    return {
        "task_dir": str(task_dir),
        "task_yaml": str(task_dir / "task.yaml"),
        "files_created": created,
        "next_steps": [
            f"Edit {task_dir}/{{source file}} with your actual workload",
            f"Edit {task_dir}/tests.py with correctness checks",
            f"Edit {task_dir}/bench.py to time your workload",
            f"Run validate_task('{task_dir / 'task.yaml'}') to check everything",
            f"Run profile_task('{task_dir / 'task.yaml'}') for a baseline",
        ],
    }


@mcp.tool(annotations={"readOnlyHint": True})
def validate_task(task_yaml: str) -> dict:
    """Validate a task.yaml without running it — catches schema errors, missing files, and contract issues.

    Returns a structured report of errors and warnings. Fix all errors before
    running profile_task or start_agent.
    """
    import yaml as yaml_mod

    errors: list[str] = []
    warnings: list[str] = []
    task_path = Path(task_yaml)

    # 1. File exists and parses
    if not task_path.exists():
        return {"valid": False, "errors": [f"File not found: {task_yaml}"], "warnings": []}

    try:
        raw = yaml_mod.safe_load(task_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- validation tool must report any failure as a structured error, never crash
        return {"valid": False, "errors": [f"YAML parse error: {exc}"], "warnings": []}

    if not isinstance(raw, dict):
        return {"valid": False, "errors": ["task.yaml must be a YAML mapping"], "warnings": []}

    # 2. Required fields
    for field in ("name", "program_type", "correctness", "benchmark"):
        if field not in raw:
            errors.append(f"Missing required field: {field}")

    valid_types = ("python", "pytorch", "jax", "triton", "cpp", "cuda")
    pt = raw.get("program_type")
    if pt and pt not in valid_types:
        errors.append(f"Invalid program_type: {pt!r}. Must be one of: {valid_types}")

    # 3. Benchmark structure
    bench = raw.get("benchmark", {})
    if isinstance(bench, dict):
        if "cmd" not in bench:
            errors.append("benchmark.cmd is required")
        if "metric" not in bench:
            errors.append("benchmark.metric is required")
        else:
            metric = bench.get("metric", {})
            if isinstance(metric, dict):
                if "name" not in metric:
                    errors.append("benchmark.metric.name is required")
                mode = metric.get("mode", "maximize")
                if mode not in ("maximize", "minimize"):
                    errors.append(f"benchmark.metric.mode must be 'maximize' or 'minimize', got {mode!r}")

    # 4. Correctness
    corr = raw.get("correctness", {})
    if isinstance(corr, dict) and "cmd" not in corr:
        errors.append("correctness.cmd is required")

    # 5. Edit policy
    ep = raw.get("edit_policy", {})
    if isinstance(ep, dict):
        allowed = ep.get("allowed_paths", [])
        if not allowed:
            warnings.append("edit_policy.allowed_paths is empty — agent can edit any file (except blocklisted). Consider restricting to specific source files.")
        blocklisted = {"tests.py", "bench.py", "task.yaml"}
        for p in allowed:
            if p in blocklisted:
                errors.append(f"edit_policy.allowed_paths contains blocklisted file: {p}")

    # 6. Workspace and file existence
    ws = task_path.parent.resolve()
    if isinstance(ep, dict):
        for p in ep.get("allowed_paths", []):
            if not (ws / p).exists():
                warnings.append(f"edit_policy file does not exist yet: {p} (will need to create it)")

    # Check for bench.py and tests.py
    if not (ws / "bench.py").exists():
        warnings.append("bench.py not found in workspace directory")
    if not (ws / "tests.py").exists():
        warnings.append("tests.py not found in workspace directory")

    # 7. Build command for compiled languages
    if pt in ("cpp", "cuda") and not raw.get("build"):
        errors.append(f"program_type={pt!r} requires a build section with a compilation command")

    # 8. Contract validation
    contract_data = raw.get("contract", {})
    if isinstance(contract_data, dict):
        from perflab.task_spec import ContractSpec
        try:
            contract = ContractSpec(
                fixed_params=dict(contract_data.get("fixed_params", {})),
                min_repeats=int(contract_data.get("min_repeats", 1)),
                min_warmup=int(contract_data.get("min_warmup", 0)),
                required_bench_fields=list(contract_data.get("required_bench_fields", ["ok"])),
            )
            for err in contract.validate():
                errors.append(f"Contract: {err}")
        except Exception as exc:  # noqa: BLE001 -- validation tool must report any failure as a structured error, never crash
            errors.append(f"Contract parse error: {exc}")

    # 9. Full load attempt
    if not errors:
        try:
            from perflab.task_spec import TaskSpec
            TaskSpec.load(task_path)
        except Exception as exc:  # noqa: BLE001 -- validation tool must report any failure as a structured error, never crash
            errors.append(f"TaskSpec.load() failed: {exc}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "task_yaml": str(task_path),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def suggest_profilers(program_type: str, target_hardware: str | None = None) -> dict:
    """Suggest which profilers to use for a given program type and hardware.

    Returns a recommended profile_plan (always/optional lists) with rationale
    for each profiler. Copy the result into your task.yaml profile_plan section.

    Args:
        program_type: python | pytorch | jax | triton | cpp | cuda.
        target_hardware: Optional hardware (e.g. "NVIDIA A100", "Apple M2 Pro", "TPU v5e").
    """
    from perflab.server.task_templates import suggest_profilers as _suggest

    valid_types = ("python", "pytorch", "jax", "triton", "cpp", "cuda")
    if program_type not in valid_types:
        return {"error": f"Invalid program_type: {program_type!r}. Must be one of: {valid_types}"}

    return _suggest(program_type, target_hardware)


@mcp.tool(annotations={"readOnlyHint": True})
def suggest_thresholds(program_type: str, target_hardware: str | None = None) -> dict:
    """Suggest analysis thresholds for bottleneck detection based on program type and hardware.

    Returns recommended threshold overrides with descriptions. Copy into
    task.yaml's analysis_thresholds section. These are starting points —
    adjust after initial profiling.

    Args:
        program_type: python | pytorch | jax | triton | cpp | cuda.
        target_hardware: Optional hardware (e.g. "NVIDIA H100", "TPU v5e").
    """
    from perflab.server.task_templates import suggest_thresholds as _suggest

    valid_types = ("python", "pytorch", "jax", "triton", "cpp", "cuda")
    if program_type not in valid_types:
        return {"error": f"Invalid program_type: {program_type!r}. Must be one of: {valid_types}"}

    return _suggest(program_type, target_hardware)


@mcp.tool(annotations={"readOnlyHint": True})
def suggest_contract(task_yaml: str) -> dict:
    """Analyze a task's bench.py and suggest contract settings (fixed_params, required_bench_fields, min_repeats).

    Scans the benchmark script for problem dimension parameters and output
    fields, then recommends which should be locked down. This prevents the
    agent from gaming benchmarks by shrinking the problem or dropping metrics.

    Args:
        task_yaml: Path to task.yaml (bench.py is read from same directory).
    """
    from perflab.server.task_templates import suggest_contract_from_bench

    task_path = Path(task_yaml)
    if not task_path.exists():
        return {"error": f"File not found: {task_yaml}"}

    ws = task_path.parent.resolve()
    bench_path = ws / "bench.py"
    if not bench_path.exists():
        return {"error": f"bench.py not found in {ws}. Create it first."}

    import yaml as yaml_mod
    raw = yaml_mod.safe_load(task_path.read_text(encoding="utf-8"))
    program_type = raw.get("program_type", "python")

    content = bench_path.read_text(encoding="utf-8")
    return suggest_contract_from_bench(content, program_type)


@mcp.tool(annotations={"readOnlyHint": True})
def lint_bench_script(task_yaml: str) -> dict:
    """Check a task's bench.py for PerfLab protocol compliance.

    Validates that bench.py: accepts --json, writes JSON output, honors
    PERFLAB_BENCH_WARMUP/REPEATS env vars, includes "ok" field, and uses
    proper GPU synchronization if applicable.

    Returns errors (must fix) and warnings (should fix).

    Args:
        task_yaml: Path to task.yaml (bench.py is read from same directory).
    """
    from perflab.server.task_templates import lint_bench_script as _lint

    task_path = Path(task_yaml)
    if not task_path.exists():
        return {"error": f"File not found: {task_yaml}"}

    ws = task_path.parent.resolve()
    bench_path = ws / "bench.py"
    if not bench_path.exists():
        return {"error": f"bench.py not found in {ws}. Create it first."}

    content = bench_path.read_text(encoding="utf-8")
    return _lint(content)


# ===========================================================================
# Run management
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": True})
def list_runs(task: str | None = None, limit: int = 20, out_dir: str = "out") -> list[dict]:
    """List stored optimization runs (newest first)."""
    from perflab.memory.run_store import RunStore
    store = RunStore(Path(out_dir))
    return store.list_runs(task=task, limit=limit)


@mcp.tool(annotations={"readOnlyHint": True})
def get_run(run_id: str, out_dir: str = "out") -> dict:
    """Get full details for a single run (meta, report, bench, profiler summaries).

    For large runs this may be truncated — use get_run_section for specific data.
    """
    from perflab.memory.run_store import RunStore
    store = RunStore(Path(out_dir))
    return _guard_output_size(store.get_run(run_id))


@mcp.tool(annotations={"readOnlyHint": True})
def get_run_section(run_id: str, section: str, out_dir: str = "out") -> dict:
    """Get a specific section of a run to avoid output size limits.

    Sections: meta, report, bench, profiler_summaries, system_info, event_log,
    or a specific profiler name (torch, nsys, ncu, jax, metal, pyspy, perf,
    memray, power, ebpf, lock_contention, thread_sched).
    """
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    run_data = store.get_run(run_id)
    run_dir = Path(out_dir) / "runs" / run_id

    if section in ("meta", "report", "bench"):
        data = run_data.get(section)
        if data is None:
            return {"error": f"Section '{section}' not found in run {run_id}"}
        return _guard_output_size(data)

    if section == "system_info":
        path = run_dir / "system_info.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"error": "system_info.json not found"}

    if section == "event_log":
        path = run_dir / "agent_events.jsonl"
        if path.exists():
            events = []
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            return _guard_output_size({"events": events})
        return {"error": "agent_events.jsonl not found"}

    if section == "profiler_summaries":
        return _guard_output_size(run_data.get("profiler_summaries", {}))

    # Try as a specific profiler name
    summaries = run_data.get("profiler_summaries", {})
    for key, data in summaries.items():
        if section in key:
            return _guard_output_size(data)

    available = ["meta", "report", "bench", "system_info", "event_log", "profiler_summaries"] + list(summaries.keys())
    return {"error": f"Unknown section: '{section}'", "available_sections": available}


@mcp.tool(annotations={"readOnlyHint": True})
def compare_runs(run_a: str, run_b: str, out_dir: str = "out") -> dict:
    """Compare two runs: values, delta, speedup, bottleneck diff."""
    from perflab.memory.run_store import RunStore
    store = RunStore(Path(out_dir))
    return store.compare_runs(run_a, run_b)


@mcp.tool(annotations={"readOnlyHint": True})
def replay_run(run_id: str, out_dir: str = "out") -> dict:
    """Replay and summarize an agent run from its event log."""
    from perflab.optimizers.event_log import replay_events

    run_dir = Path(out_dir) / "runs" / run_id
    if not run_dir.exists():
        return {"error": f"Run directory not found: {run_dir}"}

    summary = replay_events(run_dir)
    return {"run_id": run_id, "summary": summary}


# ===========================================================================
# Analysis — on-demand from stored run data
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": True})
def get_bottlenecks(run_id: str, out_dir: str = "out") -> list[dict]:
    """Load profiler summaries for a run and diagnose bottlenecks."""
    from perflab.analyzers.bottleneck_analyzer import diagnose_bottlenecks
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    run_data = store.get_run(run_id)
    program_type = run_data.get("meta", {}).get("program_type", "python")
    summaries = run_data.get("profiler_summaries", {})

    if not summaries:
        return []

    device = run_data.get("meta", {}).get("device")
    # Load system_info for CPU count etc.
    run_dir = Path(out_dir) / "runs" / run_id
    mcp_system_info: dict | None = None
    system_info_path = run_dir / "system_info.json"
    if system_info_path.exists():
        try:
            mcp_system_info = json.loads(
                system_info_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            pass
    diags = diagnose_bottlenecks(summaries, program_type, device=device, system_info=mcp_system_info)
    return [
        {
            "rank": d.rank,
            "bottleneck": d.bottleneck,
            "root_cause": d.root_cause,
            "confidence": d.confidence,
            "suggested_actions": d.suggested_actions,
        }
        for d in diags
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def get_gpu_attribution(run_id: str, out_dir: str = "out") -> dict:
    """Compute GPU attribution ranking for a run: which kernels consume the most GPU time, CPU→GPU call graph, and pipeline stalls."""
    from perflab.analyzers.gpu_attribution import compute_attribution_ranking
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    run_data = store.get_run(run_id)
    summaries = run_data.get("profiler_summaries", {})

    nsys_summary = summaries.get("nsys") or summaries.get("nsys_profiler")
    if not nsys_summary:
        return {"error": "No NSys profiler data found for this run. GPU attribution requires CUDA workloads profiled with Nsight Systems."}

    perf_summary = summaries.get("linux_perf") or summaries.get("perf")
    entries = compute_attribution_ranking(nsys_summary, perf_summary)

    return _guard_output_size({
        "run_id": run_id,
        "attribution": _to_dicts(entries),
    })


@mcp.tool(annotations={"readOnlyHint": True})
def get_profile_diff(run_a: str, run_b: str, out_dir: str = "out") -> dict:
    """Compare profiler metrics between two runs: IPC, cache misses, GPU utilization, and function-level hotspot shifts."""
    from perflab.analyzers.profile_diff import compute_hotspot_diff, compute_profile_diff
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    data_a = store.get_run(run_a)
    data_b = store.get_run(run_b)

    summaries_a = data_a.get("profiler_summaries", {})
    summaries_b = data_b.get("profiler_summaries", {})

    if not summaries_a or not summaries_b:
        return {"error": "Both runs must have profiler summaries for comparison."}

    metric_mode = data_a.get("meta", {}).get("metric_mode", "maximize")
    deltas = compute_profile_diff(summaries_a, summaries_b, metric_mode=metric_mode)
    hotspots = compute_hotspot_diff(summaries_a, summaries_b)

    return _guard_output_size({
        "run_a": run_a,
        "run_b": run_b,
        "deltas": _to_dicts(deltas),
        "hotspot_shifts": _to_dicts(hotspots),
    })


@mcp.tool(annotations={"readOnlyHint": True})
def get_hlo_attribution(run_id: str, out_dir: str = "out") -> dict:
    """Compute HLO operation attribution for a JAX/TPU run: op rankings, cost estimates, dtype distribution, and optimization suggestions."""
    from perflab.analyzers.hlo_attribution import compute_hlo_attribution
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    run_data = store.get_run(run_id)
    summaries = run_data.get("profiler_summaries", {})

    jax_summary = summaries.get("jax") or summaries.get("jax_profiler")
    if not jax_summary:
        return {"error": "No JAX profiler data found for this run. HLO attribution requires JAX workloads."}

    result = compute_hlo_attribution(jax_summary)
    if result is None:
        return {"error": "Could not compute HLO attribution (no HLO operation data in JAX summary)."}

    return _guard_output_size(dataclasses.asdict(result))


@mcp.tool(annotations={"readOnlyHint": True})
def get_build_recommendations(task_yaml: str, run_id: str | None = None, out_dir: str = "out") -> dict:
    """Get build flag recommendations based on ISA detection and profiler data.

    Without run_id: static ISA-based recommendations from the build command.
    With run_id: also includes profiler-driven recommendations (cache miss rates, TMA data, etc.).
    """
    from perflab.analyzers.build_flags import recommend_build_flags, recommend_flags_from_profiling
    from perflab.task_spec import TaskSpec

    task = TaskSpec.load(Path(task_yaml))
    build_cmd = task.build.cmd if task.build else ""
    if not build_cmd:
        return {"error": "Task has no build command — build recommendations only apply to compiled languages (C++, CUDA)."}

    cpu_isa: dict = {}
    profiler_summaries: dict = {}

    if run_id:
        run_dir = Path(out_dir) / "runs" / run_id
        sys_path = run_dir / "system_info.json"
        if sys_path.exists():
            try:
                sys_info = json.loads(sys_path.read_text(encoding="utf-8"))
                cpu_isa = sys_info.get("cpu_isa", {})
            except (json.JSONDecodeError, OSError):
                pass

        from perflab.memory.run_store import RunStore
        store = RunStore(Path(out_dir))
        run_data = store.get_run(run_id)
        profiler_summaries = run_data.get("profiler_summaries", {})

    # Static ISA-based recommendations
    static_recs = recommend_build_flags(build_cmd, cpu_isa, task.program_type)

    # Profiler-driven recommendations (if run data available)
    dynamic_recs = []
    if profiler_summaries:
        dynamic_recs = recommend_flags_from_profiling(
            build_cmd, profiler_summaries, task.program_type, cpu_isa=cpu_isa,
        )

    return {
        "build_cmd": build_cmd,
        "program_type": task.program_type,
        "isa_recommendations": _to_dicts(static_recs),
        "profiler_recommendations": _to_dicts(dynamic_recs),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def get_roofline_analysis(run_id: str, out_dir: str = "out") -> dict:
    """Compute roofline analysis for a run: arithmetic intensity, achieved TFLOPS, peak utilization %, and memory bandwidth."""
    from perflab.memory.run_store import RunStore
    from perflab.reporting.roofline import compute_roofline_point
    from perflab.roofline_peaks import infer_peaks

    store = RunStore(Path(out_dir))
    run_data = store.get_run(run_id)

    bench = run_data.get("bench", {})
    if not bench:
        return {"error": "No benchmark data found for this run."}

    summaries = run_data.get("profiler_summaries", {})

    # Try to get profiler FLOPS
    profiler_flops = None
    for key in ("torch", "pytorch_profiler", "torch_profiler"):
        s = summaries.get(key, {})
        if s.get("total_flops"):
            profiler_flops = s["total_flops"]
            break

    # Try to get measured DRAM bytes from NCU
    measured_dram = None
    for key in ("ncu", "ncu_profiler"):
        s = summaries.get(key, {})
        if s.get("dram_bytes"):
            measured_dram = s["dram_bytes"]
            break

    point = compute_roofline_point(bench, measured_dram_bytes=measured_dram, profiler_flops=profiler_flops)
    if point is None:
        return {"error": "Could not compute roofline point (bench.json lacks flops/bytes data or meta.M/N/K)."}

    result: dict = {
        "run_id": run_id,
        "roofline_point": dataclasses.asdict(point),
    }

    peaks = infer_peaks("auto")
    if peaks:
        result["peaks"] = {
            "device": peaks.device,
            "source": peaks.source,
            "peak_tflops": peaks.peak_tflops,
            "peak_mem_bw_gbs": peaks.peak_mem_bw_gbs,
        }
        if peaks.dtype_peaks:
            result["peaks"]["dtype_peaks"] = peaks.dtype_peaks
        if peaks.peak_tflops > 0 and point.tflops > 0:
            result["pct_of_peak"] = round(point.tflops / peaks.peak_tflops * 100, 1)

    return result


@mcp.tool(annotations={"readOnlyHint": True})
def get_thresholds(task_yaml: str | None = None) -> dict:
    """List analysis thresholds used for bottleneck diagnosis. Shows defaults and task-specific overrides."""
    from perflab.analyzers.bottleneck_analyzer import AnalysisThresholds

    defaults = AnalysisThresholds()
    effective = defaults

    if task_yaml:
        from perflab.task_spec import TaskSpec
        task = TaskSpec.load(Path(task_yaml))
        effective = task.analysis_thresholds

    fields_data = {}
    for f in dataclasses.fields(AnalysisThresholds):
        val = getattr(effective, f.name)
        def_val = getattr(defaults, f.name)
        entry: dict = {"value": val}
        if val != def_val:
            entry["default"] = def_val
            entry["overridden"] = True
        fields_data[f.name] = entry

    overridden = sum(1 for v in fields_data.values() if v.get("overridden"))
    return {
        "thresholds": fields_data,
        "total_fields": len(fields_data),
        "overridden_count": overridden,
    }


# ===========================================================================
# Environment
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": True})
def get_peaks(target: str = "auto", cuda_index: int | None = None) -> dict:
    """Show inferred roofline peaks and detected hardware devices (CUDA GPUs, Metal/MPS GPUs, TPU)."""
    from perflab.roofline_peaks import infer_peaks, list_cuda_gpus, list_metal_gpus, selection_hints

    result: dict = {
        "target": target,
        "cuda_gpus": list_cuda_gpus(),
        "metal_gpus": list_metal_gpus(),
    }

    peaks = infer_peaks(target, preferred_cuda_index=cuda_index)
    if peaks:
        result["peaks"] = {
            "device": peaks.device,
            "source": peaks.source,
            "peak_tflops": peaks.peak_tflops,
            "peak_mem_bw_gbs": peaks.peak_mem_bw_gbs,
        }
        if peaks.dtype_peaks:
            result["peaks"]["dtype_peaks"] = peaks.dtype_peaks
    else:
        result["peaks"] = None

    result["hints"] = selection_hints()
    return result


@mcp.tool(annotations={"readOnlyHint": True})
def doctor_check(check_profilers: bool = True, check_llm: bool = True, check_all: bool = False) -> dict:
    """Check environment readiness: Python version, packages, profiler tools, hardware detection, LLM config."""
    from perflab.doctor import run_doctor

    results = run_doctor(check_profilers=check_profilers, check_llm=check_llm, check_all=check_all)

    checks = [
        {"name": r.name, "status": r.status, "message": r.message}
        for r in results
    ]
    passes = sum(1 for r in results if r.status == "pass")
    warns = sum(1 for r in results if r.status == "warn")
    fails = sum(1 for r in results if r.status == "fail")

    return {
        "checks": checks,
        "summary": {"passed": passes, "warnings": warns, "failures": fails},
        "ready": fails == 0,
    }


# ===========================================================================
# CI regression checks
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
def ci_check(task_yaml: str, baseline_file: str | None = None) -> dict:
    """Run a CI regression check: benchmark current code against a saved baseline.

    Returns pass/fail, regression %, tolerance %, bench variance warnings,
    and profiler metric regressions (when NCU data is available in both
    the baseline and a recent profile run).
    """
    from perflab.ci import run_ci_check
    from perflab.task_spec import TaskSpec

    task = TaskSpec.load(Path(task_yaml))
    bp = Path(baseline_file) if baseline_file else None
    result = run_ci_check(task, bp)
    return result.to_dict()


@mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
def save_ci_baseline(task_yaml: str, baseline_file: str | None = None) -> dict:
    """Run benchmark and save result as CI baseline for future regression checks.

    Automatically includes NCU profiler data from the most recent profile
    run (if available) for future profiler regression detection.
    """
    from perflab.ci import save_baseline
    from perflab.task_spec import TaskSpec

    task = TaskSpec.load(Path(task_yaml))
    bp = Path(baseline_file) if baseline_file else None
    saved_path = save_baseline(task, bp)
    return {"baseline_saved": str(saved_path)}


# ===========================================================================
# Optimization — profiling and agent runs
# ===========================================================================

@mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
def profile_task(task_yaml: str) -> dict:
    """Run baseline profiling for a task. Returns profiler summary paths."""
    from perflab.orchestrator import profile_only
    from perflab.task_spec import TaskSpec

    task_file = Path(task_yaml)
    task = TaskSpec.load(task_file)
    run_dir = profile_only(task)

    # Collect profiler summaries
    summaries: dict[str, dict] = {}
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists():
        for p in artifacts_dir.glob("*_summary.json"):
            try:
                summaries[p.stem.replace("_summary", "")] = json.loads(
                    p.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                pass

    return _guard_output_size({
        "run_dir": str(run_dir),
        "profiler_summaries": summaries,
    })


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def start_agent(
    task_yaml: str,
    iters: int = 8,
    candidates: int = 4,
    suggest: str | None = None,
) -> dict:
    """Launch an agent optimization run in the background. Returns a job_id for tracking.

    Requires a configured LLM provider (run `perflab init` first).
    Use `get_agent_progress` to poll for status. Preferred for production use
    — non-blocking, full token tracking, progress polling.
    """
    job_id = uuid.uuid4().hex[:12]

    def _run() -> None:
        from perflab.llm.config import LLMConfig
        from perflab.optimizers.agent import AgentConfig, run_agent
        from perflab.optimizers.progress import ListProgress
        from perflab.task_spec import TaskSpec

        progress = ListProgress()
        with _lock:
            _active_runs[job_id]["progress"] = progress
            _active_runs[job_id]["status"] = "running"

        acquired = _agent_lock.acquire(timeout=0)
        if not acquired:
            with _lock:
                _active_runs[job_id]["status"] = "failed"
                _active_runs[job_id]["error"] = "Another agent run is already in progress."
            return

        try:
            task_file = Path(task_yaml)
            task = TaskSpec.load(task_file)
            llm_config = LLMConfig.load()
            config = AgentConfig(
                n_candidates=candidates,
                max_iters=iters,
            )
            result = run_agent(
                task, task_file, config, llm_config,
                expert_suggestion=suggest,
                progress=progress,
            )
            with _lock:
                _active_runs[job_id]["status"] = "completed"
                _active_runs[job_id]["result"] = {
                    "best_value": result.best_value,
                    "best_iter": result.best_iter,
                    "baseline_value": result.baseline_value,
                    "run_dir": str(result.run_dir),
                }
        except Exception as exc:  # noqa: BLE001 -- top-level safety net for a background job; any failure must be reported, not crash the thread
            with _lock:
                _active_runs[job_id]["status"] = "failed"
                _active_runs[job_id]["error"] = str(exc)
        finally:
            _agent_lock.release()

    with _lock:
        _active_runs[job_id] = {"status": "starting", "progress": None}

    _executor.submit(_run)
    return {"job_id": job_id, "status": "starting"}


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def optimize_task(
    task_yaml: str,
    iters: int = 8,
    candidates: int = 4,
    suggest: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Run optimization using the client's own LLM via MCP sampling (no API key needed).

    This tool uses your MCP client's LLM to generate optimization candidates.
    No `perflab init` or API key is required.

    Caveats compared to start_agent:
    - Blocks the MCP connection for the duration of the run (may be several minutes)
    - Token usage statistics are unavailable
    - Requires your MCP client to support the sampling protocol
    """
    from perflab.llm.base import Message as PerfLabMessage
    from perflab.llm.config import LLMConfig
    from perflab.llm.mcp_sampling_provider import MCPSamplingProvider
    from perflab.optimizers.agent import AgentConfig, run_agent
    from perflab.optimizers.progress import ListProgress
    from perflab.task_spec import TaskSpec

    if ctx is None:
        return {"error": "No MCP context available. This tool must be invoked by an MCP client."}

    # Pre-flight: verify sampling works
    try:
        preflight = await ctx.sample("Respond with OK.", max_tokens=16)
        if not preflight.text:
            return {"error": "MCP sampling pre-flight returned empty response. Your client may not support sampling."}
    except Exception as exc:  # noqa: BLE001 -- feature-detection probe against an arbitrary MCP client, report any failure as unsupported
        return {"error": f"MCP sampling not supported by your client: {exc}"}

    # Concurrency guard
    if not _agent_lock.acquire(timeout=0):
        return {"error": "Another agent run is already in progress. Wait for it to finish."}

    try:
        loop = asyncio.get_running_loop()

        async def sample_fn(
            *,
            messages: list[PerfLabMessage],
            system_prompt: str | None = None,
            temperature: float = 0.7,
            max_tokens: int = 4096,
        ) -> str:
            # Convert PerfLab Messages to the format ctx.sample() expects
            from fastmcp.server.context import SamplingMessage
            from mcp.types import TextContent

            sampling_msgs = []
            for m in messages:
                # System content arrives via system_prompt, so anything else maps to user
                role: Literal["user", "assistant"] = (
                    "assistant" if m.role == "assistant" else "user"
                )
                sampling_msgs.append(
                    SamplingMessage(
                        role=role,
                        content=TextContent(type="text", text=m.content),
                    )
                )

            result = await ctx.sample(
                messages=sampling_msgs,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result.text or ""

        provider = MCPSamplingProvider(
            name="mcp-sampling",
            _sample_fn=sample_fn,
            _loop=loop,
        )

        llm_config = LLMConfig(
            provider="mcp-sampling",
            model="client-llm",
        )

        task_file = Path(task_yaml)
        task = TaskSpec.load(task_file)
        progress = ListProgress()
        config = AgentConfig(
            n_candidates=candidates,
            max_iters=iters,
        )

        result = await asyncio.to_thread(
            run_agent,
            task, task_file, config, llm_config,
            expert_suggestion=suggest,
            progress=progress,
            provider=provider,
        )

        return {
            "status": "completed",
            "best_value": result.best_value,
            "best_iter": result.best_iter,
            "baseline_value": result.baseline_value,
            "run_dir": str(result.run_dir),
            "messages": progress.messages[-20:],
        }
    except Exception as exc:  # noqa: BLE001 -- top-level safety net for the MCP tool call; any failure must be reported, not crash the connection
        return {"status": "failed", "error": str(exc)}
    finally:
        _agent_lock.release()


@mcp.tool(annotations={"readOnlyHint": True})
def get_agent_progress(job_id: str) -> dict:
    """Check status and recent progress messages for a background agent run."""
    with _lock:
        run_info = _active_runs.get(job_id)
        if run_info is None:
            return {"error": f"Unknown job_id: {job_id}"}

        status = run_info["status"]
        messages: list[str] = []
        progress = run_info.get("progress")
        if progress is not None:
            messages = progress.messages[-20:]

        result: dict = {
            "job_id": job_id,
            "status": status,
            "recent_messages": messages,
        }
        if "result" in run_info:
            result["result"] = run_info["result"]
        if "error" in run_info:
            result["error"] = run_info["error"]
        return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
