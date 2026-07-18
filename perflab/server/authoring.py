"""MCP tools for task inspection, authoring, and validation."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from perflab.server.core import _PROGRAM_TYPES, _guard_output_size, mcp

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

    if program_type not in _PROGRAM_TYPES:
        return {"error": f"Invalid program_type: {program_type!r}. Must be one of: {_PROGRAM_TYPES}"}
    if metric_mode not in ("maximize", "minimize"):
        return {"error": f"Invalid metric_mode: {metric_mode!r}. Must be 'maximize' or 'minimize'"}

    for field_name, value in (("name", name), ("category", category)):
        if not value or value in (".", "..") or value != Path(value).name or value.startswith("."):
            return {"error": f"Invalid {field_name}: {value!r} — must be a plain directory name (no separators or '..')"}

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

    pt = raw.get("program_type")
    if pt and pt not in _PROGRAM_TYPES:
        errors.append(f"Invalid program_type: {pt!r}. Must be one of: {_PROGRAM_TYPES}")

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

    if program_type not in _PROGRAM_TYPES:
        return {"error": f"Invalid program_type: {program_type!r}. Must be one of: {_PROGRAM_TYPES}"}

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

    if program_type not in _PROGRAM_TYPES:
        return {"error": f"Invalid program_type: {program_type!r}. Must be one of: {_PROGRAM_TYPES}"}

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
