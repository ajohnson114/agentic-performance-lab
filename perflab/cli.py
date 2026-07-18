from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import typer
import yaml

from perflab.analyzers.bottleneck_analyzer import AnalysisThresholds
from perflab.doctor import run_doctor
from perflab.llm.config import DEFAULT_MODEL, PROVIDER_DEFAULT_MODELS
from perflab.orchestrator import optimize, profile_only
from perflab.roofline_peaks import (
    cache_path,
    infer_peaks,
    list_cuda_gpus,
    list_metal_gpus,
    selection_hints,
)
from perflab.task_spec import TaskSpec

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _setup(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Agentic Performance Lab — profile, diagnose, and optimize compute-bound programs."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s",
        level=level,
    )


def _file_uri(p: Path) -> str:
    """Return a file:// URI for a path — clickable in most terminals."""
    return f"file://{p.resolve()}"


def _hyperlink(uri: str, label: str) -> str:
    """Wrap *label* in an OSC 8 hyperlink so the terminal makes it clickable."""
    return f"\033]8;;{uri}\033\\{label}\033]8;;\033\\"


def _echo_run_links(run_dir: Path) -> None:
    """Print clickable links to the run's dashboard and report."""
    report_uri = _file_uri(run_dir / "report.md")
    dashboard_uri = _file_uri(run_dir / "dashboard.html")

    if sys.stdout.isatty():
        # Interactive terminal — use OSC 8 hyperlinks for reliable clicking.
        sys.stdout.write(f"Report:    {_hyperlink(report_uri, report_uri)}\n")
        sys.stdout.write(f"Dashboard: {_hyperlink(dashboard_uri, dashboard_uri)}\n")
        sys.stdout.flush()
    else:
        # Piped / captured — plain file:// URIs (no escape sequences).
        typer.echo(f"Report:    {report_uri}")
        typer.echo(f"Dashboard: {dashboard_uri}")


def _load_task(task_yaml: str) -> TaskSpec:
    """Load a task file with a clean CLI error instead of a raw traceback."""
    task_file = Path(task_yaml)
    if not task_file.exists():
        typer.echo(f"Error: task file not found: {task_file}")
        raise typer.Exit(code=2)
    return TaskSpec.load(task_file)


@app.command()
def profile(task_yaml: str):
    """Run correctness + benchmark + generate profiles (no code/knob search loop)."""
    task = _load_task(task_yaml)
    run_dir = profile_only(task)
    _echo_run_links(run_dir)

@app.command(name="optimize")
def optimize_cmd(
    task_yaml: str,
    iters: int = typer.Option(None, help="Max optimization iterations (overrides task.yaml max_iters)"),
    max_trials: int = typer.Option(None, "--max-trials", help="Max grid search trials (random sample if grid exceeds this)"),
):
    """Grid search over tuning.yaml knobs to find the best configuration.

    Define a ``sweep`` section in tuning.yaml to specify the search space.
    Without a sweep section, falls back to the legacy hardcoded knob sweep.
    """
    task = _load_task(task_yaml)
    run_dir = optimize(task, iters=iters, max_trials=max_trials)
    _echo_run_links(run_dir)


@app.command()
def peaks(
    target: str = typer.Option("auto", "--target", help="Hardware target: auto|cuda|mps|cpu"),
    list_devices: bool = typer.Option(True, "--list/--no-list", help="List detected devices"),
    cuda_index: int = typer.Option(None, "--cuda-index", help="Preferred CUDA GPU index for peak inference"),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass cached calibration results"),
):
    """Show inferred roofline peaks and detected devices."""
    if refresh:
        os.environ["PERFLAB_PEAKS_NO_CACHE"] = "1"

    typer.echo(f"PerfLab peaks (target={target})")
    typer.echo(f"Cache: {cache_path()}" + (" (bypassed)" if refresh else ""))

    if list_devices:
        cuda = list_cuda_gpus()
        if cuda:
            typer.echo("\nCUDA GPUs:")
            for g in cuda:
                if "raw" in g:
                    typer.echo(f"  - {g['raw']}")
                else:
                    typer.echo(f"  - idx={g.get('index','?')} name={g.get('name','')} cc={g.get('compute_cap','')} memMiB={g.get('memory_total_mib','')}")
        else:
            typer.echo("\nCUDA GPUs: (none detected via nvidia-smi)")

        metal = list_metal_gpus()
        if metal:
            typer.echo("\nMetal/MPS GPUs (system_profiler):")
            for i, d in enumerate(metal):
                typer.echo(f"  - [{i}] {d.get('name','GPU')} type={d.get('type','')} vendor={d.get('vendor','')}")
        else:
            typer.echo("\nMetal/MPS GPUs: (none / not applicable)")

    p = infer_peaks(target, preferred_cuda_index=cuda_index)
    if not p:
        typer.echo("\nInferred peaks: (unavailable)")
        hints = selection_hints()
        typer.echo("\nHints:")
        typer.echo(f"  CUDA:  {hints.get('cuda')}")
        typer.echo(f"  MPS:   {hints.get('mps')}")
        typer.echo(f"  Cache: {hints.get('cache')}")
        raise typer.Exit(code=1)

    typer.echo("\nInferred peaks:")
    typer.echo(f"  device: {p.device}")
    typer.echo(f"  source: {p.source}")
    typer.echo(f"  peak_tflops: {p.peak_tflops:.3f}")
    typer.echo(f"  peak_mem_bw_gbs: {p.peak_mem_bw_gbs:.1f}")

    hints = selection_hints()
    typer.echo("\nHints:")
    typer.echo(f"  CUDA:  {hints.get('cuda')}")
    typer.echo(f"  MPS:   {hints.get('mps')}")
    typer.echo(f"  Cache: {hints.get('cache')}")


@app.command()
def doctor(
    profilers: bool = typer.Option(True, "--profilers/--no-profilers", help="Check profiler tools"),
    llm: bool = typer.Option(True, "--llm/--no-llm", help="Check LLM provider config"),
    all_checks: bool = typer.Option(False, "--all", help="Run all checks"),
):
    """Check environment readiness: Python, packages, profiler tools, LLM config."""
    results = run_doctor(check_profilers=profilers, check_llm=llm, check_all=all_checks)

    status_icons = {"pass": "OK", "fail": "FAIL", "warn": "WARN"}
    any_fail = False

    for r in results:
        icon = status_icons.get(r.status, "?")
        typer.echo(f"  [{icon:>4}] {r.name}: {r.message}")
        if r.status == "fail":
            any_fail = True

    passes = sum(1 for r in results if r.status == "pass")
    warns = sum(1 for r in results if r.status == "warn")
    fails = sum(1 for r in results if r.status == "fail")
    typer.echo(f"\n  {passes} passed, {warns} warnings, {fails} failures")

    if any_fail:
        raise typer.Exit(code=1)


_PROVIDER_DEFAULTS = PROVIDER_DEFAULT_MODELS


@app.command()
def init(
    scrub_key: bool = typer.Option(
        False, "--scrub-key",
        help="Remove a legacy api_key from the config file and exit (no interactive setup)",
    ),
):
    """Interactive first-run setup for LLM configuration."""
    config_path = Path.home() / ".config" / "perflab" / "config.yaml"

    if scrub_key:
        from perflab.llm.config import scrub_api_key
        if scrub_api_key(config_path):
            typer.echo(f"Removed api_key from {config_path}.")
            typer.echo("Set it via: export PERFLAB_API_KEY=sk-...")
        else:
            typer.echo(f"No api_key found in {config_path} -- nothing to do.")
        raise typer.Exit(code=0)

    if config_path.exists():
        overwrite = typer.confirm(f"Config already exists at {config_path}. Overwrite?", default=False)
        if not overwrite:
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    providers = list(_PROVIDER_DEFAULTS.keys())
    typer.echo("\nAvailable LLM providers:")
    for i, p in enumerate(providers, 1):
        typer.echo(f"  {i}. {p}")

    choice = typer.prompt("Select provider (1-3)", type=int)
    if choice < 1 or choice > len(providers):
        typer.echo(f"Invalid choice. Must be between 1 and {len(providers)}.")
        raise typer.Exit(code=1)

    provider = providers[choice - 1]

    default_model = _PROVIDER_DEFAULTS[provider]
    model = typer.prompt("Model", default=default_model, type=str).strip()

    api_key = ""
    api_base = ""
    if provider == "ollama":
        typer.echo("Ollama URL (press Enter for http://localhost:11434)")
        api_base = typer.prompt("URL", default="http://localhost:11434", type=str).strip()
    else:
        api_key = typer.prompt("API key", hide_input=True, type=str).strip()
        if not api_key:
            typer.echo("API key is required for non-ollama providers.")
            raise typer.Exit(code=1)

    # Check if the provider SDK is installed; hint if not
    _SDK_PACKAGES = {"openai": "openai", "anthropic": "anthropic"}
    sdk_pkg = _SDK_PACKAGES.get(provider)
    if sdk_pkg:
        try:
            __import__(sdk_pkg)
        except ImportError:
            typer.echo(f"\nNote: the '{sdk_pkg}' package is not installed.")
            typer.echo(f'  Install it with:  pip install -e ".[{provider}]"')

    # api_key is intentionally never written to config_data -- it is only ever
    # read from the PERFLAB_API_KEY env var (see perflab/llm/config.py).
    config_data = {
        "llm": {
            "provider": provider,
            "model": model,
        }
    }
    if api_base:
        config_data["llm"]["api_base"] = api_base

    # Validate model with a live API call before writing config
    typer.echo("\nValidating model...")
    try:
        from perflab.llm.base import Message
        from perflab.llm.config import LLMConfig, create_provider

        test_config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
        )
        test_provider = create_provider(test_config)
        test_provider.complete(
            [Message(role="user", content="Respond with OK.")],
            max_tokens=5,
        )
        typer.echo("  Model validated successfully.")
    except ImportError:
        typer.echo("  Skipped (SDK not installed).")
    except Exception as exc:  # noqa: BLE001 -- best-effort validation call against an arbitrary LLM provider
        err_msg = str(exc)
        typer.echo(f"  Model validation failed: {err_msg}")
        if not typer.confirm("Save this configuration anyway?", default=False):
            typer.echo("Aborted.")
            raise typer.Exit(code=1) from None

    from perflab.llm.config import _secure_write
    _secure_write(config_path, yaml.dump(config_data, default_flow_style=False))
    typer.echo(f"\nConfig written to {config_path} (permissions: owner read/write only)")

    if api_key:
        typer.echo(
            "\nAPI keys are never stored on disk. Add this to your shell profile:\n"
            "  export PERFLAB_API_KEY=sk-...  (the key you just entered)"
        )

    typer.echo(
        "\nTip: If you're using an MCP client (Claude Desktop, Cursor), the optimize_task\n"
        "tool can use your client's LLM directly — no API key needed."
    )


@app.command(name="show-config")
def show_config():
    """Show the resolved PerfLab configuration (all layers merged)."""
    from perflab.config import (
        _USER_CONFIG_PATH,
        _find_project_config,
        load_config,
    )

    cfg = load_config(force_reload=True)
    typer.echo("Resolved PerfLab configuration")
    typer.echo("=" * 40)
    typer.echo(f"User config:    {_USER_CONFIG_PATH}" + (" (found)" if _USER_CONFIG_PATH.exists() else " (not found)"))
    project = _find_project_config()
    typer.echo(f"Project config: {project or '(not found)'}")
    typer.echo("Priority: env vars > project > user > defaults\n")

    import json
    typer.echo(json.dumps(cfg.to_dict(), indent=2))

    typer.echo("\nTo create a config file:  perflab init")
    typer.echo("Full template: perflab show-config --template")


@app.command(name="show-config-template")
def show_config_template():
    """Print the full default configuration YAML template."""
    from perflab.config import DEFAULT_CONFIG_TEMPLATE
    typer.echo(DEFAULT_CONFIG_TEMPLATE)


@app.command(name="init-config")
def init_config(
    project: bool = typer.Option(True, "--project/--user", help="Create project-level (./perflab.yaml) or user-level (~/.config/perflab/config.yaml)"),
):
    """Create a config file with the default template.

    By default creates ./perflab.yaml (project-level). Use --user for ~/.config/perflab/config.yaml.
    """
    from perflab.config import create_project_config, create_user_config

    if project:
        target = Path.cwd() / "perflab.yaml"
        if target.exists():
            if not typer.confirm(f"{target} already exists. Overwrite?", default=False):
                raise typer.Exit(code=0)
        path = create_project_config()
    else:
        from perflab.config import _USER_CONFIG_PATH
        if _USER_CONFIG_PATH.exists():
            if not typer.confirm(f"{_USER_CONFIG_PATH} already exists. Overwrite?", default=False):
                raise typer.Exit(code=0)
        path = create_user_config()

    typer.echo(f"Created {path}")
    typer.echo("Edit it to change settings — only uncomment what you need.")


@app.command(name="ci-check")
def ci_check(
    task_yaml: str = typer.Argument(..., help="Path to task YAML"),
    save_baseline: bool = typer.Option(False, "--save-baseline", help="Run benchmark and save as baseline"),
    baseline_file: str = typer.Option(None, "--baseline-file", help="Path to baseline JSON file"),
    tolerance: float = typer.Option(
        None, "--tolerance",
        help="Override the task's regression tolerance (fraction, e.g. 0.15 = 15%) — for noisy CI runners",
    ),
):
    """Run a CI regression check against a saved baseline."""
    from perflab.ci import run_ci_check
    from perflab.ci import save_baseline as save_bl

    task = _load_task(task_yaml)
    bp = Path(baseline_file) if baseline_file else None

    if save_baseline:
        saved = save_bl(task, bp)
        typer.echo(f"Baseline saved to {saved}")
        raise typer.Exit(code=0)

    result = run_ci_check(task, bp, tolerance=tolerance)
    typer.echo(json.dumps(result.to_dict(), indent=2))

    # Surface advisory warnings
    if result.bench_variance_warnings:
        for w in result.bench_variance_warnings:
            typer.echo(f"  WARNING (bench variance): {w}")
    if result.profiler_regressions:
        for r in result.profiler_regressions:
            typer.echo(f"  WARNING (profiler): {r.metric} {r.direction} ({r.baseline:.1f} -> {r.current:.1f})")

    if result.passed:
        typer.echo("CI check PASSED")
    else:
        reasons = []
        if result.regression_pct is not None and result.baseline_value is not None:
            from perflab.ci import _check_regression
            _, primary_regressed = _check_regression(
                result.current_value, result.baseline_value,
                result.metric_mode, result.tolerance_pct / 100,
            )
            if primary_regressed:
                reasons.append(f"{result.metric_name}: regression of {result.regression_pct:.1f}%")
        if result.secondary and result.secondary.regressed:
            reasons.append(f"{result.secondary.name}: regression of {result.secondary.regression_pct:.1f}%")
        detail = "; ".join(reasons) if reasons else f"regression of {result.regression_pct:.1f}%"
        typer.echo(f"CI check FAILED: {detail} (tolerance: {result.tolerance_pct:.1f}%)")
        raise typer.Exit(code=1)


@app.command()
def agent(
    task_yaml: str = typer.Argument(..., help="Path to task YAML"),
    iters: int = typer.Option(None, "--iters", help="Max agent iterations"),
    candidates: int = typer.Option(None, "--candidates", help="Candidates per iteration"),
    suggest: str = typer.Option(None, "--suggest", help="Expert optimization hint for the LLM agent"),
    no_early_stop: bool = typer.Option(False, "--no-early-stop", help="Disable early stopping / convergence detection"),
    fast_screen: bool = typer.Option(True, "--fast-screen/--no-fast-screen", help="Use fast benchmark screening for candidates"),
    max_time: int = typer.Option(3600, "--max-time", help="Wall-clock budget in seconds"),
    isolation: str = typer.Option(
        None, "--isolation",
        help="Sandbox candidate execution: none|restricted|strict (default: config/none)",
    ),
):
    """Run the agentic LLM-driven optimizer with beam search."""
    from perflab.llm.config import LLMConfig
    from perflab.optimizers.agent import AgentConfig, run_agent
    from perflab.tools.isolation import default_level_for_host, resolve_policy

    task_file = Path(task_yaml)
    task = _load_task(task_yaml)
    llm_config = LLMConfig.load()

    if not llm_config.is_configured():
        typer.echo(
            "Error: LLM is not configured. No API key or model found.\n\n"
            "Run 'perflab init' for interactive setup, or set environment variables:\n"
            "  export PERFLAB_LLM_PROVIDER=openai\n"
            f"  export PERFLAB_LLM_MODEL={DEFAULT_MODEL}\n"
            "  export PERFLAB_API_KEY=sk-...\n\n"
            "If you're using an MCP client, the optimize_task tool can use your\n"
            "client's LLM directly without an API key."
        )
        raise typer.Exit(code=1)

    # Resolution: CLI flags > task.yaml agent: > perflab.yaml agent: > AgentConfig defaults
    from perflab.config import load_config
    global_cfg = load_config()
    ga = global_cfg.agent  # global agent defaults

    # Isolation (Fix 2b): CLI flag > task.yaml `isolation.level` > perflab.yaml/
    # user config > compiled-in default ("none"). The resolved policy is passed
    # through AgentConfig into every candidate benchmark/correctness subprocess
    # (perflab/optimizers/phases/{prescreen,evaluate,autotune}.py and the
    # baseline pipeline, so sandbox overhead cancels out of speedup comparisons).
    isolation_policy = resolve_policy(task_file, global_cfg.isolation.level, cli_level=isolation)
    if isolation_policy is None:
        typer.echo("Candidate code runs unsandboxed on this host (isolation: none).")
    elif default_level_for_host() == "none":
        # bwrap unavailable (non-Linux, or user namespaces disabled):
        # wrap_command will fall back per-call, so say it loudly up front.
        typer.echo(
            f"WARNING: isolation '{isolation_policy.level}' requested, but this host "
            "cannot sandbox (bwrap unavailable or not Linux). Candidate code "
            "will run UNSANDBOXED (rlimits/env-allowlist only)."
        )
    else:
        typer.echo(f"Isolation: {isolation_policy.level} (bwrap sandbox active for candidate runs).")

    config = AgentConfig(
        n_candidates=candidates or task.agent.n_candidates or ga.n_candidates,
        top_k=task.agent.top_k,
        max_iters=iters or task.agent.max_iters or ga.max_iters,
        early_stop=not no_early_stop,
        fast_screen=fast_screen,
        max_wall_time_s=max_time or ga.max_wall_time_s,
        isolation=isolation_policy,
    )

    result = run_agent(task, task_file, config, llm_config, expert_suggestion=suggest)
    typer.echo(f"Best {task.benchmark.metric.name}: {result.best_value:.6g} (iter {result.best_iter})")
    _echo_run_links(result.run_dir)


@app.command()
def replay(
    run_dir: str = typer.Argument(..., help="Path to a completed agent run directory"),
):
    """Replay and summarize an agent run from its event log."""
    from perflab.optimizers.event_log import replay_events

    rd = Path(run_dir)
    if not rd.exists():
        typer.echo(f"Run directory not found: {rd}")
        raise typer.Exit(code=1)
    output = replay_events(rd)
    typer.echo(output)


@app.command(name="list-runs")
def list_runs_cmd(
    task: str = typer.Option(None, "--task", help="Filter by task name"),
    limit: int = typer.Option(20, "--limit", help="Max runs to display"),
    out_dir: str = typer.Option("out", "--out-dir", help="Output root directory"),
):
    """List stored optimization runs (newest first)."""
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    runs = store.list_runs(task=task, limit=limit)
    if not runs:
        typer.echo("No runs found.")
        raise typer.Exit(code=0)

    for r in runs:
        status = r.get("status", "?")
        best = r.get("best_value")
        best_str = f"  best={best:.6g}" if best is not None else ""
        ptype = r.get("program_type", "")
        ptype_str = f"  type={ptype}" if ptype else ""
        typer.echo(f"  {r['run_id']}  task={r.get('task','?')}  status={status}{best_str}{ptype_str}")


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="First run ID"),
    run_b: str = typer.Argument(..., help="Second run ID"),
    out_dir: str = typer.Option("out", "--out-dir", help="Output root directory"),
):
    """Compare two optimization runs side by side."""
    from perflab.memory.run_store import RunStore

    store = RunStore(Path(out_dir))
    try:
        result = store.compare_runs(run_a, run_b)
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    # Header: task and metric context
    if result.get("task_name"):
        typer.echo(f"Task: {result['task_name']}")
    metric_name = result.get("metric_name", "metric")
    metric_mode = result.get("metric_mode")
    if metric_name or metric_mode:
        parts = [p for p in [metric_name, metric_mode] if p]
        typer.echo(f"Metric: {' '.join(parts)}")
    typer.echo("")

    # Run details
    status_a = result.get("status_a", "?")
    status_b = result.get("status_b", "?")
    typer.echo(f"Run A (old): {result['run_a']}  ({status_a})")
    typer.echo(f"Run B (new): {result['run_b']}  ({status_b})")
    typer.echo("")

    # Values and comparison
    va = result["value_a"]
    vb = result["value_b"]
    typer.echo(f"Value A: {va:.6g}" if va is not None else "Value A: N/A")
    typer.echo(f"Value B: {vb:.6g}" if vb is not None else "Value B: N/A")
    if result["delta"] is not None:
        typer.echo(f"Delta (new-old): {result['delta']:+.6g}")
    if result["ratio"] is not None:
        ratio = result["ratio"]
        # For maximize metrics, ratio > 1 is good; for minimize, ratio < 1 is good
        if metric_mode == "minimize":
            label = "Speedup" if ratio < 1 else "Slowdown"
            factor = 1 / ratio if ratio != 0 else 0
        else:
            label = "Improvement" if ratio > 1 else "Regression"
            factor = ratio
        typer.echo(f"Ratio (new/old): {ratio:.2f}x  ({label}: {factor:.2f}x)")

    if result["resolved_bottlenecks"]:
        typer.echo("\nResolved bottlenecks:")
        for b in result["resolved_bottlenecks"]:
            typer.echo(f"  - {b}")
    if result["new_bottlenecks"]:
        typer.echo("\nNew bottlenecks:")
        for b in result["new_bottlenecks"]:
            typer.echo(f"  - {b}")


_THRESHOLD_PREFIX_LABELS = {
    "ncu_": "NCU",
    "nsys_": "NSYS",
    "perf_": "Linux perf",
    "host_device_": "Host-device",
    "metal_": "Metal trace",
    "cross_": "Cross-profiler MPS",
    "io_": "I/O",
    "nvtx_": "NVTX phases",
}


def _threshold_category(field_name: str) -> str:
    for prefix, label in _THRESHOLD_PREFIX_LABELS.items():
        if field_name.startswith(prefix):
            return label
    return "Torch trace"


def _section_matches(category: str, section: str) -> bool:
    return section.lower() in category.lower()


@app.command()
def thresholds(
    section: str = typer.Option(None, "--section", help="Filter by category (e.g. ncu, nsys, metal, perf, io)"),
    task_yaml: str = typer.Option(None, "--task", help="Show effective values from a task.yaml"),
):
    """List analysis thresholds used for bottleneck diagnosis."""
    defaults = AnalysisThresholds()
    effective = defaults

    if task_yaml:
        task = TaskSpec.load(Path(task_yaml))
        effective = task.analysis_thresholds

    typer.echo("Analysis Thresholds — defaults for bottleneck diagnosis")
    typer.echo("Override any field in task.yaml under `analysis_thresholds:`\n")

    # Group fields by category
    grouped: dict[str, list[dataclasses.Field]] = {}
    for f in dataclasses.fields(AnalysisThresholds):
        cat = _threshold_category(f.name)
        grouped.setdefault(cat, []).append(f)

    # Stable category order: match the prefix order, then fallback
    cat_order = ["Torch trace"] + [v for v in _THRESHOLD_PREFIX_LABELS.values()]

    for cat in cat_order:
        fields = grouped.get(cat)
        if not fields:
            continue
        if section and not _section_matches(cat, section):
            continue

        typer.echo(f"{cat}:")
        for f in fields:
            val = getattr(effective, f.name)
            type_name = f.type if isinstance(f.type, str) else f.type.__name__
            default_val = getattr(defaults, f.name)

            if task_yaml and val != default_val:
                typer.echo(f"  {f.name:<35s} {type_name:<8s} {val}    (default: {default_val})")
            else:
                typer.echo(f"  {f.name:<35s} {type_name:<8s} {val}")
        typer.echo()


def _fmt_val(val: object) -> str:
    """Format a value for display, handling None and containers."""
    if val is None:
        return "auto"
    if isinstance(val, dict):
        return json.dumps(val) if val else "{}"
    if isinstance(val, list):
        return json.dumps(val) if val else "[]"
    if isinstance(val, Path):
        return str(val)
    return repr(val) if isinstance(val, str) else str(val)


def _show_section(
    title: str,
    effective: Any,
    defaults: Any,
    yaml_key: str,
) -> None:
    """Print one task config section, highlighting overrides."""
    typer.echo(f"{title}:  (task.yaml key: `{yaml_key}`)")
    for f in dataclasses.fields(effective):
        val = getattr(effective, f.name)
        def_val = getattr(defaults, f.name)
        marker = ""
        if val != def_val:
            marker = f"    (default: {_fmt_val(def_val)})"
        typer.echo(f"  {f.name:<30s} {_fmt_val(val)}{marker}")
    typer.echo()


@app.command(name="show-task")
def show_task(
    task_yaml: str = typer.Argument(..., help="Path to task.yaml"),
):
    """Show effective configuration for a task, highlighting overrides."""
    from perflab.task_spec import (
        AgentSpec,
        BenchmarkSpec,
        Constraints,
        ContractSpec,
    )

    task = TaskSpec.load(Path(task_yaml))

    typer.echo(f"Task: {task.name}")
    typer.echo(f"Workspace: {task.workspace}")
    typer.echo(f"Program type: {task.program_type}")
    if task.target_hardware:
        typer.echo(f"Target hardware: {task.target_hardware}")
    typer.echo()

    # Benchmark
    typer.echo("Benchmark:  (task.yaml key: `benchmark`)")
    typer.echo(f"  {'cmd':<30s} {task.benchmark.cmd}")
    typer.echo(f"  {'metric':<30s} {task.benchmark.metric.name} ({task.benchmark.metric.mode})")
    bm_defaults = BenchmarkSpec(cmd="", metric=task.benchmark.metric)
    for fname in ("warmup", "repeats"):
        val = getattr(task.benchmark, fname)
        def_val = getattr(bm_defaults, fname)
        marker = f"    (default: {def_val})" if val != def_val else ""
        typer.echo(f"  {fname:<30s} {val}{marker}")
    typer.echo()

    # Constraints
    _show_section("Constraints", task.constraints, Constraints(), "constraints")

    # Agent
    _show_section("Agent", task.agent, AgentSpec(), "agent")

    # Contract
    _show_section("Contract", task.contract, ContractSpec(), "contract")

    # Edit policy
    typer.echo("Edit policy:  (task.yaml key: `edit_policy`)")
    if task.edit_policy.allowed_paths:
        for p in task.edit_policy.allowed_paths:
            typer.echo(f"  {p}")
    else:
        typer.echo("  (no paths — all files editable)")
    typer.echo()

    # Correctness
    typer.echo("Correctness:  (task.yaml key: `correctness`)")
    typer.echo(f"  {'cmd':<30s} {task.correctness.cmd}")
    typer.echo(f"  {'expected_exit':<30s} {task.correctness.expected_exit}")
    typer.echo()

    # Build
    if task.build:
        typer.echo("Build:  (task.yaml key: `build`)")
        typer.echo(f"  {'cmd':<30s} {task.build.cmd}")
        typer.echo()

    # Roofline
    if task.roofline:
        typer.echo("Roofline:  (task.yaml key: `roofline`)")
        typer.echo(f"  {'peak_tflops':<30s} {task.roofline.peak_tflops}")
        typer.echo(f"  {'peak_mem_bw_gbs':<30s} {task.roofline.peak_mem_bw_gbs}")
        if task.roofline.title:
            typer.echo(f"  {'title':<30s} {task.roofline.title}")
        if task.roofline.peak_fp16_tflops:
            typer.echo(f"  {'peak_fp16_tflops':<30s} {task.roofline.peak_fp16_tflops}")
        if task.roofline.dtype_peaks:
            typer.echo("  dtype_peaks:")
            for k, v in task.roofline.dtype_peaks.items():
                typer.echo(f"    {k:<28s} {v}")
        typer.echo()

    # Data hints
    import dataclasses as _dc
    dh = task.data_hints
    dh_fields = {f.name: getattr(dh, f.name) for f in _dc.fields(dh) if getattr(dh, f.name) is not None}
    if dh_fields:
        typer.echo("Data hints:  (task.yaml key: `data_hints`)")
        for k, v in dh_fields.items():
            typer.echo(f"  {k:<30s} {v}")
        typer.echo()

    # Tuning.yaml info
    tuning_path = task.workspace / "tuning.yaml"
    if tuning_path.exists():
        try:
            import yaml as _yaml
            knobs = _yaml.safe_load(tuning_path.read_text(encoding="utf-8"))
            if knobs:
                sweep = knobs.get("sweep", {})
                fixed = task.contract.fixed_params if task.contract else {}
                tunable = {k: v for k, v in knobs.items() if k != "sweep" and k not in fixed}
                typer.echo("Tuning:  (tuning.yaml)")
                if fixed:
                    typer.echo(f"  Fixed by contract (cannot change): {fixed}")
                if tunable:
                    typer.echo(f"  Tunable parameters: {tunable}")
                if sweep:
                    n_combos = 1
                    for vals in sweep.values():
                        if isinstance(vals, list):
                            n_combos *= len(vals)
                    typer.echo(f"  Sweep space: {sweep}")
                    typer.echo(f"  Total combinations: {n_combos} (auto-tuned after accepted edits)")
                else:
                    typer.echo("  Sweep: not configured (add `sweep:` section for auto-tuning)")
                typer.echo()
        except Exception:  # noqa: BLE001 -- best-effort tuning.yaml display, skip on any parse issue
            pass

    # Analysis thresholds — count overrides
    defaults_thresh = AnalysisThresholds()
    n_overrides = sum(
        1 for f in dataclasses.fields(task.analysis_thresholds)
        if getattr(task.analysis_thresholds, f.name) != getattr(defaults_thresh, f.name)
    )
    n_total = len(dataclasses.fields(AnalysisThresholds))
    typer.echo(f"Analysis thresholds: {n_overrides}/{n_total} overridden  (run `perflab thresholds --task {task_yaml}` for details)")


@app.command(name="show-task-schema")
def show_task_schema():
    """Show the full task.yaml schema with all fields and types."""
    typer.echo("=" * 70)
    typer.echo("  task.yaml Schema Reference")
    typer.echo("=" * 70)
    typer.echo()

    _SCHEMA = [
        ("REQUIRED FIELDS", [
            ("name", "str", "Task name (used in reports and file paths)"),
            ("workspace", "str", "Path to the task directory"),
            ("program_type", "str", "python | pytorch | jax | triton | cpp | cuda"),
            ("correctness.cmd", "str", "Command to run correctness test (must exit 0)"),
            ("benchmark.cmd", "str", "Command to run benchmark (must write --json)"),
            ("benchmark.metric.name", "str", "Dotted path into bench.json (e.g., tflops.median)"),
            ("benchmark.metric.mode", "str", "maximize | minimize"),
            ("edit_policy.allowed_paths", "list[str]", "Files the agent can edit (e.g., ['sgemm.cu', 'tuning.yaml'])"),
        ]),
        ("OPTIONAL FIELDS", [
            ("target_hardware", "str|null", "GPU/CPU name for hardware-specific hints (e.g., 'NVIDIA H100')"),
            ("build.cmd", "str|null", "Build command for compiled languages (e.g., 'nvcc -O2 -o bin kern.cu')"),
            ("benchmark.warmup", "int", "Warmup iterations before timing (default: 3)"),
            ("benchmark.repeats", "int", "Timed iterations (default: 20)"),
            ("benchmark.secondary_metric", "dict|null", "Secondary metric for Pareto analysis"),
        ]),
        ("CONSTRAINTS", [
            ("constraints.max_iters", "int", "Max agent iterations (default: 10)"),
            ("constraints.regression_tolerance", "float", "Min improvement fraction to accept (default: 0.02)"),
            ("constraints.rlimit_as_gb", "float|null", "Memory limit in GB (null=auto, 0=disable)"),
            ("constraints.prompt_token_budget", "int", "Max prompt tokens (0=unlimited)"),
            ("constraints.top_n", "int", "Number of bottleneck diagnoses (default: 3)"),
            ("constraints.max_history", "int", "Recent iterations in prompt (default: 3)"),
            ("constraints.allow_fast_math", "bool", "Permit -ffast-math/--use_fast_math (default: false)"),
            ("constraints.accuracy_tolerance", "str|null", "Acceptable error: 'exact', '1e-3', '1e-1'"),
        ]),
        ("CONTRACT (anti-gaming)", [
            ("contract.fixed_params", "dict", "Values enforced in bench.json meta (e.g., {M: 4096, N: 4096})"),
            ("contract.min_repeats", "int", "Minimum benchmark repeats (default: 1)"),
            ("contract.min_warmup", "int", "Minimum warmup iterations (default: 0)"),
            ("contract.required_bench_fields", "list[str]", "Fields that must exist in bench.json"),
        ]),
        ("ROOFLINE", [
            ("roofline.peak_tflops", "float", "Hardware peak TFLOPS (e.g., 989.0 for H100 TF32)"),
            ("roofline.peak_mem_bw_gbs", "float", "Peak memory bandwidth in GB/s (e.g., 3350.0)"),
            ("roofline.title", "str|null", "Plot title"),
            ("roofline.peak_fp16_tflops", "float|null", "FP16 peak for Tensor Core ceiling"),
            ("roofline.dtype_peaks", "dict|null", "Per-dtype peaks (peak_tflops_fp32, _tf32, _fp16, _bf16)"),
        ]),
        ("ANALYSIS THRESHOLDS", [
            ("analysis_thresholds.*", "float|int", "~55 fields — run `perflab thresholds` for full list"),
            ("  ncu_tc_util_low", "float", "Tensor Core utilization threshold (default: 30%)"),
            ("  ncu_stall_pct_high", "float", "Warp stall threshold (default: 30%)"),
            ("  ncu_bank_conflicts_high", "float", "Bank conflict threshold (default: 100)"),
            ("  ncu_sectors_per_request_high", "float", "Coalescing threshold (default: 4.0)"),
        ]),
        ("DATA HINTS", [
            ("data_hints.sparsity", "float|null", "Fraction of zero values (0.0-1.0). >0.9 suggests sparse formats"),
            ("data_hints.value_range", "list|null", "[min, max] of typical values. Small range → FP16 safe"),
            ("data_hints.access_pattern", "str|null", "sequential | random | strided | blocked"),
            ("data_hints.batch_size_range", "list|null", "[min, max] production batch sizes. [1,N] → latency matters"),
            ("data_hints.dtype_safety", "str|null", "fp16_safe | bf16_safe | int8_safe — confirmed precision-safe"),
            ("data_hints.sequence_lengths", "str|null", "fixed | variable_<min>_<max>"),
            ("data_hints.custom", "list|null", "Free-form hints: ['data is symmetric', 'output is sparse']"),
        ]),
        ("AGENT", [
            ("agent.n_candidates", "int", "Candidates per LLM call (default: 6)"),
            ("agent.top_k", "int", "Top candidates to re-benchmark (default: 2)"),
            ("agent.max_iters", "int", "Agent loop iterations (default: 12)"),
        ]),
    ]

    for section, fields in _SCHEMA:
        typer.echo(f"{section}:")
        for name, type_str, desc in fields:
            typer.echo(f"  {name:<40s} {type_str:<15s} {desc}")
        typer.echo()



@app.command(name="show-tuning-schema")
def show_tuning_schema():
    """Show what goes in tuning.yaml: fixed vs tunable params, sweep syntax."""
    typer.echo("=" * 70)
    typer.echo("  tuning.yaml Schema Reference")
    typer.echo("=" * 70)
    typer.echo()

    typer.echo("FIXED PARAMETERS (protected by contract.fixed_params):")
    typer.echo("  M, N, K                                 Problem dimensions (matmul)")
    typer.echo("  batch_size, seq_len                      Workload size (training)")
    typer.echo("  num_images, iterations                   Dataset/iteration count")
    typer.echo("  → These CANNOT be changed by agent or optimizer")
    typer.echo()

    typer.echo("TUNABLE PARAMETERS (fair game for optimization):")
    typer.echo("  block_size                               Thread block size (CUDA)")
    typer.echo("  TILE_M, TILE_N, TILE_K                   Tiling dimensions")
    typer.echo("  BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K Triton block sizes")
    typer.echo("  num_warps                                Warps per CTA (Triton)")
    typer.echo("  num_stages, NUM_STAGES                   Pipeline stages")
    typer.echo("  lr, momentum                             Training hyperparameters")
    typer.echo("  num_workers                              DataLoader parallelism")
    typer.echo()

    typer.echo("SWEEP SECTION (auto-tuning search space):")
    typer.echo("  sweep:")
    typer.echo("    TILE_M: [64, 128, 256]                 Values to try for each knob")
    typer.echo("    TILE_N: [64, 128, 256]                 Cartesian product of all lists")
    typer.echo("    NUM_STAGES: [2, 3, 4]                  → benchmarked automatically")
    typer.echo()
    typer.echo("  Agent auto-tunes after accepted code edits (max 15 trials).")
    typer.echo("  For CUDA, sweep ranges are centered on CUTLASS-optimal baselines.")
    typer.echo("  Each config is contract-validated (fixed_params enforced).")
    typer.echo()
