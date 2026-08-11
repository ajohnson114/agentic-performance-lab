# PerfLab Architecture Guide

This document explains how PerfLab works from the ground up. It covers every layer of the system, how data flows through it, and how each component interacts with the others. Read the [README](README.md) first for a quick overview of what PerfLab does.

---

## Table of contents

1. [Project layout](#1-project-layout)
2. [Core concepts](#2-core-concepts)
3. [Data flow overview](#3-data-flow-overview)
4. [Task specification](#4-task-specification)
5. [CLI layer](#5-cli-layer)
6. [Runners: benchmark and correctness](#6-runners-benchmark-and-correctness)
7. [Profilers](#7-profilers)
8. [Bottleneck analyzer](#8-bottleneck-analyzer)
9. [The agent loop](#9-the-agent-loop)
10. [Patch system](#10-patch-system)
11. [LLM prompt construction](#11-llm-prompt-construction)
12. [Convergence detection](#12-convergence-detection)
13. [Fast screening](#13-fast-screening)
14. [LLM providers](#14-llm-providers)
15. [Run storage and event logging](#15-run-storage-and-event-logging)
16. [Reporting](#16-reporting)
17. [Roofline analysis](#17-roofline-analysis)
18. [Safety model](#18-safety-model)
19. [Contract validation and anti-gaming](#19-contract-validation-and-anti-gaming)
20. [System info and drift detection](#20-system-info-and-drift-detection)
21. [Code snapshots](#21-code-snapshots)
22. [Statistical comparison](#22-statistical-comparison)
23. [Subprocess sandboxing](#23-subprocess-sandboxing)
24. [The knob optimizer](#24-the-knob-optimizer)
25. [CI regression checks](#25-ci-regression-checks)
26. [Adding a new task](#26-adding-a-new-task)
27. [Adding a new profiler](#27-adding-a-new-profiler)
28. [Adding a new LLM provider](#28-adding-a-new-llm-provider)
29. [Environment and configuration](#29-environment-and-configuration)
30. [MCP server](#30-mcp-server)
31. [Progress callback](#31-progress-callback)
32. [Structured configuration](#32-structured-configuration)

---

## 1. Project layout

```
perflab/                            # Main Python package
  __init__.py                       # Exports __version__
  cli.py                            # Typer CLI — entry point for all commands
  config.py                         # PerfLabConfig dataclass + YAML/env loading + config template
  task_spec.py                      # YAML schema: TaskSpec dataclass + loader
  orchestrator.py                   # Coordinates profile/optimize workflows
  doctor.py                         # `perflab doctor` environment checks
  roofline_peaks.py                 # Hardware peak detection for roofline plots
  ci.py                             # CI regression check logic

  runners/
    benchmark.py                    # Run bench command, parse bench.json
    correctness.py                  # Run correctness test
    paired.py                       # Block-interleaved (paired) A/B benchmarking
    pipeline.py                     # Shared build→correctness→benchmark→profile pipeline

  profilers/
    base.py                         # Profiler protocol + shared bench-run helpers
    python_pyspy.py                 # py-spy CPU hotspots (speedscope JSON)
    pytorch_profiler.py             # PyTorch native profiler + Chrome trace
    jax_profiler.py                 # JAX XLA compilation + TPU profiling (HLO, traces)
    linux_perf.py                   # Linux perf stat + record/script/annotate
    nsys_profiler.py                # NVIDIA nsys (SQLite + regex fallback)
    ncu_profiler.py                 # NVIDIA ncu (CSV parsing, per-kernel) + SASS
    metal_trace.py                  # Apple Metal GPU trace (xctrace)
    memray_profiler.py              # Python heap profiling (peak, allocators)
    ebpf_profiler.py                # bpftrace syscall/IO latency (Linux)
    lock_contention.py              # perf lock + perf c2c (false sharing)
    thread_sched.py                 # Thread scheduling (perf sched latency/timehist)
    power_profiler.py               # RAPL energy + nvidia-smi power/memory polling
    perfetto_export.py              # Chromium Trace Event export for ui.perfetto.dev
    interval_union.py               # Busy-interval union (overlapping trace events)

  analyzers/
    bottleneck_types.py             # AnalysisThresholds + BottleneckDiagnosis dataclasses
    bottleneck_analyzer.py          # Dispatcher — routes to sub-modules below
    bottleneck_gpu.py               # NCU, NSys, Metal, host-device, MPS cross-profiler rules
    bottleneck_cpu.py               # Linux perf, TMA Level 2/3, I/O bottleneck rules
    bottleneck_system.py            # Torch trace, JAX/TPU, memray, eBPF, locks, power rules
    decision.py                     # The accept/reject decision (noise gate, CIs)
    bench_stats.py                  # Benchmark noise detection (CV, confidence intervals)
    tma.py                          # Top-Down Microarchitecture Analysis (L1 + L2/3)
    microarch.py                    # Derived micro-arch metrics, clock-throttle detection
    compiler_diagnostics.py         # Compiler diagnostic capture, parsing, and summarization
    build_flags.py                  # Build flag recommendations from ISA detection
    build_overrides.py              # Build flag override mechanism (allowlisted flags)
    vectorization.py                # Auto-vectorization verification (SIMD in perf annotate)
    gpu_attribution.py              # CPU→GPU call graph, kernel dossiers, stall detection
    hlo_attribution.py              # HLO op attribution for JAX/TPU (weighted cost model)
    cutlass_baselines.py            # CUTLASS-derived tile configs per GPU architecture
    profile_diff.py                 # Metric + hotspot deltas between iterations
    diff_flamegraph.py              # Differential flame graph generation
    metrics_rollup.py               # Run summary stats + is_improvement()
    user_actions.py                 # Build suggestions extracted from LLM reasoning

  optimizers/
    agent.py                        # Thin orchestrator — drives the phases below
    phases/
      baseline.py                   # Baseline measurement + initial profiling
      generate.py                   # Prompt build + LLM call + patch parsing
      prescreen.py                  # Parallel build + correctness in temp workspaces
      evaluate.py                   # Benchmark survivors, accept/reject, snapshot
      autotune.py                   # Knob sweep (tuning.yaml)
      finalize.py                   # Summary LLM call, reports, user actions
    patch.py                        # Search/replace parsing + validation
    prompt.py                       # LLM prompt construction
    history.py                      # Shared builder for optimizer history entries
    convergence.py                  # Early stopping detection
    cross_run.py                    # Prior-run context injection
    forbidden.py                    # Forbidden-construct list
    event_log.py                    # Structured JSONL event logging
    propose_params.py               # Knob sweep candidate generation
    progress.py                     # AgentProgress protocol + PrintProgress + ListProgress

  llm/
    base.py                         # Message + LLMProvider protocol
    config.py                       # YAML + env var config loading, PROVIDER_DEFAULT_MODELS
    pricing.py                      # Token cost estimation (drives --max-cost)
    openai_provider.py              # OpenAI API
    anthropic_provider.py           # Anthropic API
    ollama_provider.py              # Local Ollama
    mcp_sampling_provider.py        # MCP client LLM via sampling protocol

  memory/
    run_store.py                    # Run storage + read path (list, get, compare, update_meta)
    run_export.py                   # `perflab export` — bundle a run, optional slim filter
    run_diff.py                     # `perflab diff` — before/after across run snapshots

  harness/                          # Importable by protected bench.py / tests.py
    gpu_sync.py                     # SyncTimer — full device sync around timed regions
    thread_guard.py                 # Background-thread injection detection
    tensor_check.py                 # Output type / storage / data_ptr validation
    determinism.py                  # Reproducibility + no-op detection
    precision.py                    # ULP distance against an fp64 reference
    pointer_poison.py               # Defeat pointer-keyed memoization
    tolerance.py                    # Task-declared accuracy tolerance
    _array.py                       # Backend adapter (torch/numpy/JAX/Python)

  reporting/
    report_md.py                    # Markdown report generation
    dashboard_html/                 # HTML dashboard package
      data.py                       # AnalysisData + dashboard input dataclasses
      page.py                       # Page assembly and top-level sections
      widgets.py                    # Shared HTML widgets
      diagnostics.py                # Bottleneck / diagnostic sections
      profiler_sections.py          # Per-profiler sections
      accelerator_sections.py       # GPU / TPU / Metal sections
    plots.py                        # Metric history matplotlib plots
    roofline.py                     # Roofline plot generation
    view_server.py                  # Loopback server behind `perflab view`
    generate.py                     # ReportParams dataclass + generate_reports()

  server/                           # MCP server (31 tools)
    mcp_server.py                   # Re-exporting facade — entry point + test imports
    core.py                         # Server construction, shared state, executor
    authoring.py                    # Task authoring tools
    runs.py                         # Run listing / retrieval / comparison tools
    analysis.py                     # Profiling and diagnosis tools
    environment.py                  # Environment and hardware tools
    agent_tools.py                  # Background agent runs + MCP sampling
    task_templates.py               # Templates behind create_task

  tools/
    shell.py                        # Shell command execution + resource limits + CPU plan
    isolation.py                    # Bubblewrap sandboxing (auto/none/restricted/strict)
    seccomp.py                      # Classic-BPF syscall denylist for `strict`
    sysinfo.py                      # System info collection (CPU, GPU, load)
    env_fingerprint.py              # Environment fingerprint for run comparison
    symbols.py                      # Shared demangling + kernel base-name matching

  demo_tasks/                       # Bundled demo tasks, shipped as package data
    _sample/                        # Template behind `perflab tasks init`
    matmul/
      python/ cpp/ cpp_parallel/ cuda/ cuda_h100/ cuda_tensorcore/ triton/
      pytorch/                      # Includes matmul_op.py (editable op)
      jax/                          # Includes matmul_op.py (editable op)
    transformer_train/
      pytorch/ jax/
    attention/
      jax_tpu/                      # Naive attention → jit + bf16 + vectorized heads
    dataloader_bottleneck/
      pytorch/
    inference_pipeline/
      pytorch/                      # SmallCNN inference: per-image CPU preprocess, batch_size=1, eager fp32
    gpu_inference_demo/
      pytorch/                      # GPU inference demo: similar to inference_pipeline
    reduction/
      cpp_cuda/                     # C++ host + CUDA kernel reduction (host-device bottleneck)
    stream/
      python/                       # Memory-bandwidth-bound STREAM-style kernel
```

Demo tasks live inside the package because wheel package data has to; `perflab tasks list`
and `perflab tasks copy` are the supported ways to reach them.

---

## 2. Core concepts

**Task**: A self-contained directory with naive source code, a benchmark harness, and a correctness test. The source code is deliberately slow. Tasks are the unit of work for PerfLab.

**Benchmark harness** (`bench.py`): Measures performance and writes a JSON file with metrics. The metric name in `task.yaml` is a dotted path into this JSON (e.g., `tflops.median` resolves to `bench["tflops"]["median"]`).

**Correctness test** (`tests.py`): Validates that the code still produces correct results after optimization. Must exit 0. The agent cannot accept any change that breaks this test.

**Profiler**: Collects performance data (flame graphs, hardware counters, GPU traces) and returns structured summaries. PerfLab automatically selects profilers based on the `program_type` field in `task.yaml`.

**Bottleneck diagnosis**: Rule-based analysis of profiler summaries that identifies what is limiting performance (e.g., "memory-bound", "low GPU occupancy", "CPU dispatch overhead") and suggests specific actions.

**Agent**: An LLM-driven optimization loop that reads source code + profiler data, proposes code edits (search/replace blocks), validates them, benchmarks them, and accepts improvements. It uses beam search with multiple candidates per iteration.

**Run**: A timestamped directory under `out/runs/` containing all artifacts from a single execution (profile, optimize, or agent).

---

## 3. Data flow overview

### `perflab profile`

```
task.yaml ──> TaskSpec.load()
                  │
                  ├── run correctness test (must pass)
                  ├── run benchmark ──> bench.json
                  ├── run profilers ──> *_summary.json + artifacts (SVG, traces)
                  ├── diagnose bottlenecks (from summaries)
                  └── generate reports ──> report.md, dashboard.html, plots
```

### `perflab agent`

```
task.yaml ──> TaskSpec.load()
                  │
            ┌─────┴──── BASELINE ────┐
            │  correctness + benchmark + profile
            │  → baseline metric value
            │  → profiler summaries
            └─────┬──────────────────┘
                  │
            ┌─────┴──── ITERATION LOOP (up to max_iters) ────┐
            │                                                  │
            │  1. Read source files + profiler summaries       │
            │  2. Run bottleneck analyzer                      │
            │  3. Build LLM prompt (source + profiles + diag)  │
            │  4. Call LLM → N candidate responses             │
            │  5. Parse search/replace blocks from each        │
            │                                                  │
            │  For each candidate:                             │
            │    a. Validate patch (policy, paths, exact match)│
            │    b. Backup files                               │
            │    c. Apply patch                                │
            │    d. Run correctness test                       │
            │    e. Run benchmark (fast screen or full)        │
            │    f. Restore files from backup                  │
            │                                                  │
            │  6. Accept best improving candidate              │
            │     → re-apply permanently, re-profile           │
            │  7. Check convergence → maybe early stop         │
            └─────┬──────────────────────────────────────────┘
                  │
            ┌─────┴──── FINALIZE ────┐
            │  Generate optimization summary (LLM call)
            │  Write reports, plots, event log
            └────────────────────────┘
```

### `perflab optimize`

```
task.yaml ──> TaskSpec.load()
                  │
            Baseline benchmark
                  │
            ┌─── ITERATION LOOP ────┐
            │  Load tuning.yaml knobs
            │  Generate candidate knob sets
            │  For each candidate:
            │    Save knobs → run bench → check improvement
            │  Accept or revert
            └───────────────────────┘
```

---

## 4. Task specification

**File**: `perflab/task_spec.py`

The `TaskSpec` dataclass is the central configuration object. It is loaded from a `task.yaml` file via `TaskSpec.load(path)`. Every command starts by loading a TaskSpec.

### Key fields

| Field | Type | Purpose |
|-------|------|---------|
| `name` | `str` | Human-readable task name |
| `workspace` | `Path` | Resolved absolute path to the task directory |
| `program_type` | `ProgramType` | One of: `python`, `pytorch`, `jax`, `triton`, `cpp`, `cuda` |
| `build` | `CommandSpec \| None` | Optional compile step (used by cpp, cuda) |
| `correctness` | `CommandSpec` | Correctness test command + expected exit code |
| `benchmark` | `BenchmarkSpec` | Benchmark command + metric config |
| `profile_plan` | `ProfilePlan` | Which profilers to always/optionally run |
| `constraints` | `Constraints` | `max_iters`, `regression_tolerance`, `rlimit_as_gb`, `prompt_token_budget`, `top_n`, `max_history` |
| `edit_policy` | `EditPolicy` | Glob patterns restricting which files can be edited |
| `contract` | `ContractSpec` | Anti-gaming: fixed params, required bench fields, minimum repeats |
| `target_hardware` | `str \| None` | Hardware description for optimization hints |
| `roofline` | `RooflineSpec \| None` | Peak TFLOPS and memory bandwidth |
| `agent` | `AgentSpec` | Beam search config: `n_candidates`, `top_k`, `max_iters` |
| `analysis_thresholds` | `AnalysisThresholds` | Per-task overrides for bottleneck diagnosis thresholds |

### Workspace resolution

The `workspace` field in task.yaml is relative to the YAML file's parent directory. `TaskSpec.load()` resolves it to an absolute path:

```python
ws = (yaml_file.parent / data["workspace"]).resolve()
```

All commands (correctness, benchmark, profilers) run with `cwd=workspace`.

### Metric path resolution

The `metric.name` field (e.g., `"tflops.median"`) is a dotted path into `bench.json`. The `metric_value()` function in `runners/benchmark.py` walks the path:

```python
def metric_value(bench: dict, metric_name: str) -> float:
    cur = bench
    for part in metric_name.split("."):
        cur = cur[part]
    return float(cur)
```

---

## 5. CLI layer

**File**: `perflab/cli.py`

Uses [Typer](https://typer.tiangolo.com/) to define commands. The entry point is registered in `pyproject.toml`:

```toml
[project.scripts]
perflab = "perflab.cli:app"
```

### Command routing

| Command | Function | Delegates to |
|---------|----------|-------------|
| `perflab init` | `init()` | Interactive LLM setup + live model validation |
| `perflab profile` | `profile()` | `orchestrator.profile_only()` |
| `perflab optimize` | `optimize_cmd()` | `orchestrator.optimize()` |
| `perflab agent` | `agent()` | `optimizers.agent.run_agent()` |
| `perflab replay` | `replay()` | `event_log.replay_events()` |
| `perflab peaks` | `peaks()` | `roofline_peaks.infer_peaks()` |
| `perflab doctor` | `doctor()` | `doctor.run_doctor()` |
| `perflab ci-check` | `ci_check()` | `ci.run_ci_check()` |
| `perflab list-runs` | `list_runs_cmd()` | `RunStore.list_runs()` |
| `perflab compare` | `compare()` | `RunStore.compare_runs()` |
| `perflab show-task` | `show_task()` | Effective task config with override highlighting |
| `perflab thresholds` | `thresholds()` | `AnalysisThresholds` introspection |

### `perflab init`

Interactive first-run setup for LLM configuration. Walks the user through selecting a provider (`openai`, `anthropic`, `ollama`), model, and API key. After collecting inputs, it makes a live API call (`"Respond with OK."`, max_tokens=5) to validate the model before writing the config. If validation fails (invalid model name, bad API key, unreachable endpoint), the user sees the error and can choose whether to save the config anyway.

The `agent` command constructs an `AgentConfig` from CLI flags + task.yaml defaults, loads the LLM config, and calls `run_agent()`.

---

## 6. Runners: benchmark and correctness

**Files**: `perflab/runners/benchmark.py`, `perflab/runners/correctness.py`

### Correctness runner

Runs the correctness command via `perflab/tools/shell.py:run_cmd()`. Returns a `CmdResult` with `stdout`, `stderr`, and `returncode`. The caller checks `returncode == expected_exit` (default 0).

The correctness runner enforces a **60-second timeout** and passes `program_type` to `run_cmd()` so that GPU program types (`cuda`, `pytorch`, `jax`, `triton`) skip the `RLIMIT_AS` memory cap (see [Section 23](#23-subprocess-sandboxing)).

### Benchmark runner

```python
def run_benchmark(cmd, cwd, env=None, fast_mode=False, program_type=None) -> tuple[CmdResult, dict]:
```

1. If `fast_mode=True`, sets `PERFLAB_BENCH_WARMUP=0` and `PERFLAB_BENCH_REPEATS=2` as environment variables (used only for ranking during beam search — the top candidate is always re-benchmarked at full fidelity)
2. For GPU program types, runs a **thermal gate** — if GPU temperature exceeds 80°C, waits up to 120s for cooldown to 75°C before proceeding
3. Runs the benchmark command with a **300-second timeout** and GPU-aware resource limits
3. Reads `cwd/out/bench.json` and parses it as JSON
4. If a `ContractSpec` is available, runs `validate_contract()` (see [Section 19](#19-contract-validation-and-anti-gaming))
5. Returns the `CmdResult` and parsed dict

### Contract validation

`validate_contract(bench, contract)` checks after every benchmark run:

- All `required_bench_fields` exist in the bench JSON (dotted-path traversal)
- All `fixed_params` match the values in `bench["meta"]`

A contract violation raises `RuntimeError`, which rejects the candidate immediately.

The benchmark harness is responsible for creating the `out/bench.json` file. PerfLab validates its structure against the task's `ContractSpec` if one is defined.

---

## 7. Profilers

**Files**: `perflab/profilers/base.py` + individual profiler modules

### Profiler protocol

Every profiler implements this protocol (from `base.py`):

```python
class Profiler(Protocol):
    name: str
    def is_available(self) -> bool: ...
    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult: ...
```

`ProfileResult` contains:
- `name`: profiler identifier (e.g., `"py_spy"`)
- `artifacts`: dict mapping logical names to file paths (e.g., `{"pyspy_speedscope": "artifacts/pyspy_speedscope.json"}`)
- `summary`: dict of structured metrics that feeds into the bottleneck analyzer

### Profiler selection

`perflab.profilers.select_profilers(task)` selects profilers based on `program_type`:

| program_type | Profilers |
|-------------|-----------|
| `python` | py-spy, Linux perf, memray, eBPF, power |
| `pytorch` | py-spy, PyTorch profiler, Linux perf, nsys, ncu, Metal trace, memray, eBPF, power |
| `jax` | py-spy, JAX profiler, Linux perf, nsys, ncu, Metal trace, memray, eBPF, power |
| `triton` | py-spy, Linux perf, nsys, ncu, Metal trace, memray, eBPF, power |
| `cuda` | Linux perf, nsys, ncu, eBPF, lock contention, thread sched, power |
| `cpp` | Linux perf, nsys, ncu, eBPF, lock contention, thread sched, power |

Each profiler's `is_available()` method checks whether the required tool is installed (e.g., `shutil.which("py-spy")`). Unavailable profilers are skipped silently.

### Individual profilers

**py-spy** (`python_pyspy.py`): Runs `py-spy record --native --format speedscope -o <json> -- python <bench_cmd>`. Falls back to running without `--native` if it fails (e.g., macOS SIP restrictions). Parses the speedscope JSON to extract both top hotspot functions (aggregated by leaf-frame duration) and timestamped samples for temporal GPU cross-referencing. Users can open the JSON in [Speedscope](https://www.speedscope.app/) for interactive flame graph visualization.

**PyTorch profiler** (`pytorch_profiler.py`): Wraps `torch.profiler` to collect a Chrome trace JSON. Parses the trace to extract:
- `top_ops`: CPU operators sorted by duration
- `top_gpu_kernels`: GPU kernels sorted by duration (broadened detection covers CUDA `kernel` events and MPS/Metal GPU events via `_is_gpu_event()`)
- `cpu_vs_gpu`: ratio of GPU time to CPU time
- `memory`: allocation counts, peak memory
- `sync_count` / `total_sync_time_us`: GPU synchronization overhead
- `phases`: per-phase training breakdown (forward, backward, optimizer, etc.) extracted from `record_function("## phase_name ##")` markers. Two-pass extraction: Pass 1 finds phase markers and their time ranges, Pass 2 attributes GPU kernel events to phases by timestamp containment. Each phase entry has `total_us`, `gpu_us`, `cpu_us`, `count`, and `pct`.

**nsys** (`nsys_profiler.py`): Runs `nsys profile` and exports to SQLite. Parses the database with 8 SQL queries:
- `_extract_top_kernels()`: GPU kernel durations grouped by name
- `_extract_memcpy()`: Data transfer operations by direction (HtoD, DtoH, DtoD)
- `_extract_top_api_calls()`: CUDA runtime API call frequency
- `_extract_gpu_utilization()`: Percentage of time the GPU was active
- `_extract_kernel_gaps()`: Time between consecutive kernel launches
- `_extract_nvtx_ranges()`: Top NVTX annotation ranges by duration (queries `NVTX_EVENTS` or `NVTX_RANGES` table). Produces `nvtx_ranges` list with name, duration_ms, and percentage.
- `_extract_kernel_launch_dims()`: Per-kernel grid/block dimensions. Enriches `top_kernels` entries with `avg_grid_size`, `avg_block_size`, `avg_threads_per_launch`.
- `_extract_cuda_sync_overhead()`: Filters runtime API calls for `cudaDeviceSynchronize` and `cudaStreamSynchronize`. Produces `sync_calls` list and `total_sync_ms`.

Falls back to regex parsing of `nsys stats` text output if SQLite export is unavailable.

**JAX profiler** (`jax_profiler.py`): Tracks XLA compilation activity by setting `JAX_LOG_COMPILES=1` and `XLA_FLAGS` for HLO operation dumps. Follows the standard `Profiler` protocol. Extracts:
- XLA compilation count and per-compilation timing
- Recompilation warnings (functions compiled more than once)
- Total compilation time overhead
- HLO module count and per-op type breakdown (top 10 by frequency)
- TPU device detection (chip type, count, device IDs) via `jax.devices()`
- JAX profiler trace parsing (host-device time split, MXU utilization, infeed stalls) from Chrome-format JSON traces

The summary feeds into `_analyze_jax()` for JAX-specific diagnoses and `_analyze_tpu()` for TPU-specific diagnoses when TPU devices are detected in `system_info`.

**ncu** (`ncu_profiler.py`): Profiles the benchmark once (`ncu --set full -o report.ncu-rep`), then exports CSV from the report via `ncu --import` — no second profiled run (`--set full` replays every kernel many times, making it the most expensive profiling mode; a failed report/export falls back to the old profiled `--csv` run). Groups CSV rows by kernel name. For each kernel extracts:
- SM utilization, memory throughput, compute throughput
- Achieved occupancy, registers per thread
- L1/L2 hit rates, shared memory usage
- Source file and function name (when source columns are present in CSV)
- DRAM read/write bytes (when DRAM byte columns are present in CSV), with per-kernel and aggregate `achieved_bw_gbs`
- Branch efficiency and warp execution efficiency (when branch/warp columns are present in CSV), for control divergence detection
- Tensor Core utilization and throughput (when tensor pipeline columns are present in CSV), for detecting Tensor Core underutilization
- Warp stall reasons (long_scoreboard, barrier, memory_throttle, math_pipe_throttle, etc.) for identifying why warps are idle
- Shared memory bank conflicts (when bank_conflicts columns are present), for serialization detection
- Memory coalescing efficiency via sectors per request (>1.0 means uncoalesced global loads)
- Occupancy limiters (registers, shared memory, block size) identifying which resource limits parallelism
- Instruction mix breakdown (FP32/FP64/INT/SFU pipe utilization) for detecting FP64-on-consumer-GPU and instruction bottlenecks
- TMA pipe utilization (Hopper+) for Tensor Memory Accelerator activity detection
- Register spill detection via local memory bytes (indicates register pressure causing spills to slow memory)
- Stall GMMA (Hopper-specific warp stall for Warp Group MMA completion)

**torch_profiler** (`pytorch_profiler.py`): Parses Chrome trace JSON. Extracts per-operator FLOPS (when `with_flops=True`), top operators by time and by FLOPS, GPU kernel breakdown, CPU vs GPU ratio, sync points, memory stats, phase breakdown, and operator shapes.

**jax** (`jax_profiler.py`): Extracts XLA HLO cost annotations (FLOP counts and bytes accessed) from HLO dump text files, enabling roofline analysis without ncu.

When source-line columns are detected, builds a `source_hotspots` list: groups by (source_file, function, line), sums a duration/stall metric, and returns the top 5. Each kernel entry is also enriched with `source_file` and `function` if available.

When DRAM byte columns are detected, `_parse_ncu_csv()` computes per-kernel and aggregate `achieved_bw_gbs`. The roofline plot uses profiler-measured DRAM bytes instead of theoretical estimates when available.

Computes weighted averages across invocations and identifies the dominant kernel (highest invocation count).

**Linux perf** (`linux_perf.py`): Runs a single `perf stat` benchmark run carrying the explicit events (`cycles`, `instructions`, `cache-references`, `cache-misses`, `branch-instructions`, `branch-misses`, `task-clock`) plus, when available, the `TopdownL1` metric group for TMA level 1 (and on AMD the generic cache-load events the TMA level-2 estimate needs) — previously each of those was its own benchmark run. Also runs `perf record` and post-processes with `perf script` to extract CPU function hotspots. Extracts IPC, cache miss rate, branch miss rate, `task_clock_ms`, and `cpus_utilized` (number of CPUs utilized, parsed from the `task-clock` line).

**Metal trace** (`metal_trace.py`): Uses `xctrace` on macOS to collect Metal GPU traces. Exports and parses the XML to extract submissions by encoder type (compute/render/blit), top GPU submissions, GPU idle percentage, and GPU performance counters (ALU utilization, memory bandwidth).

**memray** (`memray_profiler.py`): Runs `memray run` to capture Python heap allocations, then `memray stats` to extract peak memory, total allocations, and top allocators by size. Also generates a memory flamegraph HTML. Fills the gap where sampling profilers only show CPU time — many real-world bottlenecks are memory-bound (excessive allocations, peak memory pressure, GC thrashing) and invisible to py-spy. Only available for Python-based program types.

**eBPF** (`ebpf_profiler.py`): Uses `bpftrace` to trace read/write syscall latency distributions. Captures I/O wait time that sampling profilers miss — particularly relevant for data-loading bottlenecks where the CPU is idle waiting on disk or network I/O. Linux-only; auto-skipped on other platforms.

### Perfetto trace export

**File**: `perflab/profilers/perfetto_export.py`

After profiling completes, PerfLab exports a `perfetto_trace.json` in Chromium Trace Event JSON format. This file can be loaded into [Perfetto UI](https://ui.perfetto.dev/) for interactive timeline visualization. The export synthesizes:

- **CPU hotspots** from py-spy and perf as proportional duration events
- **Hardware counters** from perf stat as counter tracks (IPC, cache miss rate, etc.)
- **Memory allocation hotspots** from memray as allocation size events

This provides a polished timeline UI without requiring source code instrumentation or the Perfetto SDK. The dashboard includes a direct link to the trace file.

### Profiler hotspot diffing

**File**: `perflab/analyzers/profile_diff.py`

Beyond counter-level diffs (IPC, cache miss rate), the profile diff module now computes **function-level hotspot shifts** between baseline and optimized code. It merges py-spy and perf hotspots and reports which functions gained or lost CPU share, with statuses: `new` (appeared after optimization), `removed` (disappeared), `increased`, `decreased`. This gives the LLM precise feedback about whether a code change actually shifted work away from the hot function.

### Differential flame graphs

**File**: `perflab/analyzers/diff_flamegraph.py`

Compares baseline and optimized profiler summaries to produce a differential flame graph SVG. Each function bar is colored red (hotter — increased CPU share) or blue (cooler — decreased CPU share), sized by the absolute delta. Functions with <0.5pp change are filtered out. The SVG is embedded in the dashboard under the CPU profiler section.

### Lock contention profiler

**File**: `perflab/profilers/lock_contention.py`

Uses `perf lock record/report` to detect mutex/spinlock contention (acquired count, contended count, wait times) and `perf c2c record/report` to detect false sharing via HITM (Hit In Modified) cache line analysis. Auto-selected for `cpp` and `cuda` program types. Linux-only.

### Top-Down Microarchitecture Analysis (TMA)

**File**: `perflab/analyzers/tma.py`

Collects Intel's Top-Down Level 1 metrics, classifying cycles into Frontend Bound, Backend Bound, Bad Speculation, and Retiring. The metrics normally arrive pre-collected: `linux_perf` merges `-M TopdownL1` into its main `perf stat` run and passes the output text via `collect_tma(..., perf_stat_text=...)`, so no dedicated TMA benchmark run happens; a raw `topdown-*` counter run remains as fallback when the parse fails (both spellings of the metric names are handled — older perf's `frontend bound` and modern perf's `tma_frontend_bound`). Results are included in the `linux_perf` summary as `tma` dict and rendered in the dashboard as a 4-bar visualization.

### Power and energy profiler

**File**: `perflab/profilers/power_profiler.py`

Measures CPU energy via RAPL (`perf stat -e power/energy-pkg/`) and GPU power via nvidia-smi polling — both wrapped around the same single benchmark run, so CPU and GPU numbers describe the same execution window. Reports package/core energy in Joules, average power in Watts, and GPU power stats (avg, peak, P50, P95). Auto-selected for all program types.

### Cross-run learning

**File**: `perflab/optimizers/cross_run.py`

When the agent starts and prior runs exist for the same task, this module loads `report.json` and `agent_events.jsonl` from those runs. It builds a context string summarizing what worked, what didn't, and bottleneck diagnoses from up to 3 prior runs. This is injected into the LLM prompt on the first iteration only (to avoid token bloat).

**Scaling:**
- Loads at most **3 most recent** prior runs (older runs are ignored)
- Per-run summaries: accepted patches (all), rejected attempts (max 5), optimization summary (max 500 chars)
- Event insights (from `agent_events.jsonl`): capped at 20 most recent
- Total prompt addition: ~1000–3000 tokens, injected once on iteration 1 only

**Source code state:** The module does not assume the source code is at any particular state. If the user reverted to baseline, the prior context tells the LLM which strategies to try first. If the code was left optimized, the baseline benchmark captures the current performance. A disclaimer is included in the prompt so the LLM does not assume the code matches a prior run's starting state.

**Wiring:** `run_agent()` in `agent.py` calls `load_prior_run_context()` before the iteration loop and passes the result to `PromptContext.prior_run_context`. The prompt builder inserts it before the optimization history section.

### Multi-metric Pareto optimization

**Files**: `perflab/task_spec.py` (`SecondaryMetricSpec`), `perflab/reporting/plots.py` (`compute_pareto_frontier`, `plot_pareto_frontier`), `perflab/reporting/generate.py`, `perflab/reporting/dashboard_html/`

Tasks can optionally define a `secondary_metric` in their benchmark section. The agent loop **still optimizes for the primary metric only** — it accepts/rejects candidates based on primary metric improvement. The secondary metric is tracked alongside each iteration (read from `bench.json` after each accepted candidate) and used for post-hoc Pareto analysis.

**Pipeline:**
1. `agent.py` reads the secondary metric from `bench.json` after each accepted candidate and stores it as `secondary_value` in the history entry
2. `generate_reports()` calls `compute_pareto_frontier()` to find non-dominated iterations
3. `plot_pareto_frontier()` renders all points (grey) with frontier points highlighted in red diamonds, connected by a dashed line
4. The PNG is base64-embedded in the dashboard between the metric history chart and the roofline chart

**Demo task:** `perflab/demo_tasks/matmul/cpp_parallel` defines `tflops.median` (maximize) + `latency_ms.p95` (minimize).

### Benchmark noise detection

**File**: `perflab/analyzers/bench_stats.py`

Analyzes repeated benchmark measurements for statistical reliability. When `bench.json` includes `raw_values` or `samples` alongside the metric, PerfLab computes CV (coefficient of variation), 95% confidence intervals, and flags measurements with CV > 10% as noisy. The warning appears in `report.json` under `bench_stats` and as a yellow banner in the dashboard.

### Auto-vectorization verification

**File**: `perflab/analyzers/vectorization.py`

Parses `perf annotate --stdio` output to check whether hot functions contain SIMD instructions. Scans for SSE (`movaps`, `addps`), AVX (`vmovaps`, `vaddps`, `vfmadd`), AVX-512 (`zmm` registers), and NEON (`fmla`, `ld1`) mnemonics. Reports per-function: has_simd, ISA level, CPU percentage. The report appears in the dashboard under Diagnostics > Vectorization analysis.

### Thread scheduling analysis

**File**: `perflab/profilers/thread_sched.py`

Captures per-thread scheduling statistics via `perf sched record` + `perf sched latency` (runtime, switches, avg/max delay) and `perf sched timehist --summary` (per-CPU run time, migrations). Auto-selected for `cpp` and `cuda` program types. Surfaces in dashboard under Diagnostics > Thread scheduling.

### GPU memory tracking

**File**: Extended `perflab/profilers/power_profiler.py`

The power profiler's nvidia-smi polling loop now also captures `memory.used` and `memory.total` alongside `power.draw`. Computes peak/average usage and utilization percentage. Surfaces in dashboard under Diagnostics > GPU memory utilization.

### Machine fingerprint

**File**: `perflab/reporting/generate.py`

Every `report.json` now includes a `hardware` section extracted from `system_info.json`: `cpu_model`, `cpu_count`, `machine`, `platform`, `nvidia_gpus`, `cuda_version`, `cpp_compiler`. This makes reports self-documenting and helps cross-run learning assess hardware differences.

### Build flag overrides

**File**: `perflab/analyzers/build_overrides.py`

Provides a mechanism for agents to suggest compiler flag changes via a `build_overrides.json` file. Each flag is validated against an allowlist of safe options (optimization levels, architecture targeting, vectorization, OpenMP, LTO). Flags already present in the build command are not duplicated. Validation includes syntax checking and conflict detection (e.g., `-O2` and `-O3` together); invalid or conflicting flags are rejected with structured feedback via `BuildOverrideResult`, which the agent loop feeds back to the LLM.

### Summary JSON format

Each profiler writes a `<name>_summary.json` to the artifacts directory. These are the inputs to the bottleneck analyzer and are included in the LLM prompt during agent runs.

---

## 8. Bottleneck analyzer

**Files**: `perflab/analyzers/bottleneck_analyzer.py` (dispatcher), `bottleneck_gpu.py`, `bottleneck_cpu.py`, `bottleneck_system.py`, `bottleneck_types.py`

A rule-based system that consumes profiler summaries and produces ranked `BottleneckDiagnosis` objects. The analyzer is split by domain: GPU rules (NCU, NSys, Metal, host-device), CPU rules (Linux perf, TMA, I/O), and system rules (torch trace, JAX/TPU, memray, eBPF, locks, thread scheduling, power). The main file dispatches to sub-modules and re-exports all public symbols for backward compatibility.

```python
@dataclass
class BottleneckDiagnosis:
    rank: int                        # 1 = most important
    bottleneck: str                  # e.g., "Low SM utilization (23%)"
    root_cause: str                  # e.g., "Insufficient parallelism"
    confidence: str                  # "high" | "medium" | "low"
    suggested_actions: list[str]     # Specific code changes to try
```

### Analysis chain

`diagnose_bottlenecks()` calls specialized analyzers for each available profiler:

1. `_analyze_ncu()` — GPU compute vs memory bound, occupancy, register pressure, control divergence, Tensor Core underutilization, warp stall diagnosis (including Hopper Stall GMMA), bank conflict detection, uncoalesced access detection, occupancy limiter diagnosis, FP64-on-consumer-GPU detection, register spill detection, TMA pipe utilization
2. `_analyze_nsys()` — Kernel dominance, launch overhead, data transfer bottlenecks
3. `_analyze_perf()` — CPU IPC, cache misses, branch misprediction, function hotspots, single-threaded execution detection, no-SIMD vectorization detection
4. `_analyze_metal()` — GPU idle time, ALU utilization, blit transfer overhead
5. `_analyze_torch_trace()` — CPU dispatch overhead, excessive syncs, dominant GPU kernels, per-phase training breakdown (phase dominance and per-phase GPU underutilization)
6. `_analyze_jax()` — JAX-specific: recompilation detection, high compilation overhead, excessive compilations (see below)
7. `_analyze_cross_profiler_cpu_gpu()` — MPS cross-profiler CPU/GPU synthesis (see below)
8. `_analyze_io_bottleneck()` — Cross-profiler I/O and data loading detection
9. `_analyze_host_device()` — Host-device cross-analysis for C++/CUDA programs (see below)
10. `_analyze_nvtx_phases()` — NVTX annotation phase-level analysis (see below)
11. `_analyze_memray()` — Peak memory usage, dominant allocator detection
12. `_analyze_ebpf()` — I/O syscall latency (read/write p99), excessive syscall count
13. `_analyze_lock_contention()` — Lock contention ratio, total wait time, false sharing (perf c2c HITM events)
14. `_analyze_thread_sched()` — Thread scheduling delays, excessive migrations
15. `_analyze_power()` — GPU thermal throttling detection via power draw decline

Findings are sorted by confidence (high > medium > low) and the top 3 are returned. These diagnoses are:
- Included in the Markdown report and HTML dashboard
- Fed directly into the LLM prompt during agent mode

All numeric thresholds used by the analysis chain are defined in the `AnalysisThresholds` dataclass (~50 fields, including memray, eBPF, lock contention, thread scheduling, and power thresholds). Tasks can override any subset via the `analysis_thresholds:` key in `task.yaml`. Run `perflab thresholds` to list all fields, their types, defaults, and profiler categories.

### Single-threaded execution detection

`_analyze_perf()` checks `cpus_utilized` (from `task-clock`) against `perf_cpus_utilized_low` (default 1.5). When the program uses fewer CPUs than the threshold on a multi-core system (≥ 4 cores), a diagnosis fires with suggestions for OpenMP, `std::thread`, and C++17 parallel algorithms. High confidence when `cpus_utilized < 1.1` and `cpu_count >= 8`. Only fires for CPU-centric program types (`cpp`, `cuda`, `python`) — GPU-centric types (`pytorch`, `jax`, `triton`) are excluded because their parallelism comes from the GPU.

`diagnose_bottlenecks()` accepts optional `system_info` (dict with `cpu_count`) and `source_hints` (dict with `has_simd`, `has_openmp`, `has_threading` bools) parameters. Call sites pass `system_info` from `system_info.json` and `source_hints` from `compute_source_hints()`. The `top_n` parameter (default 3) controls how many diagnoses are returned; it is configurable via `constraints.top_n` in `task.yaml`.

### No-SIMD vectorization detection

`_analyze_perf()` checks `source_hints["has_simd"]` when `program_type == "cpp"` and there is a dominant hotspot (> `perf_hotspot_dominance_pct`). If no SIMD intrinsics are found, a medium-confidence diagnosis fires with suggestions for SIMD intrinsics, `-march=native -O3`, auto-vectorization hints, and `__restrict__` pointers. `compute_source_hints()` uses regex patterns matching `_mm256`, `_mm512`, `__m128`, `immintrin.h`, `vld1`, and `vaddq`.

### JAX bottleneck analysis

`_analyze_jax()` is called when the `jax_profiler` summary is present. It applies three rules:

1. **Recompilation detection** — If any function is compiled more than once, a high-confidence diagnosis is fired: "JAX recompilation detected." Suggestions: use static shapes, avoid Python-level control flow that changes trace structure, check for unintended `jax.jit` re-tracing.
2. **High compilation overhead** — If total compilation time exceeds a threshold fraction of total runtime: "High XLA compilation overhead." Suggestions: reduce the number of distinct compiled functions, use `jax.jit` at a coarser granularity, cache compilations with `jax.stages`.
3. **Excessive compilations** — If the total number of XLA compilations is unusually high: "Excessive XLA compilations." Suggestions: ensure shapes are static across calls, use `jax.jit` with explicit `in_shardings`/`out_shardings`, avoid recompilation triggers like changing batch sizes.

### TPU bottleneck analysis

`_analyze_tpu()` is called when `system_info` contains `tpu_devices`, piggybacking on the JAX profiler summary. It applies five rules:

1. **Low MXU utilization** — If MXU utilization (from JAX trace data) is below 30%, a diagnosis fires with suggestions: use bf16, pad to 128-multiples, increase batch size, use `jax.lax.scan`, remove host callbacks.
2. **XLA padding waste** — If `pad` operations exceed 20% of total HLO ops, a diagnosis fires. XLA inserts padding to align tensors to TPU tile boundaries. Suggestions: pre-pad inputs to 128-multiples, use power-of-2 batch sizes.
3. **Infeed stalls** — If host-to-device data transfer stalls exceed 10% of step time, a diagnosis fires. Suggestions: use `tf.data` with prefetch, increase data loading workers, use grain dataloader.
4. **HLO fragmentation** — If more than 10 HLO modules are generated, a diagnosis fires. Many separate XLA programs prevent efficient pipelining. Suggestions: consolidate `@jax.jit` functions, use `lax.scan`/`fori_loop`.
5. **fp32 without bf16** — If HLO ops are exclusively fp32, a diagnosis fires. TPU MXUs run bf16 at 2x the fp32 throughput. Suggestions: convert to `jnp.bfloat16`, use `jax.default_matmul_precision('bfloat16')`.

### MPS cross-profiler CPU/GPU synthesis

On MPS (Apple Silicon), the PyTorch Chrome trace doesn't emit `cat=kernel` events for Metal GPU work, so the single-profiler `cpu_vs_gpu` breakdown is empty or inaccurate. `_analyze_cross_profiler_cpu_gpu()` solves this by cross-referencing two data sources:

- **CPU op time** from the torch profiler's `top_ops` (sum of `total_us`)
- **GPU time** from the Metal trace profiler's `gpu_time_total_ms`

This function is called only when both `torch_profiler` and `metal_trace` summaries are present AND the torch summary doesn't already have valid GPU kernel data (i.e., CUDA traces are unaffected). It writes back a `cpu_vs_gpu` dict with `source: "cross_profiler_mps"` and fires findings for GPU/CPU ratio < 0.5 (CPU dispatch bottleneck) and GPU utilization < 30%.

### Per-phase training breakdown

When benchmark harnesses wrap training phases with `record_function("## phase_name ##")` markers, the torch profiler trace parser extracts per-phase timing data. The bottleneck analyzer uses this for two additional diagnoses:

1. **Phase dominance** — If any phase accounts for >60% of total time, a targeted diagnosis is fired with phase-specific suggestions (e.g., SDPA for forward-bound, gradient checkpointing/gradient accumulation/`torch.cuda.empty_cache()` for backward-bound, fused optimizer for optimizer-bound, num_workers for data-loading-bound).
2. **Per-phase GPU underutilization** — If GPU time is <30% of total time within the forward or backward phase, a diagnosis suggests `torch.compile` or larger batch size.

The phase breakdown table is also rendered directly in the LLM prompt (see [Section 11](#11-llm-prompt-construction)).

### Host-device cross-analysis (C++/CUDA)

`_analyze_host_device()` is called when `program_type` is `cpp` or `cuda`. It cross-references NSys GPU data with Linux perf CPU data to detect host-device interaction bottlenecks:

1. **Excessive sync** — If `total_sync_ms` > 20% of `cuda_kernel_time_ms`: "Excessive cudaDeviceSynchronize" (high confidence). Suggestions: use streams, async operations, CUDA graphs.
2. **Many small kernels** — If average kernel duration < 10 us and kernel count > 100: "Kernel launch overhead from many small kernels" (medium). Suggestions: fuse kernels, use CUDA graphs, batch operations.
3. **Data transfer bottleneck** — If `memcpy_time_ms` > 30% of `cuda_kernel_time_ms`: "Host-device transfer dominates" (high). Suggestions: pinned memory, overlap with streams, keep data on device.
4. **Low GPU utilization with CPU hotspot** — If `gpu_active_pct` < 50% AND perf has a hotspot > 30%: "CPU function '{name}' blocks GPU utilization" (high). Suggestions: overlap CPU/GPU, move work to GPU.

### NVTX phase analysis

`_analyze_nvtx_phases()` is called when the NSys summary contains `nvtx_ranges`:

1. **Single range dominating** — If one range > 60% of total NVTX time: "Phase '{name}' dominates execution" (medium). Suggestions: focus optimization on that phase.
2. **Many short ranges** — If > 50 ranges with avg < 1 ms: "Fine-grained work partitioning may cause overhead" (low). Suggestions: coarsen work granularity.

---

## 9. The agent loop

**Files**: `perflab/optimizers/agent.py` (orchestrator) and `perflab/optimizers/phases/` (phase logic)

This is the heart of PerfLab. `run_agent()` implements an iterative beam-search optimization loop where an LLM proposes code changes, and PerfLab evaluates them automatically.

`agent.py` is deliberately thin: it owns `AgentConfig`, `AgentContext`, `AgentResult`, the run setup, and `_run_iteration_loop()`, which sequences the phases below. Each phase lives in its own module under `perflab/optimizers/phases/`, and every phase entry point takes `AgentContext` (`ctx`) as its first parameter rather than a long argument list.

| Phase module | Entry point | Responsibility |
|---|---|---|
| `phases/baseline.py` | `run(ctx)` | Baseline measurement, initial profiling, system info, hardware-mismatch check, `snapshots/baseline.zip` |
| `phases/generate.py` | `run(ctx) -> GenerateResult` | Builds the prompt, calls the LLM, parses candidates |
| `phases/prescreen.py` | `run(...)` | Validates patches + builds + correctness tests **in parallel** in temp workspace copies |
| `phases/evaluate.py` | `evaluate_single_candidate(...)`, `accept_best(...)` | Sequential benchmarking, the accept/reject decision, re-profiling |
| `phases/autotune.py` | `run(ctx, max_trials=15)` | Post-accept knob sweep from `tuning.yaml`, CUTLASS-centered |
| `phases/finalize.py` | `run(ctx, status)`, `maybe_early_stop(ctx, it)` | Convergence check, summary LLM call, reports, user actions |

Within `phases/generate.py`, `build_iteration_prompt(ctx)` assembles the `PromptContext` for one iteration by delegating to four focused helpers: `_build_gpu_context()` (GPU attribution), `_build_diagnostics_context()` (bottleneck diagnosis + compiler cross-referencing), `_build_assembly_context()` (hot loops + SASS), and `_build_roofline_context()` (roofline point + auto-detection). The result carries all profiler summaries, failure memory, promising alternatives, and data hints.

`phases/evaluate.py` also owns the measurement-integrity paths that hang off an accept: `_remeasure_full()` (full-fidelity re-benchmark of the top fast-screened candidate), the paired A/B path (`_paired_workspaces`, `_remeasure_paired`, `_validate_paired_contract` — see [Section 6](#6-runners-benchmark-and-correctness)), `reprofile_after_accept()`, and `remeasure_baseline()` for drift.

Shared helpers that used to be private to `agent.py` now live where they're reusable: speedup math in `analyzers/metrics_rollup.py` (`calc_speedup`, `improvement_factor`, `is_improvement`), the accept gate itself in `analyzers/decision.py`, history-entry construction in `optimizers/history.py`, and source reading in `optimizers/patch.py` (`read_source_files`).

### Logging

The agent module uses Python's standard `logging` library (`logger = logging.getLogger(__name__)`). All exception handlers in profiler loading, roofline detection, profile diffs, build flag analysis, and SASS extraction log at `DEBUG` level with full tracebacks (`exc_info=True`). Use `perflab --verbose` to surface these during debugging.

### AgentConfig

```python
@dataclass
class AgentConfig:
    n_candidates: int = 6          # Candidates per LLM call
    top_k: int = 2                 # Top candidates for full re-benchmark
    max_iters: int = 12            # Maximum iterations
    early_stop: bool = True        # Enable convergence detection
    fast_screen: bool = True       # Enable two-tier benchmarking
    max_wall_time_s: int = 3600    # Wall-clock budget (default 1 hour)
    isolation: IsolationPolicy | None = None   # OS-level sandboxing
    max_cost_usd: float | None = None          # Estimated-cost budget
```

The `max_wall_time_s` field sets a hard wall-clock budget for the entire agent run. The agent checks elapsed time at the start of each iteration and stops if the budget is exceeded. Configurable via `--max-time`.

`isolation` carries the resolved `IsolationPolicy` (see [Section 18](#18-safety-model)) and is applied uniformly to baseline *and* candidate runs, so sandbox overhead cancels out of the speedup comparison rather than penalizing one arm. `max_cost_usd` stops the run gracefully — normal finalize and reports — once estimated LLM spend reaches the limit; `cli.py` fails closed at startup if it is set and the model's pricing is unknown, rather than running un-metered.

**Effective defaults differ from the dataclass.** `cli.py` resolves `candidates or task.agent.n_candidates or ga.n_candidates`, and `AgentSpec`'s own defaults (3 candidates / 12 iterations) are always truthy — so a CLI run without `--candidates` gets **3**, not the 6 above. The dataclass default applies only when `AgentConfig` is constructed directly (tests, the MCP server, library use).

### Iteration lifecycle

Each iteration proceeds as follows:

**Prompt building** (`phases/generate.build_iteration_prompt()`):

1. **Read source files**: `patch.read_source_files()` reads all files matching `edit_policy.allowed_paths` using `fnmatch`. Symlinks that resolve outside the workspace are skipped (see [Layer 19](#layer-19-symlink-protection)).
2. **Load profiler summaries**: All `*_summary.json` files from the artifacts directory
3. **Run bottleneck diagnosis**: `diagnose_bottlenecks()` on the latest profiler data
4. **Build LLM prompt**: Source code + profiler summaries + bench results + history + diagnoses

**LLM call and parsing** (`phases/generate.run()`):

5. **Call LLM**: Request `n_candidates` diverse optimization proposals
6. **Parse candidates**: Split on `--- CANDIDATE N ---` markers, extract search/replace blocks

**Parallel prescreening** (`phases/prescreen.run()`):

7. **Create temp workspace copies**: Each candidate gets an isolated copy of the workspace (excluding the task `out_dir`'s contents — `out/runs` accumulates artifacts every iteration and would make each copy progressively slower; the empty directory is kept for `out/bench.json` writes)
8. **Validate + build + test in parallel**: `ThreadPoolExecutor` runs all candidates concurrently (CPU-bound, no GPU contention)
9. **Filter**: Only candidates that pass validation, build, and correctness proceed to benchmarking
10. **Cleanup**: Temp workspaces are deleted. Original workspace is never modified during prescreen.

**Sequential benchmarking** (`phases/evaluate.evaluate_single_candidate()`, only for prescreen survivors):

11. **Backup files**: Copy originals to `backups/iterN/`
12. **Apply patch**: Replace first occurrence of each search block
13. **Run benchmark**: Fast screen or full depending on config (GPU-bound, must serialize)
14. **Restore files**: Always runs in `finally` block (even on exceptions)

**Candidate selection** (`phases/evaluate.accept_best()`):

15. **Select best**: Sort by metric value, check `is_improvement()` against current best
16. **Re-benchmark** (if fast screening): Full-precision benchmark for the top candidate
17. **Accept or reject**: If accepted, re-apply permanently and re-profile
18. **Collect promising alternatives**: Candidates that improved but weren't best are recorded for the next iteration
19. **Auto-tune sweep**: If `tuning.yaml` has a `sweep` section, sweep parameters (CUTLASS-centered)
20. **Update convergence**: Record improvement or failure
21. **Accumulate failure memory**: Failed candidates' strategies + reasons are recorded for all future iterations
22. **Check early stopping**: 5 consecutive failures, or 2 consecutive accepted improvements both below 3% → stop (`phases/finalize.maybe_early_stop()`)

### Additional agent features

**System info capture**: At the start of every agent run, `sysinfo.collect_system_info()` writes `system_info.json` to the run directory. This records CPU model, GPU info, CUDA version, load average, and OS details for reproducibility (see [Section 20](#20-system-info-and-drift-detection)).

**Code snapshots**: After the baseline and after each accepted patch, all files matching `edit_policy.allowed_paths` are archived to `snapshots/baseline.zip` and `snapshots/iter{N}.zip`. This provides a full audit trail of code evolution (see [Section 21](#21-code-snapshots)).

**Drift detection**: Every 3 accepted patches, the agent re-runs the benchmark on a clean (unpatched) version to detect measurement drift. Results are logged via the `drift_check` event type (see [Section 20](#20-system-info-and-drift-detection)).

**Parallel prescreening**: Candidates are validated, built, and correctness-tested in parallel using `ThreadPoolExecutor` with temporary workspace copies. Only survivors proceed to sequential GPU benchmarking. This reduces wall-clock time significantly — if 6 candidates each take 30s to build+test, parallel prescreening takes ~30s instead of 3 minutes. The original workspace is never modified during prescreening.

**Structured failure memory**: Across all iterations, the agent accumulates *why* candidates failed — not just that they failed. Each failure records: iteration, strategy description (from LLM reasoning), failure type (correctness/build/validation), reason (stderr excerpt), and profiler context. Shown to the LLM as "Failed approaches (DO NOT repeat these)" — prevents repeating dead-end strategies like "64x64 tiling caused register spill." Capped at last 10 failures.

**Promising alternatives**: When multiple candidates improve performance but only the best is accepted, the runner-up improvements are shown to the LLM in the next iteration: "These candidates also improved performance — consider combining them." If tiling gives 3x and WMMA gives 4x independently, the LLM learns to try tiling + WMMA together. Top 3 alternatives kept per iteration.

**Gaming detector**: If any *single* accepted iteration improves on the previous best by more than `anti_gaming.gaming_speedup_threshold` (default 10x), the agent logs a warning and an `anti_gaming_warning` event. The comparison is deliberately **incremental, not cumulative** — a naive Python baseline replaced by NumPy is a legitimate 1000x+ cumulative jump, and an earlier cumulative-vs-baseline detector fired on essentially every built-in task. It is also **warn-only**: false positives are inevitable when the agent finds a qualitatively different algorithm, and killing a real win costs more than a noisy log line. The gain is computed with `improvement_factor()`, which is mode-aware so minimize-mode metrics can cross the threshold too.

**Progress reporting**: `run_agent()` accepts an optional `progress` parameter implementing the `AgentProgress` protocol (see [Section 31](#31-progress-callback)). The agent calls `progress.on_message()` at key points (baseline complete, iteration start/end, candidate results, acceptance). The CLI uses `PrintProgress` (prints to stdout); the MCP server uses `ListProgress` (collects for polling).

**Hardware mismatch detection**: If `task.yaml` specifies `target_hardware` (e.g., `"H100"`), the agent compares it against the detected GPU using bidirectional case-insensitive substring matching (`phases/baseline._check_hardware_mismatch()`). A mismatch produces a warning in the log and a prominent amber banner in the HTML dashboard, noting that roofline analysis and optimization hints may be inaccurate. Mismatches warn but do not block the run.

**External provider injection**: `run_agent()` accepts an optional `provider: LLMProvider | None` parameter. When provided, the agent uses this provider instead of creating one from `llm_config`. This is used by the MCP server's `optimize_task` tool to inject an `MCPSamplingProvider` that delegates to the client's own LLM (see [Section 30](#30-mcp-server)).

**Iteration state artifact**: After each iteration, the agent saves a `state.json` to the run directory (iteration index, best value, accepted patches, failure memory, compiler diagnostics) — a direct `AgentContext` snapshot for debugging and post-hoc inspection.

**Resolved config**: At the start of each run, the fully resolved `PerfLabConfig` is saved as `resolved_config.json` in the run directory for reproducibility.

### Post-optimization

After the loop completes:

1. **Generate optimization summary**: An additional LLM call that explains what was changed and why
2. **Write reports**: Markdown, HTML dashboard, metric plots
3. **Return `AgentResult`**: Contains `best_value`, `best_iter`, `baseline_value`, `history`, `run_dir`

### AgentContext: the context object

**File**: `perflab/optimizers/agent.py` (`AgentContext` dataclass)

The agent loop uses the **Context design pattern** to manage its state. Instead of passing 18+ parameters between functions (the pre-refactor accept handler took 18 parameters), all mutable state lives in a single `AgentContext` dataclass that is threaded through every phase handler.

**Why Context, not State Machine:** The agent loop is mostly a linear pipeline (baseline → prompt → LLM → prescreen → benchmark → accept → repeat), not a complex state graph with many legal transitions. The real complexity is the *data* — 30+ variables that flow between phases — not the control flow. The Context pattern addresses this directly: each handler reads what it needs and writes what it produces, and the context is the single source of truth.

**Field groups** (organized by lifecycle):

| Group | Fields | Lifecycle |
|-------|--------|-----------|
| **Immutable inputs** | `task`, `config`, `llm_config`, `provider`, `progress`, `ws`, `rp`, `event_log`, `expert_suggestion` | Set at construction, never mutated |
| **Baseline state** | `baseline_val`, `sec_metric`, `baseline_sec_val`, `sysinfo`, `hardware_mismatch`, `prior_run_context` | Set once during baseline phase |
| **Evolving state** | `iteration`, `best_value`, `best_iter`, `accepted_count`, `history`, `accepted_patches`, `failure_memory`, `latest_diagnostics`, `prev_summaries`, `profiler_summaries` | Mutated across iterations by handlers |
| **Per-iteration transient** | `last_errors`, `promising_alternatives` | Reset at the start of each iteration |
| **Accumulating metadata** | `total_llm_calls`, `total_input_tokens`, `total_output_tokens`, `total_llm_latency`, `user_actions`, `early_stop_reason`, `convergence`, `wall_start` | Counters and logs that grow monotonically |

**How handlers use ctx:**

Every phase handler in the agent loop takes `ctx: AgentContext` as its first parameter. This replaces the old pattern of passing 7-18 individual arguments to each function:

```python
from perflab.optimizers.phases import autotune, baseline, evaluate, finalize, generate, prescreen

# baseline.run measures the incumbent, profiles it, and snapshots the workspace
baseline.run(ctx)

# generate.run builds the prompt from ctx, calls the LLM, returns parsed candidates
result = generate.run(ctx)

# evaluate.evaluate_single_candidate reads ctx for task/ws/rp/iteration/event_log
candidate, errors = evaluate.evaluate_single_candidate(ctx, ci, blocks, reasoning, use_fast)

# evaluate.accept_best reads and mutates ctx (best_value, history, accepted_patches, ...)
accepted, rel_improvement, accepted_value = evaluate.accept_best(ctx, candidates, backup_dir, use_fast)

# autotune.run reads ctx.task, ctx.rp, ctx.progress, ctx.event_log, ctx.iteration, ctx.best_value
autotune.run(ctx)

# finalize.maybe_early_stop consults ctx.convergence; finalize.run writes reports
if finalize.maybe_early_stop(ctx, it):
    break
finalize.run(ctx, status="completed")
```

**State serialization:**

`AgentContext.to_dict()` serializes all JSON-safe fields (iteration, best_value, history, failure_memory, latest_diagnostics, etc.) while excluding non-serializable objects (task, provider, event_log). The agent writes this to `state.json` in the run directory after each iteration, giving external tooling (or the MCP server) a direct snapshot of agent state without replaying the event log.

```python
# Saved after each iteration by run_agent()
(run_dir / "state.json").write_text(json.dumps(ctx.to_dict()))
```

---

## 10. Patch system

**File**: `perflab/optimizers/patch.py`

### Search/replace format

The LLM proposes changes using this format:

```
FILE: relative/path/to/file.py
<<<<<<< SEARCH
exact text to find in the file
=======
replacement text
>>>>>>> REPLACE
```

### Parsing (`parse_patch_response`)

Scans the LLM response line-by-line looking for `FILE:` markers followed by `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` delimiters. Handles cases where the SEARCH marker appears before the FILE marker (searches backwards for the nearest FILE line).

### Validation (`validate_patch`)

For each `SearchReplaceBlock`:

1. **Protected file blocklist**: The block's filename (basename) is checked against `PROTECTED_FILENAMES = {"tests.py", "bench.py", "task.yaml"}`. If it matches, the block is rejected immediately. This is a defense-in-depth measure that prevents the LLM from modifying test harnesses or benchmark scripts, regardless of `allowed_paths`.
2. **Path containment**: `(workspace / file_path).resolve()` must start with `workspace.resolve()`. This prevents `../../etc/passwd` attacks.
3. **Edit policy**: `fnmatch.fnmatch(file_path, pattern)` against each pattern in `allowed_paths`
4. **File existence**: The target file must exist
5. **Exact match**: `block.search in file_content` — the search text must appear verbatim

If exact match fails, `_diagnose_match_failure()` uses `difflib.SequenceMatcher` to find the closest matching region and produces a helpful diagnostic showing exactly where the text differs (including column number). This helps the LLM self-correct on the next iteration.

### Backup and restore

```python
def backup_files(blocks, workspace, backup_dir) -> dict[str, Path]:
    # Copies each affected file to backup_dir before patching

def restore_files(backed_up, workspace) -> None:
    # Copies backup files back to their original locations
```

The agent loop uses a `try/finally` pattern:

```python
backed_up = backup_files(blocks, ws, backup_dir)
try:
    apply_patch(blocks, ws)
    # ... correctness, benchmark ...
finally:
    restore_files(backed_up, ws)
```

This ensures files are always restored after evaluation, regardless of exceptions.

---

## 11. LLM prompt construction

**File**: `perflab/optimizers/prompt.py`

### System prompt

The LLM receives a system prompt that:
- Assigns the role of "expert performance engineer"
- Explains the search/replace edit format
- Lists the rules (exact match, allowed paths, correctness preservation)
- Instructs the LLM to explain reasoning before each edit
- Requests multiple candidates separated by `--- CANDIDATE N ---`

### User prompt (built by `build_prompt`)

The user message is assembled from a `PromptContext` and includes these sections in order:

1. **Source files**: Full content of each editable file, with file paths
2. **Profiler summaries**: JSON dumps of each profiler's structured output (including memray, lock contention, eBPF, and GPU memory when available — these specialized profilers feed into `PromptContext` and are rendered alongside the standard profiler data)
3. **GPU-aware profiler context**: `_add_profiler_context()` — if the workload is GPU-bound and py-spy data is present, inserts a warning that CPU hotspots reflect GPU wait time, not CPU-side inefficiency (see below)
4. **Training phase breakdown**: If the torch profiler summary contains `phases`, a Markdown table showing per-phase Time (ms), % of Total, GPU Time (ms), and CPU Time (ms)
5. **Benchmark results**: Current `bench.json` contents
6. **Device-specific guidance**: MPS backend warnings, CPU-only notes
7. **Performance vs peak**: If roofline data is available, shows % of peak and headroom
8. **Roofline analysis**: Peak hardware specs
9. **Bottleneck diagnosis**: Table of ranked diagnoses with root causes and suggested actions
10. **Hot loop assembly**: For C++/CUDA tasks with `perf annotate` data — small disassembly snippets centred on the hottest instruction in each function, showing whether the compiler is emitting SIMD or scalar code
11. **Optimization playbook**: `_build_optimization_playbook()` — a prioritized, contextual guide based on roofline utilization tier, bottleneck diagnoses, detected optimizations, and history (see below)
11. **Expert suggestion**: If the user passed `--suggest`, it's included here
12. **Error feedback**: If the previous iteration had correctness failures or contract violations, the error text is shown so the LLM can avoid repeating the same mistakes
13. **Optimization history**: Previous iterations with accepted/rejected status
14. **Request**: "Please propose N diverse optimization candidates"

### Context window management

Each agent iteration builds a complete prompt from scratch — there is no multi-turn conversation or accumulated chat history. The LLM receives the full context it needs every time (source files, profiler data, bottleneck diagnosis, history) as a single system message + user message pair.

Three layers prevent context window overflow:

**Layer 1 — Pre-send budget trimming** (`_trim_to_budget`): After assembling the full prompt, `build_prompt()` estimates total tokens (~4 chars/token) and checks against a budget. The budget comes from either `constraints.prompt_token_budget` in task.yaml (explicit) or `infer_context_budget(model)` which looks up known model context windows (e.g., GPT-5.2 → 400K, Claude Sonnet → 200K) and reserves 10% + completion tokens. If over budget, sections are removed in priority order (lowest first):

1. Prior run context
2. Profile diffs
3. GPU attribution
4. Optimization insights
5. Build flag recommendations
6. Training phase breakdown
7. Bottleneck diagnosis
8. Roofline analysis
9. (emergency) Truncate history to last 2 entries
10. (emergency) Truncate profiler summaries to 2000 chars

Source files, benchmark results, and the request itself are never trimmed.

**Layer 2 — Model context window lookup** (`infer_context_budget`): A table of known context windows for common models (GPT-5.2, Claude, Llama, DeepSeek, etc.) is used to auto-infer a safe budget when none is explicitly configured. The budget is `(context_window - max_completion_tokens) * 0.9`.

**Layer 3 — Emergency overflow recovery** (`agent.py`): If the LLM API still returns a context length error despite trimming, the agent catches the exception, halves the current token count, re-trims, and retries once. If that also fails, the iteration is skipped and the loop continues.

### Error feedback

Three types of feedback flow into the LLM prompt:

1. **`last_errors`** — Error dicts from the *previous* iteration's failed candidates. Each dict has `type` (correctness/contract_violation), `description`, and `output` (stderr, truncated to 3000 chars). Reset each iteration.

2. **`failure_memory`** — Structured failures from *all* iterations. Unlike `last_errors`, this persists across the entire run. Each entry records: iteration, strategy description, failure type, reason, and profiler context. Rendered as "Failed approaches (DO NOT repeat these)" to prevent the LLM from repeating dead-end strategies.

3. **`promising_alternatives`** — Good-but-not-best candidates from the *previous* iteration. Each entry records: description, reasoning, value, improvement ratio. Rendered as "Promising alternatives — consider combining them with the accepted approach" to encourage the LLM to merge complementary optimizations.

### GPU-aware profiler context

When a Python function dispatches GPU kernels, py-spy samples it as "hot" because the CPU thread blocks waiting for the GPU. The agent might see `forward()` at 80% and try to optimize the Python function, when the real bottleneck is the GPU kernel.

`_add_profiler_context()` detects GPU-bound workloads and warns the agent. Detection uses three helpers:

**`_is_gpu_bound(ctx)`** checks multiple signals:
- torch_profiler: `total_gpu_kernel_us > total_cpu_op_us`
- nsys: `gpu_active_pct > 50`
- metal_trace: `gpu_time_total_ms / duration_s > 0.3`
- MPS fallback: if device is "mps", assume GPU-bound (torch can't see Metal kernels)

**`_identify_gpu_dispatch_functions(summaries)`** cross-references py-spy hotspot function names against known GPU dispatch patterns: `torch`, `cuda`, `aten::`, `forward`, `backward`, `_call_impl`, `cudnn`, `cublas`, `triton`.

When GPU-bound AND py-spy data is present, the prompt includes:
```
GPU-bound workload detected. The py-spy CPU hotspots above reflect time
spent waiting for GPU kernel completion, not CPU-side inefficiency.
Functions showing high samples (`forward` (model.py:10, 80%)) are GPU
dispatch points. Focus on torch profiler / nsys / ncu data for GPU kernel analysis.
```

No annotation is added for CPU-bound workloads where py-spy data is directly useful.

### Roofline-driven optimization playbook

`_build_optimization_playbook()` builds a prioritized, contextual playbook using roofline data, bottleneck diagnoses, detected optimizations, CUTLASS baseline hints, and history.

**Step A — Utilization tier** (from % of peak TFLOPS):

| Range | Tier | Focus |
|-------|------|-------|
| <10% | structural | Wrong device, no batching, Python overhead |
| 10-30% | standard | torch.compile, AMP, precision, memory format |
| 30-60% | kernel | Memory access patterns, shared memory, custom kernels |
| 60-80% | fine_tune | Occupancy, register pressure, tile sizes |
| >80% | micro | Micro-optimizations only |

No roofline data → shows "standard" + "kernel" tiers (broadest useful coverage).

**Step B — Optimization status checklist** (`_build_optimization_checklist()`):

Merges three data sources into a unified checklist:
- `_detect_existing_optimizations()` → `[x] already present` or `[ ] not yet tried`
- History scan for rejected iterations → `[~] tried iter N, rejected`

History scanning matches optimization keywords (`torch.compile`, `autocast`, `channels_last`, etc.) in rejected history entries' `description` field.

**Step C — Priority actions**: Pulls `suggested_actions` from the #1 ranked bottleneck diagnosis and presents them as the top-priority items.

**Step D — Tier-appropriate actions**: A dict `_TIER_ACTIONS` maps `(tier, program_type)` → 3-4 prioritized actions. Covers all tiers (structural, standard, kernel, fine_tune, micro) for all program types (pytorch, cuda, triton, jax, cpp, python). Already-applied optimizations are filtered out.

Example playbook output:
```
## Optimization playbook

**Utilization: 15.0% of 989.0 TFLOPS peak (NVIDIA H100-SXM) — standard optimization tier**
**Diagnosed bottleneck: Memory-bound kernel (mem=85%, compute=23%) [high confidence]**

### Priority actions (from bottleneck analysis)
1. Use shared memory tiling to reduce global memory traffic
2. Improve data reuse via blocking / loop tiling

### Tier-appropriate optimizations (standard)
- Apply torch.compile() with inductor backend
- Enable AMP (torch.autocast) for mixed precision
- Set torch.set_float32_matmul_precision('high') for TF32

### Optimization status
- [ ] torch.compile — not yet tried
- [ ] AMP / autocast — not yet tried
- [x] SDPA / flash attention — already present
- [~] channels_last — tried iter 2, rejected
```

### Optimization detection

`_detect_existing_optimizations()` uses a data-driven pattern table (`_OPTIMIZATION_PATTERNS` dict mapping program type to `(regex, found_message, not_found_message)` tuples) to scan non-comment source lines for known patterns:

- PyTorch: `torch.compile`, autocast/AMP, SDPA/flash attention, `channels_last`, `pin_memory`, `num_workers`, `float32_matmul_precision`, `inference_mode`/`no_grad`, cuDNN benchmark mode, `persistent_workers`, `prefetch_factor`, `non_blocking` transfers, CUDA Graphs (PyTorch API: `CUDAGraph`, `make_graphed_callables`, `reduce-overhead`, `cudagraphs` backend), `torch.cuda.empty_cache`, gradient accumulation (`accumulation_steps`, `gradient_accumulation`, `accum_steps`)
- JAX: `jax.jit`, bfloat16/float16, buffer donation
- CUDA: `__shared__`, `__launch_bounds__`, tiling patterns, CUDA graphs (`cudaGraphLaunch`), pinned memory (`cudaMallocHost`), async operations (`cudaMemcpyAsync`), cooperative groups
- C++: SIMD intrinsics, OpenMP, threading, CUDA API usage (`cudaMalloc`, `<<<`), `__restrict__` pointers
- Python: numpy usage, nested loop patterns

This detection feeds into the optimization playbook's status checklist, preventing the LLM from suggesting changes that are already present or have been tried and rejected.

---

## 12. Convergence detection

**File**: `perflab/optimizers/convergence.py`

The `ConvergenceDetector` tracks the agent loop and triggers early stopping:

### Stopping conditions

1. **Consecutive failures**: If `max_consecutive_failures` (default 3) iterations in a row produce no accepted candidate, stop.
2. **Diminishing returns**: If the last 2 accepted improvements were both below `min_relative_improvement` (default 5%), stop.

### Interface

```python
convergence.record_improvement(relative_delta)  # Called on acceptance
convergence.record_failure()                      # Called on rejection
should_stop, reason = convergence.should_stop()   # Check after each iteration
```

The agent loop calls these at the end of each iteration and breaks if `should_stop` returns True. The reason string is logged to the event log and included in reports.

---

## 13. Fast screening

Fast screening is a two-tier benchmarking strategy that reduces evaluation time during agent iterations.

### How it works

1. **Screen phase**: All candidates are benchmarked with `PERFLAB_BENCH_WARMUP=0` and `PERFLAB_BENCH_REPEATS=2`. This is fast but noisy — it's only used for ranking, not decisions.
2. **Confirm phase**: Only the top candidate (the one with the best screening metric) gets a full-fidelity re-benchmark with default warmup/repeats.
3. **Decision**: The full-fidelity value is used for the accept/reject decision.

### Why it matters

Without fast screening, every candidate gets a full benchmark (potentially 3+ seconds of warmup + 20 repeats). With 6 candidates per iteration, this adds up fast. Fast screening reduces per-candidate evaluation to ~1 second while still producing accurate enough rankings to identify the best candidate for full re-benchmarking.

### Implementation

In `agent.py`:
```python
use_fast = config.fast_screen and len(candidate_blocks) > 1
# ...
_, bench = run_benchmark(task.benchmark.cmd, cwd=ws, fast_mode=use_fast)
```

In `runners/benchmark.py`:
```python
if fast_mode:
    run_env["PERFLAB_BENCH_WARMUP"] = "0"
    run_env["PERFLAB_BENCH_REPEATS"] = "2"
```

Benchmark harnesses must read these env vars:
```python
warmup  = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 20))
```

---

## 14. LLM providers

**Files**: `perflab/llm/base.py`, `perflab/llm/config.py`, `perflab/llm/openai_provider.py`, `perflab/llm/anthropic_provider.py`, `perflab/llm/ollama_provider.py`, `perflab/llm/mcp_sampling_provider.py`

### Provider protocol

```python
class LLMProvider(Protocol):
    def complete(self, messages: list[Message], temperature: float, max_tokens: int) -> CompletionResult: ...
    def stream(self, messages: list[Message], temperature: float, max_tokens: int) -> Iterator[str]: ...
```

`Message` has `role` (`system` / `user` / `assistant`) and `content` (string).

`CompletionResult` has `content`, `finish_reason`, and `usage` (dict with token counts).

### Available providers

| Provider | Class | Package | Notes |
|----------|-------|---------|-------|
| `openai` | `OpenAIProvider` | `openai>=1.0` | Default. Works with any OpenAI-compatible API via `api_base` |
| `anthropic` | `AnthropicProvider` | `anthropic>=0.30` | Claude models |
| `ollama` | `OllamaProvider` | (none) | Local models via HTTP API. SSRF-protected: `api_base` restricted to localhost by default (see [Layer 20](#layer-20-ollama-ssrf-prevention)). HTTP response status is validated on both `complete()` and `stream()` calls |
| `mcp-sampling` | `MCPSamplingProvider` | (none) | Client's LLM via MCP sampling protocol |

### MCP sampling provider

`MCPSamplingProvider` is not user-configured — it is created internally by the MCP server's `optimize_task` tool. It bridges the sync `LLMProvider.complete()` interface to the async MCP sampling protocol:

1. Constructor takes `_sample_fn` (async callable) and `_loop` (event loop)
2. `complete()` extracts the system message, calls `asyncio.run_coroutine_threadsafe(sample_fn(...), loop).result(timeout=300)` to bridge sync→async
3. Returns `CompletionResult` with `usage={}` (MCP sampling doesn't report token counts)
4. `stream()` raises `NotImplementedError` (the agent only uses `complete()`)

Zero FastMCP imports — the `sample_fn` closure is created in `mcp_server.py` and captures the MCP `Context`.

### Configuration

Config is loaded from `~/.config/perflab/config.yaml` with env var overrides. Use `perflab init` for interactive setup with live model validation. API keys (`PERFLAB_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are stripped from the subprocess environment so benchmark code cannot access them (see [Layer 17](#layer-17-secret-filtering)).

```yaml
llm:
  provider: "openai"
  model: "gpt-5.2"
  api_key: "sk-..."
  temperature: 0.7
  max_tokens: 4096
```

Environment variables (override YAML):
- `PERFLAB_LLM_PROVIDER`
- `PERFLAB_LLM_MODEL`
- `PERFLAB_API_KEY`
- `PERFLAB_API_BASE`

The `create_provider()` factory uses lazy imports so optional dependencies (`openai`, `anthropic`) are only loaded when actually needed.

---

## 15. Run storage and event logging

### Run storage (`memory/run_store.py`)

`RunStore` creates timestamped directories for each run:

```
out/runs/20250302-143622-a1b2c3d4/
  meta.json           # {"run_id": "...", "task": "...", "created_at": "..."}
  bench.json          # Latest benchmark output
  system_info.json    # CPU, GPU, OS, load average at run start
  report.md           # Markdown report
  dashboard.html      # Interactive HTML dashboard
  report.json         # Machine-readable report data
  metric_history.png  # Metric vs iteration plot
  artifacts/          # Profiler outputs
    pyspy_speedscope.json
    torch_trace.json
    *_summary.json    # Structured profiler summaries
    roofline.png
  logs/               # Command stdout/stderr
    build.stdout.txt
    correctness.stdout.txt
    bench.stdout.txt
  snapshots/          # (agent only) Source code archives
    baseline.zip      # Allowed files before any optimization
    iter1.zip         # After first accepted patch
    iter2.zip         # After second accepted patch
  llm_responses/      # (agent only)
    iter1_response.txt
    iter2_response.txt
  backups/            # (agent only) File backups per iteration
    iter1/
    iter2/
  agent_events.jsonl  # (agent only) Structured event log
  optimization_summary.md  # (agent only) LLM-generated explanation
```

Run IDs use the format `YYYYMMDD-HHMMSS-<8-hex>`. An index is maintained at `out/runs/index.jsonl`.

### RunStore read path

`RunStore` provides four read methods for querying stored runs:

| Method | Purpose |
|--------|---------|
| `list_runs(task, limit)` | List runs from `index.jsonl`, newest first. Enriches entries from `meta.json`. Optionally filter by task name. |
| `get_run(run_id)` | Load full run data: meta, report, bench, profiler summaries. Raises `FileNotFoundError` if the run directory doesn't exist. |
| `compare_runs(run_a, run_b)` | Compare two runs: extracts best values, computes delta/ratio, extracts metric context (name, mode, task), run status, and diffs bottleneck lists (resolved vs new). |
| `update_meta(run_id, updates)` | Merge key-value updates into a run's `meta.json`. |

`new_run()` accepts an optional `program_type` parameter that is stored in `meta.json` and propagated to the CLI's `list-runs` output.

Both `profile_only()` and the agent's `run_agent()` call `update_meta()` on completion, writing `status` (`"profiled"` or `"completed"`), `best_value`, and `completed_at`. This ensures `list-runs` displays consistent metadata regardless of run type, and `compare` can show the status of each run.

`compare_runs()` extracts metric context (`metric_name`, `metric_mode`, `task_name`) and per-run status from `report.json` (preferred) or `meta.json` (fallback). The CLI uses `metric_mode` to produce context-aware ratio labels: "Improvement"/"Regression" for maximize metrics, "Speedup"/"Slowdown" for minimize metrics.

### Event logging (`optimizers/event_log.py`)

The `AgentEventLog` writes JSON Lines to `agent_events.jsonl`. Each event has a timestamp, event type, iteration number, and type-specific data.

Event types in order of a typical run:

| Event | When | Key data |
|-------|------|----------|
| `baseline_complete` | After baseline | `value`, profiler names |
| `roofline_detected` | After auto-detection | Peak TFLOPS, bandwidth, source, device |
| `error_feedback` | Start of iteration | Error count, error types/descriptions from previous iteration |
| `llm_request` | Before LLM call | Prompt length, model, n_candidates, prompt_token_budget |
| `llm_response` | After LLM call | Response length, tokens used, path to full response |
| `candidate_patch` | Per candidate | Block previews (file, search, replace) |
| `candidate_validation` | Per candidate | Valid/invalid, error messages |
| `candidate_correctness` | Per candidate | Passed/failed, exit code |
| `candidate_benchmark` | Per candidate | Metric value |
| `candidate_accepted` | On acceptance | Value, delta, speedup |
| `iteration_complete` | End of iteration | Best value, whether any accepted |
| `drift_check` | Every 3 accepted patches | Clean value, last accepted value, drift % |
| `early_stop` | On convergence | Reason string |
| `run_complete` | End of run | Final stats |

Full LLM responses are saved to separate files (`llm_responses/iterN_response.txt`) to keep the JSONL compact.

`perflab replay <run_dir>` reads the JSONL and produces a human-readable timeline.

---

## 16. Reporting

**Files**: `perflab/reporting/report_md.py`, `perflab/reporting/dashboard_html/`, `perflab/reporting/plots.py`, `perflab/reporting/roofline.py`, `perflab/reporting/generate.py`

### `generate_reports()` (`reporting/generate.py`)

The report generation logic is extracted into a standalone function that can be called from both the orchestrator and the agent loop. It takes a single `ReportParams` parameter object instead of 19+ keyword arguments:

```python
@dataclass
class ReportParams:
    run_dir: Path
    run_id: str
    task_name: str
    metric_name: str
    metric_mode: str
    program_type: str
    history: list[dict]
    baseline_val: float
    best_value: float
    best_iter: int
    early_stop_reason: str | None = None
    optimization_summary_text: str | None = None
    analysis_thresholds: AnalysisThresholds | None = None
    accepted_patches: list[dict] | None = None
    roofline_peaks: dict | None = None
    llm_stats: dict | None = None
    target_hardware: str | None = None
    detected_hardware: str | None = None
    build_cmd: str | None = None
    secondary_metric_name: str | None = None
    secondary_metric_mode: str | None = None
    top_n: int = 3
    user_actions: list[dict] | None = None

def generate_reports(p: ReportParams) -> dict:
```

It orchestrates: metric history plot generation, artifact collection, run summary computation, bottleneck diagnosis from final profiler summaries, and writing of `report.md`, `dashboard.html`, and `report.json`.

### report.md

A Markdown file with:
- Task name, run ID, metric name
- Iteration table (iter, value, accepted, speedup, notes)
- Bottleneck diagnoses
- Run summary statistics (baseline, best, median speedup, success rate)
- Optimization summary (if agent mode)

### dashboard.html

A self-contained HTML file with embedded CSS/JS that displays:
- At-a-glance metrics (baseline, best, speedup, TFLOPS, % of peak, roofline source/device)
- Metric history chart (embedded PNG via base64)
- Iteration table with per-iteration results
- Optimization analysis (what worked / what didn't)
- Diagnostics card with collapsible sections:
  - Bottleneck diagnosis, GPU attribution, profile diff, build flags, hotspot diff
  - TMA (Top-Down Microarchitecture), power/energy, vectorization, GPU memory
  - Thread scheduling (perf sched), eBPF I/O tracing, lock contention (perf lock/c2c)
- Profiler insights (torch, py-spy, metal, nsys, ncu, jax, memray)
- Embedded flame graphs, roofline chart, Pareto frontier, Perfetto traces

### Metric plots

`plot_metric_history()` uses matplotlib to create a line chart of metric values over iterations, with accepted iterations highlighted.

### Roofline plots

`write_roofline_png()` generates a log-log roofline plot showing the achieved performance point relative to the compute and memory ceilings. Arithmetic intensity is computed from `M,N,K` for matmul tasks or from explicit `meta.flops` and `meta.bytes_moved` fields. When NCU profiler data includes DRAM read/write bytes, the roofline uses profiler-measured DRAM bytes instead of theoretical estimates for more accurate workload placement.

The function accepts an optional `dtype_peaks` parameter. When provided, it draws dotted horizontal ceiling lines for each dtype (FP64, FP32, TF32, FP16, BF16) with distinct colors and a legend. Per-dtype peak data is sourced from `_KNOWN_GPU_DTYPE_PEAKS` in `roofline_peaks.py`.

---

## 17. Roofline analysis

**File**: `perflab/roofline_peaks.py`

### Hardware peak detection

`infer_peaks(target)` auto-detects hardware capabilities:

1. **CUDA**: Queries `nvidia-smi` for GPU model, looks up known peak TFLOPS and memory bandwidth from a built-in table
2. **MPS/Metal**: Queries `system_profiler SPDisplaysDataType` for Apple GPU model
3. **CPU** (spec-based): `_estimate_cpu_peaks()` computes peak FLOPS as `cores × SIMD_width × clock_GHz` where SIMD width accounts for FMA (AVX-512 = 32, AVX2 = 16, NEON = 8 FP32 FLOP/cycle). On macOS, known Apple Silicon chip frequencies and memory bandwidths are used. On Linux, `/proc/cpuinfo` flags detect SIMD ISA and `lscpu` provides clock speed. Falls back to torch calibration if spec detection fails.

Results are cached to avoid repeated system queries. Use `PERFLAB_PEAKS_NO_CACHE=1` or `--refresh` to bypass.

### Roofline resolution

`resolve_roofline(task)` in `roofline_peaks.py` is the single shared entry point for resolving roofline peaks from a task spec. It checks for explicit `task.roofline` config first, then falls back to `infer_peaks()` auto-detection. Both `orchestrator.py` and `agent.py` delegate to this function.

### Roofline math

For a workload with arithmetic intensity `AI` (FLOPS per byte transferred):
- **Memory ceiling**: `peak_mem_bw * AI` TFLOPS
- **Compute ceiling**: `peak_tflops` TFLOPS
- **Attainable**: `min(memory_ceiling, compute_ceiling)`

The crossover point (knee) is at `AI = peak_tflops / peak_mem_bw`.

---

## 18. Safety model

PerfLab runs untrusted LLM-generated code edits. The safety model has **31 checks** across five categories — patch validation, execution sandboxing, result integrity, reward-hack mitigation, and hardware stability. See [SAFETY_CHECKS.md](SAFETY_CHECKS.md) for the full reference with engineering rationale for each check. The core architectural layers are:

### Layer 1: Protected file blocklist

`PROTECTED_FILENAMES = {"tests.py", "bench.py", "task.yaml"}` in `patch.py`. Any patch block targeting a file whose basename matches is rejected immediately, regardless of `allowed_paths`. This is a defense-in-depth measure preventing the LLM from modifying test harnesses, benchmark scripts, or task configuration.

### Layer 2: Edit policy

`task.yaml` declares `edit_policy.allowed_paths` as glob patterns. The agent can only modify files matching these patterns. Paths are narrowly scoped to specific source files (e.g., `["matmul.py"]` not `["**"]`).

### Layer 3: Path containment

`validate_patch()` resolves every file path against the workspace root and verifies `str(resolved).startswith(str(workspace.resolve()))`. This prevents path traversal attacks like `../../etc/passwd`.

### Layer 4: Exact text match

SEARCH blocks must match file content verbatim (character-for-character, including whitespace). `block.search in file_content` — if this fails, the patch is rejected. This prevents the LLM from modifying code it hasn't read accurately.

### Layer 5: Contract validation

After every benchmark run, `validate_contract()` checks that the bench JSON satisfies the task's `ContractSpec`: required fields exist, fixed parameters (matrix sizes, batch sizes) haven't been reduced, and minimum repeats were honored. This prevents the LLM from gaming benchmarks by shrinking the problem.

### Layer 6: Correctness gate

After every patch is applied, the correctness test runs. If it fails (non-zero exit), the candidate is immediately rejected and files are restored. The LLM cannot silently break functionality.

### Layer 7: Backup/restore

Before applying any patch, affected files are copied to `backups/iterN/`. After evaluation (success or failure), originals are restored from backup. The restore happens in a `finally` block, so even exceptions can't leave files in a modified state.

### Layer 8: Regression check

`is_improvement()` verifies that the new metric value is better than the current best by at least `regression_tolerance` (default 2%). This prevents accepting noisy measurements that appear to be improvements.

### Layer 9: Subprocess timeouts

Correctness tests time out after 60 seconds, benchmarks after 300 seconds. This prevents infinite loops or hung processes from blocking the agent indefinitely.

### Layer 10: Resource limits

On Linux, subprocesses run with `RLIMIT_AS` (4 GB for CPU tasks, 32 GB for GPU tasks), `RLIMIT_NPROC` (512 processes), and `RLIMIT_NOFILE` (1024 file descriptors). GPU tasks get a higher memory cap because CUDA runtimes and JIT compilers legitimately map large virtual address ranges, but the 32 GB cap still prevents runaway allocation. See [Section 29](#29-subprocess-sandboxing).

### Layer 11: No arbitrary commands

The LLM can only propose search/replace text edits. It cannot execute arbitrary shell commands, install packages, modify system files, or make network requests. The only code execution happens through the pre-defined correctness and benchmark commands. Note: mutated source code does run inside those commands, so the `ContractSpec` and resource limits provide additional containment.

### Layer 12: Candidate isolation

Each candidate is evaluated independently against the current best state. Candidates don't interact with each other. Only one candidate is accepted per iteration (the best improving one), and it's applied cleanly on top of the last accepted state.

### Layer 13: GPU thermal gate

Before each GPU benchmark, the runner checks GPU temperature. If above 80°C, it waits up to 120 seconds for cooldown to 75°C. This prevents thermal throttling from producing artificially degraded measurements that could cause false accept/reject decisions.

### Layer 14: GPU clock locking

`setup-h100.sh` locks GPU SM clocks at max frequency via `nvidia-smi -lgc` to eliminate boost/throttle variance (~22% swing on H100). Without clock locking, two identical runs can differ by 20%+ — well above the default 2% regression tolerance.

### Layer 15: GPU isolation

On multi-GPU nodes, `setup-h100.sh` sets `CUDA_VISIBLE_DEVICES=0` to pin benchmarks to a single GPU, preventing cross-GPU interference from other processes.

### Layer 16: Confirmation re-benchmark

The agent re-benchmarks the top candidate with full warmup/repeats before accepting (fast screening only ranks candidates). The orchestrator's grid search mode also re-benchmarks the winning configuration after the sweep completes.

### Layer 17: Secret filtering

`run_cmd()` strips `PERFLAB_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` from the subprocess environment. This prevents LLM-edited code from accessing API keys through `os.environ` and avoids accidental leakage in error messages or crash dumps.

### Layer 18: Bench.json anti-tampering

The benchmark runner computes a SHA-256 content hash of `bench.json` before the run (if it exists) and records the wall-clock start time. After the run, it verifies that (1) the content hash changed (catches stale/unmodified results) and (2) `bench.json` mtime is newer than the run start as defense-in-depth. This catches LLM-edited source code that pre-writes fake benchmark results.

### Layer 19: Symlink protection

`patch.read_source_files()` resolves each file path and verifies it stays within the workspace. Symlinks that point outside the workspace (e.g., `./link.py -> /etc/passwd`) are silently skipped. This prevents information disclosure to the LLM.

### Layer 20: Ollama SSRF prevention

The Ollama provider validates that `api_base` points to localhost (`localhost`, `127.0.0.1`, or `::1`) on port 11434 (the default Ollama port) and uses `http` or `https` scheme. Both host and port restrictions prevent SSRF attacks where a misconfigured `api_base` could target other local services. Override with `PERFLAB_OLLAMA_ALLOW_REMOTE=1` to bypass all checks, or `PERFLAB_OLLAMA_ALLOWED_PORTS=8080,11434` to allow specific additional ports.

### Layer 21: Contract early validation

`ContractSpec.validate()` checks the contract structure at task load time — before any benchmarks run. This catches malformed dotted paths in `required_bench_fields`, invalid `fixed_params` types, and negative `min_repeats`/`min_warmup` values early, avoiding wasted benchmark time on tasks with invalid contracts.

### Layer 22: Bench.json variance check

`validate_bench_variance()` walks the bench.json tree looking for numeric arrays (timing values, throughput measurements) with zero or near-zero variance. All values being identical across benchmark repeats is a strong signal of memoization or caching — the kernel is returning cached results instead of actually running computation. This check runs automatically on every benchmark result. It catches the "caching" reward hack where LLM-generated code uses a `static std::unordered_map` or Python dict keyed by tensor pointer addresses, exploiting PyTorch's deterministic memory allocator that reuses the same addresses across benchmark iterations.

### Layer 23: Determinism re-run

When `anti_gaming.determinism_rerun` is enabled (default: true), the correctness test runs twice. The second run sets `PERFLAB_DETERMINISM_SEED=42` in the subprocess environment, signaling the test harness to use different random inputs if it supports the convention. If the first run passes but the second fails, the candidate is flagged with a warning — the kernel may depend on buffer reuse from a prior run (the "no-op kernel" hack), specific input patterns, or uninitialized memory that happened to contain correct values. This adds ~1 second per candidate evaluation (one extra correctness invocation).

### Layer 24: Incremental gaming detector

The gaming detector compares each accepted candidate's metric against the *previous best* (not the baseline). Large cumulative speedups over a naive baseline are expected and legitimate — PerfLab's built-in tasks deliberately start from naive implementations (e.g., triple-nested Python loops for matmul) where 100x+ improvements are normal. What is suspicious is a single iteration producing a massive *incremental* jump over already-optimized code. The default threshold is 100x; configurable per-task via `anti_gaming.gaming_speedup_threshold`. Warnings are logged to `agent_events.jsonl` as `anti_gaming_warning` events but do not block acceptance — false positives are possible when the agent discovers a qualitatively different algorithm (e.g., switching from scalar to vectorized instructions).

### Layer 25: Thread injection check

When `anti_gaming.thread_count_check` is enabled (default: false, opt-in), the agent checks for a `thread_delta` field in `bench.json`'s `meta` section. If the benchmark harness reports that new threads were spawned during kernel execution beyond the allowed `max_thread_delta`, the candidate is rejected. This catches the "thread injection" reward hack where LLM-generated code spawns a background CPU thread to perform GPU work asynchronously while the timed kernel returns immediately. Enable this for GPU tasks where the kernel should be a synchronous computation.

---

## 19. Contract validation, anti-gaming, and reward-hack mitigations

**Files**: `perflab/task_spec.py` (ContractSpec), `perflab/runners/benchmark.py` (validate_contract)

### ContractSpec

Each task can declare a `contract:` section in `task.yaml`:

```yaml
contract:
  fixed_params: { M: 512, N: 512, K: 512 }
  min_repeats: 3
  required_bench_fields: ["ok", "tflops"]
```

This is parsed into a `ContractSpec` dataclass:

```python
@dataclass
class ContractSpec:
    fixed_params: dict[str, int | float]   # Must match bench["meta"] values
    min_repeats: int                        # Minimum benchmark iterations
    min_warmup: int                         # Minimum warmup iterations
    required_bench_fields: list[str]        # Fields that must exist in bench.json
```

### Validation flow

After `bench.json` is loaded in `run_benchmark()`, `validate_contract()` checks:

1. **Required fields**: Each entry in `required_bench_fields` is resolved as a dotted path into the bench dict. If any path doesn't resolve, the benchmark is rejected.
2. **Fixed parameters**: Each key in `fixed_params` is looked up in `bench["meta"]`. If the value doesn't match, the benchmark is rejected. This catches attempts to reduce matrix dimensions, batch sizes, or sequence lengths for a faster but meaningless result.

### Auto-tuning and contracts

When the agent loop accepts a code edit and `tuning.yaml` has a `sweep` section, `phases/autotune.run()` runs an automatic parameter sweep (max 15 trials). **Every sweep configuration is contract-validated**: `fixed_params` must still match, correctness tests must pass, and required benchmark fields must be present. This means the auto-tuner can't accidentally "optimize" by changing problem dimensions.

For CUDA tasks, sweep ranges are generated by `generate_sweep_around_baseline()` from `cutlass_baselines.py`, which centers the search on CUTLASS-optimal tile configurations (0.5×, 1×, 2× of each dimension). This explores near the known optimum rather than searching the full parameter space.

The sweep results are logged as `auto_tune_sweep` events in `agent_events.jsonl`, including the number of candidates tried, best value found, and winning knob configuration.

### Gaming detector

After each accepted iteration, the agent computes the incremental speedup — the ratio of the new metric to the *previous best*, not the baseline. If this exceeds `anti_gaming.gaming_speedup_threshold` (default 100x), a warning is logged to `agent_events.jsonl` as an `anti_gaming_warning` event. This doesn't block acceptance but flags results that warrant manual review.

The threshold is deliberately high (100x) because PerfLab's built-in tasks start from naive baselines where large first-iteration improvements are expected and correct. A Python matmul task commonly achieves 6,000x cumulative speedup when the agent replaces triple-nested loops with NumPy. The gaming detector focuses on *incremental* jumps (one iteration's gain over the previous best), where 100x+ from a single code edit is unusual and may indicate the benchmark contract was circumvented.

### AntiGamingSpec

Each task can declare an `anti_gaming:` section in `task.yaml`:

```yaml
anti_gaming:
  bench_variance_check: true      # Default: true. Detect zero-variance timing arrays.
  determinism_rerun: true          # Default: true. Run correctness twice with different seed.
  gaming_speedup_threshold: 100.0  # Default: 100. Warn if incremental speedup exceeds this.
  thread_count_check: false        # Default: false. Check bench.json meta.thread_delta.
  max_thread_delta: 0              # Default: 0. Allowed new threads during kernel execution.
```

This is parsed into an `AntiGamingSpec` dataclass in `task_spec.py`. All framework-level checks (variance, determinism, speedup) are enabled by default. The thread count check is opt-in because it requires the benchmark harness to report the `thread_delta` field.

### Reward-hack mitigations (perflab.harness)

**Files**: `perflab/harness/__init__.py`, `perflab/harness/*.py`

PerfLab includes a harness library of anti-gaming utilities for task authors to import in their protected `bench.py` and `tests.py` files. The mitigations are at two layers:

**Layer A — Framework-level (automatic, no task changes needed):**

These checks run inside the agent loop (`phases/evaluate.evaluate_single_candidate()`) for every candidate:

| Check | Hack mitigated | Mechanism | Configurable via |
|-------|---------------|-----------|------------------|
| Bench variance | Caching/memoization | `validate_bench_variance()` walks bench.json for timing arrays with zero or near-zero coefficient of variation | `anti_gaming.bench_variance_check` |
| Determinism re-run | No-op kernel, buffer reuse | `run_correctness_twice()` re-runs tests with `PERFLAB_DETERMINISM_SEED=42` | `anti_gaming.determinism_rerun` |
| Incremental speedup | Benchmark gaming | Compares single-iteration gain against previous best | `anti_gaming.gaming_speedup_threshold` |
| Thread count | Thread injection | Checks `bench.json meta.thread_delta` | `anti_gaming.thread_count_check` |

**Layer B — Harness helpers (task authors opt in by importing):**

These utilities are designed for use in `bench.py` and `tests.py`. Because those files are in the `PROTECTED_FILENAMES` blocklist, the LLM cannot remove or weaken these checks.

| Helper | Module | Hack mitigated | How it works |
|--------|--------|---------------|-------------|
| `SyncTimer` | `gpu_sync` | **Stream injection** — kernel launches work on a side CUDA stream while timing records only the default stream | Forces `torch.cuda.synchronize()` / `torch.mps.synchronize()` before starting and after stopping the timer, draining all device streams into the wall-clock measurement. Implements "hybrid timing" (event + full sync). |
| `cuda_sync_guard` | `gpu_sync` | Same | Context manager variant for wrapping existing timing blocks. |
| `ThreadGuard` | `thread_guard` | **Thread injection** — background CPU thread does GPU work while kernel returns immediately | Snapshots `threading.active_count()` and thread names before execution, asserts no new threads after. Configurable `tolerance` for frameworks that lazily start thread pools. |
| `assert_real_tensor` | `tensor_check` | **Lazy evaluation** — returns a `torch.Tensor` subclass that defers computation until `__eq__` | Validates: (1) `isinstance(t, torch.Tensor)`, (2) `type(t) is torch.Tensor` (not subclass), (3) has allocated storage, (4) non-null `data_ptr()`, (5) concrete shape/stride, (6) not nested. |
| `assert_deterministic` | `determinism` | **No-op kernel** / **shared memory overflow** — kernel returns stale buffer contents or uninitialized garbage | Phase 1: runs kernel N times with identical inputs, asserts all outputs match (catches non-determinism from uninitialized memory). Phase 2: runs with *different* inputs, asserts outputs change (catches no-ops that ignore input). Phase 3: optional reference function comparison. |
| `assert_ulp_close` | `precision` | **Precision downgrade** — computes in fp16 then casts to fp32, gaining speed while degrading accuracy | Computes ULP (units in last place) distance between output and fp64 reference on a random sample. Asserts p99 ULP distance is within `max_ulp` (default 16, catches fp16→fp32 but allows normal fp32 rounding). Also validates output dtype if `expected_dtype` is specified. |
| `assert_no_memoization` | `pointer_poison` | **Caching/memoization** — `static` map keyed by tensor pointer, exploiting PyTorch's deterministic allocator | Phase 1: creates inputs, runs kernel, verifies correctness (populates any cache). Phase 2: overwrites the *same tensor storage* with new random data (pointers unchanged, data changed). Phase 3: re-runs kernel, verifies result matches *new* reference. If kernel returns the *old* result, memoization is confirmed. |

**Usage example in a protected `tests.py`:**

```python
from perflab.harness import assert_real_tensor, assert_deterministic, assert_no_memoization
import torch
from my_kernel import kernel

dev = "cuda" if torch.cuda.is_available() else "cpu"
M, K, N = 256, 256, 256

# Check 1: output is a real, materialized tensor (not a lazy subclass)
output = kernel(torch.randn(M, K, device=dev), torch.randn(K, N, device=dev))
assert_real_tensor(output)

# Check 2: deterministic + not a no-op
assert_deterministic(
    fn=kernel,
    input_factory=lambda: (torch.randn(M, K, device=dev), torch.randn(K, N, device=dev)),
    reference_fn=lambda A, B: A @ B,
    atol=1e-3,
)

# Check 3: not memoizing via pointer addresses
assert_no_memoization(
    fn=kernel,
    input_factory=lambda: (torch.randn(M, K, device=dev), torch.randn(K, N, device=dev)),
    reference_fn=lambda A, B: A @ B,
    atol=1e-3,
)
print("ok")
```

**Usage example in a protected `bench.py`:**

```python
from perflab.harness import SyncTimer, ThreadGuard

dev = torch.device("cuda")
timer = SyncTimer(device=dev)
guard = ThreadGuard()

for _ in range(repeats):
    guard.snapshot()
    timer.start()
    result = kernel(A, B)
    elapsed = timer.stop()
    guard.check()  # raises if kernel spawned background threads
    times.append(elapsed)
```

**Why two layers?**

Framework-level checks (Layer A) provide automatic protection for *all* tasks without requiring task authors to modify their harness files. They catch the most common gaming patterns (caching, buffer reuse, suspicious speedups) at the runner/agent level. However, they cannot inspect the internal state of the benchmark subprocess — they can only observe external behavior (exit codes, bench.json contents, timing).

Harness helpers (Layer B) run *inside* the benchmark subprocess where they can inspect tensors, thread counts, and device state directly. They provide stronger guarantees but require task authors to import and use them. Since `bench.py` and `tests.py` are protected files, the LLM cannot remove these checks — once a task author adds them, they persist across all agent iterations.

The combination provides defense-in-depth: framework-level checks catch gaming that harness helpers might miss (e.g., if a task author doesn't use them), and harness helpers catch gaming that framework-level checks can't observe (e.g., tensor subclass tricks that don't affect bench.json).

---

## 20. System info and drift detection

**File**: `perflab/tools/sysinfo.py`, `perflab/optimizers/agent.py`

### System info capture

`collect_system_info()` returns a dict with:

| Field | Source |
|-------|--------|
| `platform` | `platform.platform()` |
| `cpu_model` | `sysctl` (macOS) or `/proc/cpuinfo` (Linux) |
| `cpu_count` | `os.cpu_count()` |
| `cpu_governor` | `/sys/devices/.../scaling_governor` (Linux only) |
| `nvidia_gpus` | `nvidia-smi` — name, memory, driver, persistence mode, throttle reasons |
| `cuda_version` | `nvcc --version` release line |
| `load_avg` | `os.getloadavg()` |

Each field is wrapped in `try/except` so missing tools don't crash the collection.

`warn_if_noisy()` returns warnings for:
- CPU governor is not `performance` (frequency scaling introduces variance)
- Load average exceeds 70% of CPU count (system is overloaded)
- GPU persistence mode disabled (adds latency to first CUDA call)
- Transparent hugepages set to `never` on Linux (may reduce memory-intensive performance)
- Active GPU throttling detected (thermal, power, or other throttle reasons)
- GPU clocks not locked at max frequency (boost/throttle variance can be ~22% on H100)
- GPU temperature above 70°C (approaching throttle zone) or 80°C (throttling likely)
- Multi-GPU node without `CUDA_VISIBLE_DEVICES` set (cross-GPU interference)

System info is written to `system_info.json` at the start of every agent and orchestrator run.

### Drift detection

Every 3 accepted patches, the agent re-runs the benchmark on the current workspace state without any temporary patch applied. It compares this "clean" value against the last accepted value to detect environmental drift (thermal throttling, background load changes, etc.).

Results are logged via the `drift_check` event type in `agent_events.jsonl`:

```json
{"event": "drift_check", "iteration": 6, "data": {
  "clean_value": 1.23, "last_accepted_value": 1.25, "drift_pct": -1.6
}}
```

Drift detection is informational — it logs but doesn't block. Large drift values (>5%) suggest the benchmark environment is noisy and results should be interpreted cautiously.

### Benchmark stability measures

The benchmark runner includes several measures to reduce environmental noise:

1. **GPU thermal gate**: Before each GPU benchmark, checks GPU temperature. If above 80°C, waits up to 120s for cooldown to 75°C. Prevents thermal throttling from contaminating measurements.
2. **Clock locking** (setup-h100.sh): Locks GPU SM clocks at max frequency via `nvidia-smi -lgc` to eliminate boost/throttle variance (~22% swing on H100 between base and boost clocks).
3. **GPU isolation** (setup-h100.sh): On multi-GPU nodes, sets `CUDA_VISIBLE_DEVICES=0` to pin benchmarks to a single GPU, avoiding cross-GPU interference.
4. **Confirmation re-benchmark**: The agent loop re-benchmarks the top candidate with full warmup/repeats before accepting (see [Section 13](#13-fast-screening)). The orchestrator's grid search mode also runs a confirmation benchmark on the winning configuration.
5. **Noisy-environment warnings**: `warn_if_noisy()` checks for unlocked clocks, high temperature, missing GPU pinning, non-performance CPU governor, and high system load before benchmarking starts.

---

## 21. Code snapshots

**File**: `perflab/optimizers/agent.py` (`_snapshot_workspace()`)

After the baseline run and after each accepted patch, the agent creates a zip archive of all files matching `edit_policy.allowed_paths`:

```
snapshots/
  baseline.zip    # Source files before any optimization
  iter1.zip       # After first accepted patch
  iter3.zip       # After third iteration (if accepted)
```

This provides a complete audit trail. You can diff any two snapshots to see exactly what changed between iterations, even if the workspace has been modified since.

The `_snapshot_workspace()` helper uses Python's `zipfile` module:

```python
def _snapshot_workspace(ws, allowed_paths, out_zip):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for pattern in allowed_paths:
            for f in ws.glob(pattern):
                if f.is_file():
                    zf.write(f, f.relative_to(ws))
```

---

## 22. Baseline drift detection and re-measurement

**File**: `perflab/optimizers/phases/evaluate.py`

Every 3 accepted patches, the agent re-runs the full benchmark on the current workspace and compares it to the value recorded when the latest patch was accepted (`drift_check` events in `agent_events.jsonl`). Drift beyond 5% indicates machine conditions changed (thermal throttling, background load) — the run-start baseline is no longer measured under comparable conditions.

When drift is detected, `remeasure_baseline()` reconstructs the baseline program by extracting `snapshots/baseline.zip` (the `allowed_paths` sources captured at run start) over a temporary copy of the workspace, rebuilds if the task has a build step, re-benchmarks it, and updates `ctx.baseline_val`. Subsequent history entries and the final report then compare against a baseline measured under current conditions; the swap is recorded as a `baseline_remeasured` event. The re-measure is best-effort — on any failure the original baseline is kept.

---

## 23. Compiler diagnostics

**File**: `perflab/analyzers/compiler_diagnostics.py`

Captures and parses diagnostic output from AOT compilers (GCC, Clang, NVCC) and JIT compilers (torch.compile/Inductor, JAX/XLA, Triton) to surface actionable optimization hints.

**Data model:**

```
OptimizationRemark                    (serializable via to_dict/from_dict)
├── file: str              # source file (C++) or kernel function name (CUDA)
├── line: int              # source line (0 for CUDA kernel-level remarks)
├── col: int | None        # source column
├── category: str          # "vectorize", "inline", "unroll", "loop", "alias", "fma",
│                          #  "register-pressure", "shared-memory", "other"
├── status: str            # "applied", "missed", "analysis"
├── detail: str            # human-readable detail
└── width: int | None      # vectorization width in bits (type-aware for Clang)

CompilerDiagnostics                   (serializable via to_dict/from_dict)
├── program_type: str
├── findings: list[str]
├── summary: str
└── remarks: list[OptimizationRemark]

CrossReferencedInsight
├── priority: str          # "high", "medium", "low"
├── category: str          # "missed-vec-in-hotspot", "vectorization-gap", "alias-blocking",
│                          #  "missed-fma", "non-unit-stride", "cuda-register-pressure"
├── source_location: str   # "matmul.cpp:17" or "_Z6kernelPf" (CUDA)
├── description: str
├── suggestion: str
└── perf_pct: float | None # % of CPU samples (C++) or GPU time (CUDA) at this location
```

**Pipeline:**
1. `detect_compiler(build_cmd)` — returns `"gcc"`, `"clang"`, `"nvcc"`, or `"unknown"`. Handles macOS where `g++` is Apple Clang.
2. `get_diagnostic_build_flags(program_type, compiler)` — returns extra compiler flags: GCC uses `-fopt-info-all-optall -gline-tables-only`, Clang uses `-Rpass=.* -Rpass-missed=.* -Rpass-analysis=.* -gline-tables-only`, NVCC uses `--ptxas-options=-v`.
3. `get_diagnostic_env_vars(program_type, compiler)` — returns env vars for JIT compilers and for C++/CUDA env-var-based flag injection.
4. Structured parsers — `_parse_gcc_remarks()`, `_parse_clang_remarks()`, and `_parse_nvcc_remarks()` produce `list[OptimizationRemark]` with source locations, categories, and vectorization widths. Clang width extraction is type-aware (`_infer_element_bits()` checks for `double`/`float`/`i8`/`half` in the remark detail). NVCC remarks use the kernel function name as `file` with `line=0`. Five flat parsers handle keyword counting for backward compatibility.
5. `parse_compiler_output()` dispatches to the correct parser based on `program_type` and `compiler`, populating both flat `findings` and structured `remarks`. Produces remarks for C++ (GCC/Clang) and CUDA (NVCC).
6. `cross_reference_diagnostics(remarks, perf_summary, cpu_isa, gpu_attribution)` — produces `list[CrossReferencedInsight]` by matching C++ remarks against `perf annotate` hotspots (±3 line window) and CUDA remarks against GPU attribution kernel names (fuzzy matching of mangled vs demangled names).

**Cross-referencing rules:**

| Rule | Applies to | Trigger | Priority |
|------|-----------|---------|----------|
| Missed vectorization at hotspot | C++ | missed vec remark + hotspot ≥5% CPU | high |
| Vectorization width gap | C++ | applied width < max SIMD | high/medium/low by gap ratio |
| Alias blocking at hotspot | C++ | alias remark + hotspot ≥5% | high/medium |
| Missed FMA at hotspot | C++ | fma remark + hotspot ≥5% | high/medium |
| Non-unit stride at hotspot | C++ | stride/access remark + hotspot ≥5% | high |
| Register pressure in hot GPU kernel | CUDA | missed reg remark + kernel ≥10% GPU | high/medium |

**Serialization:** Both `OptimizationRemark` and `CompilerDiagnostics` provide `to_dict()`/`from_dict()` methods. `AgentContext.to_dict()` includes `latest_diagnostics` so compiler remarks and cross-referenced insights appear in the per-iteration `state.json` artifact.

**Integration:** Both the agent phases and the orchestrator delegate to `runners/pipeline.run_pipeline()` for the build→correctness→benchmark→profile sequence. The pipeline detects the compiler, appends diagnostic build flags, and sets diagnostic env vars when `capture_diagnostics=True`. Structured remarks are cross-referenced with profiler hotspots, ISA features, and GPU attribution. Results flow into `PromptContext` as `compiler_diagnostics`, `cross_referenced_insights`, and `build_flag_recommendations`.

---

## 24. CPU ISA feature detection

**File**: `perflab/tools/sysinfo.py` — `detect_cpu_isa_features()`

Detects CPU SIMD capabilities and returns a dict with boolean flags (`sse`, `sse2`, `sse4_1`, `sse4_2`, `avx`, `avx2`, `avx512f`, `fma`, `neon`) and `max_simd_width_bits` (512/256/128/0).

- **Linux**: parses `/proc/cpuinfo` flags line
- **macOS x86**: queries `sysctl -n hw.optional.avx2_0` etc.
- **macOS ARM**: infers NEON from `platform.machine() == "arm64"`

Integrated into `collect_system_info()` as `info["cpu_isa"]`. Used by cross-referencing (vectorization gap detection) and build flag recommendations.

---

## 25. `perf annotate` integration

**File**: `perflab/profilers/linux_perf.py`

After the existing `perf script` step, runs `perf annotate --stdio` to produce source-line hotspot data. `_parse_perf_annotate()` extracts `[{function, hot_lines: [{file, line, pct}]}]` per function. Results stored in `summary["annotated_hotspots"]` and used for cross-referencing compiler remarks with profiler evidence.

**Hot loop assembly snippets:** `extract_hot_assembly(annotate_path)` parses the same `perf annotate --stdio` output and extracts small disassembly windows (default ±8 lines) centred on the hottest instruction in each function. Returns `[{function, hot_pct, snippet}]` for the top 3 functions above a configurable `min_pct` threshold (default 5%). The agent loop calls this for `cpp` and `cuda` tasks and passes the results to `PromptContext.hot_loop_assembly`. The prompt renders the snippets so the LLM can see whether the compiler is emitting SIMD instructions (e.g., `vmovaps`, `vfmadd231ps` for AVX) or scalar-only code in hot loops — a direct signal for vectorization opportunities.

---

## 26. GPU attribution engine

**File**: `perflab/analyzers/gpu_attribution.py`

Builds a CPU→GPU call graph and produces a unified attribution ranking using multiple temporal linking strategies.

**Data model:**

```
CpuGpuEdge
├── api_name: str              # CPU-side API ("cudaLaunchKernel")
├── kernel_name: str           # GPU kernel launched
├── stream_id: int
├── count: int                 # launch count
├── total_gpu_ms: float        # total GPU time for this edge
├── avg_launch_overhead_us: float
├── pct_of_total_gpu: float    # % of total GPU time
├── framework_op: str | None   # enriched: "aten::matmul", "triton_fused_relu"
└── caller_function: str | None # user-code function from call chain walking

AttributionEntry
├── rank: int
├── category: str              # "gpu-kernel", "launch-overhead", "pipeline-stall"
├── name: str
├── gpu_time_ms / gpu_pct: float
├── cpu_pct: float | None
├── launch_overhead_us: float | None
├── stream_id: int | None
├── caller_function: str | None # user-code caller (from call chain, py-spy, etc.)
├── framework_op: str | None    # framework-level op (from NVTX, torch trace, etc.)
├── diagnosis: str
└── suggestions: list[str]
```

**Functions:**
- `build_cpu_gpu_call_graph(correlations)` — groups by (api, kernel, stream), propagates most common `caller_function`, sorts by total GPU time
- `compute_attribution_ranking(nsys_summary, perf_summary, torch_summary, pyspy_summary)` — scored ranking using 5 linking strategies in priority order: call chain → temporal NVTX → torch trace → py-spy → fuzzy name matching
- `enrich_with_framework_context(edges, nvtx_ranges, correlations, program_type)` — temporal NVTX matching (preferred) + name heuristics (fallback)
- `detect_pipeline_stalls(per_stream_gaps, stream_utilization)` — idle stream and multi-stream serialization detection

**Temporal matching helpers:**
- `_build_nvtx_temporal_index()` — sorted (start_ns, end_ns, name) index from NVTX ranges
- `_build_torch_op_temporal_index()` — index from PyTorch operator events (us→ns conversion)
- `_build_pyspy_temporal_index()` — index from py-spy speedscope timestamped samples
- `_match_kernel_to_nvtx_temporal()` — finds most specific enclosing NVTX range for a kernel launch
- `_match_kernel_to_torch_op_temporal()` — finds PyTorch operator containing a kernel launch
- `_match_kernel_to_pyspy_temporal()` — finds Python function active during a kernel launch

**NSys integration** (`perflab/profilers/nsys_profiler.py`):
- `_extract_cpu_gpu_correlation()` — SQL JOIN of `CUPTI_ACTIVITY_KIND_RUNTIME` with `CUPTI_ACTIVITY_KIND_KERNEL` via `correlationId`
- `_extract_callchain_context()` — walks `callchainId` from RUNTIME table up through CUDA/framework internals to find user-code callers; enriches correlations with `caller_function` and `caller_module`
- `_extract_nvtx_ranges()` — now includes `start_ns`/`end_ns` timestamps for temporal matching
- `_extract_per_stream_gaps()` — per-stream kernel gap and utilization analysis

**Py-spy integration** (`perflab/profilers/python_pyspy.py`):
- Outputs speedscope JSON (`--format speedscope`) as the primary format — hotspots are derived from leaf-frame duration aggregation, and timestamped samples enable temporal GPU cross-referencing, all from a single py-spy run
- `_parse_speedscope_json()` extracts timestamped samples with `ts_ns`/`dur_ns` for temporal cross-reference
- Supports both sampled and evented speedscope profile formats

**Torch profiler integration** (`perflab/profilers/pytorch_profiler.py`):
- `_parse_torch_trace()` now collects timestamped CPU operator events (>100us) as `_raw_cpu_ops`
- Top 200 ops by duration stored in summary for temporal cross-reference with NSys

**Unified kernel dossier** (`build_kernel_dossiers()`): For CUDA tasks, joins GPU attribution (which kernel matters, % GPU time), NCU per-kernel metrics (what's wrong — stalls, TC util, coalescing), and SASS disassembly (exact instructions) into a single ranked view per kernel. Uses `_match_kernel()` for scored fuzzy name matching across NSys/NCU/cuobjdump name variants (exact → substring → base name → token overlap). The LLM sees one section per kernel with attribution ranking, NCU annotation header, and SASS listing — eliminating the need to mentally cross-reference separate profiler outputs.

**Why a custom layer on top of NSys?** NSys outputs flat lists of kernel statistics and raw correlation tuples — data, not interpretation. The attribution engine performs six transformations NSys doesn't: (1) full-stack caller identification via call chain walking, temporal NVTX, torch trace, and py-spy cross-referencing, (2) cross-source linking (GPU kernels ↔ CPU hotspots from Linux perf), (3) semantic grouping (100+ raw tuples → ~5 ranked entries), (4) pipeline stall diagnosis (rule-based detection on per-stream gaps), (5) framework-level translation (temporal NVTX matching preferred over name heuristics), (6) unified scoring with attribution bonus for entries with richer context. See [FEATURES.md](FEATURES.md) for details on each.

---

## 26b. HLO op attribution (JAX/TPU)

**File**: `perflab/analyzers/hlo_attribution.py`

The TPU equivalent of GPU attribution. Builds a weighted cost model from XLA HLO dumps to rank which operations dominate device time. Works for any JAX backend (TPU, GPU, CPU) that produces HLO dumps.

**Data model:**

```
HloOpEntry
├── op: str                     # "dot", "convolution", "pad", etc.
├── count: int                  # instances across all HLO modules
├── pct_of_ops: float           # % of total operations
├── category: str               # "compute", "memory", "control", "communication"
├── estimated_device_pct: float # weighted estimate of device time %
├── diagnosis: str
└── suggestions: list[str]

HloAttribution
├── entries: list[HloOpEntry]   # ranked by estimated device time
├── total_ops / total_modules: int
├── host_time_us / device_time_us: float | None  # from trace
└── device_fraction: float | None
```

**Cost model:** Each HLO op type has a weight reflecting relative device cost. `dot` and `convolution` (MXU-heavy) have weight 10.0; `reshape` has weight 0.1. The weighted count determines `estimated_device_pct`. This is a static heuristic — actual device time varies with tensor shapes and hardware — but it correctly ranks "dot > pad > reshape" in practice.

**Dashboard integration:** Renders as an "XLA/HLO op attribution" section in Diagnostics, with bar chart and per-op diagnoses. Analogous to the GPU attribution bar chart.

---

## 27. Differential profiling

**File**: `perflab/analyzers/profile_diff.py`

Compares profiler summaries between optimization iterations.

```
ProfileDelta
├── metric: str        # "ipc", "cache_miss_rate", "gpu_active_pct", etc.
├── before / after: float
├── delta / delta_pct: float
├── direction: str     # "improved", "regressed", "unchanged"
└── significance: str  # "high" (>20%), "medium" (>5%), "low"
```

- `compute_profile_diff(prev, curr, metric_mode)` — compares matching keys, classifies direction using higher-is-better/lower-is-better sets
- `format_profile_diff(deltas)` — compact rendering: `IPC: 0.82 → 1.41 (+72% ↑ improved)`

---

## 28. Build flag recommendations

**File**: `perflab/analyzers/build_flags.py`

Analyzes build commands against CPU ISA capabilities to recommend missing flags.

```
FlagRecommendation
├── flag: str       # "-march=native"
├── reason: str
├── impact: str     # "high", "medium", "low"
└── category: str   # "isa", "optimization", "debug"
```

Checks: missing `-march=native` when ISA supports AVX2+, `-O2` → `-O3` upgrade, `-g` → `-gline-tables-only`, sanitizer overhead warnings, CUDA `-arch=sm_XX`.

**Build flag mismatch warnings:** During the agent loop, `_warn_build_flag_mismatch()` checks if bottleneck diagnoses suggest optimizations (OpenMP, AVX, `-march=native`) that require build flags not present in the task's `build.cmd`. When detected, a `[agent] WARNING:` message tells the user to update `task.yaml`. This is necessary because `task.yaml` is a protected file — the LLM cannot modify the build command.

---

## 29. Subprocess sandboxing

**File**: `perflab/tools/shell.py`

All subprocess execution goes through `run_cmd()`, which applies platform-specific safety measures:

### Timeouts

Every subprocess call accepts a `timeout_s` parameter. Default timeouts:
- Benchmark: 300 seconds
- Correctness: 60 seconds

If a subprocess exceeds its timeout, it is killed and a `TimeoutError` is raised.

### Linux resource limits

On Linux, `run_cmd()` sets resource limits via a `preexec_fn`:

| Limit | Value | Purpose |
|-------|-------|---------|
| `RLIMIT_AS` | 4 GB CPU / 32 GB GPU (configurable) | Prevent memory exhaustion |
| `RLIMIT_NPROC` | 512 (configurable) | Prevent fork bombs |
| `RLIMIT_NOFILE` | 1024 | Prevent file descriptor exhaustion |

### GPU-aware memory limits

GPU frameworks (`cuda`, `pytorch`, `jax`, `triton`) legitimately map large virtual address spaces for device memory. Setting `RLIMIT_AS=4GB` on these processes would cause immediate crashes. GPU tasks use a 32 GB default instead — high enough for normal GPU workloads but still prevents runaway allocation from exhausting system memory.

The `rlimit_as_bytes` parameter controls this, with resolution priority:
1. **Explicit `rlimit_as_gb`** in `task.yaml` `constraints` → use that value (0 = disabled)
2. **GPU program types** (`cuda`, `pytorch`, `jax`, `triton`) → 32 GB default
3. **CPU program types** (`python`, `cpp`) → 4 GB default

```yaml
# Example: allow up to 16 GB for large-matrix CPU tasks
constraints:
  rlimit_as_gb: 16
```

`RLIMIT_NPROC` (512 processes) and `RLIMIT_NOFILE` (1024 file descriptors) are always applied regardless of program type.

### Thread safety

The `preexec_fn` mechanism uses `fork()` internally, which has undefined behavior when called from multithreaded processes. During parallel prescreening (which uses `ThreadPoolExecutor`), `run_cmd()` is called with `skip_preexec=True` to avoid this. Prescreening runs in temporary workspace copies with timeout enforcement, so the reduced sandbox is acceptable — the full sandbox is applied during the sequential benchmark phase.

### Secret filtering

`run_cmd()` strips secret environment variables (`PERFLAB_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) from the subprocess environment to prevent accidental leakage in benchmark error messages or crash dumps.

### Bench.json anti-tampering

The benchmark runner computes a SHA-256 content hash of `bench.json` before running the benchmark command (if it exists) and records the wall-clock start time. After the benchmark completes, it verifies that:
1. The content hash changed from its pre-run value (catches stale/unmodified results)
2. `bench.json` mtime is newer than the run start time as defense-in-depth (catches pre-written fakes)

This prevents LLM-edited source code from writing fake benchmark results.

### Portability

Resource limits are Linux-only (`platform.system() == "Linux"`). On macOS and other platforms, the `preexec_fn` is not set and no resource limits are enforced. **This means benchmarks on macOS run without memory, process, or file descriptor limits.** All limit-setting code is wrapped in `try/except` so failures are silent. For production use on macOS, consider running inside a Docker container.

### Production deployment

For stronger isolation, run PerfLab inside a Docker container with `--network=none` to prevent network access. The container provides filesystem and process namespace isolation beyond what resource limits alone can achieve. This is especially important because benchmark subprocesses have unrestricted network access — a compromised benchmark could exfiltrate data without container-level isolation.

---

## 30. The knob optimizer

**File**: `perflab/optimizers/propose_params.py`

`perflab optimize` uses a simpler optimization strategy than the agent: it sweeps numeric parameters in `tuning.yaml`.

### How it works

1. Load current knobs from `tuning.yaml`
2. `propose_knob_sweep()` generates candidate knob sets by scaling each numeric value up and down
3. For each candidate: save to `tuning.yaml`, run benchmark, check improvement
4. Accept the first improving candidate, or revert

This is safe (no code changes) and useful for finding good hyperparameters before running the full agent.

---

## 31. CI regression checks

**File**: `perflab/ci.py`

`perflab ci-check` runs the benchmark and compares against a saved baseline:

```bash
perflab ci-check perflab/demo_tasks/matmul/pytorch/task.yaml --save-baseline  # Save
perflab ci-check perflab/demo_tasks/matmul/pytorch/task.yaml                   # Check
```

If the current metric is worse than the baseline by more than `regression_tolerance`, the check fails (exit code 1). This can be integrated into CI pipelines to catch performance regressions.

### Key types

- `CICheckResult`: Primary check result with `passed`, `current_value`, `baseline_value`, `regression_pct`, optional `secondary: MetricCheckResult`, advisory `profiler_regressions: list[ProfilerRegression]`, and advisory `bench_variance_warnings: list[str]`.
- `MetricCheckResult`: Per-metric regression check result (`name`, `mode`, `current_value`, `baseline_value`, `regression_pct`, `regressed`).
- `ProfilerRegression`: A profiler metric that moved in the wrong direction (`metric`, `current`, `baseline`, `direction`).
- `_check_regression(current, baseline, mode, tolerance)`: Shared helper that computes regression percentage and whether the tolerance threshold was exceeded, for both maximize and minimize modes.

### Multi-metric CI (Pareto)

When a task defines `benchmark.secondary_metric`, `ci-check` automatically gates on both metrics:

1. `save_baseline` runs the benchmark once and saves both the primary and secondary metric values (plus names and modes) to `baseline.json`.
2. `run_ci_check` extracts both metrics from the bench output, compares each against its baseline using `_check_regression()`, and fails if *either* metric regresses beyond `regression_tolerance`.
3. If the secondary metric is absent from the bench output or the baseline file, it is silently skipped — no false failures.

The CLI reports which metric(s) regressed: `"CI check FAILED: tflops.median: regression of 5.2%; latency_ms.p95: regression of 12.0% (tolerance: 2.0%)"`.

### Bench variance anti-gaming

After every CI benchmark run, `validate_bench_variance(bench)` checks for zero-variance timing arrays in bench.json — a signal of memoization or caching. These warnings are surfaced in `CICheckResult.bench_variance_warnings` and displayed by the CLI, but are advisory (do not cause CI failure). `save_baseline` also records any variance warnings in the baseline JSON for reference.

### Profiler regression detection

`_detect_profiler_regressions(current_ncu, baseline_ncu)` compares NCU profiler metrics between two runs. Tracked regressions:

- **Higher-is-better** (5% threshold): `sm_utilization_pct`, `achieved_occupancy_pct`, `tensor_core_utilization_pct`, `branch_efficiency_pct`
- **Lower-is-better** (variable thresholds): `dominant_stall_pct` (10%), `bank_conflicts` (50), `sectors_per_request` (1.0)

NCU data flows through `baseline.json`:

1. `save_baseline` looks for NCU profiler summaries from the most recent run in the run store and saves them in the baseline. Users run `perflab profile` before `--save-baseline` to generate profiler data.
2. `run_ci_check` loads NCU data from the baseline, finds current NCU data from the run store, and compares them via `_detect_profiler_regressions()`.

NCU resolution is handled internally by the CI module — callers (CLI, MCP) don't need to look up or pass NCU data.

Profiler regressions are **advisory** — they appear in the output but do not cause CI failure. This avoids false failures when profiler metrics shift due to hardware variance rather than code changes.

### Internal flow

```
_run_bench_full(task)  →  full bench dict
  ├─ validate_bench_variance(bench)        →  bench_variance_warnings (advisory)
  ├─ metric_value(bench, primary)          →  primary check via _check_regression()
  ├─ metric_value(bench, secondary)        →  secondary check via _check_regression()
  │                                            (skipped if not configured or not in bench)
  └─ _detect_profiler_regressions(cur, bl) →  profiler_regressions (advisory)
                                               (skipped if NCU data unavailable)
```

---

## 32. Adding a new task

Copy `perflab/demo_tasks/_sample/` and customize:

```bash
cp -r perflab/demo_tasks/_sample tasks/my_category/my_task
```

### Checklist

1. **`task.yaml`**: Update `name`, `workspace`, `program_type`, `edit_policy.allowed_paths`, metric name
2. **Source code**: Write a deliberately naive implementation
3. **`bench.py`**: Measure the metric, write JSON, honor `PERFLAB_BENCH_WARMUP`/`PERFLAB_BENCH_REPEATS`
4. **`tests.py`**: Verify correctness against a reference implementation
5. **`tuning.yaml`** (optional): Numeric knobs for `perflab optimize`

### Design tips

- **Make it slow on purpose.** The baseline should have obvious inefficiencies that the agent can find. If you use a compiled language, don't enable optimizations that the agent couldn't add itself.
- **Keep the benchmark fast.** A single run should complete in 1-5 seconds. The agent will run it dozens of times.
- **Use small problem sizes.** The correctness test should run in under a second. Keep matrix sizes, sequence lengths, and batch sizes small.
- **Fixed seeds everywhere.** Use deterministic random number generation so results are reproducible.
- **The metric path must resolve.** If `task.yaml` says `metric.name: "tflops.median"`, then `bench.json` must contain `{"tflops": {"median": 0.123}}`.

### Validating

```bash
cd tasks/my_category/my_task
python tests.py                                     # Must exit 0
python bench.py --json out/bench.json               # Must create bench.json
python -c "import json; d=json.load(open('out/bench.json')); print(d)"
perflab profile tasks/my_category/my_task/task.yaml  # Full integration test
```

### Benchmark gaming prevention checklist

When writing a custom task, follow these guidelines to prevent the agent from "optimizing" by gaming the benchmark rather than improving the algorithm:

1. **Correctness tests cover multiple inputs.** Use at least 2-3 different input shapes or random seeds, not just one fixed case. For matmul/linalg, verify against a reference implementation (e.g. numpy) with tight tolerances.
2. **Contract enforces problem size.** Use `contract.fixed_params` in `task.yaml` to lock down dimensions (M, N, K, batch_size, seq_len, etc.). The framework checks these against `bench.json` meta values after every benchmark run.
3. **Protected files are untouchable.** `tests.py`, `bench.py`, and `task.yaml` are in a hardcoded blocklist — the agent cannot edit them regardless of `allowed_paths`.
4. **`allowed_paths` are narrow.** Only list the specific source files the agent should optimize — never `**` globs that include harness files.
5. **Minimum repeats enforced.** Set `contract.min_repeats` to prevent the agent from reducing measurement count.
6. **Use deterministic seeds.** Both the benchmark harness and correctness tests should use fixed random seeds for reproducibility.
7. **Consider output checksums.** For extra assurance, include an output hash or checksum in `bench.json` and verify it in `tests.py` to ensure the computation actually ran.
8. **Use `SyncTimer` for GPU timing.** In `bench.py`, use `perflab.harness.SyncTimer` instead of raw `time.perf_counter()` to force full device synchronization around timing, preventing side-stream tricks.
9. **Validate tensor types in tests.** For PyTorch tasks, call `assert_real_tensor(output)` in `tests.py` to reject lazy tensor subclasses that defer computation until comparison.
10. **Test determinism and reference correctness.** Call `assert_deterministic()` with a reference function and multiple input factories. This catches no-op kernels, buffer reuse, uninitialized memory reads, and identity copies in a single check.
11. **Poison pointer caches.** Call `assert_no_memoization()` in `tests.py` to defeat static caches keyed by tensor data pointers. This is especially important for C++/CUDA kernels where `static std::unordered_map` is easy to add.
12. **Check precision against fp64.** For tasks where the agent might downgrade precision for speed, call `assert_ulp_close(output, fp64_reference, max_ulp=4)` to enforce ULP-level accuracy.

### Benchmark variance control

PerfLab uses several mechanisms to ensure benchmark results are reliable:

- **Metric selection.** Acceptance decisions use the **median** of per-repeat timings, which is robust to outliers.
- **Two-tier benchmarking.** Fast screening uses hardcoded defaults of 0 warmup and 2 repeats (via `PERFLAB_BENCH_WARMUP=0` and `PERFLAB_BENCH_REPEATS=2`) to filter obviously bad candidates; the top candidate is re-benchmarked at full fidelity (the harness's own configured warmup + repeats) before acceptance.
- **Regression tolerance.** Candidates must beat the current best by the `regression_tolerance` (default 2%) to be accepted, preventing noise-driven "improvements."
- **Baseline drift detection.** Every 3 accepted patches, the agent re-runs the benchmark to check for system-level drift. Results are logged as `drift_check` events in `agent_events.jsonl`. When drift exceeds 5%, the baseline snapshot is re-benchmarked under current conditions and `baseline_val` is updated (logged as a `baseline_remeasured` event) so reported speedups compare measurements taken under the same machine conditions.
- **System info capture.** `system_info.json` records CPU model, GPU info, CUDA version, and load average at run start. Noise warnings are emitted for non-performance CPU governors and high system load.
- **Compile vs steady-state separation.** For JIT-compiled backends (JAX, `torch.compile`), benchmark harnesses enforce at least 1 warmup iteration to exclude compilation overhead from timing. Acceptance decisions reflect steady-state performance, not compilation cost.

---

## 33. Adding a new profiler

1. Create `perflab/profilers/my_profiler.py`
2. Implement the `Profiler` protocol:

```python
from perflab.profilers.base import Profiler, ProfileResult

class MyProfiler:
    name = "my_profiler"

    def is_available(self) -> bool:
        # Check if the required tool is installed
        return shutil.which("my-tool") is not None

    def run(self, bench_cmd, cwd, artifacts_dir) -> ProfileResult:
        # Run the profiler, parse output, return structured results
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # ... run tool, collect data ...
        return ProfileResult(
            name=self.name,
            artifacts={"output": str(artifacts_dir / "my_output.txt")},
            summary={"metric1": 42.0, "metric2": "value"},
        )
```

3. Register it in `perflab/profilers/__init__.py` (`select_profilers()`):

```python
if task.program_type in {"my_type"}:
    profs.append(MyProfiler())
```

4. Add analysis rules to the appropriate bottleneck sub-module (`bottleneck_gpu.py`, `bottleneck_cpu.py`, or `bottleneck_system.py`) and register the call in `bottleneck_analyzer.py`'s `diagnose_bottlenecks()`:

```python
if "my_profiler" in profiler_summaries:
    findings.extend(_analyze_my_profiler(profiler_summaries["my_profiler"]))
```

---

## 34. Adding a new LLM provider

1. Create `perflab/llm/my_provider.py`
2. Implement the `LLMProvider` protocol:

```python
from perflab.llm.base import LLMProvider, Message, CompletionResult

class MyProvider:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def complete(self, messages, temperature, max_tokens) -> CompletionResult:
        # Call your API, return CompletionResult
        ...

    def stream(self, messages, temperature, max_tokens):
        # Yield text chunks (optional, used for interactive output)
        ...
```

3. Register it in `llm/config.py:create_provider()`:

```python
elif name == "my_provider":
    from perflab.llm.my_provider import MyProvider
    return MyProvider(model=config.model, api_key=config.api_key)
```

4. Configure via `~/.config/perflab/config.yaml`:

```yaml
llm:
  provider: "my_provider"
  model: "my-model-name"
  api_key: "..."
```

---

## 35. Environment and configuration

### Config file

Location: `~/.config/perflab/config.yaml`. Created by `perflab init` or manually.

```yaml
llm:
  provider: "openai"        # openai | anthropic | ollama
  model: "gpt-5.2"
  api_key: "sk-..."
  api_base: ""               # Custom endpoint (e.g., for Azure OpenAI)
  temperature: 0.7
  max_tokens: 4096
```

### Environment variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `PERFLAB_LLM_PROVIDER` | Override LLM provider | `anthropic` |
| `PERFLAB_LLM_MODEL` | Override model | `claude-opus-4-6` |
| `PERFLAB_API_KEY` | Override API key | `sk-...` |
| `PERFLAB_API_BASE` | Custom API endpoint | `https://my-proxy.com/v1` |
| `PERFLAB_BENCH_WARMUP` | Override warmup iterations | `0` (fast screen) |
| `PERFLAB_BENCH_REPEATS` | Override repeat count | `2` (fast screen) |
| `PERFLAB_PEAKS_NO_CACHE` | Bypass roofline cache | `1` |

### Dependencies

Core (always required):
```
typer>=0.12     # CLI framework
PyYAML>=6.0     # Config and task.yaml parsing
matplotlib>=3.7 # Plots and roofline charts
```

Optional (per-provider):
```
openai>=1.0     # pip install perflab[openai]
anthropic>=0.30 # pip install perflab[anthropic]
fastmcp>=2.0    # pip install perflab[mcp]
```

Task-specific (not enforced by pyproject.toml):
```
torch           # PyTorch tasks
jax, optax      # JAX tasks
triton          # Triton tasks (requires CUDA GPU)
numpy           # Python matmul reference + various tasks
py-spy          # CPU hotspots (speedscope JSON)
```

### Health check

```bash
perflab doctor
```

Checks Python version, installed packages, profiler tools, GPU availability, and LLM configuration. For each missing tool, prints what it does and how to install it (e.g. `pip install py-spy`, `sudo apt install linux-tools-common`).

---

## 36. MCP server

**Files**: `perflab/server/__init__.py`, `perflab/server/mcp_server.py` (re-exporting facade — the entry point and test imports go through it), and the tool modules it composes: `core.py`, `authoring.py`, `runs.py`, `analysis.py`, `environment.py`, `agent_tools.py`, `task_templates.py`

PerfLab includes a [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes profiling and optimization tools for AI assistants. Built on [FastMCP](https://github.com/jlowin/fastmcp) with stdio transport.

### Architecture

```
AI Client (Claude Desktop, Cursor, etc.)
    |
    | stdio (JSON-RPC)
    |
perflab-mcp process
    |
    ├── FastMCP("perflab") — tool registration + request routing
    ├── ThreadPoolExecutor(max_workers=1) — background agent runs
    ├── _active_runs dict — tracks job status + ListProgress references
    ├── _lock — thread-safe access to _active_runs
    └── _agent_lock — concurrency guard (shared by start_agent + optimize_task)
```

### Tools (31)

#### Task inspection

| Tool | Description |
|------|-------------|
| `list_tasks` | Glob for `task.yaml` files, return name + program_type |
| `show_task` | Full effective config: benchmark, constraints, contract, edit policy, data hints, tuning knobs |
| `show_task_schema` | task.yaml schema reference: all fields, types, descriptions |
| `show_tuning_schema` | tuning.yaml schema: fixed vs tunable params, sweep syntax |
| `show_task_authoring_guide` | Step-by-step onboarding guide for creating a new task from scratch |

#### Task authoring (onboarding)

| Tool | Description |
|------|-------------|
| `create_task` | Scaffold a complete task directory with all required files for any program type |
| `validate_task` | Validate a task.yaml without running it — schema, contract, file existence checks |
| `suggest_profilers` | Recommend profiler plan based on program type and target hardware |
| `suggest_thresholds` | Recommend analysis thresholds for bottleneck detection |
| `suggest_contract` | Analyze bench.py and suggest fixed_params, required_bench_fields, min_repeats |
| `lint_bench_script` | Check bench.py for PerfLab protocol compliance (--json, env vars, GPU sync) |

#### Run management

| Tool | Description |
|------|-------------|
| `list_runs` | Query `RunStore.list_runs()`, newest first |
| `get_run` | Full run data: meta, report, bench, profiler summaries (may truncate at 100 KB) |
| `get_run_section` | Granular access to specific run sections (meta, report, bench, system_info, event_log, or individual profiler summaries by name) — bypasses 100 KB limit |
| `compare_runs` | Metric context, delta, ratio, status, resolved/new bottlenecks |
| `replay_run` | Replay and summarize an agent run from its event log |

#### Analysis (on-demand from stored run data)

| Tool | Description |
|------|-------------|
| `get_bottlenecks` | Load summaries + `diagnose_bottlenecks()` — ranked bottleneck list with confidence and suggested actions |
| `get_gpu_attribution` | GPU attribution ranking from NSys data: kernel GPU time %, CPU→GPU call graph, pipeline stalls |
| `get_profile_diff` | Compare profiler metrics between two runs: IPC, cache misses, GPU utilization, function-level hotspot shifts |
| `get_hlo_attribution` | HLO operation attribution for JAX/TPU runs: op rankings, cost estimates, dtype distribution |
| `get_build_recommendations` | Build flag recommendations from ISA detection (static) and profiler data (dynamic, with run_id) |
| `get_roofline_analysis` | Roofline analysis: arithmetic intensity, achieved TFLOPS, peak utilization %, memory bandwidth |
| `get_thresholds` | List analysis thresholds for bottleneck diagnosis, including task-specific overrides |

#### Environment

| Tool | Description |
|------|-------------|
| `get_peaks` | Inferred roofline peaks and detected hardware (CUDA GPUs, Metal/MPS, TPU) |
| `doctor_check` | Environment readiness: Python, packages, profiler tools, hardware, LLM config |

#### CI regression checks

| Tool | Description |
|------|-------------|
| `ci_check` | Run CI regression check against a saved baseline — returns pass/fail, regression %, profiler regressions |
| `save_ci_baseline` | Run benchmark and save result as CI baseline for future checks |

#### Optimization

| Tool | Sync? | Description |
|------|:---:|-------------|
| `profile_task` | Yes | Run baseline profiling, return summaries |
| `start_agent` | No | Submit agent run to thread pool, return `job_id` |
| `get_agent_progress` | Yes | Poll `ListProgress.messages[-20:]` + status |
| `optimize_task` | Async | Run agent using client's LLM via MCP sampling (no API key) |

### Two execution modes

**API-key mode (`start_agent`)**: Requires `perflab init`. Submits the agent run to a `ThreadPoolExecutor`. The closure creates a `ListProgress`, loads `TaskSpec` and `LLMConfig`, calls `run_agent()`, and updates `_active_runs` on completion. The MCP client polls `get_agent_progress(job_id)` for the latest 20 messages and final result. This is the preferred mode — non-blocking, full token tracking, progress polling.

**MCP sampling mode (`optimize_task`)**: No API key needed. Uses the MCP client's own LLM via the [sampling protocol](https://spec.modelcontextprotocol.io/specification/client/sampling/). The tool:
1. Runs a pre-flight sampling test (`"Respond with OK."`) to fail fast if the client doesn't support sampling
2. Creates a `sample_fn` closure that captures the MCP `Context` and converts `Message` objects to `SamplingMessage` for `ctx.sample()`
3. Wraps `sample_fn` + the event loop in an `MCPSamplingProvider` (see [Section 14](#14-llm-providers))
4. Runs the agent via `asyncio.to_thread(run_agent, ...)` to avoid blocking the event loop
5. Returns the result directly (blocking the MCP connection for the duration)

Trade-offs vs API-key mode: blocks the connection, no token usage stats, requires client sampling support.

### Tool annotations

Every MCP tool declares metadata hints per the MCP specification:

- **`readOnlyHint: true`** on all task inspection, run management, analysis, environment, and most authoring tools (25 tools) — enables MCP clients to auto-approve read-only calls
- **`readOnlyHint: false`** on `create_task`, `profile_task`, `start_agent`, `optimize_task`, `ci_check`, `save_ci_baseline` — clients should confirm before execution
- **`destructiveHint: false`** on `create_task`, `start_agent`, `optimize_task` — non-destructive writes (backups are created)
- **`idempotentHint: true`** on `profile_task`, `ci_check`, `save_ci_baseline` — safe to retry on failure
- **`openWorldHint: true`** on `list_tasks` — task list can change between calls

These annotations are advisory — the MCP server does not enforce them.

### Output size guard

`_guard_output_size()` enforces a `_MAX_OUTPUT_BYTES = 100_000` (~100 KB) limit on tool responses. When output exceeds the limit, the function returns a structured object with `_truncated: true`, a notice directing clients to `get_run_section` for granular access, `_partial_data` (first ~50 KB), and `_original_size_bytes`. The `get_run_section` tool bypasses this limit by returning individual sections (e.g., a single profiler summary) rather than the entire run.

### Concurrency guard

Both `start_agent` and `optimize_task` share `_agent_lock` (a `threading.Lock`). Only one agent run can execute at a time — benchmark results are sensitive to system contention, and concurrent runs on the same GPU would invalidate measurements. If a second run is attempted while one is active, it is rejected immediately with an error message.

### Entry point

Registered in `pyproject.toml`:

```toml
[project.scripts]
perflab-mcp = "perflab.server.mcp_server:main"
```

Install with: `pip install -e ".[mcp]"`

---

## 37. Progress callback

**File**: `perflab/optimizers/progress.py`

The `AgentProgress` protocol provides a simple callback interface for real-time progress reporting from the agent loop:

```python
@runtime_checkable
class AgentProgress(Protocol):
    def on_message(self, message: str) -> None: ...
```

### Implementations

| Class | Behavior | Used by |
|-------|----------|---------|
| `PrintProgress` | Prints each message to stdout | CLI (`perflab agent`) |
| `ListProgress` | Appends messages to an in-memory list | MCP server (`start_agent`, `optimize_task`) |

### Integration with the agent loop

`run_agent()` accepts an optional `progress` parameter. When provided, it calls `progress.on_message()` at key lifecycle points:
- Baseline profiling complete
- Iteration start (with iteration number)
- Candidate evaluation results
- Acceptance/rejection decisions
- Early stopping triggers
- Run completion with final metrics

The MCP server's `start_agent()` creates a `ListProgress` and stores it alongside the job metadata. `get_agent_progress()` reads the last 20 messages for polling-based progress display.

---

## 32. Structured configuration

**File**: `perflab/config.py`

`PerfLabConfig` is a nested dataclass hierarchy that centralizes all configuration:

```python
@dataclass
class PerfLabConfig:
    llm: LLMSection           # provider, model, api_key, api_base
    benchmark: BenchmarkSection  # warmup, repeats
    profiler: ProfilerSection    # enabled profilers, thresholds
    mps: MPSSection              # Apple MPS-specific settings
    ollama: OllamaSection        # local Ollama model settings
```

**Resolution order** (highest priority wins):
1. Environment variables (`PERFLAB_LLM_MODEL`, `PERFLAB_BENCH_WARMUP`, etc.)
2. Local project file (`./perflab.yaml`)
3. User config file (`~/.config/perflab/config.yaml`)
4. Built-in defaults

The fully resolved config is saved as `resolved_config.json` in each run directory for reproducibility. The benchmark runner reads `BenchmarkSection` defaults for `PERFLAB_BENCH_WARMUP` and `PERFLAB_BENCH_REPEATS` when env vars are not set.

No files are required — everything works with defaults and env vars. `perflab init-config` creates `./perflab.yaml` with a commented template; `perflab init-config --user` creates the user-level config. `perflab show-config` displays the resolved config. `create_project_config()` and `create_user_config()` are the underlying functions. All `PERFLAB_*` environment variables are documented in the config template's subprocess-only comments section (see `DEFAULT_CONFIG_TEMPLATE` in `config.py`).

### Hot assembly demangling

CUDA SASS extraction (`extract_cuda_sass()`) demangles kernel names via `c++filt` with an LRU cache to avoid redundant subprocess calls for repeated symbol names.

---

## Engineering rationale: profiler tooling choices

PerfLab's profiler stack was chosen for a specific architectural constraint: **automated, uninstrumented profiling of arbitrary user code**. This section documents the trade-offs.

### Why sampling profilers (py-spy, perf) instead of instrumentation (Tracy, Optick, Perfetto SDK)

Instrumentation profilers require adding macros or API calls to the source code being profiled. PerfLab's model is "point at a task directory and go" — users should not need to modify their code before profiling. Sampling profilers attach externally and work on unmodified binaries.

- **py-spy** reads Python stack frames via `process_vm_readv` without interrupting the target process. Overhead is negligible and it works on unmodified Python code.
- **Linux perf** uses hardware performance counters (PMCs) to sample call stacks at kernel level. It provides IPC, cache miss rates, and branch misprediction rates that no pure-software profiler can replicate.
- **Tracy** (2ns per span) and **Optick** are excellent tools, but they require `#include` and span macros in the profiled code. This breaks PerfLab's zero-instrumentation model.

### Why Perfetto as a viewer format, not the SDK

The Perfetto SDK (`perfetto::TracingSession`) requires linking a C++ library and instrumenting trace points. Instead, PerfLab exports existing profiler data to Chromium Trace Event JSON — the simplest format Perfetto UI accepts. This gives users interactive timeline visualization without adding any build-time dependency or code modification.

### Why memray for memory profiling

CPU sampling profilers (py-spy, perf) cannot detect memory-bound bottlenecks: excessive allocations, peak memory pressure, or GC thrashing. These are invisible in flame graphs but can dominate real-world performance. memray captures Python heap allocations with low overhead and reports top allocators by size — exactly the data the LLM needs to suggest allocation-reducing optimizations.

### Why eBPF for syscall tracing (Linux stretch goal)

Sampling profilers miss time spent waiting in syscalls (read, write, futex). For data-loading bottlenecks where the CPU is idle waiting on disk or network I/O, `perf stat` shows high wall-clock time but low CPU utilization. eBPF via `bpftrace` traces individual syscalls with latency histograms, revealing whether the bottleneck is I/O-bound rather than compute-bound. This is Linux-only and auto-skipped on other platforms.

### Why not hardware tracers (Intel PT, ARM SPE)

Hardware instruction tracers like Intel Processor Trace provide nanosecond-resolution control flow recording with near-zero overhead. However:

- **Intel PT** is Intel-only and Linux-only — it does not work on Apple Silicon or in most virtual machines
- **ARM SPE** (Statistical Profiling Extension) is not exposed through macOS; Apple does not provide userspace access to ARM CoreSight on their chips
- Both require specialized post-processing tools (magic-trace, perf intel-pt) that add complexity without proportional value for PerfLab's use case

The data these tools provide (instruction-level traces, branch-taken logs) is more granular than what the LLM optimization loop needs. The LLM works best with function-level hotspots and hardware counter ratios — exactly what py-spy and perf stat provide.

### Profiler hotspot diffing rationale

Counter-level diffs (IPC changed by +15%) tell the LLM *whether* hardware behavior changed, but not *where*. Function-level hotspot shifts tell the LLM "the hot function moved from `naive_matmul` (45%) to `memcpy` (30%)" — giving it actionable context about what to optimize next. This is computed by merging py-spy and perf hotspot lists from baseline vs. optimized profiles.
