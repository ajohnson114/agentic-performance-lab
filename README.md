# Agentic Performance Lab (PerfLab)

A local-first CLI that **profiles**, **diagnoses**, and **optimizes** compute-bound programs. Point it at a task directory containing a benchmark harness and a correctness test, and PerfLab will identify bottlenecks and, in agent mode, let an LLM propose code edits to fix them.

Every built-in task starts from a deliberately naive baseline. The only way to improve performance is to edit the source code.

```bash
pip install -e .
perflab init                                    # configure your LLM provider
perflab agent tasks/matmul/cuda/task.yaml       # LLM-driven optimization
perflab replay out/runs/<run_id>/               # review what the agent did
```

---

## What It Does

You give PerfLab a naive program + a benchmark harness. It profiles the code, diagnoses bottlenecks, and (optionally) lets an LLM rewrite the hot path in a loop until performance converges.

```
perflab init                 # one-time LLM provider setup
perflab profile  task.yaml   # baseline: flame graph, hardware counters, GPU traces
perflab agent    task.yaml   # LLM proposes code edits, benchmarks each, keeps winners
perflab replay   out/runs/…  # human-readable timeline of what the agent did
```

**What you get in `out/runs/<run_id>/`:**

| File | Contents |
|------|----------|
| `dashboard.html` | Interactive results: bottleneck diagnoses, kernel dossiers, TMA breakdown, roofline plot, iteration history |
| `report.md` | Markdown summary with bottleneck diagnoses |
| `artifacts/` | Flame graphs, GPU traces, SASS dumps, profiler summaries, roofline PNGs |
| `agent_events.jsonl` | Full audit trail: every agent decision, build flags, auto-tune sweeps |
| `snapshots/` | Source code at each accepted iteration |

**Supported backends:** Python, C++, CUDA, PyTorch, JAX, Triton on NVIDIA GPUs, Google TPUs, Apple Silicon, or CPU-only.

**Scope:** Single-device performance optimization. Does not handle multi-GPU, multi-node, or distributed training.

---

## Getting Started

```bash
pip install -e ".[openai]"                          # or .[anthropic], .[all]
pip install -e ".[tasks-all]"                       # task dependencies (optional)
perflab doctor                                       # check environment
perflab init                                         # configure LLM provider
perflab profile tasks/matmul/python/task.yaml        # one-shot profiling
perflab agent   tasks/matmul/python/task.yaml        # LLM-driven optimization
```

**One-shot setup for rented instances:** `./setup-h100.sh` (NVIDIA GPU) or `./setup-tpu-v5e.sh` (TPU VM).

| Platform | Recommended tasks |
|----------|------------------|
| **NVIDIA GPU** | `matmul/cuda`, `matmul/cuda_tensorcore`, `matmul/triton`, `transformer_train/pytorch` |
| **Apple Silicon** | `matmul/pytorch`, `matmul/jax`, `transformer_train/pytorch` |
| **CPU only** | `matmul/python`, `matmul/cpp`, `matmul/cpp_parallel`, `stream/python` |

---

## Tasks

Each task is a self-contained directory with a naive implementation, a benchmark harness (`bench.py`), a correctness test (`tests.py`), and a config (`task.yaml`). The agent must discover and apply all optimizations through code edits.

### Featured tasks

These are good starting points for seeing what the agent can do:

| Task | Command | Optimization space |
|------|---------|--------------------|
| CUDA matmul | `perflab agent tasks/matmul/cuda/task.yaml` | Tiling, coalescing, shared memory |
| CUDA Tensor Core | `perflab agent tasks/matmul/cuda_tensorcore/task.yaml` | Double buffering, warp pipelining |
| PyTorch transformer | `perflab agent tasks/transformer_train/pytorch/task.yaml` | AMP, SDPA, `torch.compile` |
| C++ matmul | `perflab agent tasks/matmul/cpp/task.yaml` | Loop reordering, tiling, SIMD |
| Triton matmul | `perflab agent tasks/matmul/triton/task.yaml` | Block tiling with `tl.dot` |

### All tasks

| Task | Type | Naive baseline | What the agent should discover |
|------|------|---------------|-------------------------------|
| C++ matmul | `cpp` | Cache-unfriendly i,j,k loop | Loop reordering (i,k,j), tiling, SIMD |
| C++ parallel matmul | `cpp` | OpenMP but untuned | Cache tiling, NUMA, false sharing, thread tuning |
| CUDA matmul | `cuda` | One thread per element, no shared mem | Shared-memory tiling, coalescing |
| CUDA Tensor Core matmul | `cuda` | Naive WMMA kernel, no shared-mem tiling | Shared-memory tiling, double buffering, warp pipelining |
| CUDA matmul (H100) | `cuda` | Same as CUDA matmul, tuned thresholds for H100 | Tensor cores, H100-specific launch config |
| Python matmul | `python` | Triple-nested Python loops | NumPy vectorization |
| PyTorch matmul | `pytorch` | Plain `A @ B` in fp16 | `torch.compile`, `nn.Linear`, AMP |
| JAX matmul | `jax` | `jnp.matmul` in float32, no jit | `@jax.jit`, dtype selection |
| Triton matmul | `triton` | One program per element, scalar loop | Block tiling with `tl.dot` |
| PyTorch transformer | `pytorch` | fp32, naive attention, no compile | AMP, SDPA, `torch.compile` |
| JAX transformer | `jax` | float32, naive attention, no jit | `jax.jit`, efficient attention, mixed precision |
| Attention (TPU) | `jax` | fp32, no jit, Python loop over heads | `@jax.jit`, bf16, vectorized heads, TPU tile alignment |
| DataLoader bottleneck | `pytorch` | `num_workers=0`, `pin_memory=false` | Parallel loading, pinned memory |
| C++/CUDA reduction | `cpp` | Per-iteration H2D/D2H, naive kernel, sync-bound | Persistent device mem, pinned memory, shared-mem reduction, streams |
| PyTorch inference | `pytorch` | Per-image CPU preprocessing, batch_size=1, eager mode, fp32 | Batching, GPU preprocess, `torch.compile`, half precision |

---

## How the Agent Works

1. **Baseline**: profile + benchmark the naive code
2. **Build prompt**: source files, profiler summaries, bottleneck diagnoses, kernel dossier, roofline playbook, failure memory, promising alternatives
3. **LLM generates** N candidate patches (search/replace edits)
4. **Parallel prescreen**: validate + build + correctness test all candidates concurrently
5. **Sequential benchmark**: only passing candidates are benchmarked on GPU
6. **Accept** the best improving candidate, re-profile
7. **Auto-tune**: if `tuning.yaml` has a `sweep` section, sweep parameters (max 15 trials, contract-validated)
8. **Learn from failures**: structured failure memory prevents repeating dead ends
9. **Repeat** until convergence, `max_iters`, or wall-clock budget
10. **Generate** dashboard + report

All activity is logged to `agent_events.jsonl`. Use `perflab replay` to review.

---

## Commands

| Command | Purpose |
|---------|---------|
| `perflab init` | Interactive LLM provider setup |
| `perflab profile <task.yaml>` | One-shot profiling (flame graphs, traces, hardware counters) |
| `perflab optimize <task.yaml>` | Grid search over `tuning.yaml` knobs (no LLM required) |
| `perflab agent <task.yaml>` | LLM-driven beam-search optimization |
| `perflab replay <run_dir>` | Replay of an agent run from its event log |
| `perflab peaks` | Show detected hardware peaks for roofline analysis |
| `perflab ci-check <task.yaml>` | CI regression check against a stored baseline |
| `perflab list-runs` | List stored runs (newest first) |
| `perflab compare <A> <B>` | Compare two runs: metric delta, ratio, bottleneck diff |
| `perflab show-task <task.yaml>` | Show effective task config with defaults filled in |
| `perflab show-config` | Display resolved configuration |
| `perflab show-config-template` | Emit commented YAML config template |
| `perflab init-config` | Create `./perflab.yaml` with default template |
| `perflab init-config --user` | Create `~/.config/perflab/config.yaml` |
| `perflab doctor` | Environment health check |

### Agent flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--suggest` | none | Expert hint for the LLM |
| `--iters` | 8 | Max agent iterations |
| `--candidates` | 4 | Candidates per LLM call |
| `--max-time` | 3600 | Wall-clock budget in seconds |
| `--no-early-stop` | off | Disable convergence detection |
| `--no-fast-screen` | off | Disable two-tier benchmarking |

---

## Safety

The agent proposes code changes within a tightly constrained sandbox. Key layers:

- **Protected files** — `tests.py`, `bench.py`, and `task.yaml` cannot be edited
- **Edit policy** — `allowed_paths` restricts editable files to specific source files
- **Path containment** — every path is resolved and checked against the workspace root
- **Correctness gate** — every candidate runs `tests.py`; rejected on failure
- **Contract validation** — benchmark output checked against `contract.fixed_params` (prevents shrinking the problem to "optimize")
- **Backup/restore** — files backed up before patching, restored on failure
- **Regression check** — candidates must beat baseline by `regression_tolerance` (default 2%)
- **Resource limits** — memory, process, and file descriptor caps (Linux)
- **Anti-gaming** — variance checks, determinism re-runs, speedup threshold alerts

PerfLab also includes `perflab.harness`, a library of anti-gaming utilities for `bench.py` and `tests.py`:

| Helper | What it does |
|--------|-------------|
| `SyncTimer` / `cuda_sync_guard` | Forces device synchronization around timing |
| `ThreadGuard` | Rejects new background threads during execution |
| `assert_real_tensor` | Validates exact `torch.Tensor` type with real storage |
| `assert_deterministic` | Same inputs must match, different inputs must differ |
| `assert_ulp_close` | ULP-distance check against fp64 reference |
| `assert_no_memoization` | Overwrites input data in-place and re-runs |

---

## Creating a Custom Task

Copy `tasks/_sample/` and customize. A task needs: `task.yaml`, `bench.py`, `tests.py`, and source files.

```yaml
# task.yaml — minimal example
name: "my_task"
program_type: "python"             # python | pytorch | jax | triton | cpp | cuda
correctness:
  cmd: "python3 tests.py"
benchmark:
  cmd: "python3 bench.py --json out/bench.json"
  metric: { name: "throughput.median", mode: "maximize" }
contract:
  fixed_params: { M: 512, N: 512 }       # agent can't shrink these
edit_policy:
  allowed_paths: ["my_source.py"]
```

Run `perflab show-task task.yaml` to see effective config with defaults.

### MCP task authoring

If you use PerfLab through the MCP server, there are task-authoring tools to walk you through this interactively: `show_task_authoring_guide`, `create_task`, `validate_task`, `suggest_profilers`, `suggest_thresholds`, `suggest_contract`, and `lint_bench_script`.

---

## Grid Search

`perflab optimize` sweeps implementation knobs defined in `tuning.yaml` without an LLM:

```yaml
# tuning.yaml
N: 1024
block_size: 16
sweep:
  block_size: [16, 32, 64, 128, 256]
```

```bash
perflab optimize tasks/matmul/cuda/task.yaml
perflab optimize tasks/matmul/triton/task.yaml --max-trials 15
```

In agent mode, parameter sweeps happen automatically after each accepted code edit.

---

## CI Integration

```bash
perflab ci-check tasks/matmul/cpp/task.yaml --save-baseline   # save baseline (once)
perflab ci-check tasks/matmul/cpp/task.yaml                    # check in CI (exit 1 on regression)
```

Compares against the stored baseline using `regression_tolerance` from `task.yaml` (default 2%).

---

## MCP Server

PerfLab includes an [MCP](https://modelcontextprotocol.io/) server (31 tools) for AI assistants like Claude Desktop or Cursor.

```bash
pip install -e ".[mcp]"
```

Add to your client config:

```json
{
  "mcpServers": {
    "perflab": {
      "command": "perflab-mcp",
      "cwd": "/path/to/perflab"
    }
  }
}
```

Tools cover task inspection, profiling, analysis, optimization, CI checks, and task authoring.

---

## LLM Configuration

```bash
perflab init    # interactive setup — provider, model, API key
```

Or set environment variables (`PERFLAB_LLM_PROVIDER`, `PERFLAB_LLM_MODEL`, `PERFLAB_API_KEY`).

Supports OpenAI (+ compatible APIs via `api_base`), Anthropic, and Ollama.

### Configuration

**No config files are required** — PerfLab works out of the box with defaults and env vars. If you want to customize settings, create a config file:

```bash
perflab init-config         # creates ./perflab.yaml for this project
perflab init-config --user  # creates ~/.config/perflab/config.yaml for all projects
```

This writes a commented YAML template — uncomment and edit only the settings you want to change. Everything you don't touch keeps its default.

**What you can configure:**

| Section | What it controls | When to change it |
|---------|-----------------|-------------------|
| `llm` | Provider, model, temperature | Switch between OpenAI/Anthropic/Ollama |
| `benchmark` | Warmup iterations, repeat count | More repeats for noisy benchmarks, fewer for fast iteration |
| `agent` | Candidates per iteration, max iterations, wall-clock budget, history depth | More candidates if you have compute budget; shorter runs for CI |
| `profiler` | FLOPS counting, roofline cache | Disable FLOPS if it adds overhead |
| `analysis_thresholds` | Bottleneck detection sensitivity | Tune for your hardware — e.g., lower occupancy threshold for register-heavy HPC kernels |
| `mps` | Apple Silicon device selection | Multi-GPU Mac setups |
| `ollama` | Remote access, port allowlist | Self-hosted LLM setups |

**Resolution order:** env vars > `./perflab.yaml` (project) > `~/.config/perflab/config.yaml` (personal) > defaults. Individual task.yaml settings override the config for that specific task.

```bash
perflab show-config   # see the final resolved values and which files were loaded
```

---

## Prerequisites

- Python 3.10+
- Run `perflab doctor` to check your environment

PerfLab gracefully skips profilers that aren't installed. Install the ones relevant to your workload:

| Tool | What it does | Install |
|------|-------------|---------|
| py-spy | CPU hotspots ([Speedscope](https://www.speedscope.app/) viewer) | `pip install py-spy` |
| memray | Memory allocation profiling | `pip install memray` |
| perf | Hardware counters (Linux) | `sudo apt install linux-tools-common` |
| nsys | NVIDIA GPU timeline | [Nsight Systems](https://developer.nvidia.com/nsight-systems) |
| ncu | NVIDIA GPU kernel profiler | [Nsight Compute](https://developer.nvidia.com/nsight-compute) |
| toplev | Intel TMA analysis | `pip install pmu-tools` |

Compilers: `g++` for C++ tasks, `nvcc` for CUDA. Runtimes: `torch`, `jax`, `triton` as needed (`pip install -e ".[tasks-pytorch]"`).

---

## License

PerfLab is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).

You are free to use, modify, and distribute PerfLab. If you run a modified version as a network service, you must make your modifications available under the same license.
