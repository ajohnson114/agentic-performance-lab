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

### From a clone

```bash
pip install -e ".[openai]"                          # or .[anthropic], .[all]
pip install -e ".[tasks-all]"                       # task dependencies (optional)
perflab doctor                                       # check environment
perflab init                                         # configure LLM provider
perflab profile tasks/matmul/python/task.yaml        # one-shot profiling
perflab agent   tasks/matmul/python/task.yaml        # LLM-driven optimization
```

**One-shot setup for rented instances:** `./setup-h100.sh` (NVIDIA GPU) or `./setup-tpu-v5e.sh` (TPU VM).

### Without cloning

PerfLab isn't published to PyPI yet, so "pip install" today means installing the wheel built from this repo (once it's on PyPI, `pip install perflab` will work the same way). Every demo task ships inside the package, so you don't need the repo checked out to try one:

```bash
pip install .                                        # from a clone; add "perflab" once it's on PyPI
perflab tasks list                                   # see what's bundled
perflab tasks copy matmul/python .                    # copies ./matmul/python/ into the cwd
perflab init                                         # configure LLM provider
perflab agent matmul/python/task.yaml                # LLM-driven optimization
```

`perflab tasks copy` refuses to overwrite an existing directory, so it's safe to run from anywhere.

| Platform | Recommended tasks |
|----------|------------------|
| **NVIDIA GPU** | `matmul/cuda`, `matmul/cuda_tensorcore`, `matmul/triton`, `transformer_train/pytorch` |
| **Apple Silicon** | `matmul/pytorch`, `matmul/jax`, `transformer_train/pytorch` |
| **CPU only** | `matmul/python`, `matmul/cpp`, `matmul/cpp_parallel`, `stream/python` |

---

## Tasks

Each task is a self-contained directory with a naive implementation, a benchmark harness (`bench.py`), a correctness test (`tests.py`), and a config (`task.yaml`). The agent must discover and apply all optimizations through code edits.

`tasks/` at the repo root is a **mirror** of `perflab/demo_tasks/`, which is the packaged source of truth (wheel package data has to live inside the package). Either path works for *running* a task. If you *edit* a bundled task, edit it under `perflab/demo_tasks/` and then run:

```bash
python scripts/sync_demo_tasks.py          # update the mirror
python scripts/sync_demo_tasks.py --check  # what CI runs; exit 1 on drift
```

A test fails if the two trees diverge, so drift is caught before it ships.

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
| STREAM (memory bandwidth) | `python` | Column-major traversal of row-major arrays, scalar loops (cache-unfriendly) | NumPy vectorization, row-major access order |
| GPU inference demo (H100) | `pytorch` | fp32 eager, per-image CPU pre/postprocess, no AMP/compile (14 antipatterns) | AMP, `channels_last`, `torch.compile`, CUDA graphs, batched pre/postprocessing |

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
| `perflab tasks list` | List demo tasks bundled with the package |
| `perflab tasks copy <name> [dest]` | Copy a bundled demo task (e.g. `matmul/python`) into `dest` |
| `perflab profile <task.yaml>` | One-shot profiling (flame graphs, traces, hardware counters) |
| `perflab optimize <task.yaml>` | Grid search over `tuning.yaml` knobs (no LLM required) |
| `perflab agent <task.yaml>` | LLM-driven beam-search optimization |
| `perflab replay <run_dir>` | Replay of an agent run from its event log |
| `perflab peaks` | Show detected hardware peaks for roofline analysis |
| `perflab ci-check <task.yaml>` | CI regression check against a stored baseline |
| `perflab list-runs` | List stored runs (newest first) |
| `perflab compare <A> <B>` | Compare two runs: metric delta, ratio, bottleneck diff |
| `perflab show-task <task.yaml>` | Show effective task config with defaults filled in |
| `perflab show-task-schema` | Show the full task.yaml schema with all fields and types |
| `perflab thresholds` | List analysis thresholds used for bottleneck diagnosis |
| `perflab show-tuning-schema` | Show what goes in tuning.yaml: fixed vs tunable params, sweep syntax |
| `perflab show-config` | Display resolved configuration |
| `perflab show-config-template` | Emit commented YAML config template |
| `perflab init-config` | Create `./perflab.yaml` with default template |
| `perflab init-config --user` | Create `~/.config/perflab/config.yaml` |
| `perflab doctor` | Environment health check |

### Agent flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--suggest` | none | Expert hint for the LLM |
| `--iters` | 12 | Max agent iterations |
| `--candidates` | 3 | Candidates per LLM call |
| `--max-time` | 3600 | Wall-clock budget in seconds |
| `--max-cost` | none (unlimited) | Stop once estimated LLM cost reaches this many USD; fails closed at startup if the model's pricing is unknown |
| `--no-early-stop` | off | Disable convergence detection |
| `--no-fast-screen` | off | Disable two-tier benchmarking |

---

## Safety

PerfLab constrains what the agent may edit through policy checks (protected files, path containment, `allowed_paths`), and can optionally constrain what candidate code may do at runtime via OS-level isolation (`--isolation=restricted`, Linux/bwrap). Without isolation enabled, candidate code runs with your user's privileges — run PerfLab on machines where that is acceptable. These two guarantees are independent; don't assume one implies the other.

### Edit policy

What the agent is allowed to change:

- **Protected files** — `tests.py`, `bench.py`, and `task.yaml` cannot be edited
- **Allowed paths** — `allowed_paths` restricts editable files to specific source files
- **Path containment** — every path is resolved and checked against the workspace root
- **Backup/restore** — files backed up before patching, restored on failure

### Runtime isolation

What the resulting code is allowed to do when it runs, controlled by `--isolation`:

- **`--isolation=auto`** (default) — resolves to `restricted` if this host has usable bwrap (Linux + working user namespaces), else `none`. Accepted from the CLI flag, `task.yaml`, or config; explicit levels below always win over `auto` per the usual CLI > task.yaml > config precedence.
- **`--isolation=none`** — no sandboxing; candidate code runs with your user's full privileges
- **`--isolation=restricted`** — Bubblewrap (`bwrap`) sandboxing on Linux: network namespace unshared unless `task.yaml` sets `network: true` (in which case `/etc/resolv.conf`, `/etc/ssl/certs`, and `/etc/hosts` are also read-only bound, so DNS resolution and TLS verification work), filesystem access scoped to the workspace, GPU devices dev-bound explicitly
- **`--isolation=strict`** — `restricted` plus a seccomp syscall-denial layer: a classic-BPF filter compiled in-process (`perflab/tools/seccomp.py`, stdlib-only — no libseccomp dependency) and applied via `bwrap --seccomp`. It denies ptrace/process-memory access, the full mount family, `bpf`, kernel keyring, module/kexec loading, and namespace-manipulation syscalls with `EPERM` (x86_64 and aarch64; syscalls entering through a foreign ABI are killed). Falls back to `restricted`-equivalent with a logged warning if this bwrap build lacks `--seccomp` or the architecture has no filter table. On non-Linux hosts both `restricted` and `strict` fall back to `none` with an explicit warning.
- **Resource limits** — memory, process, and file descriptor caps (Linux), applied regardless of isolation level

### Output validation

Checks applied to what a candidate produces, independent of edit policy or runtime isolation:

- **Disposable eval workspaces** — every candidate builds, tests, and benchmarks in a temporary copy of the workspace, so a candidate process that writes files at runtime (even `tests.py`) poisons only its own discarded copy; protected files in the real workspace are additionally hash-verified after each iteration
- **Correctness gate** — every candidate runs `tests.py`; rejected on failure
- **Contract validation** — benchmark output checked against `contract.fixed_params` (prevents shrinking the problem to "optimize") and `min_repeats`/`min_warmup` (prevents dialing down measurement counts)
- **Regression check** — candidates must beat baseline by `regression_tolerance` (default 2%)
- **Noise gate** — an improvement must also be distinguishable from measurement noise: by default, the candidate's 95% confidence interval must not overlap the incumbent's. Without it, a 2% acceptance threshold on a machine with 8% run-to-run spread accepts noise as a win, and beam search then chases it. When a task's benchmark reports only an aggregate (no per-repeat samples), the gate cannot run and the decision is logged as unverified rather than silently downgraded. Which test is applied is selectable — see **Decision rules** below
- **Anti-gaming** — determinism re-runs with a varied seed (reject on divergence), zero-variance timing detection, mode-aware single-iteration speedup alerts, and optional thread-count enforcement — configured via `anti_gaming:` in task.yaml

**Decision rules.** One module answers "is this candidate better?" for the agent, `perflab optimize`, and `perflab ci-check`. Pick the rule with `constraints.decision_rule` in `task.yaml`:

| `decision_rule` | What it requires, on top of `regression_tolerance` | Benchmark cost |
|---|---|---|
| `non_overlapping_ci` *(default)* | The candidate's 95% CI must not overlap the incumbent's | — |
| `tolerance_only` | Nothing — the bare ratio. Correct for deterministic metrics (instruction counts, bytes moved). `constraints.noise_gate: false` selects it and takes precedence | — |
| `paired_difference` | An exact Wilcoxon signed-rank test (p<0.05) over a block-interleaved A/B run | **2.6–10x** |

`paired_difference` re-measures the candidate *against* the incumbent by alternating spawns in ABBA order (6 pairs by default) instead of comparing measurements taken minutes apart, so thermal and clock drift cancels between the arms instead of being attributed to the patch. It costs roughly 2.6x the authoritative benchmark's wall clock for a task with enough repeats to spread across the blocks, and up to ~10x for a task configured with very few repeats — so it is opt-in. It earns that on drift-prone hardware (datacenter GPUs, shared CI runners, cloud VMs) and not on a quiet laptop. Where an interleaved run cannot happen — `perflab ci-check`, `perflab optimize`, or a measurement that failed part-way — the rule falls back to the default `non_overlapping_ci` gate and reports the result as unverified; it never falls back to accepting.

Why it exists and what it does and does not buy: [Why block-interleaved (paired) A/B measurement, and why is it opt-in?](ENGINEERING_RATIONALE.md#why-block-interleaved-paired-ab-measurement-and-why-is-it-opt-in)

PerfLab also includes `perflab.harness`, a library of anti-gaming utilities for `bench.py` and `tests.py`:

| Helper | What it does | Backends |
|--------|-------------|----------|
| `SyncTimer` / `cuda_sync_guard` | Forces device synchronization around timing | torch CUDA/MPS, JAX; with no device it drains every backend that is live |
| `ThreadGuard` | Rejects new background threads during execution | any |
| `assert_real_array` | Rejects proxy/unmaterialized output: exact type, real storage, non-null data pointer, no JAX tracers or deleted buffers | torch, numpy, JAX, Python |
| `assert_real_tensor` | The torch-only form of the same check | torch |
| `assert_deterministic` | Same inputs must match, different inputs must differ | torch, numpy, JAX, Python |
| `assert_ulp_close` | ULP-distance check against fp64 reference | torch, numpy, JAX, Python |
| `assert_no_memoization` | Overwrites input data in-place and re-runs | torch, numpy, lists (needs one mutable input) |

*Python* means nested lists/tuples and plain numbers. `import perflab.harness` pulls in no optional dependency — backends are identified from the value's own type, so a torch task never imports numpy and vice versa; comparisons on numpy/JAX values import numpy on demand (`pip install "perflab[tasks-python]"`), while the list/number path needs it not at all. A value the harness cannot inspect makes the check **raise**, never quietly pass.

---

## Creating a Custom Task

Copy `tasks/_sample/` and customize (or `perflab tasks copy _sample my_task` if you installed via pip without cloning). A task needs: `task.yaml`, `bench.py`, `tests.py`, and source files.

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
threadsPerBlock: 16
sweep:
  threadsPerBlock: [16, 32, 64, 128, 256]
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

The same noise gate applies here: a drop only fails CI if it is statistically distinguishable from run-to-run spread, so the check does not fail at random on a shared or unpinned runner. `--save-baseline` records the per-repeat samples needed for that comparison; baselines saved by earlier versions have none, so they fall back to the plain ratio test and the result is reported as unverified.

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

**Note:** the MCP server is the most likely place a third-party or untrusted model ends up driving PerfLab — the client (Claude Desktop, Cursor, etc.) decides what the model can invoke, not you at a terminal. The [edit policy vs. runtime isolation](#safety) distinction still applies here: the server enforces the same `allowed_paths`/protected-file checks, but candidate code still runs with your user's full privileges unless isolation (`--isolation=restricted` or `strict`) is explicitly configured for the underlying agent run.

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
| `benchmark` | Warmup iterations, repeat count, CPU pinning | More repeats for noisy benchmarks, fewer for fast iteration; `cpu_pinning: off` on a host with uncontrolled competing load (Linux-only; a logged no-op elsewhere) |
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

**A note on hardware coverage:** PerfLab's CI has no GPU or TPU runner, so the NVIDIA (`nsys`/`ncu`) and TPU analysis paths are tested against recorded-format fixtures rather than real devices. The CPU paths and the Linux isolation layer are exercised on real hardware. See [Validation Coverage: What Runs on Real Hardware](ENGINEERING_RATIONALE.md#validation-coverage-what-runs-on-real-hardware) for exactly what that does and does not guarantee.

---

## License

PerfLab is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).

You are free to use, modify, and distribute PerfLab. If you run a modified version as a network service, you must make your modifications available under the same license.
