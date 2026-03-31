# Design Decisions

This document captures the "why" behind key design choices in PerfLab to illustrate things that aren't obvious from reading the code.

---

### Search/replace patches instead of whole file generation

The LLM proposes changes as `SEARCH`/`REPLACE` blocks that must match existing source exactly. This prevents the LLM from hallucinating surrounding code or silently dropping functions. It also makes diffs reviewable since you see exactly what changed, not a full file rewrite where the interesting edit is buried.

### Parallel prescreening, sequential benchmarking

Build + correctness tests are CPU-bound and run concurrently in a thread pool with temporary workspace copies. GPU benchmarking is sequential because concurrent GPU workloads create measurement noise from contention. This means 6 candidates that each take 30s to build+test take ~30s total instead of 3 minutes, but only the survivors touch the GPU.

### Two-tier benchmarking (fast screen + full)

Full benchmarks (warmup + 20 repeats) take seconds per candidate. Fast screening (0 warmup, 2 repeats) is noisy but sufficient for ranking and verifying correctness. Only the top candidate gets a full-fidelity re-benchmark for the accept decision. Cuts per-iteration GPU time by 5-10x.

### Backup/restore instead of git

File copying is simpler and faster than git for rapid iteration testing: no branch pollution, no git state cleanup, no dependency on the user's repo being clean. Code snapshots (`snapshots/iter{N}.zip`) provide the audit trail.

### Contract enforcement

`contract.fixed_params` prevents the agent from "optimizing" by shrinking the problem (e.g., reducing matrix dimensions from 4096 to 64). Contracts are validated at task load time (before expensive benchmarks) and enforced on both LLM code edits and auto-tuning sweeps.

### Structured failure memory

Naive failure tracking only prevents exact duplicates. PerfLab records *why* each candidate failed: strategy description, failure type, stderr excerpt, profiler context. This teaches the LLM to avoid similar patterns ("64x64 tiling caused register spill") rather than just avoiding the exact same patch. Capped at 10 entries.

### Resource limits vary by program type

GPU programs (`cuda`, `pytorch`, `jax`, `triton`) skip the memory address space limit because CUDA runtimes legitimately map huge virtual regions. CPU-only programs get a 4 GB cap. Process and file descriptor limits are always enforced regardless of program type.

### CUTLASS baselines for sweep ranges

For CUDA tasks, auto-tuning sweeps are centered on CUTLASS-optimal tile configurations (0.5x, 1x, 2x of each dimension) rather than searching the full parameter space. CUTLASS encodes years of NVIDIA's tuning expertise, so searching near those optima is more efficient than brute force. Max 15 trials per sweep.

### AgentContext instead of parameter passing

The agent loop is a linear pipeline (baseline, prompt, LLM, prescreen, benchmark, accept, repeat) where complexity comes from data flow (~30 variables), not control flow. A single `AgentContext` dataclass replaces 18+ function parameters, making state mutation explicit and the data flow traceable.

### Single-worker thread pool for MCP agent runs

The MCP server uses `ThreadPoolExecutor(max_workers=1)` for agent runs. Only one optimization can run at a time because benchmark measurements are sensitive to system contention, having a second agent run would invalidate both sets of results. Subsequent requests are rejected immediately with an error rather than queued.

### 31 separate MCP tools instead of fewer

Each tool has a single responsibility with clear annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`). This lets MCP clients auto-approve read-only calls and prompt for confirmation on mutating ones. Coarser tools would force clients to treat everything as potentially destructive.

### Anti-gaming as a first-class concern

LLMs are optimization machines that will find the shortest path to a good metric, particularly those that game the benchmark rather than genuinely optimizing the code. PerfLab treats this as a systems problem with layered defenses: contract enforcement, variance checks, determinism re-runs, speedup threshold alerts, thread injection detection, and bench.json content-hash verification. The `perflab.harness` library gives task authors additional tools for their protected `bench.py` and `tests.py`.
