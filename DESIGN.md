# Design Decisions

This document captures the "why" behind key design choices in PerfLab to illustrate things that aren't obvious from reading the code.

---

### Search/replace patches instead of whole file generation

The LLM proposes changes as `SEARCH`/`REPLACE` blocks that must match existing source exactly. This prevents the LLM from hallucinating surrounding code or silently dropping functions. It also makes diffs reviewable since you see exactly what changed, not a full file rewrite where the interesting edit is buried.

### Parallel prescreening, sequential benchmarking

Build + correctness tests are CPU-bound and run concurrently in a thread pool with temporary workspace copies. GPU benchmarking is sequential because concurrent GPU workloads create measurement noise from contention. This means 6 candidates that each take 30s to build+test take ~30s total instead of 3 minutes, but only the survivors touch the GPU.

### Two-tier benchmarking (fast screen + full)

Full benchmarks (warmup + 20 repeats) take seconds per candidate. Fast screening (0 warmup, 2 repeats) is noisy but sufficient for ranking and verifying correctness. Only the top candidate gets a full-fidelity re-benchmark for the accept decision. Cuts per-iteration GPU time by 5-10x.

### Temp-copy evaluation instead of git

File copying is simpler and faster than git for rapid iteration testing: no branch pollution, no git state cleanup, no dependency on the user's repo being clean. Each candidate is evaluated (build + correctness + benchmark) in a disposable temporary copy of the workspace — a candidate process that writes files at runtime (including protected ones like `tests.py`) poisons only its own copy, never the real workspace or later candidates. Accepted patches are re-applied to the real workspace, whose protected files (`tests.py`, `bench.py`, `task.yaml`) are additionally hash-verified against a run-start snapshot after every iteration and restored if tampered. Code snapshots (`snapshots/iter{N}.zip`) provide the audit trail.

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

### Opt-in OS-level isolation instead of containers (Fix 2b)

rlimits (`perflab/tools/shell.py`) cap CPU/memory/fd usage for candidate (LLM-authored) subprocesses but do nothing about filesystem writes outside the workspace or network egress. Full container orchestration (Docker) was rejected: image management contradicts the "local-first CLI" premise, and container cold-start would pollute the fast-screen timing tier (see "Two-tier benchmarking" above). Instead, `perflab/tools/isolation.py` implements a tiered, opt-in `IsolationPolicy` (`none` | `restricted` | `strict`) that wraps the candidate subprocess in Bubblewrap (Linux only) — a launch-time namespace wrapper, not a hypervisor, with read-only binds of `/usr`, `/lib`, the venv, and detected CUDA/driver paths, and read-write binds limited to the workspace. macOS has no bwrap; `restricted`/`strict` there fall back to `none` with an explicit warning rather than a half-working `sandbox-exec` profile.

The default remains `none` (unsandboxed, current behavior) even though the spec designates `restricted` as the eventual Linux default, because that flip is gated on a benchmark-noise A/B: run `tasks/matmul/cpp` under `none` vs. `restricted` and confirm the median runtime delta is under 1% before shipping `restricted` as the default. **Status: not yet run.** This dev environment is macOS without bwrap, so the A/B could not be executed here — it needs a Linux box with bwrap installed (e.g. the ubuntu-latest CI runner). Follow-up: run the A/B, record the median-delta numbers in this section, and only then flip `IsolationSection.level`'s default in `perflab/config.py`.

`strict` currently behaves identically to `restricted` plus a warning — bwrap's `--seccomp` flag needs a compiled BPF filter (ptrace/mount/bpf/keyctl denial), which requires either a libseccomp binding or hand-rolled bytecode; neither is a dependency of this project yet, and shipping a guessed filter risks silently failing open on exactly the syscalls it's meant to deny. Tracked as a follow-up alongside the A/B above.

**DNS/TLS binds for network-allowed tasks (fixed 2026-07):** `_readonly_bind_paths()` only ever covered `/usr`, `/lib`, the venv, and CUDA/driver paths — nothing under `/etc`. A task with `constraints.network: true` correctly skips `--unshare-net`, but without `/etc/resolv.conf` also bound, glibc's resolver has no nameservers to query inside the sandbox; without `/etc/ssl/certs`, TLS verification fails for the same reason libraries can't find a CA bundle. Both symptoms look like "the network is broken" even though the network namespace itself is fine. Fix: `wrap_command` now additionally read-only binds `/etc/resolv.conf`, `/etc/ssl/certs`, and `/etc/hosts` (existence-checked, and only when `policy.network` is true — there's nothing to resolve/verify when `--unshare-net` is in effect, so skip the extra binds in that case). Bound as the literal path, not the symlink target, matching how every other `--ro-bind` in this module is done — bwrap resolves the target itself when the path is a symlink.

### Profilers share artifacts, not runs

A full profiling pass runs the benchmark ~8–12 times (linux_perf stat + record, nsys, ncu, power, py-spy, memray, eBPF, ...), at baseline and after every accepted candidate. The obvious fix — sharing executions across profilers, e.g. polling nvidia-smi during the nsys run — was considered and rejected in favor of a hard purity rule: **each profiler's measurements come from runs whose only perturbation is that profiler's own.**

Three reasons. First, comparability: if power is measured under nsys in one pass and standalone in another (nsys missing or crashed), the numbers differ by a hidden confounder; provenance tags (`measured_under: nsys`) could mark this, but then every consumer — analyzers, drift checks, prompt builder, dashboard — must become provenance-aware forever, versus purity-by-construction where none need care. Second, contamination fails silently: a diagnosis nudged into the wrong utilization tier doesn't crash, it sends the LLM down the wrong optimization path and burns candidate iterations, which cost far more than the bench runs sharing would save. Third, the legality matrix barely allows sharing anyway: CUPTI is single-subscriber per process (nsys, ncu, torch profiler, and the JAX GPU profiler are mutually exclusive by driver design), PMU wrappers each demand to own the argv, and ncu's kernel replay makes wall time and power during its run meaningless. PerfLab is an offline analysis tool for code on someone's critical path — spending longer on analysis to keep diagnoses trustworthy is the right trade.

The rule that survives: one command's output may feed multiple *parsers* (artifact sharing — the pattern TMA L1/L2 already use by parsing linux_perf's `stat_text`), but two tools never co-attach to one execution. This is why folding RAPL events into linux_perf's stat event list is legal (power's standalone run is already a `perf stat` wrapper; RAPL uncore counters don't contend with core counters — the merged run is mechanically identical for both consumers) while power-under-nsys is not. Two corollaries: the timed scoring run (pipeline step 3, which decides accept/reject) never hosts any profiler, and no Profiler protocol change is needed — the Observer/attach protocol this discussion started from existed only to legalize the contaminated pairings the purity rule forbids.

Per-accept re-profiling stays full-fleet. The fleet serves two purposes per accept: cross-axis regression visibility (a memory win that costs runtime should be attributed to the step that introduced it, not discovered at finalize) and fresh diagnosis — the next candidate is conditioned on the current profile, so stale profiles waste LLM iterations. Cost is attacked by making fleet members cheaper, not the fleet smaller: scope ncu per accept to an explicit `--metrics` list of what the analyzers actually parse (occupancy, DRAM throughput, `dram_bytes_total` for the roofline point), reserving `--set full` — which replays every kernel dozens of times for hundreds of unread metrics — for baseline and finalize. That is scope reduction, not contamination: every number still comes from a run perturbed only by ncu itself. **Follow-ups, in order: per-profiler wall timing (pipeline step 7 currently records only aggregate `profile_wall_s`) to confirm where pass time actually goes; the RAPL-into-linux_perf merge; the ncu metric-list audit.**
