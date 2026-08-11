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

rlimits (`perflab/tools/shell.py`) cap CPU/memory/fd usage for candidate (LLM-authored) subprocesses but do nothing about filesystem writes outside the workspace or network egress. Full container orchestration (Docker) was rejected: image management contradicts the "local-first CLI" premise, and container cold-start would pollute the fast-screen timing tier (see "Two-tier benchmarking" above). Instead, `perflab/tools/isolation.py` implements a tiered, opt-in `IsolationPolicy` (`auto` | `none` | `restricted` | `strict`) that wraps the candidate subprocess in Bubblewrap (Linux only) — a launch-time namespace wrapper, not a hypervisor, with read-only binds of `/usr`, `/lib`, the venv, and detected CUDA/driver paths, and read-write binds limited to the workspace. macOS has no bwrap; `restricted`/`strict` there fall back to `none` with an explicit warning rather than a half-working `sandbox-exec` profile.

**Default flipped to `auto` (2026-07-19, owner decision).** `IsolationSection.level`'s default in `perflab/config.py` is now `"auto"`, which `resolve_policy()`/`resolve_effective_level()` (`perflab/tools/isolation.py`) expand via `default_level_for_host()`: `restricted` on a host with usable bwrap (Linux + working user namespaces), else `none`. `auto` is accepted from any source — CLI `--isolation`, `task.yaml` `isolation.level`, or `perflab.yaml`/user config — with the existing CLI-flag > task.yaml > config precedence unchanged; explicit `none`/`restricted`/`strict` from any source still behave exactly as before. macOS behavior is unchanged by this flip: no bwrap means `auto` still resolves to `none` there, same as the old compiled-in default.

The original plan for this flip was a one-off benchmark-noise A/B (`tasks/matmul/cpp` under `none` vs. `restricted`, confirming <1% median runtime delta) that needed a Linux box this macOS dev environment doesn't have. That one-off was superseded by CI as the ongoing validation mechanism instead: the `test` job now installs bubblewrap on its `ubuntu-latest` runs (relaxing the AppArmor unprivileged-userns restriction 24.04 adds by default) so `TestBwrapAcceptance` in `tests/test_isolation.py` actually executes on every push — with an explicit step asserting those tests weren't silently skipped, so a future image change that breaks the bwrap probe fails loudly instead of quietly reverting to `none`. This gives continuous, per-push confirmation in place of a single recorded measurement.

**Gap:** the `real-task` job's `perflab ci-check` does not currently exercise `restricted` end-to-end — `run_ci_check`/`_run_bench_full` (`perflab/ci.py`) call `run_correctness`/`run_benchmark` with no `isolation` argument at all, and the `ci-check` CLI command has no `--isolation` flag to plumb one through. Wiring isolation into `ci-check` (mirroring how `perflab agent` and the MCP agent tools already resolve and pass an `IsolationPolicy`) is a follow-up; until then, CI's coverage of `restricted` is limited to the `TestBwrapAcceptance` unit/acceptance tests, not the real-task end-to-end job.

**`strict` implemented (2026-07-19).** `strict` layers a seccomp syscall-denial filter on top of `restricted`: `perflab/tools/seccomp.py` compiles a classic-BPF program in-process — stdlib-only, no libseccomp binding, viable because the filter is a small *denylist* (an allowlist would need the full syscall surface and real BPF tooling). It denies ptrace/`process_vm_*`/`kcmp`/`perf_event_open`, the full mount family (classic + new mount API + `pivot_root`/`chroot`), `bpf`, module/kexec loading, reboot/swap, `unshare`/`setns`/`userfaultfd`, and the kernel keyring — all with `EPERM`, so probing code degrades instead of dying; syscalls arriving through a foreign ABI (wrong audit arch, or the x32 bit on x86_64) are killed outright since their numbers can't be checked against the table. Syscall numbers are hardcoded per-arch (x86_64 + aarch64) from the kernel's frozen ABI tables. The compiled program reaches bwrap as `--seccomp FD` on a fresh memfd per spawn — bwrap reads the fd to EOF, so a wrapped argv is single-use; `run_cmd` (`perflab/tools/shell.py`) takes ownership of `pass_fds` and closes them after every spawn, and `run_correctness_twice` re-wraps for its second run.

The earlier "a guessed filter risks silently failing open" objection is answered with two independent test layers: `tests/test_seccomp.py` contains a cBPF interpreter that symbolically executes the emitted program against every denied/allowed case on both architectures (runs everywhere, including macOS), and `TestSeccompAcceptance` in `tests/test_isolation.py` verifies real kernel enforcement through bwrap — on CI's x86_64 ubuntu runners (with a step asserting the tests ran rather than skipped) and locally for aarch64 via the `docker/` dev container (see `docker/README.md`; a dev-loop tool, not a runtime — the "no container orchestration" decision above is about candidate execution and stands).

**DNS/TLS binds for network-allowed tasks (fixed 2026-07):** `_readonly_bind_paths()` only ever covered `/usr`, `/lib`, the venv, and CUDA/driver paths — nothing under `/etc`. A task with `constraints.network: true` correctly skips `--unshare-net`, but without `/etc/resolv.conf` also bound, glibc's resolver has no nameservers to query inside the sandbox; without `/etc/ssl/certs`, TLS verification fails for the same reason libraries can't find a CA bundle. Both symptoms look like "the network is broken" even though the network namespace itself is fine. Fix: `wrap_command` now additionally read-only binds `/etc/resolv.conf`, `/etc/ssl/certs`, and `/etc/hosts` (existence-checked, and only when `policy.network` is true — there's nothing to resolve/verify when `--unshare-net` is in effect, so skip the extra binds in that case). Bound as the literal path, not the symlink target, matching how every other `--ro-bind` in this module is done — bwrap resolves the target itself when the path is a symlink.

### Profilers share artifacts, not runs

A full profiling pass runs the benchmark ~8–12 times (linux_perf stat + record, nsys, ncu, power, py-spy, memray, eBPF, ...), at baseline and after every accepted candidate. The obvious fix — sharing executions across profilers, e.g. polling nvidia-smi during the nsys run — was considered and rejected in favor of a hard purity rule: **each profiler's measurements come from runs whose only perturbation is that profiler's own.**

Three reasons. First, comparability: if power is measured under nsys in one pass and standalone in another (nsys missing or crashed), the numbers differ by a hidden confounder; provenance tags (`measured_under: nsys`) could mark this, but then every consumer — analyzers, drift checks, prompt builder, dashboard — must become provenance-aware forever, versus purity-by-construction where none need care. Second, contamination fails silently: a diagnosis nudged into the wrong utilization tier doesn't crash, it sends the LLM down the wrong optimization path and burns candidate iterations, which cost far more than the bench runs sharing would save. Third, the legality matrix barely allows sharing anyway: CUPTI is single-subscriber per process (nsys, ncu, torch profiler, and the JAX GPU profiler are mutually exclusive by driver design), PMU wrappers each demand to own the argv, and ncu's kernel replay makes wall time and power during its run meaningless. PerfLab is an offline analysis tool for code on someone's critical path — spending longer on analysis to keep diagnoses trustworthy is the right trade.

The rule that survives: one command's output may feed multiple *parsers* (artifact sharing — the pattern TMA L1/L2 already use by parsing linux_perf's `stat_text`), but two tools never co-attach to one execution. This is why folding RAPL events into linux_perf's stat event list is legal (power's standalone run is already a `perf stat` wrapper; RAPL uncore counters don't contend with core counters — the merged run is mechanically identical for both consumers) while power-under-nsys is not. Two corollaries: the timed scoring run (pipeline step 3, which decides accept/reject) never hosts any profiler, and no Profiler protocol change is needed — the Observer/attach protocol this discussion started from existed only to legalize the contaminated pairings the purity rule forbids.

Per-accept re-profiling stays full-fleet. The fleet serves two purposes per accept: cross-axis regression visibility (a memory win that costs runtime should be attributed to the step that introduced it, not discovered at finalize) and fresh diagnosis — the next candidate is conditioned on the current profile, so stale profiles waste LLM iterations. Cost is attacked by making fleet members cheaper, not the fleet smaller: scope ncu per accept to an explicit `--metrics` list of what the analyzers actually parse (occupancy, DRAM throughput, `dram_bytes_total` for the roofline point), reserving `--set full` — which replays every kernel dozens of times for hundreds of unread metrics — for baseline and finalize. That is scope reduction, not contamination: every number still comes from a run perturbed only by ncu itself. **Follow-ups, in order: per-profiler wall timing (pipeline step 7 currently records only aggregate `profile_wall_s`) to confirm where pass time actually goes; the RAPL-into-linux_perf merge; the ncu metric-list audit.**
