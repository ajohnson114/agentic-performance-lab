# Safety Checks

PerfLab evaluates every LLM-generated candidate through **31 safety checks** before accepting it. These checks form a defense-in-depth chain spanning five categories:

| Category | Checks | Purpose |
|----------|--------|---------|
| **A. Patch Validation** | 1–7 | Can the edit be applied safely? |
| **B. Execution Sandboxing** | 8–14 | Can the resulting code damage the system? |
| **C. Result Integrity** | 15–22 | Are the benchmark numbers trustworthy? |
| **D. Reward-Hack Mitigation** | 23–26 | Is the optimization genuine, or gaming the metric? |
| **E. Hardware Stability** | 27–31 | Is the measurement environment reliable? |

A candidate must survive all applicable checks to be accepted. If any check rejects or flags the candidate, it is discarded and the workspace is restored to its previous state.

Checks are listed in approximate evaluation order within each category. Early checks are fast policy gates (microseconds); later checks involve running user code (seconds).

---

## Category A: Patch Validation

These checks run before any code executes. They validate the edit itself — can it be applied safely to the workspace?

### 1. Protected files

**What:** Rejects any edit block whose target file basename is in `PROTECTED_FILENAMES` (`tests.py`, `bench.py`, `task.yaml`).

**Why:** These files define the rules of the optimization game. If the LLM could edit the benchmark harness, it could trivially "improve" performance by reporting fake numbers. If it could edit the correctness test, it could remove assertions to pass a broken optimization. The task config controls what files are editable, so editing it would let the LLM escalate its own permissions.

**Action on failure:** REJECT — candidate discarded immediately.

**Source:** `perflab/optimizers/patch.py` — `PROTECTED_FILENAMES` check in `validate_patch()`

---

### 2. Path containment

**What:** Resolves each edit target to an absolute path and verifies it falls within the workspace root.

**Why:** Prevents directory traversal attacks. Without this, an LLM could craft a path like `../../etc/passwd` or `../../../home/user/.bashrc` to write outside the task directory. This is the primary filesystem sandbox boundary.

**Action on failure:** REJECT — candidate discarded with "path escapes workspace" error.

**Source:** `perflab/optimizers/patch.py` — resolved path prefix check in `validate_patch()`

---

### 3. Allowed paths

**What:** Checks each edit target against the `allowed_paths` glob patterns defined in `task.yaml` (e.g., `["*.py"]`, `["matmul.cpp"]`).

**Why:** Even within the workspace, not all files should be editable. This gives task authors fine-grained control over which source files the LLM can modify. A CUDA task might allow `kernel.cu` but not `Makefile`. Combined with protected files, this implements a whitelist-only edit policy.

**Action on failure:** REJECT — candidate discarded with "not in allowed_paths" error.

**Source:** `perflab/optimizers/patch.py` — fnmatch glob check in `validate_patch()`

---

### 4. File existence

**What:** Verifies the target file actually exists on disk.

**Why:** The agent uses search/replace edits, not file creation. If the LLM hallucinates a filename that doesn't exist, this catches it early with a clear error message rather than letting it fail cryptically downstream.

**Action on failure:** REJECT — candidate discarded with "file not found" error.

**Source:** `perflab/optimizers/patch.py` — `Path.exists()` check in `validate_patch()`

---

### 5. Search text exact match

**What:** Checks that the SEARCH block text appears verbatim in the target file.

**Why:** Search/replace edits are only safe if the search text uniquely identifies the code region being modified. An inexact match could silently replace the wrong code. This is the core correctness guarantee of the patch format — the LLM must demonstrate it knows what the current code looks like before changing it.

**Action on failure:** Falls through to fuzzy match (check 6).

**Source:** `perflab/optimizers/patch.py` — `block.search in file_content` check

---

### 6. Fuzzy match auto-correction

**What:** When an exact match fails, attempts to find a region in the file that is >=80% similar (via `difflib.SequenceMatcher`) and silently corrects the SEARCH block to match.

**Why:** LLMs frequently produce SEARCH blocks with minor errors — wrong whitespace, slightly stale code from an earlier iteration, or small hallucinated differences. Without fuzzy correction, these near-misses would all fail validation, wasting an LLM call. The 80% threshold is high enough to avoid dangerous mismatches while recovering from common LLM mistakes. This was added after observing that GPT-4o produced validation failures on ~60% of iterations due to minor SEARCH text mismatches (see `docs/gpt4o-issues.md`).

**Action on failure:** Falls through to match failure diagnosis (check 7).

**Source:** `perflab/optimizers/patch.py` — `_fuzzy_match_and_correct()`

---

### 7. Match failure diagnosis

**What:** When both exact and fuzzy matching fail, produces a detailed diagnostic: the closest matching region, line number, similarity percentage, and a column-level diff showing where the mismatch starts.

**Why:** The diagnostic is fed back to the LLM in the next iteration's error context. A good diagnostic helps the LLM self-correct — instead of just "search text not found", it sees exactly what the file actually contains and where the mismatch is. This significantly improves recovery rates across iterations.

**Action on failure:** REJECT — candidate discarded, diagnostic added to failure memory for next LLM prompt.

**Source:** `perflab/optimizers/patch.py` — `_find_closest_match()` and `_format_diagnostic()`

---

## Category B: Execution Sandboxing

These checks constrain the runtime environment when LLM-edited code executes. They prevent damage to the host system regardless of what the code does.

### 8. Backup/restore isolation

**What:** Before applying any patch, copies all affected files to a backup directory. After evaluation (pass or fail), restores the originals via `try/finally`.

**Why:** Each candidate must be evaluated against the same baseline code. Without isolation, a failed candidate's partial edits would corrupt the workspace for subsequent candidates. The `try/finally` ensures restoration even if the benchmark crashes or times out. This is what makes beam search possible — multiple candidates can be evaluated independently per iteration.

**Action on failure:** N/A — always runs. Restore is unconditional.

**Source:** `perflab/optimizers/patch.py` — `backup_files()` and `restore_files()`; `perflab/optimizers/agent.py` — `_evaluate_single_candidate()` try/finally block

---

### 9. Subprocess timeouts

**What:** Correctness tests time out after 60 seconds, benchmarks after 300 seconds.

**Why:** A patch might introduce infinite loops, deadlocks, or pathologically slow computation. Timeouts prevent any single candidate from blocking the agent indefinitely. The timeout kills the subprocess and the candidate is rejected.

**Action on failure:** REJECT — subprocess killed, candidate discarded with timeout error.

**Source:** `perflab/tools/shell.py` — `timeout_s` parameter in `run_cmd()`

---

### 10. Resource limits

**What:** On Linux, subprocesses run with `RLIMIT_AS` (4 GB for CPU tasks, 32 GB for GPU tasks), `RLIMIT_NPROC` (512 processes), and `RLIMIT_NOFILE` (1024 file descriptors).

**Why:** Prevents memory exhaustion (a runaway allocation could trigger the OOM killer), fork bombs (`RLIMIT_NPROC`), and file descriptor exhaustion (`RLIMIT_NOFILE`). GPU tasks get a higher memory cap because CUDA runtimes and JIT compilers legitimately map large virtual address regions, but 32 GB still prevents runaway allocation. Override per-task via `constraints.rlimit_as_gb` in `task.yaml`.

**Action on failure:** Subprocess killed by kernel — candidate discarded.

**Note:** Linux-only. On macOS, benchmarks run without these limits. For production on macOS, use Docker with `--network=none`.

**Source:** `perflab/tools/shell.py` — `_make_linux_preexec()` sets limits via `resource.setrlimit()`

---

### 11. No arbitrary commands

**What:** The LLM can only propose search/replace text edits. It cannot execute arbitrary shell commands, install packages, modify system files, or make network requests.

**Why:** The attack surface is limited to code edits applied to specific files. The only code execution happens through the pre-defined correctness and benchmark commands. Mutated source code does run inside those commands, so the `ContractSpec` and resource limits (checks 10, 17-18) provide additional containment for that residual risk.

**Action on failure:** N/A — architectural constraint, not a runtime check.

---

### 12. Candidate isolation

**What:** Each candidate is evaluated independently against the current best state. Parallel prescreening uses temporary workspace copies (`tempfile.mkdtemp()`). Only one candidate is accepted per iteration.

**Why:** Candidates must not interact with each other. If candidate A writes a file that candidate B reads, the evaluation is no longer independent. Temp workspace copies for prescreening and backup/restore for sequential benchmarking ensure clean isolation.

**Action on failure:** N/A — architectural constraint.

**Source:** `perflab/optimizers/agent.py` — `_prescreen_candidate()` creates temp copies; `_evaluate_single_candidate()` uses backup/restore

---

### 13. Secret filtering

**What:** `run_cmd()` strips `PERFLAB_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` from the subprocess environment.

**Why:** Prevents LLM-edited code from accessing API keys through `os.environ`. Also avoids accidental leakage in error messages, crash dumps, or benchmark output. The LLM could theoretically edit code to print `os.environ` — secret filtering ensures there's nothing sensitive to print.

**Action on failure:** N/A — always applied. No secrets available to leak.

**Source:** `perflab/tools/shell.py` — `_sanitize_env()` strips `_SECRET_ENV_PREFIXES`

---

### 14. Symlink protection

**What:** `_read_source_files()` resolves each file path and rejects symlinks that point outside the workspace.

**Why:** A symlink like `./link.py -> /etc/passwd` would let the LLM read arbitrary files through the normal source-file-reading path. Resolving and checking the target prevents information disclosure.

**Action on failure:** File silently skipped — not fed to LLM prompt.

**Source:** `perflab/optimizers/agent.py` — `_read_source_files()` resolve + prefix check

---

## Category C: Result Integrity

These checks verify that benchmark results are trustworthy — not faked, not stale, not manipulated.

### 15. Correctness gate

**What:** After applying the patch, runs the task's correctness command (typically `python tests.py`) and checks the exit code matches the expected value.

**Why:** Performance optimizations must not break correctness. This is the functional safety net — every candidate must produce correct results. A fast but wrong program is worse than a slow correct one. The 60-second timeout prevents infinite loops from hanging the agent.

**Action on failure:** REJECT — candidate discarded, error added to failure memory.

**Source:** `perflab/runners/correctness.py` — `run_correctness()`; `perflab/optimizers/agent.py` — `_evaluate_single_candidate()`

---

### 16. Benchmark execution

**What:** Runs the benchmark command and catches any exceptions (crashes, timeouts, parse errors, missing bench.json).

**Why:** A patch might pass correctness tests but crash under benchmark conditions (different input sizes, repeated execution, timing instrumentation). This gate ensures the optimization actually runs to completion under realistic conditions. The 300-second timeout prevents pathologically slow optimizations from blocking the agent.

**Action on failure:** REJECT — candidate discarded with benchmark error.

**Source:** `perflab/runners/benchmark.py` — `run_benchmark()` timeout, FileNotFoundError checks

---

### 17. Bench.json anti-tampering

**What:** The benchmark runner computes a SHA-256 content hash of `bench.json` before the run (if it exists) and records the wall-clock start time. After the run, it verifies that (1) the content hash changed and (2) `bench.json` mtime is newer than the run start.

**Why:** Without this, LLM-edited source code could pre-write a fake `bench.json` with excellent numbers before the benchmark command even executes. The hash comparison catches a stale/unmodified file. The mtime check is defense-in-depth — even if the hash check is somehow bypassed (e.g., the code writes the exact same content), the mtime must fall within the benchmark execution window.

**Action on failure:** REJECT — `RuntimeError` raised, candidate discarded.

**Source:** `perflab/runners/benchmark.py` — SHA-256 hash + mtime checks in `run_benchmark()`

---

### 18. Contract: required fields

**What:** Validates that the benchmark output JSON contains all fields listed in `contract.required_bench_fields` (e.g., `["ok", "throughput.median"]`).

**Why:** Ensures the benchmark harness still produces the expected output structure after the patch. If an optimization accidentally breaks the benchmark's output format (e.g., by removing a timing call), this catches it before the agent tries to extract a metric value and gets a confusing `KeyError`. Fields are resolved as dotted paths (e.g., `"throughput.median"` navigates to `bench["throughput"]["median"]`).

**Action on failure:** REJECT — candidate discarded with "required field missing" error.

**Source:** `perflab/runners/benchmark.py` — `validate_contract()` field traversal

---

### 19. Contract: fixed params

**What:** Checks that `meta.*` values in the benchmark output match the `fixed_params` declared in `task.yaml` (e.g., `{M: 512, N: 512}`).

**Why:** This is the primary anti-cheating mechanism for dimension gaming. Without it, the LLM could "optimize" a matrix multiply by reducing the matrix size from 4096 to 64, or speed up a transformer by reducing the sequence length from 2048 to 32. Fixed params ensure the problem size stays constant across all iterations. The bench.py harness reports the actual dimensions in `bench.json`'s `meta` section, and the contract check verifies they match — the LLM cannot forge these values because bench.py is a protected file.

**Action on failure:** REJECT — candidate discarded with "Contract violation: meta.M=64, expected 4096" error.

**Source:** `perflab/runners/benchmark.py` — `validate_contract()` meta comparison

---

### 20. Contract early validation

**What:** `ContractSpec.validate()` checks the contract structure at task load time — before any benchmarks run.

**Why:** Catches malformed dotted paths in `required_bench_fields`, invalid `fixed_params` types, and negative `min_repeats`/`min_warmup` values early. Without this, a typo in the contract would only surface after the first benchmark run, wasting LLM calls and computation.

**Action on failure:** Task load fails with validation errors before the agent starts.

**Source:** `perflab/task_spec.py` — `ContractSpec.validate()`

---

### 21. Confirmation re-benchmark

**What:** When fast screening is enabled, the top candidate is re-benchmarked with full warmup/repeats before acceptance. Fast screening uses `PERFLAB_BENCH_WARMUP=0` and `PERFLAB_BENCH_REPEATS=2` for quick directional ranking.

**Why:** Fast screening is a heuristic — 2 repeats with no warmup can be noisy. A candidate that looks best in a noisy fast screen might not hold up under full measurement. The re-benchmark with full fidelity is the decision point. This prevents accepting candidates that only appeared to improve due to measurement noise.

**Action on failure:** If re-benchmark shows no improvement, candidate is rejected despite passing fast screen.

**Source:** `perflab/optimizers/agent.py` — `_try_accept_best()` fast-mode re-benchmark logic

---

### 22. Regression check

**What:** `is_improvement()` verifies that the new metric value is better than the current best by at least `regression_tolerance` (default 2%).

**Why:** Benchmark measurements have inherent variance from OS scheduling, cache state, thermal drift, and other factors. Without a tolerance threshold, the agent would accept noise-driven "improvements" that aren't real. The 2% default is calibrated to be above typical run-to-run variance (~1-3% on locked clocks) while still accepting meaningful gains. For maximize mode: `new > best * 1.02`. For minimize mode: `new < best * 0.98`.

**Action on failure:** Candidate not accepted — agent proceeds to next iteration.

**Source:** `perflab/analyzers/metrics_rollup.py` — `is_improvement()`

---

## Category D: Reward-Hack Mitigation

These checks defend against LLM-generated code that games benchmarks rather than genuinely optimizing performance. They address specific attack patterns documented in the [Wafer.ai reward hacks field guide](https://www.wafer.ai/blog/reward-hacks-field-guide).

### 23. Bench.json variance check

**What:** After every benchmark, `validate_bench_variance()` walks the bench.json tree looking for numeric arrays (timing values, throughput measurements) with zero or near-zero coefficient of variation.

**Why:** Real benchmark timings always have some variance — from OS scheduling, cache effects, thermal state, and device contention. All timing values being *identical* across benchmark repeats is a strong signal of memoization or caching. The specific attack: LLM-generated code adds a static cache (C++ `std::unordered_map`, Python `dict`) keyed by tensor data pointer addresses, exploiting PyTorch's deterministic memory allocator that reuses the same addresses across benchmark iterations. The kernel runs once, caches the result, and returns cached values for all subsequent iterations — producing perfect timing consistency and near-zero latency.

**What it catches:** Caching/memoization hacks. Arrays with all identical values are flagged. Arrays of 5+ values with CV < 1e-9 are also flagged.

**Action on failure:** WARNING logged to `agent_events.jsonl` as `anti_gaming_warning` (check_type: `bench_variance`). Does not reject — some legitimate workloads (e.g., trivially fast operations) may produce very consistent timings.

**Configurable:** `anti_gaming.bench_variance_check` (default: `true`).

**Source:** `perflab/runners/benchmark.py` — `validate_bench_variance()` and `_check_variance_recursive()`

---

### 24. Determinism re-run

**What:** When enabled, the correctness test runs twice for each candidate. The second run sets `PERFLAB_DETERMINISM_SEED=42` in the subprocess environment, signaling the test harness to use different random inputs if it supports the convention.

**Why:** The "no-op kernel" attack: LLM-generated code launches a GPU kernel that executes zero instructions, relying on the output buffer containing correct values from a prior run. This works because GPU memory allocators often return the same buffer, and a previous correctness or benchmark invocation already wrote the correct answer there. Running the test twice — the second time with a signal to vary inputs — invalidates this stale state. If the first run passes but the second fails, the kernel isn't actually computing.

**What it catches:** No-op kernels relying on buffer reuse. Kernels that only work for specific input patterns. Code that reads uninitialized memory that happened to contain correct values from a prior execution.

**Performance cost:** ~1 second per candidate (one extra correctness invocation). The second run only executes if the first passes.

**Action on failure:** WARNING — the primary correctness result (first run) is used for the accept/reject decision. The determinism warning is logged to `agent_events.jsonl` (check_type: `determinism_rerun`) and added to the candidate's error context.

**Configurable:** `anti_gaming.determinism_rerun` (default: `true`).

**Source:** `perflab/runners/correctness.py` — `run_correctness_twice()`; `perflab/optimizers/agent.py` — `_evaluate_single_candidate()`

---

### 25. Incremental gaming detector

**What:** After each accepted iteration, computes the speedup ratio of the new metric relative to the *previous best* (not the baseline). Logs a warning if this exceeds `anti_gaming.gaming_speedup_threshold` (default 100x).

**Why:** The original gaming detector (which compared against the baseline with a 10x threshold) produced false positives on every built-in task. PerfLab's tasks deliberately start from naive implementations — a Python matmul with triple-nested loops achieves 6,000x cumulative speedup when the agent discovers NumPy. The key insight: large cumulative speedups are expected and legitimate, but large *incremental* speedups over already-optimized code are suspicious. If the code is already using vectorized operations and the agent claims another 100x from a single edit, something is likely wrong. The threshold is 100x (not 10x) because even incremental gains can be large when the agent discovers a qualitatively different algorithm (e.g., scalar → SIMD, eager → compiled).

**What it catches:** Benchmark contract circumvention that slips past contract validation. Edits that fundamentally change what is being measured (e.g., timing an empty loop instead of the kernel).

**Action on failure:** WARNING logged to `agent_events.jsonl` as `anti_gaming_warning` (check_type: `suspicious_speedup`). Does not reject — false positives are possible and the check is primarily an audit signal.

**Configurable:** `anti_gaming.gaming_speedup_threshold` (default: `100.0`).

**Source:** `perflab/optimizers/agent.py` — `_try_accept_best()` incremental speedup check

---

### 26. Thread injection check

**What:** When enabled, the agent checks for a `thread_delta` field in `bench.json`'s `meta` section. If the benchmark harness reports that new threads were spawned during kernel execution beyond `max_thread_delta`, the candidate is rejected.

**Why:** The "thread injection" attack: LLM-generated code spawns a background CPU thread that performs the actual GPU computation asynchronously, while the timed kernel function returns immediately. The benchmark records near-zero execution time because the timing only captures the thread spawn, not the computation. The background thread finishes before the correctness test reads the output buffer, so correctness passes. This check requires the benchmark harness to monitor thread counts using `perflab.harness.ThreadGuard` and report the delta in bench.json — it is opt-in because not all tasks have this instrumentation.

**What it catches:** Background thread spawning during kernel execution. Only effective when the bench.py harness reports `meta.thread_delta`.

**Action on failure:** REJECT — candidate discarded with "Thread injection detected" error.

**Configurable:** `anti_gaming.thread_count_check` (default: `false`, opt-in); `anti_gaming.max_thread_delta` (default: `0`).

**Source:** `perflab/optimizers/agent.py` — `_evaluate_single_candidate()` thread delta check

---

## Category E: Hardware Stability

These checks don't evaluate the patch itself — they ensure the measurement environment is stable enough to produce trustworthy results.

### 27. GPU thermal gate

**What:** Before each GPU benchmark, the runner checks GPU temperature. If above 80°C, it waits up to 120 seconds for cooldown to 75°C.

**Why:** Thermal throttling silently reduces GPU clock frequencies, producing artificially degraded measurements. A candidate benchmarked at 85°C will appear slower than the same code at 65°C. Without the thermal gate, the agent might reject a genuinely good optimization because it was measured during thermal throttling, or accept a mediocre one because the baseline was throttled.

**Action on failure:** If still above 75°C after 120 seconds, benchmark proceeds with a warning (results may be noisy).

**Source:** `perflab/runners/benchmark.py` — `_wait_for_gpu_cooldown()`

---

### 28. GPU clock locking

**What:** `setup-h100.sh` locks GPU SM clocks at max frequency via `nvidia-smi -lgc`.

**Why:** Without clock locking, GPU boost behavior can swing clocks by ~22% on H100 between runs of identical code. This is well above the 2% regression tolerance, meaning noise alone could cause false accept/reject decisions. Clock locking eliminates boost/throttle variance and makes measurements deterministic enough for 2% tolerance to be meaningful.

**Action on failure:** N/A — setup-time configuration, not a runtime check. `warn_if_noisy()` warns if clocks are not locked.

**Source:** `setup-h100.sh`; `perflab/tools/sysinfo.py` — `warn_if_noisy()` clock check

---

### 29. GPU isolation

**What:** On multi-GPU nodes, `setup-h100.sh` sets `CUDA_VISIBLE_DEVICES=0` to pin benchmarks to a single GPU.

**Why:** Other processes on the same node may use different GPUs, but inter-GPU communication (NVLink, PCIe bandwidth sharing) can still introduce variance. Pinning to GPU 0 ensures consistent memory bandwidth and eliminates cross-GPU interference.

**Action on failure:** N/A — setup-time configuration. `warn_if_noisy()` warns on multi-GPU nodes without `CUDA_VISIBLE_DEVICES`.

**Source:** `setup-h100.sh`; `perflab/tools/sysinfo.py` — `warn_if_noisy()` multi-GPU check

---

### 30. Drift detection

**What:** Every 3 accepted patches, the agent re-runs the benchmark from the current workspace state and compares the result against the last accepted value. If drift exceeds 5%, a warning is logged.

**Why:** System-level changes (thermal state, background processes, memory fragmentation) can cause measurements to drift over the course of a run. Without drift detection, the agent might attribute system-level degradation to a code change, or miss that accepted improvements are being eroded by environmental factors. The 3-iteration interval balances detection latency against benchmark cost.

**Action on failure:** WARNING — drift logged to `agent_events.jsonl` as `drift_check` event. Does not reject or roll back.

**Source:** `perflab/optimizers/agent.py` — `_try_accept_best()` drift check block

---

### 31. Ollama SSRF prevention

**What:** The Ollama provider validates that `api_base` points to localhost (`localhost`, `127.0.0.1`, or `::1`) on port 11434 and uses `http` or `https` scheme.

**Why:** A misconfigured `api_base` could target other local services (Redis, databases, internal APIs), turning PerfLab into an SSRF vector. Both host and port restrictions are enforced. Override with `PERFLAB_OLLAMA_ALLOW_REMOTE=1` to bypass all checks, or `PERFLAB_OLLAMA_ALLOWED_PORTS=8080,11434` to allow specific additional ports.

**Action on failure:** Provider initialization fails with validation error.

**Source:** `perflab/llm/ollama_provider.py` — URL validation in constructor

---

## Check pipeline summary

```
LLM response
  │
  ├─ parse_patch_response()           → extract SearchReplaceBlocks
  │
  ├─ validate_patch()                 → checks 1-7 (policy + content)
  │   ├─ protected file?              → REJECT (1)
  │   ├─ escapes workspace?           → REJECT (2)
  │   ├─ not in allowed_paths?        → REJECT (3)
  │   ├─ file missing?                → REJECT (4)
  │   ├─ exact match?                 → OK (5)
  │   ├─ fuzzy match ≥80%?            → auto-correct, OK (6)
  │   └─ no match?                    → diagnose, REJECT (7)
  │
  ├─ backup_files()                   → snapshot originals (8)
  ├─ apply_patch()                    → write edits to disk
  │
  ├─ run_correctness()                → correctness gate (15)
  │   └─ determinism re-run           → second run with different seed (24)
  │
  ├─ run_benchmark()                  → benchmark execution (16)
  │   ├─ thermal gate                 → wait for GPU cooldown (27)
  │   ├─ bench.json anti-tampering    → hash + mtime check (17)
  │   └─ timeout enforcement          → 300s kill (9)
  │
  ├─ validate_contract()              → required fields (18) + fixed params (19)
  ├─ validate_bench_variance()        → caching detection (23)
  ├─ thread_delta check               → thread injection (26, if enabled)
  │
  ├─ is_improvement()                 → regression check, ≥2% (22)
  │   └─ confirmation re-benchmark    → full-fidelity re-run if fast-screened (21)
  │
  ├─ gaming detector                  → incremental speedup check (25)
  ├─ drift detection                  → every 3 accepts (30)
  │
  └─ restore_files()                  → restore originals (8)
```

## Harness-level helpers (perflab.harness)

In addition to the 31 framework checks above, PerfLab provides a library of in-process mitigations that task authors can import in their protected `bench.py` and `tests.py` files. Because these files are in the `PROTECTED_FILENAMES` blocklist, the LLM cannot remove the checks once a task author adds them.

| Helper | Module | Reward hack mitigated | Mechanism |
|--------|--------|-----------------------|-----------|
| `SyncTimer` | `gpu_sync` | **Stream injection** — side-stream GPU work evades default-stream timing | Forces `torch.cuda.synchronize()` before start and after stop, draining all streams |
| `cuda_sync_guard` | `gpu_sync` | Same | Context manager variant for wrapping existing timing blocks |
| `ThreadGuard` | `thread_guard` | **Thread injection** — background thread does async work while kernel returns | Snapshots `threading.active_count()` before/after, raises on new threads |
| `assert_no_new_threads` | `thread_guard` | Same | Functional wrapper: `result = assert_no_new_threads(fn, *args)` |
| `assert_real_tensor` | `tensor_check` | **Lazy evaluation** — tensor subclass defers computation until `__eq__` | Validates exact type, storage allocation, non-null data pointer, non-nested |
| `assert_deterministic` | `determinism` | **No-op kernel** / **buffer reuse** / **shared memory overflow** | Runs N times with same inputs (must match), then with different inputs (must differ) |
| `assert_ulp_close` | `precision` | **Precision downgrade** — fp16 computation cast to fp32 | ULP distance against fp64 reference; catches fp16→fp32 casts that allclose misses |
| `assert_no_memoization` | `pointer_poison` | **Pointer-keyed caching** — static map exploits deterministic allocator | Overwrites input tensor storage in-place (same pointers, new data), re-runs kernel |

See [ARCHITECTURE.md Section 19](ARCHITECTURE.md#19-contract-validation-anti-gaming-and-reward-hack-mitigations) for usage examples and the full engineering rationale for the two-layer (framework + harness) design.
