# PerfLab Feature Reference

This document covers PerfLab's profiler backends, analysis engines, and advanced features in detail. For a quick overview, see [README.md](README.md). For architecture and internals, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Table of Contents

- [Profilers](#profilers)
- [GPU-aware profiler context](#gpu-aware-profiler-context)
- [MPS cross-profiler CPU/GPU breakdown](#mps-cross-profiler-cpugpu-breakdown)
- [Per-phase training breakdown](#per-phase-training-breakdown)
- [C++/CUDA Diagnosis](#ccuda-diagnosis)
- [Hierarchical roofline (L2/DRAM)](#hierarchical-roofline-l2dram)
- [Micro-architecture analysis](#micro-architecture-analysis)
- [TMA Level 2/3 analysis](#tma-level-23-analysis)
- [CUTLASS baselines and auto-tuning](#cutlass-baselines-and-auto-tuning)
- [Profiler-driven flag recommendations](#profiler-driven-flag-recommendations)
- [Roofline-driven optimization playbook](#roofline-driven-optimization-playbook)
- [Compiler diagnostics](#compiler-diagnostics) (includes [assembly extraction pipeline](#assembly-extraction-pipeline))
- [Tensor Core support](#tensor-core-support)
- [Unified kernel dossier](#unified-kernel-dossier)
- [GPU attribution](#gpu-attribution)
- [TPU support](#tpu-support)
- [Differential profiling](#differential-profiling)
- [Memory profiling](#memory-profiling)
- [Perfetto trace export](#perfetto-trace-export)
- [eBPF syscall tracing](#ebpf-syscall-tracing-linux)
- [Differential flame graphs](#differential-flame-graphs)
- [Lock contention profiling](#lock-contention-profiling)
- [Top-Down Microarchitecture Analysis (TMA)](#top-down-microarchitecture-analysis-tma)
- [Power and energy profiling](#power-and-energy-profiling)
- [Parallel candidate evaluation](#parallel-candidate-evaluation)
- [Structured failure memory](#structured-failure-memory)
- [Promising alternatives](#promising-alternatives)
- [Cross-run learning](#cross-run-learning)
- [Multi-metric Pareto optimization](#multi-metric-pareto-optimization-optional)
- [Benchmark noise detection](#benchmark-noise-detection)
- [Auto-vectorization verification](#auto-vectorization-verification)
- [Hot loop assembly in LLM prompt](#hot-loop-assembly-in-llm-prompt)
- [Thread scheduling analysis](#thread-scheduling-analysis)
- [GPU memory tracking](#gpu-memory-tracking)
- [Machine fingerprint](#machine-fingerprint)
- [Build flag overrides](#build-flag-overrides)
- [Error feedback to LLM](#error-feedback-to-llm)
- [Prompt token budget](#prompt-token-budget)
- [CPU roofline estimation](#cpu-roofline-estimation)
- [Bottleneck diagnosis coverage](#bottleneck-diagnosis-coverage)
- [Profiler tooling rationale](#profiler-tooling-rationale)
- [Structured configuration](#structured-configuration)
- [Iteration state artifact](#iteration-state-artifact)
- [Task authoring tools (MCP)](#task-authoring-tools-mcp)
- [Backend coverage summary](#backend-coverage-summary)

---

## Profilers

PerfLab selects profilers based on `program_type` and extracts structured data for the bottleneck analyzer. Missing profilers degrade gracefully — the agent still works with whatever is available.

| Profiler | python | pytorch | jax | triton | cpp | cuda |
|----------|--------|---------|-----|--------|-----|------|
| **py-spy** (CPU hotspots) | x | x | x | x | | |
| **PyTorch profiler** | | x | | | | |
| **JAX profiler** (XLA/TPU) | | | x | | | |
| **nsys** (Nsight Systems) | | x | x | x | x | x |
| **ncu** (Nsight Compute) | | x | x | x | x | x |
| **Linux perf** | x | x | x | x | x | x |
| **Metal trace** | | x | x | x | | |
| **memray** (memory) | x | x | x | x | | |
| **eBPF** (syscall/IO) | x | x | x | x | x | x |
| **lock contention** (perf lock/c2c) | | | | | x | x |
| **thread sched** (perf sched) | | | | | x | x |
| **power** (RAPL/nvidia-smi) | x | x | x | x | x | x |
| **TMA** (TopdownL1) | x | x | x | x | x | x |

The **bottleneck analyzer** consumes all profiler data and produces ranked diagnoses (e.g., "Kernel 'sgemm_128x128' is memory-bound (mem=82%, compute=38%)") with confidence levels and suggested actions. In agent mode, these feed directly into the LLM prompt. Diagnoses include **control divergence detection** — low branch efficiency and low warp execution efficiency trigger specific suggestions like thread coarsening, branchless arithmetic, and data partitioning.

**Source-level detection** scans editable source files for optimization patterns already present:
- **PyTorch:** `torch.compile`, AMP/autocast, SDPA/flash attention, `channels_last`, `pin_memory`, `num_workers`, `float32_matmul_precision`, `inference_mode`/`no_grad`, cuDNN benchmark mode, `persistent_workers`, `prefetch_factor`, `non_blocking` transfers, CUDA Graphs (PyTorch API), `torch.cuda.empty_cache`, gradient accumulation
- **C++:** SIMD intrinsics, OpenMP, threading, CUDA kernel launches, `__restrict__` pointers
- **CUDA:** shared memory, `__launch_bounds__`, tiling, CUDA graphs, pinned memory, async operations, cooperative groups
- **JAX:** `jax.jit`, mixed precision dtypes, buffer donation, Pallas kernels, `shard_map`
- **Python:** NumPy vectorization, nested loop detection

**System-level warnings** (`warn_if_noisy()`) check for conditions that cause noisy or suboptimal benchmarks: non-performance CPU governor, high system load, GPU persistence mode disabled, transparent hugepages set to `never` (Linux), active GPU throttling (thermal/power), GPU clocks not locked at max frequency (boost/throttle variance), GPU temperature approaching or exceeding throttle threshold, and multi-GPU nodes without `CUDA_VISIBLE_DEVICES` pinning.

The analyzer also detects **single-threaded execution** and **missing SIMD vectorization** for CPU-bound programs:

- **Single-threaded detection:** When `task-clock` reports < 1.5 CPUs utilized on a multi-core system (>= 4 cores), the analyzer flags single-threaded execution with suggestions for OpenMP, `std::thread`, and C++17 parallel algorithms. High confidence when < 1.1 CPUs utilized on >= 8 cores. Does not fire for GPU-centric program types (pytorch, jax, triton).
- **No-vectorization detection:** When source code lacks SIMD intrinsics and there is a dominant CPU hotspot (> 50% of samples), the analyzer suggests using SIMD intrinsics, `-march=native -O3`, and `__restrict__` pointers. Only fires for C++ programs.

---

## GPU-aware profiler context

When a workload is GPU-bound, py-spy CPU hotspots can be misleading — functions like `forward()` appear "hot" because the CPU thread blocks waiting for GPU kernel completion, not because they are CPU bottlenecks. PerfLab detects GPU-bound workloads (via torch profiler GPU/CPU ratio, nsys GPU active percentage, Metal trace GPU time, or MPS backend heuristic) and annotates the prompt to warn the agent. Functions matching GPU dispatch patterns (`torch`, `cuda`, `forward`, `backward`, `_call_impl`, etc.) are flagged as "GPU wait points." No annotation is added for CPU-bound workloads where py-spy data is directly useful.

---

## MPS cross-profiler CPU/GPU breakdown

On Apple Silicon (MPS), the PyTorch Chrome trace doesn't emit GPU kernel events for Metal work, leaving the CPU/GPU breakdown empty. PerfLab solves this by cross-referencing the PyTorch profiler's CPU op time with the Metal trace profiler's `gpu_time_total_ms`. This fires automatically when both profilers are present and the torch trace lacks GPU data — CUDA traces are unaffected.

---

## Per-phase training breakdown

Training benchmarks that use `record_function("## phase_name ##")` markers (the transformer and dataloader tasks do this by default) get a per-phase breakdown showing time, GPU time, and CPU time for each phase (forward, backward, optimizer, data_loading). The bottleneck analyzer uses this to:

- Identify the **dominant phase** (>60% of total) with phase-specific optimization suggestions
- Detect **per-phase GPU underutilization** (GPU time <30% of phase time in forward/backward)

The LLM prompt includes a formatted phase breakdown table so the agent knows exactly which phase to target.

---

## C++/CUDA Diagnosis

When `program_type` is `cpp` or `cuda`, PerfLab runs NSys and NCU alongside Linux perf to provide cross-profiler host-device analysis:

- **NVTX ranges**: NSys extracts NVTX annotation ranges (`nvtxRangePush`/`Pop`) to identify which phases of the host program are slowest. The bottleneck analyzer flags phases that dominate execution time.
- **Kernel launch correlation**: NSys extracts per-kernel grid/block dimensions alongside timing, enabling detection of small-kernel launch overhead patterns.
- **Sync overhead detection**: NSys identifies `cudaDeviceSynchronize` and `cudaStreamSynchronize` calls and reports total sync time. If sync time exceeds 20% of kernel time, a high-confidence bottleneck is fired.
- **NCU source-line metrics**: NCU extracts source file, function, and line information when available, producing `source_hotspots` that pinpoint where in the code GPU bottlenecks originate.
- **Control divergence detection**: NCU branch efficiency and warp execution efficiency metrics are parsed and fed to the bottleneck analyzer. Low values trigger diagnoses recommending thread coarsening, branchless arithmetic, warp-uniform control flow, and data partitioning.
- **Warp stall diagnosis**: NCU warp stall reason metrics (long_scoreboard, barrier, memory_throttle, math_pipe_throttle, etc.) identify *why* warps are idle. Each stall reason maps to specific root causes and targeted fix suggestions (e.g., long_scoreboard → add shared memory tiling + cp.async pipelining).
- **Bank conflict detection**: NCU shared memory bank conflict counts identify serialization in shared memory accesses, with suggestions for array padding and swizzled layouts.
- **Uncoalesced access detection**: NCU sectors-per-request metric flags non-contiguous global memory access patterns (ideal is 1.0 sectors/request; >4.0 triggers diagnosis with SoA layout and vectorized load suggestions).
- **Occupancy limiter identification**: NCU reports *which* resource limits occupancy (registers, shared memory, or block size). The bottleneck analyzer produces actionable suggestions specific to the tightest limiter (e.g., register-limited → `__launch_bounds__`, `-maxrregcount`).
- **FP64-on-consumer-GPU detection**: NCU instruction mix breakdown detects significant FP64 usage (>10% pipe utilization), which runs at 1/64th throughput on consumer GPUs.
- **Register spill detection**: NCU local memory bytes identify register spilling, with suggestions for `__launch_bounds__`, `-maxrregcount`, and shared memory alternatives.
- **TMA pipe utilization (Hopper+)**: NCU TMA pipeline metrics track Tensor Memory Accelerator activity for hardware-accelerated async tensor loads.
- **Stall GMMA (Hopper)**: Hopper-specific warp stall reason for Warp Group MMA completion, with guidance on warp specialization and pipeline staging.
- **Hot-path cudaMalloc detection**: NSys top API call analysis flags `cudaMalloc`/`cudaFree` consuming >5% of API time, suggesting `cudaMallocAsync` and memory pools.
- **Non-contiguous tensor detection**: Flags `aten::contiguous` and `aten::clone` when they appear in top operators (>3% of time), indicating wasted memory bandwidth on layout conversion.
- **Tensor Core alignment checking**: Checks matmul operator shapes from the torch trace and flags dimensions not aligned to multiples of 8, with suggested padded dimensions.
- **Memory fragmentation detection**: Flags high allocation counts (>500) with large peak memory (>1GB) and significant allocation time (>50ms) as potential fragmentation, suggesting `expandable_segments` and buffer pre-allocation.
- **Per-operator FLOPS counting**: PyTorch profiler extracts per-op FLOPS from trace args (via `with_flops=True`), enabling roofline analysis without ncu. Top ops ranked by FLOPS contribution.
- **XLA HLO cost analysis**: JAX profiler extracts FLOP and bytes_accessed estimates from HLO dump metadata, enabling arithmetic intensity computation for roofline analysis without hardware profilers.
- **Host-device cross-reference**: The bottleneck analyzer cross-references CPU perf hotspots with GPU utilization — if a CPU function dominates while the GPU is idle, a targeted "CPU blocks GPU" diagnosis is produced.

---

## Micro-architecture analysis

PerfLab provides deep micro-architectural analysis to help the LLM optimize beyond 80% of peak performance. These derived metrics are computed from existing profiler data and presented in a dedicated "Micro-architecture analysis" section of the LLM prompt.

### Kernel performance ceiling

Computes the theoretical maximum TFLOPS for a specific kernel based on its achieved occupancy:

```
Kernel performance ceiling:
Occupancy: 35% → theoretical max: 105.0 TFLOPS (35% of 300 peak)
Currently achieving: 5.0 TFLOPS (4.8% of kernel ceiling, 1.7% of hardware peak)
→ Occupancy is the primary limiter. Fix occupancy BEFORE optimizing compute.
```

This prevents the LLM from wasting iterations on compute optimizations when occupancy is the real bottleneck. If achieved TFLOPS is close to the kernel ceiling, the only way to improve is to increase occupancy (reduce registers, shared memory, or increase block size).

### SASS instruction efficiency

Classifies every SASS instruction into categories to identify overhead:

| Category | Instructions | What it means |
|----------|-------------|--------------|
| **Useful compute** | FFMA, FMUL, HMMA, HGMMA | FLOPs that contribute to the result |
| **Tensor Core** | HMMA, HGMMA, QGMMA | TC operations (highest throughput) |
| **Global memory** | LDG, STG, LDGSTS, ATOM | DRAM traffic |
| **Shared memory** | LDS, STS, LDSM | SRAM traffic (fast) |
| **Address math** | IADD, IMAD, LEA, SHL | Index computation overhead |
| **Control flow** | BRA, ISETP, EXIT | Branches and predication |
| **Sync** | BAR, MEMBAR, FENCE | Synchronization overhead |

A kernel with 30%+ address math overhead is spending too many instructions computing array indices — suggests restructuring indexing or using TMA (hardware addressing on Hopper). A kernel with 0% TC instructions in a matmul is not using Tensor Cores.

### Pipeline utilization heatmap

Shows all GPU execution pipes as utilization bars:

```
Pipeline utilization:
  FP32/FMA               ██████░░░░  60.0% [MED]
  Tensor Core            ░░░░░░░░░░   0.0% [LOW]
  INT/ALU                ██░░░░░░░░  20.0% [LOW]
  SFU (transcendentals)  ░░░░░░░░░░   5.0% [LOW]
```

If FP32/FMA is high but Tensor Core is 0%, switching to WMMA/HMMA is the highest-impact change. If all pipes are low, the bottleneck is memory or occupancy, not compute.

### Benchmark stability scoring

Computes coefficient of variation (CV) from benchmark timing data:
- **CV < 3%:** "Very stable — improvements >1% are real"
- **CV 3-5%:** "Stable — improvements >N% are reliable" (N = 2× CV)
- **CV 5-10%:** "Moderate noise — only improvements >N% are meaningful"
- **CV > 10%:** "High noise — results unreliable, increase repeats"

The minimum meaningful improvement threshold (2× CV) prevents the LLM from chasing noise.

### GPU clock throttle detection

Monitors GPU power draw during the benchmark via nvidia-smi samples:
- If power drops >10% during the run, the GPU is thermally throttling
- Computes effective peak TFLOPS (adjusted for clock reduction)
- Tells the LLM: "kernel is thermally limited, not algorithmically limited — further optimization may not improve wall-clock time"

All five metrics are computed in `perflab/analyzers/microarch.py` and assembled into a single `microarch_summary` dict that feeds into the prompt.

### Benchmark environment stabilization

The benchmark runner includes active measures to reduce environmental noise beyond passive warnings:

- **GPU thermal gate**: Before each GPU benchmark, the runner checks GPU temperature. If above 80°C, it waits up to 120s for cooldown to 75°C before proceeding. This prevents thermal throttling from contaminating measurements mid-run.
- **GPU clock locking** (`setup-h100.sh`): Locks SM clocks at max frequency via `nvidia-smi -lgc` to eliminate boost/throttle variance (~22% on H100 between 1620 MHz base and 1980 MHz boost).
- **GPU isolation** (`setup-h100.sh`): On multi-GPU nodes, sets `CUDA_VISIBLE_DEVICES=0` to pin benchmarks to a single GPU.
- **Fast screening + confirmation**: Screen-phase benchmarks use `warmup=0, repeats=2` for quick ranking only — the top candidate is always re-benchmarked at full fidelity before the accept/reject decision.
- **Confirmation re-benchmark**: The agent re-benchmarks the top candidate with full warmup/repeats before accepting. The orchestrator's grid search also confirms the winning configuration.

---

## TMA Level 2/3 analysis

PerfLab collects Level 1 TMA (Top-Down Microarchitecture Analysis) via `perf stat -M TopdownL1`, and extends this with Level 2/3 analysis via platform-specific tools:

**Intel (toplev from pmu-tools):** If `toplev` is installed, PerfLab runs `toplev --level 3 --single-thread` and parses the CSV output to extract:
- **Level 2:** Backend Bound → Memory Bound vs Core Bound; Frontend Bound → Fetch Latency vs Fetch Bandwidth
- **Level 3:** Memory Bound → L1 Bound, L2 Bound, L3 Bound, DRAM Bound, Store Bound; Core Bound → Divider, Port Utilization

**AMD (perf events fallback):** On AMD Zen 3/4/5, PerfLab uses `perf stat` with cache hierarchy events (`L1-dcache-load-misses`, `LLC-load-misses`, etc.) to estimate the memory hierarchy bottleneck level. Less precise than toplev but identifies whether the bottleneck is L1, L2/L3, or DRAM.

**Bottleneck rules:** TMA Level 2/3 data feeds into 6 new bottleneck rules with targeted actions:
- **L1 Bound** → tile for L1 cache (32-64 KB), stride-1 access
- **L2 Bound** → software prefetching, L2-aware blocking
- **L3 Bound** → streaming access, non-temporal stores
- **DRAM Bound** → reduce total data movement, operator fusion, lower precision
- **Store Bound** → non-temporal stores, aligned stores, avoid store-to-load forwarding
- **Core Bound** → SIMD utilization, FMA, reduce instruction count; divider and port utilization sub-diagnoses

**CPU vendor detection:** `_detect_cpu_vendor()` reads `/proc/cpuinfo` (Linux) or `sysctl` (macOS) to automatically select Intel toplev or AMD perf events.

---

## CUTLASS baselines and auto-tuning

PerfLab uses NVIDIA CUTLASS profiling data as optimal starting points for kernel tile configurations, and integrates auto-tuning into the agent loop.

### CUTLASS baseline configurations

Known-optimal GEMM tile configurations per GPU architecture, derived from CUTLASS profiling:

| GPU (SM) | Dtype | Tile M×N×K | Stages | Warps | Cluster |
|----------|-------|------------|--------|-------|---------|
| V100 (sm_70) | FP16 | 128×256×32 | 2 | 8 | — |
| A100 (sm_80) | FP16 | 128×256×32 | 4 | 8 | — |
| A100 (sm_80) | FP32 | 128×128×32 | 3 | 8 | — |
| RTX 4090 (sm_89) | FP16 | 128×256×64 | 3 | 8 | — |
| H100 (sm_90) | FP16 | 128×256×64 | 5 | 8 | 2×1 |
| H100 (sm_90) | FP8 | 128×256×128 | 5 | 8 | 2×1 |

These are injected into the LLM prompt as expert hints (e.g., "CUTLASS optimal for H100 (fp16): Tile 128×256×64, 5 stages"). The LLM uses them as starting values rather than guessing.

### Auto-tuning in the agent loop

After the LLM's code edit is accepted, if `tuning.yaml` has a `sweep` section, PerfLab automatically sweeps the parameter space:

1. **LLM writes parameterized kernel** with tunable values (TILE_M, TILE_N, NUM_STAGES, etc.)
2. **LLM updates `tuning.yaml`** with a `sweep` section listing values to try
3. **PerfLab auto-tunes** — generates all combinations, caps at 15 trials (random sample if larger), runs correctness + benchmark for each
4. **Contract validates** each config — `fixed_params` unchanged, correctness passes
5. **Best config kept** — winning parameters written back to `tuning.yaml`

The sweep is centered on CUTLASS baselines via `generate_sweep_around_baseline()`, which explores 0.5×, 1×, 2× of each tile dimension and ±1 pipeline stage. This explores near the optimum rather than searching blindly.

**Division of labor:** The LLM focuses on **what** to optimize (algorithm, data layout, instruction choice). Auto-tuning handles **how much** (parameter values). Each does what it's best at.

---

## Profiler-driven flag recommendations

Beyond static ISA-based flag recommendations, PerfLab now generates **dynamic flag suggestions based on profiler output** via `recommend_flags_from_profiling()`:

| Profiler Signal | Flag Suggested | Rationale |
|----------------|---------------|-----------|
| Cache miss rate > 5% | `-fprefetch-loop-arrays` | Software prefetching reduces cache miss penalty |
| Frontend Bound > 25% | `-falign-functions=32 -falign-loops=32` | Alignment reduces instruction fetch stalls |
| Bad Speculation > 20% + branch miss > 3% | `-fprofile-generate / -fprofile-use` | PGO trains branch predictors on actual data |
| DRAM Bound (TMA Level 3) | `-funroll-loops` | Unrolling improves bytes-per-instruction ratio |
| Hot CPU functions + AVX2 available | `-march=native` | Enables auto-vectorization with full ISA |
| IPC < 0.5 | `-funroll-loops` | Improves instruction-level parallelism |

These recommendations are deduplicated against ISA-based recommendations and logged per iteration via `build_flags_state` events for cross-iteration tracking.

**Post-optimization compilation guidance:** After the agent converges on optimized source code, the prompt includes production build guidance: `-O3 -march=native -mtune=native -flto -DNDEBUG`, plus PGO instructions (`-fprofile-generate` → run → `-fprofile-use`) for an additional 10-20%.

---

## Hierarchical roofline (L2/DRAM)

PerfLab supports hierarchical roofline analysis with both DRAM and L2 cache bandwidth ceilings:

- **L2 bandwidth specs** — Known aggregate L2 read bandwidth for A100 (6 TB/s), H100 (12 TB/s), RTX 4090 (3.2 TB/s), RTX 4080 (2.4 TB/s), RTX 3090 (2.4 TB/s), V100 (3.1 TB/s)
- **Dual-ceiling roofline plot** — The roofline PNG renders both DRAM bandwidth (black) and L2 cache bandwidth (green) ceilings, showing where data reuse shifts the effective bottleneck
- **Bottleneck level diagnosis** — When a workload is memory-bound, `_classify_bound()` determines if the bottleneck is at the DRAM level (>60% of peak DRAM BW achieved) or at L2-or-below (DRAM not saturated, suggesting cache-level inefficiency). The LLM prompt receives targeted guidance: DRAM-bottlenecked workloads should reduce total bytes; L2-bottlenecked workloads should improve tiling and data reuse
- **Profiler-based FLOPS** — PyTorch `with_flops=True` FLOPS counts and JAX HLO cost annotations (FLOP/bytes_accessed) are automatically wired into `compute_roofline_point()`, enabling roofline analysis without ncu. Meta-provided M/N/K and explicit flops/bytes_moved take priority over profiler estimates

---

## Roofline-driven optimization playbook

Instead of dumping a flat list of optimization hints, PerfLab builds a prioritized, contextual playbook based on how far the workload is from peak performance:

| Utilization | Tier | Focus |
|-------------|------|-------|
| <10% | structural | Wrong device, no batching, Python overhead |
| 10-30% | standard | torch.compile, AMP, precision, memory format |
| 30-60% | kernel | Memory access patterns, shared memory, custom kernels |
| 60-80% | fine_tune | Occupancy, register pressure, tile sizes |
| >80% | micro | Micro-optimizations only |

The playbook includes:
- **Priority actions** from the #1 ranked bottleneck diagnosis
- **Tier-appropriate optimizations** filtered by program type and already-applied optimizations
- **Optimization status checklist** merging source code detection (`[x]` present, `[ ]` absent) with history scanning (`[~]` tried and rejected)

When no roofline data is available, the playbook defaults to "standard" + "kernel" tiers for broadest coverage. The checklist and bottleneck actions still work regardless.

**Graceful degradation:** If a profiler is unavailable (e.g. `py-spy` blocked by macOS SIP, `nsys` not installed), PerfLab skips it and continues with the remaining profilers. The agent can still optimize based on benchmark numbers alone — profiler data just makes it more targeted. Run `perflab doctor` to see what's available on your system.

---

## Compiler diagnostics

PerfLab captures compiler diagnostic output and threads it into the LLM prompt so the agent can act on missed optimizations, register pressure, and JIT compilation issues.

| Toolchain | What's captured | How |
|-----------|----------------|-----|
| **GCC/G++** | Per-line optimization remarks (vectorization widths, inlining, unrolling, aliasing, FMA) | `-fopt-info-all-optall -gline-tables-only` |
| **Clang/Clang++** | Per-line remarks (vectorization widths, inlining, loop analysis) | `-Rpass=.* -Rpass-missed=.* -Rpass-analysis=.*` |
| **NVCC/ptxas** | Registers/thread, shared memory, spill stores/loads, line-level debug info | `--ptxas-options=-v --generate-line-info` |
| **PyTorch (dynamo/inductor)** | Graph breaks, eager fallbacks, fusion events, recompilations | `TORCH_LOGS=+dynamo,+inductor` |
| **JAX/XLA** | XLA compilation events, total compile time, recompilations, HLO op breakdown | `JAX_LOG_COMPILES=1` + XLA HLO dump |
| **JAX/TPU** | MXU utilization, host-device time split, infeed stalls, HLO padding waste | JAX profiler trace + XLA HLO analysis |
| **Triton** | Shared memory, register usage, num_warps, compilation events | `TRITON_DEBUG=1` |

For C++ and CUDA, structured remarks are cross-referenced with profiler hotspots to produce targeted insights:

- **C++ remarks** are matched against `perf annotate` hotspots (±3 line window) and CPU ISA features (e.g., "hottest loop vectorizes at 128-bit but hardware supports 256-bit"). Clang vectorization width extraction is type-aware — it infers element size from the remark context (`double` → 64-bit, `float` → 32-bit, `i8` → 8-bit) instead of assuming 32-bit.
- **CUDA remarks** are matched against GPU attribution data — kernel function names from ptxas (mangled) are fuzzy-matched against NSys kernel names (demangled). Register pressure or spills in a kernel consuming significant GPU time produce high-priority insights with `__launch_bounds__` and shared memory caching suggestions.

The agent also receives build flag recommendations based on ISA detection — missing `-march=native`, `-O2` to `-O3` upgrades, and sanitizer overhead warnings. Agent-proposed build flag overrides go through syntax checking and conflict detection; invalid flags are rejected with structured feedback via `BuildOverrideResult` so the LLM can correct its proposal.

For C++ and CUDA tasks, the `build` step in `task.yaml` owns compilation. `bench.py` and `tests.py` expect the pre-built binary. See [ARCHITECTURE.md](ARCHITECTURE.md) for data models and cross-referencing rules.

### Assembly extraction pipeline

PerfLab extracts and feeds low-level assembly directly to the LLM so it can verify whether the compiler is generating optimal code. Two complementary pipelines:

**CPU assembly (perf annotate):**
1. `perf record -g` samples call stacks during the benchmark
2. `perf annotate --stdio` maps samples to disassembled instructions with % CPU per line
3. `extract_hot_assembly()` finds the hottest instruction in each function, extracts ±8 lines of surrounding assembly, returns top 3 functions
4. The LLM sees the assembly and can identify: scalar vs SIMD instructions (`vmulss` vs `vmulps`), unaligned loads, missing FMA, branch-heavy inner loops

**CUDA SASS (cuobjdump):**
1. `cuobjdump --dump-sass <binary>` disassembles the compiled GPU binary to SASS (GPU machine code)
2. `extract_cuda_sass()` parses per-kernel SASS listings, demangles kernel names via `c++filt`
3. Large kernels are truncated to head + tail with instruction count shown
4. The LLM sees the SASS and can identify: `HMMA`/`HGMMA` (Tensor Core ops), `LDG`/`STG` (global loads/stores), `LDS`/`STS` (shared memory), `FFMA` (FP32 FMA), `LDGSTS` (async copy), `BAR.SYNC` (barriers)
5. Absence of `HMMA` in a matmul kernel means Tensor Cores are not engaged; frequent `LDG` without `LDS` suggests missing shared memory tiling

**Cross-referencing with compiler remarks:**

Compiler optimization remarks (missed vectorization, aliasing, FMA) are matched to perf hotspot lines within a ±3 line window. A missed vectorization is only flagged HIGH priority if perf also identifies that line as hot (>5% CPU). Five cross-referencing rules:
1. Missed vectorization at hot line → suggest `__restrict__`, alignment, loop restructuring
2. Vectorization width gap → hardware supports wider SIMD than compiler used (e.g., 128-bit on AVX2-capable hardware)
3. Alias blocking → `__restrict__` qualifiers missing
4. Missed FMA → compile with `-mfma` or `-march=native`
5. Non-unit stride → restructure for contiguous access (AoS→SoA)

**Build flag recommendations:**

`build_flags.py` analyzes the task's build command and recommends missing flags:
- ISA: `-march=native`, `-mavx2`, `-mavx512f`, `-mfma` (based on detected CPU features)
- Optimization: `-O2` → `-O3` upgrade, `-flto` for link-time optimization
- Debug: `-g` → `-gline-tables-only` (lighter debug info for profiling)
- Warnings: `-fsanitize` removal (2-5x overhead flagged)

The agent can also inject safe flags via `build_overrides.json` from a 28-flag allowlist (ISA, optimization, debug, OpenMP — no arbitrary CFLAGS injection).

---

## Tensor Core support

PerfLab provides full Tensor Core awareness for NVIDIA GPUs (Volta+):

- **Profiling** — NCU profiler extracts Tensor Core utilization and throughput metrics (`sm__pipe_tensor_cycles_active`, `tensor_throughput`) per kernel. Handles multiple ncu CSV column name variants (`Tensor Active`, `pipe_tensor_cycles_active`, `Tensor Utilization`).
- **Bottleneck detection** — Two rules in `_analyze_ncu()`:
  1. *Low Tensor Core utilization* — fires when TC utilization is below `ncu_tc_util_low` (default 30%). Suggests WMMA/mma.sync, dimension alignment, TF32, cuBLAS/CUTLASS.
  2. *Compute-bound on CUDA cores* — fires when a kernel saturates CUDA core ALUs but shows no Tensor Core activity. Suggests switching to FP16/BF16 inputs or TF32 precision.
- **Multi-dtype roofline** — Per-GPU peak tables in `roofline_peaks.py` cover FP32, TF32, FP16, BF16 for A100, H100, RTX 4090/4080, RTX 3090, V100. Roofline visualization renders separate ceiling lines per dtype.
- **Task YAML support** — `RooflineSpec.dtype_peaks` allows task authors to specify all dtype peaks directly. The H100 task includes FP32 (67 TFLOPS), TF32 (989), FP16/BF16 (1979).
- **LLM hints** — Compute-bound action playbook includes WMMA/mma.sync, cp.async pipelining, TMA, CUB primitives for CUDA; warp specialization, persistent kernels, SplitK for Triton; Pallas kernels, XLA GPU flags, FP8/AQT, shard_map for JAX; FlexAttention, torchao quantization, semi-structured sparsity for PyTorch.
- **Optimization detection** — Source code scanning detects already-applied optimizations: WMMA/mma.sync, cp.async, CUB/Thrust, cudaMallocAsync (CUDA); FlexAttention, torchao, nested tensors, 2:4 sparsity (PyTorch); shard_map, Pallas, gradient checkpointing (JAX). The agent's checklist marks these as `[x]` to avoid re-suggesting them.
- **Demo task** — `perflab/demo_tasks/matmul/cuda_tensorcore/` provides a naive WMMA kernel (FP16 inputs, FP32 accumulator) that loads fragments directly from global memory with no shared-memory tiling. The agent should discover shared-memory tiling, double buffering, warp-level pipelining, and cooperative tile loading.

---

## Unified kernel dossier

For CUDA tasks, PerfLab joins three independent data sources into a single ranked "kernel dossier" for each hot GPU kernel, so the LLM sees everything about a kernel in one place:

```
### #1: sgemm_naive (85% GPU time, 120.0 ms)
NCU: SM util 45% | Memory-bound (mem=82%, compute=23%) | TC util 0%
Dominant stall: long_scoreboard (42%)
Issues: sectors/req: 8.5 (uncoalesced)

SASS (50 instructions):
  /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
  /*0010*/ FFMA R2, R5, R6, R2 ;
      ... (30 instructions omitted) ...
  /*00f0*/ STG.E [R4.64], R2 ;
→ Use shared memory tiling
```

**Data flow:**

```
GPU attribution (NSys)          → which kernel matters, % GPU time
    ↓ ranked kernel list
NCU per-kernel metrics          → what's wrong (stalls, TC util, coalescing, occupancy)
    ↓ matched by kernel name
SASS disassembly (cuobjdump)    → exact GPU machine instructions
    ↓ combined into unified dossier
LLM prompt                      → one section per kernel, #1 kernel first
```

**Fuzzy kernel name matching:** The three tools use different name formats — NSys reports CUDA runtime names (`volta_sgemm_128x128_nn`), NCU reports demangled C++ names (`sgemm_naive(int, int, ...)`), and cuobjdump reports mangled symbols (`_Z12sgemm_naivePfS_S_iii`). The `_match_kernel()` function uses scored matching (exact → substring → base name → token overlap) to join them correctly.

**NCU annotation header:** Each dossier includes a one-line NCU summary showing:
- SM utilization, memory-bound vs compute-bound classification
- Tensor Core utilization
- Achieved occupancy with limiter identification (registers, shared mem, block size)
- Dominant warp stall reason and percentage
- Issues: bank conflicts, uncoalesced access (sectors/request)

**Graceful degradation:** If NSys correlation data is unavailable (no GPU attribution), the dossier is skipped and GPU attribution, NCU metrics, and SASS render as separate sections. If NCU or SASS data is missing for a kernel, the dossier still renders with whatever data is available.

---

## GPU attribution

PerfLab builds a CPU-to-GPU call graph from NSys correlation data, linking `cudaLaunchKernel` calls to GPU kernel executions. This works across all CUDA frameworks (PyTorch, Triton, JAX/XLA, raw CUDA) because they all go through `cudaLaunchKernel` at the CUDA runtime layer.

### How the CPU→GPU graph is built

**Step 1 — NSys data collection.** NSys runs the benchmark with full CUPTI tracing enabled. This captures two key event tables in the NSys SQLite database:
- `CUPTI_ACTIVITY_KIND_RUNTIME` — CPU-side CUDA API calls (`cudaLaunchKernel`, `cudaMemcpy`, `cudaDeviceSynchronize`) with timestamps, thread IDs, a `correlationId`, and a `callchainId` for host call stack recovery
- `CUPTI_ACTIVITY_KIND_KERNEL` — GPU kernel executions with start/end timestamps, grid/block dimensions, stream ID, and the same `correlationId`

**Step 2 — Correlation extraction.** `_extract_cpu_gpu_correlation()` in `nsys_profiler.py` performs a SQL JOIN on `correlationId` to link each CPU-side `cudaLaunchKernel` call to its GPU kernel execution. Each correlation tuple contains: `{api_name, kernel_name, stream_id, gpu_duration_ns, launch_overhead_ns, cpu_start_ns, gpu_start_ns}`. The launch overhead is the time delta between the CPU API call and the GPU kernel start.

**Step 2b — Call chain walking.** `_extract_callchain_context()` walks the CPU call stack from each `cudaLaunchKernel` upward through CUDA runtime and framework internals to find the first user-code function that triggered the kernel launch. This enriches each correlation with `caller_function` and `caller_module`, enabling attribution like "your `forward()` at line 42 triggers `volta_sgemm`." Handles multiple NSys schema versions and filters out internal frames (cuda*, libtorch, c10::, at::, pybind11, etc.).

**Step 3 — Graph construction.** `build_cpu_gpu_call_graph()` groups correlation tuples by `{api_name, kernel_name, stream_id}`, aggregates per-edge statistics (launch count, total GPU time, average launch overhead), propagates the most common `caller_function` per edge, and computes each edge's percentage of total GPU time. The result is a list of `CpuGpuEdge` objects sorted by total GPU time descending.

**Step 4 — Attribution ranking.** `compute_attribution_ranking()` combines the call graph edges with multiple linking strategies applied in priority order:

1. **Call chain caller matching** — If call chain walking found a user-code `caller_function`, match it against CPU hotspots from Linux perf. This succeeds where name matching fails (e.g., `train_step()` → `cudaLaunchKernel` → `volta_sgemm`).

2. **Temporal NVTX matching** — If no framework op is found yet, check whether the kernel's CPU-side launch timestamp falls within an NVTX annotation's time window. Prefers the most specific (shortest duration) enclosing range. This correctly attributes `volta_sgemm` to `aten::linear` even though they share no name substrings. NVTX ranges now include `start_ns`/`end_ns` timestamps for this purpose.

3. **Torch trace cross-reference** — For PyTorch tasks, cross-references timestamped CPU operator events (`_raw_cpu_ops` from the torch profiler trace) with NSys kernel launch timestamps. Operators with duration >100us are indexed for efficient temporal lookup.

4. **Py-spy temporal join** — For Python-based GPU backends (PyTorch, JAX, Triton), matches kernel launch CPU timestamps against timestamped py-spy samples from speedscope JSON output. This identifies which Python function was on the CPU when each kernel was launched, providing full Python → GPU attribution regardless of framework.

5. **Fuzzy name matching** (fallback) — CPU hotspot names matched to kernel names via substring and base-name matching. Also includes kernel name heuristics (`volta_sgemm` → `aten::mm`, `triton_poi_fused_relu_0` → `triton:fused_relu`).

Each entry is scored: `gpu_pct * 2.0 + cpu_pct * 0.5 + overhead_penalties + attribution_bonus`, producing a single ranked list. Entries with richer attribution (caller function, framework op) receive a scoring bonus since they are more actionable. The top 5 entries are passed to the LLM prompt (or to the kernel dossier builder when NCU and SASS data are also available).

**Step 5 — Per-stream analysis.** `_extract_per_stream_gaps()` computes per-stream kernel gap statistics and utilization. `detect_pipeline_stalls()` applies rules: if a stream is idle >50% of trace time or the max inter-kernel gap exceeds the threshold, a pipeline stall diagnosis is generated with suggestions for CUDA graphs, kernel fusion, or stream overlap.

### Cross-backend coverage

The temporal attribution strategies benefit multiple backends:

| Strategy | Backends |
|----------|----------|
| Call chain walking (NSys callchainId) | All CUDA: raw CUDA, PyTorch, Triton, JAX/XLA |
| Temporal NVTX matching | PyTorch, Triton, any framework emitting NVTX annotations |
| Torch trace cross-reference | PyTorch |
| Py-spy temporal join | All Python GPU backends: PyTorch, JAX, Triton |
| Fuzzy name matching | All (fallback) |

**Why not just pass raw NSys output to the LLM?** NSys is an excellent data collection tool but produces flat lists of kernel statistics, memory transfers, and correlation tuples — not actionable optimization guidance. The attribution engine adds six things NSys doesn't provide:

1. **Full-stack caller identification** — NSys records correlation IDs and call chains, but doesn't surface "user function X triggered kernel Y." The engine walks call chains, cross-references NVTX ranges, torch operator events, and py-spy samples to build full attribution: "Python function `forward()` → `aten::mm` → `volta_sgemm` (85% GPU time)."

2. **Cross-source linking** — NSys knows kernel `volta_sgemm_128x128_nn` took 85% of GPU time. Linux perf knows `matmul()` took 60% of CPU time. Neither tool connects them. The attribution engine joins the two, giving the LLM a full-stack picture.

3. **Semantic grouping** — NSys `cpu_gpu_correlations` is a flat list of 100+ raw correlation tuples (one per kernel launch). The engine groups by `{API call, kernel name, stream}`, aggregates launch counts and total time, and returns ~5 ranked entries. Sending 100 raw tuples wastes tokens and dilutes focus.

4. **Pipeline stall diagnosis** — NSys provides per-stream gap durations and utilization percentages. The engine applies rules on top: "Stream 3 is idle 70% of the time, max gap 250 us" → category: `pipeline-stall` → suggestion: "overlap with other streams, fuse small kernels, consider CUDA graphs."

5. **Framework-level translation** — NSys reports CUDA kernel names (`triton_poi_fused_relu_0`). The engine maps these back to framework operations using temporal NVTX matching (preferred) and name heuristics (fallback), making prompts directly relatable to user code.

6. **Unified scoring and ranking** — The engine scores entries with `gpu_pct * 2.0 + cpu_pct * 0.5 + overhead_penalties + attribution_bonus`, producing a single ranked list across kernel time, launch overhead, pipeline stalls, and memory transfers. Entries with richer attribution are boosted. NSys presents separate tables for each category with no concept of "this is the #1 thing to fix."

---

## TPU support

PerfLab provides first-class TPU support through the JAX profiler:

- **Device detection** — auto-detects TPU chips via `jax.devices()` (v4, v5e, v5p, v6e)
- **Roofline model** — known BF16 TFLOPS and HBM bandwidth specs for each TPU generation
- **Profiling** — XLA HLO dump analysis (op breakdown, module count, padding detection) + JAX trace parsing (host-device time split, MXU utilization)
- **Bottleneck analysis** — 5 TPU-specific rules: low MXU utilization, XLA padding waste, infeed stalls, HLO fragmentation, fp32-instead-of-bf16
- **HLO op attribution** — weighted cost model ranking XLA operations by estimated device time (the TPU equivalent of GPU kernel attribution). Shows which operations dominate MXU time with per-op diagnoses and suggestions
- **Dashboard** — dedicated TPU section showing MXU utilization, host vs device time bar, infeed stall metrics, plus XLA/HLO op attribution bar chart in Diagnostics
- **LLM hints** — TPU-specific guidance in the optimization prompt (Pallas, shard_map, tile alignment, bf16, scan, compilation cache)

**Recommended TPU for demos:** TPU v5e is the best choice — cheapest on-demand pricing (~$1.20/chip-hour), most mature JAX software stack (GA since 2023), widely available across US/EU/Asia regions. See `setup-tpu-v5e.sh` for one-command setup.

**Demo task:** `perflab/demo_tasks/attention/jax_tpu/` provides a deliberately naive attention implementation (fp32, no jit, Python loops over heads) that the agent should optimize to achieve 10-50x speedup via `@jax.jit`, bf16 dtype, and vectorized multi-head attention.

TPU analysis triggers automatically when `system_info` contains TPU device data. The `perflab doctor` command checks for JAX and TPU availability.

---

## Differential profiling

Between iterations, PerfLab computes a profile diff comparing key metrics (IPC, cache miss rate, GPU active %, kernel gap times) with direction and significance classification. This tells the agent whether its last change had the intended microarchitectural effect, not just whether the benchmark score moved.

The diff now includes **function-level hotspot shifts** — which functions gained or lost CPU share between baseline and optimized code. The LLM sees "naive_matmul dropped from 45% to 2%, memcpy rose to 30%" rather than just "IPC improved by 15%."

---

## Memory profiling

CPU sampling profilers cannot detect memory-bound bottlenecks. PerfLab includes a **memray** profiler for Python-based tasks that captures peak memory, total allocations, and top allocators by size. This fills a critical gap: many real-world optimizations involve reducing allocations, reusing buffers, or avoiding copies — and the LLM cannot suggest these without memory data. Memray summaries feed into `PromptContext` and are rendered in the LLM prompt alongside other profiler data.

---

## Perfetto trace export

After profiling, PerfLab exports a `perfetto_trace.json` in Chromium Trace Event JSON format. Open it in [Perfetto UI](https://ui.perfetto.dev/) for interactive timeline visualization of CPU hotspots and hardware counters. No instrumentation SDK required — this reuses data already collected by the sampling profilers.

---

## eBPF syscall tracing (Linux)

On Linux, PerfLab can trace syscall latency using `bpftrace`. This captures I/O wait time that sampling profilers miss — particularly relevant for data-loading bottlenecks where the CPU is idle waiting on disk or network I/O. Auto-skipped on non-Linux platforms. eBPF summaries feed into `PromptContext` and are rendered in the LLM prompt.

---

## Differential flame graphs

When both baseline and optimized profiles are available, PerfLab generates a **differential flame graph** (`diff_flamegraph.svg`) that overlays function-level CPU share changes. Red bars indicate functions that got hotter (increased CPU share); blue bars indicate functions that got cooler. The diff is sorted by absolute delta so the most impactful changes appear first. This complements the existing hotspot diff table with a visual representation. (The diff flamegraph is generated from computed stack diffs, separate from py-spy's speedscope output.)

---

## Lock contention profiling

For multi-threaded C/C++ code, PerfLab runs `perf lock` and `perf c2c` to detect:

- **Mutex/spinlock contention:** Which locks are most contended, with acquisition counts, wait times, and average/max wait durations
- **False sharing detection:** `perf c2c` identifies cache lines with high HITM (Hit In Modified) counts, indicating cores fighting over the same cache line — a common hidden performance killer in parallel code

These are Linux-only and auto-skipped on other platforms. The lock contention data feeds into the bottleneck analyzer, is visible in the dashboard under Diagnostics, and is rendered in the LLM prompt via `PromptContext`.

---

## Top-Down Microarchitecture Analysis (TMA)

PerfLab collects Intel's Top-Down Microarchitecture Analysis metrics via `perf stat -M TopdownL1`, classifying CPU cycles into four buckets:

- **Frontend Bound:** instruction fetch/decode stalls (I-cache misses, complex decoding)
- **Backend Bound:** execution unit / memory stalls (cache misses, memory bandwidth)
- **Bad Speculation:** mispredicted branches, cancelled work
- **Retiring:** useful work done

This tells the LLM *why* the CPU is slow, not just *where*. A backend-bound program needs tiling or prefetching; a frontend-bound program needs code layout optimization. Falls back to raw topdown-* counters on older perf versions.

---

## Power and energy profiling

PerfLab measures energy consumption during benchmarks:

- **CPU:** RAPL (Running Average Power Limit) via `perf stat -e power/energy-pkg/` reports package and core energy in Joules with average power in Watts
- **GPU:** nvidia-smi polling during benchmark execution captures average, peak, and P50/P95 power draw in Watts

Power data appears in the dashboard Diagnostics section and helps identify whether optimizations are also energy-efficient — important for data center and battery-constrained environments.

---

## Parallel candidate evaluation

PerfLab evaluates candidates in two phases to minimize wall-clock time:

**Phase 1 — Parallel prescreening (CPU-bound):** All candidates are validated, built, and correctness-tested concurrently using `ThreadPoolExecutor`. Each candidate gets a temporary workspace copy to avoid file conflicts. The original workspace is never modified. Typical speedup: 6 candidates × 30s each = 3 minutes sequential → ~30 seconds parallel.

**Phase 2 — Sequential benchmarking (GPU-bound):** Only candidates that pass prescreening proceed to GPU benchmarking. Benchmarks must run sequentially to avoid GPU contention (concurrent benchmarks would invalidate measurements). With 6 candidates and typically 2-3 passing prescreen, this reduces benchmark time by 50-60%.

The prescreen uses `_prescreen_candidate()` which creates a `tempfile.mkdtemp()` workspace copy, applies the patch, runs build + correctness, and cleans up — regardless of success or failure. Prescreening subprocess calls use `skip_preexec=True` to avoid the `preexec_fn` + `fork()` thread-safety issue (undefined behavior in multithreaded processes). Resource limits are enforced during the sequential benchmark phase instead.

---

## Security and resource limits

PerfLab runs LLM-generated code edits in a layered sandbox with **31 safety checks** across five categories. See [SAFETY_CHECKS.md](SAFETY_CHECKS.md) for the full reference and [ARCHITECTURE.md Section 18](ARCHITECTURE.md#18-safety-model) for the architectural overview. Key runtime protections:

**Resource limits (Linux):**

| Limit | CPU tasks | GPU tasks | Purpose |
|-------|-----------|-----------|---------|
| `RLIMIT_AS` | 4 GB | 32 GB | Prevent memory exhaustion |
| `RLIMIT_NPROC` | 512 | 512 | Prevent fork bombs |
| `RLIMIT_NOFILE` | 1024 | 1024 | Prevent file descriptor exhaustion |

GPU tasks get a higher memory cap because CUDA runtimes and JIT compilers map large virtual address regions. The 32 GB cap still prevents runaway allocation. Override per-task via `constraints.rlimit_as_gb` in `task.yaml`.

**Secret filtering:** API keys (`PERFLAB_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are stripped from the subprocess environment before running benchmarks. LLM-edited code cannot access API keys through `os.environ`.

**Bench.json anti-tampering:** The benchmark runner computes a SHA-256 content hash of `bench.json` before execution. After the benchmark, it verifies the content changed (hash comparison) and the file mtime is within the run window (defense-in-depth) — catches LLM-edited code that pre-writes fake results.

**Symlink protection:** `_read_source_files()` resolves each path and rejects symlinks that point outside the workspace, preventing information disclosure (e.g., `./link.py -> /etc/passwd`).

**Ollama SSRF prevention:** The Ollama provider restricts `api_base` to localhost on port 11434 by default. Override with `PERFLAB_OLLAMA_ALLOW_REMOTE=1` for remote deployments, or `PERFLAB_OLLAMA_ALLOWED_PORTS=8080` to allow specific additional ports.

**Contract early validation:** `ContractSpec.validate()` checks contract structure at task load time — malformed field paths, invalid types, and negative values are caught before any benchmarks run.

**Thread safety:** Parallel prescreening uses `skip_preexec=True` to avoid the `preexec_fn` + `fork()` undefined behavior in multithreaded processes. Full resource limits are enforced during the sequential benchmark phase.

**Reward-hack mitigations:** PerfLab defends against [known reward hacks](https://www.wafer.ai/blog/reward-hacks-field-guide) at two layers. Framework-level checks run automatically: bench.json variance detection (catches caching/memoization via zero-variance timing arrays), determinism re-run (runs correctness twice with different random seeds to catch buffer-reuse dependent code), and an incremental speedup detector (flags single-iteration gains exceeding a configurable threshold). Harness-level helpers (`perflab.harness`) provide in-process defenses for task authors: `SyncTimer` drains all CUDA streams before timing (stream injection), `ThreadGuard` detects background thread spawning (thread injection), `assert_real_tensor` rejects tensor subclasses (lazy evaluation), `assert_deterministic` verifies output reproducibility (no-op kernels), `assert_ulp_close` checks ULP distance against fp64 references (precision downgrade), and `assert_no_memoization` poisons tensor pointers to defeat static caches. All framework checks are configurable via `anti_gaming:` in `task.yaml`. See [ARCHITECTURE.md Section 19](ARCHITECTURE.md#19-contract-validation-anti-gaming-and-reward-hack-mitigations) for the full engineering rationale.

**macOS limitations:** Resource limits are Linux-only. On macOS, benchmarks run without memory, process, or file descriptor limits. For production use on macOS, run inside a Docker container with `--network=none`.

---

## Structured failure memory

The agent accumulates structured information about *why* candidates failed across all iterations, preventing the LLM from repeating dead-end strategies:

```
## Failed approaches (DO NOT repeat these)
- Iter 2 [correctness]: Added 64x64 shared memory tiling with 4-stage pipeline
  Reason: Register spill caused occupancy drop to 12%
- Iter 3 [correctness]: WMMA with fp16 accumulator
  Reason: FP16 accumulation too imprecise for verification tolerance
  Context: Max error: 0.15, tolerance: 0.01
```

Each failure records: iteration, strategy description (from LLM reasoning), failure type (correctness/build/validation), reason (stderr excerpt), and profiler context. Capped at last 10 failures to prevent prompt bloat.

Unlike `last_errors` (which only carries the previous iteration's errors), failure memory persists across the entire run — the LLM always knows the full history of what didn't work and why.

---

## Promising alternatives

When multiple candidates improve performance but only the best is accepted, the good-but-not-best candidates are shown to the LLM in the next iteration:

```
## Promising alternatives from last iteration
These candidates also improved performance — consider combining them.
- candidate 1: shared memory tiling → 3.2 TFLOPS (2.4x vs baseline)
  Strategy: Added 64x64 shared memory tiles to reduce global memory traffic
- candidate 4: loop reordering → 2.8 TFLOPS (2.1x vs baseline)
  Strategy: Reordered k-loop to innermost for better cache locality
```

This solves the "lost improvement" problem: if tiling gives 3x and WMMA gives 4x independently, the LLM now knows both worked and is encouraged to combine them (tiling + WMMA → potentially 8x+). Top 3 alternatives are kept per iteration, sorted by value.

---

## Cross-run learning

When the agent is invoked on a task that has prior runs, PerfLab automatically loads context from previous optimization attempts. The LLM prompt receives:

- What worked in prior runs (accepted patches with descriptions)
- What didn't work (rejected attempts)
- Prior optimization summaries and bottleneck diagnoses

This prevents the agent from repeating failed approaches and lets it build on successful strategies from earlier runs.

**Scaling:** Context is loaded from at most 3 most recent prior runs, with summaries capped at ~500 chars each. It is injected into the LLM prompt only on iteration 1 (not every iteration), adding roughly 1000-3000 tokens. Older runs are ignored.

**Starting from scratch:** If the source code has been reverted to baseline between runs, the prior context still applies — it tells the LLM which strategies to try first. If the source code was left in an optimized state from a prior run, the baseline benchmark captures the current performance and the LLM works from there. A disclaimer is included in the prompt: "The current source code may differ from what these runs started with. Use these results as strategy guidance, not as assumptions about current code state."

---

## Multi-metric Pareto optimization (optional)

Tasks can optionally define a `secondary_metric` in their `benchmark` section for multi-objective optimization:

```yaml
benchmark:
  cmd: "python bench.py --json out/bench.json"
  metric:
    name: "tflops.median"
    mode: "maximize"
  secondary_metric:
    name: "latency_ms.p95"
    mode: "minimize"
```

When both metrics are tracked, PerfLab computes the **Pareto frontier** — the set of iterations where no other iteration is better on both metrics simultaneously — and generates a frontier graph showing the tradeoff space. This is purely optional; tasks without `secondary_metric` work exactly as before.

**How it works:** The agent loop still **optimizes for the primary metric only** — it accepts/rejects candidates based on whether the primary metric improved. The secondary metric is tracked alongside each iteration and used for **post-hoc Pareto analysis**. The Pareto frontier graph appears in the dashboard between the metric history chart and the roofline chart.

**CI gating:** When a task has a `secondary_metric`, `perflab ci-check` gates on both metrics — a regression on *either* metric beyond `regression_tolerance` fails the check. CI also runs bench variance anti-gaming checks (flags zero-variance timing arrays as possible memoization/caching) and, when NCU profiler data is available in both the baseline and a recent profile run, detects profiler metric regressions (SM utilization, Tensor Core utilization, occupancy, bank conflicts, warp stalls, memory coalescing). Bench variance warnings and profiler regressions are advisory — they surface in the output but do not cause CI failure. See the README [CI Integration](README.md#ci-integration) section for details.

**Demo task:** `perflab/demo_tasks/matmul/cpp_parallel` uses `tflops.median` (maximize) as the primary metric and `latency_ms.p95` (minimize) as the secondary metric.

---

## Benchmark noise detection

PerfLab analyzes repeated benchmark measurements for statistical reliability. When the benchmark harness includes `raw_values` or `samples` in its JSON output, PerfLab computes:

- **Coefficient of Variation (CV):** Standard deviation / mean. If CV > 10%, a warning banner appears in the dashboard and is injected into report.json.
- **95% Confidence Interval:** Helps assess whether speedup claims are statistically meaningful.
- **Noise warning:** When CV exceeds the threshold, the dashboard shows a yellow warning: "High measurement variance detected — speedup claims may be unreliable."

**Implementation:** `perflab/analyzers/bench_stats.py`.

---

## Auto-vectorization verification

For C++ tasks profiled with `perf annotate`, PerfLab checks whether hot functions actually contain SIMD instructions (SSE, AVX, AVX-512, or ARM NEON). This answers "did the compiler actually vectorize my hot loop?"

The analysis:
1. Parses `perf annotate --stdio` output for each function
2. Scans assembly lines for SIMD mnemonics (e.g., `vmovaps`, `vaddps`, `fmla`)
3. Reports per-function: has_simd (yes/no), ISA level, CPU percentage
4. Generates a warning for hot functions lacking SIMD instructions

The vectorization report appears in the dashboard under Diagnostics > Vectorization analysis.

**Implementation:** `perflab/analyzers/vectorization.py`.

---

## Hot loop assembly in LLM prompt

For C++ and CUDA tasks profiled with `perf annotate`, PerfLab extracts small disassembly snippets from the hottest loops and includes them directly in the LLM prompt. This gives the agent concrete, low-level evidence to inform its optimization strategy — not just "this function is hot" but "this function is hot and the compiler is emitting scalar `vmulss` instead of vectorized `vfmadd231ps`."

The extraction:
1. Parses `perf annotate --stdio` output
2. Finds the instruction with the highest CPU sample percentage in each function
3. Extracts a window of ±8 lines around that instruction
4. Returns the top 3 functions above a 5% CPU threshold

The prompt section appears between **Bottleneck diagnosis** and **Optimization playbook**, providing the bridge from "what is slow" to "why it's slow at the instruction level." The prompt text guides the LLM to look for specific SIMD mnemonics (x86: `vmovaps`, `vfmadd`, `vaddps`; ARM: `fmla`, `ld1`) and notes that scalar-only code in a hot loop signals vectorization opportunities.

**Implementation:** `perflab/profilers/linux_perf.py` (`extract_hot_assembly()`), wired through `perflab/optimizers/agent.py` into `PromptContext.hot_loop_assembly`.

---

## Thread scheduling analysis

For multi-threaded C/C++ tasks, PerfLab captures thread scheduling statistics via `perf sched`:

- **`perf sched latency`:** Per-thread runtime, context switches, average/max scheduling delay
- **`perf sched timehist --summary`:** Per-CPU run time, thread migrations

High scheduling delays indicate lock contention or NUMA effects. Frequent thread migrations suggest poor affinity or over-subscription.

Auto-selected for `cpp` and `cuda` program types. Linux-only, auto-skipped elsewhere.

**Implementation:** `perflab/profilers/thread_sched.py`.

---

## GPU memory tracking

The power profiler also polls GPU memory utilization alongside power draw during benchmark execution. For each nvidia-smi sample, it captures:

- **Total VRAM** (MiB)
- **Peak/average used** (MiB)
- **Utilization percentage** (peak used / total)

High utilization (> 90%) indicates the workload is near OOM and may benefit from reduced batch sizes or mixed precision. The data appears in the dashboard under Diagnostics > GPU memory utilization. GPU memory summaries also feed into `PromptContext` and are rendered in the LLM prompt.

---

## Machine fingerprint

Every run's `report.json` includes a `hardware` section with the machine's key specs:

```json
{
  "hardware": {
    "cpu_model": "Apple M4 Max",
    "cpu_count": 16,
    "machine": "arm64",
    "platform": "macOS-15.3-arm64",
    "nvidia_gpus": [...],
    "cuda_version": "...",
    "cpp_compiler": "Apple clang version 16.0.0"
  }
}
```

This makes reports self-documenting and helps cross-run learning identify when strategies were developed on different hardware.

---

## Build flag overrides

PerfLab includes a build flag override mechanism that lets the agent suggest compiler flag changes without modifying `task.yaml`. The agent writes a `build_overrides.json` file:

```json
{"flags": ["-O3", "-march=native", "-flto"]}
```

The runner validates each flag against an allowlist of safe compiler options (optimization levels, architecture targeting, vectorization, OpenMP, LTO) and rejects anything dangerous. Flags already present in the build command are not duplicated.

**Allowlist includes:** `-O2`, `-O3`, `-Ofast`, `-march=native`, `-mtune=native`, `-mavx2`, `-mavx512f`, `-fopenmp`, `-funroll-loops`, `-ftree-vectorize`, `-ffast-math`, `-flto`, `-g`, and similar safe flags.

**Implementation:** `perflab/analyzers/build_overrides.py`.

---

**Demo task — `stream/python`:** A memory-bound STREAM benchmark (copy, scale, add, triad) with deliberately cache-unfriendly column-major traversal using scalar Python loops. The agent can discover numpy vectorization (eliminating Python loops) and row-major access patterns (improving cache locality). This exercises memray profiling, bench stats noise detection (via `raw_values`), and bottleneck analysis for memory-bound workloads.

---

## Error feedback to LLM

When the agent evaluates candidate patches that fail (correctness errors, contract violations), PerfLab captures the error output and feeds it back into the next iteration's prompt. This lets the LLM learn from its mistakes rather than repeating the same failing patterns.

Error feedback includes:
- **Correctness failures:** Exit code and stderr (truncated to 3000 chars)
- **Contract violations:** Which contract constraints were violated

The errors appear in the prompt under `## Errors from previous iteration` with the full error output in fenced code blocks. This is automatic — no configuration needed.

---

## Context window management

Each agent iteration builds a complete prompt from scratch — there is no multi-turn conversation or accumulated chat history between iterations. The LLM receives the full context it needs each time (source files, profiler data, bottleneck diagnosis, history) as a single system + user message pair.

Three layers prevent context window overflow:

### Layer 1 — Pre-send budget trimming

After assembling the full prompt, `build_prompt()` estimates total tokens (~4 chars/token) and checks against a budget. The budget comes from either an explicit `constraints.prompt_token_budget` in task.yaml or auto-inference from the model name.

Configure in `task.yaml`:

```yaml
constraints:
  prompt_token_budget: 8000   # approximate token limit (0 = auto-infer from model)
```

When the estimated token count exceeds the budget, sections are removed in priority order (lowest priority first):
1. Prior run context
2. Profile diff from previous iteration
3. GPU attribution
4. Cross-referenced optimization insights
5. Build flag recommendations
6. Training phase breakdown
7. Bottleneck diagnosis
8. Roofline analysis
9. *(emergency)* Optimization history truncated to last 2 entries
10. *(emergency)* Profiler summaries truncated to 2000 chars

Source files, benchmark results, and the request itself are never trimmed.

### Layer 2 — Auto context budget inference

Even without an explicit `prompt_token_budget`, PerfLab auto-infers a safe budget from the model name using known context window sizes:

| Model | Context Window | Safe Budget |
|-------|---------------|-------------|
| GPT-5.2 | 400K | ~356K |
| GPT-4o / GPT-5 | 128K | ~111K |
| Claude Sonnet/Opus | 200K | ~176K |
| Llama 3.2 | 128K | ~111K |

The formula is `(context_window - max_completion_tokens) * 0.9`, reserving space for the LLM's response plus a 10% safety margin. Optimization history is also capped to the last `max_history` iterations (default 3) to prevent prompt bloat.

### Layer 3 — Emergency overflow recovery

If the LLM API still returns a context length error despite pre-send trimming, the agent catches the exception, halves the current token estimate, re-trims with the aggressive budget, and retries once. If that also fails, the iteration is skipped and the agent loop continues to the next iteration. This ensures a single oversized prompt never crashes the optimization run.

---

## CPU roofline estimation

PerfLab can estimate CPU peak FLOPS and memory bandwidth from hardware specs, enabling roofline analysis for CPU-only tasks without requiring torch calibration.

The estimation uses:
- **Peak FLOPS:** `cores x SIMD_width x clock_GHz` where SIMD width accounts for FMA (e.g., AVX-512 = 32 FP32 FLOP/cycle, AVX2 = 16, NEON = 8)
- **Peak bandwidth:** Known specs for Apple Silicon chips, or `dmidecode` DDR info on Linux

Supported platforms:
- **macOS/Apple Silicon:** Detects M1-M4 variants (base/Pro/Max/Ultra) with known frequencies and bandwidths
- **Linux x86:** Reads `/proc/cpuinfo` flags for SSE/AVX/AVX-512, `lscpu` for clock speed
- **Fallback:** torch calibration (matmul + copy micro-benchmarks)

No configuration needed — `infer_cpu_peaks()` automatically tries spec-based estimation first.

---

## Bottleneck diagnosis coverage

The bottleneck diagnosis engine (`diagnose_bottlenecks()`) analyzes profiler summaries and generates ranked diagnoses with root causes and suggested actions. All profilers flow into diagnosis:

| Profiler | Detects |
|----------|---------|
| torch_profiler | GPU underutilization, CPU/GPU sync overhead, memory allocation overhead |
| pyspy | I/O hotspots, data loading bottlenecks |
| linux_perf | Low IPC, cache misses, branch misprediction, single-threaded execution |
| nsys | GPU idle time, kernel gaps, API overhead, host-device transfers |
| ncu | SM utilization, memory throughput, occupancy, warp divergence |
| metal_trace | Metal GPU utilization, submission overhead |
| jax | XLA recompilation, compilation fraction |
| memray | High peak memory, dominant allocator functions |
| ebpf | I/O syscall latency (read/write p99), excessive syscall count |
| lock_contention | Lock contention ratio, total wait time, false sharing (perf c2c HITM) |
| thread_sched | High scheduling delay, excessive thread migrations |
| power | GPU thermal throttling (power draw decline) |

All diagnoses include configurable thresholds in `task.yaml` under `analysis_thresholds`; thresholds are configurable per-task (see [ARCHITECTURE.md](ARCHITECTURE.md) for the full list). The number of bottlenecks surfaced is controlled by `constraints.top_n` (default: 3).

---

## Profiler tooling rationale

PerfLab uses **sampling profilers** (py-spy, perf) rather than instrumentation profilers (Tracy, Optick, Perfetto SDK) because PerfLab profiles arbitrary user code without requiring source modifications. Instrumentation profilers produce lower-overhead, higher-resolution traces, but they require adding macros or API calls to the profiled code — fundamentally incompatible with PerfLab's "point at a directory and go" model.

Py-spy outputs speedscope JSON format, which provides both structured hotspot data for the agent and timestamped samples for temporal GPU cross-referencing. Users can open the JSON in [Speedscope](https://www.speedscope.app/) for interactive flame graph visualization — the dashboard includes a direct link.

**Hardware tracers** (Intel PT/magic-trace, ARM SPE) are not included because they are platform-locked: Intel PT requires Intel Skylake+ on Linux, ARM SPE is not exposed on macOS. The data they produce (instruction-level traces) is more granular than what the LLM optimization loop needs — the LLM works best with function-level hotspots and hardware counter ratios, exactly what py-spy and perf stat provide.

**LBR (Last Branch Record)** is not included because: (1) Intel-only (AMD has BRS on Zen 3+ but different interface), (2) the existing stack already catches branch problems — TMA "Bad Speculation" identifies the bottleneck, profiler-driven flags suggest PGO (`-fprofile-generate/-fprofile-use`) which fixes all branches automatically using real data. LBR would tell you *which specific branch* mispredicts, but PGO fixes them all without needing per-branch diagnosis. For GPU code, branch divergence is covered by NCU warp-level metrics (branch efficiency, warp execution efficiency). LBR would be a valuable addition if CPU-heavy C++ users request per-branch granularity.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed rationale on each tooling decision.

---

## Data hints

Task authors can provide hints about their input data characteristics via the `data_hints` section in `task.yaml`. These help the LLM make better **algorithmic** choices that profiler data alone can't suggest:

```yaml
data_hints:
  sparsity: 0.95              # 95% zeros → consider sparse formats (CSR, COO)
  value_range: [-1.0, 1.0]    # small values → FP16 is safe, no overflow
  access_pattern: "sequential" # sequential → prefetching works; random → tiling critical
  batch_size_range: [1, 128]  # includes batch_size=1 → optimize for latency too
  dtype_safety: "fp16_safe"   # task author confirms FP16 won't hurt accuracy
  sequence_lengths: "variable_32_2048"
  custom:
    - "data is symmetric"
    - "output is always positive"
```

The profiler captures the *effects* of data characteristics (cache misses, branch patterns), but can't infer *why*. Telling the agent "data is 95% sparse" lets it suggest CSR format or sparse matmul — an algorithmic change the profiler alone would never propose.

Data hints are displayed by `perflab show-task` and documented in `perflab show-task-schema`.

---

## Platform-specific notes

### JAX on NVIDIA GPU

JAX on GPU gets the full NCU/NSys profiling pipeline — SM utilization, Tensor Core metrics, warp stall diagnosis, cache hierarchy (L1/L2/DRAM), bank conflicts, coalescing, occupancy limiters, and memory bandwidth. All GPU-side diagnostics work because XLA compiles to CUDA kernels that NCU and NSys can profile.

**What's different from CUDA:** SASS extraction is not available because XLA JIT-compiles at runtime (no static binary to disassemble). HLO attribution compensates — the LLM sees which XLA operations dominate device time (the TPU/GPU equivalent of SASS instruction classification, but at the operation level rather than instruction level).

### TPU architecture differences

TPU has a fundamentally different memory hierarchy from GPUs:
- **GPU:** Registers → L1/Shared Memory → L2 Cache → DRAM (HBM)
- **TPU:** Registers → VMEM (Vector Memory) → SMEM (Scalar Memory) → HBM

There is no L1/L2 cache on TPU — the memory hierarchy is explicitly managed by the XLA compiler. PerfLab's TPU bottleneck analysis focuses on what matters for TPU: MXU utilization (matrix unit efficiency), HLO padding waste (XLA's alignment overhead), infeed stalls (host-device data pipeline), and bf16 vs fp32 precision choices. These are the TPU equivalents of GPU cache hierarchy analysis.

### Apple Silicon (macOS)

On Apple Silicon, PerfLab runs with reduced profiling depth:

| Available | Not available (hardware limitation) |
|-----------|-------------------------------------|
| py-spy (CPU hotspots, needs sudo for SIP) | Linux perf (no hardware counters on macOS) |
| Metal trace via xctrace (GPU timing) | NCU / NSys (no NVIDIA GPU) |
| memray (Python memory profiling) | TMA Level 1/2/3 (requires perf or toplev) |
| MPS cross-profiler (torch + Metal join) | eBPF (no bpftrace on macOS) |
| CPU roofline (Apple chip specs known) | SASS extraction (no NVIDIA GPU) |

The agent still optimizes effectively using benchmark numbers + py-spy hotspots + Metal GPU timing, but can't provide NCU-level kernel diagnostics. Py-spy outputs speedscope JSON which can be opened in [Speedscope](https://www.speedscope.app/) for interactive flame graph visualization. For PyTorch MPS workloads, the MPS cross-profiler joins torch trace CPU data with Metal GPU data to identify the CPU/GPU split.

**Instruments integration:** macOS Instruments (via `xctrace`) provides deeper Metal GPU profiling (shader profiling, GPU counters, memory bandwidth) but the output format is complex XML/binary that requires Instruments-specific parsing. PerfLab currently uses xctrace for basic Metal trace data (GPU time, submissions, idle percentage). Deeper Instruments integration is possible but non-trivial — it would require parsing the Instruments trace archive format, which Apple does not document as a stable API.

---

## Structured configuration

PerfLab supports layered configuration via a `PerfLabConfig` dataclass hierarchy (`perflab/config.py`). Resolution order (highest priority wins):

1. **Environment variables** (`PERFLAB_LLM_MODEL`, `PERFLAB_BENCH_WARMUP`, etc.)
2. **Local project file** (`./perflab.yaml`)
3. **User config file** (`~/.config/perflab/config.yaml`)
4. **Built-in defaults**

**No files are required** — everything works out of the box with defaults and env vars. Config files are only needed when you want to change settings. Run `perflab init-config` to create a commented template, then uncomment and edit what you need.

The config is organized into sections, each controlling a different aspect of PerfLab's behavior:

| Section | What you can customize | Example |
|---------|----------------------|---------|
| `llm` | Provider, model, temperature, max tokens | Switch to Anthropic, use a different model |
| `benchmark` | Warmup iterations, repeat count | More repeats for noisy benchmarks |
| `agent` | Candidates per iteration, max iterations, wall-clock budget, LLM history depth, token budget | More candidates for thorough search, shorter runs for CI |
| `profiler` | FLOPS counting, roofline cache behavior | Disable FLOPS counting if it adds overhead |
| `analysis_thresholds` | All bottleneck detection thresholds (occupancy, IPC, cache miss rate, TC utilization, etc.) | Relax occupancy threshold for register-heavy HPC kernels; tighten cache miss threshold for latency-sensitive code |
| `mps` | Apple Silicon device selection | Choose which GPU on a multi-GPU Mac |
| `ollama` | Remote access, port allowlist | Allow connecting to a remote Ollama server |

Individual `task.yaml` settings override the config for that specific task, so you can set team-wide defaults in `perflab.yaml` and still fine-tune per task. Each agent run saves the fully resolved config as `resolved_config.json` for reproducibility.

CLI commands:
- `perflab init-config` — create `./perflab.yaml` with the default template (edit what you need)
- `perflab init-config --user` — create `~/.config/perflab/config.yaml` for personal defaults
- `perflab show-config` — display the resolved configuration with source indicators
- `perflab show-config-template` — emit a commented YAML template with all settings

Subprocess-only env vars (`PERFLAB_TORCH_PROFILE`, `PERFLAB_DETERMINISM_SEED`, etc.) are documented in the YAML template comments but are set automatically by PerfLab — users don't need to touch them.

---

## Iteration state artifact

The agent saves a `state.json` after each iteration, recording the current iteration index, best value, accepted patches, failure memory, and compiler diagnostics. This is a direct snapshot of `AgentContext` for debugging and post-hoc inspection — reading it is a single `json.load()` instead of replaying the event log. (An earlier checkpoint/resume mechanism was removed: it was never wired to a CLI flag and could not fire, since every run gets a fresh run directory.)

---

## Task authoring tools (MCP)

The MCP server includes 7 task-authoring tools designed to help new users create valid task specifications with AI assistance. These tools complement the existing profiling and optimization tools.

### Onboarding guide

`show_task_authoring_guide` returns a step-by-step walkthrough covering directory structure, required files, common pitfalls, and which tools to use at each step. This is the recommended starting point for new users.

### Scaffolding

`create_task` generates a complete task directory with all required files — task.yaml, bench.py, tests.py, a source file template, and tuning.yaml — pre-configured for any of the 6 supported program types (python, pytorch, jax, triton, cpp, cuda). Templates include:

- **GPU-aware timing** — PyTorch and Triton templates include `torch.cuda.synchronize()`, JAX templates include `block_until_ready()`
- **Benchmark protocol** — Generated bench.py honors `PERFLAB_BENCH_WARMUP` / `PERFLAB_BENCH_REPEATS` env vars, writes JSON to `--json` path
- **Build integration** — C++ and CUDA templates include appropriate build commands (g++/nvcc)
- **Contract stubs** — Optional fixed_params are wired into task.yaml's contract section

### Validation

`validate_task` checks a task.yaml without running it. Catches:

- Missing required fields (name, program_type, correctness, benchmark)
- Invalid program_type or metric mode
- Blocklisted files in edit_policy (tests.py, bench.py, task.yaml)
- Missing build command for compiled languages (cpp, cuda)
- Contract validation errors (malformed dotted paths, non-numeric fixed_params)
- Missing workspace files (bench.py, tests.py, allowed_paths)
- Full `TaskSpec.load()` round-trip

### Bench.py linting

`lint_bench_script` checks a benchmark harness for PerfLab protocol compliance:

- `--json` argument acceptance
- JSON output writing
- `PERFLAB_BENCH_WARMUP` / `PERFLAB_BENCH_REPEATS` env var support
- `"ok"` field in output
- GPU synchronization (CUDA `synchronize()`, JAX `block_until_ready()`)

### Intelligent suggestions

Three tools provide context-aware recommendations:

- **`suggest_profilers`** — Recommends profiler plan based on program type and target hardware (e.g., removes NVIDIA profilers for Apple Silicon, adds Metal trace; adds JAX profiler for TPU). Includes rationale for each profiler.
- **`suggest_thresholds`** — Recommends analysis threshold overrides (e.g., lower IPC threshold for Python, tighter SM utilization for CUDA, TPU-specific MXU/padding/infeed thresholds).
- **`suggest_contract`** — Analyzes bench.py to identify problem dimensions (M, N, K, batch_size, etc.) that should be locked as fixed_params, detects output metric fields for required_bench_fields, and recommends min_repeats based on program type.

### Typical workflow

```
AI assistant                          PerfLab MCP
    │                                      │
    ├─ show_task_authoring_guide ──────────►│ "Here's how to create a task..."
    │                                      │
    ├─ create_task(name="attention",  ─────►│ Scaffolds tasks/custom/attention/
    │      program_type="pytorch")         │   with 5 files
    │                                      │
    │  [user edits source + tests]         │
    │                                      │
    ├─ lint_bench_script ──────────────────►│ "2 warnings: missing CUDA sync..."
    ├─ suggest_contract ───────────────────►│ "Lock batch_size, seq_len as fixed"
    ├─ suggest_profilers ──────────────────►│ "Use torch_trace + ncu"
    ├─ suggest_thresholds ─────────────────►│ "Set gpu_cpu_ratio_low: 0.5"
    │                                      │
    ├─ validate_task ──────────────────────►│ "Valid ✓ (2 warnings)"
    │                                      │
    ├─ profile_task ───────────────────────►│ Baseline profiling
    └─ start_agent ────────────────────────►│ Optimization begins
```

## Backend coverage summary

Comprehensive single-device optimization coverage across all supported backends.

### CUDA / NCU

| Layer | Coverage |
|-------|----------|
| **Profiling (20+ metric categories)** | SM utilization, memory/compute throughput, achieved occupancy, L1/L2 hit rates, DRAM bytes + achieved bandwidth, branch efficiency, warp execution efficiency, Tensor Core utilization + throughput, warp stall reasons (15 categories including Hopper Stall GMMA), bank conflicts, sectors per request (coalescing), occupancy limiters (registers/shared mem/block size), instruction mix (FP32/FP64/INT/SFU), TMA pipe utilization, register spills (local memory bytes), source-line hotspots |
| **Bottleneck rules (12)** | Compute-bound vs memory-bound, low SM utilization, memory bandwidth saturation, low occupancy, high register pressure, branch divergence, low warp execution efficiency, Tensor Core underutilization, CUDA-cores-only compute-bound, dominant warp stall diagnosis, bank conflict detection, uncoalesced access detection, occupancy limiter diagnosis, FP64-on-consumer-GPU, register spill detection |
| **Prompt suggestions** | WMMA/mma.sync, cp.async pipelining, TMA descriptors, Thread Block Clusters, warp specialization, CUB/Thrust primitives, cudaMallocAsync, bank conflict padding, tile swizzling, CUDA Graphs, vectorized loads, thread coarsening, cooperative groups |
| **Optimization detection** | Shared memory, launch_bounds, loop tiling, CUDA Graphs, pinned memory, async operations, cooperative groups, WMMA/mma.sync, cp.async, CUB/Thrust, cudaMallocAsync |

### PyTorch

| Layer | Coverage |
|-------|----------|
| **Profiling** | Per-operator timing + FLOPS counting (`with_flops=True`), GPU kernel breakdown, CPU vs GPU ratio, sync point detection, memory allocation stats, phase extraction (forward/backward/optimizer/data_loading), operator shapes |
| **Bottleneck rules (7)** | GPU underutilization / CPU dispatch bottleneck, excessive GPU synchronization, memory allocation overhead, dominant GPU kernel, phase dominance + per-phase GPU underutilization, non-contiguous tensor copies, Tensor Core alignment checking, memory fragmentation |
| **Prompt suggestions** | FlexAttention, torchao int8/int4/float8 quantization, semi-structured 2:4 sparsity, nested tensors, torch.compile modes (reduce-overhead, max-autotune, fullgraph), SDPA, AMP, channels_last, CUDA Graphs, fused optimizers |
| **Optimization detection** | torch.compile, AMP/autocast, SDPA/flash attention, channels_last, pin_memory, num_workers, float32_matmul_precision, inference_mode/no_grad, cuDNN benchmark, persistent_workers, prefetch_factor, non_blocking, CUDA Graphs, FlexAttention, torchao, nested tensors, 2:4 sparsity |

### JAX

| Layer | Coverage |
|-------|----------|
| **Profiling** | XLA compilation metrics (count, time, recompilations), HLO dump analysis (module count, op breakdown), HLO cost analysis (FLOP + bytes_accessed estimates), TPU device detection, JAX trace parsing (host-device time split, MXU utilization, infeed stalls) |
| **Bottleneck rules (8)** | XLA recompilation, high compilation overhead, excessive compilations, low MXU utilization (TPU), XLA padding waste (TPU), infeed stalls (TPU), fragmented HLO modules (TPU), fp32-instead-of-bf16 (TPU) |
| **Prompt suggestions** | Pallas kernels (GPU + TPU), shard_map, XLA GPU flags (latency hiding scheduler, async collectives, Triton GEMM routing), FP8/AQT, AOT lowering cost analysis, donate_argnums, jax.checkpoint with rematerialization policies, scan vs Python loops |
| **Optimization detection** | jax.jit, mixed precision (bfloat16/float16), buffer donation (donate_argnums), shard_map, Pallas kernels, gradient checkpointing |

### Triton

| Layer | Coverage |
|-------|----------|
| **Profiling** | Via NCU (all CUDA metrics apply to Triton-compiled kernels) + NSys (kernel timeline, launch gaps) |
| **Bottleneck rules** | All NCU rules apply (Tensor Core, warp stalls, bank conflicts, coalescing, etc.) |
| **Prompt suggestions** | @triton.autotune, warp specialization (num_consumer_groups, num_buffers_warp_spec), TMA descriptors, persistent kernels, SplitK decomposition, tl.dot with allow_tf32, tile index swizzling, num_warps/num_stages tuning |
| **Optimization detection** | Via CUDA detection patterns |

### C++ / CPU

| Layer | Coverage |
|-------|----------|
| **Profiling** | Linux perf (IPC, cache miss rate, branch miss rate, cpus_utilized, function hotspots, annotated hotspots), TMA (Top-Down Microarchitecture Analysis), ISA feature detection |
| **Bottleneck rules (6)** | Low IPC, high cache miss rate, high branch misprediction, CPU hotspot dominance, single-threaded execution detection, vectorization width mismatch |
| **Prompt suggestions** | SIMD intrinsics (AVX-512/AVX2/NEON), FMA, loop tiling, OpenMP, software prefetching, cache-line alignment, streaming stores, restrict pointers, LTO |
| **Optimization detection** | SIMD intrinsics, OpenMP, threading (std::thread/pthread), CUDA kernel launches, __restrict__ pointers |

### Roofline

| Layer | Coverage |
|-------|----------|
| **Hierarchical model** | DRAM bandwidth ceiling + L2 cache bandwidth ceiling (A100, H100, RTX 4090/4080, RTX 3090, V100) |
| **Per-dtype ceilings** | FP64, FP32, TF32, FP16, BF16 peak TFLOPS lines for each GPU model |
| **Profiler FLOPS integration** | PyTorch `with_flops=True` + JAX HLO cost annotations feed into roofline point computation without ncu |
| **Bound classification** | Memory-bound vs compute-bound with DRAM vs L2 bottleneck level identification |
| **Tiered playbook** | 5 utilization tiers (structural → micro) × 6 program types = 30 action sets, plus 12 bound × program_type action sets |

### Cross-profiler

| Feature | Coverage |
|---------|----------|
| **NSys** | CPU-bound detection, CUDA API overhead, kernel dominance, launch gaps, data transfer bottleneck, hot-path cudaMalloc/cudaFree detection |
| **Host-device** | Excessive synchronization, small kernel overhead, transfer dominance, CPU-blocks-GPU |
| **GPU attribution** | CPU-to-GPU call graph from NSys correlation, ranked kernel attribution, launch overhead, pipeline stall flags |
| **Metal (Apple GPU)** | GPU underutilization, blit transfer overhead, GPU idle time, ALU utilization |
| **Memory (memray)** | Peak memory usage, dominant allocator detection |
| **I/O (eBPF)** | Read/write latency p99, excessive syscall count |
| **Lock contention** | Contention ratio, total wait time, false sharing (HITM) |
| **Thread scheduling** | Scheduling delays, excessive thread migrations |
| **Power/thermal** | GPU thermal throttling detection |
