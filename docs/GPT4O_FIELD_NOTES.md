# GPT-4o Issues with SEARCH/REPLACE Patch Format

Observed during agent optimization runs on the `inference_pipeline/pytorch` task
using `gpt-4o` via the OpenAI API. All evidence can be found in the agent event
logs and raw LLM response files under `out/runs/`.

---

## Issue 1: Hallucinated SEARCH Text

**Severity: Critical** — Causes validation failures, wasting an entire candidate slot.

GPT-4o frequently produces SEARCH blocks that don't match the actual file
content. Instead of copying the code verbatim, it *imagines* what the code looks
like, introducing small changes such as added function parameters, modified
docstrings, or invented comments.

### Example A: Added function parameter (99% similar, still fails)

The actual code:
```python
def preprocess_image(image_tensor: torch.Tensor) -> torch.Tensor:
```

What GPT-4o wrote in the SEARCH block:
```python
def preprocess_image(image_tensor: torch.Tensor, device: str) -> torch.Tensor:
```

**Evidence:** `out/runs/20260302-235851-68f56259/llm_responses/iter1_response.txt`,
Candidate 1. Validation error logged in `agent_events.jsonl` (iter 1, cand 0):
"99% similar" match but exact-match fails.

### Example B: Added keyword arguments to tensor constructors

The actual code:
```python
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
```

What GPT-4o wrote:
```python
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device)
```

**Evidence:** `out/runs/20260302-235354-adc5b996/agent_events.jsonl` (iter 4,
cand 3, "67% similar") and `out/runs/20260302-235851-68f56259/agent_events.jsonl`
(iter 1, cand 4, "86% similar").

### Example C: Changed docstring text

The actual code:
```python
    """Convert logits to predictions — per-image .cpu() and softmax."""
```

What GPT-4o wrote:
```python
    """Convert logits to predictions — softmax on GPU."""
```

**Evidence:** `out/runs/20260302-235354-adc5b996/agent_events.jsonl` (iter 4,
cand 1, "71% similar").

### Example D: Searching for code that doesn't exist yet

After the model *describes* an optimization in its reasoning, it sometimes puts
the *optimized* code (not the original) in the SEARCH block, as if the change
has already been applied.

```
SEARCH: '            processed_batch = torch.stack([preprocess_image(img) for img in batch_images])'
ACTUAL: '            processed = preprocess_image(img)'
```

**Evidence:** `out/runs/20260302-235354-adc5b996/agent_events.jsonl` (iter 2,
cand 4, "67% similar"). Also `out/runs/20260303-001401-cea99c95/agent_events.jsonl`
(iter 1, cand 2 and cand 5).

### Impact

In the run *without* fuzzy matching (`20260302-235851-68f56259`), this issue
caused **9 validation failures** across 5 iterations and **0 accepted patches**
(complete failure). With fuzzy matching enabled (`20260303-001401-cea99c95`),
validation failures dropped to **2** and the agent achieved a **5.08x speedup**.

---

## Issue 2: Markdown Code Fences Wrapping Edit Blocks

**Severity: High** — Causes 0 candidates to parse from an entire LLM response.

Despite the system prompt explicitly saying "Do NOT wrap edit blocks in markdown
code fences", GPT-4o frequently wraps the FILE/SEARCH/REPLACE blocks inside
triple-backtick code fences.

### Example

```
```python
FILE: pipeline.py
<<<<<<<
    model = SmallCNN().to(device)
=======
    model = torch.compile(SmallCNN().to(device))
>>>>>>> REPLACE
```                                                    <-- fence closes here
```

The parser can't find the `FILE:` marker because it's inside a code block.

**Evidence:** Found in 9 out of 11 runs. Example:
`out/runs/20260302-235851-68f56259/llm_responses/iter3_response.txt` (all 6
candidates wrapped in fences, 0 parsed).

Full list of affected responses:
- `20260302-225839-1879cdaf/iter1_response.txt`
- `20260302-230517-dbc6dd96/iter1_response.txt`
- `20260302-231000-6696cd34/iter1_response.txt`
- `20260302-231349-de1ab45f/iter2_response.txt`
- `20260302-234354-cc18bb32/iter5_response.txt`
- `20260302-234711-600b3ad1/iter1_response.txt`
- `20260302-235354-adc5b996/iter7_response.txt`
- `20260302-235851-68f56259/iter2_response.txt`, `iter3_response.txt`
- `20260303-001401-cea99c95/iter3_response.txt`

### Fix Applied

Added `_strip_code_fences()` in `perflab/optimizers/patch.py` to remove
triple-backtick lines before parsing.

---

## Issue 3: Missing SEARCH/REPLACE Labels on Markers

**Severity: Medium** — Causes candidates to fail parsing.

GPT-4o sometimes uses bare conflict markers (`<<<<<<<`) without the required
`SEARCH` label (`<<<<<<< SEARCH`). The parser requires the exact string
`<<<<<<< SEARCH` to identify the start of a search block.

### Example

Correct format:
```
<<<<<<< SEARCH
    model.eval()
=======
    model = torch.compile(model).eval()
>>>>>>> REPLACE
```

What GPT-4o sometimes produces:
```
<<<<<<<
    model.eval()
=======
    model = torch.compile(model).eval()
>>>>>>> REPLACE
```

**Evidence:** `out/runs/20260302-235851-68f56259/llm_responses/iter3_response.txt`
— all candidates use bare `<<<<<<<` markers. Across all runs: 312 correct
`<<<<<<< SEARCH` markers vs 8 bare `<<<<<<<` markers.

---

## Issue 4: Correctness Failures from Broken Code

**Severity: Medium** — Candidate evaluates but produces code that fails tests.

Many candidates pass validation (SEARCH text matches) but produce code that
doesn't actually work. Common causes:
- Calling `torch.cuda.*` APIs on Apple MPS backend
- Breaking the output contract (wrong return shape or missing keys)
- Introducing syntax errors in multi-block patches

### Impact by Run

| Run | Validation Fails | Correctness Fails | Accepted |
|-----|------------------|--------------------|----------|
| `20260302-235851-68f56259` (no fuzzy) | 9 | 1 | 0 |
| `20260302-235354-adc5b996` (n_cand bug) | 6 | 10 | 1 |
| `20260303-001401-cea99c95` (with fuzzy) | 2 | 11 | 3 |

With fuzzy matching, more candidates pass validation but then fail correctness
— the bottleneck shifts from "can't match the code" to "produces broken code".

**Evidence:** `agent_events.jsonl` in each run, events with
`"event_type": "candidate_correctness"` and `"passed": false`.

---

## Issue 5: Inconsistent Multi-Block Patches

**Severity: Medium** — Later blocks in a multi-block patch assume earlier blocks
have already been applied.

When GPT-4o produces a candidate with multiple SEARCH/REPLACE blocks for the
same file, it sometimes writes Block 2's SEARCH text as if Block 1 has already
been applied. Since blocks are validated against the *original* file content
before any are applied, Block 2 fails validation.

### Example

Block 1 changes `batch_size: 1` to `batch_size: 8` in tuning.yaml.
Block 2's SEARCH text references `batch_size: 8` (the post-Block-1 state).

**Evidence:** `out/runs/20260302-235851-68f56259/agent_events.jsonl` (iter 1,
cand 1): Block 1 for pipeline.py is valid but Block 2 for tuning.yaml searches
for `batch_size: 16` when the file contains `batch_size: 1`.

---

## Issue 6: Hallucinated GPU Claims in Optimization Summary

**Severity: Medium** — Misleads users about what optimizations actually achieved.

When GPT-4o generates the post-run optimization summary, it fabricates claims about
GPU throughput improvements even when the profiler data shows **0% GPU kernel time**.
On Apple MPS, the torch profiler cannot observe Metal GPU kernels — so
`total_gpu_kernel_us` is always 0 — but GPT-4o interprets the optimization
descriptions (batching, `torch.compile`, reducing synchronization) as GPU
improvements and invents GPU-specific claims.

### Example

The optimization summary from run `cea99c95` states:

> "Initially, batch processing was implemented to process multiple images
> simultaneously, minimizing CPU dispatch overhead and **increasing GPU workload
> per dispatch**. In a subsequent iteration, **the batch size was further
> increased to maximize GPU throughput**, and `torch.compile` was used to reduce
> Python-level and dispatch overhead…"

However, the torch profiler summary for the same run shows:

```json
"cpu_vs_gpu": {
    "total_cpu_op_us": 323911.3,
    "total_gpu_kernel_us": 0,
    "ratio": 0.0
}
```

The bottleneck diagnosis from `report.md` in the same run confirms:

| Rank | Bottleneck | Confidence |
|---:|---|:---:|
| 1 | GPU underutilized — CPU dispatch is bottleneck (GPU/CPU ratio=0.00) | high |
| 3 | MPS GPU active only 0% of trace time | medium |

**Evidence:**
- `out/runs/20260303-001401-cea99c95/optimization_summary.md` — full text of
  the hallucinated summary
- `out/runs/20260303-001401-cea99c95/artifacts/torch_profiler_summary.json` —
  `cpu_vs_gpu.total_gpu_kernel_us = 0`
- `out/runs/20260303-001401-cea99c95/report.md` — bottleneck table showing 0%
  GPU utilization

### Root Cause

The optimization summary LLM call originally received only the patch
descriptions (which mention batching, compile, etc.) without any profiler
context. GPT-4o assumed that these optimizations improved GPU throughput because
that's the typical effect on CUDA — but on MPS the torch profiler can't see
Metal GPU kernels, so GPU time always reads 0.

### Fix Applied

Added device name and CPU/GPU profiler breakdown to the summary prompt in
`perflab/optimizers/agent.py` (`_generate_optimization_summary`). When
`device == "mps"`, an explicit instruction is appended: "Do NOT claim GPU
utilization improvements — say 'device' or 'MPS' instead."

---

## Issue 7: Unable to Discover Core Algorithmic Optimizations

**Severity: High** — Agent fails to improve tasks where the key optimization requires a fundamental code rewrite rather than incremental edits.

### matmul/python — Failed to discover numpy vectorization

The naive baseline uses triple-nested Python loops for matrix multiplication (~0.0001 TFLOPS). The expected optimization is replacing the loops with `numpy` (100x+ speedup). Across 5 iterations and ~25 candidates, GPT-4o:

- Attempted minor loop tweaks (loop unrolling, variable hoisting) that produced no measurable improvement
- Generated 7 correctness failures from broken rewrites
- Never proposed the key insight: replace the loops with `np.matmul()` or `@` operator

**Evidence:** `tasks/matmul/python/out/runs/20260303-033339-c77dc9e7/`

### transformer_train/pytorch — Massive correctness failure rate

Baseline: 72,900 tokens/sec. Expected optimizations: AMP, SDPA attention, `torch.compile`. Across 5 iterations and ~30 candidates:

- **22 out of ~30 candidates failed correctness** — the dominant failure mode
- Candidates that compiled typically regressed performance (18k–26k tok/s vs 73k baseline)
- Multi-block patches frequently had inconsistent SEARCH text (see Issue 5)
- The model repeatedly tried to restructure the entire `TransformerBlock` class, producing patches that couldn't match the file

**Evidence:** `tasks/transformer_train/pytorch/out/runs/20260303-085114-85da36d2/`

### matmul/pytorch — All candidates regressed on MPS

Baseline: 3.16 TFLOPS on Apple MPS. Across 5 iterations:

- Every candidate that passed correctness was slower than baseline (0.06–2.85 TFLOPS)
- GPT-4o tried `torch.compile`, `nn.Linear`, dtype changes — all regressed on MPS
- The model lacks awareness that MPS has different performance characteristics than CUDA

**Evidence:** `tasks/matmul/pytorch/out/runs/20260303-033844-2b78cf24/`

### dataloader_bottleneck/pytorch — Improved but repeatedly violated contract

Baseline: 365 samples/sec → Best: 892 samples/sec (**2.4x**). The agent found `num_workers` and `pin_memory` optimizations, but:

- **Repeatedly tried to change `batch_size`** (to 64, 128, 256) despite `contract.fixed_params` enforcing `batch_size: 32`. This wasted ~40% of candidate slots across 12 iterations.
- The model sees batch_size as an obvious throughput lever and ignores the contract constraint even after multiple rejections with explicit "Contract violation: meta.batch_size=128, expected 32" error messages.

**Evidence:** `tasks/dataloader_bottleneck/pytorch/out/runs/20260303-085425-56c0c616/`

---

## Full Mac Validation Results (gpt-4o, Apple Silicon M4)

| Task | Baseline | Best | Speedup | Outcome |
|------|----------|------|---------|---------|
| inference_pipeline/pytorch | 961 img/s | 5,707 img/s | 5.9x | Improved |
| dataloader_bottleneck/pytorch | 365 samp/s | 892 samp/s | 2.4x | Improved (but contract violations) |
| matmul/cpp | 0.003 TFLOPS | 0.029 TFLOPS | ~10x | Improved |
| matmul/pytorch | 3.16 TFLOPS | 3.16 TFLOPS | 1x | All candidates regressed |
| matmul/python | 0.0001 TFLOPS | 0.0001 TFLOPS | 1x | Failed — couldn't find numpy |
| transformer_train/pytorch | 72,900 tok/s | 72,900 tok/s | 1x | Failed — correctness failures |
| matmul/jax | — | — | — | JAX not installed |
| transformer_train/jax | — | — | — | JAX not installed |

---

## Summary Statistics

Across the three key runs on `inference_pipeline/pytorch`:

| Metric | No Fuzzy | With n_cand Bug | With Fuzzy |
|--------|----------|-----------------|------------|
| Run ID | `68f56259` | `adc5b996` | `cea99c95` |
| Iterations | 5 | 7 | 11 |
| Total candidates evaluated | 16 | ~20 | ~60 |
| Validation failures | 9 (56%) | 6 (30%) | 2 (3%) |
| Correctness failures | 1 | 10 | 11 |
| Zero-parse responses | 2 | 1 | 2 |
| Accepted patches | 0 | 1 | 3 |
| Final speedup | 1.0x | 2.48x | 5.08x |

## Mitigations Applied

1. **`_strip_code_fences()`** — Strips markdown code fences before parsing
   (`perflab/optimizers/patch.py`)
2. **`_fuzzy_match_and_correct()`** — Auto-corrects SEARCH blocks that are
   >=80% similar to actual file content (`perflab/optimizers/patch.py`)
3. **Explicit prompt instruction** — "Do NOT wrap edit blocks in markdown code
   fences" added to system prompt (`perflab/optimizers/prompt.py`)
4. **Device-specific guidance** — Warning against `torch.cuda.*` on MPS
   (`perflab/optimizers/prompt.py`)
5. **Profiler-grounded summaries** — Summary LLM call now receives device name,
   CPU/GPU breakdown, and MPS-specific instruction to avoid GPU claims
   (`perflab/optimizers/agent.py`)

## Where to Find the Evidence

The runs referenced in this document were generated before per-task output
directories were implemented. They are located in the repo-level `out/runs/`.
Future runs will appear under `tasks/<task_name>/<variant>/out/runs/`.

All raw data is in `out/runs/<run_id>/`:

```
out/runs/<run_id>/
  agent_events.jsonl     # Structured event log (validation errors, correctness results)
  llm_responses/         # Raw LLM response text per iteration
    iter1_response.txt
    iter2_response.txt
    ...
  report.md              # Human-readable run summary
  bench.json             # Final benchmark results
  dashboard.html         # Visual dashboard with metric history
```
