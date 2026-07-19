"""End-to-end test for the optimizer agent loop (perflab.optimizers.agent.run_agent).

Every layer below run_agent (phases/{baseline,generate,prescreen,evaluate,
autotune,finalize}, the LLM-response patch parser, the benchmark/correctness
runners) already has unit coverage, but nothing exercises the *composed*
loop end to end. This module does: real subprocesses for bench/correctness,
real file I/O, real phase orchestration, real report generation. The only
faked boundary is the LLM provider.

Injection seam: run_agent(..., provider=...) already accepts a provider
instance directly -- the exact seam perflab.cli's `agent` command leaves at
its default (None -> create_provider(llm_config)) and that
tests/test_crash_finalize.py already uses to hand in a stub. That is
narrower than monkeypatching perflab.llm.config.create_provider (which would
require patching a module-level name and still funnel through the same
parameter), so ScriptedProvider is passed straight in as a constructor
argument -- no monkeypatching needed anywhere in this file.

Fixture task: a tiny pure-Python "sum of squares over range(n)" workload
modeled on tasks/matmul/python (same task.yaml/bench.json shape). The
baseline implementation is an explicit O(n) Python loop (~25ms for
n=1_000_000, so warmup=1/repeats=3 keeps one full benchmark invocation under
~150ms); the scripted "genuinely faster" patch replaces it with the O(1)
closed-form triangular/square-pyramidal formula, which numpy-free tests.py
can verify exactly (integer arithmetic, no floating-point tolerance needed).
"""
from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from perflab.llm.base import CompletionResult, Message
from perflab.llm.config import LLMConfig
from perflab.optimizers.agent import AgentConfig, run_agent
from perflab.optimizers.progress import ListProgress
from perflab.task_spec import TaskSpec

# --- Fixture task source, shared between the on-disk file and the scripted
# --- LLM patch response so the SEARCH block is byte-identical to the file
# --- by construction (no risk of a hand-copied mismatch).

_ORIGINAL_FUNC = '''def sum_of_squares(n: int) -> int:
    """Compute sum_{i=0}^{n-1} i**2 via an explicit loop (deliberately slow)."""
    total = 0
    for i in range(n):
        total += i * i
    return total'''

_PATCHED_FUNC = '''def sum_of_squares(n: int) -> int:
    """Compute sum_{i=0}^{n-1} i**2 via the closed-form formula (O(1))."""
    return (n - 1) * n * (2 * n - 1) // 6'''

_SUMSQ_SOURCE = (
    '"""Intentionally slow sum-of-squares over range(n) using an explicit '
    'Python loop.\n\n'
    'An optimizing agent should replace the loop with a closed-form formula.\n'
    '"""\n'
    "from __future__ import annotations\n\n\n"
    f"{_ORIGINAL_FUNC}\n"
)

_BENCH_N = 1_000_000

_BENCH_SOURCE = textwrap.dedent(f'''\
    """Benchmark for the sum-of-squares e2e fixture task."""
    from __future__ import annotations

    import argparse
    import json
    import os
    import time
    from pathlib import Path

    from sumsq import sum_of_squares

    N = {_BENCH_N}


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--json", required=True)
        args = ap.parse_args()

        warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 1))
        repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 3))

        for _ in range(warmup):
            sum_of_squares(N)

        times_ms = []
        result = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = sum_of_squares(N)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        sorted_times = sorted(times_ms)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(0.95 * (len(sorted_times) - 1))]

        out = {{
            "meta": {{"n": N, "warmup": warmup, "repeats": repeats}},
            "times_ms": times_ms,
            "latency_ms": {{"median": p50, "p95": p95}},
            "result": result,
            "ok": True,
        }}
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps({{"latency_ms_median": p50}}, indent=2))


    if __name__ == "__main__":
        main()
''')

_TESTS_SOURCE = textwrap.dedent('''\
    """Correctness test for the sum-of-squares e2e fixture task (no numpy;
    the closed-form patch is exact integer arithmetic, so equality holds for
    both the baseline loop and the optimized candidate)."""
    from __future__ import annotations

    from sumsq import sum_of_squares


    def _reference(n: int) -> int:
        return sum(i * i for i in range(n))


    def main() -> None:
        for n in (0, 1, 5, 37, 200):
            got = sum_of_squares(n)
            want = _reference(n)
            assert got == want, f"sum_of_squares({n}) = {got}, expected {want}"
        print("ok")


    if __name__ == "__main__":
        main()
''')

_TASK_YAML = textwrap.dedent('''\
    name: "sumsq_e2e_fixture"
    workspace: "."
    program_type: "python"
    target_hardware: null
    build: null
    correctness:
      cmd: "python tests.py"
      expected_exit: 0
    benchmark:
      cmd: "python bench.py --json out/bench.json"
      metric:
        name: "latency_ms.median"
        mode: "minimize"
      warmup: 1
      repeats: 3
    constraints:
      regression_tolerance: 0.02
      rlimit_as_gb: 2
    contract:
      min_repeats: 3
      required_bench_fields: ["ok", "latency_ms"]
    edit_policy:
      allowed_paths:
        - "sumsq.py"
    out_dir: "out"
''')

_PROTECTED_FILENAMES = ("task.yaml", "tests.py", "bench.py")


def _write_task(tmp_path: Path) -> tuple[TaskSpec, Path]:
    """Write a tiny real python task (source, bench, tests, task.yaml) to tmp_path."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "sumsq.py").write_text(_SUMSQ_SOURCE, encoding="utf-8")
    (ws / "bench.py").write_text(_BENCH_SOURCE, encoding="utf-8")
    (ws / "tests.py").write_text(_TESTS_SOURCE, encoding="utf-8")
    task_file = ws / "task.yaml"
    task_file.write_text(_TASK_YAML, encoding="utf-8")
    task = TaskSpec.load(task_file)
    return task, task_file


def _agent_config(max_iters: int) -> AgentConfig:
    """Mirror perflab.cli's `agent` command construction, scaled down for a test."""
    return AgentConfig(
        n_candidates=1,
        top_k=1,
        max_iters=max_iters,
        early_stop=True,
        fast_screen=True,
        max_wall_time_s=60,
        isolation=None,
    )


def _good_patch_response() -> str:
    """A well-formed single-candidate search/replace response that genuinely
    speeds up sum_of_squares (O(n) loop -> O(1) closed form)."""
    return (
        "--- CANDIDATE 1 ---\n"
        "Replace the O(n) Python loop with the closed-form sum-of-squares "
        "formula to remove the per-element interpreter overhead entirely.\n\n"
        "FILE: sumsq.py\n"
        "<<<<<<< SEARCH\n"
        f"{_ORIGINAL_FUNC}\n"
        "=======\n"
        f"{_PATCHED_FUNC}\n"
        ">>>>>>> REPLACE\n"
    )


def _garbage_patch_response() -> str:
    """A malformed response: has FILE:/SEARCH markers but is cut off before
    the '=======' divider ever arrives -- simulates a truncated/misbehaving
    LLM completion. parse_patch_response must drop it (never apply a partial
    edit) and surface a warning for the next prompt's error feedback."""
    return (
        "--- CANDIDATE 1 ---\n"
        "I will fix the slow loop by rewriting it, one moment...\n\n"
        "FILE: sumsq.py\n"
        "<<<<<<< SEARCH\n"
        "def sum_of_squares(n: int) -> int:\n"
        "    total = 0\n"
    )


@dataclass
class ScriptedProvider:
    """LLMProvider test double: no network calls, returns pre-scripted
    responses from a queue, one per call to complete().

    Satisfies the LLMProvider protocol (perflab.llm.base) exactly: `name`
    attribute, `is_available()`, `complete()` with the same keyword-only
    signature real providers use, and a `usage` dict shaped so
    perflab.optimizers.progress.usage_input_tokens/usage_output_tokens (and
    the truncation check in phases/generate.py, which reads
    result.finish_reason) read it correctly.
    """

    responses: list[str]
    name: str = "scripted"
    calls: list[list[Message]] = field(default_factory=list)
    _idx: int = field(default=0, init=False)

    def is_available(self) -> bool:
        return True

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
    ) -> CompletionResult:
        self.calls.append(list(messages))
        # Repeat the last scripted response if called more times than
        # scripted (e.g. an unexpected extra summary call) rather than
        # raising IndexError and masking the real assertion failure.
        content = self.responses[min(self._idx, len(self.responses) - 1)]
        self._idx += 1
        return CompletionResult(
            content=content,
            finish_reason="stop",
            usage={"input_tokens": 111, "output_tokens": 42},
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
    ):
        result = self.complete(
            messages, temperature=temperature, max_tokens=max_tokens,
            json_mode=json_mode, stop=stop,
        )
        yield result.content


def _protected_snapshot(ws: Path) -> dict[str, bytes]:
    return {name: (ws / name).read_bytes() for name in _PROTECTED_FILENAMES}


def _event_types(run_dir: Path) -> list[str]:
    events_path = run_dir / "agent_events.jsonl"
    types = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            types.append(json.loads(line)["event_type"])
    return types


def _events_of_type(run_dir: Path, event_type: str) -> list[dict]:
    events_path = run_dir / "agent_events.jsonl"
    out = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev["event_type"] == event_type:
            out.append(ev)
    return out


class TestAgentLoopHappyPath:
    """Scenario A: a genuinely-faster scripted patch should be measured,
    accepted, applied to the real workspace, and reported."""

    def test_accepts_faster_candidate_and_reports(self, tmp_path: Path) -> None:
        task, task_file = _write_task(tmp_path)
        before = _protected_snapshot(task.workspace)

        provider = ScriptedProvider(responses=[
            _good_patch_response(),
            "Replaced an O(n) Python loop with an O(1) closed-form formula, "
            "eliminating per-element interpreter overhead.",
        ])
        progress = ListProgress()

        result = run_agent(
            task, task_file,
            _agent_config(max_iters=1),
            LLMConfig(provider="anthropic", model="test-scripted-model"),
            progress=progress,
            provider=provider,
        )

        # --- Baseline was measured, candidate beat it ---
        assert result.baseline_value > 0
        assert result.best_value > 0
        assert result.best_value < result.baseline_value
        speedup = result.baseline_value / result.best_value
        assert speedup > 1.0
        assert result.best_iter == 1

        # --- History has baseline + one accepted candidate entry ---
        assert len(result.history) == 2
        assert result.history[0]["description"] == "baseline"
        assert result.history[0]["accepted"] is True
        accepted_entry = result.history[1]
        assert accepted_entry["accepted"] is True
        assert accepted_entry["value"] == result.best_value

        # --- Event log recorded the expected phases ---
        types = _event_types(result.run_dir)
        for expected in (
            "baseline_complete", "llm_request", "llm_response",
            "candidate_validation", "candidate_correctness",
            "candidate_benchmark", "candidate_accepted",
            "iteration_complete", "run_complete",
        ):
            assert expected in types, f"missing event {expected!r} in {types}"
        accepted_events = _events_of_type(result.run_dir, "candidate_accepted")
        assert len(accepted_events) == 1
        assert accepted_events[0]["iteration"] == 1
        assert accepted_events[0]["speedup"] > 1.0

        # --- The LLM was actually called twice: 1 iteration + 1 finalize summary ---
        assert len(provider.calls) == 2

        # --- Patch was permanently applied to the real workspace ---
        patched_source = (task.workspace / "sumsq.py").read_text(encoding="utf-8")
        assert "// 6" in patched_source
        assert "for i in range(n)" not in patched_source

        # --- Report artifacts were written ---
        assert (result.run_dir / "report.md").exists()
        assert (result.run_dir / "report.json").exists()
        assert (result.run_dir / "dashboard.html").exists()
        assert (result.run_dir / "optimization_summary.md").exists()

        # --- Protected files (bench/correctness/task.yaml) are untouched ---
        after = _protected_snapshot(task.workspace)
        assert after == before


class TestAgentLoopMisbehavingLLM:
    """Scenario B: a truncated/garbage scripted response must never be
    applied as a patch, must feed an error back to the next iteration's
    prompt, and the run must still finalize cleanly (reports written,
    no crash) instead of raising."""

    def test_garbage_response_rejected_and_run_still_finalizes(
        self, tmp_path: Path,
    ) -> None:
        task, task_file = _write_task(tmp_path)
        before = _protected_snapshot(task.workspace)
        original_source = (task.workspace / "sumsq.py").read_text(encoding="utf-8")

        # Two iterations of garbage: max_consecutive_failures defaults to 5,
        # so early-stop convergence won't cut this short before max_iters.
        provider = ScriptedProvider(responses=[
            _garbage_patch_response(),
            _garbage_patch_response(),
        ])
        progress = ListProgress()

        result = run_agent(
            task, task_file,
            _agent_config(max_iters=2),
            LLMConfig(provider="anthropic", model="test-scripted-model"),
            progress=progress,
            provider=provider,
        )

        # --- No candidate was ever accepted ---
        assert result.best_value == result.baseline_value
        assert result.best_iter == 0
        assert len(result.history) == 3  # baseline + 2 rejected iterations
        for entry in result.history[1:]:
            assert entry["accepted"] is False
            assert entry["description"] == "no candidates parsed"

        # --- The workspace source was never modified ---
        assert (task.workspace / "sumsq.py").read_text(encoding="utf-8") == original_source

        # --- No LLM call for a finalize summary (nothing was accepted) ---
        assert len(provider.calls) == 2

        # --- Iteration errors were recorded and fed back to the next prompt ---
        # Note: state.json (the per-iteration debug snapshot) is NOT written
        # for a "no candidates parsed" iteration -- that branch `continue`s
        # before reaching the state_path.write_text() call at the bottom of
        # the loop (see agent.py's _run_iteration_loop). The event log is
        # the only durable record of iteration_errors/last_errors for a
        # fully-garbled LLM response; assert against that instead.
        error_feedback_events = _events_of_type(result.run_dir, "error_feedback")
        assert len(error_feedback_events) == 1  # only iteration 2 has prior errors to feed back
        assert error_feedback_events[0]["iteration"] == 2
        assert error_feedback_events[0]["error_count"] > 0
        assert any(
            e["type"] == "incomplete_block" for e in error_feedback_events[0]["errors"]
        )

        # --- No candidate_accepted event ever fired ---
        assert _events_of_type(result.run_dir, "candidate_accepted") == []

        # --- Both iterations completed (not accepted), and the run finalized ---
        iter_complete_events = _events_of_type(result.run_dir, "iteration_complete")
        assert [e["iteration"] for e in iter_complete_events] == [1, 2]
        assert all(e["accepted_any"] is False for e in iter_complete_events)
        assert _events_of_type(result.run_dir, "run_complete")

        # --- The run still finalized cleanly: reports exist despite the crashless "no-op" run ---
        assert (result.run_dir / "report.md").exists()
        assert (result.run_dir / "report.json").exists()
        assert (result.run_dir / "dashboard.html").exists()
        # No candidate ever improved on baseline, so no optimization summary
        # LLM call was made and no summary file was written.
        assert not (result.run_dir / "optimization_summary.md").exists()

        # --- Protected files (bench/correctness/task.yaml) are untouched ---
        after = _protected_snapshot(task.workspace)
        assert after == before
