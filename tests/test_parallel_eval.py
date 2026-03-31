"""Tests for parallel candidate evaluation and structured failure memory."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Structured failure memory in prompt
# ---------------------------------------------------------------------------

class TestFailureMemoryInPrompt:
    def test_failure_memory_rendered(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"kern.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[{"iteration": 1, "description": "baseline", "value": 1.0, "accepted": True}],
            allowed_paths=["kern.cu"],
            program_type="cuda",
            failure_memory=[
                {
                    "iteration": 2,
                    "strategy": "Added 64x64 shared memory tiling with 4-stage pipeline",
                    "failure_type": "correctness",
                    "reason": "Register spill caused occupancy drop to 12%",
                },
                {
                    "iteration": 3,
                    "strategy": "WMMA with fp16 accumulator",
                    "failure_type": "correctness",
                    "reason": "FP16 accumulation too imprecise for verification tolerance",
                    "profiler_context": "Max error: 0.15, tolerance: 0.01",
                },
            ],
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "Failed approaches" in full_text
        assert "avoid repeating" in full_text.lower()
        assert "64x64 shared memory tiling" in full_text
        assert "Register spill" in full_text
        assert "WMMA with fp16" in full_text
        assert "FP16 accumulation" in full_text
        assert "Max error: 0.15" in full_text

    def test_no_failures_no_section(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"kern.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["kern.cu"],
            program_type="cuda",
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "Failed approaches" not in full_text

    def test_failure_memory_capped_at_10(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        failures = [
            {
                "iteration": i,
                "strategy": f"Strategy {i}",
                "failure_type": "correctness",
                "reason": f"Reason {i}",
            }
            for i in range(20)
        ]

        ctx = PromptContext(
            source_files={"kern.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["kern.cu"],
            program_type="cuda",
            failure_memory=failures,
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        # Should show last 10 failures (iterations 10-19), not first 10
        assert "Strategy 19" in full_text
        assert "Strategy 10" in full_text
        # First failures should be dropped
        assert "Strategy 0" not in full_text


# ---------------------------------------------------------------------------
# Parallel prescreen
# ---------------------------------------------------------------------------

class TestPromisingAlternatives:
    def test_alternatives_rendered_in_prompt(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"kern.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 4.1}},
            roofline=None,
            history=[{"iteration": 1, "description": "WMMA", "value": 4.1, "accepted": True}],
            allowed_paths=["kern.cu"],
            program_type="cuda",
            promising_alternatives=[
                {
                    "description": "candidate 1: shared memory tiling",
                    "reasoning": "Added 64x64 shared memory tiles to reduce global memory traffic",
                    "value": 3.2,
                    "improvement": 2.4,
                },
                {
                    "description": "candidate 4: loop reordering",
                    "reasoning": "Reordered k-loop to innermost for better cache locality",
                    "value": 2.8,
                    "improvement": 2.1,
                },
            ],
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "Promising alternatives" in full_text
        assert "combining" in full_text.lower()
        assert "shared memory tiling" in full_text
        assert "3.2" in full_text
        assert "2.4x" in full_text
        assert "loop reordering" in full_text

    def test_no_alternatives_no_section(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"kern.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["kern.cu"],
            program_type="cuda",
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "Promising alternatives" not in full_text


class TestParallelPrescreen:
    def test_prescreen_with_invalid_patch(self, tmp_path):
        """Prescreening catches validation errors without building."""
        from perflab.optimizers.agent import _prescreen_candidate
        from perflab.optimizers.patch import SearchReplaceBlock
        from perflab.task_spec import TaskSpec, CommandSpec, BenchmarkSpec, MetricSpec, EditPolicy

        # Create a minimal workspace
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "source.py").write_text("x = 1\n")

        task = TaskSpec(
            name="test",
            workspace=ws,
            program_type="python",
            build=None,
            correctness=CommandSpec(cmd="echo ok"),
            benchmark=BenchmarkSpec(cmd="echo bench", metric=MetricSpec(name="v")),
            edit_policy=EditPolicy(allowed_paths=["source.py"]),
        )

        # Block that tries to edit a non-allowed file
        blocks = [SearchReplaceBlock(file_path="tests.py", search="x", replace="y")]
        result = _prescreen_candidate(0, blocks, "test", task, ws, 1)
        assert result["passed"] is False
        assert result["error"]["type"] == "validation"

    def test_prescreen_with_valid_patch(self, tmp_path):
        """Valid patch that passes build+correctness."""
        from perflab.optimizers.agent import _prescreen_candidate
        from perflab.optimizers.patch import SearchReplaceBlock
        from perflab.task_spec import TaskSpec, CommandSpec, BenchmarkSpec, MetricSpec, EditPolicy

        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "source.py").write_text("x = 1\n")

        task = TaskSpec(
            name="test",
            workspace=ws,
            program_type="python",
            build=None,
            correctness=CommandSpec(cmd="echo ok"),
            benchmark=BenchmarkSpec(cmd="echo bench", metric=MetricSpec(name="v")),
            edit_policy=EditPolicy(allowed_paths=["source.py"]),
        )

        blocks = [SearchReplaceBlock(file_path="source.py", search="x = 1", replace="x = 2")]
        result = _prescreen_candidate(0, blocks, "change x", task, ws, 1)
        assert result["passed"] is True
        assert result["error"] is None

        # Original file should be UNCHANGED (prescreen uses temp copy)
        assert (ws / "source.py").read_text() == "x = 1\n"
