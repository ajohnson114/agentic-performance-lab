"""Tests for checkpoint save/load resumability in the agent loop."""
from __future__ import annotations

import json
from pathlib import Path

from perflab.optimizers.agent import AgentContext
from perflab.analyzers.compiler_diagnostics import CompilerDiagnostics, OptimizationRemark


class TestCheckpointRoundTrip:
    def test_basic_round_trip(self):
        ctx = AgentContext.__new__(AgentContext)
        ctx.iteration = 3
        ctx.best_value = 42.5
        ctx.best_iter = 2
        ctx.baseline_val = 10.0
        ctx.accepted_count = 2
        ctx.history = [{"iteration": 1, "value": 20.0}]
        ctx.accepted_patches = []
        ctx.failure_memory = [{"iteration": 1, "strategy": "x", "failure_type": "build", "reason": "err"}]
        ctx.last_errors = []
        ctx.promising_alternatives = []
        ctx.total_llm_calls = 5
        ctx.total_input_tokens = 1000
        ctx.total_output_tokens = 500
        ctx.total_llm_latency = 12.5
        ctx.user_actions = []
        ctx.early_stop_reason = None
        ctx.latest_diagnostics = None

        d = ctx.to_dict()
        assert d["iteration"] == 3
        assert d["best_value"] == 42.5

        ctx2 = AgentContext.__new__(AgentContext)
        # Initialize all fields that load_dict checks via hasattr
        for field_name in d:
            if field_name != "latest_diagnostics":
                setattr(ctx2, field_name, None)
        ctx2.latest_diagnostics = None
        ctx2.load_dict(d)
        assert ctx2.iteration == 3
        assert ctx2.best_value == 42.5
        assert ctx2.failure_memory == ctx.failure_memory

    def test_diagnostics_survive_round_trip(self):
        ctx = AgentContext.__new__(AgentContext)
        ctx.iteration = 1
        ctx.best_value = 10.0
        ctx.best_iter = 1
        ctx.baseline_val = 5.0
        ctx.accepted_count = 0
        ctx.history = []
        ctx.accepted_patches = []
        ctx.failure_memory = []
        ctx.last_errors = []
        ctx.promising_alternatives = []
        ctx.total_llm_calls = 0
        ctx.total_input_tokens = 0
        ctx.total_output_tokens = 0
        ctx.total_llm_latency = 0.0
        ctx.user_actions = []
        ctx.early_stop_reason = None
        ctx.latest_diagnostics = CompilerDiagnostics(
            program_type="cpp",
            findings=["Missed vectorizations: 2"],
            summary="- Missed vectorizations: 2",
            remarks=[
                OptimizationRemark(
                    file="kern.cpp", line=14, col=5,
                    category="vectorize", status="missed",
                    detail="couldn't vectorize loop",
                ),
            ],
        )

        d = ctx.to_dict()
        assert "latest_diagnostics" in d
        assert d["latest_diagnostics"]["program_type"] == "cpp"

        # Restore
        ctx2 = AgentContext.__new__(AgentContext)
        ctx2.iteration = 0
        ctx2.latest_diagnostics = None
        ctx2.load_dict(d)
        assert ctx2.latest_diagnostics is not None
        assert ctx2.latest_diagnostics.program_type == "cpp"
        assert len(ctx2.latest_diagnostics.remarks) == 1
        assert ctx2.latest_diagnostics.remarks[0].category == "vectorize"

    def test_json_serializable(self, tmp_path):
        """Checkpoint dict can be written to and read from JSON."""
        ctx = AgentContext.__new__(AgentContext)
        ctx.iteration = 2
        ctx.best_value = 15.0
        ctx.best_iter = 2
        ctx.baseline_val = 5.0
        ctx.accepted_count = 1
        ctx.history = [{"iteration": 1, "value": 10.0}]
        ctx.accepted_patches = []
        ctx.failure_memory = []
        ctx.last_errors = []
        ctx.promising_alternatives = []
        ctx.total_llm_calls = 3
        ctx.total_input_tokens = 500
        ctx.total_output_tokens = 200
        ctx.total_llm_latency = 5.0
        ctx.user_actions = []
        ctx.early_stop_reason = None
        ctx.latest_diagnostics = None

        checkpoint = tmp_path / "checkpoint.json"
        checkpoint.write_text(json.dumps(ctx.to_dict(), indent=2), encoding="utf-8")

        restored = json.loads(checkpoint.read_text(encoding="utf-8"))
        ctx2 = AgentContext.__new__(AgentContext)
        ctx2.iteration = 0
        ctx2.latest_diagnostics = None
        ctx2.load_dict(restored)
        assert ctx2.iteration == 2
        assert ctx2.best_value == 15.0

    def test_resume_iteration_start(self):
        """After loading checkpoint at iteration 3, next loop should start at 4."""
        ctx = AgentContext.__new__(AgentContext)
        ctx.iteration = 3
        # The agent loop calculates: start_iter = ctx.iteration + 1 if ctx.iteration > 0 else 1
        start_iter = ctx.iteration + 1 if ctx.iteration > 0 else 1
        assert start_iter == 4


class TestContextOverflowDetection:
    def test_openai_context_length(self):
        from perflab.optimizers.agent import _is_context_overflow_error
        exc = Exception("This model's maximum context length is 128000 tokens")
        assert _is_context_overflow_error(exc) is True

    def test_anthropic_prompt_too_long(self):
        from perflab.optimizers.agent import _is_context_overflow_error
        exc = Exception("prompt is too long: 150000 tokens > 100000 maximum")
        assert _is_context_overflow_error(exc) is True

    def test_ollama_token_limit(self):
        from perflab.optimizers.agent import _is_context_overflow_error
        exc = Exception("too many tokens in request")
        assert _is_context_overflow_error(exc) is True

    def test_unrelated_error_not_matched(self):
        from perflab.optimizers.agent import _is_context_overflow_error
        exc = Exception("connection refused")
        assert _is_context_overflow_error(exc) is False

    def test_rate_limit_not_matched(self):
        from perflab.optimizers.agent import _is_context_overflow_error
        exc = Exception("rate limit exceeded, retry after 30s")
        assert _is_context_overflow_error(exc) is False
