"""Tests for the per-iteration state.json serialization (AgentContext.to_dict)."""
from __future__ import annotations

import json

from perflab.analyzers.compiler_diagnostics import CompilerDiagnostics, OptimizationRemark
from perflab.optimizers.agent import AgentContext


def _bare_ctx(**overrides) -> AgentContext:
    """Build an AgentContext without required constructor args, for serialization tests."""
    ctx = AgentContext.__new__(AgentContext)
    ctx.iteration = 0
    ctx.best_value = 0.0
    ctx.best_iter = 0
    ctx.baseline_val = 0.0
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
    ctx.latest_diagnostics = None
    for key, val in overrides.items():
        setattr(ctx, key, val)
    return ctx


class TestStateSerialization:
    def test_to_dict_fields(self):
        ctx = _bare_ctx(
            iteration=3,
            best_value=42.5,
            best_iter=2,
            baseline_val=10.0,
            accepted_count=2,
            history=[{"iteration": 1, "value": 20.0}],
            failure_memory=[{"iteration": 1, "strategy": "x", "failure_type": "build", "reason": "err"}],
            total_llm_calls=5,
        )
        d = ctx.to_dict()
        assert d["iteration"] == 3
        assert d["best_value"] == 42.5
        assert d["failure_memory"] == ctx.failure_memory
        assert "latest_diagnostics" not in d

    def test_diagnostics_serialized(self):
        ctx = _bare_ctx(
            iteration=1,
            latest_diagnostics=CompilerDiagnostics(
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
            ),
        )
        d = ctx.to_dict()
        assert d["latest_diagnostics"]["program_type"] == "cpp"
        assert len(d["latest_diagnostics"]["remarks"]) == 1

    def test_json_serializable(self, tmp_path):
        """The state dict can be written to and read back from JSON."""
        ctx = _bare_ctx(iteration=2, best_value=15.0, history=[{"iteration": 1, "value": 10.0}])
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps(ctx.to_dict(), indent=2), encoding="utf-8")

        restored = json.loads(state_path.read_text(encoding="utf-8"))
        assert restored["iteration"] == 2
        assert restored["best_value"] == 15.0


class TestContextOverflowDetection:
    def test_openai_context_length(self):
        from perflab.optimizers.phases.generate import _is_context_overflow_error
        exc = Exception("This model's maximum context length is 128000 tokens")
        assert _is_context_overflow_error(exc) is True

    def test_anthropic_prompt_too_long(self):
        from perflab.optimizers.phases.generate import _is_context_overflow_error
        exc = Exception("prompt is too long: 150000 tokens > 100000 maximum")
        assert _is_context_overflow_error(exc) is True

    def test_ollama_token_limit(self):
        from perflab.optimizers.phases.generate import _is_context_overflow_error
        exc = Exception("too many tokens in request")
        assert _is_context_overflow_error(exc) is True

    def test_unrelated_error_not_matched(self):
        from perflab.optimizers.phases.generate import _is_context_overflow_error
        exc = Exception("connection refused")
        assert _is_context_overflow_error(exc) is False

    def test_rate_limit_not_matched(self):
        from perflab.optimizers.phases.generate import _is_context_overflow_error
        exc = Exception("rate limit exceeded, retry after 30s")
        assert _is_context_overflow_error(exc) is False
