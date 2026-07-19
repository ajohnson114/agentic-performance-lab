"""Tests for the per-iteration state.json serialization (AgentContext.to_dict)."""
from __future__ import annotations

import json
from types import SimpleNamespace

from perflab.analyzers.compiler_diagnostics import CompilerDiagnostics, OptimizationRemark
from perflab.optimizers.agent import AgentConfig, AgentContext


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
    ctx.total_estimated_cost_usd = None
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

    def test_estimated_cost_serialized(self):
        ctx = _bare_ctx(total_estimated_cost_usd=1.23)
        assert ctx.to_dict()["total_estimated_cost_usd"] == 1.23

    def test_estimated_cost_none_serialized_as_null(self, tmp_path):
        # None (unknown model pricing) must round-trip through JSON as null,
        # never coerced into a fabricated 0.0.
        ctx = _bare_ctx(total_estimated_cost_usd=None)
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps(ctx.to_dict()), encoding="utf-8")
        assert json.loads(state_path.read_text())["total_estimated_cost_usd"] is None


def _cost_limit_ctx(tmp_path, *, max_cost_usd, model="claude-opus-4-8", pricing=None,
                     total_input_tokens=0, total_output_tokens=0):
    """Build a minimal real AgentContext for exercising the iteration loop's
    cost-guard and state-snapshot helpers (agent._update_and_check_cost_limit
    / agent._write_state_snapshot / agent._run_iteration_loop)."""
    from perflab.optimizers.event_log import AgentEventLog
    from perflab.optimizers.progress import ListProgress

    return AgentContext(
        task=SimpleNamespace(
            constraints=SimpleNamespace(prompt_token_budget=0),
            benchmark=SimpleNamespace(metric=SimpleNamespace(mode="maximize")),
        ),
        config=AgentConfig(max_wall_time_s=3600, max_iters=1, max_cost_usd=max_cost_usd),
        llm_config=SimpleNamespace(model=model, pricing=pricing or {}),
        provider=None,
        progress=ListProgress(),
        ws=tmp_path,
        rp=None,
        event_log=AgentEventLog(run_dir=tmp_path),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )


class TestUpdateAndCheckCostLimit:
    """agent._update_and_check_cost_limit: the --max-cost stop-wiring helper."""

    def test_no_limit_configured_never_stops(self, tmp_path):
        from perflab.optimizers.agent import _update_and_check_cost_limit

        ctx = _cost_limit_ctx(tmp_path, max_cost_usd=None,
                               total_input_tokens=10_000_000, total_output_tokens=10_000_000)
        assert _update_and_check_cost_limit(ctx, 1) is False
        # Still recomputed for reporting purposes, even though unenforced.
        assert ctx.total_estimated_cost_usd == 300.0  # 10*5 + 10*25 usd

    def test_below_limit_does_not_stop(self, tmp_path):
        from perflab.optimizers.agent import _update_and_check_cost_limit

        ctx = _cost_limit_ctx(tmp_path, max_cost_usd=100.0,
                               total_input_tokens=1_000_000, total_output_tokens=0)
        assert _update_and_check_cost_limit(ctx, 1) is False
        assert ctx.early_stop_reason is None

    def test_limit_reached_stops_and_records_everything(self, tmp_path):
        from perflab.optimizers.agent import _update_and_check_cost_limit

        ctx = _cost_limit_ctx(tmp_path, max_cost_usd=1.0,
                               total_input_tokens=1_000_000, total_output_tokens=0)
        assert _update_and_check_cost_limit(ctx, 3) is True
        assert ctx.total_estimated_cost_usd == 5.0
        assert ctx.early_stop_reason == "cost limit reached (est. $5.00 >= $1.00)"
        assert len(ctx.history) == 1
        assert ctx.history[0]["description"] == "early stop: cost limit reached (est. $5.00 >= $1.00)"

        events_path = tmp_path / "agent_events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        cost_events = [e for e in events if e["event_type"] == "cost_limit_reached"]
        assert len(cost_events) == 1
        assert cost_events[0]["estimated_cost_usd"] == 5.0
        assert cost_events[0]["max_cost_usd"] == 1.0

    def test_unknown_model_pricing_never_stops(self, tmp_path):
        # Defensive: the CLI is supposed to fail closed before a run with
        # unknown pricing + --max-cost ever starts, but this helper must not
        # crash or falsely stop if it's reached anyway.
        from perflab.optimizers.agent import _update_and_check_cost_limit

        ctx = _cost_limit_ctx(tmp_path, max_cost_usd=0.01, model="totally-unknown-model",
                               total_input_tokens=1_000_000, total_output_tokens=1_000_000)
        assert _update_and_check_cost_limit(ctx, 1) is False
        assert ctx.total_estimated_cost_usd is None

    def test_config_pricing_override_is_honored(self, tmp_path):
        from perflab.optimizers.agent import _update_and_check_cost_limit

        ctx = _cost_limit_ctx(
            tmp_path, max_cost_usd=1.0, model="my-custom-model",
            pricing={"my-custom-model": (10.0, 10.0)},
            total_input_tokens=1_000_000, total_output_tokens=0,
        )
        assert _update_and_check_cost_limit(ctx, 1) is True
        assert ctx.total_estimated_cost_usd == 10.0


class TestWriteStateSnapshot:
    def test_writes_valid_json(self, tmp_path):
        from perflab.optimizers.agent import _write_state_snapshot

        ctx = _bare_ctx(iteration=4, best_value=7.0)
        state_path = tmp_path / "state.json"
        _write_state_snapshot(ctx, state_path)

        assert state_path.exists()
        assert json.loads(state_path.read_text())["iteration"] == 4

    def test_write_failure_is_swallowed_not_raised(self, tmp_path, caplog):
        from perflab.optimizers.agent import _write_state_snapshot

        ctx = _bare_ctx()
        # A directory in place of the target file makes write_text raise
        # IsADirectoryError, a subclass of OSError.
        state_path = tmp_path / "state_dir"
        state_path.mkdir()
        _write_state_snapshot(ctx, state_path)  # must not raise


class TestNoCandidatesParsedWritesStateSnapshot:
    """Regression test: the "no candidates parsed" branch used to `continue`
    before reaching the per-iteration state.json snapshot at the bottom of
    the loop, so state.json was never written for exactly the iterations
    where the LLM misbehaved worst (no parseable candidates)."""

    def test_state_json_written_when_no_candidates_parsed(self, tmp_path, monkeypatch):
        from perflab.optimizers import agent as agent_module
        from perflab.optimizers.phases.generate import GenerateResult

        ctx = _cost_limit_ctx(tmp_path, max_cost_usd=None)
        state_path = tmp_path / "state.json"
        protected_dir = tmp_path / "protected"

        # Force the "no candidates parsed" branch: llm_failed=False but
        # candidate_blocks is empty.
        monkeypatch.setattr(agent_module.generate, "run", lambda ctx: GenerateResult())

        agent_module._run_iteration_loop(ctx, state_path, protected_dir, {})

        assert state_path.exists(), "state.json must be written even when no candidates parsed"
        data = json.loads(state_path.read_text(encoding="utf-8"))
        assert data["iteration"] == 1
        assert data["history"][-1]["description"] == "no candidates parsed"


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


class TestCollectPromisingAlternatives:
    def _cands(self, values_accepted):
        from perflab.optimizers.phases.evaluate import BeamCandidate
        return [
            BeamCandidate(
                iteration=1, index=i, blocks=[], description=f"cand {i}",
                reasoning=f"r{i}", value=v, accepted=a,
            )
            for i, (v, a) in enumerate(values_accepted)
        ]

    def test_maximize_keeps_above_baseline_only(self):
        from perflab.optimizers.agent import _collect_promising_alternatives
        cands = self._cands([(12.0, True), (11.0, False), (9.0, False), (None, False)])
        alts = _collect_promising_alternatives(cands, 10.0, "maximize")
        assert [a["value"] for a in alts] == [11.0]
        assert alts[0]["improvement"] == 1.1

    def test_minimize_keeps_below_baseline_only(self):
        # Lower is better: 9.0 improved on baseline 10.0; 12.0 regressed and
        # must not be reported as promising.
        from perflab.optimizers.agent import _collect_promising_alternatives
        cands = self._cands([(8.0, True), (9.0, False), (12.0, False)])
        alts = _collect_promising_alternatives(cands, 10.0, "minimize")
        assert [a["value"] for a in alts] == [9.0]
        assert alts[0]["improvement"] == round(10.0 / 9.0, 2)

    def test_minimize_sorted_best_first_and_capped_at_three(self):
        from perflab.optimizers.agent import _collect_promising_alternatives
        cands = self._cands([
            (5.0, True), (9.5, False), (7.0, False), (8.0, False), (9.0, False),
        ])
        alts = _collect_promising_alternatives(cands, 10.0, "minimize")
        assert [a["value"] for a in alts] == [7.0, 8.0, 9.0]

    def test_accepted_candidate_excluded(self):
        from perflab.optimizers.agent import _collect_promising_alternatives
        cands = self._cands([(12.0, True)])
        assert _collect_promising_alternatives(cands, 10.0, "maximize") == []

    def test_zero_baseline_yields_no_alternatives(self):
        from perflab.optimizers.agent import _collect_promising_alternatives
        cands = self._cands([(12.0, True), (11.0, False)])
        assert _collect_promising_alternatives(cands, 0.0, "maximize") == []
