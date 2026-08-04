"""Tests for perflab.llm.pricing -- built-in price table + cost estimation."""
from __future__ import annotations

from perflab.llm.pricing import (
    estimate_cost_usd,
    format_cost_usd,
    is_known_model,
)


class TestEstimateCostUsdKnownModels:
    def test_exact_match_anthropic(self):
        # claude-opus-4-8 is (5.00, 25.00) usd/mtok
        cost = estimate_cost_usd("claude-opus-4-8", 1_000_000, 1_000_000)
        assert cost == 30.00

    def test_exact_match_openai_default(self):
        # gpt-5.6-sol is the PROVIDER_DEFAULT_MODELS openai entry -- must be
        # priced. (5.00, 30.00) usd/mtok -> 2M input tokens = $10.00
        cost = estimate_cost_usd("gpt-5.6-sol", 2_000_000, 0)
        assert cost == 10.00

    def test_zero_tokens_is_zero_cost(self):
        assert estimate_cost_usd("claude-opus-4-8", 0, 0) == 0.0

    def test_fractional_mtok(self):
        # 500k input tokens at $5/mtok input -> $2.50
        cost = estimate_cost_usd("claude-opus-4-8", 500_000, 0)
        assert cost == 2.50

    def test_output_priced_separately_from_input(self):
        in_only = estimate_cost_usd("claude-opus-4-8", 1_000_000, 0)
        out_only = estimate_cost_usd("claude-opus-4-8", 0, 1_000_000)
        assert in_only == 5.00
        assert out_only == 25.00
        assert out_only > in_only


class TestEstimateCostUsdUnknownModel:
    def test_unknown_model_returns_none(self):
        assert estimate_cost_usd("totally-made-up-model-xyz", 1000, 1000) is None

    def test_none_never_fabricated_as_zero(self):
        # An unknown model must be distinguishable from a genuinely free one
        result = estimate_cost_usd("nonexistent-model", 1_000_000, 1_000_000)
        assert result is None
        ollama_result = estimate_cost_usd("llama3.2", 1_000_000, 1_000_000)
        assert ollama_result == 0.0
        assert result != ollama_result


class TestOllamaModelsAreFree:
    def test_llama_is_zero_cost(self):
        assert estimate_cost_usd("llama3.2", 10_000_000, 10_000_000) == 0.0

    def test_mistral_is_zero_cost(self):
        assert estimate_cost_usd("mistral", 5_000_000, 5_000_000) == 0.0

    def test_ollama_default_model_is_known(self):
        # perflab.llm.config.PROVIDER_DEFAULT_MODELS["ollama"] == "llama3.2"
        assert is_known_model("llama3.2")


class TestPrefixMatching:
    def test_dated_suffix_resolves_to_base_model(self):
        base = estimate_cost_usd("claude-opus-4-8", 1_000_000, 1_000_000)
        dated = estimate_cost_usd("claude-opus-4-8-20260701", 1_000_000, 1_000_000)
        assert dated == base

    def test_longest_prefix_wins_over_shorter_one(self):
        # "gpt-5.4-mini" is a more specific (longer) key than "gpt-5.4" -- a
        # dated gpt-5.4-mini snapshot must resolve to the mini pricing, not
        # accidentally match the shorter "gpt-5.4" prefix.
        mini = estimate_cost_usd("gpt-5.4-mini", 1_000_000, 1_000_000)
        dated_mini = estimate_cost_usd("gpt-5.4-mini-2026-08-01", 1_000_000, 1_000_000)
        full = estimate_cost_usd("gpt-5.4", 1_000_000, 1_000_000)
        assert dated_mini == mini
        assert dated_mini != full

    def test_gpt_5_4_mini_not_shadowed_by_gpt_5_4(self):
        # Real prefix collision introduced by the 2026-08-02 pricing update:
        # "gpt-5.4" is itself a priced entry AND a prefix of "gpt-5.4-mini",
        # "gpt-5.4-nano", and "gpt-5.4-pro". Confirm the longer, more
        # specific key wins for each -- not the shorter "gpt-5.4" entry.
        mini = estimate_cost_usd("gpt-5.4-mini", 1_000_000, 1_000_000)
        nano = estimate_cost_usd("gpt-5.4-nano", 1_000_000, 1_000_000)
        pro = estimate_cost_usd("gpt-5.4-pro", 1_000_000, 1_000_000)
        base = estimate_cost_usd("gpt-5.4", 1_000_000, 1_000_000)
        assert mini == 5.25   # (0.75, 4.50)
        assert nano == 1.45   # (0.20, 1.25)
        assert pro == 210.00  # (30.00, 180.00)
        assert base == 17.50  # (2.50, 15.00)
        assert len({mini, nano, pro, base}) == 4  # all four resolve distinctly

    def test_gpt_5_5_pro_not_shadowed_by_gpt_5_5(self):
        # Same collision shape for "gpt-5.5" vs "gpt-5.5-pro".
        pro = estimate_cost_usd("gpt-5.5-pro", 1_000_000, 1_000_000)
        dated_pro = estimate_cost_usd("gpt-5.5-pro-2026-08-01", 1_000_000, 1_000_000)
        base = estimate_cost_usd("gpt-5.5", 1_000_000, 1_000_000)
        assert pro == 210.00  # (30.00, 180.00)
        assert base == 35.00  # (5.00, 30.00)
        assert dated_pro == pro
        assert dated_pro != base

    def test_gpt_5_6_family_shares_stem_but_resolves_distinctly(self):
        # "gpt-5.6-sol" / "gpt-5.6-terra" / "gpt-5.6-luna" all share the
        # "gpt-5.6-" stem but none is a prefix of another -- each must
        # resolve to its own price.
        sol = estimate_cost_usd("gpt-5.6-sol", 1_000_000, 1_000_000)
        terra = estimate_cost_usd("gpt-5.6-terra", 1_000_000, 1_000_000)
        luna = estimate_cost_usd("gpt-5.6-luna", 1_000_000, 1_000_000)
        assert sol == 35.00   # (5.00, 30.00)
        assert terra == 14.00  # (2.00, 12.00)
        assert luna == 1.40   # (0.20, 1.20)
        assert len({sol, terra, luna}) == 3

    def test_shared_prefix_matches_not_none(self):
        # "gpt-5.4" is a genuine prefix of "gpt-5.4-some-variant" -- confirms
        # the match resolves (0 tokens -> $0.00) rather than falling through
        # to None, distinguishing "matched, zero cost" from "unknown model".
        assert estimate_cost_usd("gpt-5.4-some-variant", 0, 0) == 0.0

    def test_no_match_returns_none(self):
        assert estimate_cost_usd("z-completely-unrelated", 1000, 1000) is None

    def test_claude_opus_5_and_opus_4_8_resolve_to_own_entries(self):
        # Both are distinct table entries -- each must resolve on its own
        # (not fall through to None, and not accidentally match the other).
        assert is_known_model("claude-opus-5")
        assert is_known_model("claude-opus-4-8")
        opus5 = estimate_cost_usd("claude-opus-5", 1_000_000, 1_000_000)
        opus48 = estimate_cost_usd("claude-opus-4-8", 1_000_000, 1_000_000)
        assert opus5 == 30.00   # (5.00, 25.00)
        assert opus48 == 30.00  # (5.00, 25.00) -- same price, separate entries

    def test_claude_opus_5_dated_snapshot_resolves_to_base_entry(self):
        base = estimate_cost_usd("claude-opus-5", 1_000_000, 1_000_000)
        dated = estimate_cost_usd("claude-opus-5-20260701", 1_000_000, 1_000_000)
        assert dated == base

    def test_unknown_model_returns_none_never_fabricated(self):
        assert estimate_cost_usd("not-a-real-model-5000", 1_000_000, 1_000_000) is None


class TestConfigOverrides:
    def test_override_adds_new_model(self):
        assert estimate_cost_usd("my-custom-model", 1_000_000, 1_000_000) is None
        cost = estimate_cost_usd(
            "my-custom-model", 1_000_000, 1_000_000,
            overrides={"my-custom-model": (1.5, 6.0)},
        )
        assert cost == 7.5

    def test_override_wins_over_builtin(self):
        # Override a built-in entry entirely -- the override price applies,
        # not the shipped default.
        cost = estimate_cost_usd(
            "claude-opus-4-8", 1_000_000, 1_000_000,
            overrides={"claude-opus-4-8": (1.0, 1.0)},
        )
        assert cost == 2.0

    def test_is_known_model_reflects_overrides(self):
        assert not is_known_model("brand-new-model")
        assert is_known_model("brand-new-model", overrides={"brand-new-model": (1.0, 1.0)})

    def test_empty_overrides_behaves_like_none(self):
        assert estimate_cost_usd("claude-opus-4-8", 100, 100, overrides={}) == \
            estimate_cost_usd("claude-opus-4-8", 100, 100, overrides=None)


class TestIsKnownModel:
    def test_known_builtin(self):
        assert is_known_model("claude-opus-4-8") is True

    def test_unknown(self):
        assert is_known_model("does-not-exist") is False

    def test_known_via_prefix(self):
        assert is_known_model("claude-sonnet-5-20260601") is True


class TestFormatCostUsd:
    def test_known_cost_formatted_as_dollars(self):
        assert format_cost_usd(12.3) == "$12.30"
        assert format_cost_usd(0.0) == "$0.00"
        assert format_cost_usd(1.999) == "$2.00"

    def test_none_shows_unknown_marker(self):
        assert format_cost_usd(None) == "n/a (unknown model pricing)"
