"""LLM cost estimation.

Provides a built-in USD-per-million-token price table plus helpers to
estimate the cost of a run from its accumulated token counts. Used by the
optimizer's ``--max-cost`` guard and by the dashboard/report's "est. cost"
line.

IMPORTANT -- ALL PRICES BELOW ARE ESTIMATES, not billing truth. They are a
best-effort snapshot for cost *tracking* and the ``--max-cost`` guardrail,
not a source of record for invoicing. Prices change over time and vary by
region/contract/volume discount. If a number here is wrong for your account,
override it via the ``pricing:`` mapping in the ``llm:`` section of your
perflab config (see perflab/llm/config.py -- ``LLMConfig.pricing``), e.g.:

    llm:
      pricing:
        my-custom-model: [1.5, 6.0]   # [usd_per_mtok_input, usd_per_mtok_output]

Lookup is exact-match first, then longest-matching-prefix -- so a dated
snapshot suffix (e.g. "claude-opus-4-8-20260701") still resolves against the
"claude-opus-4-8" entry.
"""
from __future__ import annotations

from collections.abc import Mapping

# usd_per_mtok_input, usd_per_mtok_output -- USD per 1,000,000 tokens.
# ESTIMATES ONLY (see module docstring). Sourced by transcription from the
# official OpenAI and Anthropic pricing pages on 2026-08-02 -- prices change
# over time and this is a point-in-time snapshot, not a live feed; override
# via the ``pricing:`` config mapping if a number here goes stale.
_BUILTIN_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # --- OpenAI (standard tier, short-context column; sourced 2026-08-02) ---
    # NOTE: a bare "gpt-5.6" and "gpt-5.2" are NOT real OpenAI models as of
    # this snapshot -- the 5.6 family ships only as sol/terra/luna, and 5.2
    # was removed/never existed. Do not add either as a table entry.
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5-pro": (30.00, 180.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4": (2.50, 15.00),

    # --- Anthropic (base input/output tokens; sourced 2026-08-02) ---
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-opus-4-1": (15.00, 75.00),  # deprecated
    "claude-opus-4-0": (15.00, 75.00),  # retired except on Google Cloud
    # Introductory rate, effective through 2026-08-31; rises to (3.00, 15.00)
    # on 2026-09-01 -- update this entry then (or override via `pricing:`
    # in the meantime if a run straddles the cutover).
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),

    # --- Ollama: local inference, no per-token API charge ---
    "llama3.2": (0.0, 0.0),
    "llama3.1": (0.0, 0.0),
    "llama3": (0.0, 0.0),
    "llama2": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "mixtral": (0.0, 0.0),
    "qwen2.5": (0.0, 0.0),
    "qwen2": (0.0, 0.0),
    "codellama": (0.0, 0.0),
    "gemma2": (0.0, 0.0),
    "gemma": (0.0, 0.0),
    "phi3": (0.0, 0.0),
    "deepseek": (0.0, 0.0),
}


def _merged_table(
    overrides: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    table = dict(_BUILTIN_PRICES_USD_PER_MTOK)
    if overrides:
        table.update(overrides)
    return table


def _lookup_price(
    model: str, table: Mapping[str, tuple[float, float]],
) -> tuple[float, float] | None:
    """Exact match first, else the longest key that is a prefix of ``model``.

    The longest-prefix rule is order-independent (unlike a substring/"in"
    check): among all table keys that are a prefix of ``model``, the longest
    one always wins, so a more specific entry (e.g. "gpt-5.4-mini") is never
    shadowed by a shorter, more general one (e.g. "gpt-5.4") regardless of
    dict insertion order.
    """
    if not model:
        return None
    if model in table:
        return table[model]
    best: tuple[float, float] | None = None
    best_len = -1
    for key, price in table.items():
        if key and model.startswith(key) and len(key) > best_len:
            best = price
            best_len = len(key)
    return best


def is_known_model(
    model: str, overrides: Mapping[str, tuple[float, float]] | None = None,
) -> bool:
    """Whether pricing (built-in or override) is known for ``model``."""
    return _lookup_price(model, _merged_table(overrides)) is not None


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    overrides: Mapping[str, tuple[float, float]] | None = None,
) -> float | None:
    """Estimate USD cost for the given token counts on ``model``.

    Returns None when the model's pricing is unknown -- callers must then
    show token counts only, never fabricate a $0 (or any other) cost.
    ``overrides`` (typically ``LLMConfig.pricing``, a user config override)
    takes priority over the built-in table on a per-model basis.
    """
    prices = _lookup_price(model, _merged_table(overrides))
    if prices is None:
        return None
    price_in, price_out = prices
    return (input_tokens / 1_000_000.0) * price_in + (output_tokens / 1_000_000.0) * price_out


def format_cost_usd(cost_usd: float | None) -> str:
    """Format an estimated cost for display, e.g. '$1.23', or the unknown-model marker."""
    if cost_usd is None:
        return "n/a (unknown model pricing)"
    return f"${cost_usd:.2f}"
