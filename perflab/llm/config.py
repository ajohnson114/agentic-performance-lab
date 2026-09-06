from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from perflab.llm.base import LLMProvider

_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "perflab" / "config.yaml"
_PROJECT_CONFIG_NAME = "perflab.yaml"

# Single source of truth for per-provider default models. Referenced by
# perflab.config, perflab.cli, and the provider defaults so they can't drift.
PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-5.6-sol",
    "anthropic": "claude-opus-5",
    "ollama": "llama3.2",
}
DEFAULT_MODEL = PROVIDER_DEFAULT_MODELS["openai"]

# Top-level keys that mark a YAML file as a general PerfLabConfig-style file
# (see perflab.config.PerfLabConfig) rather than a dedicated flat llm-only
# config -- used by LLMConfig._load_yaml_section to decide whether a missing
# `llm:` key means "flat llm data at top level" or "no llm data here".
_PERFLAB_CONFIG_SECTION_KEYS = frozenset({
    "benchmark", "profiler", "mps", "ollama", "agent", "isolation", "analysis_thresholds",
})


def find_project_config(filename: str = _PROJECT_CONFIG_NAME) -> Path | None:
    """Walk up from cwd looking for a project-level config file.

    Shared by LLMConfig.load() and perflab.config.load_config() -- both
    honor the same documented resolution order (env > project > user >
    defaults), so there is exactly one place that decides what "project
    config" means.
    """
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / filename
        if candidate.exists():
            return candidate
        if parent == Path.home() or parent == parent.parent:
            break
    return None


def _check_config_permissions(path: Path) -> None:
    """Warn if config file containing API keys is world-readable."""
    import platform
    import stat
    import warnings

    if platform.system() == "Windows":
        return  # Windows doesn't use Unix permissions

    try:
        mode = path.stat().st_mode
        if mode & stat.S_IROTH:  # World-readable
            warnings.warn(
                f"Security warning: {path} is world-readable (mode {oct(mode)}). "
                f"This file contains your API key. Fix with: chmod 600 {path}",
                stacklevel=3,
            )
            # Auto-fix: tighten permissions
            try:
                path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
            except OSError:
                pass
    except OSError:
        pass


def _parse_pricing_overrides(data: dict) -> dict[str, tuple[float, float]]:
    """Parse the optional ``pricing:`` mapping from the llm config section.

    Each value is a 2-element ``[input, output]`` list of USD-per-million-
    token prices, e.g. ``pricing: {my-model: [1.5, 6.0]}`` -- merged over the
    built-in table in perflab.llm.pricing (this override wins on conflicts).
    Malformed entries are skipped with a warning rather than aborting config
    load, mirroring perflab.config._safe_set's per-key tolerance.
    """
    raw = data.get("pricing")
    if not isinstance(raw, dict):
        return {}
    overrides: dict[str, tuple[float, float]] = {}
    for model, prices in raw.items():
        try:
            price_in, price_out = prices
            overrides[str(model)] = (float(price_in), float(price_out))
        except (TypeError, ValueError):
            warnings.warn(
                f"Ignoring invalid pricing override for {model!r} in config "
                f"(expected [input_per_mtok, output_per_mtok]): {prices!r}",
                stacklevel=2,
            )
    return overrides


def _secure_write(path: Path, content: str) -> None:
    """Write a file with secure permissions (owner read/write only)."""
    import stat
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except OSError:
        pass


def scrub_api_key(path: Path | None = None) -> bool:
    """Remove a legacy on-disk ``api_key`` from the LLM config file, if present.

    API keys are only ever read from PERFLAB_API_KEY -- a file-based
    ``api_key`` is ignored (see load()). This rewrites the file to drop it,
    so the on-disk copy is no longer left behind. Returns True if a key was
    found and removed, False otherwise.
    """
    config_path = path or _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return False

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return False

    section = raw.get("llm", raw)
    if not isinstance(section, dict) or "api_key" not in section:
        return False

    del section["api_key"]
    _secure_write(config_path, yaml.dump(raw, default_flow_style=False))
    return True


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = DEFAULT_MODEL
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.7
    # 64000, not a lower "safe for non-streaming" figure: every provider's
    # complete() now streams internally (see anthropic_provider.py), and
    # real-hardware runs showed a full reasoning-plus-patch response for a
    # CUDA/tensor-core task routinely needs 20-30k+ output tokens on Claude
    # Opus 5, where thinking (on by default, no separate budget_tokens knob
    # anymore) shares this same ceiling with the response text. A lower
    # default here silently discards the tail of the model's patch and wastes
    # the whole turn's cost on a candidate that never gets parsed.
    max_tokens: int = 64000
    # Optional per-model USD-per-million-token overrides, merged over the
    # built-in table in perflab.llm.pricing (see estimate_cost_usd). Loaded
    # from an optional `pricing:` mapping in the llm: config section.
    pricing: dict[str, tuple[float, float]] = field(default_factory=dict)

    @staticmethod
    def _load_yaml_section(config_path: Path) -> dict:
        if not config_path.exists():
            return {}
        _check_config_permissions(config_path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        if "llm" in raw:
            section = raw["llm"]
            return section if isinstance(section, dict) else {}
        if _PERFLAB_CONFIG_SECTION_KEYS & raw.keys():
            # A general PerfLabConfig-style file (a project perflab.yaml, or
            # a user config with other sections) that simply has no llm:
            # key -- not the dedicated flat llm-only file the fallback below
            # supports. Falling through to "flat" here would leak unrelated
            # keys like "benchmark" into llm resolution (data.get("model"),
            # data.get("provider"), etc. would just miss, but data.get(...)
            # for a coincidentally-matching key -- e.g. a section literally
            # named "model" -- would not).
            return {}
        # Flat style: a dedicated llm-only config file with no `llm:` nesting.
        return raw

    @staticmethod
    def load(path: Path | None = None) -> LLMConfig:
        """Load config, layering user config < project config < env vars.

        Resolution order (later wins), matching the order documented for
        perflab.config.load_config(): dataclass defaults, then
        ~/.config/perflab/config.yaml, then a discovered ./perflab.yaml
        (walking up from cwd), then environment variables. Passing an
        explicit ``path`` overrides the user-config step only -- project
        config discovery and env overrides still apply on top of it, same
        as the default case.

        The API key is never read from either config file -- only
        PERFLAB_API_KEY is honored, so a key never has to live on disk. A
        legacy file with an ``api_key`` field still loads (the value is
        ignored) but emits a deprecation warning; run
        ``perflab init --scrub-key`` to remove it from the file.

        If PERFLAB_API_KEY is unset, the provider's conventional env var is
        used instead -- OPENAI_API_KEY for the openai provider,
        ANTHROPIC_API_KEY for anthropic -- so a key already exported for
        direct SDK use is picked up automatically. PERFLAB_API_KEY always
        takes precedence when set.
        """
        user_config_path = path or _DEFAULT_CONFIG_PATH
        user_data = LLMConfig._load_yaml_section(user_config_path)

        project_data: dict = {}
        project_config_path = find_project_config()
        if project_config_path is not None:
            project_data = LLMConfig._load_yaml_section(project_config_path)

        # Project config layers over user config key-by-key, not as a whole
        # dict replacement -- a project file that only sets `model:` must
        # not blow away a user-level `api_base:` for a self-hosted proxy.
        data: dict = {**user_data, **project_data}

        for source_path, source_data in ((user_config_path, user_data), (project_config_path, project_data)):
            if source_path is not None and source_data.get("api_key"):
                warnings.warn(
                    f"{source_path} contains an 'api_key' field, which is no longer "
                    "read from disk. Set the PERFLAB_API_KEY environment variable "
                    "instead, then run 'perflab init --scrub-key' to remove it from "
                    "the file.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        defaults = LLMConfig()

        # Provider must be resolved before model: an env override of just
        # PERFLAB_LLM_PROVIDER (leaving PERFLAB_LLM_MODEL unset) must not
        # silently keep the OLD provider's default/configured model name --
        # that model id is very likely invalid for the new provider's API.
        # Model is only carried over from the file/default when nothing --
        # neither env nor file -- names one explicitly for the resolved
        # provider; otherwise it's re-derived from PROVIDER_DEFAULT_MODELS.
        provider = str(data.get("provider", defaults.provider))
        if env_provider := os.environ.get("PERFLAB_LLM_PROVIDER"):
            provider = env_provider

        if env_model := os.environ.get("PERFLAB_LLM_MODEL"):
            model = env_model
        elif "model" in data:
            model = str(data["model"])
        else:
            model = PROVIDER_DEFAULT_MODELS.get(provider.lower(), defaults.model)

        cfg = LLMConfig(
            provider=provider,
            model=model,
            api_key="",
            api_base=str(data.get("api_base", defaults.api_base)),
            temperature=float(data.get("temperature", defaults.temperature)),
            max_tokens=int(data.get("max_tokens", defaults.max_tokens)),
            pricing=_parse_pricing_overrides(data),
        )

        if env_key := os.environ.get("PERFLAB_API_KEY"):
            cfg.api_key = env_key
        elif not cfg.api_key:
            # PERFLAB_API_KEY is unset/empty and no api_key came from either
            # config file (file-sourced keys are never honored -- see
            # above). Fall back to the provider's own conventional env var
            # so a user who already has OPENAI_API_KEY / ANTHROPIC_API_KEY
            # exported doesn't hit an opaque "provider configured but not
            # available" with no clue why. PERFLAB_API_KEY, when set, always
            # wins -- this branch only runs when it's absent.
            provider_lower = cfg.provider.lower()
            if provider_lower == "openai":
                if openai_key := os.environ.get("OPENAI_API_KEY"):
                    cfg.api_key = openai_key
            elif provider_lower == "anthropic":
                if anthropic_key := os.environ.get("ANTHROPIC_API_KEY"):
                    cfg.api_key = anthropic_key
        if env_base := os.environ.get("PERFLAB_API_BASE"):
            cfg.api_base = env_base

        return cfg

    def is_configured(self) -> bool:
        """Check whether the config has enough info to create a working provider."""
        if self.provider.lower() == "ollama":
            return bool(self.model)
        return bool(self.api_key and self.model)


def create_provider(config: LLMConfig) -> LLMProvider:
    """Factory with lazy imports to avoid pulling in optional deps at import time."""
    name = config.provider.lower()

    if name == "openai":
        from perflab.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model=config.model,
            api_key=config.api_key,
            api_base=config.api_base or None,
        )
    elif name == "anthropic":
        from perflab.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            model=config.model,
            api_key=config.api_key,
        )
    elif name == "ollama":
        from perflab.llm.ollama_provider import OllamaProvider
        return OllamaProvider(
            model=config.model,
            api_base=config.api_base or "http://localhost:11434",
        )
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider!r}")
