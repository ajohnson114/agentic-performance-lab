from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from perflab.llm.base import LLMProvider


_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "perflab" / "config.yaml"


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


def _secure_write(path: Path, content: str) -> None:
    """Write a file with secure permissions (owner read/write only)."""
    import stat
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except OSError:
        pass


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-5.2"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096

    @staticmethod
    def load(path: Path | None = None) -> LLMConfig:
        """Load config from YAML file, then override with env vars."""
        config_path = path or _DEFAULT_CONFIG_PATH
        data: dict = {}

        if config_path.exists():
            # Security: warn if config file is world-readable (contains API keys)
            _check_config_permissions(config_path)
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # Support nested llm: section or flat
                data = raw.get("llm", raw)

        cfg = LLMConfig(
            provider=str(data.get("provider", "openai")),
            model=str(data.get("model", "gpt-4o")),
            api_key=str(data.get("api_key", "")),
            api_base=str(data.get("api_base", "")),
            temperature=float(data.get("temperature", 0.7)),
            max_tokens=int(data.get("max_tokens", 4096)),
        )

        # Env var overrides
        if env_provider := os.environ.get("PERFLAB_LLM_PROVIDER"):
            cfg.provider = env_provider
        if env_model := os.environ.get("PERFLAB_LLM_MODEL"):
            cfg.model = env_model
        if env_key := os.environ.get("PERFLAB_API_KEY"):
            cfg.api_key = env_key
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
