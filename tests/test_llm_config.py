"""Tests for perflab.llm.config -- API key is env-var only (Fix 3, Option A).

Covers: a config file's api_key field is never loaded, a deprecation warning
fires when one is present, PERFLAB_API_KEY still populates api_key, and
scrub_api_key() removes a legacy on-disk key.
"""
from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest
import yaml

from perflab.llm.config import PROVIDER_DEFAULT_MODELS, LLMConfig, scrub_api_key


@pytest.fixture(autouse=True)
def _no_project_config(monkeypatch):
    """Isolate LLMConfig.load() from whatever cwd pytest happens to run in.

    load() now also discovers a project-level ./perflab.yaml (walking up
    from cwd) -- without this, a stray perflab.yaml in a parent of the test
    runner's cwd could pollute every test in this file. Tests that
    specifically exercise project-config layering override this locally.
    """
    monkeypatch.setattr("perflab.llm.config.find_project_config", lambda *a, **k: None)


class TestApiKeyNeverLoadedFromFile:
    def test_file_api_key_is_ignored(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.dump({"llm": {"provider": "openai", "model": "gpt-5.2", "api_key": "sk-test"}})
        )

        with patch.dict("os.environ", {}, clear=True):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                cfg = LLMConfig.load(config_path)

        assert cfg.api_key == ""
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-5.2"

    def test_file_api_key_emits_deprecation_warning(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"api_key": "sk-test"}}))

        with patch.dict("os.environ", {}, clear=True):
            with pytest.warns(DeprecationWarning, match="perflab init --scrub-key"):
                LLMConfig.load(config_path)

    def test_no_warning_when_no_api_key_in_file(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "openai", "model": "gpt-5.2"}}))

        with patch.dict("os.environ", {}, clear=True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                LLMConfig.load(config_path)

        assert not any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_env_var_still_populates_api_key(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "openai"}}))

        with patch.dict("os.environ", {"PERFLAB_API_KEY": "sk-from-env"}, clear=True):
            cfg = LLMConfig.load(config_path)

        assert cfg.api_key == "sk-from-env"

    def test_env_var_overrides_ignored_file_key(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"api_key": "sk-file"}}))

        with patch.dict("os.environ", {"PERFLAB_API_KEY": "sk-from-env"}, clear=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cfg = LLMConfig.load(config_path)

        assert cfg.api_key == "sk-from-env"


class TestApiKeyProviderEnvFallback:
    """PERFLAB_API_KEY wins when set; otherwise fall back to the provider's
    own conventional env var (OPENAI_API_KEY / ANTHROPIC_API_KEY) so a key
    already exported for direct SDK use is picked up automatically."""

    def test_perflab_api_key_wins_over_openai_api_key(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "openai"}}))

        with patch.dict(
            "os.environ",
            {"PERFLAB_API_KEY": "sk-perflab", "OPENAI_API_KEY": "sk-openai"},
            clear=True,
        ):
            cfg = LLMConfig.load(config_path)

        assert cfg.api_key == "sk-perflab"

    def test_perflab_api_key_wins_over_anthropic_api_key(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "anthropic"}}))

        with patch.dict(
            "os.environ",
            {"PERFLAB_API_KEY": "sk-perflab", "ANTHROPIC_API_KEY": "sk-ant-key"},
            clear=True,
        ):
            cfg = LLMConfig.load(config_path)

        assert cfg.api_key == "sk-perflab"

    def test_openai_api_key_used_when_perflab_api_key_absent(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "openai"}}))

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-openai"}, clear=True):
            cfg = LLMConfig.load(config_path)

        assert cfg.api_key == "sk-openai"

    def test_anthropic_api_key_used_when_perflab_api_key_absent(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "anthropic"}}))

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-key"}, clear=True):
            cfg = LLMConfig.load(config_path)

        assert cfg.api_key == "sk-ant-key"

    def test_openai_provider_does_not_pick_up_anthropic_api_key(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "openai"}}))

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-key"}, clear=True):
            cfg = LLMConfig.load(config_path)

        assert cfg.api_key == ""

    def test_anthropic_provider_does_not_pick_up_openai_api_key(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "anthropic"}}))

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-openai"}, clear=True):
            cfg = LLMConfig.load(config_path)

        assert cfg.api_key == ""

    def test_provider_env_override_resolved_before_fallback(self, tmp_path):
        # Config file says openai, but PERFLAB_LLM_PROVIDER overrides to
        # anthropic -- the fallback must read ANTHROPIC_API_KEY, not
        # OPENAI_API_KEY, proving provider resolution happens first.
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "openai"}}))

        with patch.dict(
            "os.environ",
            {
                "PERFLAB_LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "sk-ant-key",
                "OPENAI_API_KEY": "sk-openai",
            },
            clear=True,
        ):
            cfg = LLMConfig.load(config_path)

        assert cfg.provider == "anthropic"
        assert cfg.api_key == "sk-ant-key"

    def test_no_fallback_for_ollama(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "ollama"}}))

        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "sk-openai", "ANTHROPIC_API_KEY": "sk-ant-key"},
            clear=True,
        ):
            cfg = LLMConfig.load(config_path)

        assert cfg.api_key == ""


class TestPricingOverrides:
    def test_pricing_override_parsed_from_llm_section(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({
            "llm": {
                "provider": "openai",
                "model": "gpt-5.2",
                "pricing": {"my-custom-model": [1.5, 6.0]},
            },
        }))

        with patch.dict("os.environ", {}, clear=True):
            cfg = LLMConfig.load(config_path)

        assert cfg.pricing == {"my-custom-model": (1.5, 6.0)}

    def test_pricing_override_flat_config_style(self, tmp_path):
        # LLMConfig.load() supports a flat (non-nested-under-llm) config file
        # too -- pricing should parse the same way in that shape.
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({
            "provider": "openai",
            "pricing": {"another-model": [2.0, 8.0]},
        }))

        with patch.dict("os.environ", {}, clear=True):
            cfg = LLMConfig.load(config_path)

        assert cfg.pricing == {"another-model": (2.0, 8.0)}

    def test_no_pricing_section_yields_empty_dict(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "openai"}}))

        with patch.dict("os.environ", {}, clear=True):
            cfg = LLMConfig.load(config_path)

        assert cfg.pricing == {}

    def test_malformed_pricing_entry_skipped_with_warning(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({
            "llm": {
                "pricing": {
                    "good-model": [1.0, 2.0],
                    "bad-model": "not-a-pair",
                },
            },
        }))

        with patch.dict("os.environ", {}, clear=True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cfg = LLMConfig.load(config_path)

        assert cfg.pricing == {"good-model": (1.0, 2.0)}
        assert any("bad-model" in str(w.message) for w in caught)

    def test_pricing_override_used_by_estimate_cost_usd(self, tmp_path):
        from perflab.llm.pricing import estimate_cost_usd

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({
            "llm": {"pricing": {"my-model": [1.0, 4.0]}},
        }))

        with patch.dict("os.environ", {}, clear=True):
            cfg = LLMConfig.load(config_path)

        cost = estimate_cost_usd("my-model", 1_000_000, 1_000_000, overrides=cfg.pricing)
        assert cost == 5.0


class TestProviderModelResolution:
    """Regression coverage: overriding just PERFLAB_LLM_PROVIDER via env
    (leaving PERFLAB_LLM_MODEL unset) must re-derive the model default from
    the NEW provider, not silently keep the OLD provider's default/configured
    model name -- which the new provider's API would very likely reject."""

    def test_env_provider_override_rederives_default_model(self, tmp_path):
        # No file at all -- provider comes purely from env, so the model
        # must resolve to anthropic's default, not openai's.
        config_path = tmp_path / "missing.yaml"
        with patch.dict(
            "os.environ", {"PERFLAB_LLM_PROVIDER": "anthropic"}, clear=True
        ):
            cfg = LLMConfig.load(config_path)
        assert cfg.provider == "anthropic"
        assert cfg.model == PROVIDER_DEFAULT_MODELS["anthropic"]

    def test_env_provider_override_with_file_default_model_rederives(self, tmp_path):
        # File configures openai with no explicit model (so the file's
        # "model" is really just the dataclass default). Overriding the
        # provider via env must not carry the openai default over.
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "openai"}}))
        with patch.dict(
            "os.environ", {"PERFLAB_LLM_PROVIDER": "anthropic"}, clear=True
        ):
            cfg = LLMConfig.load(config_path)
        assert cfg.provider == "anthropic"
        assert cfg.model == PROVIDER_DEFAULT_MODELS["anthropic"]

    def test_env_model_override_wins_even_with_env_provider_override(self, tmp_path):
        config_path = tmp_path / "missing.yaml"
        with patch.dict(
            "os.environ",
            {"PERFLAB_LLM_PROVIDER": "anthropic", "PERFLAB_LLM_MODEL": "my-custom-model"},
            clear=True,
        ):
            cfg = LLMConfig.load(config_path)
        assert cfg.model == "my-custom-model"

    def test_explicit_file_model_survives_env_provider_override(self, tmp_path):
        # An explicit model in the file for the file's own provider is user
        # intent -- it is not silently discarded just because the provider
        # was overridden via env (only the *default-derived* case is fixed
        # up; an explicit choice is respected either way).
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.dump({"llm": {"provider": "openai", "model": "explicit-model"}})
        )
        with patch.dict(
            "os.environ", {"PERFLAB_LLM_PROVIDER": "anthropic"}, clear=True
        ):
            cfg = LLMConfig.load(config_path)
        assert cfg.provider == "anthropic"
        assert cfg.model == "explicit-model"

    def test_no_env_override_keeps_file_provider_and_default_model(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "anthropic"}}))
        with patch.dict("os.environ", {}, clear=True):
            cfg = LLMConfig.load(config_path)
        assert cfg.provider == "anthropic"
        assert cfg.model == PROVIDER_DEFAULT_MODELS["anthropic"]


class TestProjectConfigLayering:
    """LLMConfig.load() now also discovers and layers a project-level
    ./perflab.yaml on top of the user config, matching the resolution order
    perflab.config.load_config() documents (env > project > user >
    defaults) -- previously it only ever read the single user-level path
    passed in, so a project-level llm: override was silently inert."""

    def test_project_config_overrides_user_config(self, tmp_path, monkeypatch):
        user_path = tmp_path / "user.yaml"
        user_path.write_text(yaml.dump({"llm": {"provider": "openai", "model": "user-model"}}))
        project_path = tmp_path / "perflab.yaml"
        project_path.write_text(yaml.dump({"llm": {"model": "project-model"}}))
        monkeypatch.setattr("perflab.llm.config.find_project_config", lambda: project_path)

        with patch.dict("os.environ", {}, clear=True):
            cfg = LLMConfig.load(user_path)

        assert cfg.provider == "openai"       # from user config, untouched by project
        assert cfg.model == "project-model"   # project overrides user

    def test_project_config_without_llm_section_does_not_leak_other_sections(
        self, tmp_path, monkeypatch
    ):
        # A real project perflab.yaml typically has benchmark:/agent:/etc.
        # sections and no llm: key at all -- that must not be misread as
        # "flat" llm data (see _load_yaml_section / _PERFLAB_CONFIG_SECTION_KEYS).
        user_path = tmp_path / "user.yaml"
        user_path.write_text(yaml.dump({"llm": {"provider": "anthropic", "model": "user-model"}}))
        project_path = tmp_path / "perflab.yaml"
        project_path.write_text(yaml.dump({"benchmark": {"warmup": 10}, "agent": {"max_iters": 5}}))
        monkeypatch.setattr("perflab.llm.config.find_project_config", lambda: project_path)

        with patch.dict("os.environ", {}, clear=True):
            cfg = LLMConfig.load(user_path)

        assert cfg.provider == "anthropic"
        assert cfg.model == "user-model"

    def test_env_still_wins_over_project_config(self, tmp_path, monkeypatch):
        user_path = tmp_path / "user.yaml"
        user_path.write_text(yaml.dump({"llm": {"provider": "openai"}}))
        project_path = tmp_path / "perflab.yaml"
        project_path.write_text(yaml.dump({"llm": {"model": "project-model"}}))
        monkeypatch.setattr("perflab.llm.config.find_project_config", lambda: project_path)

        with patch.dict("os.environ", {"PERFLAB_LLM_MODEL": "env-model"}, clear=True):
            cfg = LLMConfig.load(user_path)

        assert cfg.model == "env-model"


class TestScrubApiKey:
    def test_removes_key_and_rewrites_file(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "openai", "api_key": "sk-test"}}))

        removed = scrub_api_key(config_path)

        assert removed is True
        data = yaml.safe_load(config_path.read_text())
        assert "api_key" not in data["llm"]
        assert data["llm"]["provider"] == "openai"

    def test_returns_false_when_no_key_present(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "openai"}}))

        assert scrub_api_key(config_path) is False

    def test_returns_false_when_file_missing(self, tmp_path):
        assert scrub_api_key(tmp_path / "missing.yaml") is False


class TestInitScrubKeyCLI:
    def test_scrub_key_removes_and_reports(self, tmp_path):
        from typer.testing import CliRunner

        from perflab.cli import app

        config_dir = tmp_path / ".config" / "perflab"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text(yaml.dump({"llm": {"provider": "openai", "api_key": "sk-test"}}))

        runner = CliRunner()
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["init", "--scrub-key"])

        assert result.exit_code == 0
        assert "Removed api_key" in result.output
        data = yaml.safe_load(config_path.read_text())
        assert "api_key" not in data["llm"]

    def test_scrub_key_no_op_when_absent(self, tmp_path):
        from typer.testing import CliRunner

        from perflab.cli import app

        config_dir = tmp_path / ".config" / "perflab"
        config_dir.mkdir(parents=True)
        (config_dir / "config.yaml").write_text(yaml.dump({"llm": {"provider": "openai"}}))

        runner = CliRunner()
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["init", "--scrub-key"])

        assert result.exit_code == 0
        assert "nothing to do" in result.output
