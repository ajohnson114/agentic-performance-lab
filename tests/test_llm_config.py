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

from perflab.llm.config import LLMConfig, scrub_api_key


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
