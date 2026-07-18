"""Tests for PerfLabConfig structured configuration system."""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import yaml

from perflab.config import (
    DEFAULT_CONFIG_TEMPLATE,
    PerfLabConfig,
    _overlay_env,
    _overlay_yaml,
    load_config,
)

# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_config_values(self):
        cfg = PerfLabConfig()
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-5.6"
        assert cfg.benchmark.warmup == 3
        assert cfg.benchmark.repeats == 20
        assert cfg.profiler.torch_with_flops is True
        assert cfg.mps.device_index is None
        assert cfg.ollama.allow_remote is False

    def test_to_dict_round_trip(self):
        cfg = PerfLabConfig()
        d = cfg.to_dict()
        assert d["llm"]["provider"] == "openai"
        assert d["benchmark"]["warmup"] == 3
        # Verify JSON serializable
        json.loads(json.dumps(d))

    def test_save_and_read(self, tmp_path):
        cfg = PerfLabConfig()
        cfg.llm.model = "claude-sonnet-4-20250514"
        path = tmp_path / "config.json"
        cfg.save(path)
        restored = json.loads(path.read_text(encoding="utf-8"))
        assert restored["llm"]["model"] == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# YAML overlay
# ---------------------------------------------------------------------------

class TestYamlOverlay:
    def test_overlay_all_sections(self):
        cfg = PerfLabConfig()
        data = {
            "llm": {"provider": "anthropic", "model": "claude-opus-4-20250514"},
            "benchmark": {"warmup": 5, "repeats": 50},
            "profiler": {"peaks_no_cache": True},
            "mps": {"device_match": "M3 Max"},
            "ollama": {"allow_remote": True, "allowed_ports": [8080]},
        }
        _overlay_yaml(cfg, data)
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-opus-4-20250514"
        assert cfg.benchmark.warmup == 5
        assert cfg.profiler.peaks_no_cache is True
        assert cfg.mps.device_match == "M3 Max"
        assert cfg.ollama.allow_remote is True
        assert cfg.ollama.allowed_ports == [8080]

    def test_partial_overlay_preserves_defaults(self):
        cfg = PerfLabConfig()
        _overlay_yaml(cfg, {"llm": {"model": "custom"}})
        assert cfg.llm.model == "custom"
        assert cfg.llm.provider == "openai"  # default preserved
        assert cfg.benchmark.warmup == 3     # untouched section

    def test_empty_yaml(self):
        cfg = PerfLabConfig()
        _overlay_yaml(cfg, {})
        assert cfg.llm.provider == "openai"

    def test_none_yaml(self):
        cfg = PerfLabConfig()
        _overlay_yaml(cfg, None)
        assert cfg.llm.provider == "openai"

    def test_agent_section_overlay(self):
        cfg = PerfLabConfig()
        _overlay_yaml(cfg, {
            "agent": {
                "n_candidates": 12,
                "max_iters": 20,
                "max_wall_time_s": 7200,
                "fast_screen": False,
                "max_history": 5,
                "prompt_token_budget": 8000,
            },
        })
        assert cfg.agent.n_candidates == 12
        assert cfg.agent.max_iters == 20
        assert cfg.agent.max_wall_time_s == 7200
        assert cfg.agent.fast_screen is False
        assert cfg.agent.max_history == 5
        assert cfg.agent.prompt_token_budget == 8000

    def test_agent_partial_overlay(self):
        cfg = PerfLabConfig()
        _overlay_yaml(cfg, {"agent": {"n_candidates": 10}})
        assert cfg.agent.n_candidates == 10
        assert cfg.agent.max_iters == 12  # default preserved


# ---------------------------------------------------------------------------
# Env var overlay
# ---------------------------------------------------------------------------

class TestEnvOverlay:
    def test_env_overrides_yaml(self):
        cfg = PerfLabConfig()
        cfg.llm.provider = "openai"
        with patch.dict(os.environ, {"PERFLAB_LLM_PROVIDER": "anthropic"}):
            _overlay_env(cfg)
        assert cfg.llm.provider == "anthropic"

    def test_env_benchmark_settings(self):
        cfg = PerfLabConfig()
        with patch.dict(os.environ, {
            "PERFLAB_BENCH_WARMUP": "10",
            "PERFLAB_BENCH_REPEATS": "100",
        }):
            _overlay_env(cfg)
        assert cfg.benchmark.warmup == 10
        assert cfg.benchmark.repeats == 100

    def test_env_invalid_int_ignored(self):
        cfg = PerfLabConfig()
        with patch.dict(os.environ, {"PERFLAB_BENCH_WARMUP": "not_a_number"}):
            _overlay_env(cfg)
        assert cfg.benchmark.warmup == 3  # default preserved

    def test_env_peaks_no_cache(self):
        cfg = PerfLabConfig()
        with patch.dict(os.environ, {"PERFLAB_PEAKS_NO_CACHE": "1"}):
            _overlay_env(cfg)
        assert cfg.profiler.peaks_no_cache is True

    def test_env_mps_device(self):
        cfg = PerfLabConfig()
        with patch.dict(os.environ, {
            "PERFLAB_MPS_DEVICE_MATCH": "M3",
            "PERFLAB_MPS_DEVICE_INDEX": "2",
        }):
            _overlay_env(cfg)
        assert cfg.mps.device_match == "M3"
        assert cfg.mps.device_index == 2

    def test_env_ollama(self):
        cfg = PerfLabConfig()
        with patch.dict(os.environ, {
            "PERFLAB_OLLAMA_ALLOW_REMOTE": "1",
            "PERFLAB_OLLAMA_ALLOWED_PORTS": "8080,9090",
        }):
            _overlay_env(cfg)
        assert cfg.ollama.allow_remote is True
        assert cfg.ollama.allowed_ports == [8080, 9090]


# ---------------------------------------------------------------------------
# Full load_config()
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_from_user_config(self, tmp_path, monkeypatch):
        import perflab.config as config_mod
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({
            "llm": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            "benchmark": {"warmup": 7},
        }), encoding="utf-8")
        monkeypatch.setattr(config_mod, "_USER_CONFIG_PATH", config_path)
        monkeypatch.setattr(config_mod, "_cached_config", None)
        # No project config
        monkeypatch.setattr(config_mod, "_find_project_config", lambda: None)

        cfg = load_config(force_reload=True)
        assert cfg.llm.provider == "anthropic"
        assert cfg.benchmark.warmup == 7

    def test_project_config_overrides_user(self, tmp_path, monkeypatch):
        import perflab.config as config_mod
        user_path = tmp_path / "user_config.yaml"
        user_path.write_text(yaml.dump({
            "llm": {"model": "user-model"},
            "benchmark": {"warmup": 5},
        }), encoding="utf-8")
        project_path = tmp_path / "perflab.yaml"
        project_path.write_text(yaml.dump({
            "benchmark": {"warmup": 10},  # project overrides user
        }), encoding="utf-8")
        monkeypatch.setattr(config_mod, "_USER_CONFIG_PATH", user_path)
        monkeypatch.setattr(config_mod, "_cached_config", None)
        monkeypatch.setattr(config_mod, "_find_project_config", lambda: project_path)

        cfg = load_config(force_reload=True)
        assert cfg.llm.model == "user-model"  # from user config
        assert cfg.benchmark.warmup == 10      # project overrides

    def test_env_overrides_all(self, tmp_path, monkeypatch):
        import perflab.config as config_mod
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({
            "llm": {"model": "yaml-model"},
        }), encoding="utf-8")
        monkeypatch.setattr(config_mod, "_USER_CONFIG_PATH", config_path)
        monkeypatch.setattr(config_mod, "_cached_config", None)
        monkeypatch.setattr(config_mod, "_find_project_config", lambda: None)
        monkeypatch.setenv("PERFLAB_LLM_MODEL", "env-model")

        cfg = load_config(force_reload=True)
        assert cfg.llm.model == "env-model"  # env wins

    def test_caching(self, monkeypatch):
        import perflab.config as config_mod
        monkeypatch.setattr(config_mod, "_cached_config", None)
        monkeypatch.setattr(config_mod, "_find_project_config", lambda: None)

        cfg1 = load_config(force_reload=True)
        cfg2 = load_config()
        assert cfg1 is cfg2  # same instance

    def test_force_reload(self, monkeypatch):
        import perflab.config as config_mod
        monkeypatch.setattr(config_mod, "_cached_config", None)
        monkeypatch.setattr(config_mod, "_find_project_config", lambda: None)

        cfg1 = load_config(force_reload=True)
        cfg2 = load_config(force_reload=True)
        assert cfg1 is not cfg2  # different instances


# ---------------------------------------------------------------------------
# Config template
# ---------------------------------------------------------------------------

class TestConfigTemplate:
    def test_template_is_valid_yaml(self):
        """The default config template should parse as valid YAML."""
        data = yaml.safe_load(DEFAULT_CONFIG_TEMPLATE)
        assert isinstance(data, dict)
        assert "llm" in data
        assert "benchmark" in data

    def test_template_has_subprocess_docs(self):
        """Template documents which env vars stay as subprocess-only."""
        assert "PERFLAB_API_KEY" in DEFAULT_CONFIG_TEMPLATE
        assert "PERFLAB_TORCH_PROFILE" in DEFAULT_CONFIG_TEMPLATE
        assert "PERFLAB_DETERMINISM_SEED" in DEFAULT_CONFIG_TEMPLATE
        assert "Subprocess-only" in DEFAULT_CONFIG_TEMPLATE


# ---------------------------------------------------------------------------
# Analysis thresholds in config
# ---------------------------------------------------------------------------

class TestAnalysisThresholdsInConfig:
    def test_thresholds_overlay_from_yaml(self):
        cfg = PerfLabConfig()
        _overlay_yaml(cfg, {
            "analysis_thresholds": {
                "ncu_occupancy_low": 25.0,
                "perf_ipc_low": 0.5,
            },
        })
        assert cfg.analysis_thresholds["ncu_occupancy_low"] == 25.0
        assert cfg.analysis_thresholds["perf_ipc_low"] == 0.5

    def test_thresholds_empty_by_default(self):
        cfg = PerfLabConfig()
        assert cfg.analysis_thresholds == {}

    def test_thresholds_merge_accumulates(self):
        cfg = PerfLabConfig()
        _overlay_yaml(cfg, {"analysis_thresholds": {"ncu_occupancy_low": 25.0}})
        _overlay_yaml(cfg, {"analysis_thresholds": {"perf_ipc_low": 0.5}})
        # Both should be present (project config overlays user config)
        assert cfg.analysis_thresholds["ncu_occupancy_low"] == 25.0
        assert cfg.analysis_thresholds["perf_ipc_low"] == 0.5

    def test_thresholds_in_serialized_output(self):
        cfg = PerfLabConfig()
        cfg.analysis_thresholds = {"ncu_occupancy_low": 25.0}
        d = cfg.to_dict()
        assert d["analysis_thresholds"]["ncu_occupancy_low"] == 25.0

    def test_task_yaml_overrides_config_thresholds(self, tmp_path, monkeypatch):
        """task.yaml analysis_thresholds override global config thresholds."""
        import perflab.config as config_mod
        from perflab.task_spec import TaskSpec

        # Set global config threshold
        mock_cfg = PerfLabConfig()
        mock_cfg.analysis_thresholds = {
            "ncu_occupancy_low": 25.0,  # global: relaxed
            "perf_ipc_low": 0.5,        # global: strict
        }
        monkeypatch.setattr(config_mod, "_cached_config", mock_cfg)

        # task.yaml overrides one threshold
        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text("""\
name: test
workspace: "."
program_type: cuda
correctness:
  cmd: "echo ok"
benchmark:
  cmd: "echo bench"
  metric:
    name: "tflops.median"
analysis_thresholds:
  ncu_occupancy_low: 40.0
""", encoding="utf-8")

        spec = TaskSpec.load(task_yaml)
        # task.yaml wins for ncu_occupancy_low
        assert spec.analysis_thresholds.ncu_occupancy_low == 40.0
        # config wins for perf_ipc_low (not overridden in task.yaml)
        assert spec.analysis_thresholds.perf_ipc_low == 0.5


# ---------------------------------------------------------------------------
# Config saved per run (reproducibility)
# ---------------------------------------------------------------------------

class TestRunReproducibility:
    def test_config_saved_to_run_dir(self, tmp_path):
        cfg = PerfLabConfig()
        cfg.llm.model = "specific-model"
        cfg.benchmark.warmup = 7
        run_config = tmp_path / "resolved_config.json"
        cfg.save(run_config)

        # A different session can load and compare
        saved = json.loads(run_config.read_text(encoding="utf-8"))
        assert saved["llm"]["model"] == "specific-model"
        assert saved["benchmark"]["warmup"] == 7
        # No api_key in serialized output
        assert "api_key" not in saved["llm"]
