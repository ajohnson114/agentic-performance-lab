"""Tests for perflab.tools.env_fingerprint and its wiring into RunStore.compare_runs,
perflab.ci (save_baseline / run_ci_check), and the compare / ci-check CLI commands.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from perflab.ci import run_ci_check, save_baseline
from perflab.cli import app
from perflab.memory.run_store import RunStore
from perflab.tools.env_fingerprint import (
    ADVISORY_FIELDS,
    BLOCKING_FIELDS,
    FingerprintComparison,
    blocking_differences,
    compare_fingerprints,
    fingerprint_from_sysinfo,
    format_comparison,
)

runner = CliRunner()

# A realistic full collect_system_info() dict, including fields the
# fingerprint must NOT pick up (platform, cpu_governor, load_average,
# cpu_isa, openmp_version, torch_cuda_version, and the raw nvidia_gpus /
# tpu_devices lists themselves).
FULL_SYSINFO = {
    "platform": "macOS-14.0-arm64-arm-64bit",
    "python_version": "3.12.3",
    "machine": "arm64",
    "system": "Darwin",
    "cpu_model": "Apple M2 Max",
    "cpu_count": 12,
    "cpu_governor": "performance",
    "nvidia_gpus": [
        {
            "name": "NVIDIA H100 80GB HBM3",
            "memory_mib": "81559",
            "driver_version": "535.129.03",
            "persistence_mode": "Enabled",
            "throttle_reasons": "0x0000000000000000",
        },
        {
            "name": "NVIDIA H100 80GB HBM3",
            "memory_mib": "81559",
            "driver_version": "535.129.03",
        },
    ],
    "cuda_version": "release 12.2, V12.2.140",
    "torch_version": "2.5.1",
    "torch_cuda_version": "12.1",
    "jax_version": "0.4.30",
    "tpu_devices": [{"name": "TPU v5e", "id": 0, "platform": "tpu"}],
    "tpu_chip": "TPU v5e",
    "tpu_count": 1,
    "triton_version": "3.0.0",
    "cpp_compiler": "g++ (GCC) 13.2.0",
    "openmp_version": "#define _OPENMP 201811",
    "load_average": {"1min": 1.2, "5min": 1.1, "15min": 0.9},
    "cpu_isa": {"avx2": True, "max_simd_width_bits": 256},
}


# ---------------------------------------------------------------------------
# fingerprint_from_sysinfo
# ---------------------------------------------------------------------------

class TestFingerprintFromSysinfo:
    def test_projects_full_sysinfo(self):
        fp = fingerprint_from_sysinfo(FULL_SYSINFO)
        assert fp == {
            "system": "Darwin",
            "machine": "arm64",
            "python_version": "3.12.3",
            "cpu_model": "Apple M2 Max",
            "cpu_count": 12,
            "tpu_chip": "TPU v5e",
            "tpu_count": 1,
            "cuda_version": "release 12.2, V12.2.140",
            "torch_version": "2.5.1",
            "jax_version": "0.4.30",
            "triton_version": "3.0.0",
            "cpp_compiler": "g++ (GCC) 13.2.0",
            "gpu_count": 2,
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_driver": "535.129.03",
        }

    def test_excludes_non_identity_fields(self):
        fp = fingerprint_from_sysinfo(FULL_SYSINFO)
        for excluded in (
            "platform", "cpu_governor", "load_average", "cpu_isa",
            "openmp_version", "torch_cuda_version", "nvidia_gpus", "tpu_devices",
        ):
            assert excluded not in fp

    def test_absent_keys_omitted_not_none(self):
        fp = fingerprint_from_sysinfo({"system": "Linux"})
        assert fp == {"system": "Linux"}
        assert "machine" not in fp
        assert "gpu_name" not in fp
        assert "cpu_model" not in fp

    def test_empty_sysinfo_yields_empty_fingerprint(self):
        assert fingerprint_from_sysinfo({}) == {}

    def test_no_nvidia_gpus_key_omits_gpu_fields(self):
        fp = fingerprint_from_sysinfo({"system": "Darwin", "machine": "arm64"})
        assert "gpu_name" not in fp
        assert "gpu_driver" not in fp
        assert "gpu_count" not in fp

    def test_empty_nvidia_gpus_list_omits_gpu_fields(self):
        fp = fingerprint_from_sysinfo({"system": "Linux", "nvidia_gpus": []})
        assert "gpu_name" not in fp
        assert "gpu_count" not in fp


# ---------------------------------------------------------------------------
# compare_fingerprints — verdict matrix
# ---------------------------------------------------------------------------

class TestCompareFingerprintsVerdict:
    def test_identical_is_comparable(self):
        a = {"system": "Linux", "machine": "x86_64", "torch_version": "2.5.1"}
        b = dict(a)
        cmp = compare_fingerprints(a, b)
        assert cmp.verdict == "comparable"
        assert cmp.differing == []
        assert set(cmp.matched) == set(a)

    def test_differing_torch_version_only_is_advisory(self):
        a = {"system": "Linux", "machine": "x86_64", "torch_version": "2.5.0"}
        b = {"system": "Linux", "machine": "x86_64", "torch_version": "2.5.1"}
        cmp = compare_fingerprints(a, b)
        assert cmp.verdict == "advisory"
        assert cmp.differing == [("torch_version", "2.5.0", "2.5.1")]

    def test_differing_gpu_name_is_incomparable(self):
        a = {"system": "Linux", "machine": "x86_64", "gpu_name": "H100"}
        b = {"system": "Linux", "machine": "x86_64", "gpu_name": "A100"}
        cmp = compare_fingerprints(a, b)
        assert cmp.verdict == "incomparable"
        assert ("gpu_name", "H100", "A100") in cmp.differing

    def test_blocking_and_advisory_both_differ_is_incomparable(self):
        # Blocking always wins, even alongside an advisory difference.
        a = {"system": "Linux", "gpu_name": "H100", "torch_version": "2.5.0"}
        b = {"system": "Linux", "gpu_name": "A100", "torch_version": "2.5.1"}
        cmp = compare_fingerprints(a, b)
        assert cmp.verdict == "incomparable"

    def test_none_on_either_side_is_unverified(self):
        assert compare_fingerprints(None, {"system": "Linux"}).verdict == "unverified"
        assert compare_fingerprints({"system": "Linux"}, None).verdict == "unverified"
        assert compare_fingerprints(None, None).verdict == "unverified"

    def test_empty_dict_counts_as_no_fingerprint(self):
        # fingerprint_from_sysinfo never has a reason to return one, but an
        # empty dict must behave exactly like None -- "no fingerprint",
        # never a fingerprint with zero fields that happens to match anything.
        assert compare_fingerprints({}, {"system": "Linux"}).verdict == "unverified"


# ---------------------------------------------------------------------------
# compare_fingerprints — one-sided fields land in `unknown`, not `differing`
# ---------------------------------------------------------------------------

class TestCompareFingerprintsUnknown:
    def test_field_present_on_one_side_only_is_unknown(self):
        a = {"system": "Linux", "machine": "x86_64", "gpu_name": "H100"}
        b = {"system": "Linux", "machine": "x86_64"}  # no GPU at all
        cmp = compare_fingerprints(a, b)
        assert "gpu_name" in cmp.unknown
        assert all(f != "gpu_name" for f, _, _ in cmp.differing)

    def test_one_sided_field_does_not_block_verdict(self):
        # A GPU-only-on-one-side situation is "unknown", not a hardware
        # mismatch -- it must not force incomparable on its own.
        a = {"system": "Linux", "machine": "x86_64", "gpu_name": "H100"}
        b = {"system": "Linux", "machine": "x86_64"}
        cmp = compare_fingerprints(a, b)
        assert cmp.verdict == "comparable"


# ---------------------------------------------------------------------------
# FingerprintComparison.to_dict / from_dict
# ---------------------------------------------------------------------------

class TestFingerprintComparisonSerialization:
    def test_to_dict_shape(self):
        cmp = compare_fingerprints(
            {"system": "Linux", "gpu_name": "H100"},
            {"system": "Linux", "gpu_name": "A100"},
        )
        d = cmp.to_dict()
        assert d["verdict"] == "incomparable"
        assert d["differing"] == [["gpu_name", "H100", "A100"]]
        assert isinstance(d["matched"], list)
        assert isinstance(d["unknown"], list)

    def test_round_trip(self):
        cmp = compare_fingerprints(
            {"system": "Linux", "torch_version": "2.5.0"},
            {"system": "Linux", "torch_version": "2.5.1"},
        )
        restored = FingerprintComparison.from_dict(cmp.to_dict())
        assert restored.verdict == cmp.verdict
        assert restored.differing == cmp.differing
        assert restored.matched == cmp.matched

    def test_from_dict_defaults_to_unverified(self):
        assert FingerprintComparison.from_dict({}).verdict == "unverified"


# ---------------------------------------------------------------------------
# blocking_differences / format_comparison
# ---------------------------------------------------------------------------

class TestFormattingHelpers:
    def test_blocking_differences_filters_out_advisory(self):
        cmp = compare_fingerprints(
            {"system": "Linux", "gpu_name": "H100", "torch_version": "2.5.0"},
            {"system": "Linux", "gpu_name": "A100", "torch_version": "2.5.1"},
        )
        blocking = blocking_differences(cmp.differing)
        assert [f for f, _, _ in blocking] == ["gpu_name"]

    def test_blocking_differences_accepts_json_lists(self):
        # The shape that comes back out of a JSON round-trip (lists, not tuples).
        differing = [["gpu_name", "H100", "A100"], ["torch_version", "2.5.0", "2.5.1"]]
        blocking = blocking_differences(differing)
        assert blocking == [("gpu_name", "H100", "A100")]

    def test_format_comparison_unverified(self):
        cmp = FingerprintComparison(verdict="unverified")
        lines = format_comparison(cmp)
        assert len(lines) == 1
        assert "unavailable" in lines[0] or "not known" in lines[0]

    def test_format_comparison_lists_differences(self):
        cmp = compare_fingerprints(
            {"system": "Linux", "torch_version": "2.5.0"},
            {"system": "Linux", "torch_version": "2.5.1"},
        )
        lines = format_comparison(cmp)
        assert lines == ["torch_version: 2.5.0 != 2.5.1"]

    def test_format_comparison_comparable_is_empty(self):
        cmp = compare_fingerprints({"system": "Linux"}, {"system": "Linux"})
        assert format_comparison(cmp) == []


# ---------------------------------------------------------------------------
# Field lists sanity
# ---------------------------------------------------------------------------

class TestFieldLists:
    def test_blocking_and_advisory_are_disjoint(self):
        assert set(BLOCKING_FIELDS).isdisjoint(ADVISORY_FIELDS)


# ---------------------------------------------------------------------------
# RunStore.compare_runs — environment block
# ---------------------------------------------------------------------------

class TestRunStoreCompareRunsEnvironment:
    def test_unverified_when_neither_run_has_system_info(self, tmp_path: Path):
        store = RunStore(tmp_path)
        ra = store.new_run("t")
        rb = store.new_run("t")
        result = store.compare_runs(ra.run_id, rb.run_id)
        assert result["environment"]["verdict"] == "unverified"

    def test_incomparable_on_different_gpu_name(self, tmp_path: Path):
        store = RunStore(tmp_path)
        ra = store.new_run("t")
        rb = store.new_run("t")
        (ra.run_dir / "system_info.json").write_text(json.dumps({
            "system": "Linux", "machine": "x86_64",
            "nvidia_gpus": [{"name": "NVIDIA H100 80GB HBM3", "driver_version": "535.129.03"}],
        }))
        (rb.run_dir / "system_info.json").write_text(json.dumps({
            "system": "Linux", "machine": "x86_64",
            "nvidia_gpus": [{"name": "NVIDIA A100-SXM4-40GB", "driver_version": "535.129.03"}],
        }))
        result = store.compare_runs(ra.run_id, rb.run_id)
        assert result["environment"]["verdict"] == "incomparable"
        assert ["gpu_name", "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-40GB"] in result["environment"]["differing"]

    def test_comparable_when_fingerprints_match(self, tmp_path: Path):
        store = RunStore(tmp_path)
        ra = store.new_run("t")
        rb = store.new_run("t")
        info = {"system": "Linux", "machine": "x86_64", "cpu_model": "Xeon"}
        (ra.run_dir / "system_info.json").write_text(json.dumps(info))
        (rb.run_dir / "system_info.json").write_text(json.dumps(info))
        result = store.compare_runs(ra.run_id, rb.run_id)
        assert result["environment"]["verdict"] == "comparable"

    def test_corrupt_system_info_json_yields_unverified_not_crash(self, tmp_path: Path):
        store = RunStore(tmp_path)
        ra = store.new_run("t")
        rb = store.new_run("t")
        (ra.run_dir / "system_info.json").write_text("{not valid json")
        (rb.run_dir / "system_info.json").write_text(json.dumps({"system": "Linux"}))
        result = store.compare_runs(ra.run_id, rb.run_id)
        assert result["environment"]["verdict"] == "unverified"


# ---------------------------------------------------------------------------
# `perflab compare` CLI — environment gate
# ---------------------------------------------------------------------------

def _make_run(store: RunStore, task: str = "t", best_value: float = 10.0, sysinfo: dict | None = None):
    rp = store.new_run(task)
    (rp.run_dir / "report.json").write_text(json.dumps({
        "best_value": best_value, "metric_name": "gflops", "metric_mode": "maximize",
    }))
    if sysinfo is not None:
        (rp.run_dir / "system_info.json").write_text(json.dumps(sysinfo))
    return rp


class TestCompareCLIEnvironmentGate:
    def test_incomparable_refuses_without_force(self, tmp_path: Path):
        out_dir = tmp_path / "out"
        store = RunStore(out_dir)
        ra = _make_run(store, best_value=10.0, sysinfo={
            "system": "Linux", "machine": "x86_64",
            "nvidia_gpus": [{"name": "NVIDIA H100 80GB HBM3", "driver_version": "535.0"}],
        })
        rb = _make_run(store, best_value=20.0, sysinfo={
            "system": "Linux", "machine": "x86_64",
            "nvidia_gpus": [{"name": "NVIDIA A100-SXM4-40GB", "driver_version": "535.0"}],
        })
        result = runner.invoke(app, ["compare", ra.run_id, rb.run_id, "--out-dir", str(out_dir)])
        assert result.exit_code == 1
        assert "Refusing to compare" in result.output
        assert "Ratio" not in result.output
        assert "gpu_name" in result.output

    def test_incomparable_force_shows_exploratory_and_numbers(self, tmp_path: Path):
        out_dir = tmp_path / "out"
        store = RunStore(out_dir)
        ra = _make_run(store, best_value=10.0, sysinfo={
            "system": "Linux", "machine": "x86_64",
            "nvidia_gpus": [{"name": "NVIDIA H100 80GB HBM3", "driver_version": "535.0"}],
        })
        rb = _make_run(store, best_value=20.0, sysinfo={
            "system": "Linux", "machine": "x86_64",
            "nvidia_gpus": [{"name": "NVIDIA A100-SXM4-40GB", "driver_version": "535.0"}],
        })
        result = runner.invoke(
            app, ["compare", ra.run_id, rb.run_id, "--out-dir", str(out_dir), "--force"]
        )
        assert result.exit_code == 0
        assert "EXPLORATORY COMPARISON" in result.output
        assert "NOT VALID" in result.output
        assert "Ratio" in result.output

    def test_force_never_changes_the_stored_verdict(self, tmp_path: Path):
        # --force only changes what is displayed. RunStore.compare_runs (the
        # source of truth) must report "incomparable" either way.
        out_dir = tmp_path / "out"
        store = RunStore(out_dir)
        ra = _make_run(store, sysinfo={"system": "Linux", "gpu_name_marker": "irrelevant",
                                        "nvidia_gpus": [{"name": "H100", "driver_version": "1"}]})
        rb = _make_run(store, sysinfo={"system": "Linux",
                                        "nvidia_gpus": [{"name": "A100", "driver_version": "1"}]})
        without_force = store.compare_runs(ra.run_id, rb.run_id)
        with_force_still = store.compare_runs(ra.run_id, rb.run_id)
        assert without_force["environment"]["verdict"] == "incomparable"
        assert with_force_still["environment"]["verdict"] == "incomparable"

    def test_comparable_runs_print_normally(self, tmp_path: Path):
        out_dir = tmp_path / "out"
        store = RunStore(out_dir)
        info = {"system": "Linux", "machine": "x86_64"}
        ra = _make_run(store, best_value=10.0, sysinfo=info)
        rb = _make_run(store, best_value=20.0, sysinfo=dict(info))
        result = runner.invoke(app, ["compare", ra.run_id, rb.run_id, "--out-dir", str(out_dir)])
        assert result.exit_code == 0
        assert "Ratio" in result.output

    def test_unverified_prints_note_and_still_compares(self, tmp_path: Path):
        out_dir = tmp_path / "out"
        store = RunStore(out_dir)
        ra = _make_run(store, best_value=10.0, sysinfo=None)
        rb = _make_run(store, best_value=20.0, sysinfo=None)
        result = runner.invoke(app, ["compare", ra.run_id, rb.run_id, "--out-dir", str(out_dir)])
        assert result.exit_code == 0
        assert "could not be verified" in result.output
        assert "Ratio" in result.output


# ---------------------------------------------------------------------------
# perflab.ci — save_baseline / run_ci_check environment wiring
# ---------------------------------------------------------------------------

def _make_task(tmp_path, metric_name="tflops.median", metric_mode="maximize", regression_tolerance=0.05):
    """Minimal mock TaskSpec, mirroring tests/test_ci.py's helper."""
    task = MagicMock()
    task.workspace = tmp_path
    task.name = "test_task"
    task.out_dir = tmp_path / "out"
    task.benchmark.metric.name = metric_name
    task.benchmark.metric.mode = metric_mode
    task.benchmark.secondary_metric = None
    task.constraints.regression_tolerance = regression_tolerance
    return task


def _write_baseline(path: Path, value: float, environment: dict | None = None,
                    metric_name="tflops.median", metric_mode="maximize"):
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"metric_name": metric_name, "metric_mode": metric_mode, "value": value}
    if environment is not None:
        data["environment"] = environment
    path.write_text(json.dumps(data), encoding="utf-8")


class TestSaveBaselineEnvironment:
    @patch("perflab.ci._find_latest_ncu_summary", return_value=None)
    @patch("perflab.tools.sysinfo.collect_system_info",
           return_value={"system": "Linux", "machine": "x86_64", "cpu_model": "Xeon", "irrelevant": "x"})
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 2.5}})
    def test_records_projected_environment(self, mock_bench, mock_sysinfo, mock_ncu, tmp_path):
        task = _make_task(tmp_path)
        result_path = save_baseline(task)
        data = json.loads(result_path.read_text())
        assert data["environment"] == {"system": "Linux", "machine": "x86_64", "cpu_model": "Xeon"}

    @patch("perflab.ci._find_latest_ncu_summary", return_value=None)
    @patch("perflab.tools.sysinfo.collect_system_info", side_effect=RuntimeError("probe blew up"))
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 2.5}})
    def test_probe_failure_does_not_abort_save(self, mock_bench, mock_sysinfo, mock_ncu, tmp_path):
        task = _make_task(tmp_path)
        result_path = save_baseline(task)
        data = json.loads(result_path.read_text())
        assert data["value"] == 2.5
        assert "environment" not in data


class TestRunCICheckEnvironmentGate:
    @patch("perflab.tools.sysinfo.collect_system_info",
           return_value={"system": "Linux", "machine": "x86_64",
                          "nvidia_gpus": [{"name": "A100", "driver_version": "1"}]})
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_gpu_mismatch_fails_closed_with_reason(self, mock_bench, mock_sysinfo, tmp_path):
        task = _make_task(tmp_path)
        _write_baseline(
            tmp_path / "baseline.json", 1.0,
            environment={"system": "Linux", "machine": "x86_64", "gpu_name": "H100", "gpu_count": 1},
        )
        result = run_ci_check(task)
        assert result.passed is False
        assert result.environment is not None
        assert result.environment["verdict"] == "incomparable"
        assert result.environment_warnings
        reason = result.environment_warnings[0]
        assert "different hardware" in reason
        assert "gpu_name" in reason
        assert "--save-baseline" in reason

    @patch("perflab.tools.sysinfo.collect_system_info",
           return_value={"system": "Linux", "machine": "x86_64",
                          "nvidia_gpus": [{"name": "A100", "driver_version": "1"}]})
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_force_downgrades_mismatch_and_runs_check(self, mock_bench, mock_sysinfo, tmp_path):
        task = _make_task(tmp_path)
        _write_baseline(
            tmp_path / "baseline.json", 1.0,
            environment={"system": "Linux", "machine": "x86_64", "gpu_name": "H100", "gpu_count": 1},
        )
        result = run_ci_check(task, force=True)
        assert result.environment is not None
        assert result.environment["verdict"] == "incomparable"
        # Value is unchanged vs. baseline -- the (now-run) regression check passes.
        assert result.passed is True
        assert any("--force" in w for w in result.environment_warnings)

    @patch("perflab.tools.sysinfo.collect_system_info",
           return_value={"system": "Linux", "machine": "x86_64"})
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_baseline_without_environment_is_unverified_not_crash(self, mock_bench, mock_sysinfo, tmp_path):
        """A baseline saved before this feature existed (no "environment" key)
        must not crash run_ci_check, and must not be silently treated as
        matching -- it goes through the unverified path and says so."""
        task = _make_task(tmp_path)
        _write_baseline(tmp_path / "baseline.json", 1.0)  # no environment key at all
        result = run_ci_check(task)
        assert result.environment is not None
        assert result.environment["verdict"] == "unverified"
        assert result.environment_warnings
        assert any("could not be verified" in w for w in result.environment_warnings)
        # Unverified never blocks -- the regression check ran and passed normally.
        assert result.passed is True

    @patch("perflab.tools.sysinfo.collect_system_info",
           return_value={"system": "Linux", "machine": "x86_64", "torch_version": "2.6.0"})
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_advisory_mismatch_does_not_fail(self, mock_bench, mock_sysinfo, tmp_path):
        task = _make_task(tmp_path)
        _write_baseline(
            tmp_path / "baseline.json", 1.0,
            environment={"system": "Linux", "machine": "x86_64", "torch_version": "2.5.0"},
        )
        result = run_ci_check(task)
        assert result.environment["verdict"] == "advisory"
        assert result.passed is True
        assert any("torch_version" in w for w in result.environment_warnings)

    @patch("perflab.tools.sysinfo.collect_system_info",
           return_value={"system": "Linux", "machine": "x86_64"})
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_matching_environment_is_comparable_and_silent(self, mock_bench, mock_sysinfo, tmp_path):
        task = _make_task(tmp_path)
        _write_baseline(
            tmp_path / "baseline.json", 1.0,
            environment={"system": "Linux", "machine": "x86_64"},
        )
        result = run_ci_check(task)
        assert result.environment["verdict"] == "comparable"
        assert result.environment_warnings == []
        assert result.passed is True

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_no_baseline_leaves_environment_none(self, mock_bench, tmp_path):
        task = _make_task(tmp_path)
        result = run_ci_check(task)
        assert result.environment is None
        assert result.environment_warnings == []


class TestCICheckResultToDictEnvironment:
    @patch("perflab.tools.sysinfo.collect_system_info",
           return_value={"system": "Linux", "machine": "x86_64",
                          "nvidia_gpus": [{"name": "A100", "driver_version": "1"}]})
    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_environment_in_to_dict(self, mock_bench, mock_sysinfo, tmp_path):
        task = _make_task(tmp_path)
        _write_baseline(
            tmp_path / "baseline.json", 1.0,
            environment={"system": "Linux", "machine": "x86_64", "gpu_name": "H100"},
        )
        result = run_ci_check(task)
        d = result.to_dict()
        assert "environment" in d
        assert d["environment"]["verdict"] == "incomparable"
        assert "environment_warnings" in d

    @patch("perflab.ci._run_bench_full", return_value={"tflops": {"median": 1.0}})
    def test_to_dict_omits_environment_when_none(self, mock_bench, tmp_path):
        task = _make_task(tmp_path)
        result = run_ci_check(task)  # no baseline -> environment stays None
        d = result.to_dict()
        assert "environment" not in d
        assert "environment_warnings" not in d


# ---------------------------------------------------------------------------
# `perflab ci-check` CLI — --force
# ---------------------------------------------------------------------------

class TestCICheckCLIEnvironmentGate:
    @patch("perflab.tools.sysinfo.collect_system_info",
           return_value={"system": "Linux", "machine": "x86_64",
                          "nvidia_gpus": [{"name": "A100", "driver_version": "1"}]})
    @patch("perflab.ci._run_bench_full", return_value={"throughput": {"median": 1.0}, "ok": True})
    def test_cli_fails_on_mismatch_without_force(self, mock_bench, mock_sysinfo, tmp_path, sample_task_yaml):
        _write_baseline(
            sample_task_yaml.parent / "baseline.json", 1.0,
            environment={"system": "Linux", "machine": "x86_64", "gpu_name": "H100"},
            metric_name="throughput.median", metric_mode="maximize",
        )
        result = runner.invoke(app, ["ci-check", str(sample_task_yaml)])
        assert result.exit_code == 1
        assert "environment mismatch" in result.output

    @patch("perflab.tools.sysinfo.collect_system_info",
           return_value={"system": "Linux", "machine": "x86_64",
                          "nvidia_gpus": [{"name": "A100", "driver_version": "1"}]})
    @patch("perflab.ci._run_bench_full", return_value={"throughput": {"median": 1.0}, "ok": True})
    def test_cli_force_flag_is_accepted_and_downgrades(self, mock_bench, mock_sysinfo, tmp_path, sample_task_yaml):
        _write_baseline(
            sample_task_yaml.parent / "baseline.json", 1.0,
            environment={"system": "Linux", "machine": "x86_64", "gpu_name": "H100"},
            metric_name="throughput.median", metric_mode="maximize",
        )
        result = runner.invoke(app, ["ci-check", str(sample_task_yaml), "--force"])
        assert result.exit_code == 0
        assert "CI check PASSED" in result.output
        # Regression check ran (force downgrades, it does not silence) --
        # the mismatch is still surfaced as a warning, just not a failure.
        assert "WARNING (environment)" in result.output
        assert "downgraded by --force" in result.output
