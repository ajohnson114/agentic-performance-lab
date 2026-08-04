"""Tests for the CPU peak-FLOPS model and the measured-peak microbenchmark.

Covers the two halves of `perflab/roofline_peaks.py`'s CPU path:

* the ISA / clock model (`_cpu_isa_profile`, `_linux_sustained_clock_ghz`,
  `_estimate_cpu_peaks`) -- exercised with synthetic flag strings and fake
  `_run` output, since macOS has neither `lscpu` nor `/proc/cpuinfo`;
* the measured probe (`_measured_cpu_peaks`) -- its guard rails, every failure
  mode, and cache reuse.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from perflab import roofline_peaks as rp
from perflab.roofline_peaks import Peaks

# ---------------------------------------------------------------- ISA table


def _isa(flags: str, name: str = "Generic CPU", machine: str = "x86_64"):
    return rp._cpu_isa_profile(set(flags.split()), name, machine)


class TestIsaFlopsPerCycle:
    """One case per branch of the fp32 FLOP/cycle table.

    Every expected number is fp32_lanes * 2 (mul+add) * pipes.
    """

    def test_avx512_dual_fma_server_part(self):
        # Xeon Platinum 8380: two 512-bit FMA units -> 16 * 2 * 2.
        isa = _isa("avx avx2 fma avx512f avx512bw", "Intel Xeon Platinum 8380")
        assert isa.name == "avx512"
        assert isa.pipes == 2
        assert isa.flops_per_cycle == 64

    def test_avx512_single_fma_is_the_default(self):
        # Client 512-bit parts have one FMA unit; unknown parts get the same
        # conservative assumption.
        isa = _isa("avx avx2 fma avx512f", "Intel Core i7-1165G7")
        assert isa.name == "avx512"
        assert isa.pipes == 1
        assert isa.flops_per_cycle == 32

    def test_xeon_gold_5xxx_stays_single_fma(self):
        """Gold 51xx/52xx really do have one FMA512 -- only Gold 6xxx gets two."""
        assert _isa("avx512f", "Intel Xeon Gold 5218").flops_per_cycle == 32
        assert _isa("avx512f", "Intel Xeon Gold 6248").flops_per_cycle == 64

    def test_sapphire_rapids_workstation_dual_fma(self):
        assert _isa("avx512f", "Intel Xeon w9-3495X").flops_per_cycle == 64

    def test_avx2_with_fma_gets_two_units(self):
        # Haswell onward / Zen 2 onward: 8 * 2 * 2.
        isa = _isa("avx avx2 fma", "Intel Core i9-9900K")
        assert isa.name == "avx2+fma"
        assert isa.flops_per_cycle == 32

    def test_zen1_avx2_is_half_rate(self):
        isa = _isa("avx avx2 fma", "AMD Ryzen 7 1700 Eight-Core Processor")
        assert isa.name == "avx2+fma"
        assert isa.pipes == 1
        assert isa.flops_per_cycle == 16

    def test_atom_avx2_is_half_rate(self):
        assert _isa("avx avx2 fma", "Intel Atom x6425E").flops_per_cycle == 16

    def test_pre_fma_avx_is_not_lumped_with_avx2(self):
        """Sandy/Ivy Bridge: 256-bit mul + add co-issue, no FMA -> 8 * 2 * 1."""
        isa = _isa("sse sse2 avx", "Intel Core i7-2600K")
        assert isa.name == "avx"
        assert isa.pipes == 1
        assert isa.flops_per_cycle == 16
        # ...and it is half of what a real AVX2+FMA part gets.
        assert _isa("avx avx2 fma").flops_per_cycle == 2 * isa.flops_per_cycle

    def test_sse2_has_no_fma(self):
        isa = _isa("fpu sse sse2", "Intel Xeon X5690")
        assert isa.name == "sse2"
        assert isa.pipes == 1
        assert isa.flops_per_cycle == 8

    def test_scalar_when_no_simd_flags(self):
        isa = _isa("fpu tsc msr")
        assert isa.name == "scalar"
        assert isa.flops_per_cycle == 2

    def test_aarch64_neon(self):
        isa = _isa("fp asimd aes crc32", "Neoverse N1", machine="aarch64")
        assert isa.name == "neon"
        assert isa.flops_per_cycle == 16

    def test_aarch64_without_asimd_is_scalar(self):
        assert _isa("fp aes", "weird-arm", machine="aarch64").name == "scalar"

    def test_arm64_alias_is_recognized(self):
        assert _isa("asimd", "M", machine="arm64").name == "neon"


class TestIsaFlagMatchingIsExact:
    """The old code substring-matched a raw blob; these are the traps that hid there."""

    def test_avx512_machine_is_not_classified_as_pre_fma_avx(self):
        # "avx" is a substring of the flag list of every AVX-512 machine.
        assert _isa("avx avx2 avx512f avx512dq fma").name == "avx512"

    def test_sse4_alone_does_not_imply_sse(self):
        # `"sse" in blob` used to be true because of sse4_1/ssse3.
        assert _isa("fpu sse4_1 ssse3").name == "scalar"

    def test_avx2_without_fma_does_not_claim_fma(self):
        assert _isa("avx avx2").name == "avx"

    def test_non_x86_machine_string_does_not_crash(self):
        assert _isa("", "", machine="").name == "scalar"


# ------------------------------------------------------------------- clocks


def _fake_run(mapping: dict[str, str | None]):
    """Build a `_run` stub keyed by a substring of the command."""

    def run(cmd):
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        for needle, value in mapping.items():
            if needle in cmd_str:
                return value
        return None

    return run


class TestSustainedClock:
    def test_prefers_sysfs_base_frequency(self):
        with patch.object(rp, "_run", side_effect=_fake_run({
            "base_frequency": "2300000",       # kHz
            "CPU max MHz": "3900.0",
        })):
            assert rp._linux_sustained_clock_ghz("Intel Xeon Gold 6248") == (2.3, "sysfs-base")

    def test_falls_back_to_base_clock_in_the_brand_string(self):
        with patch.object(rp, "_run", side_effect=_fake_run({"CPU max MHz": "3900.0"})):
            ghz, src = rp._linux_sustained_clock_ghz("Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz")
        assert (ghz, src) == (2.5, "model-name-base")

    def test_derates_single_core_max_turbo_when_that_is_all_there_is(self):
        """The headline bug: max turbo must not be multiplied by every core."""
        with patch.object(rp, "_run", side_effect=_fake_run({"CPU max MHz": "4800.0"})):
            ghz, src = rp._linux_sustained_clock_ghz("Intel Xeon w9-3495X")
        assert src == "max-turbo-derated"
        assert ghz == pytest.approx(4.8 * rp._ALL_CORE_TURBO_DERATE)
        assert ghz < 4.8

    def test_cpuinfo_spot_reading_is_the_last_resort_and_is_not_derated(self):
        with patch.object(rp, "_run", side_effect=_fake_run({"cpu MHz": "2494.140"})):
            ghz, src = rp._linux_sustained_clock_ghz("QEMU Virtual CPU")
        assert src == "cpuinfo-spot"
        assert ghz == pytest.approx(2.49414)

    def test_returns_none_when_nothing_reports_a_clock(self):
        with patch.object(rp, "_run", return_value=None):
            assert rp._linux_sustained_clock_ghz("CPU") is None

    def test_garbage_values_are_skipped(self):
        with patch.object(rp, "_run", side_effect=_fake_run({
            "base_frequency": "not-a-number",
            "CPU max MHz": "",
            "cpu MHz": "3000.0",
        })):
            ghz, src = rp._linux_sustained_clock_ghz("CPU")
        assert (src, ghz) == ("cpuinfo-spot", 3.0)


class TestAvx512LicenseDerate:
    def test_skylake_sp_and_cascade_lake_are_derated(self):
        assert rp._avx512_clock_factor("Intel Xeon Platinum 8180") == rp._AVX512_HEAVY_LICENSE_FACTOR
        assert rp._avx512_clock_factor("Intel Xeon Gold 6248") == rp._AVX512_HEAVY_LICENSE_FACTOR
        assert rp._avx512_clock_factor("Intel Xeon Silver 4114") == rp._AVX512_HEAVY_LICENSE_FACTOR

    def test_ice_lake_and_later_are_not(self):
        assert rp._avx512_clock_factor("Intel Xeon Platinum 8380") == 1.0   # Ice Lake-SP
        assert rp._avx512_clock_factor("Intel Xeon Platinum 8480+") == 1.0  # Sapphire Rapids
        assert rp._avx512_clock_factor("AMD EPYC 9654 96-Core Processor") == 1.0
        assert rp._avx512_clock_factor("") == 1.0


# ------------------------------------------------------- model, end to end


def _linux_model(flags: str, name: str, cores: int, run_extra: dict[str, str | None]):
    mapping = {"model name": name, "flags": flags}
    mapping.update(run_extra)
    with patch.object(rp.platform, "system", return_value="Linux"), \
         patch.object(rp.platform, "machine", return_value="x86_64"), \
         patch.object(rp, "_run", side_effect=_fake_run(mapping)), \
         patch.object(rp, "_physical_cpu_count", return_value=cores):
        return rp._estimate_cpu_peaks()


class TestEstimateCpuPeaksLinux:
    def test_cascade_lake_applies_both_derates(self):
        """Gold 6248: 20 cores, base 2.5 GHz, 2x FMA512, heavy-license 0.70."""
        peaks = _linux_model(
            "fpu sse sse2 avx avx2 fma avx512f avx512bw",
            "Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz",
            cores=20,
            run_extra={"CPU max MHz": "3900.0"},
        )
        assert peaks is not None
        assert peaks.source == "cpu-spec"
        assert peaks.peak_tflops == pytest.approx(20 * 64 * 2.5 * 0.70 / 1000.0)

    def test_sandy_bridge_is_half_of_an_equivalent_avx2_part(self):
        common = dict(cores=8, run_extra={"CPU max MHz": "4000.0"})
        pre_fma = _linux_model("fpu sse sse2 avx", "Intel Core i7-2600K", **common)
        with_fma = _linux_model("fpu sse sse2 avx avx2 fma", "Intel Core i7-4790K", **common)
        assert pre_fma is not None and with_fma is not None
        assert with_fma.peak_tflops == pytest.approx(2 * pre_fma.peak_tflops)

    def test_aarch64_linux_is_modeled_rather_than_treated_as_scalar(self):
        mapping = {"model name": None, "Model name": "Neoverse-N1",
                   "Features": "fp asimd aes", "flags": "fp asimd aes",
                   "CPU max MHz": "3000.0"}
        with patch.object(rp.platform, "system", return_value="Linux"), \
             patch.object(rp.platform, "machine", return_value="aarch64"), \
             patch.object(rp, "_run", side_effect=_fake_run(mapping)), \
             patch.object(rp, "_physical_cpu_count", return_value=64):
            peaks = rp._estimate_cpu_peaks()
        assert peaks is not None
        assert peaks.peak_tflops == pytest.approx(64 * 16 * 3.0 * 0.75 / 1000.0)

    def test_no_clock_means_no_model(self):
        peaks = _linux_model("avx2 fma", "Some CPU", cores=8, run_extra={})
        assert peaks is None


class TestEstimateCpuPeaksDarwin:
    def test_apple_p_cores_use_four_neon_pipes(self):
        mapping = {
            "machdep.cpu.brand_string": "Apple M4",
            "hw.perflevel0.logicalcpu": "4",
        }
        with patch.object(rp.platform, "system", return_value="Darwin"), \
             patch.object(rp, "_run", side_effect=_fake_run(mapping)):
            peaks = rp._estimate_cpu_peaks()
        assert peaks is not None
        # 4 P-cores x 32 FLOP/cycle x 4.4 GHz
        assert peaks.peak_tflops == pytest.approx(4 * 32 * 4.4 / 1000.0)
        assert peaks.peak_mem_bw_gbs == 120.0


# ------------------------------------------------------------ measured path


def _rates(value: float, n: int = 5) -> list[float]:
    return [value * (1.0 - 0.01 * i) for i in range(n)]


def _probe_payload(tflops: float = 2.0, gbs: float = 100.0) -> dict:
    return {"numpy": "2.5.1", "gemm_tflops": _rates(tflops), "stream_gbs": _rates(gbs)}


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "_CACHE_PATH", tmp_path / "peaks.json")
    monkeypatch.delenv("PERFLAB_PEAKS_NO_CACHE", raising=False)
    monkeypatch.setenv("PERFLAB_CPU_PEAK_MEASURE", "1")
    return tmp_path / "peaks.json"


MODEL = Peaks(0.5, 50.0, "cpu-spec", "Test CPU")


class TestMeasuredCpuPeaks:
    def test_uses_the_measurement_when_it_is_sound(self, isolated_cache):
        with patch.object(rp, "_run_cpu_probe", return_value=_probe_payload()):
            peaks = rp._measured_cpu_peaks(MODEL)
        assert peaks is not None
        assert peaks.source == "cpu-measured"
        assert peaks.peak_tflops == pytest.approx(2.0)   # max across trials
        assert peaks.peak_mem_bw_gbs == pytest.approx(100.0)
        assert peaks.device == "Test CPU"

    def test_peak_is_the_max_not_the_mean(self, isolated_cache):
        payload = {"gemm_tflops": [1.0, 1.4, 1.2], "stream_gbs": [50.0, 55.0, 52.0]}
        with patch.object(rp, "_run_cpu_probe", return_value=payload):
            peaks = rp._measured_cpu_peaks(MODEL)
        assert peaks is not None
        assert peaks.peak_tflops == pytest.approx(1.4)
        assert peaks.peak_mem_bw_gbs == pytest.approx(55.0)

    def test_best_thread_configuration_wins(self, isolated_cache):
        """The plan tries more than one width; the roof is the best of them."""
        results = [
            {"gemm_tflops": _rates(1.0), "stream_gbs": _rates(40.0)},
            {"gemm_tflops": _rates(2.5), "stream_gbs": _rates(30.0)},
        ]
        with patch.object(rp, "_cpu_probe_threads", return_value=8), \
             patch.object(rp, "_run_cpu_probe", side_effect=results) as probe:
            peaks = rp._measured_cpu_peaks(MODEL)
        assert probe.call_count == 2
        assert peaks is not None
        assert peaks.peak_tflops == pytest.approx(2.5)
        assert peaks.peak_mem_bw_gbs == pytest.approx(40.0)

    def test_disabled_by_env(self, isolated_cache, monkeypatch):
        monkeypatch.setenv("PERFLAB_CPU_PEAK_MEASURE", "0")
        with patch.object(rp, "_run_cpu_probe") as probe:
            assert rp._measured_cpu_peaks(MODEL) is None
        probe.assert_not_called()

    def test_device_name_without_a_model(self, isolated_cache):
        with patch.object(rp, "_run_cpu_probe", return_value=_probe_payload()):
            peaks = rp._measured_cpu_peaks(None)
        assert peaks is not None
        assert peaks.device


class TestMeasuredCpuPeaksFallsBack:
    """Every failure mode must yield None (caller uses the model), never raise."""

    def test_numpy_missing(self, isolated_cache):
        payload = {"error": "numpy-unavailable: ModuleNotFoundError()"}
        with patch.object(rp, "_run_cpu_probe", return_value=payload) as probe:
            assert rp._measured_cpu_peaks(MODEL) is None
        # No point retrying a second thread count when numpy is simply absent.
        assert probe.call_count == 1

    def test_probe_process_failed(self, isolated_cache):
        """A crashed/timed-out child stops the sweep: no second full timeout."""
        with patch.object(rp, "_cpu_probe_threads", return_value=16), \
             patch.object(rp, "_run_cpu_probe", return_value=None) as probe:
            assert rp._measured_cpu_peaks(MODEL) is None
        assert probe.call_count == 1

    def test_gemm_raised_inside_the_probe(self, isolated_cache):
        payload = {"gemm_error": "MemoryError()", "stream_gbs": _rates(100.0)}
        with patch.object(rp, "_run_cpu_probe", return_value=payload):
            assert rp._measured_cpu_peaks(MODEL) is None

    def test_too_few_trials(self, isolated_cache):
        payload = {"gemm_tflops": [2.0, 1.9], "stream_gbs": _rates(100.0)}
        with patch.object(rp, "_run_cpu_probe", return_value=payload):
            assert rp._measured_cpu_peaks(MODEL) is None

    def test_unstable_trials_are_rejected(self, isolated_cache):
        """A wide max/median spread means the box was contended, not fast."""
        payload = {"gemm_tflops": [0.1, 0.12, 0.11, 5.0], "stream_gbs": _rates(100.0)}
        with patch.object(rp, "_run_cpu_probe", return_value=payload):
            assert rp._measured_cpu_peaks(MODEL) is None

    def test_implausibly_high_result_is_rejected(self, isolated_cache):
        payload = {"gemm_tflops": _rates(50_000.0), "stream_gbs": _rates(100.0)}
        with patch.object(rp, "_run_cpu_probe", return_value=payload):
            assert rp._measured_cpu_peaks(MODEL) is None

    def test_implausibly_low_result_is_rejected(self, isolated_cache):
        payload = {"gemm_tflops": _rates(1e-9), "stream_gbs": _rates(100.0)}
        with patch.object(rp, "_run_cpu_probe", return_value=payload):
            assert rp._measured_cpu_peaks(MODEL) is None

    def test_non_numeric_rates_are_ignored(self, isolated_cache):
        payload = {"gemm_tflops": ["fast", None, 2.0], "stream_gbs": _rates(100.0)}
        with patch.object(rp, "_run_cpu_probe", return_value=payload):
            assert rp._measured_cpu_peaks(MODEL) is None

    def test_rates_of_the_wrong_shape_are_ignored(self, isolated_cache):
        payload = {"gemm_tflops": {"median": 2.0}, "stream_gbs": _rates(100.0)}
        with patch.object(rp, "_run_cpu_probe", return_value=payload):
            assert rp._measured_cpu_peaks(MODEL) is None

    def test_a_slow_machine_is_believed_not_overridden(self, isolated_cache):
        """Measured far below modeled is usually the truth (VM, cgroup quota)."""
        payload = {"gemm_tflops": _rates(0.01), "stream_gbs": _rates(5.0)}
        with patch.object(rp, "_run_cpu_probe", return_value=payload):
            peaks = rp._measured_cpu_peaks(Peaks(10.0, 200.0, "cpu-spec", "Big CPU"))
        assert peaks is not None
        assert peaks.peak_tflops == pytest.approx(0.01)


class TestMeasuredBandwidthFallback:
    def test_bandwidth_failure_keeps_measured_flops_and_says_so(self, isolated_cache):
        payload = {"gemm_tflops": _rates(2.0), "stream_error": "MemoryError()"}
        with patch.object(rp, "_run_cpu_probe", return_value=payload):
            peaks = rp._measured_cpu_peaks(MODEL)
        assert peaks is not None
        assert peaks.source == "cpu-measured-flops"
        assert peaks.peak_tflops == pytest.approx(2.0)
        assert peaks.peak_mem_bw_gbs == MODEL.peak_mem_bw_gbs

    def test_bandwidth_failure_with_no_model_bandwidth_falls_back_entirely(self, isolated_cache):
        payload = {"gemm_tflops": _rates(2.0)}
        with patch.object(rp, "_run_cpu_probe", return_value=payload):
            assert rp._measured_cpu_peaks(None) is None


class TestMeasuredCpuPeaksCache:
    def test_second_call_reuses_the_cache_and_does_not_re_probe(self, isolated_cache):
        with patch.object(rp, "_run_cpu_probe", return_value=_probe_payload()) as probe:
            first = rp._measured_cpu_peaks(MODEL)
            calls_after_first = probe.call_count
            second = rp._measured_cpu_peaks(MODEL)
        assert probe.call_count == calls_after_first
        assert first is not None and second is not None
        assert second.peak_tflops == first.peak_tflops
        assert second.source == first.source
        assert isolated_cache.exists()

    def test_failures_are_cached_too(self, isolated_cache):
        with patch.object(rp, "_run_cpu_probe", return_value=None) as probe:
            assert rp._measured_cpu_peaks(MODEL) is None
            calls_after_first = probe.call_count
            assert rp._measured_cpu_peaks(MODEL) is None
        assert probe.call_count == calls_after_first
        cached = json.loads(isolated_cache.read_text())
        assert any(v.get("failed") for v in cached.values())

    def test_no_cache_env_forces_a_re_probe(self, isolated_cache, monkeypatch):
        monkeypatch.setenv("PERFLAB_PEAKS_NO_CACHE", "1")
        with patch.object(rp, "_cpu_probe_threads", return_value=1), \
             patch.object(rp, "_run_cpu_probe", return_value=_probe_payload()) as probe:
            rp._measured_cpu_peaks(MODEL)
            rp._measured_cpu_peaks(MODEL)
        assert probe.call_count == 2

    def test_malformed_cache_entry_is_ignored_not_fatal(self, isolated_cache):
        with patch.object(rp, "_cpu_probe_threads", return_value=1):
            key = rp._cpu_measured_cache_key(MODEL.device, 1)
            isolated_cache.parent.mkdir(parents=True, exist_ok=True)
            isolated_cache.write_text(json.dumps({key: {"peak_tflops": "???"}}))
            with patch.object(rp, "_run_cpu_probe", return_value=_probe_payload()):
                peaks = rp._measured_cpu_peaks(MODEL)
        assert peaks is not None
        assert peaks.peak_tflops == pytest.approx(2.0)

    def test_cache_key_changes_with_the_probe_version(self):
        key = rp._cpu_measured_cache_key("Test CPU", 8)
        assert f"v{rp._CPU_PROBE_VERSION}" in key
        assert "8t" in key
        assert rp._cpu_measured_cache_key("Test CPU", 4) != key


class TestThreadPlan:
    def test_wide_machines_also_try_half_width(self):
        assert rp._cpu_probe_thread_plan(16) == [16, 8]

    def test_narrow_machines_only_try_one_configuration(self):
        assert rp._cpu_probe_thread_plan(2) == [2]
        assert rp._cpu_probe_thread_plan(1) == [1]


# --------------------------------------------------- probe subprocess layer


class TestRunCpuProbe:
    def test_timeout_returns_none(self):
        with patch.object(rp.subprocess, "run",
                          side_effect=subprocess.TimeoutExpired("python", 30.0)):
            assert rp._run_cpu_probe(4) is None

    def test_oserror_returns_none(self):
        with patch.object(rp.subprocess, "run", side_effect=OSError("no exec")):
            assert rp._run_cpu_probe(4) is None

    def test_no_marker_line_returns_none(self):
        completed = subprocess.CompletedProcess(["python"], 1, stdout="Traceback...\n", stderr="")
        with patch.object(rp.subprocess, "run", return_value=completed):
            assert rp._run_cpu_probe(4) is None

    def test_unparseable_marker_line_returns_none(self):
        completed = subprocess.CompletedProcess(
            ["python"], 0, stdout=rp._CPU_PROBE_MARKER + "{not json\n", stderr="")
        with patch.object(rp.subprocess, "run", return_value=completed):
            assert rp._run_cpu_probe(4) is None

    def test_non_dict_payload_returns_none(self):
        completed = subprocess.CompletedProcess(
            ["python"], 0, stdout=rp._CPU_PROBE_MARKER + "[1, 2]\n", stderr="")
        with patch.object(rp.subprocess, "run", return_value=completed):
            assert rp._run_cpu_probe(4) is None

    def test_sets_blas_thread_env_before_the_child_imports_numpy(self):
        completed = subprocess.CompletedProcess(
            ["python"], 0, stdout=rp._CPU_PROBE_MARKER + json.dumps(_probe_payload()) + "\n",
            stderr="")
        with patch.object(rp.subprocess, "run", return_value=completed) as run:
            payload = rp._run_cpu_probe(6)
        assert payload is not None
        env = run.call_args.kwargs["env"]
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
            assert env[var] == "6"
        # The child must not inherit a candidate workspace as cwd: `python -c`
        # puts cwd on sys.path.
        assert run.call_args.kwargs["cwd"]
        assert run.call_args.kwargs["timeout"] == rp._CPU_PROBE_TIMEOUT_S

    def test_probe_really_runs_and_reports_plausible_rates(self, monkeypatch):
        """End-to-end exercise of the probe script itself, shrunk to be quick."""
        pytest.importorskip("numpy")
        monkeypatch.setattr(rp, "_CPU_PROBE_WARMUP_S", 0.01)
        monkeypatch.setattr(rp, "_CPU_PROBE_MIN_TRIAL_S", 0.005)
        monkeypatch.setattr(rp, "_CPU_PROBE_TRIALS", 3)
        monkeypatch.setattr(rp, "_CPU_PROBE_GEMM_N", 256)
        monkeypatch.setattr(rp, "_CPU_PROBE_BW_MB", 4)
        payload = rp._run_cpu_probe(2)
        assert payload is not None, "probe subprocess produced no result"
        assert "error" not in payload
        assert len(payload["gemm_tflops"]) == 3
        assert len(payload["stream_gbs"]) == 3
        assert all(r > 0 for r in payload["gemm_tflops"])
        assert all(r > 0 for r in payload["stream_gbs"])


# ---------------------------------------------------------------- ordering


class TestInferCpuPeaksOrdering:
    def test_measurement_wins_over_the_model(self):
        measured = Peaks(2.0, 100.0, "cpu-measured", "CPU")
        with patch.object(rp, "_estimate_cpu_peaks", return_value=MODEL), \
             patch.object(rp, "_measured_cpu_peaks", return_value=measured) as m:
            result = rp.infer_cpu_peaks()
        assert result is measured
        # The model is still computed first: it names the device and supplies
        # the bandwidth roof if only the compute half survives validation.
        m.assert_called_once_with(MODEL)

    def test_model_is_used_when_measurement_declines(self):
        with patch.object(rp, "_estimate_cpu_peaks", return_value=MODEL), \
             patch.object(rp, "_measured_cpu_peaks", return_value=None):
            assert rp.infer_cpu_peaks() is MODEL
