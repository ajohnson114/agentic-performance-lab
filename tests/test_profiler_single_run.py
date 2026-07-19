"""Profilers must not re-run the benchmark more often than necessary.

Each benchmark execution costs real wall-clock (and under ncu --set full,
many replayed kernel launches). These tests pin the consolidated behavior:
- linux_perf: ONE perf stat run carries the generic counters plus the TMA
  level-1 metric group (and on AMD the cache-load events the level-2
  estimate needs) -- previously up to three separate stat runs.
- ncu: ONE profiled run produces the .ncu-rep; the CSV is exported from the
  report via `ncu --import` without re-profiling.
- power: ONE bench run measures RAPL (perf stat wrapper) with nvidia-smi
  polling alongside, instead of two runs whose CPU/GPU numbers weren't even
  from the same execution.
"""
from __future__ import annotations

import threading
from pathlib import Path

import perflab.analyzers.tma as tma_mod
import perflab.profilers.base as base
import perflab.profilers.ncu_profiler as ncu_mod
import perflab.profilers.power_profiler as power_mod
from perflab.profilers.linux_perf import LinuxPerfProfiler
from perflab.profilers.ncu_profiler import NcuProfiler
from perflab.profilers.power_profiler import PowerProfiler
from perflab.tools.shell import CmdResult


def _ok(cmd, rc=0) -> CmdResult:
    return CmdResult(cmd=list(cmd), returncode=rc, stdout="", stderr="", duration_s=0.01)


_STAT_COUNTERS = """\
 Performance counter stats for 'python3 bench.py':

     1,234,567,890      cycles
     2,469,135,780      instructions
        10,000,000      cache-references
         1,000,000      cache-misses
       500,000,000      branch-instructions
         5,000,000      branch-misses
         2,000,000      L1-dcache-load-misses
           300,000      LLC-load-misses
          4,002.12 msec task-clock                  #    3.998 CPUs utilized
"""

_TMA_METRIC_LINES = """\
                        #     23.4 %  tma_frontend_bound
                        #     45.2 %  tma_backend_bound
                        #      8.1 %  tma_bad_speculation
                        #     23.3 %  tma_retiring
"""

_AMD_LOAD_LINES = """\
       100,000,000      L1-dcache-loads
         1,000,000      LLC-loads
"""

# Combined -M run where the base ``cycles`` counter was multiplexed: perf
# appends a "(73.50%)" scaling annotation. The other counters are unscaled.
_STAT_COUNTERS_CYCLES_MUX = """\
 Performance counter stats for 'python3 bench.py':

     1,234,567,890      cycles                                            (73.50%)
     2,469,135,780      instructions
        10,000,000      cache-references
         1,000,000      cache-misses
       500,000,000      branch-instructions
         5,000,000      branch-misses
         2,000,000      L1-dcache-load-misses
           300,000      LLC-load-misses
          4,002.12 msec task-clock                  #    3.998 CPUs utilized
"""

# The unmultiplexed plain re-run (into base_stat.txt): distinct cycle count so
# the test can prove base counters were parsed from THIS run, not the combined.
_STAT_COUNTERS_REMEASURED = """\
 Performance counter stats for 'python3 bench.py':

     9,000,000,000      cycles
     2,469,135,780      instructions
        10,000,000      cache-references
         1,000,000      cache-misses
       500,000,000      branch-instructions
         5,000,000      branch-misses
         2,000,000      L1-dcache-load-misses
           300,000      LLC-load-misses
          4,002.12 msec task-clock                  #    3.998 CPUs utilized
"""


class _PerfStatFake:
    """Fake base.run_cmd for linux_perf: writes the -o stat file per flags."""

    def __init__(self, fail_combined: bool = False):
        self.calls: list[list[str]] = []
        self.fail_combined = fail_combined

    def __call__(self, cmd, cwd=None, **kwargs):
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[:2] == ["perf", "stat"]:
            out_path = Path(cmd[cmd.index("-o") + 1])
            if "-M" in cmd:
                if self.fail_combined:
                    # perf rejects the -M/-e combination: errors out before
                    # running the benchmark, no output file written.
                    return _ok(cmd, rc=129)
                out_path.write_text(
                    _STAT_COUNTERS + _TMA_METRIC_LINES, encoding="utf-8",
                )
            else:
                text = _STAT_COUNTERS
                if "L1-dcache-loads" in cmd[cmd.index("-e") + 1]:
                    text += _AMD_LOAD_LINES
                out_path.write_text(text, encoding="utf-8")
        # perf record: don't create perf.data, so script/annotate are skipped
        return _ok(cmd)


class TestLinuxPerfCombinedStat:
    def _run(self, tmp_path, monkeypatch, *, vendor: str, tma: bool, fail_combined=False):
        fake = _PerfStatFake(fail_combined=fail_combined)
        tma_calls: list[list[str]] = []

        def fake_tma_run_cmd(cmd, cwd=None, **kwargs):
            tma_calls.append(list(cmd))
            return _ok(cmd, rc=1)

        monkeypatch.setattr(base, "run_cmd", fake)
        monkeypatch.setattr(tma_mod, "run_cmd", fake_tma_run_cmd)
        monkeypatch.setattr(tma_mod, "is_tma_available", lambda: tma)
        monkeypatch.setattr(tma_mod, "_detect_cpu_vendor", lambda: vendor)

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        result = LinuxPerfProfiler().run("python3 bench.py", tmp_path, artifacts)
        return result, fake, tma_calls

    def test_intel_tma_rides_on_single_stat_run(self, tmp_path, monkeypatch):
        result, fake, tma_calls = self._run(
            tmp_path, monkeypatch, vendor="intel", tma=True,
        )

        # Exactly two benchmark executions: combined stat + record.
        assert len(fake.calls) == 2
        stat_cmd, record_cmd = fake.calls
        assert stat_cmd[:2] == ["perf", "stat"]
        assert "-M" in stat_cmd and "TopdownL1" in stat_cmd
        assert "cycles" in stat_cmd[stat_cmd.index("-e") + 1]
        assert record_cmd[:2] == ["perf", "record"]
        # TMA parsed from the combined output -- no dedicated TMA bench run.
        assert tma_calls == []
        assert result.summary["tma"]["frontend_bound_pct"] == 23.4
        assert result.summary["tma"]["backend_bound_pct"] == 45.2
        # Generic counters parsed from the same file.
        assert result.summary["cycles"] == 1_234_567_890
        assert round(result.summary["ipc"], 2) == 2.0

    def test_no_tma_support_omits_metric_group(self, tmp_path, monkeypatch):
        result, fake, tma_calls = self._run(
            tmp_path, monkeypatch, vendor="unknown", tma=False,
        )

        assert len(fake.calls) == 2
        assert "-M" not in fake.calls[0]
        # collect_tma still attempts its raw-counter fallback run (fails
        # fast on machines without topdown-* events).
        assert len(tma_calls) == 1
        assert "topdown-fetch-bubbles" in tma_calls[0][tma_calls[0].index("-e") + 1]
        assert "tma" not in result.summary

    def test_amd_level2_parsed_from_same_run(self, tmp_path, monkeypatch):
        result, fake, tma_calls = self._run(
            tmp_path, monkeypatch, vendor="amd", tma=False,
        )

        assert len(fake.calls) == 2
        events = fake.calls[0][fake.calls[0].index("-e") + 1]
        assert "L1-dcache-loads" in events and "LLC-loads" in events
        # Level 2 came from the combined stat text: the only extra TMA
        # attempt is the level-1 raw-counter fallback.
        assert len(tma_calls) == 1
        assert result.summary["tma_level2"]["source"] == "amd-perf"
        assert result.summary["tma_level2"]["dram_bound_pct"] == 30.0

    def test_perf_rejecting_metric_group_retries_plain(self, tmp_path, monkeypatch):
        result, fake, _ = self._run(
            tmp_path, monkeypatch, vendor="intel", tma=True, fail_combined=True,
        )

        # Combined run failed at the perf level -> retried without -M so the
        # base counters aren't lost. Three runs total: failed stat, plain
        # stat, record.
        assert len(fake.calls) == 3
        assert "-M" in fake.calls[0]
        assert "-M" not in fake.calls[1]
        assert fake.calls[1][:2] == ["perf", "stat"]
        assert fake.calls[2][:2] == ["perf", "record"]
        assert result.summary["cycles"] == 1_234_567_890


class _MuxStatFake:
    """base.run_cmd fake for the multiplexing fallback: the combined -M run
    scales the ``cycles`` counter; the plain re-run writes clean counters."""

    def __init__(self, mux: bool = True):
        self.calls: list[list[str]] = []
        self.mux = mux

    def __call__(self, cmd, cwd=None, **kwargs):
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[:2] == ["perf", "stat"]:
            out_path = Path(cmd[cmd.index("-o") + 1])
            if "-M" in cmd:
                base_counters = (
                    _STAT_COUNTERS_CYCLES_MUX if self.mux else _STAT_COUNTERS
                )
                out_path.write_text(base_counters + _TMA_METRIC_LINES, encoding="utf-8")
            else:
                out_path.write_text(_STAT_COUNTERS_REMEASURED, encoding="utf-8")
        # perf record: don't create perf.data, so script/annotate are skipped.
        return _ok(cmd)


class TestLinuxPerfBaseCounterMultiplexing:
    def _run(self, tmp_path, monkeypatch, *, mux: bool):
        fake = _MuxStatFake(mux=mux)
        monkeypatch.setattr(base, "run_cmd", fake)
        monkeypatch.setattr(
            tma_mod, "run_cmd", lambda cmd, cwd=None, **kw: _ok(cmd, rc=1),
        )
        monkeypatch.setattr(tma_mod, "is_tma_available", lambda: True)
        monkeypatch.setattr(tma_mod, "_detect_cpu_vendor", lambda: "intel")

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        result = LinuxPerfProfiler().run("python3 bench.py", tmp_path, artifacts)
        return result, fake

    def test_multiplexed_base_counters_trigger_plain_remeasure(self, tmp_path, monkeypatch):
        result, fake = self._run(tmp_path, monkeypatch, mux=True)

        stat_cmds = [c for c in fake.calls if c[:2] == ["perf", "stat"]]
        # Combined -M run + exactly one plain re-run into base_stat.txt.
        assert len(stat_cmds) == 2
        assert "-M" in stat_cmds[0]
        assert "-M" not in stat_cmds[1]
        assert stat_cmds[1][stat_cmds[1].index("-o") + 1].endswith("base_stat.txt")
        # Base counters come from the unmultiplexed re-run...
        assert result.summary["cycles"] == 9_000_000_000
        # ...while TMA still parses from the combined run's text.
        assert result.summary["tma"]["frontend_bound_pct"] == 23.4
        assert "perf_base_stat_txt" in result.artifacts

    def test_no_multiplexing_annotation_no_extra_run(self, tmp_path, monkeypatch):
        result, fake = self._run(tmp_path, monkeypatch, mux=False)

        stat_cmds = [c for c in fake.calls if c[:2] == ["perf", "stat"]]
        # No scaling annotation -> single combined stat run, no re-measure.
        assert len(stat_cmds) == 1
        assert result.summary["cycles"] == 1_234_567_890
        assert result.summary["tma"]["frontend_bound_pct"] == 23.4
        assert "perf_base_stat_txt" not in result.artifacts


class TestNcuSingleProfiledRun:
    # A layout _parse_ncu_csv recognizes: "SM Throughput (%)" maps to a real
    # per-kernel metric, so the parsed summary is usable.
    _USABLE_CSV = '"Kernel Name","SM Throughput (%)"\n"k1","50.0"\n'
    # A layout _parse_ncu_csv can't map: rows parse but yield no per-kernel
    # metric (only name + invocations), so the summary is unusable.
    _UNUSABLE_CSV = '"Mystery Col","Other Col"\n"a","b"\n'

    def _fakes(self, monkeypatch, *, report_ok=True, import_ok=True, import_usable=True):
        bench_runs: list[list[str]] = []
        export_calls: list[list[str]] = []

        def fake_bench_run_cmd(cmd, cwd=None, **kwargs):
            cmd = list(cmd)
            bench_runs.append(cmd)
            if "-o" in cmd and report_ok:
                Path(cmd[cmd.index("-o") + 1]).write_text("", encoding="utf-8")
            if "--log-file" in cmd:
                # A live profiled --csv run always emits a recognized layout.
                Path(cmd[cmd.index("--log-file") + 1]).write_text(
                    self._USABLE_CSV, encoding="utf-8",
                )
            return _ok(cmd, rc=0 if report_ok else 1)

        def fake_export_run_cmd(cmd, cwd=None, **kwargs):
            cmd = list(cmd)
            export_calls.append(cmd)
            if import_ok:
                content = self._USABLE_CSV if import_usable else self._UNUSABLE_CSV
                Path(cmd[cmd.index("--log-file") + 1]).write_text(
                    content, encoding="utf-8",
                )
            return _ok(cmd, rc=0 if import_ok else 1)

        monkeypatch.setattr(base, "run_cmd", fake_bench_run_cmd)
        monkeypatch.setattr(ncu_mod, "run_cmd", fake_export_run_cmd)
        return bench_runs, export_calls

    def test_csv_exported_from_report_without_second_profile(self, tmp_path, monkeypatch):
        bench_runs, export_calls = self._fakes(monkeypatch)
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = NcuProfiler().run("python3 bench.py", tmp_path, artifacts)

        # ONE profiled benchmark run (--set full is the most expensive mode).
        assert len(bench_runs) == 1
        assert "-o" in bench_runs[0]
        assert len(export_calls) == 1
        assert export_calls[0][:2] == ["ncu", "--import"]
        assert result.summary["csv_returncode"] == 0
        assert result.summary["report_returncode"] == 0
        assert "ncu_metrics_csv" in result.artifacts
        assert "ncu_report" in result.artifacts
        # The export CSV parsed to real per-kernel metrics -> no live fallback.
        assert result.summary["kernels"][0]["sm_utilization_pct"] == 50.0

    def test_missing_report_falls_back_to_profiled_csv_run(self, tmp_path, monkeypatch):
        bench_runs, export_calls = self._fakes(monkeypatch, report_ok=False)
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = NcuProfiler().run("python3 bench.py", tmp_path, artifacts)

        # No report -> no import possible -> old two-run behavior.
        assert export_calls == []
        assert len(bench_runs) == 2
        assert "--csv" in bench_runs[1]
        assert result.summary["csv_returncode"] == 1

    def test_failed_export_falls_back_to_profiled_csv_run(self, tmp_path, monkeypatch):
        bench_runs, export_calls = self._fakes(monkeypatch, import_ok=False)
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = NcuProfiler().run("python3 bench.py", tmp_path, artifacts)

        assert len(export_calls) == 1
        assert len(bench_runs) == 2
        assert "--csv" in bench_runs[1]
        assert result.summary["csv_returncode"] == 0  # fallback run succeeded

    def test_unusable_export_csv_falls_back_to_live_run(self, tmp_path, monkeypatch):
        # Export succeeds (file written) but its column layout yields no
        # per-kernel metrics -> the summary is unusable, so the live profiled
        # --csv fallback fires and re-parses, keeping the roofline honest.
        bench_runs, export_calls = self._fakes(monkeypatch, import_usable=False)
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = NcuProfiler().run("python3 bench.py", tmp_path, artifacts)

        assert len(export_calls) == 1
        assert len(bench_runs) == 2
        assert "--csv" in bench_runs[1]
        # csv_returncode reflects the live fallback run that produced the CSV.
        assert result.summary["csv_returncode"] == 0
        assert result.summary["kernels"][0]["sm_utilization_pct"] == 50.0


_RAPL_TEXT = """\
 12.34 Joules power/energy-pkg/
  8.56 Joules power/energy-cores/

       1.234567 seconds time elapsed
"""


class TestPowerSingleBenchRun:
    def _run(self, tmp_path, monkeypatch, *, rapl: bool, smi: bool, rapl_usable: bool = True):
        bench_runs: list[dict] = []
        first_sample = threading.Event()

        def fake_run_bench_under(wrapper, bench_cmd, cwd, **kwargs):
            wrapper = list(wrapper)
            bench_runs.append({"wrapper": wrapper})
            if "-o" in wrapper:
                Path(wrapper[wrapper.index("-o") + 1]).write_text(
                    _RAPL_TEXT, encoding="utf-8",
                )
            if smi:
                # Deterministic: hold the "benchmark" until the poller
                # thread has collected at least one nvidia-smi sample.
                assert first_sample.wait(timeout=5)
            return _ok(wrapper)

        def fake_run_cmd(cmd, cwd=None, timeout_s=None, **kwargs):
            first_sample.set()
            return CmdResult(
                cmd=list(cmd), returncode=0,
                stdout="215.5, 1024, 40960\n", stderr="", duration_s=0.01,
            )

        monkeypatch.setattr(power_mod, "run_bench_under", fake_run_bench_under)
        monkeypatch.setattr(power_mod, "run_cmd", fake_run_cmd)
        monkeypatch.setattr(power_mod, "_has_rapl", lambda: rapl)
        # Patch the "can perf actually open RAPL?" probe directly so it never
        # spawns perf (and never runs the fake run_cmd, which would trip the
        # first_sample gate prematurely).
        monkeypatch.setattr(power_mod, "_rapl_usable", lambda: rapl_usable)
        monkeypatch.setattr(
            power_mod.shutil, "which",
            lambda name: "/usr/bin/nvidia-smi" if smi else None,
        )

        artifacts = tmp_path / "artifacts"
        result = PowerProfiler().run("python3 bench.py", tmp_path, artifacts)
        return result, bench_runs

    def test_rapl_and_gpu_polling_share_one_run(self, tmp_path, monkeypatch):
        result, bench_runs = self._run(tmp_path, monkeypatch, rapl=True, smi=True)

        assert len(bench_runs) == 1
        wrapper = bench_runs[0]["wrapper"]
        assert wrapper[:2] == ["perf", "stat"]
        assert "power/energy-pkg/" in wrapper[wrapper.index("-e") + 1]
        # CPU and GPU power both measured, from the same execution.
        assert result.summary["rapl"]["package_joules"] == 12.34
        assert result.summary["gpu_power"]["sample_count"] >= 1
        assert result.summary["gpu_memory"]["total_mib"] == 40960

    def test_rapl_only_single_wrapped_run(self, tmp_path, monkeypatch):
        result, bench_runs = self._run(tmp_path, monkeypatch, rapl=True, smi=False)

        assert len(bench_runs) == 1
        assert bench_runs[0]["wrapper"][:2] == ["perf", "stat"]
        assert "rapl" in result.summary
        assert "gpu_power" not in result.summary

    def test_smi_only_single_bare_run(self, tmp_path, monkeypatch):
        result, bench_runs = self._run(tmp_path, monkeypatch, rapl=False, smi=True)

        assert len(bench_runs) == 1
        assert bench_runs[0]["wrapper"] == []
        assert "rapl" not in result.summary
        assert result.summary["gpu_power"]["sample_count"] >= 1

    def test_neither_source_runs_nothing(self, tmp_path, monkeypatch):
        result, bench_runs = self._run(tmp_path, monkeypatch, rapl=False, smi=False)

        assert bench_runs == []
        assert result.summary == {}

    def test_rapl_listed_but_unusable_runs_bench_bare_for_gpu(self, tmp_path, monkeypatch):
        # perf lists the RAPL PMU but can't open the events (perf_event_paranoid
        # / CAP_PERFMON). The old code still wrapped the run, so perf exited
        # before exec'ing the benchmark and nvidia-smi sampled an idle GPU.
        # Now the benchmark runs bare so GPU polling gets a real window.
        result, bench_runs = self._run(
            tmp_path, monkeypatch, rapl=True, smi=True, rapl_usable=False,
        )

        assert len(bench_runs) == 1
        assert bench_runs[0]["wrapper"] == []  # no perf stat wrapper
        assert "rapl" not in result.summary
        assert "rapl_unavailable" in result.summary
        assert "perf_event_paranoid" in result.summary["rapl_unavailable"]
        assert result.summary["gpu_power"]["sample_count"] >= 1

    def test_rapl_usable_probe_gates_the_perf_wrapper(self, tmp_path, monkeypatch):
        # Same has_rapl=True, but the probe passes -> wrapper is used.
        result, bench_runs = self._run(
            tmp_path, monkeypatch, rapl=True, smi=True, rapl_usable=True,
        )

        assert bench_runs[0]["wrapper"][:2] == ["perf", "stat"]
        assert "rapl" in result.summary
        assert "rapl_unavailable" not in result.summary

    def test_rapl_unusable_and_no_gpu_skips_run(self, tmp_path, monkeypatch):
        # Nothing measurable: no usable RAPL and no GPU. Skip the benchmark run
        # entirely, but still record why CPU energy is missing.
        result, bench_runs = self._run(
            tmp_path, monkeypatch, rapl=True, smi=False, rapl_usable=False,
        )

        assert bench_runs == []
        assert result.summary == {"rapl_unavailable": power_mod._RAPL_UNAVAILABLE_REASON}
