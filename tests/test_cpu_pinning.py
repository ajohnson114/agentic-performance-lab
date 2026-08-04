"""CPU environment control for measured subprocesses.

Covers perflab.tools.shell's pinning policy (physical-core selection, SMT
sibling avoidance, core-0 avoidance, socket locality), the governor/turbo
reporting that deliberately stops short of applying anything, the honest
nice(2) capability check, and -- most importantly -- the invariant that the
baseline and every candidate are measured under the *same* CPU environment.

Sections
  1. CPU list parsing
  2. Physical core selection policy (fake sysfs topology)
  3. Plan resolution: specs, precedence, non-Linux degradation
  4. Baseline/candidate parity -- the correctness requirement
  5. Governor / turbo detection and reporting
  6. nice(2) capability
  7. OpenMP binding env
  8. Linux acceptance: affinity actually reaches the child
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import platform
import textwrap
from pathlib import Path

import pytest

from perflab.tools import shell
from perflab.tools.shell import (
    CpuPlan,
    _build_cpu_plan,
    _can_renice,
    _cpu_freq_state,
    _parse_cpu_list,
    _physical_cores,
    _select_cpus,
    resolve_cpu_plan,
    run_cmd,
    set_task_cpu_pinning,
)

IS_LINUX = platform.system() == "Linux"
linux_only = pytest.mark.skipif(not IS_LINUX, reason="Linux-only CPU affinity API")

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _reset_cpu_plan_cache(monkeypatch):
    """The plan is a process-wide singleton; keep tests from leaking into each
    other (and from leaking into the rest of the suite)."""
    monkeypatch.setattr(shell, "_cached_cpu_plan", None, raising=False)
    monkeypatch.setattr(shell, "_task_cpu_pinning", None, raising=False)
    monkeypatch.delenv("PERFLAB_CPU_PINNING", raising=False)
    yield
    shell._cached_cpu_plan = None
    shell._task_cpu_pinning = None


def _fake_sysfs(
    tmp_path: Path,
    *,
    siblings: dict[int, list[int]],
    packages: dict[int, int] | None = None,
    governor: str | None = None,
    no_turbo: str | None = None,
    boost: str | None = None,
) -> Path:
    """Build a fake /sys/devices/system/cpu tree.

    siblings maps cpu id -> its thread_siblings_list, so an SMT box is
    expressed as {0: [0, 4], 4: [0, 4], ...}.
    """
    root = tmp_path / "sys_cpu"
    for cpu, sibs in siblings.items():
        topo = root / f"cpu{cpu}" / "topology"
        topo.mkdir(parents=True, exist_ok=True)
        (topo / "thread_siblings_list").write_text(
            ",".join(str(s) for s in sibs), encoding="utf-8",
        )
        pkg = (packages or {}).get(cpu, 0)
        (topo / "physical_package_id").write_text(str(pkg), encoding="utf-8")
        if governor is not None:
            freq = root / f"cpu{cpu}" / "cpufreq"
            freq.mkdir(parents=True, exist_ok=True)
            (freq / "scaling_governor").write_text(governor, encoding="utf-8")
    if no_turbo is not None:
        (root / "intel_pstate").mkdir(parents=True, exist_ok=True)
        (root / "intel_pstate" / "no_turbo").write_text(no_turbo, encoding="utf-8")
    if boost is not None:
        (root / "cpufreq").mkdir(parents=True, exist_ok=True)
        (root / "cpufreq" / "boost").write_text(boost, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# 1. CPU list parsing
# ---------------------------------------------------------------------------

class TestParseCpuList:
    @pytest.mark.parametrize("text,expected", [
        ("0", [0]),
        ("0-3", [0, 1, 2, 3]),
        ("2,4,6", [2, 4, 6]),
        ("0-1,4-5", [0, 1, 4, 5]),
        (" 2 , 3 ", [2, 3]),
        ("", []),
    ])
    def test_valid_forms(self, text, expected):
        assert _parse_cpu_list(text) == expected

    @pytest.mark.parametrize("text", ["garbage", "3-1", "a-b", "-", ",,", "-5"])
    def test_malformed_fragments_are_dropped_not_raised(self, text):
        """A typo in perflab.yaml must degrade, not abort the run."""
        assert _parse_cpu_list(text) == []

    def test_mixed_valid_and_invalid(self):
        assert _parse_cpu_list("1,oops,3-4") == [1, 3, 4]


# ---------------------------------------------------------------------------
# 2. Physical core selection policy
# ---------------------------------------------------------------------------

class TestPhysicalCoreSelection:
    def test_smt_siblings_collapse_to_one_core(self, tmp_path, monkeypatch):
        # 4 logical CPUs, 2 physical cores: (0,2) and (1,3).
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={0: [0, 2], 2: [0, 2], 1: [1, 3], 3: [1, 3]},
        ))
        cores = _physical_cores({0, 1, 2, 3})
        assert len(cores) == 2
        assert sorted(rep for _, rep, _ in cores) == [0, 1]

    def test_pinning_never_picks_two_siblings_of_one_core(self, tmp_path, monkeypatch):
        """The whole point of reading thread_siblings_list: a naive 'first N
        cpus' pick would return (0, 1) here, which on this topology is one
        physical core twice."""
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path,
            siblings={0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3],
                      4: [4, 5], 5: [4, 5]},
        ))
        chosen, _ = _select_cpus({0, 1, 2, 3, 4, 5}, 2)
        assert chosen == (2, 4)  # two distinct physical cores, core 0 avoided

    def test_core_zero_is_avoided(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={c: [c] for c in range(4)},
        ))
        chosen, reason = _select_cpus({0, 1, 2, 3}, None)
        assert 0 not in chosen
        assert chosen == (1, 2, 3)
        assert "core 0 excluded" in reason

    def test_core_zero_sibling_is_also_avoided(self, tmp_path, monkeypatch):
        """Excluding cpu 0 is not enough -- cpu 0's SMT sibling shares the same
        physical core and the same IRQ pressure."""
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]},
        ))
        chosen, _ = _select_cpus({0, 1, 2, 3}, None)
        assert chosen == (2,)

    def test_core_zero_reinstated_when_too_few_cores(self, tmp_path, monkeypatch):
        """A single-core box must still get a plan rather than nothing."""
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={0: [0]},
        ))
        chosen, reason = _select_cpus({0}, None)
        assert chosen == (0,)
        assert "core 0 included" in reason

    def test_prefers_a_single_package(self, tmp_path, monkeypatch):
        """Straddling sockets is itself a noise source; a job that fits on one
        socket should stay there."""
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path,
            siblings={c: [c] for c in range(6)},
            packages={0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1},
        ))
        chosen, reason = _select_cpus(set(range(6)), None)
        assert chosen == (2, 3, 4, 5)  # package 1: the larger one
        assert "package 1 of 2" in reason

    def test_requested_core_count_is_honored(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={c: [c] for c in range(8)},
        ))
        chosen, reason = _select_cpus(set(range(8)), 3)
        assert len(chosen) == 3
        assert 0 not in chosen
        assert "3 physical cores" in reason

    def test_request_larger_than_machine_is_clamped_and_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={c: [c] for c in range(4)},
        ))
        chosen, reason = _select_cpus(set(range(4)), 99)
        assert len(chosen) == 4
        assert "only 4 available" in reason

    def test_respects_the_inherited_affinity_mask(self, tmp_path, monkeypatch):
        """A container with --cpuset-cpus already narrows sched_getaffinity;
        we must never hand back a CPU outside it."""
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={c: [c] for c in range(8)},
        ))
        chosen, _ = _select_cpus({5, 6, 7}, None)
        assert set(chosen) <= {5, 6, 7}

    def test_representative_is_a_cpu_inside_the_mask(self, tmp_path, monkeypatch):
        """When only the high sibling of a core is available, the core is still
        usable -- via that sibling, not via the masked-out low one."""
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]},
        ))
        cores = _physical_cores({1, 3})
        assert sorted(rep for _, rep, _ in cores) == [1, 3]

    def test_missing_topology_treats_each_cpu_as_a_core(self, tmp_path, monkeypatch):
        """VMs and containers often expose no topology at all."""
        monkeypatch.setattr(shell, "_SYS_CPU", tmp_path / "does_not_exist")
        cores = _physical_cores({0, 1, 2})
        assert sorted(rep for _, rep, _ in cores) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 3. Plan resolution
# ---------------------------------------------------------------------------

class TestPlanResolution:
    @pytest.mark.parametrize("spec", ["off", "none", "false", "no", "0", "disabled",
                                      "OFF", " off "])
    def test_off_spellings_disable(self, spec):
        plan = _build_cpu_plan(spec)
        assert not plan.enabled
        assert plan.cpus == ()

    def test_non_linux_degrades_with_a_reason_and_never_raises(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        plan = _build_cpu_plan("auto")
        assert not plan.enabled
        assert "Darwin" in plan.reason
        assert "sched_setaffinity" in plan.reason

    def test_windows_degrades_too(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        assert not _build_cpu_plan("2-5").enabled

    @linux_only
    def test_auto_selects_cpus(self):
        plan = _build_cpu_plan("auto")
        assert plan.enabled
        assert set(plan.cpus) <= set(os.sched_getaffinity(0))

    @linux_only
    def test_integer_spec_requests_that_many_cores(self):
        plan = _build_cpu_plan("1")
        assert len(plan.cpus) == 1

    @linux_only
    def test_explicit_list_is_intersected_with_the_affinity_mask(self):
        available = sorted(os.sched_getaffinity(0))
        if len(available) < 2:
            pytest.skip("needs >= 2 available CPUs")
        spec = f"{available[0]},{available[1]},9999"
        plan = _build_cpu_plan(spec)
        assert plan.cpus == (available[0], available[1])
        assert "outside the affinity mask dropped" in plan.reason

    @linux_only
    def test_explicit_list_entirely_outside_the_mask_degrades(self):
        plan = _build_cpu_plan("9990-9999")
        assert not plan.enabled
        assert "no CPU inside this process's affinity mask" in plan.reason

    @linux_only
    def test_unparseable_spec_degrades_instead_of_raising(self):
        plan = _build_cpu_plan("!!!nonsense!!!")
        assert not plan.enabled

    def test_resolution_is_cached(self, monkeypatch):
        calls = []

        def _spy(spec):
            calls.append(spec)
            return CpuPlan(reason="stub")

        monkeypatch.setattr(shell, "_build_cpu_plan", _spy)
        resolve_cpu_plan()
        resolve_cpu_plan()
        resolve_cpu_plan()
        assert len(calls) == 1, "plan must be resolved once per process"

    def test_env_var_outranks_task_and_config(self, monkeypatch):
        monkeypatch.setenv("PERFLAB_CPU_PINNING", "off")
        set_task_cpu_pinning("auto")
        assert shell._pinning_spec() == "off"

    def test_task_setting_outranks_config(self, monkeypatch):
        set_task_cpu_pinning("2-3")
        assert shell._pinning_spec() == "2-3"

    def test_config_default_when_nothing_else_set(self):
        assert shell._pinning_spec() == "auto"

    def test_changing_the_task_setting_invalidates_the_cache(self, monkeypatch):
        monkeypatch.setattr(shell, "_SYS_CPU", Path("/nonexistent-for-test"))
        resolve_cpu_plan()
        assert shell._cached_cpu_plan is not None
        set_task_cpu_pinning("off")
        assert shell._cached_cpu_plan is None

    def test_task_spec_publishes_constraints_cpu_pinning(self, tmp_path):
        """The task.yaml setting must actually reach the runners."""
        from perflab.task_spec import TaskSpec

        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text(textwrap.dedent("""
            name: t
            program_type: python
            correctness: {cmd: "python3 tests.py"}
            benchmark:
              cmd: "python3 bench.py"
              metric: {name: "ok"}
            constraints:
              cpu_pinning: "off"
        """), encoding="utf-8")
        spec = TaskSpec.load(task_yaml)
        assert spec.constraints.cpu_pinning == "off"
        assert shell._task_cpu_pinning == "off"
        assert not resolve_cpu_plan().enabled

    def test_task_without_the_setting_clears_the_override(self, tmp_path):
        from perflab.task_spec import TaskSpec

        set_task_cpu_pinning("off")
        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text(textwrap.dedent("""
            name: t
            program_type: python
            correctness: {cmd: "python3 tests.py"}
            benchmark:
              cmd: "python3 bench.py"
              metric: {name: "ok"}
        """), encoding="utf-8")
        spec = TaskSpec.load(task_yaml)
        assert spec.constraints.cpu_pinning == "auto"
        assert shell._task_cpu_pinning is None

    def test_config_section_accepts_cpu_pinning(self, tmp_path, monkeypatch):
        from perflab.config import PerfLabConfig, _overlay_yaml

        cfg = PerfLabConfig()
        assert cfg.benchmark.cpu_pinning == "auto"
        _overlay_yaml(cfg, {"benchmark": {"cpu_pinning": "2-5"}})
        assert cfg.benchmark.cpu_pinning == "2-5"

    def test_config_yaml_bare_off_survives_as_text(self):
        """PyYAML turns a bare `off` into False; it must still disable."""
        from perflab.config import PerfLabConfig, _overlay_yaml

        cfg = PerfLabConfig()
        _overlay_yaml(cfg, {"benchmark": {"cpu_pinning": False}})
        assert cfg.benchmark.cpu_pinning.lower() in shell._PINNING_OFF


# ---------------------------------------------------------------------------
# 4. Baseline/candidate parity -- THE correctness requirement
# ---------------------------------------------------------------------------

_RUNNER_FUNCS = {"run_benchmark", "run_correctness", "run_correctness_twice"}
_PINNING_KWARGS = {"cpu_pinning", "cpu_affinity", "cpus", "nice_adj", "affinity"}


class TestBaselineCandidateParity:
    """If the baseline were measured on a different core set than the
    candidates, every candidate would look faster (or slower) for free -- a
    systematic bias strictly worse than the noise pinning removes.

    The design defends this structurally: the plan is a process-wide singleton
    and the runners read it themselves, so there is no per-call knob any call
    site could set differently. These tests hold that door shut.
    """

    def test_runners_expose_no_per_call_pinning_override(self):
        """A per-call parameter is exactly the mechanism by which the baseline
        and the candidates could diverge. There must not be one."""
        from perflab.runners.benchmark import run_benchmark
        from perflab.runners.correctness import run_correctness, run_correctness_twice

        for func in (run_benchmark, run_correctness, run_correctness_twice):
            params = set(inspect.signature(func).parameters)
            leaked = params & _PINNING_KWARGS
            assert not leaked, (
                f"{func.__name__} exposes {leaked}: a per-call CPU-environment "
                f"override lets the baseline and candidates be measured "
                f"differently. Resolve through shell.resolve_cpu_plan() instead."
            )

    def test_no_call_site_passes_a_pinning_argument(self):
        """Static sweep of every run_benchmark/run_correctness call in the
        package: all of them must inherit the single process-wide policy."""
        offenders: list[str] = []
        for py in sorted((REPO_ROOT / "perflab").rglob("*.py")):
            if "demo_tasks" in py.parts:
                continue
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute)
                    else None
                )
                if name not in _RUNNER_FUNCS:
                    continue
                for kw in node.keywords:
                    if kw.arg in _PINNING_KWARGS:
                        offenders.append(
                            f"{py.relative_to(REPO_ROOT)}:{node.lineno} "
                            f"{name}({kw.arg}=...)"
                        )
        assert not offenders, (
            "these call sites override the CPU environment per call, so the "
            "baseline and candidates can be measured differently: "
            + "; ".join(offenders)
        )

    def test_all_runner_call_sites_are_accounted_for(self):
        """Guard against the sweep above silently matching nothing (e.g. after
        a rename) and passing vacuously."""
        found = 0
        for py in sorted((REPO_ROOT / "perflab").rglob("*.py")):
            if "demo_tasks" in py.parts:
                continue
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = (
                        func.id if isinstance(func, ast.Name)
                        else func.attr if isinstance(func, ast.Attribute)
                        else None
                    )
                    if name in _RUNNER_FUNCS:
                        found += 1
        assert found >= 8, f"expected the agent/pipeline/ci call sites, found {found}"

    def test_benchmark_and_correctness_read_the_same_plan(self, monkeypatch):
        """Both runners must consult resolve_cpu_plan(), not build their own."""
        from perflab.runners import benchmark as benchmark_mod
        from perflab.runners import correctness as correctness_mod

        sentinel = CpuPlan(cpus=(3, 5), omp_env={"OMP_PROC_BIND": "close"}, nice_adj=None)
        seen: list[tuple[int, ...]] = []

        def _fake_run_cmd(cmd, **kwargs):
            seen.append(tuple(kwargs.get("cpu_affinity") or ()))
            raise _StopRun

        class _StopRun(Exception):
            pass

        monkeypatch.setattr(benchmark_mod, "resolve_cpu_plan", lambda: sentinel)
        monkeypatch.setattr(correctness_mod, "resolve_cpu_plan", lambda: sentinel)
        monkeypatch.setattr(benchmark_mod, "run_cmd", _fake_run_cmd)
        monkeypatch.setattr(correctness_mod, "run_cmd", _fake_run_cmd)

        with pytest.raises(_StopRun):
            benchmark_mod.run_benchmark("true", cwd=Path("."))
        with pytest.raises(_StopRun):
            correctness_mod.run_correctness("true", cwd=Path("."))

        assert seen == [(3, 5), (3, 5)]

    @linux_only
    def test_baseline_and_candidate_children_land_on_identical_cores(self, tmp_path):
        """End to end, with real subprocesses: the affinity mask a 'baseline'
        benchmark run sees is byte-identical to what a later 'candidate' run
        sees, including across a fast-screen run and a correctness run."""
        from perflab.runners.benchmark import run_benchmark
        from perflab.runners.correctness import run_correctness

        ws = tmp_path / "ws"
        (ws / "out").mkdir(parents=True)
        # "t" varies per run: run_benchmark's anti-tamper check rejects a
        # bench.json whose contents are byte-identical to the previous run's.
        (ws / "bench.py").write_text(textwrap.dedent("""
            import json, os, pathlib, time
            pathlib.Path("out").mkdir(exist_ok=True)
            pathlib.Path("out/bench.json").write_text(json.dumps({
                "ok": True,
                "t": time.time_ns(),
                "affinity": sorted(os.sched_getaffinity(0)),
                "omp": {k: v for k, v in os.environ.items() if k.startswith("OMP_")},
            }))
        """), encoding="utf-8")
        (ws / "tests.py").write_text(
            "import os;print('AFF', sorted(os.sched_getaffinity(0)))\n", encoding="utf-8",
        )

        _, baseline = run_benchmark("python3 bench.py", cwd=ws)
        _, candidate = run_benchmark("python3 bench.py", cwd=ws, fast_mode=True)
        _, rebench = run_benchmark("python3 bench.py", cwd=ws)
        cres = run_correctness("python3 tests.py", cwd=ws)

        assert baseline["affinity"] == candidate["affinity"] == rebench["affinity"]
        assert f"AFF {baseline['affinity']}" in cres.stdout

        plan = resolve_cpu_plan()
        if plan.enabled:
            assert baseline["affinity"] == list(plan.cpus)
            assert baseline["omp"]["OMP_PROC_BIND"] == "close"
            assert baseline["omp"]["OMP_PLACES"] == "cores"


# ---------------------------------------------------------------------------
# 5. Governor / turbo detection and reporting
# ---------------------------------------------------------------------------

class TestGovernorReporting:
    def test_powersave_governor_is_detected_and_a_fix_offered(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={c: [c] for c in range(4)}, governor="powersave",
        ))
        governors, _turbo, advice = _cpu_freq_state()
        assert governors == ("powersave",)
        assert "sudo cpupower frequency-set -g performance" in advice

    def test_performance_governor_needs_no_advice(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={c: [c] for c in range(4)}, governor="performance",
        ))
        governors, _turbo, advice = _cpu_freq_state()
        assert governors == ("performance",)
        assert advice == ()

    def test_mixed_governors_are_all_reported(self, tmp_path, monkeypatch):
        root = _fake_sysfs(tmp_path, siblings={c: [c] for c in range(2)},
                           governor="performance")
        (root / "cpu1" / "cpufreq" / "scaling_governor").write_text(
            "powersave", encoding="utf-8",
        )
        monkeypatch.setattr(shell, "_SYS_CPU", root)
        governors, _turbo, advice = _cpu_freq_state()
        assert governors == ("performance", "powersave")
        assert advice

    def test_intel_pstate_turbo_on_is_detected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={0: [0]}, no_turbo="0",
        ))
        _governors, turbo, advice = _cpu_freq_state()
        assert turbo == "on"
        assert any("no_turbo" in a for a in advice)

    def test_intel_pstate_turbo_off_needs_no_advice(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={0: [0]}, no_turbo="1",
        ))
        _governors, turbo, advice = _cpu_freq_state()
        assert turbo == "off"
        assert advice == ()

    def test_generic_cpufreq_boost_is_detected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={0: [0]}, boost="1",
        ))
        _governors, turbo, advice = _cpu_freq_state()
        assert turbo == "on"
        assert any("cpufreq/boost" in a for a in advice)

    def test_absent_cpufreq_reports_nothing_rather_than_guessing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shell, "_SYS_CPU", tmp_path / "no_such_tree")
        assert _cpu_freq_state() == ((), None, ())

    def test_governor_is_reported_even_when_pinning_is_off(self, tmp_path, monkeypatch):
        """The frequency finding is independent of pinning -- turning pinning
        off must not hide it."""
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={c: [c] for c in range(2)}, governor="powersave",
        ))
        plan = _build_cpu_plan("off")
        assert not plan.enabled
        assert plan.governors == ("powersave",)
        assert plan.advice

    def test_never_applies_the_fix_itself(self, tmp_path, monkeypatch):
        """PerfLab refuses to change a machine-wide setting on the user's
        behalf -- same stance as profilers/base.py declining to auto-sudo. The
        governor file must be untouched after a full plan resolution."""
        root = _fake_sysfs(tmp_path, siblings={0: [0]}, governor="powersave")
        gov_file = root / "cpu0" / "cpufreq" / "scaling_governor"
        monkeypatch.setattr(shell, "_SYS_CPU", root)
        before = gov_file.read_text(encoding="utf-8")
        resolve_cpu_plan(force=True)
        assert gov_file.read_text(encoding="utf-8") == before == "powersave"

    def test_warning_names_the_exact_command(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(shell, "_SYS_CPU", _fake_sysfs(
            tmp_path, siblings={c: [c] for c in range(2)}, governor="powersave",
        ))
        with caplog.at_level("WARNING", logger="perflab.tools.shell"):
            resolve_cpu_plan(force=True)
        text = caplog.text
        assert "powersave" in text
        assert "sudo cpupower frequency-set -g performance" in text


# ---------------------------------------------------------------------------
# 6. nice(2) capability
# ---------------------------------------------------------------------------

class TestRenice:
    def test_unprivileged_default_limits_mean_no_renice(self, monkeypatch):
        """Without root or a raised RLIMIT_NICE, nice(2) can only *lower*
        priority, which would make the measurement worse. We skip it rather
        than pretend."""
        import resource

        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        # RLIMIT_NICE is Linux-only; supply it so the arithmetic under test
        # runs on the macOS dev box too.
        monkeypatch.setattr(resource, "RLIMIT_NICE", 13, raising=False)
        monkeypatch.setattr(resource, "getrlimit", lambda which: (0, 0))
        assert _can_renice() is False

    def test_raised_rlimit_nice_permits_renice(self, monkeypatch):
        import resource

        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        # soft=30 -> nice floor of 20-30 = -10, below _BENCH_NICE_ADJ (-5).
        monkeypatch.setattr(resource, "RLIMIT_NICE", 13, raising=False)
        monkeypatch.setattr(resource, "getrlimit", lambda which: (30, 30))
        assert _can_renice() is True

    def test_marginal_rlimit_nice_is_rejected(self, monkeypatch):
        import resource

        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        # soft=22 -> floor of -2, not low enough for -5.
        monkeypatch.setattr(resource, "RLIMIT_NICE", 13, raising=False)
        monkeypatch.setattr(resource, "getrlimit", lambda which: (22, 22))
        assert _can_renice() is False

    def test_root_may_renice(self, monkeypatch):
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        assert _can_renice() is True

    def test_plan_omits_nice_when_not_permitted(self, monkeypatch):
        monkeypatch.setattr(shell, "_can_renice", lambda: False)
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(shell.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3},
                            raising=False)
        plan = _build_cpu_plan("auto")
        assert plan.nice_adj is None

    def test_plan_sets_a_negative_nice_when_permitted(self, monkeypatch):
        monkeypatch.setattr(shell, "_can_renice", lambda: True)
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(shell.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3},
                            raising=False)
        plan = _build_cpu_plan("auto")
        assert plan.nice_adj is not None and plan.nice_adj < 0


# ---------------------------------------------------------------------------
# 7. OpenMP binding env
# ---------------------------------------------------------------------------

class TestOpenMPBinding:
    @linux_only
    def test_plan_sets_proc_bind_and_places(self, monkeypatch):
        monkeypatch.delenv("OMP_PROC_BIND", raising=False)
        monkeypatch.delenv("OMP_PLACES", raising=False)
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        plan = _build_cpu_plan("auto")
        assert plan.omp_env["OMP_PROC_BIND"] == "close"
        assert plan.omp_env["OMP_PLACES"] == "cores"
        assert plan.omp_env["OMP_NUM_THREADS"] == str(len(plan.cpus))

    @linux_only
    def test_an_explicit_user_setting_is_left_alone(self, monkeypatch):
        """`export OMP_NUM_THREADS=2` is a deliberate choice; PerfLab does not
        silently override it."""
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        plan = _build_cpu_plan("auto")
        assert "OMP_NUM_THREADS" not in plan.omp_env

    def test_runner_uses_setdefault_so_the_caller_wins(self, monkeypatch, tmp_path):
        from perflab.runners import benchmark as benchmark_mod

        captured: dict = {}

        class _Stop(Exception):
            pass

        def _fake_run_cmd(cmd, **kwargs):
            captured.update(kwargs.get("env") or {})
            raise _Stop

        monkeypatch.setattr(
            benchmark_mod, "resolve_cpu_plan",
            lambda: CpuPlan(cpus=(1,), omp_env={"OMP_NUM_THREADS": "1",
                                                "OMP_PROC_BIND": "close"}),
        )
        monkeypatch.setattr(benchmark_mod, "run_cmd", _fake_run_cmd)
        with pytest.raises(_Stop):
            benchmark_mod.run_benchmark(
                "true", cwd=tmp_path, env={"OMP_NUM_THREADS": "8"},
            )
        assert captured["OMP_NUM_THREADS"] == "8"   # caller's value survives
        assert captured["OMP_PROC_BIND"] == "close"  # plan fills the rest

    def test_correctness_gets_affinity_but_not_the_omp_overlay(self, monkeypatch, tmp_path):
        """Correctness runs are confined to the same cores (so candidate test
        code cannot perturb a measurement) but are not timed, so the OMP
        binding overlay -- purely a measurement concern -- stays out of their
        environment. Keeping it out also leaves env=None when nothing else
        needs forwarding, which callers rely on."""
        from perflab.runners import correctness as correctness_mod

        captured: dict = {}

        class _Stop(Exception):
            pass

        def _fake_run_cmd(cmd, **kwargs):
            captured.update(kwargs)
            raise _Stop

        monkeypatch.setattr(
            correctness_mod, "resolve_cpu_plan",
            lambda: CpuPlan(cpus=(1,), omp_env={"OMP_PLACES": "cores"}),
        )
        monkeypatch.setattr(correctness_mod, "run_cmd", _fake_run_cmd)
        with pytest.raises(_Stop):
            correctness_mod.run_correctness("true", cwd=tmp_path)
        assert captured["cpu_affinity"] == (1,)
        assert captured["env"] is None


# ---------------------------------------------------------------------------
# 8. Linux acceptance: the affinity actually reaches the child
# ---------------------------------------------------------------------------

@linux_only
class TestAffinityAcceptance:
    def test_run_cmd_pins_the_child(self):
        available = sorted(os.sched_getaffinity(0))
        target = available[-1]
        res = run_cmd(
            ["python3", "-c",
             "import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))"],
            cpu_affinity=[target],
        )
        assert res.returncode == 0, res.stderr
        assert json.loads(res.stdout.strip()) == [target]
        assert res.cpu_env_applied is True

    def test_pinning_is_inherited_by_threads(self):
        """OpenMP/BLAS parallelism is threads, and affinity is per-thread on
        Linux -- a thread that did not inherit the mask would run off the
        pinned cores and undo the whole point.

        rlimit_nproc headroom: threads count against RLIMIT_NPROC, which the
        kernel tracks per real UID machine-wide rather than per container, so
        run_cmd's 512 default can leave no room for even one extra thread on a
        shared-UID host. See test_pinning_survives_into_grandchildren.
        """
        available = sorted(os.sched_getaffinity(0))
        target = available[-1]
        script = (
            "import os,json,threading;"
            "out=[];"
            "t=threading.Thread(target=lambda: out.append("
            "sorted(os.sched_getaffinity(0))));"
            "t.start();t.join();print(json.dumps(out[0]))"
        )
        res = run_cmd(["python3", "-c", script], cpu_affinity=[target],
                      rlimit_nproc=4096)
        assert res.returncode == 0, res.stderr
        assert json.loads(res.stdout.strip()) == [target]

    def test_pinning_survives_into_grandchildren(self):
        """A benchmark that forks workers must inherit the mask too --
        otherwise pinning the launcher buys nothing.

        rlimit_nproc is raised for this case only: the point under test is
        affinity inheritance, and run_cmd's default fork guard (512) is
        counted per real UID kernel-wide, so on a shared-UID host (Docker
        Desktop's Linux VM) the deliberate extra process cannot spawn and the
        test would fail for a reason unrelated to pinning.
        """
        available = sorted(os.sched_getaffinity(0))
        if len(available) < 2:
            pytest.skip("needs >= 2 available CPUs")
        target = available[-1]
        script = (
            "import os,json,subprocess,sys;"
            "print(subprocess.run([sys.executable,'-c',"
            "\"import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))\"],"
            "capture_output=True,text=True).stdout.strip())"
        )
        res = run_cmd(["python3", "-c", script], cpu_affinity=[target],
                      rlimit_nproc=4096)
        assert res.returncode == 0, res.stderr
        assert json.loads(res.stdout.strip()) == [target]

    def test_no_affinity_requested_leaves_the_child_alone(self):
        res = run_cmd(
            ["python3", "-c",
             "import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))"],
        )
        assert res.returncode == 0
        assert json.loads(res.stdout.strip()) == sorted(os.sched_getaffinity(0))
        assert res.cpu_env_applied is None

    def test_bad_cpu_id_is_reported_not_silently_ignored(self, caplog):
        """A run that was supposed to be pinned but wasn't is a mismeasurement;
        it must surface, and the marker must not pollute the child's stderr."""
        with caplog.at_level("WARNING", logger="perflab.tools.shell"):
            res = run_cmd(["python3", "-c", "print('hi')"], cpu_affinity=[9999])
        assert res.returncode == 0
        assert res.cpu_env_applied is False
        assert "perflab-cpuenv-failed" not in res.stderr
        assert "NOT" in caplog.text and "pinning policy" in caplog.text

    def test_rlimit_reporting_is_unaffected_by_the_cpu_marker(self):
        res = run_cmd(["python3", "-c", "print('hi')"], cpu_affinity=list(
            sorted(os.sched_getaffinity(0))[:1]))
        assert res.rlimits_applied is True
        assert res.cpu_env_applied is True

    def test_skip_preexec_skips_pinning(self):
        """Documented interaction: prescreen runs correctness from a thread
        pool, where preexec_fn + fork is unsafe. Nothing is measured there."""
        available = sorted(os.sched_getaffinity(0))
        res = run_cmd(
            ["python3", "-c",
             "import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))"],
            cpu_affinity=[available[-1]], skip_preexec=True,
        )
        assert json.loads(res.stdout.strip()) == available
        assert res.cpu_env_applied is None
