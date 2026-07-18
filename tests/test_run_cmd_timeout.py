"""Tests for run_cmd default timeout + TimeoutExpired handling, and the
non-interactive sudo (-n) invocation in roofline_peaks.

Covers:
  1. perflab.tools.shell.run_cmd -- DEFAULT_TIMEOUT_S is the signature
     default, a wedged subprocess is killed on timeout and surfaced as a
     failure CmdResult (returncode=TIMEOUT_RETURNCODE, clear stderr
     message) rather than an unhandled subprocess.TimeoutExpired.
  2. timeout_s=None still means "no timeout" (explicit opt-out).
  3. perflab.roofline_peaks -- dmidecode probing uses `sudo -n` (never
     blocks on a password prompt) and degrades gracefully to the
     conservative bandwidth fallback when sudo credentials are absent.
"""
from __future__ import annotations

import inspect
import time

import perflab.roofline_peaks as roofline_peaks
import perflab.tools.shell as shell

# ---------------------------------------------------------------------------
# 1. Default timeout wiring
# ---------------------------------------------------------------------------

class TestDefaultTimeout:
    def test_signature_default_is_module_constant(self):
        sig = inspect.signature(shell.run_cmd)
        assert sig.parameters["timeout_s"].default == shell.DEFAULT_TIMEOUT_S
        assert shell.DEFAULT_TIMEOUT_S == 600

    def test_normal_command_unaffected(self):
        res = shell.run_cmd(["python3", "-c", "print('ok')"])
        assert res.returncode == 0
        assert "ok" in res.stdout


class TestTimeoutExpiredHandling:
    def test_short_timeout_kills_sleeping_subprocess(self):
        t0 = time.time()
        res = shell.run_cmd(
            ["python3", "-c", "import time; time.sleep(60)"], timeout_s=1,
        )
        elapsed = time.time() - t0
        # Must not have waited for the full sleep -- child was killed.
        assert elapsed < 30
        assert res.returncode == shell.TIMEOUT_RETURNCODE
        assert "timed out after 1s" in res.stderr
        assert "[perflab-timeout]" in res.stderr
        # Failure is a CmdResult, matching how other failed runs surface.
        assert isinstance(res, shell.CmdResult)
        assert isinstance(res.stdout, str)
        assert res.duration_s >= 1
        assert res.rlimits_applied is None

    def test_explicit_none_disables_timeout(self):
        res = shell.run_cmd(["python3", "-c", "print('no-timeout')"], timeout_s=None)
        assert res.returncode == 0
        assert "no-timeout" in res.stdout


# ---------------------------------------------------------------------------
# 3. roofline_peaks: sudo -n for dmidecode, graceful degradation
# ---------------------------------------------------------------------------

class TestRooflineSudoNonInteractive:
    def _drive_linux_cpu_peaks(self, monkeypatch, run_impl):
        """Run _estimate_cpu_peaks on a simulated Linux box with a fake _run."""
        monkeypatch.setattr(roofline_peaks.platform, "system", lambda: "Linux")
        monkeypatch.setattr(roofline_peaks, "_run", run_impl)
        return roofline_peaks._estimate_cpu_peaks()

    def test_dmidecode_probes_use_sudo_dash_n(self, monkeypatch):
        seen = []

        def fake_run(cmd):
            script = cmd[-1]
            seen.append(script)
            if "model name" in script:
                return "FakeCPU 9000"
            if "CPU max MHz" in script:
                return "3000"
            if "flags" in script:
                return "avx2\nfma"
            if "dmidecode" in script:
                return "3200" if "speed:" in script else "2"
            return None

        peaks = self._drive_linux_cpu_peaks(monkeypatch, fake_run)
        sudo_scripts = [s for s in seen if "sudo" in s]
        assert sudo_scripts, "expected dmidecode probes to run"
        for script in sudo_scripts:
            assert "sudo -n " in script, f"plain sudo (may block on prompt): {script}"
        # With dmidecode data available, bandwidth comes from it.
        assert peaks is not None
        assert peaks.peak_mem_bw_gbs == (3200 * 8 * 2) / 1000.0

    def test_sudo_failure_falls_back_to_conservative_bandwidth(self, monkeypatch):
        def fake_run(cmd):
            script = cmd[-1]
            if "model name" in script:
                return "FakeCPU 9000"
            if "CPU max MHz" in script:
                return "3000"
            if "flags" in script:
                return "avx2\nfma"
            # sudo -n with no cached credentials: non-zero exit -> _run None
            if "dmidecode" in script:
                return None
            return None

        peaks = self._drive_linux_cpu_peaks(monkeypatch, fake_run)
        assert peaks is not None
        assert peaks.peak_mem_bw_gbs == 50.0

    def test_run_helper_returns_none_on_nonzero_exit(self):
        # sudo -n without cached credentials exits non-zero; _run must
        # swallow that (SubprocessError) rather than raising.
        assert roofline_peaks._run(["bash", "-c", "exit 1"]) is None
