"""Tests for Fix 4: observable rlimit application (CmdResult.rlimits_applied).

Covers:
  1. perflab.tools.shell.run_cmd — rlimits_applied True/False/None semantics
  2. Marker text never leaks into CmdResult.stdout/stderr
  3. The narrowed (ValueError, OSError) except clause in _preexec doesn't
     swallow other exceptions (e.g. KeyboardInterrupt)
  4. perflab.optimizers.event_log.AgentEventLog.rlimit_warning + replay_events

Note: real rlimit contention (e.g. setrlimit(RLIMIT_NPROC, (1, 1)) while the
process already owns more procs than that) does not reliably raise on
Linux/macOS -- setrlimit() itself succeeds even when lowering the limit
below current usage; enforcement only kicks in on the *next* resource-
consuming syscall (fork/open). So the failure-path tests here mock
resource.setrlimit directly to deterministically simulate a rejected
limit (e.g. a hardened kernel policy), which is portable across dev
machines and CI. subprocess's preexec_fn runs post-fork in a COW child,
so a monkeypatch applied in the parent before the fork is still in effect
inside the child.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

import perflab.tools.shell as shell

# ---------------------------------------------------------------------------
# 1 & 2. rlimits_applied semantics + marker stripping
# ---------------------------------------------------------------------------

class TestRlimitsAppliedNotApplicable:
    def test_non_linux_platform_gives_none(self, monkeypatch):
        monkeypatch.setattr(shell.platform, "system", lambda: "Darwin")
        res = shell.run_cmd(["python3", "-c", "print('hi')"])
        assert res.rlimits_applied is None
        assert "hi" in res.stdout

    def test_skip_preexec_gives_none_even_on_linux(self, monkeypatch):
        monkeypatch.setattr(shell.platform, "system", lambda: "Linux")
        res = shell.run_cmd(["python3", "-c", "print('hi')"], skip_preexec=True)
        assert res.rlimits_applied is None


class TestRlimitsAppliedSuccess:
    def test_normal_invocation_true_and_no_marker_leak(self, monkeypatch):
        monkeypatch.setattr(shell.platform, "system", lambda: "Linux")
        # rlimit_as_bytes=None: RLIMIT_AS enforcement is notoriously
        # inconsistent across POSIX kernels (e.g. some macOS configurations
        # reject any RLIMIT_AS soft/hard pair outright) -- NPROC/NOFILE are
        # the portable ones to assert a clean apply against.
        res = shell.run_cmd(
            ["python3", "-c", "import sys; print('out1'); print('err1', file=sys.stderr)"],
            rlimit_as_bytes=None,
        )
        assert res.rlimits_applied is True
        assert "perflab-rlimit-failed" not in res.stdout
        assert "perflab-rlimit-failed" not in res.stderr
        assert "out1" in res.stdout
        assert "err1" in res.stderr


class TestRlimitsAppliedFailure:
    def test_setrlimit_failure_reported_stripped_and_logged(self, monkeypatch, caplog):
        monkeypatch.setattr(shell.platform, "system", lambda: "Linux")

        import resource

        def _boom(*_args, **_kwargs):
            raise OSError("simulated: rejected by kernel policy")

        monkeypatch.setattr(resource, "setrlimit", _boom)

        with caplog.at_level(logging.WARNING, logger="perflab.tools.shell"):
            res = shell.run_cmd(
                ["python3", "-c", "print('still runs')"], rlimit_nproc=1,
            )

        assert res.rlimits_applied is False
        assert res.returncode == 0
        assert "still runs" in res.stdout
        assert "perflab-rlimit-failed" not in res.stdout
        assert "perflab-rlimit-failed" not in res.stderr
        assert any("rlimit application failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 3. Narrowed exception clause: KeyboardInterrupt must not be swallowed
# ---------------------------------------------------------------------------

class TestPreexecExceptionNarrowing:
    def test_keyboard_interrupt_not_swallowed_by_preexec(self, monkeypatch):
        """If resource.setrlimit raises something other than (ValueError,
        OSError), _preexec must not catch it. subprocess.run then surfaces
        the (unhandled-by-us) preexec_fn exception to the parent as a
        SubprocessError, rather than run_cmd silently returning a normal
        CmdResult as if nothing happened.
        """
        monkeypatch.setattr(shell.platform, "system", lambda: "Linux")
        import resource

        def _boom(*_args, **_kwargs):
            raise KeyboardInterrupt("simulated")

        monkeypatch.setattr(resource, "setrlimit", _boom)

        with pytest.raises(subprocess.SubprocessError):
            shell.run_cmd(["python3", "-c", "print('x')"])


# ---------------------------------------------------------------------------
# 4. AgentEventLog.rlimit_warning + replay
# ---------------------------------------------------------------------------

class TestAgentEventLogRlimitWarning:
    def test_rlimit_warning_event_written_and_replayed(self, tmp_path: Path):
        from perflab.optimizers.event_log import AgentEventLog, replay_events

        log = AgentEventLog(run_dir=tmp_path)
        log.rlimit_warning(2, "rlimit failed for candidate 1 during benchmark", candidate_index=0)

        events_path = tmp_path / "agent_events.jsonl"
        assert events_path.exists()
        content = events_path.read_text(encoding="utf-8")
        assert '"event_type": "rlimit_warning"' in content
        assert "rlimit failed for candidate 1 during benchmark" in content

        replay = replay_events(tmp_path)
        assert "RLIMIT WARNING" in replay
        assert "rlimit failed for candidate 1 during benchmark" in replay
