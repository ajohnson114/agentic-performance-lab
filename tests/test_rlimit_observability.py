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
# 5. skip_preexec's ulimit-shell fallback (prescreen's ThreadPoolExecutor path)
# ---------------------------------------------------------------------------
#
# skip_preexec=True previously meant NO resource limit at all -- preexec_fn
# can't be used from a thread (preexec_fn + fork() in a multithreaded process
# is undefined behavior), so it was skipped entirely rather than degraded.
# That left prescreen's parallel candidate build+correctness (untrusted,
# LLM-authored code) running with zero memory ceiling. _rlimit_shell_wrap
# closes that gap via a bash `ulimit` shim, which needs no preexec_fn (so
# Popen can use posix_spawn, never forking the parent at all).
#
# Real enforcement can't be proven on this dev box: _rlimit_shell_wrap (like
# _make_linux_preexec) is gated to Linux only, and forcing platform.system()
# to report "Linux" doesn't change the real underlying kernel -- macOS's own
# bash rejects `ulimit -v` outright ("cannot modify limit: Invalid
# argument"), it just doesn't abort the script. So these tests cover the
# wrap's shape and that it doesn't break normal execution, not OS-level
# enforcement (that's CI/docker-dev-container territory, like the bwrap/
# seccomp acceptance tests).

class TestSkipPreexecRlimitShellWrap:
    def test_wrap_shape_includes_as_nproc_nofile(self, monkeypatch):
        monkeypatch.setattr(shell.platform, "system", lambda: "Linux")
        wrapped = shell._rlimit_shell_wrap(["echo", "hi"], 4 * 1024**3, 512)
        assert wrapped[:2] == ["bash", "-c"]
        script = wrapped[2]
        assert f"ulimit -v {4 * 1024**2}" in script  # bytes -> KiB
        assert "ulimit -u 512" in script
        assert "ulimit -n 1024" in script
        assert 'exec "$@"' in script
        assert wrapped[3:] == ["bash", "echo", "hi"]

    def test_wrap_omits_as_limit_when_rlimit_as_bytes_is_none(self, monkeypatch):
        # None means "explicitly disabled" (see _resolve_rlimit) -- must not
        # silently impose some other AS ceiling.
        monkeypatch.setattr(shell.platform, "system", lambda: "Linux")
        wrapped = shell._rlimit_shell_wrap(["echo", "hi"], None, 512)
        script = wrapped[2]
        assert "ulimit -v" not in script
        assert "ulimit -u 512" in script

    def test_non_linux_leaves_cmd_unwrapped(self, monkeypatch):
        monkeypatch.setattr(shell.platform, "system", lambda: "Darwin")
        wrapped = shell._rlimit_shell_wrap(["echo", "hi"], 4 * 1024**3, 512)
        assert wrapped == ["echo", "hi"]

    def test_run_cmd_skip_preexec_still_wraps_and_runs_correctly_on_linux(self, monkeypatch):
        # End-to-end through run_cmd: platform forced to "Linux" so the wrap
        # activates, proving it doesn't corrupt argv/output on a real
        # (unpatched-kernel) subprocess run -- see the module note above for
        # why this can't assert real memory-ceiling enforcement here.
        monkeypatch.setattr(shell.platform, "system", lambda: "Linux")
        res = shell.run_cmd(
            ["python3", "-c", "print('wrapped-ok')"],
            rlimit_as_bytes=4 * 1024**3,
            skip_preexec=True,
        )
        assert res.returncode == 0
        assert "wrapped-ok" in res.stdout

    def test_run_cmd_skip_preexec_non_linux_runs_unwrapped(self, monkeypatch):
        monkeypatch.setattr(shell.platform, "system", lambda: "Darwin")
        res = shell.run_cmd(
            ["python3", "-c", "print('unwrapped-ok')"],
            rlimit_as_bytes=4 * 1024**3,
            skip_preexec=True,
        )
        assert res.returncode == 0
        assert "unwrapped-ok" in res.stdout


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
