"""Regression coverage: capture_system_info's JAX detection must never
import jax into the calling process.

Found on real 2x-H100 hardware: capture_system_info() used to call
jax.devices() in-process. jax.devices() initializes JAX's runtime
(background threads, heavier still with a real CUDA backend), leaving the
whole perflab process multithreaded for the rest of its life. Every
build/correctness/benchmark subprocess perflab spawns afterward goes
through run_cmd's preexec_fn (rlimit/cpu pinning), which forces classic
fork()+exec() instead of posix_spawn -- and fork() in a multithreaded
process only carries the calling thread into the child, so a lock held by
any other thread at fork time stays locked forever there. This reliably
broke the very next nvcc build in the same run with an opaque "Build failed
with code 1" and no matching stderr. Fixed by probing jax.devices() in a
throwaway subprocess instead.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from perflab.tools.sysinfo import capture_system_info


class TestJaxDetectionNeverImportsInProcess:
    def test_jax_not_in_sys_modules_after_capture(self):
        # A subprocess, not this test's own interpreter, so a prior test
        # importing jax elsewhere in the suite can't hide a regression here.
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys, tempfile\n"
             "from pathlib import Path\n"
             "from perflab.tools.sysinfo import capture_system_info\n"
             "capture_system_info(Path(tempfile.mkdtemp()))\n"
             "assert 'jax' not in sys.modules, 'jax leaked into the calling process'\n"
             "print('OK')\n"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_jax_version_still_populated(self, tmp_path: Path):
        info = capture_system_info(tmp_path)
        # Only asserts the field is populated when jax is actually
        # installed in this environment; a jax-less CI box is fine too.
        try:
            import jax  # noqa: F401
        except ImportError:
            return
        assert "jax_version" in info

    def test_subprocess_failure_does_not_raise(self, tmp_path: Path):
        with patch("subprocess.run", side_effect=OSError("no such file")):
            info = capture_system_info(tmp_path)
        assert "jax_version" not in info

    def test_malformed_subprocess_output_does_not_raise(self, tmp_path: Path):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="not json", stderr="")
        with patch("subprocess.run", return_value=fake):
            info = capture_system_info(tmp_path)
        assert "jax_version" not in info
