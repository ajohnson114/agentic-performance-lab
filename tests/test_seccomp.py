"""Unit tests for perflab/tools/seccomp.py (the strict-isolation BPF filter).

The centerpiece is a ~30-line classic-BPF interpreter that symbolically
executes the compiled program against every (arch, syscall) case we care
about. That makes "the filter denies what it claims to deny" a property
checked on every platform and every CI job -- not just where a Linux kernel
happens to enforce it. Real-kernel enforcement through bwrap is covered by
TestSeccompAcceptance in tests/test_isolation.py (CI's x86_64 runners; the
docker/ dev container for aarch64).
"""
from __future__ import annotations

import struct
from unittest.mock import patch

import pytest

from perflab.tools import seccomp
from perflab.tools.seccomp import (
    DENIED_SYSCALL_NAMES,
    SeccompUnavailableError,
    build_filter,
)

ALLOW = 0x7FFF0000
ERRNO_EPERM = 0x00050001
KILL_PROCESS = 0x80000000

ARCH_X86_64 = 0xC000003E
ARCH_AARCH64 = 0xC00000B7
X32_BIT = 0x40000000


def run_bpf(prog: bytes, *, nr: int, arch: int) -> int:
    """Execute a classic-BPF seccomp program against one syscall.

    Implements exactly the opcode subset the builder emits (W-sized absolute
    loads of seccomp_data, JEQ/JGE immediate jumps, RET immediate) and fails
    the test on anything else -- including reads outside the two seccomp_data
    fields the filter is supposed to look at, and running off the program end.
    """
    assert len(prog) % 8 == 0
    insns = [struct.unpack_from("<HBBI", prog, i * 8) for i in range(len(prog) // 8)]
    data = {0: nr & 0xFFFFFFFF, 4: arch & 0xFFFFFFFF}
    acc = 0
    pc = 0
    while pc < len(insns):
        code, jt, jf, k = insns[pc]
        if code == 0x20:  # BPF_LD | BPF_W | BPF_ABS
            assert k in data, f"load from unexpected seccomp_data offset {k}"
            acc = data[k]
        elif code == 0x15:  # BPF_JMP | BPF_JEQ | BPF_K
            pc += jt if acc == k else jf
        elif code == 0x35:  # BPF_JMP | BPF_JGE | BPF_K
            pc += jt if acc >= k else jf
        elif code == 0x06:  # BPF_RET | BPF_K
            return k
        else:
            pytest.fail(f"unexpected BPF opcode 0x{code:02x} at pc={pc}")
        pc += 1
    pytest.fail("BPF program ran off the end without returning")


# (machine string, audit arch value) pairs the builder supports.
ARCH_CASES = [("x86_64", ARCH_X86_64), ("aarch64", ARCH_AARCH64)]


class TestFilterSemantics:
    @pytest.mark.parametrize("machine,arch", ARCH_CASES)
    def test_benign_syscalls_allowed(self, machine, arch):
        prog = build_filter(machine)
        # read, write, close, mmap-ish, and a high never-assigned number:
        # everything not in the denied table must pass through untouched.
        for nr in (0, 1, 3, 9, 57, 220, 1000, 5000):
            if nr in set(seccomp._ARCHES[machine][1].values()):
                continue
            assert run_bpf(prog, nr=nr, arch=arch) == ALLOW, nr

    @pytest.mark.parametrize("machine,arch", ARCH_CASES)
    def test_every_denied_syscall_returns_eperm(self, machine, arch):
        prog = build_filter(machine)
        table = seccomp._ARCHES[machine][1]
        for name, nr in table.items():
            assert run_bpf(prog, nr=nr, arch=arch) == ERRNO_EPERM, name

    @pytest.mark.parametrize("machine,arch", ARCH_CASES)
    def test_foreign_arch_is_killed(self, machine, arch):
        """A syscall entering through a different ABI has numbers this
        filter's table knows nothing about -- it must die, not slip through."""
        prog = build_filter(machine)
        foreign = ARCH_AARCH64 if arch == ARCH_X86_64 else ARCH_X86_64
        # nr 0 is benign on the native arch; the arch check must fire first.
        assert run_bpf(prog, nr=0, arch=foreign) == KILL_PROCESS
        assert run_bpf(prog, nr=1, arch=0x40000003) == KILL_PROCESS  # i386

    def test_x32_bit_is_killed_on_x86_64(self):
        prog = build_filter("x86_64")
        # Both a benign-looking number and a denied one: the x32 guard must
        # fire before the per-syscall comparisons can mis-match.
        assert run_bpf(prog, nr=X32_BIT | 1, arch=ARCH_X86_64) == KILL_PROCESS
        assert run_bpf(prog, nr=X32_BIT | 101, arch=ARCH_X86_64) == KILL_PROCESS
        assert run_bpf(prog, nr=X32_BIT, arch=ARCH_X86_64) == KILL_PROCESS

    def test_high_nr_allowed_on_aarch64(self):
        # aarch64 has no x32 ABI, so numbers above 0x40000000 are just
        # unassigned syscalls -- the kernel answers ENOSYS, not the filter.
        prog = build_filter("aarch64")
        assert run_bpf(prog, nr=X32_BIT | 1, arch=ARCH_AARCH64) == ALLOW

    def test_arm64_alias_matches_aarch64(self):
        assert build_filter("arm64") == build_filter("aarch64")


class TestTableIntegrity:
    def test_documented_families_present(self):
        # The README/DESIGN contract for strict: ptrace, mount, bpf, keyctl.
        for name in ("ptrace", "mount", "umount2", "pivot_root", "bpf",
                     "keyctl", "add_key", "request_key", "unshare", "setns"):
            assert name in DENIED_SYSCALL_NAMES, name

    def test_tables_cover_identical_names(self):
        assert seccomp._DENIED_X86_64.keys() == seccomp._DENIED_AARCH64.keys()

    @pytest.mark.parametrize("table", [seccomp._DENIED_X86_64, seccomp._DENIED_AARCH64])
    def test_numbers_are_distinct_within_an_arch(self, table):
        assert len(set(table.values())) == len(table)

    def test_spot_check_kernel_abi_numbers(self):
        # A few well-known numbers transcribed independently of the module
        # source, as a typo tripwire (these are frozen kernel ABI).
        assert seccomp._DENIED_X86_64["ptrace"] == 101
        assert seccomp._DENIED_X86_64["mount"] == 165
        assert seccomp._DENIED_X86_64["bpf"] == 321
        assert seccomp._DENIED_AARCH64["ptrace"] == 117
        assert seccomp._DENIED_AARCH64["mount"] == 40
        assert seccomp._DENIED_AARCH64["bpf"] == 280


class TestBuilder:
    def test_unknown_machine_raises(self):
        with pytest.raises(SeccompUnavailableError):
            build_filter("riscv64")

    def test_program_shape(self):
        prog = build_filter("x86_64")
        assert len(prog) % 8 == 0
        first = struct.unpack_from("<HBBI", prog, 0)
        assert first == (0x20, 0, 0, 4)  # load seccomp_data.arch
        # Last three instructions: ret ALLOW / ret ERRNO(EPERM) / ret KILL.
        tail = [struct.unpack_from("<HBBI", prog, off)
                for off in range(len(prog) - 24, len(prog), 8)]
        assert [(c, k) for c, _, _, k in tail] == [
            (0x06, ALLOW), (0x06, ERRNO_EPERM), (0x06, KILL_PROCESS),
        ]

    def test_oversized_table_fails_loudly(self):
        # cBPF jumps carry 8-bit displacements; a table too large for them
        # must be a build-time error, never a silently mis-assembled program.
        huge = {f"fake_{i}": 10_000 + i for i in range(300)}
        with patch.dict(seccomp._ARCHES, {"x86_64": (ARCH_X86_64, huge, True)}):
            with pytest.raises(ValueError, match="displacement"):
                build_filter("x86_64")


class TestFilterMemfd:
    def test_memfd_contains_program_at_offset_zero(self):
        if not hasattr(seccomp.os, "memfd_create"):
            pytest.skip("no os.memfd_create on this platform")
        fd = seccomp.filter_memfd("x86_64")
        try:
            assert seccomp.os.read(fd, 1 << 16) == build_filter("x86_64")
        finally:
            seccomp.os.close(fd)

    def test_unavailable_without_memfd_create(self, monkeypatch):
        monkeypatch.delattr(seccomp.os, "memfd_create", raising=False)
        with pytest.raises(SeccompUnavailableError):
            seccomp.filter_memfd("x86_64")
