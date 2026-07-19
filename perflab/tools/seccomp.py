"""Classic-BPF seccomp filter backing the ``strict`` isolation level.

``strict`` (perflab.tools.isolation) layers a syscall-denial filter on top of
``restricted``'s bwrap sandbox, passed to ``bwrap --seccomp FD`` as raw
compiled cBPF. This module builds that program with the stdlib only -- no
libseccomp binding -- which is viable because the filter is a small *denylist*
(an allowlist would need the full syscall surface and real BPF tooling):

    load arch      -> not this build's arch?          KILL_PROCESS
    load nr        -> x32 ABI bit set? (x86-64 only)  KILL_PROCESS
                   -> nr in the denied table?         ERRNO(EPERM)
                   -> otherwise                       ALLOW

Denied syscalls are escape/tamper primitives no legitimate benchmark or
correctness run needs: tracing another process's memory, (un)mounting
filesystems, loading kernel code, joining/creating namespaces, and the kernel
keyring. Denials return EPERM rather than killing, so candidate code that
merely probes (some runtimes do) degrades gracefully instead of dying; the
arch/x32 checks kill outright because a syscall through a foreign ABI has no
trustworthy number to check against the table.

Syscall numbers are copied per-architecture from the kernel's syscall tables
(x86_64: arch/x86/entry/syscalls/syscall_64.tbl; aarch64: the asm-generic
table). They are kernel ABI -- frozen forever once assigned -- so hardcoding
them is safe. Correctness of the emitted program is tested two ways: a cBPF
interpreter in tests/test_seccomp.py symbolically executes it against every
denied/allowed case on both arches, and Linux acceptance tests (CI x86_64
runners, the docker/ dev container for aarch64) verify real kernel
enforcement through bwrap.
"""
from __future__ import annotations

import os
import platform
import struct

# The four BPF opcodes this filter uses (linux/bpf_common.h encodings).
_BPF_LD_W_ABS = 0x20   # A = seccomp_data[k]
_BPF_JEQ_K = 0x15      # pc += (A == k) ? jt : jf
_BPF_JGE_K = 0x35      # pc += (A >= k) ? jt : jf
_BPF_RET_K = 0x06      # return k

# seccomp return actions (linux/seccomp.h).
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000  # low 16 bits carry the errno
_SECCOMP_RET_KILL_PROCESS = 0x80000000
_EPERM = 1

# struct seccomp_data field offsets.
_OFF_NR = 0
_OFF_ARCH = 4

_AUDIT_ARCH_X86_64 = 0xC000003E
_AUDIT_ARCH_AARCH64 = 0xC00000B7

# On x86-64 the x32 ABI reuses the 64-bit entry path with this bit OR'd into
# the syscall number, so a denylist keyed on plain x86_64 numbers would fail
# open for x32 calls. No process perflab runs uses x32; kill it outright.
_X32_SYSCALL_BIT = 0x40000000

# Denied syscalls per architecture. The name -> number tables must stay in
# name-lockstep (asserted below); numbers intentionally differ per arch.
_DENIED_X86_64: dict[str, int] = {
    # Tracing / foreign-process memory access.
    "ptrace": 101,
    "process_vm_readv": 310,
    "process_vm_writev": 311,
    "kcmp": 312,
    "perf_event_open": 298,
    # Filesystem topology, classic + new mount API.
    "mount": 165,
    "umount2": 166,
    "pivot_root": 155,
    "chroot": 161,
    "open_tree": 428,
    "move_mount": 429,
    "fsopen": 430,
    "fsconfig": 431,
    "fsmount": 432,
    "fspick": 433,
    "mount_setattr": 442,
    # Kernel code loading and machine state.
    "bpf": 321,
    "init_module": 175,
    "finit_module": 313,
    "delete_module": 176,
    "kexec_load": 246,
    "kexec_file_load": 320,
    "reboot": 169,
    "swapon": 167,
    "swapoff": 168,
    # Namespace manipulation (bwrap has already built the sandbox by the time
    # this filter is installed) and the userfaultfd exploit primitive.
    "unshare": 272,
    "setns": 308,
    "userfaultfd": 323,
    # Kernel keyring.
    "keyctl": 250,
    "add_key": 248,
    "request_key": 249,
}

_DENIED_AARCH64: dict[str, int] = {
    "ptrace": 117,
    "process_vm_readv": 270,
    "process_vm_writev": 271,
    "kcmp": 272,
    "perf_event_open": 241,
    "mount": 40,
    "umount2": 39,
    "pivot_root": 41,
    "chroot": 51,
    "open_tree": 428,
    "move_mount": 429,
    "fsopen": 430,
    "fsconfig": 431,
    "fsmount": 432,
    "fspick": 433,
    "mount_setattr": 442,
    "bpf": 280,
    "init_module": 105,
    "finit_module": 273,
    "delete_module": 106,
    "kexec_load": 104,
    "kexec_file_load": 294,
    "reboot": 142,
    "swapon": 224,
    "swapoff": 225,
    "unshare": 97,
    "setns": 268,
    "userfaultfd": 282,
    "keyctl": 219,
    "add_key": 217,
    "request_key": 218,
}

assert _DENIED_X86_64.keys() == _DENIED_AARCH64.keys(), (
    "per-arch denied-syscall tables drifted apart"
)

# Public: the set of syscall names strict denies, for docs/tests.
DENIED_SYSCALL_NAMES: frozenset[str] = frozenset(_DENIED_X86_64)

_ARCHES: dict[str, tuple[int, dict[str, int], bool]] = {
    # machine -> (audit arch, denied table, has x32 ABI to guard against)
    "x86_64": (_AUDIT_ARCH_X86_64, _DENIED_X86_64, True),
    "aarch64": (_AUDIT_ARCH_AARCH64, _DENIED_AARCH64, False),
    "arm64": (_AUDIT_ARCH_AARCH64, _DENIED_AARCH64, False),
}


class SeccompUnavailableError(RuntimeError):
    """This host can't get a seccomp filter (unknown arch, no memfd_create).

    Callers (isolation.wrap_command) treat this like every other capability
    gap in that module: log and degrade to restricted-equivalent, don't raise.
    """


def _insn(code: int, jt: int, jf: int, k: int) -> bytes:
    # struct sock_filter { __u16 code; __u8 jt; __u8 jf; __u32 k; } -- 8 bytes,
    # no padding; little-endian on every arch this module supports.
    if not (0 <= jt <= 0xFF and 0 <= jf <= 0xFF):
        # cBPF conditional jumps carry 8-bit displacements. Unreachable while
        # the denied table stays small (~30 entries => offsets ~35); a guard so
        # future growth fails loudly at build time, never as a mis-jump.
        raise ValueError(f"BPF jump displacement out of range: jt={jt} jf={jf}")
    return struct.pack("<HBBI", code, jt, jf, k)


def build_filter(machine: str | None = None) -> bytes:
    """Compile the strict-mode seccomp program for ``machine``.

    machine defaults to platform.machine(). Raises SeccompUnavailableError for
    architectures without a syscall table here (the filter must never guess:
    wrong numbers would silently deny nothing).
    """
    machine = machine or platform.machine()
    try:
        audit_arch, denied, has_x32 = _ARCHES[machine]
    except KeyError:
        raise SeccompUnavailableError(
            f"no seccomp syscall table for architecture {machine!r} "
            f"(known: {sorted(_ARCHES)})"
        ) from None

    # Layout (indices relative to program start):
    #   0            ld arch
    #   1            jeq audit_arch ? fall through : KILL
    #   2            ld nr
    #   [3]          jge X32_SYSCALL_BIT ? KILL : fall through   (x86_64 only)
    #   base .. +n-1 jeq denied_nr ? DENY : next
    #   base + n     ret ALLOW
    #   base + n + 1 ret ERRNO(EPERM)     <- DENY
    #   base + n + 2 ret KILL_PROCESS     <- KILL
    nrs = sorted(denied.values())
    base = 4 if has_x32 else 3
    allow_idx = base + len(nrs)
    deny_idx = allow_idx + 1
    kill_idx = deny_idx + 1

    insns = [
        _insn(_BPF_LD_W_ABS, 0, 0, _OFF_ARCH),
        _insn(_BPF_JEQ_K, 0, kill_idx - 1 - 1, audit_arch),
        _insn(_BPF_LD_W_ABS, 0, 0, _OFF_NR),
    ]
    if has_x32:
        insns.append(_insn(_BPF_JGE_K, kill_idx - 3 - 1, 0, _X32_SYSCALL_BIT))
    for i, nr in enumerate(nrs):
        insns.append(_insn(_BPF_JEQ_K, deny_idx - (base + i) - 1, 0, nr))
    insns.append(_insn(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW))
    insns.append(_insn(_BPF_RET_K, 0, 0, _SECCOMP_RET_ERRNO | _EPERM))
    insns.append(_insn(_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL_PROCESS))
    return b"".join(insns)


def filter_memfd(machine: str | None = None) -> int:
    """The compiled program in an anonymous memfd, rewound to offset 0.

    bwrap reads ``--seccomp FD`` to EOF, so every bwrap invocation needs its
    own fresh fd (a shared one would be at EOF after the first read). The fd
    is CLOEXEC as created; subprocess pass_fds re-marks it inheritable in the
    child, and run_cmd closes the parent's copy after the spawn -- see
    perflab.tools.shell.run_cmd's pass_fds ownership contract.

    Raises SeccompUnavailableError (unknown arch / no os.memfd_create) or
    OSError (memfd creation failed).
    """
    prog = build_filter(machine)
    memfd_create = getattr(os, "memfd_create", None)
    if memfd_create is None:  # non-Linux; only reachable from tests
        raise SeccompUnavailableError("os.memfd_create is not available on this platform")
    fd = memfd_create("perflab-seccomp")
    try:
        os.write(fd, prog)
        os.lseek(fd, 0, os.SEEK_SET)
    except OSError:
        os.close(fd)
        raise
    return fd
