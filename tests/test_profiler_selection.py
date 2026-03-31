"""Tests for profiler selection logic in perflab.profilers.select_profilers."""
from __future__ import annotations

from dataclasses import dataclass

from perflab.profilers import select_profilers


@dataclass
class _FakeTask:
    program_type: str


def _profiler_names(task_type: str) -> set[str]:
    """Return set of profiler class names selected for a program type."""
    task = _FakeTask(program_type=task_type)
    profs = select_profilers(task)
    return {type(p).__name__ for p in profs}


class TestSelectProfilers:
    def test_python_gets_pyspy_and_memray(self):
        names = _profiler_names("python")
        assert "PySpyProfiler" in names
        assert "MemrayProfiler" in names
        # No GPU profilers
        assert "NsysProfiler" not in names
        assert "NcuProfiler" not in names

    def test_pytorch_gets_all_gpu_profilers(self):
        names = _profiler_names("pytorch")
        assert "PySpyProfiler" in names
        assert "TorchProfiler" in names
        assert "NsysProfiler" in names
        assert "NcuProfiler" in names
        assert "MetalTraceProfiler" in names
        assert "MemrayProfiler" in names

    def test_jax_gets_jax_profiler(self):
        names = _profiler_names("jax")
        assert "JaxProfiler" in names
        assert "PySpyProfiler" in names
        assert "NsysProfiler" in names

    def test_triton_gets_gpu_profilers_no_torch(self):
        names = _profiler_names("triton")
        assert "PySpyProfiler" in names
        assert "NsysProfiler" in names
        assert "NcuProfiler" in names
        assert "TorchProfiler" not in names

    def test_cuda_gets_nsys_ncu_lock_thread(self):
        names = _profiler_names("cuda")
        assert "NsysProfiler" in names
        assert "NcuProfiler" in names
        assert "LockContentionProfiler" in names
        assert "ThreadSchedProfiler" in names
        # No py-spy for compiled CUDA
        assert "PySpyProfiler" not in names

    def test_cpp_gets_lock_thread_profilers(self):
        names = _profiler_names("cpp")
        assert "LockContentionProfiler" in names
        assert "ThreadSchedProfiler" in names
        assert "LinuxPerfProfiler" in names
        assert "PySpyProfiler" not in names
        assert "MemrayProfiler" not in names

    def test_all_types_get_linux_perf(self):
        for ptype in ("python", "pytorch", "jax", "triton", "cuda", "cpp"):
            names = _profiler_names(ptype)
            assert "LinuxPerfProfiler" in names, f"{ptype} missing LinuxPerfProfiler"

    def test_all_types_get_power(self):
        for ptype in ("python", "pytorch", "jax", "triton", "cuda", "cpp"):
            names = _profiler_names(ptype)
            assert "PowerProfiler" in names, f"{ptype} missing PowerProfiler"

    def test_all_types_get_ebpf(self):
        for ptype in ("python", "pytorch", "jax", "triton", "cuda", "cpp"):
            names = _profiler_names(ptype)
            assert "EbpfProfiler" in names, f"{ptype} missing EbpfProfiler"

    def test_mps_types_get_metal(self):
        for ptype in ("pytorch", "jax", "triton"):
            names = _profiler_names(ptype)
            assert "MetalTraceProfiler" in names, f"{ptype} missing MetalTraceProfiler"

    def test_non_mps_types_no_metal(self):
        for ptype in ("python", "cuda", "cpp"):
            names = _profiler_names(ptype)
            assert "MetalTraceProfiler" not in names, f"{ptype} should not have MetalTraceProfiler"
