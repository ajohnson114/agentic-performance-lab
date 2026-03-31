"""Tests for specialized profiler → LLM prompt integration.

Verifies that memray, lock contention, and GPU memory data flows
from profiler summaries through PromptContext into the rendered prompt.
"""
from __future__ import annotations

from perflab.optimizers.prompt import PromptContext, build_prompt


def _make_ctx(**overrides) -> PromptContext:
    """Create a minimal PromptContext with overrides."""
    defaults = dict(
        source_files={"main.py": "code"},
        profiler_summaries={},
        bench_results={"throughput": {"median": 100.0}},
        roofline=None,
        history=[],
        allowed_paths=["main.py"],
        program_type="python",
    )
    defaults.update(overrides)
    return PromptContext(**defaults)


def _prompt_text(ctx: PromptContext) -> str:
    messages = build_prompt(ctx)
    return " ".join(m.content for m in messages)


# ---------------------------------------------------------------------------
# Memray prompt integration
# ---------------------------------------------------------------------------

class TestMemrayPromptIntegration:
    def test_memray_top_allocators_rendered(self):
        ctx = _make_ctx(memray_summary={
            "peak_memory_mb": 512.0,
            "total_allocated_mb": 2048.0,
            "total_allocations": 150000,
            "top_allocators": [
                {"function": "torch.zeros", "location": "model.py:42", "size_mb": 256.0, "count": 500},
                {"function": "np.array", "location": "data.py:10", "size_mb": 128.0, "count": 1000},
            ],
        })
        text = _prompt_text(ctx)
        assert "Memory allocation hotspots" in text
        assert "512 MB" in text
        assert "torch.zeros" in text
        assert "model.py:42" in text
        assert "np.array" in text

    def test_memray_no_data_no_section(self):
        ctx = _make_ctx(memray_summary=None)
        text = _prompt_text(ctx)
        assert "Memory allocation hotspots" not in text

    def test_memray_empty_allocators_no_section(self):
        ctx = _make_ctx(memray_summary={
            "peak_memory_mb": 0,
            "total_allocations": 0,
            "top_allocators": [],
        })
        text = _prompt_text(ctx)
        assert "Memory allocation hotspots" not in text


# ---------------------------------------------------------------------------
# Lock contention prompt integration
# ---------------------------------------------------------------------------

class TestLockContentionPromptIntegration:
    def test_lock_contention_rendered(self):
        ctx = _make_ctx(lock_contention_summary={
            "lock_stats": {
                "total_contended": 42,
                "total_wait_ns": 5_000_000,
                "locks": [
                    {"name": "data_mutex", "acquired": 500, "contended": 42,
                     "avg_wait_ns": 119047, "total_wait_ns": 5_000_000, "max_wait_ns": 500_000},
                ],
            },
            "c2c_stats": {"total_hitm": 0, "false_sharing_lines": []},
        })
        text = _prompt_text(ctx)
        assert "Lock contention" in text
        assert "data_mutex" in text
        assert "42 contentions" in text

    def test_false_sharing_rendered(self):
        ctx = _make_ctx(lock_contention_summary={
            "lock_stats": {"total_contended": 0, "locks": []},
            "c2c_stats": {
                "total_hitm": 1500,
                "total_store": 5000,
                "false_sharing_lines": [
                    {"address": "0x7fff1234", "hitm": 800, "store": 2000},
                ],
            },
        })
        text = _prompt_text(ctx)
        assert "False sharing detected" in text
        assert "1500 HITM" in text
        assert "0x7fff1234" in text
        assert "cache line" in text.lower()

    def test_no_contention_no_section(self):
        ctx = _make_ctx(lock_contention_summary={
            "lock_stats": {"total_contended": 0, "locks": []},
            "c2c_stats": {"total_hitm": 0, "false_sharing_lines": []},
        })
        text = _prompt_text(ctx)
        assert "Lock contention" not in text

    def test_none_summary_no_section(self):
        ctx = _make_ctx(lock_contention_summary=None)
        text = _prompt_text(ctx)
        assert "Lock contention" not in text


# ---------------------------------------------------------------------------
# GPU memory prompt integration
# ---------------------------------------------------------------------------

class TestGpuMemoryPromptIntegration:
    def test_gpu_memory_rendered(self):
        ctx = _make_ctx(gpu_memory_summary={
            "total_mib": 16384,
            "max_used_mib": 12000,
            "avg_used_mib": 10000,
            "utilization_pct": 73.2,
            "sample_count": 20,
        })
        text = _prompt_text(ctx)
        assert "GPU memory" in text
        assert "12000" in text
        assert "16384" in text
        assert "73%" in text

    def test_gpu_memory_oom_warning(self):
        ctx = _make_ctx(gpu_memory_summary={
            "total_mib": 16384,
            "max_used_mib": 15000,
            "avg_used_mib": 14000,
            "utilization_pct": 91.6,
            "sample_count": 20,
        })
        text = _prompt_text(ctx)
        assert "OOM" in text
        assert "gradient checkpointing" in text

    def test_gpu_memory_zero_total_no_section(self):
        ctx = _make_ctx(gpu_memory_summary={
            "total_mib": 0,
            "max_used_mib": 0,
            "utilization_pct": 0,
        })
        text = _prompt_text(ctx)
        assert "GPU memory" not in text

    def test_gpu_memory_none_no_section(self):
        ctx = _make_ctx(gpu_memory_summary=None)
        text = _prompt_text(ctx)
        assert "GPU memory" not in text


# ---------------------------------------------------------------------------
# Agent helper extraction functions
# ---------------------------------------------------------------------------

class TestAgentExtractHelpers:
    def test_extract_memray_summary(self):
        from perflab.optimizers.agent import _extract_memray_summary
        summaries = {
            "memray": {
                "returncode": 0,
                "peak_memory_mb": 512.0,
                "total_allocated_mb": 2048.0,
                "total_allocations": 150000,
                "top_allocators": [{"function": "f", "size_mb": 100}],
            },
        }
        result = _extract_memray_summary(summaries)
        assert result is not None
        assert result["peak_memory_mb"] == 512.0

    def test_extract_memray_failed_returns_none(self):
        from perflab.optimizers.agent import _extract_memray_summary
        assert _extract_memray_summary({"memray": {"returncode": 1}}) is None
        assert _extract_memray_summary({}) is None

    def test_extract_lock_contention(self):
        from perflab.optimizers.agent import _extract_lock_contention_summary
        summaries = {
            "lock_contention": {
                "lock_stats": {"total_contended": 10, "locks": []},
                "c2c_stats": {"total_hitm": 0, "false_sharing_lines": []},
            },
        }
        result = _extract_lock_contention_summary(summaries)
        assert result is not None

    def test_extract_lock_contention_no_contention_returns_none(self):
        from perflab.optimizers.agent import _extract_lock_contention_summary
        summaries = {
            "lock_contention": {
                "lock_stats": {"total_contended": 0},
                "c2c_stats": {"total_hitm": 0},
            },
        }
        assert _extract_lock_contention_summary(summaries) is None

    def test_extract_gpu_memory(self):
        from perflab.optimizers.agent import _extract_gpu_memory_summary
        summaries = {
            "power": {
                "gpu_memory": {
                    "total_mib": 16384,
                    "max_used_mib": 12000,
                    "utilization_pct": 73.2,
                },
            },
        }
        result = _extract_gpu_memory_summary(summaries)
        assert result is not None
        assert result["total_mib"] == 16384

    def test_extract_gpu_memory_no_data(self):
        from perflab.optimizers.agent import _extract_gpu_memory_summary
        assert _extract_gpu_memory_summary({}) is None
        assert _extract_gpu_memory_summary({"power": {}}) is None


# ---------------------------------------------------------------------------
# eBPF prompt integration
# ---------------------------------------------------------------------------

class TestEbpfPromptIntegration:
    def test_ebpf_syscalls_rendered(self):
        ctx = _make_ctx(ebpf_summary={
            "read_syscalls": 5000,
            "write_syscalls": 1200,
            "read_bytes": 104857600,
            "write_bytes": 52428800,
            "read_latency": {
                "total_count": 5000,
                "p50_ns": 8000,
                "p90_ns": 50000,
                "p99_ns": 200000,
            },
            "write_latency": {
                "total_count": 1200,
                "p50_ns": 5000,
                "p90_ns": 30000,
                "p99_ns": 150000,
            },
        })
        text = _prompt_text(ctx)
        assert "Syscall tracing (eBPF)" in text
        assert "read=5,000" in text
        assert "write=1,200" in text
        assert "100.0 MB read" in text
        assert "p50=" in text
        assert "p99=" in text

    def test_ebpf_no_data_no_section(self):
        ctx = _make_ctx(ebpf_summary={
            "read_syscalls": 0,
            "write_syscalls": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "read_latency": None,
            "write_latency": None,
        })
        text = _prompt_text(ctx)
        assert "Syscall tracing (eBPF)" not in text

    def test_ebpf_none_no_section(self):
        ctx = _make_ctx(ebpf_summary=None)
        text = _prompt_text(ctx)
        assert "Syscall tracing (eBPF)" not in text


# ---------------------------------------------------------------------------
# Agent helper: eBPF extraction
# ---------------------------------------------------------------------------

class TestAgentExtractEbpf:
    def test_extract_ebpf_summary(self):
        from perflab.optimizers.agent import _extract_ebpf_summary
        summaries = {
            "ebpf": {
                "returncode": 0,
                "read_syscalls": 5000,
                "write_syscalls": 1200,
                "read_bytes": 104857600,
                "write_bytes": 52428800,
                "read_latency": {"total_count": 5000, "p50_ns": 8000},
                "write_latency": None,
            },
        }
        result = _extract_ebpf_summary(summaries)
        assert result is not None
        assert result["read_syscalls"] == 5000
        assert result["write_syscalls"] == 1200

    def test_extract_ebpf_failed_returns_none(self):
        from perflab.optimizers.agent import _extract_ebpf_summary
        assert _extract_ebpf_summary({"ebpf": {"returncode": 1}}) is None
        assert _extract_ebpf_summary({}) is None

    def test_extract_ebpf_no_syscalls_returns_none(self):
        from perflab.optimizers.agent import _extract_ebpf_summary
        assert _extract_ebpf_summary({
            "ebpf": {"returncode": 0, "read_syscalls": 0, "write_syscalls": 0},
        }) is None
