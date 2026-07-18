"""Tests for data_hints in task.yaml and prompt integration."""
from __future__ import annotations

import textwrap


class TestDataHintsParsing:
    def test_data_hints_loaded(self, tmp_path):
        from perflab.task_spec import TaskSpec

        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text(textwrap.dedent("""\
            name: "test_hints"
            workspace: "."
            program_type: "cuda"
            correctness:
              cmd: "echo ok"
            benchmark:
              cmd: "echo bench"
              metric:
                name: "tflops.median"
            data_hints:
              sparsity: 0.95
              value_range: [-1.0, 1.0]
              access_pattern: "sequential"
              batch_size_range: [1, 128]
              dtype_safety: "fp16_safe"
              custom:
                - "data is symmetric"
                - "output is always positive"
        """), encoding="utf-8")

        spec = TaskSpec.load(task_yaml)
        assert spec.data_hints.sparsity == 0.95
        assert spec.data_hints.value_range == [-1.0, 1.0]
        assert spec.data_hints.access_pattern == "sequential"
        assert spec.data_hints.batch_size_range == [1, 128]
        assert spec.data_hints.dtype_safety == "fp16_safe"
        assert len(spec.data_hints.custom) == 2

    def test_no_data_hints(self, tmp_path):
        from perflab.task_spec import TaskSpec

        task_yaml = tmp_path / "task.yaml"
        task_yaml.write_text(textwrap.dedent("""\
            name: "test_no_hints"
            workspace: "."
            program_type: "python"
            correctness:
              cmd: "echo ok"
            benchmark:
              cmd: "echo bench"
              metric:
                name: "throughput.median"
        """), encoding="utf-8")

        spec = TaskSpec.load(task_yaml)
        assert spec.data_hints.sparsity is None
        assert spec.data_hints.custom is None


class TestDataHintsInPrompt:
    def test_sparsity_hint_in_prompt(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"kern.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["kern.cu"],
            program_type="cuda",
            data_hints={
                "sparsity": 0.95,
                "dtype_safety": "fp16_safe",
                "custom": ["data is symmetric"],
            },
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "95%" in full_text
        assert "sparse" in full_text.lower()
        assert "FP16" in full_text
        assert "symmetric" in full_text

    def test_no_data_hints_no_section(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"kern.cu": "code"},
            profiler_summaries={},
            bench_results={"tflops": {"median": 1.0}},
            roofline=None,
            history=[],
            allowed_paths=["kern.cu"],
            program_type="cuda",
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "Data characteristics" not in full_text

    def test_batch_1_latency_hint(self):
        from perflab.optimizers.prompt import PromptContext, build_prompt

        ctx = PromptContext(
            source_files={"model.py": "code"},
            profiler_summaries={},
            bench_results={"latency_ms": {"p50": 10.0}},
            roofline=None,
            history=[],
            allowed_paths=["model.py"],
            program_type="pytorch",
            data_hints={
                "batch_size_range": [1, 64],
            },
        )
        messages = build_prompt(ctx)
        full_text = " ".join(m.content for m in messages)

        assert "batch_size=1" in full_text or "batch-1" in full_text or "latency" in full_text.lower()
