"""Tests for accuracy-tolerance rendering in build_prompt.

Regression coverage: the accuracy_tolerance paragraph used to be nested inside
_add_build_flag_recommendations, so framework tasks (PyTorch/JAX/Triton) that
declare a tolerance but carry no build-flag recommendations never saw it. It now
renders as its own _add_accuracy_tolerance section, independent of build flags.
"""
from perflab.optimizers.prompt import PromptContext, build_prompt


def _minimal_ctx(**kwargs):
    defaults = dict(
        source_files={"main.py": "print('hello')"},
        profiler_summaries={},
        bench_results={"ok": True, "throughput": {"median": 1.0}},
    )
    defaults.update(kwargs)
    return PromptContext(**defaults)


def _user_content(ctx):
    return build_prompt(ctx)[1].content


def test_tolerance_rendered_without_build_flags():
    # Framework-task case: tolerance set, no build-flag recommendations.
    ctx = _minimal_ctx(accuracy_tolerance="1e-3", build_flag_recommendations=None)
    content = _user_content(ctx)
    assert "## Accuracy tolerance" in content
    assert "Accuracy tolerance: **1e-3**" in content
    assert "results may differ" in content
    # The build-flag section must NOT appear when there are no recommendations.
    assert "## Build flag recommendations" not in content


def test_tolerance_absent_when_unset():
    ctx = _minimal_ctx(accuracy_tolerance=None)
    content = _user_content(ctx)
    assert "## Accuracy tolerance" not in content
    assert "Accuracy tolerance:" not in content


def test_tolerance_rendered_once_with_build_flags():
    ctx = _minimal_ctx(
        accuracy_tolerance="1e-2",
        program_type="cpp",
        build_flag_recommendations=[
            {"impact": "high", "flag": "-mavx2", "reason": "AVX2 supported"}
        ],
    )
    content = _user_content(ctx)
    # Tolerance text appears exactly once even alongside build flags.
    assert content.count("Accuracy tolerance: **1e-2**") == 1
    assert "## Accuracy tolerance" in content


def test_build_flag_section_renders_without_tolerance_line():
    ctx = _minimal_ctx(
        accuracy_tolerance=None,
        program_type="cpp",
        build_flag_recommendations=[
            {"impact": "high", "flag": "-mavx2", "reason": "AVX2 supported"}
        ],
    )
    content = _user_content(ctx)
    assert "## Build flag recommendations" in content
    assert "-mavx2" in content
    # No accuracy-tolerance line leaks into the build-flag section.
    assert "Accuracy tolerance:" not in content
    assert "## Accuracy tolerance" not in content
