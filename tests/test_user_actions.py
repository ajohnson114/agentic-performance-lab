"""Tests for user action extraction from LLM reasoning text."""
from __future__ import annotations

import json

from perflab.analyzers.user_actions import UserAction, extract_build_suggestions


class TestExtractBuildSuggestions:
    def test_fopenmp_with_build_phrase(self):
        text = "You should compile with -fopenmp to enable parallel loops."
        actions = extract_build_suggestions(text, iteration=3)
        assert len(actions) == 1
        assert actions[0].flag == "-fopenmp"
        assert actions[0].iteration == 3
        assert actions[0].source == "llm_reasoning"
        assert "-fopenmp" in actions[0].suggestion

    def test_march_native(self):
        text = "Add -march=native to your build command for auto-vectorization."
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert actions[0].flag == "-march=native"

    def test_multiple_flags_one_line(self):
        text = "Use -O3 and -ffast-math in your compilation flags."
        actions = extract_build_suggestions(text)
        flags = {a.flag for a in actions}
        assert "-O3" in flags
        assert "-ffast-math" in flags

    def test_multiple_flags_separate_lines(self):
        text = (
            "Add -fopenmp to enable OpenMP parallelism.\n"
            "Also add -march=native for better vectorization.\n"
        )
        actions = extract_build_suggestions(text)
        flags = {a.flag for a in actions}
        assert "-fopenmp" in flags
        assert "-march=native" in flags

    def test_cuda_use_fast_math(self):
        text = "Pass --use_fast_math to nvcc for faster transcendentals."
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert actions[0].flag == "--use_fast_math"

    def test_cuda_arch(self):
        text = "Add -arch=sm_80 to target Ampere GPUs."
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert actions[0].flag == "-arch=sm_80"

    def test_linker_flag(self):
        text = "You need to add -lgomp to link with the OpenMP runtime."
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert actions[0].flag == "-lgomp"

    def test_task_yaml_mention(self):
        text = "Update task.yaml build.cmd to include -O3 for optimization."
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert actions[0].flag == "-O3"

    def test_no_suggestions_in_plain_text(self):
        text = "I improved the loop by using shared memory tiling."
        actions = extract_build_suggestions(text)
        assert actions == []

    def test_no_suggestions_empty_text(self):
        assert extract_build_suggestions("") == []
        assert extract_build_suggestions("   \n  \n") == []

    def test_flag_without_build_context_ignored(self):
        # A flag mentioned in passing without an action verb or build phrase
        text = "The -O3 optimization level enables aggressive inlining."
        actions = extract_build_suggestions(text)
        assert actions == []

    def test_deduplication(self):
        text = (
            "Add -fopenmp to your build command.\n"
            "Make sure to compile with -fopenmp enabled.\n"
        )
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert actions[0].flag == "-fopenmp"

    def test_bullet_list_formatting(self):
        text = "- Add -ffast-math for faster floating-point math"
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert not actions[0].suggestion.startswith("-")

    def test_long_suggestion_truncated(self):
        text = "Add -O3 to your build command " + "x" * 250
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert len(actions[0].suggestion) <= 203  # 200 + "..."

    def test_action_verb_enable(self):
        text = "Enable -funroll-loops for loop unrolling."
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert actions[0].flag == "-funroll-loops"

    def test_action_verb_switch_to(self):
        text = "Switch to -O3 from -O2 for better performance."
        actions = extract_build_suggestions(text)
        assert len(actions) >= 1
        flags = {a.flag for a in actions}
        assert "-O3" in flags

    def test_cpp_standard(self):
        text = "Use -std=c++17 to enable structured bindings."
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert actions[0].flag == "-std=c++17"

    def test_flto(self):
        text = "Add -flto for link-time optimization."
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert actions[0].flag == "-flto"

    def test_ndebug(self):
        text = "Add -DNDEBUG to disable assertions in release builds."
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert actions[0].flag == "-DNDEBUG"

    def test_code_snippet_not_extracted(self):
        # Flags inside code blocks should still be extracted if they have
        # action verbs — the LLM often puts suggestions in mixed format
        text = "I added `#pragma omp parallel` to the code. You'll need to add -fopenmp to the build."
        actions = extract_build_suggestions(text)
        assert len(actions) == 1
        assert actions[0].flag == "-fopenmp"


class TestUserActionDataclass:
    def test_fields(self):
        a = UserAction(
            suggestion="Add -fopenmp",
            flag="-fopenmp",
            iteration=2,
            source="llm_reasoning",
        )
        assert a.suggestion == "Add -fopenmp"
        assert a.flag == "-fopenmp"
        assert a.iteration == 2
        assert a.source == "llm_reasoning"

    def test_serializable(self):
        a = UserAction(
            suggestion="Add -fopenmp",
            flag="-fopenmp",
            iteration=2,
            source="llm_reasoning",
        )
        d = {"suggestion": a.suggestion, "flag": a.flag,
             "iteration": a.iteration, "source": a.source}
        # Should be JSON-serializable
        assert json.loads(json.dumps(d)) == d
