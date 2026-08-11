"""Tests for the forbidden-construct policy (perflab.optimizers.forbidden).

Motivating case: an agent wrote `#pragma GCC optimize("O3")` into the one file
it was allowed to edit, silently overriding a build command pinned in the
protected task.yaml. These tests cover that route and the general mechanism.
"""
from __future__ import annotations

import pytest

from perflab.optimizers.forbidden import (
    FORBIDDEN_CONSTRUCTS,
    check_text,
    compile_rules,
    describe,
    validate_spec,
)
from perflab.optimizers.patch import SearchReplaceBlock, validate_patch
from perflab.task_spec import TaskSpec


class TestNamedConstructs:
    @pytest.mark.parametrize(
        ("construct", "snippet"),
        [
            ("optimization_pragmas", '#pragma GCC optimize("O3","unroll-loops")'),
            ("optimization_pragmas", "#pragma GCC target(\"avx2\")"),
            ("optimization_pragmas", '__attribute__((optimize("O3"))) void f(){}'),
            ("optimization_pragmas", "#pragma clang optimize on"),
            ("openmp", "#pragma omp parallel for"),
            ("openmp", "#include <omp.h>"),
            ("openmp", "omp_set_num_threads(8);"),
            ("threading", "std::thread t(worker);"),
            ("threading", "from concurrent.futures import ThreadPoolExecutor"),
            ("threading", "import multiprocessing"),
            ("blas", "cblas_sgemm(CblasRowMajor, ...);"),
            ("blas", "#include <cblas.h>"),
            ("simd_intrinsics", "#include <arm_neon.h>"),
            ("simd_intrinsics", "__m256 v = _mm256_load_ps(p);"),
            ("inline_asm", '__asm__ volatile("nop");'),
        ],
    )
    def test_construct_is_detected(self, construct: str, snippet: str) -> None:
        rules = compile_rules([construct], [])
        violations = check_text(snippet, rules)
        assert violations, f"{construct} should reject {snippet!r}"
        assert construct in violations[0]

    def test_only_named_constructs_are_active(self) -> None:
        """Forbidding openmp must not incidentally reject a pragma."""
        rules = compile_rules(["openmp"], [])
        assert not check_text('#pragma GCC optimize("O3")', rules)

    def test_nothing_forbidden_by_default(self) -> None:
        assert compile_rules([], []) == []
        assert check_text('#pragma GCC optimize("O3")', []) == []


class TestLegitimateCodePasses:
    @pytest.mark.parametrize(
        "snippet",
        [
            "for (int kb = 0; kb < K; kb += KC) { /* cache blocking */ }",
            "float acc[4][8]; // register tiling",
            "const float* __restrict A;",
            "np.multiply(A, scalar, out=B)",
            "std::memset(C, 0, n * sizeof(float));",
        ],
    )
    def test_real_optimizations_are_allowed(self, snippet: str) -> None:
        rules = compile_rules(sorted(FORBIDDEN_CONSTRUCTS), [])
        assert not check_text(snippet, rules), f"false positive on {snippet!r}"


class TestCustomPatterns:
    def test_custom_regex_matches(self) -> None:
        rules = compile_rules([], [r"\bmy_fast_lib\b"])
        violations = check_text("result = my_fast_lib(x)", rules)
        assert violations and "forbidden pattern" in violations[0]

    def test_custom_regex_does_not_overmatch(self) -> None:
        rules = compile_rules([], [r"\bmy_fast_lib\b"])
        assert not check_text("my_fast_library_wrapper_name_x", rules)


class TestSpecValidation:
    def test_unknown_construct_reported(self) -> None:
        errors = validate_spec(["nope"], [])
        assert errors and "unknown forbidden_construct" in errors[0]

    def test_bad_regex_reported(self) -> None:
        errors = validate_spec([], ["([unclosed"])
        assert errors and "invalid forbidden_pattern" in errors[0]

    def test_valid_spec_has_no_errors(self) -> None:
        assert validate_spec(["openmp"], [r"\bfoo\b"]) == []

    def test_bad_regex_is_skipped_not_raised_at_compile(self) -> None:
        """compile_rules must not explode mid-run on a config typo."""
        assert compile_rules(["nope"], ["([unclosed"]) == []


class TestDescribe:
    def test_describe_lists_constructs_and_patterns(self) -> None:
        lines = describe(["openmp"], [r"\bfoo\b"])
        assert any("openmp" in ln for ln in lines)
        assert any("foo" in ln for ln in lines)

    def test_describe_skips_unknown(self) -> None:
        assert describe(["nope"], []) == []


class TestValidatePatchIntegration:
    def _ws(self, tmp_path):
        (tmp_path / "kernel.cpp").write_text("int slow() { return 0; }\n")
        return tmp_path

    def test_forbidden_replace_is_rejected(self, tmp_path) -> None:
        ws = self._ws(tmp_path)
        block = SearchReplaceBlock(
            file_path="kernel.cpp",
            search="int slow() { return 0; }",
            replace='#pragma GCC optimize("O3")\nint slow() { return 0; }',
        )
        errors = validate_patch(
            [block], ["*.cpp"], ws,
            forbidden_rules=compile_rules(["optimization_pragmas"], []),
        )
        assert errors and "optimization_pragmas" in errors[0]

    def test_clean_replace_is_accepted(self, tmp_path) -> None:
        ws = self._ws(tmp_path)
        block = SearchReplaceBlock(
            file_path="kernel.cpp",
            search="int slow() { return 0; }",
            replace="int slow() { return 1; }",
        )
        errors = validate_patch(
            [block], ["*.cpp"], ws,
            forbidden_rules=compile_rules(["optimization_pragmas"], []),
        )
        assert errors == []

    def test_only_replace_side_is_checked(self, tmp_path) -> None:
        """A baseline that already contains a construct is not a violation."""
        ws = tmp_path
        (ws / "kernel.cpp").write_text('#pragma omp parallel for\nint x;\n')
        block = SearchReplaceBlock(
            file_path="kernel.cpp",
            search="#pragma omp parallel for\nint x;",
            replace="int x;",
        )
        errors = validate_patch(
            [block], ["*.cpp"], ws,
            forbidden_rules=compile_rules(["openmp"], []),
        )
        assert errors == []

    def test_no_rules_means_no_checking(self, tmp_path) -> None:
        ws = self._ws(tmp_path)
        block = SearchReplaceBlock(
            file_path="kernel.cpp",
            search="int slow() { return 0; }",
            replace='#pragma GCC optimize("O3")\nint slow() { return 0; }',
        )
        assert validate_patch([block], ["*.cpp"], ws) == []


class TestTaskSpecWiring:
    BASE = (
        "name: t\n"
        "program_type: python\n"
        "correctness:\n"
        '  cmd: "python tests.py"\n'
        "benchmark:\n"
        '  cmd: "python bench.py --json out/bench.json"\n'
        "  metric:\n"
        '    name: "t.median"\n'
        '    mode: "maximize"\n'
        "edit_policy:\n"
        '  allowed_paths: ["*.py"]\n'
    )

    def _write_task(self, tmp_path, anti_gaming: str) -> str:
        (tmp_path / "kernel.py").write_text("x = 1\n")
        task = tmp_path / "task.yaml"
        task.write_text(self.BASE + anti_gaming)
        return str(task)

    def test_forbidden_constructs_parsed(self, tmp_path) -> None:
        path = self._write_task(
            tmp_path,
            "anti_gaming:\n"
            '  forbidden_constructs: ["openmp"]\n'
            '  forbidden_patterns: ["\\\\bfoo\\\\b"]\n',
        )
        spec = TaskSpec.load(path)
        assert spec.anti_gaming.forbidden_constructs == ["openmp"]
        assert spec.anti_gaming.forbidden_patterns == [r"\bfoo\b"]

    def test_defaults_are_empty(self, tmp_path) -> None:
        spec = TaskSpec.load(self._write_task(tmp_path, ""))
        assert spec.anti_gaming.forbidden_constructs == []
        assert spec.anti_gaming.forbidden_patterns == []

    def test_unknown_construct_fails_task_load(self, tmp_path) -> None:
        path = self._write_task(
            tmp_path,
            "anti_gaming:\n"
            '  forbidden_constructs: ["definitely_not_a_construct"]\n',
        )
        with pytest.raises(ValueError, match="unknown forbidden_construct"):
            TaskSpec.load(path)


class TestShippedMatmulTask:
    def test_cpp_matmul_forbids_the_known_escape_hatches(self) -> None:
        """The demo task must actually carry the policy, not just document it."""
        from perflab.cli import _demo_tasks_root

        spec = TaskSpec.load(_demo_tasks_root() / "matmul/cpp/task.yaml")
        forbidden = spec.anti_gaming.forbidden_constructs
        assert "optimization_pragmas" in forbidden
        assert "blas" in forbidden
        # Hand-vectorizing is real work and must stay legal.
        assert "simd_intrinsics" not in forbidden

    def test_cpp_matmul_rejects_the_exact_pragma_that_was_used(self) -> None:
        from perflab.cli import _demo_tasks_root

        spec = TaskSpec.load(_demo_tasks_root() / "matmul/cpp/task.yaml")
        rules = compile_rules(
            spec.anti_gaming.forbidden_constructs,
            spec.anti_gaming.forbidden_patterns,
        )
        assert check_text('#pragma GCC optimize("O3", "unroll-loops")', rules)
