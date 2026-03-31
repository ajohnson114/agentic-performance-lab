"""Compiler diagnostics capture and parsing.

Captures and parses diagnostic output from AOT compilers (g++, nvcc) and
JIT compilers (torch.compile/Inductor, JAX/XLA, Triton) to surface actionable
optimization hints — missed vectorizations, register spills, graph breaks,
recompilations — that the agent can act on.

Structured optimization remarks (GCC -fopt-info-all-optall and Clang -Rpass)
provide per-function/per-line detail that can be cross-referenced with perf
annotate hotspots.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field

# 2048 chars: keeps compiler summary under ~500 tokens to avoid consuming the LLM's context budget
_MAX_SUMMARY_CHARS = 2048


# ---------------------------------------------------------------------------
# Structured optimization remark
# ---------------------------------------------------------------------------

@dataclass
class OptimizationRemark:
    """A structured compiler optimization remark with source location."""
    file: str
    line: int
    col: int | None
    category: str   # "vectorize", "inline", "unroll", "loop", "alias", "fma", "register-pressure", "shared-memory", "other"
    status: str     # "applied", "missed", "analysis"
    detail: str     # human-readable detail
    width: int | None = None  # vectorization width in bits

    def to_dict(self) -> dict:
        return {
            "file": self.file, "line": self.line, "col": self.col,
            "category": self.category, "status": self.status,
            "detail": self.detail, "width": self.width,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OptimizationRemark:
        return cls(
            file=d["file"], line=d["line"], col=d.get("col"),
            category=d["category"], status=d["status"],
            detail=d["detail"], width=d.get("width"),
        )


@dataclass
class CrossReferencedInsight:
    """An insight from cross-referencing compiler remarks with profiler data."""
    priority: str           # "high", "medium", "low"
    category: str           # "missed-vec-in-hotspot", "vectorization-gap", "alias-blocking", "missed-fma"
    source_location: str    # "matmul.cpp:17"
    description: str
    suggestion: str
    perf_pct: float | None = None  # % of samples at this location


# ---------------------------------------------------------------------------
# Compiler detection
# ---------------------------------------------------------------------------

def detect_compiler(build_cmd: str) -> str:
    """Detect compiler from build command. Returns 'gcc', 'clang', or 'unknown'.

    On macOS, g++ is often Apple Clang — checks --version output.
    """
    parts = build_cmd.split()
    if not parts:
        return "unknown"

    binary = os.path.basename(parts[0])

    # nvcc is always nvcc
    if binary == "nvcc":
        return "nvcc"

    # Check if the binary looks like gcc/g++ or clang/clang++
    if binary in ("clang", "clang++"):
        return "clang"

    if binary in ("gcc", "g++", "cc", "c++"):
        # On macOS, g++ might be Apple Clang. Check --version output.
        try:
            result = subprocess.run(
                [parts[0], "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and "clang" in result.stdout.lower():
                return "clang"
        except (OSError, subprocess.SubprocessError):
            pass
        return "gcc"

    return "unknown"


# ---------------------------------------------------------------------------
# Diagnostic build flags
# ---------------------------------------------------------------------------

def get_diagnostic_build_flags(program_type: str, compiler: str = "gcc") -> list[str]:
    """Return extra compiler flags that enable diagnostic output for AOT compilers."""
    if program_type == "cpp":
        if compiler in ("clang", "clang++"):
            return ["-Rpass=.*", "-Rpass-missed=.*", "-Rpass-analysis=.*", "-gline-tables-only"]
        return ["-fopt-info-all-optall", "-gline-tables-only"]  # GCC default
    if program_type == "cuda":
        return ["--ptxas-options=-v", "--generate-line-info"]
    return []


def get_diagnostic_env_vars(program_type: str, compiler: str = "gcc") -> dict[str, str]:
    """Return env vars that enable diagnostic output for JIT/runtime compilers.

    For AOT types (cpp, cuda) the env vars tell bench.py to pick up
    diagnostic flags via PERFLAB_CXXFLAGS / PERFLAB_NVCCFLAGS.
    """
    if program_type == "pytorch":
        return {"TORCH_LOGS": "+dynamo,+inductor"}
    if program_type == "jax":
        return {"JAX_LOG_COMPILES": "1"}
    if program_type == "triton":
        return {"TRITON_DEBUG": "1"}
    if program_type == "cpp":
        if compiler in ("clang", "clang++"):
            return {"PERFLAB_CXXFLAGS": "-Rpass=.* -Rpass-missed=.* -Rpass-analysis=.* -gline-tables-only"}
        return {"PERFLAB_CXXFLAGS": "-fopt-info-all-optall -gline-tables-only"}
    if program_type == "cuda":
        return {"PERFLAB_NVCCFLAGS": "--ptxas-options=-v"}
    return {}


# ---------------------------------------------------------------------------
# Structured remark parsers
# ---------------------------------------------------------------------------

_GCC_REMARK_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s+(?P<status>optimized|missed|note):\s+(?P<detail>.+)$"
)

_CLANG_REMARK_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s+remark:\s+(?P<detail>.+?)\s*"
    r"\[-R(?P<pass_type>pass|pass-missed|pass-analysis)=(?P<pass_name>[^\]]+)\]$"
)

_VEC_WIDTH_BYTES_RE = re.compile(r"(\d+)\s+byte\s+vectors?", re.IGNORECASE)
_VEC_WIDTH_ELEMENTS_RE = re.compile(r"vectorization\s+width:\s+(\d+)", re.IGNORECASE)

# NVCC ptxas patterns
_NVCC_FUNC_RE = re.compile(
    r"(?:Compiling entry function|Function properties for)\s+'?(\S+?)'?(?:\s|$)",
    re.IGNORECASE,
)
_NVCC_REGS_RE = re.compile(r"Used\s+(\d+)\s+registers", re.IGNORECASE)
_NVCC_SMEM_RE = re.compile(r"(\d+)\s+bytes\s+smem", re.IGNORECASE)
_NVCC_SPILL_STORE_RE = re.compile(r"(\d+)\s+bytes\s+spill\s+stores", re.IGNORECASE)
_NVCC_SPILL_LOAD_RE = re.compile(r"(\d+)\s+bytes\s+spill\s+loads", re.IGNORECASE)


def _classify_category(detail: str) -> str:
    """Classify a remark detail string into a category."""
    lower = detail.lower()
    if "vector" in lower or "vec" in lower:
        return "vectorize"
    if "inline" in lower or "inlined" in lower or "inlinable" in lower:
        return "inline"
    if "unroll" in lower:
        return "unroll"
    if "alias" in lower or "restrict" in lower:
        return "alias"
    if "fma" in lower or "multiply-add" in lower or "fused multiply" in lower:
        return "fma"
    if "loop" in lower:
        return "loop"
    if "register" in lower or "spill" in lower:
        return "register-pressure"
    if "shared" in lower and "mem" in lower:
        return "shared-memory"
    return "other"


def _infer_element_bits(detail: str) -> int:
    """Infer element bit width from a remark detail string.

    Checks for type keywords (double, float, i64, etc.) to determine the
    element size. Returns 32 (float/int32) as default — the most common
    element type in performance-critical loops.
    """
    lower = detail.lower()
    if any(kw in lower for kw in ("double", "f64", "i64", "int64", "long")):
        return 64
    if any(kw in lower for kw in ("half", "fp16", "f16", "i16", "int16", "short", "bfloat")):
        return 16
    if any(kw in lower for kw in ("i8", "int8", "char", "byte")):
        return 8
    # Default: float/int32 — most common in compute-intensive loops
    return 32


def _extract_vec_width(detail: str) -> int | None:
    """Extract vectorization width in bits from a remark detail string."""
    # GCC: "using 32 byte vectors" → 256 bits
    m = _VEC_WIDTH_BYTES_RE.search(detail)
    if m:
        return int(m.group(1)) * 8

    # Clang: "vectorization width: 8" — element count, infer type from context
    m = _VEC_WIDTH_ELEMENTS_RE.search(detail)
    if m:
        element_bits = _infer_element_bits(detail)
        return int(m.group(1)) * element_bits

    return None


def _parse_gcc_remarks(stderr: str) -> list[OptimizationRemark]:
    """Parse structured GCC optimization remarks from -fopt-info-all-optall output."""
    remarks: list[OptimizationRemark] = []

    for line in stderr.splitlines():
        m = _GCC_REMARK_RE.match(line.strip())
        if not m:
            continue

        status_raw = m.group("status")
        status = {"optimized": "applied", "missed": "missed", "note": "analysis"}.get(
            status_raw, "analysis"
        )
        detail = m.group("detail")
        category = _classify_category(detail)
        width = _extract_vec_width(detail) if category == "vectorize" else None

        remarks.append(OptimizationRemark(
            file=m.group("file"),
            line=int(m.group("line")),
            col=int(m.group("col")),
            category=category,
            status=status,
            detail=detail,
            width=width,
        ))

    return remarks


def _parse_clang_remarks(stderr: str) -> list[OptimizationRemark]:
    """Parse structured Clang optimization remarks from -Rpass output."""
    remarks: list[OptimizationRemark] = []

    for line in stderr.splitlines():
        m = _CLANG_REMARK_RE.match(line.strip())
        if not m:
            continue

        pass_type = m.group("pass_type")
        status = {"pass": "applied", "pass-missed": "missed", "pass-analysis": "analysis"}.get(
            pass_type, "analysis"
        )
        detail = m.group("detail")
        pass_name = m.group("pass_name")

        # Classify from pass name and detail
        if "vectorize" in pass_name:
            category = "vectorize"
        elif "inline" in pass_name:
            category = "inline"
        elif "unroll" in pass_name:
            category = "unroll"
        else:
            category = _classify_category(detail)

        width = _extract_vec_width(detail) if category == "vectorize" else None

        remarks.append(OptimizationRemark(
            file=m.group("file"),
            line=int(m.group("line")),
            col=int(m.group("col")),
            category=category,
            status=status,
            detail=detail,
            width=width,
        ))

    return remarks


def _parse_nvcc_remarks(stderr: str) -> list[OptimizationRemark]:
    """Parse structured NVCC/ptxas remarks from --ptxas-options=-v output.

    ptxas emits per-kernel resource info:
        ptxas info    : Compiling entry function '_Z12sgemm_naivePfS_S_iii'
        ptxas info    : Function properties for _Z12sgemm_naivePfS_S_iii
                            0 bytes stack frame, 16 bytes spill stores, 16 bytes spill loads
        ptxas info    : Used 64 registers, 8192 bytes smem, 360 bytes cmem[0]

    Creates OptimizationRemark per finding with the kernel function name as
    the ``file`` field (line=0 since ptxas doesn't emit source lines).
    """
    remarks: list[OptimizationRemark] = []
    current_func: str = ""

    for line in stderr.splitlines():
        stripped = line.strip()

        # Track current kernel function
        m = _NVCC_FUNC_RE.search(stripped)
        if m:
            current_func = m.group(1).strip("'")
            continue

        if not current_func:
            continue

        # Register usage
        m = _NVCC_REGS_RE.search(stripped)
        if m:
            regs = int(m.group(1))
            # 64 regs: NVIDIA GPUs have 65536 regs/SM; >64 regs/thread limits occupancy to ≤50%
            status = "missed" if regs > 64 else "analysis"
            detail = f"Used {regs} registers/thread"
            if regs > 64:
                detail += " — high register pressure may limit occupancy"
            remarks.append(OptimizationRemark(
                file=current_func, line=0, col=None,
                category="register-pressure", status=status,
                detail=detail,
            ))

        # Shared memory
        m = _NVCC_SMEM_RE.search(stripped)
        if m:
            smem = int(m.group(1))
            remarks.append(OptimizationRemark(
                file=current_func, line=0, col=None,
                category="shared-memory", status="analysis",
                detail=f"Shared memory: {smem} bytes",
            ))

        # Spill stores
        m = _NVCC_SPILL_STORE_RE.search(stripped)
        if m:
            spill = int(m.group(1))
            if spill > 0:
                remarks.append(OptimizationRemark(
                    file=current_func, line=0, col=None,
                    category="register-pressure", status="missed",
                    detail=f"Register spill stores: {spill} bytes",
                ))

        # Spill loads
        m = _NVCC_SPILL_LOAD_RE.search(stripped)
        if m:
            spill = int(m.group(1))
            if spill > 0:
                remarks.append(OptimizationRemark(
                    file=current_func, line=0, col=None,
                    category="register-pressure", status="missed",
                    detail=f"Register spill loads: {spill} bytes",
                ))

    return remarks


# ---------------------------------------------------------------------------
# Legacy flat parsers (backward compatibility)
# ---------------------------------------------------------------------------

def _parse_gcc_diagnostics(stderr: str) -> list[str]:
    """Parse GCC/G++ optimization diagnostics from -fopt-info output.

    Wrapper around _parse_gcc_remarks for backward compatibility.
    Falls back to keyword counting when structured parsing yields nothing.
    """
    remarks = _parse_gcc_remarks(stderr)
    if remarks:
        findings: list[str] = []
        vec_missed = sum(1 for r in remarks if r.category == "vectorize" and r.status == "missed")
        vec_success = sum(1 for r in remarks if r.category == "vectorize" and r.status == "applied")
        inline_missed = sum(1 for r in remarks if r.category == "inline" and r.status == "missed")

        if vec_missed > 0:
            findings.append(f"Missed vectorizations: {vec_missed} loops not vectorized")
        if vec_success > 0:
            findings.append(f"Successful vectorizations: {vec_success} loops vectorized")
        if inline_missed > 0:
            findings.append(f"Missed inlines: {inline_missed} call sites not inlined")
        return findings

    # Fall back to keyword counting for non-structured output
    findings = []
    vec_missed = 0
    inline_missed = 0
    vec_success = 0

    for line in stderr.splitlines():
        if "missed:" in line.lower() or "note:" in line.lower():
            if "vectorized" in line.lower() or "vec" in line.lower():
                if "missed" in line.lower() or "not vectorized" in line.lower():
                    vec_missed += 1
                else:
                    vec_success += 1
            if ("inline" in line.lower() or "inlin" in line.lower()) and "missed" in line.lower():
                inline_missed += 1

    if vec_missed > 0:
        findings.append(f"Missed vectorizations: {vec_missed} loops not vectorized")
    if vec_success > 0:
        findings.append(f"Successful vectorizations: {vec_success} loops vectorized")
    if inline_missed > 0:
        findings.append(f"Missed inlines: {inline_missed} call sites not inlined")

    return findings


def _parse_nvcc_diagnostics(stderr: str) -> list[str]:
    """Parse NVCC/ptxas diagnostics from --ptxas-options=-v output."""
    findings: list[str] = []

    for line in stderr.splitlines():
        line_stripped = line.strip()

        # ptxas info: Used N registers, M bytes smem, ...
        m = re.search(
            r"ptxas\s+info\s*:\s*Used\s+(\d+)\s+registers",
            line_stripped, re.IGNORECASE,
        )
        if m:
            regs = int(m.group(1))
            findings.append(f"Registers/thread: {regs}")
            # 64 regs: 65536 regs/SM ÷ 64 regs/thread = 1024 max threads; >64 caps occupancy
            if regs > 64:
                findings.append(f"WARNING: High register usage ({regs} > 64) may limit occupancy")

        m_smem = re.search(
            r"(\d+)\s+bytes\s+smem",
            line_stripped, re.IGNORECASE,
        )
        if m_smem:
            smem = int(m_smem.group(1))
            findings.append(f"Shared memory: {smem} bytes")

        # Spill stores/loads
        m_spill_st = re.search(r"(\d+)\s+bytes\s+spill\s+stores", line_stripped, re.IGNORECASE)
        m_spill_ld = re.search(r"(\d+)\s+bytes\s+spill\s+loads", line_stripped, re.IGNORECASE)
        if m_spill_st:
            findings.append(f"WARNING: Spill stores: {m_spill_st.group(1)} bytes")
        if m_spill_ld:
            findings.append(f"WARNING: Spill loads: {m_spill_ld.group(1)} bytes")

    return findings


def _parse_pytorch_diagnostics(stderr: str) -> list[str]:
    """Parse PyTorch dynamo/inductor diagnostics from TORCH_LOGS output."""
    findings: list[str] = []
    graph_breaks = 0
    eager_fallbacks = 0
    recompilations = 0
    fusions = 0

    for line in stderr.splitlines():
        lower = line.lower()
        if "graph break" in lower or "graph_break" in lower:
            graph_breaks += 1
        if "eager fallback" in lower or "fallback" in lower and "eager" in lower:
            eager_fallbacks += 1
        if "recompil" in lower:
            recompilations += 1
        if "fuse" in lower or "fusi" in lower:
            fusions += 1

    if graph_breaks > 0:
        findings.append(f"Graph breaks: {graph_breaks} (each forces eager execution boundary)")
    if eager_fallbacks > 0:
        findings.append(f"Eager fallbacks: {eager_fallbacks} ops fell back to eager mode")
    if recompilations > 0:
        findings.append(f"WARNING: Recompilations: {recompilations} (dynamic shapes or cache eviction)")
    if fusions > 0:
        findings.append(f"Fusion events: {fusions}")

    return findings


def _parse_jax_diagnostics(stderr: str) -> list[str]:
    """Parse JAX/XLA diagnostics from JAX_LOG_COMPILES output."""
    findings: list[str] = []
    compilations = 0
    recompilations = 0
    total_compile_ms = 0.0

    for line in stderr.splitlines():
        lower = line.lower()
        # JAX logs compilation events like "Finished XLA compilation of ..."
        if "compil" in lower and ("finished" in lower or "compiled" in lower):
            compilations += 1
            # Try to extract compilation time
            m = re.search(r"in\s+([\d.]+)\s*(?:ms|s)", line, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                unit = line[m.end() - 1:m.end()].lower() if m.end() <= len(line) else "s"
                # Check if the matched unit was ms or s
                if "ms" in line[m.start():m.end() + 2].lower():
                    total_compile_ms += val
                else:
                    total_compile_ms += val * 1000
        if "recompil" in lower:
            recompilations += 1

    if compilations > 0:
        findings.append(f"XLA compilations: {compilations}")
    if total_compile_ms > 0:
        findings.append(f"Total compile time: {total_compile_ms:.0f} ms")
    if recompilations > 0:
        findings.append(f"WARNING: Recompilations: {recompilations} (shape changes or cache misses)")

    return findings


def _parse_triton_diagnostics(stderr: str) -> list[str]:
    """Parse Triton diagnostics from TRITON_DEBUG output."""
    findings: list[str] = []
    compilations = 0

    for line in stderr.splitlines():
        lower = line.lower()

        # Shared memory usage
        m = re.search(r"shared\s*(?:memory|mem)[:\s]+(\d+)\s*(?:bytes)?", lower)
        if m:
            findings.append(f"Shared memory: {m.group(1)} bytes")

        # Register usage
        m = re.search(r"registers[:\s]+(\d+)", lower)
        if m:
            findings.append(f"Registers: {m.group(1)}")

        # num_warps
        m = re.search(r"num_warps[:\s]+(\d+)", lower)
        if m:
            findings.append(f"num_warps: {m.group(1)}")

        if "compil" in lower:
            compilations += 1

    if compilations > 0:
        findings.append(f"Compilation events: {compilations}")

    return findings


# ---------------------------------------------------------------------------
# Dispatcher + summarizer
# ---------------------------------------------------------------------------

_AOT_TYPES = {"cpp", "cuda"}
_JIT_TYPES = {"pytorch", "jax", "triton"}

_PARSER_MAP = {
    "cpp": _parse_gcc_diagnostics,
    "cuda": _parse_nvcc_diagnostics,
    "pytorch": _parse_pytorch_diagnostics,
    "jax": _parse_jax_diagnostics,
    "triton": _parse_triton_diagnostics,
}


@dataclass
class CompilerDiagnostics:
    program_type: str
    findings: list[str] = field(default_factory=list)
    summary: str = ""
    remarks: list[OptimizationRemark] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "program_type": self.program_type,
            "findings": self.findings,
            "summary": self.summary,
            "remarks": [r.to_dict() for r in self.remarks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> CompilerDiagnostics:
        return cls(
            program_type=d["program_type"],
            findings=d.get("findings", []),
            summary=d.get("summary", ""),
            remarks=[OptimizationRemark.from_dict(r) for r in d.get("remarks", [])],
        )


def _summarize_findings(findings: list[str]) -> str:
    """Build a token-budgeted summary string, warnings first."""
    # Sort: WARNING lines first, then the rest
    warnings = [f for f in findings if f.startswith("WARNING")]
    others = [f for f in findings if not f.startswith("WARNING")]
    ordered = warnings + others

    lines = ["- " + f for f in ordered]
    text = "\n".join(lines)

    if len(text) > _MAX_SUMMARY_CHARS:
        text = text[:_MAX_SUMMARY_CHARS - 20] + "\n... (truncated)"

    return text


def parse_compiler_output(
    program_type: str,
    build_stderr: str = "",
    bench_stderr: str = "",
    compiler: str = "gcc",
) -> CompilerDiagnostics:
    """Parse compiler output and return structured diagnostics.

    AOT types (cpp, cuda): parse both build_stderr and bench_stderr.
    JIT types (pytorch, jax, triton): parse bench_stderr only.
    """
    parser = _PARSER_MAP.get(program_type)
    if parser is None:
        return CompilerDiagnostics(program_type=program_type)

    findings: list[str] = []
    if program_type in _AOT_TYPES:
        if build_stderr:
            findings.extend(parser(build_stderr))
        if bench_stderr:
            findings.extend(parser(bench_stderr))
    elif program_type in _JIT_TYPES:
        if bench_stderr:
            findings.extend(parser(bench_stderr))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for f in findings:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    summary = _summarize_findings(unique) if unique else ""

    # Structured remarks for C++ (GCC/Clang) and CUDA (NVCC)
    remarks: list[OptimizationRemark] = []
    combined_stderr = ""
    if build_stderr:
        combined_stderr += build_stderr + "\n"
    if bench_stderr:
        combined_stderr += bench_stderr
    if combined_stderr.strip():
        if program_type == "cpp":
            if compiler in ("clang", "clang++"):
                remarks = _parse_clang_remarks(combined_stderr)
            else:
                remarks = _parse_gcc_remarks(combined_stderr)
        elif program_type == "cuda":
            remarks = _parse_nvcc_remarks(combined_stderr)

    return CompilerDiagnostics(
        program_type=program_type,
        findings=unique,
        summary=summary,
        remarks=remarks,
    )


# ---------------------------------------------------------------------------
# Cross-referencing: compiler remarks × profiler hotspots × ISA
# ---------------------------------------------------------------------------

def _fuzzy_kernel_match(name_a: str, name_b: str) -> bool:
    """Check if two kernel names refer to the same kernel (mangled vs demangled)."""
    a = name_a.lower()
    b = name_b.lower()
    if a == b or a in b or b in a:
        return True
    # Extract base name: strip leading _Z\d+, template args, namespaces
    def _base(s: str) -> str:
        # Demangle prefix: _Z<len><name>...
        m = re.match(r"_Z\d+(\w+)", s)
        if m:
            s = m.group(1)
        return s.split("::")[-1].split("<")[0].split("(")[0].lower()
    base_a = _base(name_a)
    base_b = _base(name_b)
    return bool(base_a and base_b and (base_a in base_b or base_b in base_a))


def cross_reference_diagnostics(
    remarks: list[OptimizationRemark],
    perf_summary: dict,
    cpu_isa: dict | None = None,
    gpu_attribution: list[dict] | None = None,
) -> list[CrossReferencedInsight]:
    """Cross-reference compiler optimization remarks with profiler hotspots and ISA info.

    For C++ remarks: matches by source file:line against perf annotate hotspots.
    For CUDA remarks: matches kernel function names against GPU attribution data.

    Returns prioritized insights for the LLM prompt.
    """
    insights: list[CrossReferencedInsight] = []
    if not remarks and not cpu_isa:
        return insights

    max_simd = (cpu_isa or {}).get("max_simd_width_bits", 0)
    annotated_hotspots = perf_summary.get("annotated_hotspots", [])
    hotspots = perf_summary.get("hotspots", [])

    # Build hot-line index: (basename, line) -> pct
    hot_lines: dict[tuple[str, int], float] = {}
    for hs in annotated_hotspots:
        for hl in hs.get("hot_lines", []):
            basename = os.path.basename(hl.get("file", ""))
            if basename:
                hot_lines[(basename, hl["line"])] = hl["pct"]

    # Also index function-level hotspots for broader matching
    hot_functions: dict[str, float] = {}
    for h in hotspots:
        hot_functions[h.get("function", "")] = h.get("pct", 0)

    # GPU kernel hotspot index: kernel_name -> gpu_pct
    gpu_hot_kernels: dict[str, float] = {}
    if gpu_attribution:
        for a in gpu_attribution:
            name = a.get("name", "")
            if name:
                gpu_hot_kernels[name] = a.get("gpu_pct", 0)

    # ±3 lines: compiler remarks often land on loop headers while profiler samples land on
    # inner-loop bodies; 3 lines bridges this gap without false positives
    window = 3

    for remark in remarks:
        remark_basename = os.path.basename(remark.file)
        loc = f"{remark_basename}:{remark.line}" if remark.line > 0 else remark_basename

        # Find nearby CPU hotspot percentage (file:line match)
        nearby_pct: float | None = None
        if remark.line > 0:
            for (hb, hl), pct in hot_lines.items():
                if hb == remark_basename and abs(hl - remark.line) <= window:
                    if nearby_pct is None or pct > nearby_pct:
                        nearby_pct = pct

        # Find matching GPU kernel percentage (for CUDA remarks where file=kernel name)
        gpu_pct: float | None = None
        if remark.line == 0 and gpu_hot_kernels:
            for kernel_name, kpct in gpu_hot_kernels.items():
                if _fuzzy_kernel_match(remark.file, kernel_name):
                    gpu_pct = kpct
                    break

        # Rule 1: Missed vectorization at perf hotspot
        if remark.category == "vectorize" and remark.status == "missed":
            # 5%: a line with >=5% of CPU samples is a genuine hotspot worth optimizing
            if nearby_pct is not None and nearby_pct >= 5.0:
                insights.append(CrossReferencedInsight(
                    priority="high",
                    category="missed-vec-in-hotspot",
                    source_location=loc,
                    description=f"Missed vectorization at hot line ({nearby_pct:.0f}% of samples): {remark.detail}",
                    suggestion="Add __restrict__ qualifiers, ensure aligned accesses, consider loop restructuring",
                    perf_pct=nearby_pct,
                ))

        # Rule 2: Vectorization width gap
        if (
            remark.category == "vectorize"
            and remark.status == "applied"
            and remark.width is not None
            and max_simd > 0
            and remark.width < max_simd
        ):
            gap_ratio = max_simd / remark.width
            priority = "high" if gap_ratio >= 4 else "medium" if gap_ratio >= 2 else "low"
            insights.append(CrossReferencedInsight(
                priority=priority,
                category="vectorization-gap",
                source_location=loc,
                description=(
                    f"Loop vectorizes at {remark.width}-bit but hardware supports "
                    f"{max_simd}-bit ({gap_ratio:.0f}x gap)"
                ),
                suggestion=f"Compile with -march=native, add __restrict__, ensure {max_simd//8}-byte alignment",
                perf_pct=nearby_pct,
            ))

        # Rule 3: Alias blocking at hotspot
        if remark.category == "alias" and remark.status == "missed":
            prio = "high" if nearby_pct and nearby_pct >= 5.0 else "medium"
            insights.append(CrossReferencedInsight(
                priority=prio,
                category="alias-blocking",
                source_location=loc,
                description=f"Aliasing prevents optimization: {remark.detail}",
                suggestion="Add __restrict__ qualifiers to pointer parameters",
                perf_pct=nearby_pct,
            ))

        # Rule 4: Missed FMA at hot location
        if remark.category == "fma" and remark.status == "missed":
            prio = "high" if nearby_pct and nearby_pct >= 5.0 else "medium"
            insights.append(CrossReferencedInsight(
                priority=prio,
                category="missed-fma",
                source_location=loc,
                description=f"Missed FMA opportunity: {remark.detail}",
                suggestion="Compile with -mfma or -march=native, or use manual fma() intrinsic",
                perf_pct=nearby_pct,
            ))

        # Rule 5: Non-unit stride / complicated access at hotspot
        if (
            remark.status == "missed"
            and ("complicated access" in remark.detail.lower() or "stride" in remark.detail.lower())
        ):
            if nearby_pct is not None and nearby_pct >= 5.0:
                insights.append(CrossReferencedInsight(
                    priority="high",
                    category="non-unit-stride",
                    source_location=loc,
                    description=f"Non-unit stride or complicated access at hot line: {remark.detail}",
                    suggestion="Restructure data layout for unit-stride access (e.g., AoS → SoA, transpose)",
                    perf_pct=nearby_pct,
                ))

        # Rule 6: CUDA register pressure in hot GPU kernel
        if remark.category == "register-pressure" and remark.status == "missed" and gpu_pct is not None:
            # 10%: kernel using >=10% of GPU time with register pressure is high-priority
            prio = "high" if gpu_pct >= 10.0 else "medium"
            insights.append(CrossReferencedInsight(
                priority=prio,
                category="cuda-register-pressure",
                source_location=loc,
                description=f"Register pressure in hot GPU kernel ({gpu_pct:.0f}% GPU time): {remark.detail}",
                suggestion="Reduce register usage with __launch_bounds__, simplify kernel logic, or use shared memory to cache intermediate values",
                perf_pct=gpu_pct,
            ))

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    insights.sort(key=lambda i: priority_order.get(i.priority, 3))

    return insights
