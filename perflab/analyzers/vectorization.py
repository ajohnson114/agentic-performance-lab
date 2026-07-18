"""Auto-vectorization verification from perf annotate output.

Checks whether hot functions contain SIMD instructions (SSE/AVX/NEON)
to verify that the compiler actually vectorized critical loops.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# SIMD instruction mnemonics by ISA. Packed forms only: scalar SSE/AVX ops
# (movss/addss/vmovsd, ... — "ss"/"sd" = single element) say nothing about
# vectorization and previously produced false "SIMD present" verdicts.
_SSE_RE = re.compile(r"\b(movaps|movups|addps|mulps|subps|divps|xorps|andps|"
                     r"movapd|movupd|addpd|mulpd|subpd|divpd|maxps|minps|"
                     r"shufps)\b", re.I)
_AVX_RE = re.compile(r"\b(vmovaps|vmovups|vaddps|vmulps|vsubps|vfmadd|vfmsub|"
                     r"vfnmadd|vfnmsub|vmovapd|vaddpd|vmulpd|vsubpd|"
                     r"vbroadcast|vperm|vblend|vzeroall|vzeroupper|"
                     r"vxorps|vandps|vorps)\b", re.I)
_AVX512_RE = re.compile(r"\b(zmm\d+|vmovaps\s.*zmm|vaddps\s.*zmm|"
                        r"vmulps\s.*zmm|vfmadd\d+ps\s.*zmm)\b", re.I)
# AArch64 mnemonics like fadd/and/shl/mov also exist in scalar form, so the
# ambiguous group only counts when the operand is a vector register (v0.4s);
# ld1-4/st1-4/addv/faddp are inherently vector.
_NEON_RE = re.compile(r"\b(?:fmla|fmul|fadd|fsub|fdiv|fmaxnm|fminnm|"
                      r"dup|ins|movi?|shl|ushr|sshr|and|orr)\s+v\d|"
                      r"\b(?:ld[1-4]|st[1-4]|addv|faddp)\b", re.I)


@dataclass
class VectorizationReport:
    """Per-function vectorization analysis."""
    function: str
    has_simd: bool
    simd_isa: str  # "none", "sse", "avx", "avx512", "neon"
    simd_instruction_count: int = 0
    total_instruction_count: int = 0
    simd_ratio: float = 0.0  # fraction of instructions that are SIMD
    hot_pct: float = 0.0  # CPU percentage from perf annotate


@dataclass
class VectorizationSummary:
    """Overall vectorization summary."""
    functions: list[VectorizationReport] = field(default_factory=list)
    vectorized_count: int = 0
    not_vectorized_count: int = 0
    warning: str = ""


def check_vectorization_from_perf_annotate(annotate_text: str) -> VectorizationSummary:
    """Analyze raw perf annotate --stdio text for SIMD instructions.

    This works directly with the raw text rather than parsed hotspots.
    """
    summary = VectorizationSummary()

    func_header_re = re.compile(r"Source code & Disassembly of\s+(\S+)")
    pct_line_re = re.compile(r"^\s*([\d.]+)\s+:")
    # Disassembly lines carry "<addr>: <insn>" after the percent column
    # (e.g. "  45.20 :  14: vmovaps %ymm0, (%rax)"). Interleaved source
    # lines have only the bare percent colon — matching mnemonics against
    # them counted source tokens like `and`/`shl` as SIMD and inflated the
    # simd_ratio denominator.
    asm_line_re = re.compile(r"^\s*(?:[\d.]+\s+)?:\s*[0-9a-fA-F]+:\s+(.+)$")

    current_func = None
    simd_count = 0
    total_asm_lines = 0
    hot_pct = 0.0
    best_isa = "none"

    def _flush():
        nonlocal current_func, simd_count, total_asm_lines, hot_pct, best_isa
        if current_func:
            report = VectorizationReport(
                function=current_func,
                has_simd=simd_count > 0,
                simd_isa=best_isa,
                simd_instruction_count=simd_count,
                total_instruction_count=total_asm_lines,
                simd_ratio=simd_count / total_asm_lines if total_asm_lines > 0 else 0.0,
                hot_pct=hot_pct,
            )
            summary.functions.append(report)
            if report.has_simd:
                summary.vectorized_count += 1
            else:
                summary.not_vectorized_count += 1
        current_func = None
        simd_count = 0
        total_asm_lines = 0
        hot_pct = 0.0
        best_isa = "none"

    for line in annotate_text.splitlines():
        m = func_header_re.search(line)
        if m:
            _flush()
            current_func = m.group(1)
            continue

        if current_func is None:
            continue

        # Track percentage lines
        pm = pct_line_re.match(line)
        if pm:
            try:
                hot_pct += float(pm.group(1))
            except ValueError:
                pass

        am = asm_line_re.match(line)
        if am is None:
            continue
        asm_part = am.group(1)

        total_asm_lines += 1

        if _AVX512_RE.search(asm_part):
            simd_count += 1
            if best_isa in ("none", "sse", "avx"):
                best_isa = "avx512"
        elif _AVX_RE.search(asm_part):
            simd_count += 1
            if best_isa in ("none", "sse"):
                best_isa = "avx"
        elif _SSE_RE.search(asm_part):
            simd_count += 1
            if best_isa == "none":
                best_isa = "sse"
        elif _NEON_RE.search(asm_part):
            simd_count += 1
            if best_isa == "none":
                best_isa = "neon"

    _flush()

    if summary.not_vectorized_count > 0:
        unvectorized = [f.function for f in summary.functions if not f.has_simd]
        summary.warning = (
            f"{summary.not_vectorized_count} hot function(s) lack SIMD instructions: "
            f"{', '.join(unvectorized[:3])}. "
            f"Consider -O3 -march=native or manual vectorization."
        )

    return summary


def format_vectorization_for_prompt(summary: VectorizationSummary) -> str:
    """Format vectorization summary for inclusion in LLM prompt."""
    if not summary.functions:
        return ""

    lines = ["Vectorization report:"]
    for f in summary.functions:
        status = f"SIMD ({f.simd_isa})" if f.has_simd else "NO SIMD"
        lines.append(f"  {f.function}: {status} ({f.hot_pct:.1f}% CPU)")

    if summary.warning:
        lines.append(f"WARNING: {summary.warning}")

    return "\n".join(lines)
