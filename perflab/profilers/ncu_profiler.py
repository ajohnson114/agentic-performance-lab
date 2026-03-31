from __future__ import annotations

import csv
import io
import logging
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from perflab.profilers.base import ProfileResult
from perflab.tools.shell import run_cmd

logger = logging.getLogger(__name__)


@dataclass
class NcuProfiler:
    name: str = "ncu"

    def is_available(self) -> bool:
        return shutil.which("ncu") is not None

    def run(self, bench_cmd: str, cwd: Path, artifacts_dir: Path) -> ProfileResult:
        csv_path = artifacts_dir / "ncu_metrics.csv"
        report_path = artifacts_dir / "ncu_report.ncu-rep"
        cmd_parts = shlex.split(bench_cmd)

        # ncu with CSV output for programmatic analysis
        csv_cmd = [
            "ncu", "--set", "full", "--csv",
            "--log-file", str(csv_path),
        ] + cmd_parts
        csv_res = run_cmd(csv_cmd, cwd=cwd)

        # Also generate a report file for interactive viewing
        report_cmd = [
            "ncu", "--set", "full",
            "-o", str(report_path),
        ] + cmd_parts
        report_res = run_cmd(report_cmd, cwd=cwd)

        summary = _parse_ncu_csv(csv_path)
        summary["csv_returncode"] = csv_res.returncode
        summary["report_returncode"] = report_res.returncode

        artifacts: dict[str, str] = {}
        if csv_path.exists():
            artifacts["ncu_metrics_csv"] = str(csv_path)
        if report_path.exists():
            artifacts["ncu_report"] = str(report_path)

        return ProfileResult(name=self.name, artifacts=artifacts, summary=summary)


def extract_cuda_sass(
    binary_path: Path,
    *,
    max_kernels: int = 3,
    context_lines: int = 10,
    artifacts_dir: Path | None = None,
) -> list[dict]:
    """Extract SASS disassembly from a CUDA binary using cuobjdump.

    Runs ``cuobjdump --dump-sass <binary>`` and parses the output to extract
    per-kernel SASS listings. Returns a list of dicts:
        {kernel, snippet, instruction_count}
    where *snippet* is the SASS listing for the kernel (truncated to a
    reasonable size for LLM consumption).
    """
    if not binary_path.exists():
        return []
    if not shutil.which("cuobjdump"):
        return []

    cmd = ["cuobjdump", "--dump-sass", str(binary_path)]
    res = run_cmd(cmd, cwd=binary_path.parent)
    if res.returncode != 0 or not res.stdout:
        return []

    # Optionally save raw output
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "sass_dump.txt").write_text(res.stdout, encoding="utf-8")

    return _parse_sass_dump(res.stdout, max_kernels=max_kernels, context_lines=context_lines)


def _parse_sass_dump(
    text: str,
    *,
    max_kernels: int = 3,
    context_lines: int = 10,
) -> list[dict]:
    """Parse cuobjdump --dump-sass output into per-kernel snippets.

    cuobjdump output format:
        code for sm_XX
            Function : _Z12kernel_namePfS_S_iii
            .headerflags ...
            /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
            /*0010*/ S2R R0, SR_CTAID.X ;
            ...

    Returns list of {kernel, snippet, instruction_count}.
    """
    import re

    results: list[dict] = []
    func_re = re.compile(r"^\s*Function\s*:\s*(\S+)")
    inst_re = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s+(.+)")

    current_kernel: str | None = None
    current_lines: list[str] = []
    inst_count = 0

    for line in text.splitlines():
        m = func_re.match(line)
        if m:
            # Save previous kernel
            if current_kernel is not None and current_lines:
                results.append({
                    "kernel": _demangle_kernel_name(current_kernel),
                    "snippet": "\n".join(current_lines),
                    "instruction_count": inst_count,
                })
            current_kernel = m.group(1)
            current_lines = []
            inst_count = 0
            continue

        if current_kernel is not None:
            mi = inst_re.match(line)
            if mi:
                inst_count += 1
                current_lines.append(line.rstrip())

    # Save last kernel
    if current_kernel is not None and current_lines:
        results.append({
            "kernel": _demangle_kernel_name(current_kernel),
            "snippet": "\n".join(current_lines),
            "instruction_count": inst_count,
        })

    # Sort by instruction count (largest kernels first — likely the hot ones)
    results.sort(key=lambda r: r["instruction_count"], reverse=True)

    # Classify instructions for efficiency analysis (before truncation)
    for r in results:
        efficiency = classify_sass_instructions(r["snippet"])
        if efficiency:
            r["instruction_efficiency"] = efficiency

    # Truncate snippets for LLM consumption — show first and last N lines
    for r in results:
        lines = r["snippet"].splitlines()
        if len(lines) > context_lines * 2:
            head = lines[:context_lines]
            tail = lines[-context_lines:]
            omitted = len(lines) - context_lines * 2
            r["snippet"] = "\n".join(head + [f"    ... ({omitted} instructions omitted) ..."] + tail)

    return results[:max_kernels]


def _demangle_kernel_name(mangled: str) -> str:
    """Best-effort demangling of C++ kernel names.

    Tries c++filt if available, otherwise extracts the base name from
    the mangled symbol.
    """
    if shutil.which("c++filt"):
        try:
            res = run_cmd(["c++filt", mangled], cwd=Path("."))
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except (OSError, ValueError):
            pass

    # Fallback: extract name between _Z<len> prefix and parameter types
    import re
    m = re.match(r"_Z\d+(\w+)", mangled)
    if m:
        return m.group(1)
    return mangled


def classify_sass_instructions(sass_text: str) -> dict:
    """Classify SASS instructions into categories for efficiency analysis.

    Returns a dict with:
      - category counts: compute, tensor_core, memory_global, memory_shared,
        address_math, control_flow, sync, other
      - total instruction count
      - efficiency_pct (useful compute as % of total)
      - category_pcts (each category as % of total)
    """
    import re

    # SASS instruction classification patterns
    _CATEGORIES: dict[str, list[str]] = {
        "compute": [
            "FFMA", "FMUL", "FADD", "FSUB", "FMNMX", "FSET", "FSETP",
            "HFMA2", "HMUL2", "HADD2",  # FP16 packed
            "DFMA", "DMUL", "DADD",  # FP64
            "MUFU",  # transcendentals (sin, cos, rsqrt, exp, log)
        ],
        "tensor_core": [
            "HMMA", "HGMMA", "QGMMA", "IGMMA",  # Tensor Core MMA
            "WGMMA",  # Warp Group MMA
        ],
        "memory_global": [
            "LDG", "STG",  # Global load/store
            "LDGSTS",  # Async global→shared copy (cp.async)
            "RED",  # Atomic reduction to global
            "ATOM",  # Atomic operations
        ],
        "memory_shared": [
            "LDS", "STS",  # Shared memory load/store
            "LDSM",  # Load shared memory matrix (for TC)
        ],
        "address_math": [
            "IMAD", "IADD", "IADD3", "ISCADD",  # Integer multiply-add, add
            "LEA", "SHL", "SHR",  # Shift, lea
            "LOP3", "LOP",  # Logic ops (often used for address computation)
            "IMNMX",  # Integer min/max
        ],
        "control_flow": [
            "BRA", "BRX", "JMP", "JMXU",  # Branches/jumps
            "EXIT", "RET", "BREAK", "CONT",  # Exit/return
            "ISETP", "ISET", "FSETP",  # Predicate set (comparison for branching)
            "@P", "@!P",  # Predicated instructions (counted as control overhead)
        ],
        "sync": [
            "BAR", "MEMBAR", "DEPBAR",  # Barriers
            "FENCE",  # Memory fences
        ],
        "register_move": [
            "MOV", "PRMT", "SHFL", "S2R", "CS2R",  # Moves, shuffles, special reg reads
            "R2P", "P2R",  # Register-predicate conversions
        ],
    }

    inst_re = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?(\w+)")
    counts: dict[str, int] = {cat: 0 for cat in _CATEGORIES}
    counts["other"] = 0
    total = 0

    for line in sass_text.splitlines():
        m = inst_re.search(line)
        if not m:
            continue
        opcode = m.group(1).upper()
        total += 1

        classified = False
        for cat, opcodes in _CATEGORIES.items():
            for op in opcodes:
                if opcode.startswith(op):
                    counts[cat] += 1
                    classified = True
                    break
            if classified:
                break
        if not classified:
            counts["other"] += 1

    if total == 0:
        return {}

    result: dict = {
        "total_instructions": total,
        "categories": {cat: count for cat, count in counts.items() if count > 0},
        "category_pcts": {
            cat: round(count / total * 100, 1)
            for cat, count in counts.items() if count > 0
        },
    }

    # Efficiency: useful compute (FMA + tensor core) as % of total
    useful = counts["compute"] + counts["tensor_core"]
    result["efficiency_pct"] = round(useful / total * 100, 1)

    # Overhead: address math + control flow as % of total
    overhead = counts["address_math"] + counts["control_flow"]
    result["overhead_pct"] = round(overhead / total * 100, 1)

    return result


def _find_column(headers: list[str], *keywords: str) -> str | None:
    """Find a column header containing all keywords (case-insensitive)."""
    for h in headers:
        h_lower = h.lower()
        if all(kw in h_lower for kw in keywords):
            return h
    return None


def _safe_float(val: str | None) -> float | None:
    """Parse a numeric string, stripping % signs. Returns None on failure."""
    if val is None:
        return None
    try:
        return float(val.strip().rstrip("%").replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _safe_int(val: str | None) -> int | None:
    """Parse an integer string. Returns None on failure."""
    if val is None:
        return None
    try:
        return int(float(val.strip().rstrip("%").replace(",", "")))
    except (ValueError, AttributeError):
        return None


def _parse_ncu_csv(path: Path) -> dict:
    """Extract per-kernel metrics from ncu CSV output.

    Groups rows by kernel name, computes per-kernel and weighted-average
    aggregate metrics.
    """
    result: dict = {}
    if not path.exists():
        return result

    text = path.read_text(encoding="utf-8", errors="replace")

    # ncu CSV may have header lines starting with "=="; skip them
    csv_lines = [
        line for line in text.splitlines()
        if line and not line.startswith("==")
    ]
    if not csv_lines:
        return result

    try:
        reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
        rows = list(reader)
    except (csv.Error, ValueError):
        logger.warning("Failed to parse NCU CSV output", exc_info=True)
        return result

    if not rows:
        return result

    headers = list(rows[0].keys()) if rows else []

    # Detect the kernel name column
    kernel_name_col = _find_column(headers, "kernel", "name")
    if not kernel_name_col:
        # Fallback: try just "kernel"
        kernel_name_col = _find_column(headers, "kernel")

    # Map metric names to column headers
    sm_col = (
        _find_column(headers, "sm", "throughput")
        or _find_column(headers, "sm", "utilization")
        or _find_column(headers, "sm", "active")
        or _find_column(headers, "sm", "busy")
    )
    mem_col = (
        _find_column(headers, "dram", "throughput")
        or _find_column(headers, "memory", "throughput")
        or _find_column(headers, "memory", "bandwidth")
    )
    occ_col = (
        _find_column(headers, "occupancy", "achieved")
        or _find_column(headers, "warps_active", "pct")
        or _find_column(headers, "occupancy")
    )
    compute_col = (
        _find_column(headers, "compute", "throughput")
        or _find_column(headers, "sm__pipe", "utilization")
    )
    l1_col = _find_column(headers, "l1", "hit_rate")
    l2_col = _find_column(headers, "l2", "hit_rate")
    reg_col = _find_column(headers, "registers_per_thread") or _find_column(headers, "registers", "thread")
    smem_col = (
        _find_column(headers, "shared_mem_per_block")
        or _find_column(headers, "shared_mem")
    )

    # Tensor Core columns (available when ncu runs with --set full)
    tensor_active_col = (
        _find_column(headers, "tensor", "active")
        or _find_column(headers, "pipe_tensor", "cycles")
        or _find_column(headers, "tensor", "utilization")
    )
    tensor_throughput_col = (
        _find_column(headers, "tensor", "throughput")
        or _find_column(headers, "pipe_tensor", "throughput")
    )

    # Branch divergence columns (available when ncu runs with --set full)
    branch_eff_col = (
        _find_column(headers, "branch", "efficiency")
        or _find_column(headers, "branch", "uniform")
    )
    warp_exec_eff_col = (
        _find_column(headers, "thread", "inst", "executed", "per")
        or _find_column(headers, "warp", "execution", "efficiency")
    )

    # Warp stall reason columns (ncu --set full provides these)
    _STALL_REASONS = [
        "long_scoreboard", "short_scoreboard", "barrier", "lg_throttle",
        "not_selected", "wait", "math_pipe_throttle", "mio_throttle",
        "drain", "tex_throttle", "imc_miss", "dispatch_stall",
        "memory_throttle", "membar", "gmma",
    ]
    stall_cols: dict[str, str | None] = {}
    for reason in _STALL_REASONS:
        stall_cols[reason] = (
            _find_column(headers, "stalled", reason)
            or _find_column(headers, "stall", reason)
            or _find_column(headers, reason, "stall")
        )

    # Shared memory bank conflicts
    bank_conflict_col = (
        _find_column(headers, "bank", "conflict")
        or _find_column(headers, "bank_conflicts")
    )

    # Memory coalescing: sectors per request (>1.0 means uncoalesced)
    sectors_per_req_col = (
        _find_column(headers, "sectors_per_request")
        or _find_column(headers, "sectors", "request")
        or _find_column(headers, "average", "sectors")
    )

    # Occupancy limiters
    occ_limit_reg_col = _find_column(headers, "occupancy", "limit", "register")
    occ_limit_smem_col = _find_column(headers, "occupancy", "limit", "shared")
    occ_limit_block_col = (
        _find_column(headers, "occupancy", "limit", "block")
        or _find_column(headers, "occupancy", "limit", "warp")
    )
    theoretical_occ_col = (
        _find_column(headers, "occupancy", "theoretical")
        or _find_column(headers, "theoretical", "occupancy")
    )

    # Instruction mix: pipe utilization breakdown
    pipe_fp32_col = (
        _find_column(headers, "pipe_fma", "active")
        or _find_column(headers, "pipe_fma", "utilization")
        or _find_column(headers, "fp32", "pipe")
    )
    pipe_fp64_col = (
        _find_column(headers, "pipe_fp64")
        or _find_column(headers, "fp64", "pipe")
        or _find_column(headers, "inst_executed", "fp64")
    )
    pipe_int_col = (
        _find_column(headers, "pipe_alu")
        or _find_column(headers, "int", "pipe")
    )
    pipe_sfu_col = (
        _find_column(headers, "pipe_xu")
        or _find_column(headers, "sfu", "pipe")
        or _find_column(headers, "pipe_sfu")
    )

    # TMA pipe utilization (Hopper+)
    pipe_tma_col = (
        _find_column(headers, "pipe_tma", "active")
        or _find_column(headers, "pipe_tma", "utilization")
        or _find_column(headers, "tma", "pipe")
    )

    # Register spill: local memory bytes (indicates register spilling)
    local_mem_col = (
        _find_column(headers, "local", "memory", "bytes")
        or _find_column(headers, "local_memory")
        or _find_column(headers, "spill", "bytes")
        or _find_column(headers, "lmem", "bytes")
    )

    # DRAM byte columns (available when ncu runs with --set full)
    dram_read_col = (
        _find_column(headers, "dram", "bytes", "read")
        or _find_column(headers, "dram__bytes_read")
    )
    dram_write_col = (
        _find_column(headers, "dram", "bytes", "write")
        or _find_column(headers, "dram__bytes_write")
    )
    duration_ns_col = (
        _find_column(headers, "duration")
        or _find_column(headers, "gpu__time_duration")
    )

    # Source-line columns
    source_file_col = _find_column(headers, "source", "file") or _find_column(headers, "source")
    source_line_col = _find_column(headers, "line")
    source_func_col = _find_column(headers, "function")

    # Group rows by kernel name
    kernel_groups: dict[str, list[dict]] = {}
    for row in rows:
        kname = row.get(kernel_name_col, "(unknown)") if kernel_name_col else "(unknown)"
        kernel_groups.setdefault(kname, []).append(row)

    # Extract per-kernel metrics
    per_kernel: list[dict] = []
    for kname, krows in kernel_groups.items():
        invocations = len(krows)
        sm_vals = [v for r in krows if (v := _safe_float(r.get(sm_col))) is not None] if sm_col else []
        mem_vals = [v for r in krows if (v := _safe_float(r.get(mem_col))) is not None] if mem_col else []
        occ_vals = [v for r in krows if (v := _safe_float(r.get(occ_col))) is not None] if occ_col else []
        compute_vals = [v for r in krows if (v := _safe_float(r.get(compute_col))) is not None] if compute_col else []
        l1_vals = [v for r in krows if (v := _safe_float(r.get(l1_col))) is not None] if l1_col else []
        l2_vals = [v for r in krows if (v := _safe_float(r.get(l2_col))) is not None] if l2_col else []
        reg_vals = [v for r in krows if (v := _safe_int(r.get(reg_col))) is not None] if reg_col else []
        smem_vals = [v for r in krows if (v := _safe_int(r.get(smem_col))) is not None] if smem_col else []
        branch_eff_vals = [v for r in krows if (v := _safe_float(r.get(branch_eff_col))) is not None] if branch_eff_col else []
        warp_exec_vals = [v for r in krows if (v := _safe_float(r.get(warp_exec_eff_col))) is not None] if warp_exec_eff_col else []
        tensor_active_vals = [v for r in krows if (v := _safe_float(r.get(tensor_active_col))) is not None] if tensor_active_col else []
        tensor_throughput_vals = [v for r in krows if (v := _safe_float(r.get(tensor_throughput_col))) is not None] if tensor_throughput_col else []

        entry: dict = {"name": kname, "invocations": invocations}
        if sm_vals:
            entry["sm_utilization_pct"] = round(sum(sm_vals) / len(sm_vals), 1)
        if mem_vals:
            entry["memory_throughput_pct"] = round(sum(mem_vals) / len(mem_vals), 1)
        if occ_vals:
            entry["achieved_occupancy_pct"] = round(sum(occ_vals) / len(occ_vals), 1)
        if compute_vals:
            entry["compute_throughput_pct"] = round(sum(compute_vals) / len(compute_vals), 1)
        if l1_vals:
            entry["l1_hit_rate"] = round(sum(l1_vals) / len(l1_vals), 1)
        if l2_vals:
            entry["l2_hit_rate"] = round(sum(l2_vals) / len(l2_vals), 1)
        if reg_vals:
            entry["registers_per_thread"] = max(reg_vals)
        if smem_vals:
            entry["shared_mem_per_block_bytes"] = max(smem_vals)
        if branch_eff_vals:
            entry["branch_efficiency_pct"] = round(sum(branch_eff_vals) / len(branch_eff_vals), 1)
        if warp_exec_vals:
            entry["warp_execution_efficiency_pct"] = round(sum(warp_exec_vals) / len(warp_exec_vals), 1)
        if tensor_active_vals:
            entry["tensor_core_utilization_pct"] = round(sum(tensor_active_vals) / len(tensor_active_vals), 1)
        if tensor_throughput_vals:
            entry["tensor_core_throughput_pct"] = round(sum(tensor_throughput_vals) / len(tensor_throughput_vals), 1)

        # Warp stall reasons (per-kernel dominant stall)
        stall_pcts: dict[str, float] = {}
        for reason, col in stall_cols.items():
            if col:
                vals = [v for r in krows if (v := _safe_float(r.get(col))) is not None]
                if vals:
                    stall_pcts[reason] = round(sum(vals) / len(vals), 1)
        if stall_pcts:
            entry["warp_stall_reasons"] = stall_pcts
            dominant_stall = max(stall_pcts, key=stall_pcts.get)  # type: ignore[arg-type]
            entry["dominant_stall_reason"] = dominant_stall
            entry["dominant_stall_pct"] = stall_pcts[dominant_stall]

        # Bank conflicts
        if bank_conflict_col:
            bc_vals = [v for r in krows if (v := _safe_float(r.get(bank_conflict_col))) is not None]
            if bc_vals:
                entry["bank_conflicts"] = round(sum(bc_vals), 0)

        # Sectors per request (coalescing efficiency)
        if sectors_per_req_col:
            spr_vals = [v for r in krows if (v := _safe_float(r.get(sectors_per_req_col))) is not None]
            if spr_vals:
                entry["sectors_per_request"] = round(sum(spr_vals) / len(spr_vals), 2)

        # Occupancy limiters
        if occ_limit_reg_col:
            olr = [v for r in krows if (v := _safe_float(r.get(occ_limit_reg_col))) is not None]
            if olr:
                entry["occupancy_limit_registers_pct"] = round(sum(olr) / len(olr), 1)
        if occ_limit_smem_col:
            ols = [v for r in krows if (v := _safe_float(r.get(occ_limit_smem_col))) is not None]
            if ols:
                entry["occupancy_limit_shared_mem_pct"] = round(sum(ols) / len(ols), 1)
        if occ_limit_block_col:
            olb = [v for r in krows if (v := _safe_float(r.get(occ_limit_block_col))) is not None]
            if olb:
                entry["occupancy_limit_block_pct"] = round(sum(olb) / len(olb), 1)
        if theoretical_occ_col:
            tocc = [v for r in krows if (v := _safe_float(r.get(theoretical_occ_col))) is not None]
            if tocc:
                entry["theoretical_occupancy_pct"] = round(sum(tocc) / len(tocc), 1)

        # Instruction mix: pipe utilization
        inst_mix: dict[str, float] = {}
        for pipe_name, pipe_col in [
            ("fp32_fma", pipe_fp32_col), ("fp64", pipe_fp64_col),
            ("int_alu", pipe_int_col), ("sfu", pipe_sfu_col),
        ]:
            if pipe_col:
                pv = [v for r in krows if (v := _safe_float(r.get(pipe_col))) is not None]
                if pv:
                    inst_mix[pipe_name] = round(sum(pv) / len(pv), 1)
        if inst_mix:
            entry["instruction_mix"] = inst_mix

        # TMA pipe utilization (Hopper+)
        if pipe_tma_col:
            tma_vals = [v for r in krows if (v := _safe_float(r.get(pipe_tma_col))) is not None]
            if tma_vals:
                entry["tma_pipe_utilization_pct"] = round(sum(tma_vals) / len(tma_vals), 1)

        # Register spill: local memory bytes
        if local_mem_col:
            lm_vals = [v for r in krows if (v := _safe_float(r.get(local_mem_col))) is not None]
            if lm_vals:
                total_local = sum(lm_vals)
                if total_local > 0:
                    entry["local_memory_bytes"] = round(total_local, 0)

        # DRAM bytes per kernel
        if dram_read_col or dram_write_col:
            dr_vals = [v for r in krows if (v := _safe_float(r.get(dram_read_col))) is not None] if dram_read_col else []
            dw_vals = [v for r in krows if (v := _safe_float(r.get(dram_write_col))) is not None] if dram_write_col else []
            k_dram_read = sum(dr_vals)
            k_dram_write = sum(dw_vals)
            k_dram_total = k_dram_read + k_dram_write
            if k_dram_read > 0:
                entry["dram_bytes_read"] = k_dram_read
            if k_dram_write > 0:
                entry["dram_bytes_written"] = k_dram_write
            if k_dram_total > 0:
                entry["dram_bytes_total"] = k_dram_total
                # Per-kernel achieved bandwidth
                if duration_ns_col:
                    dur_vals = [v for r in krows if (v := _safe_float(r.get(duration_ns_col))) is not None]
                    k_dur_ns = sum(dur_vals)
                    if k_dur_ns > 0:
                        entry["achieved_bw_gbs"] = round(k_dram_total / (k_dur_ns * 1e-9) / 1e9, 2)

        # Enrich with source info from first row if available
        if source_file_col or source_func_col:
            first_row = krows[0]
            if source_file_col:
                sf = first_row.get(source_file_col)
                if sf:
                    entry["source_file"] = sf.strip()
            if source_func_col:
                fn = first_row.get(source_func_col)
                if fn:
                    entry["function"] = fn.strip()

        per_kernel.append(entry)

    # Source-line hotspots: group by (source_file, function, line), sum a metric
    if source_file_col or source_func_col:
        # Pick a duration/stall metric column for aggregation
        duration_col = (
            _find_column(headers, "duration")
            or _find_column(headers, "elapsed")
            or _find_column(headers, "stall")
            or sm_col
        )
        hotspot_key_sums: dict[tuple, float] = {}
        metric_name_used = duration_col or "invocations"
        for row in rows:
            sf = (row.get(source_file_col, "") if source_file_col else "").strip()
            fn = (row.get(source_func_col, "") if source_func_col else "").strip()
            ln = _safe_int(row.get(source_line_col)) if source_line_col else None
            key = (sf, fn, ln)
            if not sf and not fn:
                continue
            if duration_col:
                val = _safe_float(row.get(duration_col))
                hotspot_key_sums[key] = hotspot_key_sums.get(key, 0.0) + (val or 0.0)
            else:
                hotspot_key_sums[key] = hotspot_key_sums.get(key, 0.0) + 1.0

        if hotspot_key_sums:
            sorted_hotspots = sorted(hotspot_key_sums.items(), key=lambda x: x[1], reverse=True)[:5]
            result["source_hotspots"] = [
                {
                    "file": k[0] or "",
                    "function": k[1] or "",
                    "line": k[2],
                    "metric_name": metric_name_used or "count",
                    "value": round(v, 2),
                }
                for k, v in sorted_hotspots
            ]

    # Sort by invocations descending, keep top 5
    per_kernel.sort(key=lambda k: k["invocations"], reverse=True)
    result["kernels"] = per_kernel[:5]

    # Dominant kernel
    if per_kernel:
        result["dominant_kernel"] = per_kernel[0]

    # Weighted-average aggregates (weighted by invocation count)
    total_invocations = sum(k["invocations"] for k in per_kernel)
    if total_invocations > 0:
        for metric in (
            "sm_utilization_pct", "memory_throughput_pct", "achieved_occupancy_pct",
            "branch_efficiency_pct", "warp_execution_efficiency_pct",
            "tensor_core_utilization_pct", "tensor_core_throughput_pct",
        ):
            weighted_sum = sum(
                k.get(metric, 0) * k["invocations"]
                for k in per_kernel
                if metric in k
            )
            weighted_count = sum(
                k["invocations"] for k in per_kernel if metric in k
            )
            if weighted_count > 0:
                result[metric] = round(weighted_sum / weighted_count, 1)

    # Propagate dominant kernel's new metrics to result level for bottleneck analyzer
    if per_kernel:
        dk = per_kernel[0]
        if "dominant_stall_reason" in dk:
            result["dominant_stall_reason"] = dk["dominant_stall_reason"]
            result["dominant_stall_pct"] = dk["dominant_stall_pct"]
        if "warp_stall_reasons" in dk:
            result["warp_stall_reasons"] = dk["warp_stall_reasons"]
        if "bank_conflicts" in dk:
            result["bank_conflicts"] = dk["bank_conflicts"]
        if "sectors_per_request" in dk:
            result["sectors_per_request"] = dk["sectors_per_request"]
        if "theoretical_occupancy_pct" in dk:
            result["theoretical_occupancy_pct"] = dk["theoretical_occupancy_pct"]
        for occ_limiter in ("occupancy_limit_registers_pct", "occupancy_limit_shared_mem_pct",
                            "occupancy_limit_block_pct"):
            if occ_limiter in dk:
                result[occ_limiter] = dk[occ_limiter]
        if "instruction_mix" in dk:
            result["instruction_mix"] = dk["instruction_mix"]
        if "tma_pipe_utilization_pct" in dk:
            result["tma_pipe_utilization_pct"] = dk["tma_pipe_utilization_pct"]
        if "local_memory_bytes" in dk:
            result["local_memory_bytes"] = dk["local_memory_bytes"]

    # Aggregate DRAM totals across all kernels
    if dram_read_col or dram_write_col:
        total_dram_read = sum(k.get("dram_bytes_read", 0) for k in per_kernel)
        total_dram_write = sum(k.get("dram_bytes_written", 0) for k in per_kernel)
        total_dram = total_dram_read + total_dram_write
        if total_dram_read > 0:
            result["dram_bytes_read_total"] = total_dram_read
        if total_dram_write > 0:
            result["dram_bytes_written_total"] = total_dram_write
        if total_dram > 0:
            result["dram_bytes_total"] = total_dram
            # Aggregate achieved bandwidth
            if duration_ns_col:
                total_dur_ns = 0.0
                for kname_d, krows_d in kernel_groups.items():
                    dur_vals_d = [v for r in krows_d if (v := _safe_float(r.get(duration_ns_col))) is not None]
                    total_dur_ns += sum(dur_vals_d)
                if total_dur_ns > 0:
                    result["achieved_bw_gbs"] = round(total_dram / (total_dur_ns * 1e-9) / 1e9, 2)

    result["kernel_count"] = len(rows)
    return result
