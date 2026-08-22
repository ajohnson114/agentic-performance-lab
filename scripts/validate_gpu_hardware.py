#!/usr/bin/env python3
"""Real-hardware validation for perflab's GPU/TPU profiler-parsing code.

docs/ENGINEERING_RATIONALE.md's "Validation Coverage" section flags a real,
self-acknowledged gap: no GPU or TPU has ever been in the loop for an
automated test of this repository. ~3800 lines of GPU/TPU parsing code
(nsys_profiler.py, ncu_profiler.py, gpu_attribution.py, jax_profiler.py, the
whole TPU bottleneck path -- including everything added in the multi-GPU/
multi-TPU work this session) is validated only against hand-written fixtures
that approximate a profiler output format observed once. That class of bug
-- a profiler renaming a column, a metric name that no longer exists, a
join that silently returns nothing -- is invisible to the fixture suite by
construction, because the same person wrote both the fixture and the parser
from the same one-time sample.

This script closes that gap for a single manual/occasional run, not a CI
gate: it runs perflab's REAL profiling pipeline (perflab.orchestrator.
profile_only, the same function `perflab profile` calls) against whatever
real hardware is visible, then feeds the REAL artifacts through perflab's
own parsing functions and prints the result for a human to eyeball. It is
a diagnostic pass, not pass/fail -- "does this still look sane," not
"assert exact value."

Run on the hardware you want to validate (e.g. after ./setup-h100.sh on a
rented multi-GPU box, or on a TPU VM after setup-tpu-v5e.sh):

    python3 scripts/validate_gpu_hardware.py
    python3 scripts/validate_gpu_hardware.py --skip-tests --skip-tpu
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _run_task(task_yaml: Path):
    """Run perflab's real profile_only() pipeline; return the run dir, or None on failure."""
    from perflab.orchestrator import profile_only
    from perflab.task_spec import TaskSpec

    print(f"Profiling {task_yaml.relative_to(REPO_ROOT)} ...")
    task = TaskSpec.load(task_yaml)
    try:
        run_dir = profile_only(task)
    except Exception as exc:  # noqa: BLE001 -- report and let other checks still run
        print(f"  FAILED: {exc!r}")
        return None
    print(f"  -> {run_dir}")
    return run_dir


def validate_nsys(run_dir: Path) -> None:
    _header("nsys: real SQLite export vs perflab's parser")
    sqlite_path = run_dir / "artifacts" / "nsys_report.sqlite"
    if not sqlite_path.exists():
        print(f"  nsys sqlite not found at {sqlite_path} (nsys unavailable, or profile_plan skipped it)")
        return

    conn = sqlite3.connect(str(sqlite_path))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for expected in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME", "StringIds"):
        print(f"  table {expected}: {'present' if expected in tables else 'MISSING'}")
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        for expected in ("deviceId", "streamId", "correlationId", "demangledName", "start", "end"):
            print(f"  CUPTI_ACTIVITY_KIND_KERNEL.{expected}: {'present' if expected in cols else 'MISSING'}")
    conn.close()

    from perflab.profilers.nsys_profiler import _parse_nsys_sqlite
    summary = _parse_nsys_sqlite(sqlite_path)
    print(f"  parsed keys: {sorted(summary.keys())}")
    for key in ("gpu_active_pct", "gpu_device_count", "cuda_kernel_time_ms"):
        print(f"  {key}: {summary.get(key)}")
    if summary.get("top_kernels"):
        print(f"  top_kernels[0]: {summary['top_kernels'][0]}")
    else:
        print("  top_kernels: EMPTY")
    if summary.get("cpu_gpu_correlations"):
        print(f"  cpu_gpu_correlations: {len(summary['cpu_gpu_correlations'])} entries")
    else:
        print("  cpu_gpu_correlations: EMPTY -- correlationId join produced nothing, check nsys version/flags")
    # Multi-GPU-specific fields: only present when the traced process used >1 device.
    for key in ("gpu_active_pct_by_device", "top_kernels_by_device", "nccl_pct", "nccl_pct_by_device"):
        if key in summary:
            print(f"  {key}: {summary[key]}")


def validate_ncu(run_dir: Path) -> None:
    _header("ncu: real CSV export vs perflab's fuzzy column matcher")
    csv_path = run_dir / "artifacts" / "ncu_metrics.csv"
    if not csv_path.exists():
        print(f"  ncu csv not found at {csv_path} (ncu unavailable, or profile_plan skipped it)")
        return

    from perflab.profilers.ncu_profiler import _parse_ncu_csv
    summary = _parse_ncu_csv(csv_path)
    print(f"  parsed keys: {sorted(summary.keys())}")
    # _find_column does fuzzy substring matching against the real CSV header,
    # so a renamed column doesn't crash -- it silently resolves to nothing.
    # That's exactly the class of bug the fixture suite can't catch; this is
    # the first time these have been checked against real ncu output.
    for key in (
        "sm_utilization_pct", "compute_throughput_pct", "l1_hit_rate_pct", "l2_hit_rate_pct",
        "registers_per_thread", "tensor_core_active_pct", "tensor_core_throughput_pct",
        "branch_efficiency_pct", "warp_execution_efficiency_pct",
    ):
        if key in summary:
            print(f"  {key}: resolved -> {summary[key]}")
        else:
            print(f"  {key}: NOT RESOLVED -- check ncu --set full's column names for this ncu version")


def validate_jax_tpu(run_dir: Path) -> None:
    _header("JAX/TPU: real trace + HLO dump vs perflab's parser")
    from perflab.profilers.jax_profiler import _collect_jax_trace_metrics, _parse_hlo_dump

    hlo_dir = run_dir / "artifacts" / "xla_hlo_dump"
    trace_dir = run_dir / "artifacts" / "jax_trace"

    hlo_summary = _parse_hlo_dump(hlo_dir) if hlo_dir.exists() else {}
    print(f"  HLO dump dir exists: {hlo_dir.exists()}; parsed keys: {sorted(hlo_summary.keys())}")

    trace_summary = _collect_jax_trace_metrics(trace_dir) if trace_dir.exists() else {}
    print(f"  trace dir exists: {trace_dir.exists()}; parsed keys: {sorted(trace_summary.keys())}")
    for key in ("mxu_utilization_pct", "mxu_utilization_pct_by_device", "infeed_stall_pct", "device_fraction"):
        if key in trace_summary:
            print(f"  {key}: {trace_summary[key]}")


def validate_mps(run_dir: Path) -> None:
    _header("MPS/Metal: real torch_trace vs perflab's parser")

    trace_path = run_dir / "artifacts" / "torch_trace.json"
    if not trace_path.exists():
        print(f"  torch_trace.json not found at {trace_path}")
    else:
        from perflab.profilers.pytorch_profiler import _parse_torch_trace

        summary = _parse_torch_trace(trace_path)
        print(f"  parsed keys: {sorted(summary.keys())}")
        cpu_vs_gpu = summary.get("cpu_vs_gpu", {})
        gpu_us = cpu_vs_gpu.get("total_gpu_kernel_us", 0)
        print(f"  cpu_vs_gpu: {cpu_vs_gpu}")
        # torch.profiler cannot observe Metal GPU kernels on MPS -- gpu_us
        # should legitimately be 0 here. The thing actually worth checking
        # is whether the bottleneck analyzer recognizes that as expected
        # (an informational finding) rather than misreading it as a real
        # GPU-idle bug -- see _analyze_torch_trace's is_mps branch.
        print(f"  total_gpu_kernel_us: {gpu_us} (expected 0 on MPS -- torch.profiler can't see Metal kernels)")

        from perflab.analyzers.bottleneck_analyzer import diagnose_bottlenecks
        diags = diagnose_bottlenecks({"torch_profiler": summary}, "pytorch", device="mps")
        matched = [d for d in diags if "mps" in d.bottleneck.lower() or "unavailable" in d.bottleneck.lower()]
        if matched:
            print(f"  MPS-aware finding present: {matched[0].bottleneck}")
        else:
            print("  NO MPS-aware finding produced -- check _analyze_torch_trace's is_mps branch against this real trace")

    # metal_trace (xctrace/Instruments) needs full Xcode, not just Command
    # Line Tools -- report which one this machine has rather than silently
    # skipping, since that distinction itself is useful to know.
    from perflab.profilers.metal_trace import MetalTraceProfiler
    mt_available = MetalTraceProfiler().is_available()
    print(f"  metal_trace (xctrace/Instruments) available: {mt_available}"
          + ("" if mt_available else " (needs full Xcode, not just Command Line Tools)"))
    if mt_available:
        xml_path = run_dir / "artifacts" / "metal_trace_export.xml"
        if xml_path.exists():
            from perflab.profilers.metal_trace import _parse_xctrace_export
            mt_summary = _parse_xctrace_export(xml_path)
            print(f"  metal_trace parsed keys: {sorted(mt_summary.keys())}")
            print(
                "  NOTE: _parse_gpu_counters' patterns were written for M1/M2/M3 -- "
                "this machine's chip is worth checking explicitly."
            )


def validate_peaks() -> None:
    _header("perflab peaks: compare these against the device's published spec sheet by hand")
    subprocess.run([sys.executable, "-m", "perflab.cli", "peaks"], cwd=REPO_ROOT, check=False)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--skip-tests", action="store_true", help="skip the full pytest run")
    ap.add_argument("--skip-multi-gpu", action="store_true", help="skip the multi-GPU task even if >=2 GPUs are visible")
    ap.add_argument("--skip-tpu", action="store_true", help="skip the TPU check")
    ap.add_argument("--skip-mps", action="store_true", help="skip the Apple Silicon MPS check")
    args = ap.parse_args()

    if not args.skip_tests:
        _header("Full pytest suite on this hardware")
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO_ROOT, check=False).returncode
        print(f"pytest exit code: {rc}")

    try:
        import torch
        has_cuda = torch.cuda.is_available()
        n_gpus = torch.cuda.device_count() if has_cuda else 0
    except ImportError:
        has_cuda, n_gpus = False, 0
    print(f"\nDetected {n_gpus} CUDA device(s) (torch.cuda.is_available()={has_cuda}).")

    if has_cuda and n_gpus >= 1:
        run_dir = _run_task(REPO_ROOT / "perflab/demo_tasks/matmul/cuda/task.yaml")
        if run_dir:
            validate_nsys(run_dir)
            validate_ncu(run_dir)

    if has_cuda and n_gpus >= 2 and not args.skip_multi_gpu:
        # The one that actually exercises deviceId disambiguation, per-device
        # breakdowns, and nccl_pct against real hardware for the first time.
        run_dir = _run_task(REPO_ROOT / "perflab/demo_tasks/multi_gpu_matmul/pytorch/task.yaml")
        if run_dir:
            validate_nsys(run_dir)

    if not args.skip_tpu:
        has_tpu = False
        try:
            import jax
            has_tpu = any(d.platform == "tpu" for d in jax.devices())
        except Exception:  # noqa: BLE001 -- jax not installed or no TPU visible
            pass
        if has_tpu:
            run_dir = _run_task(REPO_ROOT / "perflab/demo_tasks/attention/jax_tpu/task.yaml")
            if run_dir:
                validate_jax_tpu(run_dir)
        else:
            print("\nNo TPU visible (or jax not installed) -- skipping TPU validation.")

    if not args.skip_mps and not has_cuda:
        try:
            import torch
            has_mps = torch.backends.mps.is_available()
        except ImportError:
            has_mps = False
        if has_mps:
            run_dir = _run_task(REPO_ROOT / "perflab/demo_tasks/matmul/pytorch/task.yaml")
            if run_dir:
                validate_mps(run_dir)
        else:
            print("\nNo MPS device visible -- skipping MPS validation.")

    validate_peaks()

    _header("Done. This is a diagnostic pass, not a gate -- eyeball the output above")
    print(
        "Anything marked MISSING / NOT RESOLVED / EMPTY above is worth a closer\n"
        "look: it's either a real profiler-format drift, or a fixture that was\n"
        "wrong about the real format all along. Either way, that's the gap this\n"
        "script exists to surface."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
