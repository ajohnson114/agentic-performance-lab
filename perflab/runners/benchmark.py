from __future__ import annotations
from pathlib import Path
import hashlib
import json
import logging
import os
import shlex
import subprocess
from perflab.tools.shell import run_cmd, CmdResult

_GPU_PROGRAM_TYPES = {"cuda", "pytorch", "jax", "triton"}

_logger = logging.getLogger(__name__)

_GPU_THERMAL_THROTTLE_C = 80
_GPU_THERMAL_WAIT_C = 75
_GPU_THERMAL_WAIT_MAX_S = 120
_GPU_THERMAL_POLL_S = 5


def _gpu_temperature() -> int | None:
    """Return current GPU 0 temperature in Celsius, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().splitlines()[0].strip())
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return None


def _wait_for_gpu_cooldown() -> None:
    """If GPU is above thermal throttle threshold, wait for it to cool down."""
    import time
    temp = _gpu_temperature()
    if temp is None or temp <= _GPU_THERMAL_THROTTLE_C:
        return
    _logger.warning(
        "GPU temperature is %d°C (threshold %d°C) — waiting for cooldown",
        temp, _GPU_THERMAL_WAIT_C,
    )
    waited = 0
    while temp is not None and temp > _GPU_THERMAL_WAIT_C and waited < _GPU_THERMAL_WAIT_MAX_S:
        time.sleep(_GPU_THERMAL_POLL_S)
        waited += _GPU_THERMAL_POLL_S
        temp = _gpu_temperature()
    if temp is not None and temp > _GPU_THERMAL_WAIT_C:
        _logger.warning(
            "GPU still at %d°C after %ds — benchmarking anyway (results may be noisy)",
            temp, waited,
        )
    else:
        _logger.info("GPU cooled to %d°C after %ds — proceeding", temp, waited)


def _resolve_rlimit(program_type: str | None, rlimit_as_gb: float | None) -> int | None:
    """Compute RLIMIT_AS in bytes from task config.

    Priority:
      1. Explicit rlimit_as_gb from task.yaml → use that value (0 means disabled)
      2. GPU program type → DEFAULT_GPU_RLIMIT_AS_BYTES (32 GB)
      3. CPU program type → DEFAULT_RLIMIT_AS_BYTES (4 GB)
    """
    from perflab.tools.shell import DEFAULT_RLIMIT_AS_BYTES, DEFAULT_GPU_RLIMIT_AS_BYTES
    if rlimit_as_gb is not None:
        if rlimit_as_gb <= 0:
            return None  # Explicitly disabled
        return int(rlimit_as_gb * 1024**3)
    if program_type in _GPU_PROGRAM_TYPES:
        return DEFAULT_GPU_RLIMIT_AS_BYTES
    return DEFAULT_RLIMIT_AS_BYTES


def run_benchmark(
    cmd: str,
    cwd: Path,
    env: dict[str, str] | None = None,
    fast_mode: bool = False,
    program_type: str | None = None,
    rlimit_as_gb: float | None = None,
) -> tuple[CmdResult, dict]:
    """Run the benchmark command and parse bench.json output.

    When fast_mode=True, sets PERFLAB_BENCH_WARMUP=0 and PERFLAB_BENCH_REPEATS=2
    as environment variables for quick directional ranking during beam search.
    The fast screen is only used to rank candidates — the top candidate is always
    re-benchmarked with full warmup/repeats before the accept/reject decision.
    Benchmark harnesses should honor these env vars when present.

    When program_type is a GPU type (cuda, pytorch, jax, triton), RLIMIT_AS is
    set to 32 GB (instead of 4 GB for CPU tasks) because CUDA runtimes and JIT
    compilers map large virtual address regions. This still prevents runaway
    allocation from exhausting system memory.

    rlimit_as_gb overrides the default when set in task.yaml constraints.
    """
    run_env = dict(env or {})
    if fast_mode:
        run_env["PERFLAB_BENCH_WARMUP"] = "0"
        run_env["PERFLAB_BENCH_REPEATS"] = "2"
    else:
        # Set config defaults only if not already in env (env vars win)
        try:
            from perflab.config import load_config
            cfg = load_config()
            run_env.setdefault("PERFLAB_BENCH_WARMUP", str(cfg.benchmark.warmup))
            run_env.setdefault("PERFLAB_BENCH_REPEATS", str(cfg.benchmark.repeats))
        except Exception:
            pass  # Config loading is best-effort — defaults still come from bench.py

    # Thermal gate: wait for GPU to cool if above throttle threshold
    if program_type in _GPU_PROGRAM_TYPES:
        _wait_for_gpu_cooldown()

    rlimit = _resolve_rlimit(program_type, rlimit_as_gb)

    # Anti-tampering: record bench.json state before the run.
    # If LLM-edited code pre-writes a fake bench.json, we detect it via
    # content hash comparison and mtime timing checks.
    bench_path = cwd / "out" / "bench.json"
    pre_hash: str | None = None
    if bench_path.exists():
        pre_hash = hashlib.sha256(bench_path.read_bytes()).hexdigest()
    import time as _time
    run_start = _time.time()

    res = run_cmd(
        shlex.split(cmd), cwd=cwd, env=run_env if run_env else None,
        timeout_s=300, rlimit_as_bytes=rlimit,
    )
    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark did not create {bench_path}. Stdout/stderr:\n{res.stdout}\n{res.stderr}")

    # Anti-tampering: verify bench.json was actually written by the benchmark
    if pre_hash is not None:
        post_hash = hashlib.sha256(bench_path.read_bytes()).hexdigest()
        if post_hash == pre_hash:
            raise RuntimeError(
                "bench.json was not modified by the benchmark run. "
                "The benchmark harness must overwrite bench.json on each execution."
            )
    # Defense in depth: mtime must be within the benchmark run window
    post_mtime = bench_path.stat().st_mtime
    if post_mtime < run_start - 1.0:
        raise RuntimeError(
            f"bench.json appears stale (mtime {post_mtime:.1f} < run start {run_start:.1f}). "
            f"The benchmark harness must write bench.json during execution."
        )

    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    return res, bench

def validate_contract(bench: dict, contract) -> list[str]:
    """Validate bench.json against the task contract. Returns list of errors."""
    errors: list[str] = []

    # Check required fields exist (dotted path traversal)
    for field_path in contract.required_bench_fields:
        cur = bench
        for part in field_path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                errors.append(f"Required bench field '{field_path}' missing (part '{part}' not found)")
                break
            cur = cur[part]

    # Check fixed_params match meta values
    meta = bench.get("meta", {})
    for param, expected in contract.fixed_params.items():
        actual = meta.get(param)
        if actual is None:
            errors.append(f"Contract fixed_param '{param}' not found in bench.meta")
        elif actual != expected:
            errors.append(
                f"Contract violation: meta.{param}={actual}, expected {expected}"
            )

    return errors


def metric_value(bench: dict, metric_name: str) -> float:
    # simple dotted-path access: "tflops.median" etc.
    cur = bench
    for part in metric_name.split("."):
        if part not in cur:
            raise KeyError(f"Metric path '{metric_name}' missing part '{part}' in bench.json")
        cur = cur[part]
    if not isinstance(cur, (int, float)):
        raise TypeError(f"Metric '{metric_name}' is not a number: {cur!r}")
    return float(cur)


def validate_bench_variance(bench: dict) -> list[str]:
    """Detect suspiciously low variance in benchmark timing arrays.

    Checks all lists of numbers in bench.json (e.g., "times_ms", "all" arrays).
    If a timing array has zero variance (all values identical), this likely
    indicates memoization/caching — the kernel is returning cached results
    instead of actually running computation.

    Returns list of warning strings (empty = ok).
    """
    warnings: list[str] = []
    _check_variance_recursive(bench, "", warnings)
    return warnings


def _check_variance_recursive(obj: dict | list, path: str, warnings: list[str]) -> None:
    """Walk bench.json tree looking for numeric arrays with zero variance."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            child_path = f"{path}.{key}" if path else key
            _check_variance_recursive(val, child_path, warnings)
    elif isinstance(obj, list) and len(obj) >= 3:
        # Only check lists that look like timing arrays (all numbers)
        if all(isinstance(x, (int, float)) for x in obj):
            vals = [float(x) for x in obj]
            if max(vals) == min(vals):
                warnings.append(
                    f"Zero variance in {path}: all {len(vals)} values are "
                    f"identical ({vals[0]}). Possible memoization/caching."
                )
            elif len(vals) >= 5:
                mean = sum(vals) / len(vals)
                if mean > 0:
                    cv = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 / mean
                    if cv < 1e-9:
                        warnings.append(
                            f"Suspiciously low variance in {path}: CV={cv:.2e} "
                            f"across {len(vals)} values. Possible caching."
                        )
