"""Correctness test: compile and run CUDA reduction selftest."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path


def main():
    binary = Path("reduce_bin")

    # Binary is built by task.yaml build step; do not recompile here.
    if not binary.exists():
        raise FileNotFoundError(f"Binary {binary} not found. Run the build step first.")

    # Run built-in selftest (multiple sizes + random seeds)
    selftest_result = subprocess.run(
        [str(binary.resolve()), "--selftest"],
        capture_output=True, text=True,
    )
    assert selftest_result.returncode == 0, (
        f"Selftest failed (rc={selftest_result.returncode}): {selftest_result.stderr}"
    )

    # Small benchmark run to verify JSON output
    N = 1024
    result = subprocess.run(
        [str(binary.resolve()), "--N", str(N), "--iterations", "2",
         "--json", "--warmup", "1", "--repeats", "1"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    assert data["ok"], "Binary reported failure"
    assert "throughput_gbs" in data, "Missing throughput_gbs in output"
    assert data["throughput_gbs"]["median"] >= 0, "Negative throughput"

    print("ok", {"N": N, "iterations": 2})


if __name__ == "__main__":
    main()
