"""Correctness test: compile and run CUDA sgemm at small size, verify output."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def main():
    binary = Path("sgemm_bin")

    # Binary is built by task.yaml build step; do not recompile here.
    if not binary.exists():
        raise FileNotFoundError(f"Binary {binary} not found. Run the build step first.")

    # Run built-in selftest (hardcoded 4x4 GPU verification)
    selftest_result = subprocess.run(
        [str(binary.resolve()), "--selftest"],
        capture_output=True, text=True,
    )
    assert selftest_result.returncode == 0, (
        f"Selftest failed (rc={selftest_result.returncode}): {selftest_result.stderr}"
    )

    M, N, K = 64, 64, 64

    result = subprocess.run(
        [str(binary.resolve()), "--M", str(M), "--N", str(N), "--K", str(K),
         "--json", "--warmup", "1", "--repeats", "1"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    assert data["ok"], "Binary reported failure"
    assert "tflops" in data, "Missing tflops in output"
    assert data["tflops"]["median"] >= 0, "Negative tflops"

    print("ok", {"M": M, "N": N, "K": K})


if __name__ == "__main__":
    main()
