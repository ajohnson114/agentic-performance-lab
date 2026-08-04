"""Correctness test: compile and run CUDA Tensor Core hgemm, verify output."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def main():
    binary = Path("hgemm_bin")

    # Binary is built by task.yaml build step; do not recompile here.
    if not binary.exists():
        raise FileNotFoundError(f"Binary {binary} not found. Run the build step first.")

    # Run built-in selftest (16x16 GPU verification with FP16 tolerance)
    selftest_result = subprocess.run(
        [str(binary.resolve()), "--selftest"],
        capture_output=True, text=True,
    )
    assert selftest_result.returncode == 0, (
        f"Selftest failed (rc={selftest_result.returncode}): {selftest_result.stderr}"
    )

    # Small benchmark run to verify JSON output
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
    assert data["meta"]["dtype"] == "fp16", "Expected fp16 dtype in meta"

    print("ok", {"M": M, "N": N, "K": K, "dtype": "fp16"})


if __name__ == "__main__":
    main()
