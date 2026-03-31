"""Compile and benchmark the C++ matmul binary."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Output JSON path")
    args = ap.parse_args()

    knobs = yaml.safe_load(Path("tuning.yaml").read_text(encoding="utf-8"))
    M = int(knobs.get("M", 512))
    N = int(knobs.get("N", 512))
    K = int(knobs.get("K", 512))

    binary = Path("matmul_bin")

    # Binary is built by task.yaml build step; do not recompile here.
    if not binary.exists():
        raise FileNotFoundError(f"Binary {binary} not found. Run the build step first.")

    # Run
    warmup = os.environ.get("PERFLAB_BENCH_WARMUP")
    repeats = os.environ.get("PERFLAB_BENCH_REPEATS")
    run_cmd = [
        str(binary.resolve()),
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--json",
    ]
    if warmup is not None:
        run_cmd += ["--warmup", warmup]
    if repeats is not None:
        run_cmd += ["--repeats", repeats]
    print(f"[bench] running: {' '.join(run_cmd)}")
    result = subprocess.run(run_cmd, capture_output=True, text=True, check=True)

    bench_data = json.loads(result.stdout)

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bench_data, indent=2), encoding="utf-8")
    print(json.dumps({"tflops_median": bench_data["tflops"]["median"]}, indent=2))


if __name__ == "__main__":
    main()
