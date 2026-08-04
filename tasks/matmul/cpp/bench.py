"""Compile and benchmark the C++ matmul binary."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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

    # Run. Defaults here mirror matmul.cpp's own hardcoded --warmup/--repeats
    # defaults, but since we always pass them explicitly below, the binary's
    # defaults never actually get exercised -- these are the real values.
    warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 2))
    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 5))
    run_cmd = [
        str(binary.resolve()),
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--warmup", str(warmup), "--repeats", str(repeats),
        "--json",
    ]
    print(f"[bench] running: {' '.join(run_cmd)}")
    result = subprocess.run(run_cmd, capture_output=True, text=True, check=True)

    bench_data = json.loads(result.stdout)
    # matmul.cpp's own JSON doesn't report the sampling counts it used --
    # inject the actual (post-env-override) values for the contract's
    # min_repeats/min_warmup anti-gaming check.
    bench_data.setdefault("meta", {})
    bench_data["meta"]["warmup"] = warmup
    bench_data["meta"]["repeats"] = repeats

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bench_data, indent=2), encoding="utf-8")
    print(json.dumps({"tflops_median": bench_data["tflops"]["median"]}, indent=2))


if __name__ == "__main__":
    main()
