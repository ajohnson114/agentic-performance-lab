"""Compile and benchmark the multi-GPU CUDA sgemm binary."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Output JSON path")
    # GPU-scale problem, pinned by contract.fixed_params -- see task.yaml's
    # comment for why (prevents a candidate from shrinking the problem to
    # manufacture a "win").
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--threadsPerBlock", type=int, default=16)
    args = ap.parse_args()

    binary = Path("sgemm_multi_gpu_bin")

    # Binary is built by task.yaml's build step; do not recompile here.
    if not binary.exists():
        raise FileNotFoundError(f"Binary {binary} not found. Run the build step first.")

    warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 10))
    run_cmd = [
        str(binary.resolve()),
        "--M", str(args.M), "--N", str(args.N), "--K", str(args.K),
        "--threadsPerBlock", str(args.threadsPerBlock),
        "--warmup", str(warmup), "--repeats", str(repeats),
        "--json",
    ]
    print(f"[bench] running: {' '.join(run_cmd)}")
    result = subprocess.run(run_cmd, capture_output=True, text=True, check=True)

    bench_data = json.loads(result.stdout)
    bench_data.setdefault("meta", {})
    bench_data["meta"]["warmup"] = warmup
    bench_data["meta"]["repeats"] = repeats

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bench_data, indent=2), encoding="utf-8")
    print(json.dumps({
        "tflops_median": bench_data["tflops"]["median"],
        "num_gpus": bench_data.get("num_gpus"),
    }, indent=2))


if __name__ == "__main__":
    main()
