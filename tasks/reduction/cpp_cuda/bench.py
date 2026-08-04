"""Compile and benchmark the CUDA reduction binary."""
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
    N = int(knobs.get("N", 16777216))
    iterations = int(knobs.get("iterations", 100))
    threadsPerBlock = int(knobs.get("threadsPerBlock", 256))

    binary = Path("reduce_bin")

    # Binary is built by task.yaml build step; do not recompile here.
    if not binary.exists():
        raise FileNotFoundError(f"Binary {binary} not found. Run the build step first.")

    # Run. Defaults here mirror reduce.cu's own hardcoded --warmup/--repeats
    # defaults, but since we always pass them explicitly below, the binary's
    # defaults never actually get exercised -- these are the real values.
    warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
    repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 10))
    run_cmd = [
        str(binary.resolve()),
        "--N", str(N),
        "--iterations", str(iterations),
        "--threadsPerBlock", str(threadsPerBlock),
        "--warmup", str(warmup), "--repeats", str(repeats),
        "--json",
    ]
    print(f"[bench] running: {' '.join(run_cmd)}")
    result = subprocess.run(run_cmd, capture_output=True, text=True, check=True)

    bench_data = json.loads(result.stdout)
    # reduce.cu's own JSON already reports meta.repeats but not meta.warmup --
    # set both explicitly here (harmless overwrite for repeats) so the
    # contract's min_repeats/min_warmup anti-gaming check has both.
    bench_data.setdefault("meta", {})
    bench_data["meta"]["warmup"] = warmup
    bench_data["meta"]["repeats"] = repeats

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bench_data, indent=2), encoding="utf-8")
    print(json.dumps({"throughput_gbs_median": bench_data["throughput_gbs"]["median"]}, indent=2))


if __name__ == "__main__":
    main()
