"""File templates for scaffolding new PerfLab tasks.

Each program type gets a tailored set of starter files: source code to
optimize, a benchmark harness, a correctness test, tuning knobs, and the
task.yaml configuration.
"""
from __future__ import annotations

from textwrap import dedent

# ---------------------------------------------------------------------------
# Per-program-type templates
# ---------------------------------------------------------------------------

_PROFILER_DEFAULTS: dict[str, dict[str, list[str]]] = {
    "python":  {"always": ["cpu_flame"], "optional": []},
    "pytorch": {"always": ["cpu_flame", "torch_trace"], "optional": ["nsys", "ncu"]},
    "jax":     {"always": ["cpu_flame", "jax"], "optional": []},
    "triton":  {"always": ["cpu_flame", "ncu"], "optional": ["nsys"]},
    "cpp":     {"always": ["cpu_flame", "linux_perf"], "optional": []},
    "cuda":    {"always": ["cpu_flame", "ncu"], "optional": ["nsys"]},
}

_BUILD_TEMPLATES: dict[str, str | None] = {
    "python": None,
    "pytorch": None,
    "jax": None,
    "triton": None,
    "cpp": 'g++ -O2 -march=native -o {name}_bin {name}.cpp',
    "cuda": 'nvcc -O2 -o {name}_bin {name}.cu',
}


def _source_template(program_type: str, name: str, description: str) -> tuple[str, str]:
    """Return (filename, content) for the source file to optimize."""
    if program_type == "python":
        return f"{name}.py", dedent(f'''\
            """{description}

            This is the file the agent will optimize. Write a correct but
            deliberately naive implementation here.
            """
            from __future__ import annotations


            def compute(n: int) -> float:
                """Naive implementation — replace with your workload."""
                total = 0.0
                for i in range(n):
                    total += i * i
                return total
        ''')

    if program_type == "pytorch":
        return f"{name}.py", dedent(f'''\
            """{description}

            This is the file the agent will optimize.
            """
            from __future__ import annotations

            import torch


            def compute(x: torch.Tensor) -> torch.Tensor:
                """Naive implementation — replace with your workload."""
                return x @ x.T
        ''')

    if program_type == "jax":
        return f"{name}.py", dedent(f'''\
            """{description}

            This is the file the agent will optimize.
            """
            from __future__ import annotations

            import jax
            import jax.numpy as jnp


            def compute(x: jnp.ndarray) -> jnp.ndarray:
                """Naive implementation — replace with your workload."""
                return jnp.dot(x, x.T)
        ''')

    if program_type == "triton":
        return f"{name}.py", dedent(f'''\
            """{description}

            This is the file the agent will optimize.
            """
            from __future__ import annotations

            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offs = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offs < N
                x = tl.load(x_ptr + offs, mask=mask)
                tl.store(out_ptr + offs, x * x, mask=mask)


            def compute(x: torch.Tensor) -> torch.Tensor:
                """Naive implementation — replace with your workload."""
                out = torch.empty_like(x)
                N = x.numel()
                BLOCK = 1024
                grid = ((N + BLOCK - 1) // BLOCK,)
                kernel[grid](x, out, N, BLOCK)
                return out
        ''')

    if program_type == "cpp":
        return f"{name}.cpp", dedent(f'''\
            // {description}
            //
            // This is the file the agent will optimize.
            #include <cmath>
            #include <cstdlib>
            #include <iostream>
            #include <vector>

            extern "C" double compute(int n) {{
                double total = 0.0;
                for (int i = 0; i < n; ++i) {{
                    total += static_cast<double>(i) * i;
                }}
                return total;
            }}

            #ifndef PERFLAB_LIB
            int main(int argc, char* argv[]) {{
                int n = argc > 1 ? std::atoi(argv[1]) : 1000;
                std::cout << compute(n) << std::endl;
                return 0;
            }}
            #endif
        ''')

    if program_type == "cuda":
        return f"{name}.cu", dedent(f'''\
            // {description}
            //
            // This is the file the agent will optimize.
            #include <cstdio>
            #include <cstdlib>
            #include <cuda_runtime.h>

            __global__ void kernel(const float* __restrict__ x,
                                   float* __restrict__ out, int N) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < N) out[i] = x[i] * x[i];
            }}

            extern "C" void compute(const float* x, float* out, int N) {{
                int threads = 256;
                int blocks = (N + threads - 1) / threads;
                kernel<<<blocks, threads>>>(x, out, N);
                cudaDeviceSynchronize();
            }}

            #ifndef PERFLAB_LIB
            int main() {{
                const int N = 1 << 20;
                float *h_x = new float[N], *h_out = new float[N];
                for (int i = 0; i < N; ++i) h_x[i] = static_cast<float>(i);
                float *d_x, *d_out;
                cudaMalloc(&d_x, N * sizeof(float));
                cudaMalloc(&d_out, N * sizeof(float));
                cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
                compute(d_x, d_out, N);
                cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
                printf("out[0]=%f out[1]=%f\\n", h_out[0], h_out[1]);
                cudaFree(d_x); cudaFree(d_out);
                delete[] h_x; delete[] h_out;
                return 0;
            }}
            #endif
        ''')

    # Fallback — should not happen
    return f"{name}.py", f'"""{ description }"""\n\ndef compute():\n    pass\n'


def _bench_template(program_type: str, name: str, metric_name: str) -> str:
    """Return bench.py content tailored to the program type."""
    # Determine import and timing style
    if program_type in ("pytorch", "triton"):
        return dedent(f'''\
            """Benchmark harness for {name}.

            Writes JSON metrics to --json path.  Honors PERFLAB_BENCH_WARMUP
            and PERFLAB_BENCH_REPEATS env vars for fast screening.
            """
            from __future__ import annotations

            import argparse
            import json
            import os
            import time
            from pathlib import Path
            from statistics import median

            import torch
            import yaml

            from {name} import compute


            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--json", required=True)
                args = parser.parse_args()

                knobs = yaml.safe_load(Path("tuning.yaml").read_text(encoding="utf-8"))
                N = int(knobs.get("N", 1024))

                device = "cuda" if torch.cuda.is_available() else "cpu"
                x = torch.randn(N, N, device=device)

                warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
                for _ in range(warmup):
                    compute(x)
                if device == "cuda":
                    torch.cuda.synchronize()

                repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 20))
                times_ms = []
                for _ in range(repeats):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    compute(x)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times_ms.append((t1 - t0) * 1000)

                med = median(times_ms)
                flops = 2 * N ** 3  # adjust for your workload
                tflops = flops / (med / 1000) / 1e12

                out = {{
                    "latency_ms": {{"median": med, "all": times_ms}},
                    "tflops": {{"median": tflops}},
                    "meta": {{"N": N, "warmup": warmup, "repeats": repeats}},
                    "ok": True,
                }}

                Path(args.json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
                print(f"{metric_name} = {{med:.3f}}")


            if __name__ == "__main__":
                main()
        ''')

    if program_type == "jax":
        return dedent(f'''\
            """Benchmark harness for {name}.

            Writes JSON metrics to --json path.  Honors PERFLAB_BENCH_WARMUP
            and PERFLAB_BENCH_REPEATS env vars for fast screening.
            """
            from __future__ import annotations

            import argparse
            import json
            import os
            import time
            from pathlib import Path
            from statistics import median

            import jax
            import jax.numpy as jnp
            import yaml

            from {name} import compute


            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--json", required=True)
                args = parser.parse_args()

                knobs = yaml.safe_load(Path("tuning.yaml").read_text(encoding="utf-8"))
                N = int(knobs.get("N", 1024))

                key = jax.random.PRNGKey(0)
                x = jax.random.normal(key, (N, N))

                warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
                for _ in range(warmup):
                    result = compute(x)
                    result.block_until_ready()

                repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 20))
                times_ms = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    result = compute(x)
                    result.block_until_ready()
                    t1 = time.perf_counter()
                    times_ms.append((t1 - t0) * 1000)

                med = median(times_ms)
                flops = 2 * N ** 3  # adjust for your workload
                tflops = flops / (med / 1000) / 1e12

                out = {{
                    "latency_ms": {{"median": med, "all": times_ms}},
                    "tflops": {{"median": tflops}},
                    "meta": {{"N": N, "warmup": warmup, "repeats": repeats}},
                    "ok": True,
                }}

                Path(args.json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
                print(f"{metric_name} = {{med:.3f}}")


            if __name__ == "__main__":
                main()
        ''')

    if program_type in ("cpp", "cuda"):
        run_cmd = f"./{name}_bin" if program_type == "cpp" else f"./{name}_bin"
        return dedent(f'''\
            """Benchmark harness for {name} (compiled).

            Writes JSON metrics to --json path.  Honors PERFLAB_BENCH_WARMUP
            and PERFLAB_BENCH_REPEATS env vars for fast screening.
            """
            from __future__ import annotations

            import argparse
            import json
            import os
            import subprocess
            import time
            from pathlib import Path
            from statistics import median

            import yaml


            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--json", required=True)
                args = parser.parse_args()

                knobs = yaml.safe_load(Path("tuning.yaml").read_text(encoding="utf-8"))
                N = int(knobs.get("N", 1024))

                binary = "{run_cmd}"

                warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
                for _ in range(warmup):
                    subprocess.run([binary, str(N)], check=True,
                                   capture_output=True)

                repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 20))
                times_ms = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    subprocess.run([binary, str(N)], check=True,
                                   capture_output=True)
                    t1 = time.perf_counter()
                    times_ms.append((t1 - t0) * 1000)

                med = median(times_ms)

                out = {{
                    "latency_ms": {{"median": med, "all": times_ms}},
                    "meta": {{"N": N, "warmup": warmup, "repeats": repeats}},
                    "ok": True,
                }}

                Path(args.json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
                print(f"{metric_name} = {{med:.3f}}")


            if __name__ == "__main__":
                main()
        ''')

    # python (default)
    return dedent(f'''\
        """Benchmark harness for {name}.

        Writes JSON metrics to --json path.  Honors PERFLAB_BENCH_WARMUP
        and PERFLAB_BENCH_REPEATS env vars for fast screening.
        """
        from __future__ import annotations

        import argparse
        import json
        import os
        import time
        from pathlib import Path
        from statistics import median

        import yaml

        from {name} import compute


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--json", required=True)
            args = parser.parse_args()

            knobs = yaml.safe_load(Path("tuning.yaml").read_text(encoding="utf-8"))
            N = int(knobs.get("N", 1024))

            warmup = int(os.environ.get("PERFLAB_BENCH_WARMUP", 3))
            for _ in range(warmup):
                compute(N)

            repeats = int(os.environ.get("PERFLAB_BENCH_REPEATS", 20))
            times_ms = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                compute(N)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000)

            med = median(times_ms)
            throughput = N / (med / 1000)

            out = {{
                "throughput": {{"median": throughput}},
                "latency_ms": {{"median": med, "all": times_ms}},
                "meta": {{"N": N, "warmup": warmup, "repeats": repeats}},
                "ok": True,
            }}

            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"{metric_name} = {{med:.3f}}")


        if __name__ == "__main__":
            main()
    ''')


def _tests_template(program_type: str, name: str) -> str:
    """Return tests.py content."""
    if program_type in ("cpp", "cuda"):
        binary = f"./{name}_bin"
        return dedent(f'''\
            """Correctness test for {name} (compiled).

            Must exit 0 on success.
            """
            import subprocess
            import sys


            def main():
                result = subprocess.run(
                    ["{binary}"], capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"FAIL: exit code {{result.returncode}}", file=sys.stderr)
                    print(result.stderr, file=sys.stderr)
                    sys.exit(1)

                # TODO: Add numerical correctness checks here.
                # Parse stdout or a results file and compare to expected values.
                print("ok")


            if __name__ == "__main__":
                main()
        ''')

    return dedent(f'''\
        """Correctness test for {name}.

        Must exit 0 on success. Any non-zero exit causes the agent to reject
        the candidate that produced this failure.
        """
        from {name} import compute


        def main():
            # TODO: Replace with your own correctness checks.
            # Compare compute() output against known-good values.
            result = compute(100)
            assert result is not None, "compute() returned None"

            print("ok")


        if __name__ == "__main__":
            main()
    ''')


def _tuning_template(name: str, fixed_params: dict[str, int | float] | None = None) -> str:
    """Return tuning.yaml content."""
    lines = [
        f"# Tuning knobs for {name}",
        f"# Fixed params (protected by contract) should not be changed by the agent.",
        f"# Tunable params are fair game for optimization.",
        "",
    ]

    if fixed_params:
        lines.append("# Fixed (problem dimensions)")
        for k, v in fixed_params.items():
            lines.append(f"{k}: {v}")
        lines.append("")

    lines.extend([
        "# Tunable (implementation knobs) — add your own",
        "N: 1024",
        "",
        "# Uncomment to enable auto-tuning sweep:",
        "# sweep:",
        "#   N: [512, 1024, 2048]",
    ])
    return "\n".join(lines) + "\n"


def _task_yaml_template(
    name: str,
    program_type: str,
    workspace: str,
    source_file: str,
    *,
    target_hardware: str | None = None,
    metric_name: str = "latency_ms.median",
    metric_mode: str = "minimize",
    fixed_params: dict[str, int | float] | None = None,
) -> str:
    """Return task.yaml content."""
    build_cmd = _BUILD_TEMPLATES.get(program_type)
    profilers = _PROFILER_DEFAULTS.get(program_type, {"always": ["cpu_flame"], "optional": []})

    lines = [
        f'name: "{name}"',
        f'workspace: "{workspace}"',
        f'program_type: "{program_type}"',
    ]

    if target_hardware:
        lines.append(f'target_hardware: "{target_hardware}"')
    else:
        lines.append("target_hardware: null")

    lines.append("")

    if build_cmd:
        cmd = build_cmd.format(name=name)
        lines.extend([
            "build:",
            f'  cmd: "{cmd}"',
            "  expected_exit: 0",
        ])
    else:
        lines.append("build: null")

    lines.extend([
        "",
        "correctness:",
        '  cmd: "python3 tests.py"',
        "  expected_exit: 0",
        "",
        "benchmark:",
        '  cmd: "python3 bench.py --json out/bench.json"',
        "  metric:",
        f'    name: "{metric_name}"',
        f'    mode: "{metric_mode}"',
        "  warmup: 3",
        "  repeats: 20",
        "",
        "profile_plan:",
        f'  always: {_yaml_list(profilers["always"])}',
    ])
    if profilers["optional"]:
        lines.append(f'  optional: {_yaml_list(profilers["optional"])}')

    lines.extend([
        "",
        "constraints:",
        "  max_iters: 10",
        "  regression_tolerance: 0.02",
    ])

    if fixed_params:
        fp_items = ", ".join(f"{k}: {v}" for k, v in fixed_params.items())
        lines.extend([
            "",
            "contract:",
            f"  fixed_params: {{{fp_items}}}",
            "  min_repeats: 3",
            '  required_bench_fields: ["ok"]',
        ])

    lines.extend([
        "",
        "edit_policy:",
        "  allowed_paths:",
        f'    - "{source_file}"',
        '    - "tuning.yaml"',
        "",
        'out_dir: "out"',
    ])

    return "\n".join(lines) + "\n"


def _yaml_list(items: list[str]) -> str:
    """Format a list as inline YAML."""
    return "[" + ", ".join(f'"{i}"' for i in items) + "]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_task_files(
    name: str,
    program_type: str,
    workspace: str,
    description: str = "Performance optimization task",
    *,
    target_hardware: str | None = None,
    metric_name: str = "latency_ms.median",
    metric_mode: str = "minimize",
    fixed_params: dict[str, int | float] | None = None,
) -> dict[str, str]:
    """Generate all files for a new task.

    Returns a dict of {relative_filename: content}.
    """
    source_file, source_content = _source_template(program_type, name, description)

    return {
        source_file: source_content,
        "bench.py": _bench_template(program_type, name, metric_name),
        "tests.py": _tests_template(program_type, name),
        "tuning.yaml": _tuning_template(name, fixed_params),
        "task.yaml": _task_yaml_template(
            name=name,
            program_type=program_type,
            workspace=workspace,
            source_file=source_file,
            target_hardware=target_hardware,
            metric_name=metric_name,
            metric_mode=metric_mode,
            fixed_params=fixed_params,
        ),
    }


# ---------------------------------------------------------------------------
# Profiler suggestions
# ---------------------------------------------------------------------------

_PROFILER_RATIONALE: dict[str, str] = {
    "cpu_flame": "CPU flame graph — always useful for finding hotspots in host code",
    "torch_trace": "PyTorch profiler — GPU/CPU timeline, operator breakdown, memory",
    "jax": "JAX/XLA trace — compilation times, HLO ops, device utilization",
    "ncu": "NVIDIA Compute Profiler — kernel-level SM util, memory throughput, occupancy",
    "nsys": "NVIDIA Systems Profiler — system-wide GPU/CPU timeline, API traces",
    "linux_perf": "Linux perf — hardware counters (IPC, cache misses, branch mispredicts)",
    "metal_trace": "Apple Metal trace — GPU timeline on Apple Silicon",
}


def suggest_profilers(program_type: str, target_hardware: str | None = None) -> dict:
    """Suggest profilers for a given program type and hardware."""
    base = _PROFILER_DEFAULTS.get(program_type, {"always": ["cpu_flame"], "optional": []})
    always = list(base["always"])
    optional = list(base["optional"])

    # Hardware-specific adjustments
    if target_hardware:
        hw = target_hardware.lower()
        if "apple" in hw or "m1" in hw or "m2" in hw or "m3" in hw or "m4" in hw or "mps" in hw:
            if "metal_trace" not in always and "metal_trace" not in optional:
                optional.append("metal_trace")
            # Remove NVIDIA profilers
            always = [p for p in always if p not in ("ncu", "nsys")]
            optional = [p for p in optional if p not in ("ncu", "nsys")]
        elif "tpu" in hw:
            if "jax" not in always:
                always.append("jax")
            always = [p for p in always if p not in ("ncu", "nsys")]
            optional = [p for p in optional if p not in ("ncu", "nsys")]

    result = {
        "always": always,
        "optional": optional,
        "rationale": {
            p: _PROFILER_RATIONALE.get(p, "")
            for p in always + optional
            if p in _PROFILER_RATIONALE
        },
    }

    return result


# ---------------------------------------------------------------------------
# Threshold suggestions
# ---------------------------------------------------------------------------

_THRESHOLD_PRESETS: dict[str, dict[str, float]] = {
    "python": {
        "perf_ipc_low": 0.5,
        "perf_hotspot_dominance_pct": 30.0,
    },
    "pytorch": {
        "gpu_cpu_ratio_low": 0.5,
        "sync_count_warn": 5,
        "mem_alloc_overhead_pct": 0.10,
    },
    "jax": {
        "jax_recompilation_warn": 1,
        "jax_compilation_time_high_ms": 5000.0,
        "jax_compilations_excessive": 5,
    },
    "triton": {
        "ncu_sm_util_low": 60.0,
        "ncu_tc_util_low": 30.0,
        "ncu_occupancy_low": 50.0,
    },
    "cpp": {
        "perf_ipc_low": 1.5,
        "perf_cache_miss_rate_high": 0.02,
        "perf_hotspot_dominance_pct": 50.0,
    },
    "cuda": {
        "ncu_sm_util_low": 60.0,
        "ncu_tc_util_low": 30.0,
        "ncu_occupancy_low": 50.0,
        "ncu_bank_conflicts_high": 100.0,
    },
}

_THRESHOLD_DESCRIPTIONS: dict[str, str] = {
    "perf_ipc_low": "Instructions per cycle — below this flags CPU inefficiency",
    "perf_cache_miss_rate_high": "L1/LLC miss rate — above this flags memory access issues",
    "perf_hotspot_dominance_pct": "Single function time % — above this flags hotspot dominance",
    "gpu_cpu_ratio_low": "GPU/CPU time ratio — below this flags GPU underutilization",
    "sync_count_warn": "Host-device syncs — above this flags excessive synchronization",
    "mem_alloc_overhead_pct": "Memory allocation overhead — above this flags allocation churn",
    "ncu_sm_util_low": "SM utilization % — below this flags underutilized GPU SMs",
    "ncu_tc_util_low": "Tensor Core utilization % — below this flags missed TC opportunity",
    "ncu_occupancy_low": "Warp occupancy % — below this flags low parallelism",
    "ncu_bank_conflicts_high": "Shared memory bank conflicts — above this flags memory bottleneck",
    "jax_recompilation_warn": "XLA recompilations — above this flags tracing issues",
    "jax_compilation_time_high_ms": "Compilation time — above this flags long compile",
    "jax_compilations_excessive": "Total compilations — above this flags excessive tracing",
}


def suggest_thresholds(program_type: str, target_hardware: str | None = None) -> dict:
    """Suggest analysis thresholds for a given program type."""
    preset = _THRESHOLD_PRESETS.get(program_type, {})
    result: dict = {}
    for key, val in preset.items():
        result[key] = {
            "value": val,
            "description": _THRESHOLD_DESCRIPTIONS.get(key, ""),
        }

    # TPU-specific additions
    if target_hardware and "tpu" in target_hardware.lower():
        tpu_thresholds = {
            "tpu_mxu_util_low": (30.0, "MXU utilization % — below this flags underused matrix unit"),
            "tpu_padding_waste_pct_high": (15.0, "Padding waste % — above this flags inefficient tiling"),
            "tpu_infeed_stall_pct_high": (5.0, "Infeed stall % — above this flags data pipeline bottleneck"),
        }
        for key, (val, desc) in tpu_thresholds.items():
            result[key] = {"value": val, "description": desc}

    return {
        "program_type": program_type,
        "target_hardware": target_hardware,
        "suggested_thresholds": result,
        "note": "These are starting points. Adjust after running initial profiling based on your workload's characteristics.",
    }


# ---------------------------------------------------------------------------
# Bench.py linting
# ---------------------------------------------------------------------------

def lint_bench_script(content: str) -> dict:
    """Check a bench.py script for PerfLab protocol compliance.

    Returns a dict with 'passed', 'warnings', and 'errors' lists.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Must accept --json
    if "--json" not in content:
        errors.append("bench.py must accept --json <path> argument for writing results")

    # Must write JSON
    if "json.dumps" not in content and "json.dump" not in content:
        errors.append("bench.py must write JSON output (use json.dumps or json.dump)")

    # Should honor env vars
    if "PERFLAB_BENCH_WARMUP" not in content:
        warnings.append(
            "bench.py should honor PERFLAB_BENCH_WARMUP env var for fast screening "
            "(e.g., warmup = int(os.environ.get('PERFLAB_BENCH_WARMUP', 3)))"
        )
    if "PERFLAB_BENCH_REPEATS" not in content:
        warnings.append(
            "bench.py should honor PERFLAB_BENCH_REPEATS env var for fast screening "
            "(e.g., repeats = int(os.environ.get('PERFLAB_BENCH_REPEATS', 20)))"
        )

    # Should include 'ok' field
    if '"ok"' not in content and "'ok'" not in content:
        warnings.append(
            'bench.py output should include an "ok" field (required by default contract)'
        )

    # Should use statistics for timing
    if "median" not in content and "mean" not in content:
        warnings.append(
            "bench.py should compute median or mean of timing samples for stable metrics"
        )

    # GPU sync check for CUDA workloads
    has_cuda = "torch.cuda" in content or "cuda" in content.lower()
    has_sync = "synchronize" in content
    if has_cuda and not has_sync:
        warnings.append(
            "bench.py uses CUDA but doesn't call synchronize() — GPU timing will be inaccurate"
        )

    # JAX block_until_ready check
    has_jax = "import jax" in content or "jax." in content
    has_block = "block_until_ready" in content
    if has_jax and not has_block:
        warnings.append(
            "bench.py uses JAX but doesn't call block_until_ready() — timing will include only dispatch, not execution"
        )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks_run": len(errors) + len(warnings) + (4 if len(errors) == 0 else 2),
    }


# ---------------------------------------------------------------------------
# Contract suggestion from bench.py analysis
# ---------------------------------------------------------------------------

def suggest_contract_from_bench(content: str, program_type: str) -> dict:
    """Analyze a bench.py and suggest contract settings."""
    import re

    suggestions: dict = {
        "fixed_params": {},
        "required_bench_fields": ["ok"],
        "min_repeats": 3,
        "reasoning": [],
    }

    # Look for common problem dimension patterns
    dimension_patterns = [
        (r'["\']?(M|N|K|batch_size|seq_len|d_model|n_heads|num_images|size|dim|width|height)\b["\']?\s*[:=]', None),
        (r'knobs\.get\(["\'](\w+)["\']', None),
        (r'knobs\[["\'](\w+)["\']\]', None),
    ]

    found_params: set[str] = set()
    for pattern, _ in dimension_patterns:
        for match in re.finditer(pattern, content):
            param = match.group(1)
            # Skip known tunable params
            if param.upper().startswith(("BLOCK", "TILE", "NUM_WARPS", "NUM_STAGES")):
                continue
            found_params.add(param)

    if found_params:
        suggestions["reasoning"].append(
            f"Found potential problem dimensions in bench.py: {sorted(found_params)}. "
            "These should be fixed_params to prevent the agent from shrinking the problem."
        )
        for p in sorted(found_params):
            suggestions["fixed_params"][p] = f"<fill in the value for {p}>"

    # Look for metric fields in the output JSON
    json_field_pattern = r'["\'](\w+(?:\.\w+)*)["\']'
    # Find dict literal patterns that look like bench output
    output_patterns = re.findall(r'"(\w+)":\s*\{', content)
    metric_fields = [f for f in output_patterns if f not in ("meta",)]
    if metric_fields:
        suggestions["required_bench_fields"] = ["ok"] + metric_fields
        suggestions["reasoning"].append(
            f"Found metric sections in output: {metric_fields}. "
            "Adding to required_bench_fields prevents the agent from dropping metrics."
        )

    # Determine min_repeats based on program type
    if program_type in ("cuda", "pytorch", "triton"):
        suggestions["min_repeats"] = 5
        suggestions["reasoning"].append(
            "GPU workloads benefit from more repeats (5+) due to higher variance from device scheduling."
        )
    elif program_type in ("cpp",):
        suggestions["min_repeats"] = 10
        suggestions["reasoning"].append(
            "CPU compiled workloads can use more repeats (10) for tighter confidence intervals."
        )

    return suggestions
