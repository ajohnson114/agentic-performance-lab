"""Benchmark for attention op on TPU/GPU/CPU.

Measures TFLOPS and tokens/sec for the attention operation.
The naive baseline should score very low on TPU; an optimized version
using jit + bf16 + vectorized heads should approach roofline.
"""

import argparse
import json
import os
import time
from pathlib import Path
from statistics import median

import jax
import jax.numpy as jnp
import yaml
from attention_op import attention


def _compute_attention_flops(batch, seq_len, d_model, n_heads):
    """Compute FLOPs for one attention pass (QK^T + softmax + AV)."""
    head_dim = d_model // n_heads
    # QK^T: batch * n_heads * seq_len * seq_len * head_dim * 2
    qkt_flops = batch * n_heads * seq_len * seq_len * head_dim * 2
    # AV: batch * n_heads * seq_len * head_dim * seq_len * 2
    av_flops = batch * n_heads * seq_len * head_dim * seq_len * 2
    # Softmax ~ 5 * seq_len * seq_len * n_heads * batch
    softmax_flops = 5 * batch * n_heads * seq_len * seq_len
    return qkt_flops + av_flops + softmax_flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Output JSON path")
    args = parser.parse_args()

    knobs = yaml.safe_load(Path("tuning.yaml").read_text())
    batch_size = int(knobs.get("batch_size", 16))
    seq_len = int(knobs.get("seq_len", 512))
    d_model = int(knobs.get("d_model", 512))
    n_heads = int(knobs.get("n_heads", 8))

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (batch_size, seq_len, d_model), dtype=jnp.float32)
    k_tensor = jax.random.normal(k2, (batch_size, seq_len, d_model), dtype=jnp.float32)
    v = jax.random.normal(k3, (batch_size, seq_len, d_model), dtype=jnp.float32)

    # Warmup
    warmup_steps = max(1, int(os.environ.get("PERFLAB_BENCH_WARMUP", 3)))
    for _ in range(warmup_steps):
        out = attention(q, k_tensor, v, n_heads)
        out.block_until_ready()

    # Timed iterations
    n_iters = int(os.environ.get("PERFLAB_BENCH_REPEATS", 20))
    flops_per_call = _compute_attention_flops(batch_size, seq_len, d_model, n_heads)
    tokens_per_call = batch_size * seq_len
    times_s = []

    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = attention(q, k_tensor, v, n_heads)
        out.block_until_ready()
        t1 = time.perf_counter()
        times_s.append(t1 - t0)

    tflops_list = [flops_per_call / t / 1e12 for t in times_s]
    tokens_per_sec_list = [tokens_per_call / t for t in times_s]

    result = {
        "tflops": {"median": median(tflops_list), "all": tflops_list},
        "tokens_per_sec": {"median": median(tokens_per_sec_list), "all": tokens_per_sec_list},
        "raw_values": tflops_list,
        "meta": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "device": str(jax.devices()[0]),
            "platform": jax.devices()[0].platform,
        },
        "ok": True,
    }

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2))
    print(f"tflops.median = {median(tflops_list):.2f}")
    print(f"tokens_per_sec.median = {median(tokens_per_sec_list):.0f}")


if __name__ == "__main__":
    main()
