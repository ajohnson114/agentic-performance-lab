"""Training throughput benchmark for small JAX transformer.

Runs in float32 with naive attention and no jax.jit. An optimizing agent
should discover and apply jax.jit, efficient attention, and mixed precision.
"""

import argparse
import json
import os
import time
from pathlib import Path
from statistics import median

import jax
import optax
import yaml
from model import (
    cross_entropy_loss,
    init_transformer,
    transformer_forward,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Output JSON path")
    args = parser.parse_args()

    knobs = yaml.safe_load(Path("tuning.yaml").read_text())
    batch_size = int(knobs.get("batch_size", 8))
    seq_len = int(knobs.get("seq_len", 128))
    lr = float(knobs.get("lr", 1e-3))

    vocab_size = 1024
    d_model = 256
    n_heads = 4
    n_layers = 4
    d_ff = 512

    key = jax.random.PRNGKey(0)
    params = init_transformer(key, vocab_size, d_model, n_heads, n_layers, d_ff, seq_len)

    optimizer = optax.adamw(learning_rate=lr)
    opt_state = optimizer.init(params)

    # TODO: Per-phase annotation is future work using jax.profiler.TraceAnnotation
    # (e.g. with jax.profiler.TraceAnnotation("forward"): ...)
    def train_step(params, opt_state, key):
        key, subkey = jax.random.split(key)
        tokens = jax.random.randint(subkey, (batch_size, seq_len), 0, vocab_size)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]

        def loss_fn(p):
            logits = transformer_forward(p, inputs, n_heads)
            return cross_entropy_loss(logits, targets)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, key

    # Warmup (at least 1 for JIT compilation cost during fast screening)
    warmup_steps = max(1, int(os.environ.get("PERFLAB_BENCH_WARMUP", 3)))
    step_key = jax.random.PRNGKey(42)
    for _ in range(warmup_steps):
        params, opt_state, loss, step_key = train_step(params, opt_state, step_key)
        loss.block_until_ready()

    # Timed steps
    n_steps = int(os.environ.get("PERFLAB_BENCH_REPEATS", 10))
    tokens_per_step = batch_size * (seq_len - 1)
    times_s = []

    for _ in range(n_steps):
        t0 = time.perf_counter()
        params, opt_state, loss, step_key = train_step(params, opt_state, step_key)
        loss.block_until_ready()
        t1 = time.perf_counter()
        times_s.append(t1 - t0)

    tokens_per_sec_list = [tokens_per_step / t for t in times_s]
    med = median(tokens_per_sec_list)

    out = {
        "tokens_per_sec": {"median": med, "all": tokens_per_sec_list},
        "meta": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "n_steps": n_steps,
        },
        "ok": True,
    }

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"tokens_per_sec.median = {med:.1f}")


if __name__ == "__main__":
    main()
