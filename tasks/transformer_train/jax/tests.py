"""Correctness test: small JAX transformer, 5 training steps, loss finite and decreasing."""

import jax
import jax.numpy as jnp
import optax

from model import (
    cross_entropy_loss,
    init_transformer,
    transformer_forward,
)


def main():
    vocab_size = 1024
    d_model = 64
    n_heads = 2
    n_layers = 2
    d_ff = 128
    seq_len = 32
    batch_size = 4

    key = jax.random.PRNGKey(42)
    params = init_transformer(key, vocab_size, d_model, n_heads, n_layers, d_ff, seq_len)

    optimizer = optax.adamw(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    # No jit in tests — keep it simple and deterministic
    losses = []
    step_key = jax.random.PRNGKey(123)
    for step in range(5):
        step_key, subkey = jax.random.split(step_key)
        tokens = jax.random.randint(subkey, (batch_size, seq_len), 0, vocab_size)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]

        def loss_fn(p):
            logits = transformer_forward(p, inputs, n_heads)
            return cross_entropy_loss(logits, targets)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        loss_val = float(loss)
        losses.append(loss_val)

    # Check: all losses finite
    for i, l in enumerate(losses):
        assert jnp.isfinite(l), f"Loss at step {i} is not finite: {l}"

    # Check: loss is decreasing overall (last < first)
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )

    # Logit sanity check: verify model architecture produces reasonable outputs
    test_key = jax.random.PRNGKey(999)
    test_tokens = jax.random.randint(test_key, (1, seq_len), 0, vocab_size)
    test_logits = transformer_forward(params, test_tokens[:, :-1], n_heads)
    logit_mean = float(jnp.mean(test_logits))
    logit_std = float(jnp.std(test_logits))
    assert -5 <= logit_mean <= 5, f"Logit mean out of range: {logit_mean}"
    assert 0.01 <= logit_std <= 50, f"Logit std out of range: {logit_std}"

    print(f"ok  losses={[f'{l:.4f}' for l in losses]}")


if __name__ == "__main__":
    main()
