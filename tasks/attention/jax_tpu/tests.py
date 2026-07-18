"""Correctness test for attention op.

Checks:
1. Output shape matches input shape
2. Output values are finite
3. Causal property: output at position t depends only on positions <= t
4. Numerical agreement with a reference implementation (within tolerance)
"""

import jax
import jax.numpy as jnp
from attention_op import attention


def _reference_attention(q, k, v, n_heads):
    """Reference vectorized attention for correctness checking."""
    B, T, D = q.shape
    head_dim = D // n_heads

    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)

    scale = head_dim ** 0.5
    attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
    mask = jnp.triu(jnp.ones((T, T), dtype=jnp.bool_), k=1)
    attn = jnp.where(mask, -1e9, attn)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.matmul(attn, v)
    return out.transpose(0, 2, 1, 3).reshape(B, T, D)


def main():
    key = jax.random.PRNGKey(42)
    batch_size = 2
    seq_len = 64
    d_model = 128
    n_heads = 4

    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (batch_size, seq_len, d_model), dtype=jnp.float32)
    k_tensor = jax.random.normal(k2, (batch_size, seq_len, d_model), dtype=jnp.float32)
    v = jax.random.normal(k3, (batch_size, seq_len, d_model), dtype=jnp.float32)

    # Test 1: Output shape
    out = attention(q, k_tensor, v, n_heads)
    assert out.shape == (batch_size, seq_len, d_model), (
        f"Shape mismatch: expected {(batch_size, seq_len, d_model)}, got {out.shape}"
    )

    # Test 2: Output is finite
    assert jnp.all(jnp.isfinite(out)), "Output contains NaN or Inf"

    # Test 3: Numerical agreement with reference
    ref_out = _reference_attention(q, k_tensor, v, n_heads)

    # Allow tolerance for bf16 implementations (wider tolerance)
    if out.dtype == jnp.bfloat16 or out.dtype == jnp.float16:
        atol = 0.1
        rtol = 0.05
    else:
        atol = 1e-4
        rtol = 1e-4

    # Cast both to float32 for comparison
    out_f32 = out.astype(jnp.float32)
    ref_f32 = ref_out.astype(jnp.float32)

    max_diff = float(jnp.max(jnp.abs(out_f32 - ref_f32)))
    assert jnp.allclose(out_f32, ref_f32, atol=atol, rtol=rtol), (
        f"Numerical mismatch: max diff = {max_diff:.6f} (atol={atol}, rtol={rtol})"
    )

    # Test 4: Causal property — changing future tokens shouldn't affect past output
    k2_mod = k_tensor.at[:, -1, :].set(999.0)  # change last position keys
    out_modified = attention(q, k2_mod, v, n_heads)
    # Output at position 0 should be identical (causal: can't see position -1... well,
    # position 0 only attends to itself)
    diff_pos0 = float(jnp.max(jnp.abs(
        out[:, 0, :].astype(jnp.float32) - out_modified[:, 0, :].astype(jnp.float32)
    )))
    assert diff_pos0 < 1e-5, (
        f"Causal violation: modifying future keys changed position 0 output by {diff_pos0}"
    )

    print(f"ok  shape={out.shape} dtype={out.dtype} max_diff={max_diff:.6f}")


if __name__ == "__main__":
    main()
