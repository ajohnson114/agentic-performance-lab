"""Scaled dot-product attention — deliberately naive baseline for TPU optimization.

Starts with:
- fp32 everywhere (TPU MXUs run bf16 at 2x throughput)
- No jax.jit (recompiles every call)
- Explicit Python loops over heads (prevents XLA fusion)
- Naive O(n^2) attention materialization
- No padding alignment for TPU tile boundaries

An optimizing agent should discover:
- @jax.jit for compilation caching
- bfloat16 dtype for 2x MXU throughput
- Vectorized multi-head (no Python loop)
- Tiled/flash attention or jax.nn.dot_product_attention
- Padding to multiples of 128 for TPU tile alignment
- jax.lax.scan for sequence-parallel attention
"""

import jax
import jax.numpy as jnp


def attention(q, k, v, n_heads):
    """Multi-head scaled dot-product attention (naive implementation).

    Args:
        q: Query tensor, shape (batch, seq_len, d_model), float32
        k: Key tensor, shape (batch, seq_len, d_model), float32
        v: Value tensor, shape (batch, seq_len, d_model), float32
        n_heads: Number of attention heads

    Returns:
        Output tensor, shape (batch, seq_len, d_model)
    """
    B, T, D = q.shape
    head_dim = D // n_heads

    # Split into heads — deliberately using a Python loop (prevents fusion)
    outputs = []
    for h in range(n_heads):
        start = h * head_dim
        end = start + head_dim
        q_h = q[:, :, start:end]  # (B, T, head_dim)
        k_h = k[:, :, start:end]
        v_h = v[:, :, start:end]

        # Scaled dot-product attention
        scale = jnp.float32(head_dim) ** 0.5
        attn_weights = jnp.matmul(q_h, jnp.swapaxes(k_h, -2, -1)) / scale  # (B, T, T)

        # Causal mask
        mask = jnp.triu(jnp.ones((T, T), dtype=jnp.bool_), k=1)
        attn_weights = jnp.where(mask, jnp.float32(-1e9), attn_weights)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        head_out = jnp.matmul(attn_weights, v_h)  # (B, T, head_dim)
        outputs.append(head_out)

    # Concatenate heads
    return jnp.concatenate(outputs, axis=-1)  # (B, T, D)
