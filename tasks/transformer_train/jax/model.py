"""Pure JAX decoder-only transformer (no Flax dependency).

Params are plain pytrees (nested dicts of jax arrays), clean for optax.
Config values (n_heads) are passed as function args, not stored in the pytree.

Uses naive manual attention. An optimizing agent should discover and apply
jax.jit, efficient attention, and mixed precision.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Parameter initialization
# ---------------------------------------------------------------------------

def _init_linear(key, in_dim, out_dim):
    """Xavier-uniform initialized linear layer."""
    limit = (6.0 / (in_dim + out_dim)) ** 0.5
    w = jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)
    b = jnp.zeros(out_dim)
    return {"w": w, "b": b}


def _init_layer_norm(d_model):
    return {"scale": jnp.ones(d_model), "bias": jnp.zeros(d_model)}


def _init_attention(key, d_model):
    k1, k2 = jax.random.split(key)
    return {
        "qkv": _init_linear(k1, d_model, 3 * d_model),
        "out_proj": _init_linear(k2, d_model, d_model),
    }


def _init_block(key, d_model, d_ff):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "ln1": _init_layer_norm(d_model),
        "attn": _init_attention(k1, d_model),
        "ln2": _init_layer_norm(d_model),
        "ffn_up": _init_linear(k2, d_model, d_ff),
        "ffn_down": _init_linear(k3, d_ff, d_model),
    }


def init_transformer(key, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
    """Initialize transformer parameters as a plain pytree."""
    keys = jax.random.split(key, n_layers + 2)
    tok_emb = jax.random.normal(keys[0], (vocab_size, d_model)) * 0.02
    pos_emb = jax.random.normal(keys[1], (max_seq_len, d_model)) * 0.02
    blocks = [_init_block(keys[i + 2], d_model, d_ff) for i in range(n_layers)]
    return {
        "tok_emb": tok_emb,
        "pos_emb": pos_emb,
        "blocks": blocks,
        "ln_f": _init_layer_norm(d_model),
        "head_w": tok_emb,  # weight tying
    }


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _layer_norm(x, params):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return params["scale"] * (x - mean) / jnp.sqrt(var + 1e-5) + params["bias"]


def _linear(x, params):
    return x @ params["w"] + params["b"]


def _naive_attention(q, k, v, n_heads):
    """Manual scaled dot-product attention with causal mask (deliberately slow)."""
    B, T, C = q.shape
    head_dim = C // n_heads

    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)  # (B, H, T, D)
    k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)

    scale = head_dim ** 0.5
    attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale  # (B, H, T, T)

    # Causal mask
    mask = jnp.triu(jnp.ones((T, T), dtype=jnp.bool_), k=1)
    attn = jnp.where(mask, -1e9, attn)
    attn = jax.nn.softmax(attn, axis=-1)

    out = jnp.matmul(attn, v)  # (B, H, T, D)
    return out.transpose(0, 2, 1, 3).reshape(B, T, C)


def _attn_forward(params, x, n_heads):
    qkv = _linear(x, params["qkv"])
    q, k, v = jnp.split(qkv, 3, axis=-1)
    out = _naive_attention(q, k, v, n_heads)
    return _linear(out, params["out_proj"])


def _block_forward(params, x, n_heads):
    h = _layer_norm(x, params["ln1"])
    h = _attn_forward(params["attn"], h, n_heads)
    x = x + h
    h = _layer_norm(x, params["ln2"])
    h = _linear(h, params["ffn_up"])
    h = jax.nn.gelu(h)
    h = _linear(h, params["ffn_down"])
    return x + h


def transformer_forward(params, idx, n_heads):
    """Forward pass returning logits. idx: (B, T) integer tokens."""
    B, T = idx.shape
    tok = params["tok_emb"][idx]                         # (B, T, D)
    pos = params["pos_emb"][jnp.arange(T)]               # (T, D)
    x = tok + pos

    for block_params in params["blocks"]:
        x = _block_forward(block_params, x, n_heads)

    x = _layer_norm(x, params["ln_f"])
    logits = x @ params["head_w"].T                      # weight-tied output
    return logits


def cross_entropy_loss(logits, targets):
    """Cross-entropy loss for language modeling."""
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])
