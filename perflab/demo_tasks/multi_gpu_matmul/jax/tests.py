import jax
import jax.numpy as jnp
from matmul_op import matmul_op


def main():
    devices = jax.devices()
    if len(devices) < 2:
        raise RuntimeError(
            f"multi_gpu_matmul requires >=2 JAX devices, found {len(devices)}. "
            "This task demonstrates multi-GPU profiling and is meant to run "
            "on a multi-GPU box (e.g. a 2x+ GPU RunPod instance)."
        )

    # Small correctness test vs a plain (single-device) jnp.matmul reference.
    M = N = K = 64
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (M, K), dtype=jnp.float32)
    B = jax.random.normal(k2, (K, N), dtype=jnp.float32)

    C = matmul_op(A, B, devices)
    C_ref = jnp.matmul(A, B)

    max_abs = float(jnp.abs(C - C_ref).max())
    tol = 1e-3
    assert max_abs < tol, f"max_abs too large: {max_abs} (tol={tol})"
    print("ok", {"num_gpus": len(devices), "max_abs": max_abs})


if __name__ == "__main__":
    main()
