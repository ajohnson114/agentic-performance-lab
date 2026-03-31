"""Naive Triton matmul kernel — one program instance per output element.

Each program computes a single element of C by looping over the full K
dimension. An optimizing agent should rewrite this to use block tiling with
tl.dot for much higher throughput.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Naive: each program computes one element of C."""
    pid = tl.program_id(axis=0)
    row = pid // N
    col = pid % N

    if row < M and col < N:
        acc = tl.zeros((1,), dtype=tl.float32)
        for k in range(K):
            a_val = tl.load(A_ptr + row * stride_am + k * stride_ak)
            b_val = tl.load(B_ptr + k * stride_bk + col * stride_bn)
            acc += a_val * b_val
        tl.store(C_ptr + row * stride_cm + col * stride_cn, acc)


def triton_matmul(A, B, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32):
    """Run the Triton matmul kernel on torch tensors A (MxK) and B (KxN)."""
    import torch

    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # One program per output element (naive grid)
    grid = (M * N,)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C
