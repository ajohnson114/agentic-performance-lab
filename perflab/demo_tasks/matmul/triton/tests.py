"""Correctness test: verify Triton matmul kernel against torch reference."""
from __future__ import annotations

import torch
from matmul_kernel import triton_matmul


def main():
    M, N, K = 128, 128, 128
    dev = torch.device("cuda")

    torch.manual_seed(0)
    A = torch.randn(M, K, device=dev, dtype=torch.float32)
    B = torch.randn(K, N, device=dev, dtype=torch.float32)

    C_triton = triton_matmul(A, B)
    C_ref = torch.matmul(A, B)

    max_abs = (C_triton - C_ref).abs().max().item()
    # float32 accumulation should be quite accurate
    assert max_abs < 1e-3, f"max_abs too large: {max_abs}"
    print("ok", {"M": M, "N": N, "K": K, "max_abs": max_abs})


if __name__ == "__main__":
    main()
