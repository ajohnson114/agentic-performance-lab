from pathlib import Path

import torch
import yaml


def main():
    knobs = yaml.safe_load(Path("tuning.yaml").read_text())
    dtype = knobs.get("dtype", "fp16")
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # Small correctness test vs CPU float32 reference
    M = N = K = 256
    torch.manual_seed(0)
    A = torch.randn(M, K, device=dev, dtype=torch_dtype)
    B = torch.randn(K, N, device=dev, dtype=torch_dtype)

    from matmul_op import matmul_op
    C = matmul_op(A, B)

    A_cpu = A.detach().to("cpu", dtype=torch.float32)
    B_cpu = B.detach().to("cpu", dtype=torch.float32)
    C_ref = A_cpu @ B_cpu
    C_test = C.detach().to("cpu", dtype=torch.float32)

    max_abs = (C_test - C_ref).abs().max().item()
    tol = 5e-2 if dtype == "fp16" else 1e-5
    assert max_abs < tol, f"max_abs too large: {max_abs} (tol={tol}, dtype={dtype})"
    print("ok", {"device": dev, "dtype": dtype, "max_abs": max_abs})

if __name__ == "__main__":
    main()
