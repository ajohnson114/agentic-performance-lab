import torch
from matmul_op import matmul_op


def main():
    n = torch.cuda.device_count()
    if n < 2:
        raise RuntimeError(
            f"multi_gpu_matmul requires >=2 CUDA devices, found {n}. "
            "This task demonstrates multi-GPU profiling and is meant to run "
            "on a multi-GPU box (e.g. a 2x+ GPU RunPod instance)."
        )
    devices = [torch.device(f"cuda:{i}") for i in range(n)]
    dev0 = devices[0]

    # Small correctness test vs CPU float32 reference.
    M = N = K = 256
    torch.manual_seed(0)
    A = torch.randn(M, K, device=dev0, dtype=torch.float16)
    B = torch.randn(K, N, device=dev0, dtype=torch.float16)

    C = matmul_op(A, B, devices)

    A_cpu = A.detach().to("cpu", dtype=torch.float32)
    B_cpu = B.detach().to("cpu", dtype=torch.float32)
    C_ref = A_cpu @ B_cpu
    C_test = C.detach().to("cpu", dtype=torch.float32)

    max_abs = (C_test - C_ref).abs().max().item()
    tol = 5e-2  # fp16 matmul tolerance, matches matmul/pytorch
    assert max_abs < tol, f"max_abs too large: {max_abs} (tol={tol})"
    print("ok", {"num_gpus": n, "max_abs": max_abs})


if __name__ == "__main__":
    main()
