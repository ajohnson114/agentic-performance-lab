// Naive HGEMM (Half-precision GEMM) using WMMA Tensor Core intrinsics.
// Each warp computes one 16x16 tile of C using a single wmma::mma_sync call
// per K-step — no shared memory, no tiling across warps.
// An optimizing agent should add shared-memory tiling, double buffering,
// warp-level pipelining, and multi-stage async copies.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <string>
#include <vector>

using namespace nvcuda;

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// WMMA tile dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Naive WMMA kernel: each warp handles one 16x16 output tile.
// Loads A and B fragments directly from global memory — no shared memory.
__global__ void hgemm_kernel(int M, int N, int K,
                                  const half* A, const half* B, float* C) {
    // Warp-level coordinates
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * WMMA_N;

    if (warpM >= M || warpN >= N) return;

    // Accumulator fragment (FP32)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Walk the K dimension in steps of WMMA_K
    for (int k = 0; k < K; k += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

        // Load directly from global memory (naive — no shared mem tiling)
        wmma::load_matrix_sync(a_frag, A + warpM * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN, N);

        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // Store result to global memory
    wmma::store_matrix_sync(C + warpM * N + warpN, acc, N, wmma::mem_row_major);
}

static double tflops(int M, int N, int K, double seconds) {
    double flops = 2.0 * M * N * K;
    return flops / seconds / 1e12;
}

// Convert FP32 array to FP16 on host
static std::vector<half> float_to_half(const std::vector<float>& src) {
    std::vector<half> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = __float2half(src[i]);
    }
    return dst;
}

static int selftest() {
    // 16x16 matmul verification: A * identity = A
    const int N = 16;  // Must be multiple of WMMA_M/N/K = 16
    std::vector<float> h_A_fp32(N * N), h_B_fp32(N * N, 0.0f);
    std::vector<float> h_C(N * N, 0.0f);

    for (int i = 0; i < N * N; ++i) h_A_fp32[i] = static_cast<float>(i + 1) / (N * N);
    for (int i = 0; i < N; ++i) h_B_fp32[i * N + i] = 1.0f;

    auto h_A = float_to_half(h_A_fp32);
    auto h_B = float_to_half(h_B_fp32);

    half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, N * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, N * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), N * N * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, N * N * sizeof(float)));

    // 1 warp = 32 threads, need 1 warp for 16x16 tile
    dim3 block(32, 1);
    dim3 grid(1, 1);
    hgemm_kernel<<<grid, block>>>(N, N, N, d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // C should approximate A (A * I = A), with FP16 tolerance
    for (int i = 0; i < N * N; ++i) {
        float expected = h_A_fp32[i];
        float diff = h_C[i] - expected;
        if (diff < -0.01f || diff > 0.01f) {
            fprintf(stderr, "selftest FAILED at %d: got %f expected %f\n", i, h_C[i], expected);
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return 1;
        }
    }

    // Second test: B = all-ones, verify row-sums
    for (int i = 0; i < N * N; ++i) h_B_fp32[i] = 1.0f;
    h_B = float_to_half(h_B_fp32);
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), N * N * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, N * N * sizeof(float)));
    hgemm_kernel<<<grid, block>>>(N, N, N, d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        float expected_sum = 0.0f;
        for (int j = 0; j < N; ++j) expected_sum += h_A_fp32[i * N + j];
        for (int j = 0; j < N; ++j) {
            float diff = h_C[i * N + j] - expected_sum;
            if (diff < -0.05f || diff > 0.05f) {
                fprintf(stderr, "selftest FAILED row-sum at (%d,%d): got %f expected %f\n",
                        i, j, h_C[i * N + j], expected_sum);
                cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
                return 1;
            }
        }
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    printf("selftest passed\n");
    return 0;
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    bool json_output = false;
    int warmup = 3, repeats = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--selftest") return selftest();
        if (arg == "--M" && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (arg == "--repeats" && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (arg == "--json") json_output = true;
    }

    // Validate dimensions are multiples of 16 for WMMA
    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        fprintf(stderr, "Error: M, N, K must be multiples of 16 for WMMA. Got M=%d N=%d K=%d\n", M, N, K);
        return 1;
    }

    // Host allocation in FP32, convert to FP16
    std::vector<float> h_A_fp32(M * K), h_B_fp32(K * N);
    std::srand(42);
    for (auto& v : h_A_fp32) v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
    for (auto& v : h_B_fp32) v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;

    auto h_A = float_to_half(h_A_fp32);
    auto h_B = float_to_half(h_B_fp32);

    size_t sA = M * K * sizeof(half);
    size_t sB = K * N * sizeof(half);
    size_t sC = M * N * sizeof(float);

    half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sA));
    CHECK_CUDA(cudaMalloc(&d_B, sB));
    CHECK_CUDA(cudaMalloc(&d_C, sC));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), sB, cudaMemcpyHostToDevice));

    // Launch config: each warp handles a 16x16 tile
    // Use 4 warps per block (128 threads), arranged as 2x2 tiles
    int warps_per_block_x = 2;
    int warps_per_block_y = 2;
    dim3 block(warps_per_block_x * 32, warps_per_block_y);
    int grid_x = (N + warps_per_block_x * WMMA_N - 1) / (warps_per_block_x * WMMA_N);
    int grid_y = (M + warps_per_block_y * WMMA_M - 1) / (warps_per_block_y * WMMA_M);
    dim3 grid(grid_x, grid_y);

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        hgemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    std::vector<double> times_ms;
    for (int r = 0; r < repeats; ++r) {
        CHECK_CUDA(cudaMemset(d_C, 0, sC));
        CHECK_CUDA(cudaDeviceSynchronize());

        auto t0 = std::chrono::high_resolution_clock::now();
        hgemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    // Percentiles
    std::vector<double> sorted_times = times_ms;
    std::sort(sorted_times.begin(), sorted_times.end());
    double p50 = sorted_times[sorted_times.size() / 2];
    double p95 = sorted_times[static_cast<int>(0.95 * (sorted_times.size() - 1))];
    double tflops_med = tflops(M, N, K, p50 / 1000.0);

    if (json_output) {
        printf("{\n");
        printf("  \"meta\": {\"M\": %d, \"N\": %d, \"K\": %d, \"dtype\": \"fp16\"},\n", M, N, K);
        printf("  \"times_ms\": [");
        for (size_t i = 0; i < times_ms.size(); ++i) {
            if (i) printf(", ");
            printf("%.4f", times_ms[i]);
        }
        printf("],\n");
        printf("  \"latency_ms\": {\"p50\": %.4f, \"p95\": %.4f},\n", p50, p95);
        printf("  \"tflops\": {\"median\": %.6f},\n", tflops_med);
        printf("  \"ok\": true\n");
        printf("}\n");
    } else {
        printf("M=%d N=%d K=%d dtype=fp16\n", M, N, K);
        printf("p50=%.4f ms  p95=%.4f ms\n", p50, p95);
        printf("tflops_median=%.6f\n", tflops_med);
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}
