// Naive SGEMM CUDA kernel.
// Each thread computes one element of C by walking the full K dimension.
// An optimizing agent should add shared-memory tiling, loop unrolling, etc.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

__global__ void sgemm_naive(int M, int N, int K,
                            const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

static double tflops(int M, int N, int K, double seconds) {
    double flops = 2.0 * M * N * K;
    return flops / seconds / 1e12;
}

static int selftest() {
    // Hardcoded 4x4 GPU matmul verification against CPU reference
    const int N = 4;
    float h_A[16], h_B[16], h_C[16];

    // A = [[1..16]], B = identity
    for (int i = 0; i < N * N; ++i) h_A[i] = static_cast<float>(i + 1);
    memset(h_B, 0, sizeof(h_B));
    for (int i = 0; i < N; ++i) h_B[i * N + i] = 1.0f;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(h_A)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(h_B)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(h_C)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeof(h_C)));

    dim3 block(N, N);
    dim3 grid(1, 1);
    sgemm_naive<<<grid, block>>>(N, N, N, d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost));

    // C should equal A (A * I = A)
    for (int i = 0; i < N * N; ++i) {
        float diff = h_C[i] - h_A[i];
        if (diff < -1e-5f || diff > 1e-5f) {
            fprintf(stderr, "selftest FAILED at %d: got %f expected %f\n", i, h_C[i], h_A[i]);
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return 1;
        }
    }

    // Second test: B = all-ones, verify row-sums
    for (int i = 0; i < N * N; ++i) h_B[i] = 1.0f;
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeof(h_C)));
    sgemm_naive<<<grid, block>>>(N, N, N, d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        float expected_sum = 0.0f;
        for (int j = 0; j < N; ++j) expected_sum += h_A[i * N + j];
        for (int j = 0; j < N; ++j) {
            float diff = h_C[i * N + j] - expected_sum;
            if (diff < -1e-4f || diff > 1e-4f) {
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
    int block_size = 16;
    bool json_output = false;
    int warmup = 3, repeats = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--selftest") return selftest();
        if (arg == "--M" && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "--block_size" && i + 1 < argc) block_size = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (arg == "--repeats" && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (arg == "--json") json_output = true;
    }

    size_t sA = M * K * sizeof(float);
    size_t sB = K * N * sizeof(float);
    size_t sC = M * N * sizeof(float);

    // Host allocation
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f);
    std::srand(42);
    for (auto& v : h_A) v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
    for (auto& v : h_B) v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;

    // Device allocation
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sA));
    CHECK_CUDA(cudaMalloc(&d_B, sB));
    CHECK_CUDA(cudaMalloc(&d_C, sC));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), sB, cudaMemcpyHostToDevice));

    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        sgemm_naive<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    std::vector<double> times_ms;
    for (int r = 0; r < repeats; ++r) {
        CHECK_CUDA(cudaMemset(d_C, 0, sC));
        CHECK_CUDA(cudaDeviceSynchronize());

        auto t0 = std::chrono::high_resolution_clock::now();
        sgemm_naive<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
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
        printf("  \"meta\": {\"M\": %d, \"N\": %d, \"K\": %d, \"block_size\": %d},\n",
               M, N, K, block_size);
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
        printf("M=%d N=%d K=%d block_size=%d\n", M, N, K, block_size);
        printf("p50=%.4f ms  p95=%.4f ms\n", p50, p95);
        printf("tflops_median=%.6f\n", tflops_med);
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}
