// Naive SGEMM CUDA kernel.
// Each thread computes one element of C by walking the full K dimension.
// An optimizing agent should add shared-memory tiling, loop unrolling, etc.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

// --- Benchmark barriers (host side) -----------------------------------------
// These constrain the HOST compiler only. What makes the timing of a kernel
// launch correct is the cudaDeviceSynchronize() inside the timed region below --
// a launch is asynchronous, so without the sync the host clock would measure
// only the launch overhead. The barriers are here so the surrounding host code
// (buffer setup, result handling) cannot be sunk out of or hoisted into the
// timed region if this benchmark loop is ever restructured.
//
// do_not_optimize(v)  makes an opaque asm block consume `v`, so `v` must really
//   be computed. The "r,m" multiple-alternative constraint lets the operand stay
//   in a REGISTER when it fits in one and falls back to MEMORY only when it does
//   not -- a bare "m" would force a spill to the stack on every call and inflate
//   the measurement.
// clobber_memory()    tells the compiler the asm may read and write any memory,
//   so pending stores are committed before the clock is read and nothing stays
//   cached in a register across the barrier.
//
// Both emit zero instructions; they constrain only the optimizer. They are host
// functions (no __device__ annotation) and are never called from device code.
#if defined(__GNUC__) || defined(__clang__)
template <class T>
inline void do_not_optimize(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}
inline void clobber_memory() {
    asm volatile("" : : : "memory");
}
#else
// MSVC has no inline asm on x64/ARM64. These tasks build with nvcc over g++/clang,
// so this path exists only to keep the file compilable; it is a weaker barrier.
#include <atomic>
inline void clobber_memory() {
    std::atomic_signal_fence(std::memory_order_acq_rel);
}
template <class T>
inline void do_not_optimize(const T& value) {
    volatile const T* sink = &value;
    (void)sink;
    clobber_memory();
}
#endif

#define CHECK_CUDA(call)                                                     \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

__global__ void sgemm_kernel(int M, int N, int K,
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

// Ceiling-indexed percentile -- mirrors
// perflab.harness.precision._ceil_percentile_index. Floor indexing
// (static_cast<int>(fraction * (n - 1))) rounds toward the middle of the
// distribution, so at small sample counts (n=2 during fast screening, n=10 by
// default here) the extreme value falls just past the computed index and is
// never reported -- e.g. n=2: floor(0.95*1)=0 returns the MINIMUM as "p95".
static double ceil_percentile(const std::vector<double>& sorted_values, double fraction) {
    int n = static_cast<int>(sorted_values.size());
    int idx = static_cast<int>(std::ceil(fraction * (n - 1)));
    if (idx > n - 1) idx = n - 1;
    if (idx < 0) idx = 0;
    return sorted_values[idx];
}

// True median: averages the middle pair for an even sample count.
// sorted_values[n / 2] alone (the previous implementation here) is biased
// high for even n -- at n=2 it always returns the larger of the two samples.
// Matches perflab.analyzers.bench_stats.compute_bench_stats's median.
static double true_median(const std::vector<double>& sorted_values) {
    int n = static_cast<int>(sorted_values.size());
    if (n % 2 == 0) {
        return (sorted_values[n / 2 - 1] + sorted_values[n / 2]) / 2.0;
    }
    return sorted_values[n / 2];
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
    sgemm_kernel<<<grid, block>>>(N, N, N, d_A, d_B, d_C);
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
    sgemm_kernel<<<grid, block>>>(N, N, N, d_A, d_B, d_C);
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
    int threadsPerBlock = 16;
    bool json_output = false;
    int warmup = 3, repeats = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--selftest") return selftest();
        if (arg == "--M" && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "--threadsPerBlock" && i + 1 < argc) threadsPerBlock = std::atoi(argv[++i]);
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

    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 grid((N + threadsPerBlock - 1) / threadsPerBlock, (M + threadsPerBlock - 1) / threadsPerBlock);

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        sgemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    std::vector<double> times_ms;
    for (int r = 0; r < repeats; ++r) {
        CHECK_CUDA(cudaMemset(d_C, 0, sC));
        CHECK_CUDA(cudaDeviceSynchronize());

        // steady_clock, not high_resolution_clock: on libstdc++ the latter is a
        // typedef for system_clock, i.e. the wall clock, which is NOT monotonic
        // -- an NTP step mid-benchmark yields a wrong or negative interval.
        auto t0 = std::chrono::steady_clock::now();
        clobber_memory();
        sgemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
        // The sync is what makes this measurement meaningful: the launch above
        // is asynchronous, so without it t1 would capture launch overhead only.
        CHECK_CUDA(cudaDeviceSynchronize());
        do_not_optimize(d_C);
        clobber_memory();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    // Percentiles
    std::vector<double> sorted_times = times_ms;
    std::sort(sorted_times.begin(), sorted_times.end());
    double p50 = true_median(sorted_times);
    double p95 = ceil_percentile(sorted_times, 0.95);
    double tflops_med = tflops(M, N, K, p50 / 1000.0);

    // Per-repeat tflops, in measurement order -- what the accept gate's
    // variance check (extract_repeated_values) needs alongside the aggregate.
    std::vector<double> tflops_list;
    tflops_list.reserve(times_ms.size());
    for (double ms : times_ms) tflops_list.push_back(tflops(M, N, K, ms / 1000.0));

    if (json_output) {
        printf("{\n");
        printf("  \"meta\": {\"M\": %d, \"N\": %d, \"K\": %d, \"threadsPerBlock\": %d},\n",
               M, N, K, threadsPerBlock);
        printf("  \"times_ms\": [");
        for (size_t i = 0; i < times_ms.size(); ++i) {
            if (i) printf(", ");
            printf("%.4f", times_ms[i]);
        }
        printf("],\n");
        printf("  \"latency_ms\": {\"p50\": %.4f, \"p95\": %.4f, \"raw_values\": [", p50, p95);
        for (size_t i = 0; i < times_ms.size(); ++i) {
            if (i) printf(", ");
            printf("%.4f", times_ms[i]);
        }
        printf("]},\n");
        printf("  \"tflops\": {\"median\": %.6f, \"raw_values\": [", tflops_med);
        for (size_t i = 0; i < tflops_list.size(); ++i) {
            if (i) printf(", ");
            printf("%.6f", tflops_list[i]);
        }
        printf("]},\n");
        printf("  \"ok\": true\n");
        printf("}\n");
    } else {
        printf("M=%d N=%d K=%d threadsPerBlock=%d\n", M, N, K, threadsPerBlock);
        printf("p50=%.4f ms  p95=%.4f ms\n", p50, p95);
        printf("tflops_median=%.6f\n", tflops_med);
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}
