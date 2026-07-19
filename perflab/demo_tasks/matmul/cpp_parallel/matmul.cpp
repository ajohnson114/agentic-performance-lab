// Naive single-threaded matrix multiplication with cache-unfriendly access pattern.
// Anti-patterns:
//   1. Single-threaded — does not exploit multiple cores
//   2. i,j,k loop order — inner k-loop strides across rows of B, causing cache misses
//   3. No tiling — poor cache locality for large matrices
//   4. No SIMD hints — leaves vectorization on the table
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static void matmul(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];  // B access strides by N
            }
            C[i * N + j] = sum;
        }
    }
}

static double tflops(int M, int N, int K, double seconds) {
    double flops = 2.0 * M * N * K;
    return flops / seconds / 1e12;
}

static int selftest() {
    // Hardcoded 4x4 matmul verification
    const int N = 4;
    // A = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    float A[16], B[16], C[16];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A[i * N + j] = static_cast<float>(i * N + j + 1);
    // B = identity
    std::memset(B, 0, sizeof(B));
    for (int i = 0; i < N; ++i) B[i * N + i] = 1.0f;

    std::memset(C, 0, sizeof(C));
    matmul(A, B, C, N, N, N);

    // C should equal A (A * I = A)
    for (int i = 0; i < N * N; ++i) {
        float diff = C[i] - A[i];
        if (diff < -1e-5f || diff > 1e-5f) {
            std::cerr << "selftest FAILED at index " << i
                      << ": got " << C[i] << " expected " << A[i] << "\n";
            return 1;
        }
    }

    // Second test: A * B where B is all-ones, result should be row-sums
    for (int i = 0; i < N * N; ++i) B[i] = 1.0f;
    std::memset(C, 0, sizeof(C));
    matmul(A, B, C, N, N, N);
    for (int i = 0; i < N; ++i) {
        float expected_sum = 0.0f;
        for (int j = 0; j < N; ++j) expected_sum += A[i * N + j];
        for (int j = 0; j < N; ++j) {
            float diff = C[i * N + j] - expected_sum;
            if (diff < -1e-4f || diff > 1e-4f) {
                std::cerr << "selftest FAILED row-sum at (" << i << "," << j
                          << "): got " << C[i * N + j] << " expected " << expected_sum << "\n";
                return 1;
            }
        }
    }
    std::cout << "selftest passed\n";
    return 0;
}

int main(int argc, char** argv) {
    int M = 2048, N = 2048, K = 2048;
    bool json_output = false;
    int warmup = 2, repeats = 5;

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

    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);

    // Initialize with deterministic values
    std::srand(42);
    for (auto& v : A) v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
    for (auto& v : B) v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        matmul(A.data(), B.data(), C.data(), M, N, K);
    }

    // Benchmark
    std::vector<double> times_ms;
    for (int r = 0; r < repeats; ++r) {
        std::memset(C.data(), 0, C.size() * sizeof(float));
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul(A.data(), B.data(), C.data(), M, N, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    // Sort for percentiles
    std::vector<double> sorted_times = times_ms;
    std::sort(sorted_times.begin(), sorted_times.end());
    double p50 = sorted_times[sorted_times.size() / 2];
    double p95 = sorted_times[static_cast<int>(0.95 * (sorted_times.size() - 1))];
    double tflops_med = tflops(M, N, K, p50 / 1000.0);

    if (json_output) {
        std::cout << "{\n";
        std::cout << "  \"meta\": {\"M\": " << M << ", \"N\": " << N
                  << ", \"K\": " << K << "},\n";
        std::cout << "  \"times_ms\": [";
        for (size_t i = 0; i < times_ms.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << times_ms[i];
        }
        std::cout << "],\n";
        std::cout << "  \"latency_ms\": {\"p50\": " << p50
                  << ", \"p95\": " << p95 << "},\n";
        std::cout << "  \"tflops\": {\"median\": " << tflops_med << "},\n";
        std::cout << "  \"ok\": true\n";
        std::cout << "}\n";
    } else {
        std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";
        std::cout << "p50=" << p50 << " ms  p95=" << p95 << " ms\n";
        std::cout << "tflops_median=" << tflops_med << "\n";
    }
    return 0;
}
