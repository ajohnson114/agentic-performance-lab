// Naive matrix multiplication with cache-unfriendly access pattern.
// The i,j,k loop order means the inner k-loop strides across rows of B,
// causing frequent cache misses.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// --- Benchmark barriers -----------------------------------------------------
// Without these the optimizer is free to delete the work being timed. At -O2 and
// above a computation whose result is never read is dead code (measured: an
// unobserved sum-of-squares loop compiles to zero FP instructions and the
// benchmark then reports 0.000000 ms), and a loop-invariant call can be hoisted
// out of the repeat loop so what gets timed is an empty loop.
//
// do_not_optimize(v)  makes an opaque asm block consume `v`, so `v` must really
//   be computed. The "r,m" multiple-alternative constraint lets the operand stay
//   in a REGISTER when it fits in one and falls back to MEMORY only when it does
//   not -- a bare "m" would force a spill to the stack on every call and inflate
//   the measurement. Passing a pointer (e.g. C.data()) makes the pointed-to
//   buffer escape, which is what keeps the stores into it live.
// clobber_memory()    tells the compiler the asm may read and write any memory,
//   so pending stores are committed before the clock is read and nothing stays
//   cached in a register across the barrier -- that is also what stops the
//   repeat loop from collapsing into a single iteration.
//
// Both emit zero instructions; they constrain only the optimizer.
#if defined(__GNUC__) || defined(__clang__)
template <class T>
inline void do_not_optimize(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}
inline void clobber_memory() {
    asm volatile("" : : : "memory");
}
#else
// MSVC has no inline asm on x64/ARM64. These tasks build with g++/nvcc, so this
// path exists only to keep the file compilable elsewhere; it is a weaker barrier.
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

// Ceiling-indexed percentile -- mirrors
// perflab.harness.precision._ceil_percentile_index. Floor indexing
// (static_cast<int>(fraction * (n - 1))) rounds toward the middle of the
// distribution, so at small sample counts (n=2 during fast screening, n=5 by
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
    int M = 512, N = 512, K = 512;
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

    // Warmup. The barrier matters here too: without it the warmup calls are
    // dead code and the caches never get warmed.
    for (int w = 0; w < warmup; ++w) {
        matmul(A.data(), B.data(), C.data(), M, N, K);
        do_not_optimize(C.data());
        clobber_memory();
    }

    // Benchmark. steady_clock, not high_resolution_clock: on libstdc++ the
    // latter is a typedef for system_clock, i.e. the wall clock, which is NOT
    // monotonic -- an NTP step mid-benchmark yields a wrong or negative
    // interval. steady_clock is guaranteed monotonic and is the correct clock
    // for measuring a duration.
    std::vector<double> times_ms;
    for (int r = 0; r < repeats; ++r) {
        std::memset(C.data(), 0, C.size() * sizeof(float));
        auto t0 = std::chrono::steady_clock::now();
        clobber_memory();               // work below may not be hoisted above t0
        matmul(A.data(), B.data(), C.data(), M, N, K);
        do_not_optimize(C.data());      // C escapes: the stores into it are live
        clobber_memory();               // and are committed before t1 is read
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    // Sort for percentiles
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
                  << ", \"p95\": " << p95 << ", \"raw_values\": [";
        for (size_t i = 0; i < times_ms.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << times_ms[i];
        }
        std::cout << "]},\n";
        std::cout << "  \"tflops\": {\"median\": " << tflops_med << ", \"raw_values\": [";
        for (size_t i = 0; i < tflops_list.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << tflops_list[i];
        }
        std::cout << "]},\n";
        std::cout << "  \"ok\": true\n";
        std::cout << "}\n";
    } else {
        std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";
        std::cout << "p50=" << p50 << " ms  p95=" << p95 << " ms\n";
        std::cout << "tflops_median=" << tflops_med << "\n";
    }
    return 0;
}
