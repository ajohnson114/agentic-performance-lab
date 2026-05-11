/*
 * Naive CUDA reduction: sum an array of floats.
 *
 * Deliberate anti-patterns (for PerfLab to discover and fix):
 *   1. Pageable (non-pinned) host memory
 *   2. Per-iteration cudaMemcpy H2D + D2H
 *   3. cudaDeviceSynchronize after every kernel launch
 *   4. Naive reduction kernel (one thread per element, no shared mem)
 *   5. No streams, no overlap
 *
 * Build:  nvcc -O2 -o reduce_bin reduce.cu
 * Usage:  ./reduce_bin [--json] [--selftest] [--N <n>] [--iterations <n>]
 *         [--threadsPerBlock <n>] [--warmup <n>] [--repeats <n>]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#else
#error "This file must be compiled with nvcc"
#endif

// ---------------------------------------------------------------------------
// Naive reduction kernel: one thread accumulates across the full array using
// a stride loop.  No shared memory, no warp shuffle, no tree reduction.
// ---------------------------------------------------------------------------
__global__ void reduce_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N) {
    float sum = 0.0f;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N;
         i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    // Naive: atomic add from every thread (no tree reduction)
    atomicAdd(output, sum);
}

// ---------------------------------------------------------------------------
// Self-test: verify GPU reduction matches CPU reference
// ---------------------------------------------------------------------------
static bool selftest() {
    const int sizes[] = {1, 7, 256, 1024, 65536};
    for (int s = 0; s < 5; ++s) {
        int N = sizes[s];
        std::vector<float> h_data(N);
        srand(42 + s);
        for (int i = 0; i < N; ++i) h_data[i] = (float)(rand() % 1000) / 100.0f;

        float cpu_sum = 0.0f;
        for (int i = 0; i < N; ++i) cpu_sum += h_data[i];

        float *d_in, *d_out;
        cudaMalloc(&d_in, N * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_in, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_out, 0, sizeof(float));

        int block = 256;
        int grid = (N + block - 1) / block;
        if (grid > 1024) grid = 1024;
        reduce_kernel<<<grid, block>>>(d_in, d_out, N);
        cudaDeviceSynchronize();

        float gpu_sum = 0.0f;
        cudaMemcpy(&gpu_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost);

        float rel_err = fabsf(gpu_sum - cpu_sum) / fmaxf(fabsf(cpu_sum), 1e-6f);
        if (rel_err > 0.01f) {
            fprintf(stderr, "FAIL N=%d: cpu=%.4f gpu=%.4f rel_err=%.6f\n",
                    N, cpu_sum, gpu_sum, rel_err);
            cudaFree(d_in);
            cudaFree(d_out);
            return false;
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Benchmark: per-iteration H2D -> kernel -> D2H -> host accumulate
// ---------------------------------------------------------------------------
struct BenchResult {
    double median_time_s;
    double median_throughput_gbs;
    bool ok;
};

static BenchResult run_bench(int N, int iterations, int threadsPerBlock,
                              int warmup, int repeats) {
    size_t bytes = (size_t)N * sizeof(float);

    // Pageable host memory (anti-pattern #1)
    std::vector<float> h_input(N);
    srand(123);
    for (int i = 0; i < N; ++i) h_input[i] = (float)(rand() % 1000) / 100.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));

    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;

    auto run_iterations = [&]() -> double {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        float host_accum = 0.0f;
        for (int it = 0; it < iterations; ++it) {
            // Anti-pattern #2: per-iteration H2D copy
            nvtxRangePush("H2D");
            cudaMemcpy(d_in, h_input.data(), bytes, cudaMemcpyHostToDevice);
            nvtxRangePop();

            // Anti-pattern #4: naive kernel
            nvtxRangePush("kernel");
            cudaMemset(d_out, 0, sizeof(float));
            reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
            nvtxRangePop();

            // Anti-pattern #3: sync after every launch
            nvtxRangePush("sync");
            cudaDeviceSynchronize();
            nvtxRangePop();

            // Anti-pattern #2: per-iteration D2H copy
            nvtxRangePush("D2H");
            float partial = 0.0f;
            cudaMemcpy(&partial, d_out, sizeof(float), cudaMemcpyDeviceToHost);
            nvtxRangePop();

            host_accum += partial;
        }
        // Prevent compiler from optimizing away
        volatile float sink = host_accum;
        (void)sink;

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        return elapsed;
    };

    // Warmup
    for (int w = 0; w < warmup; ++w) run_iterations();

    // Timed repeats
    std::vector<double> times(repeats);
    for (int r = 0; r < repeats; ++r) {
        times[r] = run_iterations();
    }

    cudaFree(d_in);
    cudaFree(d_out);

    std::sort(times.begin(), times.end());
    double median_time = times[repeats / 2];

    // Throughput: total data touched per iteration = N*4 bytes (read) + 4 bytes (write)
    // Over `iterations` iterations
    double total_bytes = (double)iterations * ((double)N * sizeof(float) + sizeof(float));
    double throughput_gbs = (total_bytes / median_time) / 1e9;

    return {median_time, throughput_gbs, true};
}

// ---------------------------------------------------------------------------
// Main: parse args, dispatch selftest or benchmark
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int N = 16777216;    // 16M elements = 64 MB
    int iterations = 100;
    int threadsPerBlock = 256;
    int warmup = 3;
    int repeats = 10;
    bool json_output = false;
    bool do_selftest = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--selftest") == 0) {
            do_selftest = true;
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = true;
        } else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threadsPerBlock") == 0 && i + 1 < argc) {
            threadsPerBlock = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = atoi(argv[++i]);
        }
    }

    if (do_selftest) {
        bool ok = selftest();
        return ok ? 0 : 1;
    }

    BenchResult res = run_bench(N, iterations, threadsPerBlock, warmup, repeats);

    if (json_output) {
        printf("{\n");
        printf("  \"ok\": %s,\n", res.ok ? "true" : "false");
        printf("  \"throughput_gbs\": {\"median\": %.4f},\n", res.median_throughput_gbs);
        printf("  \"time_s\": {\"median\": %.6f},\n", res.median_time_s);
        printf("  \"meta\": {\"N\": %d, \"iterations\": %d, \"threadsPerBlock\": %d, \"repeats\": %d}\n",
               N, iterations, threadsPerBlock, repeats);
        printf("}\n");
    } else {
        printf("N=%d  iterations=%d  threadsPerBlock=%d\n", N, iterations, threadsPerBlock);
        printf("Median time: %.4f s\n", res.median_time_s);
        printf("Median throughput: %.2f GB/s\n", res.median_throughput_gbs);
    }

    return res.ok ? 0 : 1;
}
