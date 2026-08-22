// Multi-GPU tensor-parallel SGEMM: the K (reduction) dimension is split
// across every visible CUDA device, each device computes a partial GEMM on
// its own K-shard with the SAME naive per-thread kernel as matmul/cuda, and
// NCCL's ncclAllReduce sums the partial results into the final output --
// genuine NCCL collective communication, using ncclCommInitAll (the
// documented single-process, multiple-GPUs-per-thread initialization
// pattern; distinct from the one-rank-per-process pattern DDP/FSDP use).
//
// The split below is deliberately lopsided (device 0 gets ~90% of K,
// everyone else splits the rest) -- a load-imbalance bug for the agent to
// find (perflab's gpu_active_pct_by_device / "GPU load imbalance"
// bottleneck rule) and fix by balancing the split.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <nccl.h>
#include <string>
#include <vector>

// --- Benchmark barriers (host side) -----------------------------------------
// See matmul/cuda/sgemm.cu for the full rationale -- identical here.
#if defined(__GNUC__) || defined(__clang__)
template <class T>
inline void do_not_optimize(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}
inline void clobber_memory() {
    asm volatile("" : : : "memory");
}
#else
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

#define CHECK_NCCL(call)                                                     \
    do {                                                                       \
        ncclResult_t res = (call);                                             \
        if (res != ncclSuccess) {                                              \
            fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    ncclGetErrorString(res));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Same naive per-thread kernel as matmul/cuda/sgemm.cu, unchanged: each
// thread computes one element of the LOCAL partial C by walking this
// device's own K-shard.
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

// See matmul/cuda/sgemm.cu for the full rationale -- identical here.
static double ceil_percentile(const std::vector<double>& sorted_values, double fraction) {
    int n = static_cast<int>(sorted_values.size());
    int idx = static_cast<int>(std::ceil(fraction * (n - 1)));
    if (idx > n - 1) idx = n - 1;
    if (idx < 0) idx = 0;
    return sorted_values[idx];
}

static double true_median(const std::vector<double>& sorted_values) {
    int n = static_cast<int>(sorted_values.size());
    if (n % 2 == 0) {
        return (sorted_values[n / 2 - 1] + sorted_values[n / 2]) / 2.0;
    }
    return sorted_values[n / 2];
}

// Naive split: device 0 gets the lion's share of K, everyone else splits
// what's left evenly. See matmul_op.py in the pytorch/jax siblings of this
// task for the same split, in the same words.
static std::vector<int> split_sizes(int K, int numDevices) {
    std::vector<int> sizes(numDevices);
    if (numDevices == 1) {
        sizes[0] = K;
        return sizes;
    }
    int first = static_cast<int>(K * 0.9);
    int rest = K - first;
    int base = rest / (numDevices - 1);
    sizes[0] = first;
    for (int i = 1; i < numDevices; ++i) sizes[i] = base;
    sizes[numDevices - 1] += rest - base * (numDevices - 1);
    return sizes;
}

// Runs one multi-GPU SGEMM: splits K across `numDevices` per split_sizes(),
// computes each device's local partial on its own shard, all-reduces the
// partials via NCCL, and copies the final (M, N) result back into h_C
// (host, size M*N floats, caller-allocated). Devices 0..numDevices-1 are
// used in device-index order.
static void multi_gpu_sgemm(int M, int N, int K, int numDevices,
                            int threadsPerBlock,
                            const std::vector<float>& h_A,
                            const std::vector<float>& h_B,
                            std::vector<float>& h_C,
                            ncclComm_t* comms) {
    std::vector<int> sizes = split_sizes(K, numDevices);

    std::vector<float*> d_A(numDevices), d_B(numDevices), d_C(numDevices);
    std::vector<cudaStream_t> streams(numDevices);

    int offset = 0;
    for (int i = 0; i < numDevices; ++i) {
        int k_i = sizes[i];
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        size_t sA = static_cast<size_t>(M) * k_i * sizeof(float);
        size_t sB = static_cast<size_t>(k_i) * N * sizeof(float);
        size_t sC = static_cast<size_t>(M) * N * sizeof(float);
        CHECK_CUDA(cudaMalloc(&d_A[i], sA));
        CHECK_CUDA(cudaMalloc(&d_B[i], sB));
        CHECK_CUDA(cudaMalloc(&d_C[i], sC));

        // A's K-shard: rows are contiguous in K, so each row's shard is a
        // contiguous k_i-float run starting at offset within that row.
        std::vector<float> A_shard(static_cast<size_t>(M) * k_i);
        for (int m = 0; m < M; ++m) {
            std::memcpy(&A_shard[static_cast<size_t>(m) * k_i],
                        &h_A[static_cast<size_t>(m) * K + offset],
                        static_cast<size_t>(k_i) * sizeof(float));
        }
        // Synchronous, not Async: A_shard is a stack-local temporary that
        // goes out of scope at the end of this loop iteration. An async
        // copy would race the host buffer being freed before the transfer
        // completes -- cudaMemcpy blocks until the copy is done, so it's
        // safe for A_shard to be destroyed right after.
        CHECK_CUDA(cudaMemcpy(d_A[i], A_shard.data(), sA, cudaMemcpyHostToDevice));
        // B's K-shard: rows offset..offset+k_i are contiguous in row-major
        // B. Safe to keep async -- h_B is the caller's buffer, alive for
        // the whole benchmark run, not freed until well after this
        // function (and any pending copy from it) has completed.
        CHECK_CUDA(cudaMemcpyAsync(d_B[i], &h_B[static_cast<size_t>(offset) * N], sB, cudaMemcpyHostToDevice, streams[i]));

        dim3 block(threadsPerBlock, threadsPerBlock);
        dim3 grid((N + threadsPerBlock - 1) / threadsPerBlock, (M + threadsPerBlock - 1) / threadsPerBlock);
        sgemm_kernel<<<grid, block, 0, streams[i]>>>(M, N, k_i, d_A[i], d_B[i], d_C[i]);

        offset += k_i;
    }

    // All-reduce the per-device (M, N) partials in place: every device ends
    // up holding the same summed result. Grouped, per NCCL's documented
    // requirement for issuing multiple collective calls across devices from
    // a single thread (ncclCommInitAll's use case).
    int count = M * N;
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < numDevices; ++i) {
        // The allocation/kernel-launch loop above leaves the CUDA context on
        // numDevices-1 -- ncclAllReduce validates d_C[i] against whichever
        // device is CURRENTLY active, not the device the pointer was
        // allocated on, so every i other than the last one fails with
        // "Cuda failure 1 'invalid argument'" without this.
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_NCCL(ncclAllReduce(d_C[i], d_C[i], count, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    CHECK_NCCL(ncclGroupEnd());

    for (int i = 0; i < numDevices; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // Every device holds the identical all-reduced result; device 0's copy
    // is as good as any.
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C[0], static_cast<size_t>(M) * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < numDevices; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(d_A[i]));
        CHECK_CUDA(cudaFree(d_B[i]));
        CHECK_CUDA(cudaFree(d_C[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
}

static int selftest(int numDevices) {
    // A = sequential values, B = identity -> C should equal A (A * I = A).
    // Exercises the real multi-GPU K-split + NCCL all-reduce path with
    // whatever device count is actually available, not a hardcoded 2.
    //
    // N=128, not a tiny 4: split_sizes()'s 90/10 split gives the non-first
    // devices rest/(numDevices-1) each of K -- with a K of just 4, that's
    // 0 on any box with more than ~5 GPUs (an all-zero shard, which the
    // rest of this file happens to handle gracefully, but there's no
    // reason to rely on that in the one place meant to catch bugs). 128
    // keeps every device's shard non-empty for any GPU count this task is
    // realistically run on, and it's still a trivially fast selftest.
    const int N = 128;
    const int count = N * N;
    std::vector<float> h_A(count), h_B(count, 0.0f), h_C(count, 0.0f);
    for (int i = 0; i < count; ++i) h_A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < N; ++i) h_B[i * N + i] = 1.0f;

    std::vector<ncclComm_t> comms(numDevices);
    std::vector<int> devList(numDevices);
    for (int i = 0; i < numDevices; ++i) devList[i] = i;
    CHECK_NCCL(ncclCommInitAll(comms.data(), numDevices, devList.data()));

    multi_gpu_sgemm(N, N, N, numDevices, 16, h_A, h_B, h_C, comms.data());

    for (int i = 0; i < numDevices; ++i) CHECK_NCCL(ncclCommDestroy(comms[i]));

    for (int i = 0; i < count; ++i) {
        float diff = h_C[i] - h_A[i];
        if (diff < -1e-2f || diff > 1e-2f) {
            fprintf(stderr, "selftest FAILED at %d: got %f expected %f\n", i, h_C[i], h_A[i]);
            return 1;
        }
    }
    printf("selftest passed\n");
    return 0;
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    int threadsPerBlock = 16;
    bool json_output = false;
    bool run_selftest = false;
    int warmup = 3, repeats = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--selftest") run_selftest = true;
        else if (arg == "--M" && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "--threadsPerBlock" && i + 1 < argc) threadsPerBlock = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (arg == "--repeats" && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (arg == "--json") json_output = true;
    }

    int numDevices = 0;
    CHECK_CUDA(cudaGetDeviceCount(&numDevices));
    if (numDevices < 2) {
        fprintf(stderr,
                "multi_gpu_matmul requires >=2 CUDA devices, found %d. "
                "This task demonstrates multi-GPU profiling and is meant to "
                "run on a multi-GPU box (e.g. a 2x+ GPU RunPod instance).\n",
                numDevices);
        return 1;
    }

    if (run_selftest) return selftest(numDevices);

    std::vector<ncclComm_t> comms(numDevices);
    std::vector<int> devList(numDevices);
    for (int i = 0; i < numDevices; ++i) devList[i] = i;
    CHECK_NCCL(ncclCommInitAll(comms.data(), numDevices, devList.data()));

    std::vector<float> h_A(static_cast<size_t>(M) * K), h_B(static_cast<size_t>(K) * N), h_C(static_cast<size_t>(M) * N, 0.0f);
    std::srand(42);
    for (auto& v : h_A) v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
    for (auto& v : h_B) v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        multi_gpu_sgemm(M, N, K, numDevices, threadsPerBlock, h_A, h_B, h_C, comms.data());
    }

    // Benchmark
    std::vector<double> times_ms;
    for (int r = 0; r < repeats; ++r) {
        // steady_clock, not high_resolution_clock: see matmul/cuda/sgemm.cu.
        auto t0 = std::chrono::steady_clock::now();
        clobber_memory();
        multi_gpu_sgemm(M, N, K, numDevices, threadsPerBlock, h_A, h_B, h_C, comms.data());
        do_not_optimize(h_C.data());
        clobber_memory();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    for (int i = 0; i < numDevices; ++i) CHECK_NCCL(ncclCommDestroy(comms[i]));

    std::vector<double> sorted_times = times_ms;
    std::sort(sorted_times.begin(), sorted_times.end());
    double p50 = true_median(sorted_times);
    double p95 = ceil_percentile(sorted_times, 0.95);
    double tflops_med = tflops(M, N, K, p50 / 1000.0);

    std::vector<double> tflops_list;
    tflops_list.reserve(times_ms.size());
    for (double ms : times_ms) tflops_list.push_back(tflops(M, N, K, ms / 1000.0));

    if (json_output) {
        printf("{\n");
        printf("  \"meta\": {\"M\": %d, \"N\": %d, \"K\": %d, \"threadsPerBlock\": %d, \"num_gpus\": %d},\n",
               M, N, K, threadsPerBlock, numDevices);
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
        printf("  \"num_gpus\": %d,\n", numDevices);
        printf("  \"ok\": true\n");
        printf("}\n");
    } else {
        printf("M=%d N=%d K=%d threadsPerBlock=%d num_gpus=%d\n", M, N, K, threadsPerBlock, numDevices);
        printf("p50=%.4f ms  p95=%.4f ms\n", p50, p95);
        printf("tflops_median=%.6f\n", tflops_med);
    }

    return 0;
}
