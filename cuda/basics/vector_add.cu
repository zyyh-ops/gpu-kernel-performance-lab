#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err_)         \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

struct BenchResult {
    float single_ms;  // 单次 launch 的端到端时间（中位数），反映真实内核延迟
    float batch_ms;   // 背靠背批量 launch 的平均周期，反映流水线吞吐（含 launch 间隙）
};

BenchResult benchmark(const float* d_a, const float* d_b, float* d_c,
                      size_t N, int block_size) {
    const int grid_size = static_cast<int>((N + block_size - 1) / block_size);

    constexpr int warmup = 10;
    constexpr int repeat = 100;

    for (int i = 0; i < warmup; i++) {
        vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---- 单次延迟：每次 launch 都用 event 包裹并同步，取中位数 ----
    std::vector<float> single_times(repeat);
    for (int i = 0; i < repeat; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&single_times[i], start, stop));
    }
    std::sort(single_times.begin(), single_times.end());
    const float single_ms = single_times[repeat / 2];

    // ---- 批量吞吐：背靠背 launch，总时间 / repeat ----
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float batch_ms = total_ms / repeat;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return {single_ms, batch_ms};
}

int main() {
    // 1<<26 = 67,108,864 个 float，每个数组 256 MB，总访存 768 MB > L2(~96 MB)
    const size_t N = static_cast<size_t>(1) << 26;
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (size_t i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // ---- 正确性校验（必须在任何性能测量之前）----
    {
        const int block = 256;
        const int grid = static_cast<int>((N + block - 1) / block);
        vector_add<<<grid, block>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

        bool ok = true;
        size_t bad_index = 0;
        for (size_t i = 0; i < N; i++) {
            const float expected = h_a[i] + h_b[i];
            // 单次 float 加法在 GPU 与 CPU(SSE) 上均为 IEEE-754 确定性运算，
            // 输入相同时结果应 bit-exact，直接比较即可。
            if (h_c[i] != expected) {
                ok = false;
                bad_index = i;
                break;
            }
        }
        if (!ok) {
            std::cout << "Correctness: FAIL at index " << bad_index
                      << " (got " << h_c[bad_index]
                      << ", expected " << h_a[bad_index] + h_b[bad_index] << ")\n";
            return EXIT_FAILURE;
        }
        std::cout << "Correctness: PASS\n";
    }

    const double bytes_moved = static_cast<double>(N) * 3 * sizeof(float);

    const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int block_size : block_sizes) {
        const int grid_size = static_cast<int>((N + block_size - 1) / block_size);
        const BenchResult r = benchmark(d_a, d_b, d_c, N, block_size);

        const double bw_single = bytes_moved / (r.single_ms * 1e-3) / 1e9;
        const double bw_batch  = bytes_moved / (r.batch_ms  * 1e-3) / 1e9;

        std::cout << "block=" << block_size
                  << ", grid=" << grid_size
                  << ", warps/block=" << block_size / 32
                  << " | single=" << r.single_ms * 1e3 << "us"
                  << ", BW(single)=" << bw_single << "GB/s"
                  << " | batch=" << r.batch_ms * 1e3 << "us"
                  << ", BW(batch)=" << bw_batch << "GB/s\n";
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}