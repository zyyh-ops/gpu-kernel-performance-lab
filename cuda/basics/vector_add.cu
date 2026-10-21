#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


float benchmark(
    const float* d_a,
    const float* d_b,
    float* d_c,
    int N,
    int block_size
) {
    int grid_size = (N + block_size - 1) / block_size;

    constexpr int warmup = 10;
    constexpr int repeat = 1000;

    for(int i = 0; i < warmup; i++) {
        vector_add<<<grid_size, block_size>>>(
            d_a,
            d_b,
            d_c,
            N
        );
    }

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++) {
        vector_add<<<grid_size, block_size>>>(
            d_a,
            d_b,
            d_c,
            N
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;

    cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_ms / repeat;
}


int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float* d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(
        d_a,
        h_a.data(),
        bytes,
        cudaMemcpyHostToDevice
    );
    
    cudaMemcpy(
        d_b,
        h_b.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;

    // vector_add<<<grid_size, block_size>>>(
    //     d_a,
    //     d_b,
    //     d_c,
    //     N
    // );

    // cudaDeviceSynchronize();

    // cudaMemcpy(
    //     h_c.data(),
    //     d_c,
    //     bytes,
    //     cudaMemcpyDeviceToHost
    // );

    // bool ok = true;

    // for (int i = 0; i < N; i++) {
    //     float expected = h_a[i] + h_b[i];

    //     if (std::fabs(h_c[i] - expected) > 1e-6f) {
    //         std::cout << "Mismatch at " << i << ": "
    //         << " vs " << expected << '\n';

    //         ok = false;
    //         break;
    //     }
    // }

    // std::cout << (ok ? "Correctness: PASS\n" : "Correctness: FAIL\n");

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int warmup = 10;
    const int repeat = 1000;

    for(int i = 0; i < warmup; i++) {
        vector_add<<<grid_size, block_size>>>(
            d_a, d_b, d_c, N
        );
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++) {
        vector_add<<<grid_size, block_size>>>(
            d_a, d_b, d_c, N
        );
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    float avg_ms = milliseconds / repeat;

    std::cout << "Average kernel time: " << avg_ms << "ms\n";

    int block_sizes[] = {
        32, 64, 128, 256, 512, 1024
    };

    for(int block_size : block_sizes) {
        int grid_size = (N + block_size - 1) / block_size;

        float ms = benchmark(
            d_a, d_b, d_c, N, block_size
        );

        double bytes = static_cast<double>(N) * 3 * sizeof(float);

        double bandwidth = bytes / (ms * 1e-3) / 1e9;

        std::cout << "block=" << block_size << ", grid=" << grid_size
            << ", warps/block=" << block_size / 32 << ", time=" << ms << "ms"
            << ", effective BW=" << bandwidth << "GB/s\n";
    }


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}