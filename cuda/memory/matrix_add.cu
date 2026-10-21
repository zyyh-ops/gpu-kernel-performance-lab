#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

__global__ void matrix_add(
    const float* a,
    const float* b,
    float* c,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < rows && col < cols) {
        int idx = row * cols + col;

        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matrix_add_bad(
    const float* a,
    const float* b,
    float* c,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < rows && col < cols) {
        int idx = row * cols + col;

        c[idx] = a[idx] + b[idx];
    }
}

float benchmark(
    const float* d_a,
    const float* d_b,
    float* d_c,
    int rows,
    int cols,
    dim3 block,
    bool is_good
) {
    dim3 grid;

    if(is_good) {
        grid = dim3(
            (cols + block.x - 1) / block.x,
            (rows + block.y - 1) / block.y
        );
    } else{
        grid = dim3(
            (rows + block.x - 1) / block.x,
            (cols + block.y - 1) / block.y
        );
    }

    int warmup = 10, repeat = 100;

    cudaEvent_t start, stop;

    if(is_good) {
        for(int i = 0; i < warmup; i++){
            matrix_add<<<grid, block>>>(d_a, d_b, d_c, rows, cols);
        }
    } else {
        for(int i = 0; i < warmup; i++){
            matrix_add_bad<<<grid, block>>>(d_a, d_b, d_c, rows, cols);
        }
    }

    cudaDeviceSynchronize();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    if(is_good) {
        for(int i = 0; i < repeat; i++){
            matrix_add<<<grid, block>>>(d_a, d_b, d_c, rows, cols);
        }
    } else {
        for(int i = 0; i < repeat; i++){
            matrix_add_bad<<<grid, block>>>(d_a, d_b, d_c, rows, cols);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;

    cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    );

    total_ms /= repeat;
    const size_t bytes = rows * cols * 12;

    float BW = bytes / (total_ms * 1e-3) / 1e9;

    if(is_good) {
        std::cout << "Good time: " << total_ms << "ms" << '\n';
        std::cout << "Good effective BW: " << BW << "G/s" << '\n';
    } else {
        std::cout << "Bad time: " << total_ms << "ms" << '\n';
        std::cout << "Bad effective BW: " << BW << "G/s" << '\n';
    }

    return BW;
}

int main() { 
    dim3 block(32, 8);

    int rows, cols;
    rows = cols = 8192;

    const size_t bytes = rows * cols * sizeof(float);
    int N = rows * cols;

    std::cout << "the size of matrix: " << rows << " × " << cols << '\n';
    std::cout << "block: " << block.x << " × " << block.y << '\n';

    std::vector<float> h_a(N), h_b(N), h_c(N);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    for(int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

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

    dim3 grid_good(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    dim3 grid_bad(
        (rows + block.x - 1) / block.x,
        (cols + block.y - 1) / block.y
    );

    matrix_add<<<grid_good, block>>>(d_a, d_b, d_c, rows, cols);

    cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    );

    bool f = true;

    for(int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];

        if(fabs(expected - h_c[i]) > 1e-6) {
            f = false; 
            std::cout << "Mismatch at " << i << ": " << h_c[i] << " vs " << expected << '\n';
            break;
        }
    }

    std::cout << "Good correctness: " << (f ? "PASS" : "FAIL") << '\n';

    matrix_add<<<grid_bad, block>>>(d_a, d_b, d_c, rows, cols);

    cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    );

    f = true;

    for(int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];

        if(fabs(expected - h_c[i]) > 1e-6) {
            f = false; 
            std::cout << "Mismatch at " << i << ": " << h_c[i] << " vs " << expected << '\n';
            break;
        }
    }

    std::cout << "Bad correctness: " << (f ? "PASS" : "FAIL") << "\n\n";

    float Good_BW = benchmark(d_a, d_b, d_c, rows, cols, block, true);
    std::cout << "\n";
    float Bad_BW = benchmark(d_a, d_b, d_c, rows, cols, block, false);
    std::cout << "\n";

    std::cout << "slowdown: " << Bad_BW / Good_BW << '\n';

    return 0;
}