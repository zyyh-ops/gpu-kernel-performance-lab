#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <vector>

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_naive(
    const float* input,
    float* output,
    int n
) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;

    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if(x < n && y + j < n) {
            output[x * n + y + j] = input[(y + j) * n + x];
        }
    }
}

__global__ void transpose_shared(
    const float* input,
    float* output,
    int n
) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;

    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        if(x < n && y + j < n){
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * n + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;

    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        if(x < n && y + j < n){
            output[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void transpose_shared_padded(
    const float* input,
    float* output,
    int n
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;

    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        if(x < n && y + j < n){
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * n + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;

    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        if(x < n && y + j < n){
            output[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void copy_kernel(
    const float* input,
    float* output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n){
        output[i] = input[i];
    }
}

int main(){
    int n = 8192;

    dim3 block(TILE_DIM, BLOCK_ROWS);

    dim3 grid(
        (n + TILE_DIM - 1) / TILE_DIM,
        (n + TILE_DIM - 1) / TILE_DIM
    );

    std::vector<float> h_input(n * n), h_output(n * n);

    const size_t bytes = n * n * sizeof(float);

    float *input ,*output;

    cudaMalloc(&input, bytes);
    cudaMalloc(&output, bytes);

    for(int i = 0; i < n * n; i++) h_input[i] = static_cast<float>(i);

    cudaMemcpy(
        input, h_input.data(), bytes, cudaMemcpyHostToDevice
    );

    transpose_naive<<<grid, block>>>(input, output, n);

    cudaMemcpy(
        h_output.data(), output, bytes, cudaMemcpyDeviceToHost
    );

    bool f = true;
    for(int r = 0; r < n; r++) {
        for(int c = 0; c < n; c++){
            float expected = h_input[r * n + c];
            float actual = h_output[c * n + r];
            if(fabs(expected - actual) > 1e-6f){
                f = false; break;
            }
        }
    }

    std::cout << "Naive correctness: " << (f ? "PASS" : "FAIL") << '\n';

    transpose_shared<<<grid, block>>>(input, output, n);

    cudaMemcpy(
        h_output.data(), output, bytes, cudaMemcpyDeviceToHost
    );

    f = true;
    for(int r = 0; r < n; r++) {
        for(int c = 0; c < n; c++){
            float expected = h_input[r * n + c];
            float actual = h_output[c * n + r];
            if(fabs(expected - actual) > 1e-6f){
                f = false; break;
            }
        }
    }

    std::cout << "Shared correctness: " << (f ? "PASS" : "FAIL") << '\n';

    transpose_shared_padded<<<grid, block>>>(input, output, n);

    cudaMemcpy(
        h_output.data(), output, bytes, cudaMemcpyDeviceToHost
    );

    f = true;
    for(int r = 0; r < n; r++) {
        for(int c = 0; c < n; c++){
            float expected = h_input[r * n + c];
            float actual = h_output[c * n + r];
            if(fabs(expected - actual) > 1e-6f){
                f = false; break;
            }
        }
    }

    std::cout << "Shared padded correctness: " << (f ? "PASS" : "FAIL") << '\n';


    int warmup = 5, repeat = 50;

    for(int i = 0; i < warmup; i++){
        transpose_naive<<<grid, block>>>(input, output, n);
    }

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++){
        transpose_naive<<<grid, block>>>(input, output, n);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naive_time = 0.0f;

    cudaEventElapsedTime(&naive_time, start, stop);


    for(int i = 0; i < warmup; i++) {
        transpose_shared<<<grid, block>>>(input, output, n);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++) {
        transpose_shared<<<grid, block>>>(input, output, n);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float shared_time = 0.0f;

    cudaEventElapsedTime(&shared_time, start, stop);


    for(int i = 0; i < warmup; i++) {
        transpose_shared_padded<<<grid, block>>>(input, output, n);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++) {
        transpose_shared_padded<<<grid, block>>>(input, output, n);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float shared_padded_time = 0.0f;

    cudaEventElapsedTime(&shared_padded_time, start, stop);

    int N = n * n;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    for(int i = 0; i < warmup; i++) {
        copy_kernel<<<blocks, threads>>>(input, output, N);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++) {
        copy_kernel<<<blocks, threads>>>(input, output, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float copy_kernel_time = 0.0f;

    cudaEventElapsedTime(&copy_kernel_time, start, stop);

    float naive_BW = 2 * n * n * 4 / (naive_time / repeat / 1e3) / 1e9;
    float shared_BW = 2 * n * n * 4 / (shared_time / repeat / 1e3) / 1e9;
    float shared_padded_BW = 2 * n * n * 4 / (shared_padded_time / repeat / 1e3) / 1e9;
    float copy_kernel_BW = 2 * n * n * 4 / (copy_kernel_time / repeat / 1e3) / 1e9;

    std::cout << "naive_BW: " << naive_BW << "GB/s\n";
    std::cout << "shared_BW: " << shared_BW << "GB/s\n";
    std::cout << "shared_padded_BW: " << shared_padded_BW << "GB/s\n";
    std::cout << "Baseline BW: " << copy_kernel_BW << "GB/s\n";
    std::cout << "Shared / Navie speedup: " << shared_BW / naive_BW << "x\n";
    std::cout << "Shared padded / Navie speedup: " << shared_padded_BW / naive_BW << "x\n";
    std::cout << "Shared padded / shared speedup: " << shared_padded_BW / shared_BW << "x\n";

    return 0;
}