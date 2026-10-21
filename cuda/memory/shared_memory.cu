#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <vector>

constexpr int TILE = 32;
constexpr int REPEAT = 1024;

__global__ void shared_conflict(
    const float* input,
    float* output,
    int n
) {
    __shared__ float tile[TILE][TILE];

    int ty = blockIdx.y * TILE + threadIdx.y;
    int tx = blockIdx.x * TILE + threadIdx.x;
    
    if(ty < n && tx < n){
        tile[threadIdx.y][threadIdx.x] = input[ty * n + tx];
    }

    __syncthreads();

    float sum = 0.0f;

    #pragma unroll 1
    for(int r = 0; r < REPEAT; r++){
        int col = (threadIdx.y + r) & 31;
        
        sum += tile[threadIdx.x][col];
    }

    if(tx < n && ty < n) {
        output[ty * n + tx] = sum;
    }
}

__global__ void shared_padded(
    const float* input,
    float* output,
    int n
) {
    __shared__ float tile[TILE][TILE + 1];

    int ty = blockIdx.y * TILE + threadIdx.y;
    int tx = blockIdx.x * TILE + threadIdx.x;

    if(tx < n && ty < n){
        tile[threadIdx.y][threadIdx.x] = input[ty * n + tx];
    }

    __syncthreads();

    float sum = 0.0f;

    #pragma unroll 1
    for(int r = 0; r < REPEAT; r++){
        int col = (threadIdx.y + r) & 31;

        sum += tile[threadIdx.x][col];
    }

    if(tx < n && ty < n) {
        output[ty * n + tx] = sum;
    }
}

int main(){
    float *input, *output;

    int n = 8192;

    int N = n * n;

    dim3 block(TILE, TILE);

    dim3 grid(
        (n + TILE - 1) / TILE,
        (n + TILE - 1) / TILE
    );

    const size_t bytes = N * sizeof(float);

    cudaMalloc(&input, bytes);
    cudaMalloc(&output, bytes);

    std::vector<float> h_input(N), h_output(N);

    for(int i = 0; i < N; i++) h_input[i] = static_cast<float>(i);

    cudaMemcpy(
        input, h_input.data(), bytes, cudaMemcpyHostToDevice
    );

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warmup = 5, repeat = 50;

    for(int i = 0; i < warmup; i++){
        shared_conflict<<<grid, block>>>(
            input, output, n
        );
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++){
        shared_conflict<<<grid, block>>>(
            input, output, n
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float conflict_time = 0.0f;

    cudaEventElapsedTime(&conflict_time, start, stop);

    for(int i = 0; i < warmup; i++){
        shared_padded<<<grid, block>>>(
            input, output, n
        );
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++){
        shared_padded<<<grid, block>>>(
            input, output, n
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float padded_time = 0.0f;

    cudaEventElapsedTime(&padded_time, start, stop);

    std::cout << "conflict time per launch: " 
            << conflict_time / repeat << " ms\n";
    std::cout << "padded time per launch: " 
            << padded_time / repeat << " ms\n";
    std::cout << "speedup: " 
            << conflict_time / padded_time << "x\n";

    return 0;
}