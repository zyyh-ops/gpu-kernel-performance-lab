#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>

__global__ void copy_scalar(
    const float* input,
    float* output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n) {
        output[i] = input[i];
    }
}

__global__ void copy_float2(
    const float* input,
    float* output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int vec_n = n / 2;

    if(i < vec_n) {
        const float2* src = reinterpret_cast<const float2*>(input);
        
        float2* dst = reinterpret_cast<float2*>(output);

        dst[i] = src[i];
    }
}

__global__ void copy_float4(
    const float* input,
    float* output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int vec_n = n / 4;

    if(i < vec_n) {
        const float4* src = reinterpret_cast<const float4*>(input);

        float4* dst = reinterpret_cast<float4*>(output);

        dst[i] = src[i];
    }
}

using COPY_FLOAT = void(*)(const float* , float* , int);

COPY_FLOAT kernels[] = {copy_scalar, copy_float2, copy_float4};

void benchmark(COPY_FLOAT kernel, int grid, int block, 
    const float* input, float* output, int N) {
    int warmup = 10, repeat = 100;
    
    for(int i = 0; i < warmup; i++)
        kernel<<<grid, block>>>(input, output, N);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < repeat; i++)
        kernel<<<grid, block>>>(input, output, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float seconds = 0.0f;

    cudaEventElapsedTime(&seconds, start, stop);

    float BW = 8 * N / (seconds / 1e3 / repeat) / 1e9;

    std::cout << "time: " << seconds << "ms\n";
    std::cout << "BW: " << BW << "GB/s\n";
}

int main(){
    int block = 256, N = 1 << 26;
    int grid = (N + block - 1) / block;
    const size_t bytes = N * sizeof(float);

    float *input, *output;
    cudaMalloc(&input, bytes);
    cudaMalloc(&output, bytes);

    std::vector<float> h_input(N);
    for(int i = 0; i < N; i++) h_input[i] = static_cast<float>(i);
    cudaMemcpy(input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    for(int k = 0; k < 3; k++){
        auto kernel = kernels[k];

        if(k == 0) std::cout << "scalar:\n";
        else if(k == 1) std::cout << "float2:\n";
        else std::cout << "float4:\n";

        benchmark(kernel, grid, block, input, output, N);
    }

    return 0;
}