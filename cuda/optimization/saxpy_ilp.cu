#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>

__global__ void saxpy_like_scalar(
    const float* input,
    float* output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n) {
        float x = input[i];

        output[i] = 2.0f * x + 1.0f;
    }
}

template<int NUM>
__global__ void saxpy_like_ilp(
    const float* input,
    float* output,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = gridDim.x * blockDim.x;

    #pragma unroll
    for(int j = 0; j < NUM; j++){
        int idx = tid + j * stride;

        if(idx < n) {
            output[idx] = 2.0f * input[idx] + 1.0f;
        }
    }
}

using SAXPY_ILP = void(*)(const float*, float*, int);

SAXPY_ILP kernels[] = {saxpy_like_scalar, saxpy_like_ilp<1>, saxpy_like_ilp<2>, saxpy_like_ilp<4>, saxpy_like_ilp<8>};
std::string kernel_names[] = {"scalar", "ilp1", "ilp2", "ilp4", "ilp8"};

void benchmark(SAXPY_ILP kernel, int grid, int block, int N, 
    const float* input, float* output, std::string kernel_name) {
    kernel<<<grid, block>>>(input, output, N);
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

    float miliseconds = 0.0f;

    cudaEventElapsedTime(&miliseconds, start, stop);

    float BW = 8 * N / (miliseconds / 1e3 / repeat) / 1e9;

    std::cout << kernel_name << ' ' << miliseconds << "ms " << BW << "GB/s\n";
}

int main(){
    int block = 256, N = 1 << 26;
    const size_t bytes = N * sizeof(float);

    float *input, *output;
    cudaMalloc(&input, bytes);
    cudaMalloc(&output, bytes);

    std::vector<float> h_input(N), h_output(N);
    for(int i = 0; i < N; i++) h_input[i] = static_cast<float>(i);

    cudaMemcpy(input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    std::cout << "ILP | time | effective BW\n";

    for(int k = 0; k < 5; k++){
        auto kernel = kernels[k];
        std::string kernel_name = kernel_names[k];

        cudaMemcpy(h_output.data(), output, bytes, cudaMemcpyDeviceToHost);

        bool ok = true;
        for(int i = 0; i < N; i++){
            float expected = 2.0f * h_input[i] + 1.0f;
            
            if(fabs(expected - h_output[i]) > 1e-6f) {
                ok = false; break;
            }
        }

        if(ok) std::cout << kernel_name << " Correctness: PASS\n";
        else std::cout << kernel_name << " Correctness: FAIL\n";

        int NUM;
        if(k == 0 || k == 1) NUM = 1;
        else NUM = (1 << (k - 1));
        
        int grid = (N + block * NUM - 1) / (block * NUM);

        benchmark(kernel, grid, block, N, input, output, kernel_name);
    }

    return 0;
}