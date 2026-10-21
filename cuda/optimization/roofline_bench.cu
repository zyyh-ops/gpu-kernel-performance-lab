#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <vector>

template<int ITER>
__global__ void roofline_bench(
    const float* input,
    float* output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    float x = input[i];

    float a0 = x + 0.01f;
    float a1 = x + 0.02f;
    float a2 = x + 0.03f;
    float a3 = x + 0.04f;
    float a4 = x + 0.05f;
    float a5 = x + 0.06f;
    float a6 = x + 0.07f;
    float a7 = x + 0.08f;

    #pragma unroll 1
    for (int k = 0; k < ITER; ++k) {
        a0 = a0 * 1.000001f + 0.000001f;
        a1 = a1 * 1.000002f + 0.000002f;
        a2 = a2 * 1.000003f + 0.000003f;
        a3 = a3 * 1.000004f + 0.000004f;
        a4 = a4 * 1.000005f + 0.000005f;
        a5 = a5 * 1.000006f + 0.000006f;
        a6 = a6 * 1.000007f + 0.000007f;
        a7 = a7 * 1.000008f + 0.000008f;
    }

    output[i] =
        a0 + a1 + a2 + a3
        + a4 + a5 + a6 + a7;
}

using ROOF_BENCH = void(*)(const float*, float*, int);

ROOF_BENCH kernels[] = {roofline_bench<1>, roofline_bench<2>,
    roofline_bench<4>, roofline_bench<8>, roofline_bench<16>,
    roofline_bench<32>, roofline_bench<64>};

void benchmark(ROOF_BENCH kernel, int grid, int block, int N,
    const float* input, float* output, int k) {
    int warmup = 5, repeat = 50;
    
    for(int i = 0; i < warmup; i++){
        kernel<<<grid, block>>>(input, output, N);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < repeat; i++) {
        kernel<<<grid, block>>>(input, output, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float miliseconds = 0.0f;
    cudaEventElapsedTime(&miliseconds, start, stop);

    int iter = (1 << k);
    double tflops = 16 * iter * static_cast<double>(N) / (miliseconds / repeat / 1e3) / 1e9;

    std::cout << iter << ' ' << 2 * iter << ' ' << tflops << "GFLOP/s\n";
}

int main(){
    float *input, *output;
    int N = 1 << 26;
    const size_t bytes = N * sizeof(float);
    int block = 256;
    int grid = (N + block - 1) / block;

    cudaMalloc(&input, bytes);
    cudaMalloc(&output, bytes);

    std::vector<float> h_input(N);

    for(int i = 0; i < N; i++) h_input[i] = static_cast<float>(i);

    cudaMemcpy(input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    std::cout << "ITER | AI | GFLOP/s\n";

    for(int k = 0; k < 7; k++){
        benchmark(kernels[k], grid, block, N, input, output, k);
    }

    return 0;
}