#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>
using ll = long long;


__global__ void stride_read(
    const float* input,
    float* output,
    int n,
    int stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] = input[(ll)i * stride];
    }
}


int main(){
    int N = 1 << 20;

    int strides[] = {1, 2, 4, 8, 16, 32};

    int block_size = 256;

    dim3 grid((N + block_size - 1) / block_size);

    for(int i = 0; i < 6; i++){
        int stride = strides[i];

        float *input, *output;

        const size_t bytes = N * sizeof(float) * stride;

        std::vector<float> h_input(N * stride);

        for(int i = 0; i < N * stride; i += stride) {
            h_input[i] = static_cast<float>(i);
        }

        cudaMalloc(&input, bytes);
        cudaMalloc(&output, bytes);

        cudaMemcpy(
            input, h_input.data(), bytes, cudaMemcpyHostToDevice
        );

        int warmup = 10, repeat = 100;

        for(int i = 0; i < warmup; i++) {
            stride_read<<<grid, block_size>>>(
                input, output, N, stride
            );
        }

        cudaDeviceSynchronize();

        cudaEvent_t start, stop;

        cudaEventCreate(&stop);
        cudaEventCreate(&start);

        cudaEventRecord(start);

        for(int i = 0; i < repeat; i++) {
            stride_read<<<grid, block_size>>>(
                input, output, N, stride
            );
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

        float BW = 8 * N / (total_ms * 1e-3) / 1e9;

        std::cout << "stride = " << stride << ": " << BW << " GB/s\n";
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout
        << "GPU: "
        << prop.name
        << '\n';

    std::cout
        << "L2 cache: "
        << prop.l2CacheSize / 1024.0 / 1024.0
        << " MiB\n";

    std::cout
        << "SM count: "
        << prop.multiProcessorCount
        << '\n';

    return 0;
}