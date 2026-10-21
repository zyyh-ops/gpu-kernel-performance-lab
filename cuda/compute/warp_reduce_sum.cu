#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

__device__ __forceinline__ float warp_reduce_sum(float value) {
    constexpr unsigned FULL_MASK = 0xffffffffu;

    for(int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(
            FULL_MASK,
            value,
            offset
        );
    }

    return value;
}

__global__ void reduce_shuffle(
    const float* input,
    float* output,
    int n
) {
    __shared__ float warp_nums[32];

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    float sum = 0.0f;
    
    int i = blockIdx.x * blockDim.x * 2 + tid;
    if(i < n) sum += input[i];

    if(i + blockDim.x < n) sum += input[i + blockDim.x];

    sum = warp_reduce_sum(sum);

    if(lane == 0) warp_nums[warp_id] = sum;

    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;

    if(warp_id == 0){
        float block_num = (
            lane < num_warps ? warp_nums[lane] : 0.0f
        );

        block_num = warp_reduce_sum(block_num);

        if(lane == 0) {
            output[blockIdx.x] = block_num;
        }
    }
}

__global__ void reduce_shared(
    const float* input,
    float* output,
    int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    int i = blockIdx.x * blockDim.x * 2 + tid;

    float sum = 0.0f;

    if(i < n) sum += input[i];

    if(i + blockDim.x < n) sum += input[i + blockDim.x];

    sdata[tid] = sum;

    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    if(tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

using ReduceKernel = void(*)(const float*, float*, int);

void launch_kernel(ReduceKernel kernel, const float* input, 
        float* output, int N, int grid, int block, size_t shared_mem) {
    kernel<<<grid, block, shared_mem>>>(input, output, N);
}

float benchmark(ReduceKernel kernel, const float* input,
        float* output, int N, int grid, int block, size_t shared_mem) {
    int warmup = 10, repeat = 100;

    cudaEvent_t start, stop;

    cudaEventCreate(&stop);
    cudaEventCreate(&start);

    for(int i = 0; i < warmup; i++){
        launch_kernel(kernel, input, output, N, grid, block, shared_mem);
    }
    cudaDeviceSynchronize();

    float seconds = 0.0f;
    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++){
        launch_kernel(kernel, input, output, N, grid, block, shared_mem);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&seconds, start, stop);

    seconds /= repeat;

    return seconds;
}


int main(){
    int N = 1 << 24;

    float *input, *output;

    const size_t bytes = N * sizeof(float);

    cudaMalloc(&input, bytes);
    cudaMalloc(&output, bytes);

    std::vector<float> h_input(N), h_output(N);

    for(int i = 0; i < N; i++) h_input[i] = 1.0f;

    cudaMemcpy(input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    int block = 64, grid = (N + block * 2 - 1) / (block * 2);
    size_t shared_mem = block * sizeof(float);

    reduce_shared<<<grid, block, shared_mem>>>(input, output, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), output, bytes, cudaMemcpyDeviceToHost);

    double shared_total = 0.0f;

    for(int i = 0; i < grid; i++) shared_total += h_output[i];

    reduce_shuffle<<<grid, block>>>(input, output, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), output, bytes, cudaMemcpyDeviceToHost);

    double shuffle_total = 0.0f;

    for(int i = 0; i < grid; i++) shuffle_total += h_output[i];

    int warmup = 10, repeat = 100;

    for(int i = 0; i < warmup; i++) reduce_shared<<<grid, block, shared_mem>>>(input, output, N);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;

    cudaEventCreate(&stop);
    cudaEventCreate(&start);

    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++) {
        reduce_shared<<<grid, block, shared_mem>>>(input, output, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float shared_time = 0.0f;
    cudaEventElapsedTime(
        &shared_time,
        start,
        stop
    );

    shared_time /= repeat;

    for(int i = 0; i < warmup; i++) reduce_shuffle<<<grid, block>>>(input, output, N);

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++) {
        reduce_shuffle<<<grid, block>>>(input, output, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float shuffle_time = 0.0f;
    cudaEventElapsedTime(
        &shuffle_time,
        start,
        stop
    );

    shuffle_time /= repeat;

    float shared_BW = 4 * N / (shared_time / 1e3) / 1e9;
    float shuffle_BW = 4 * N / (shuffle_time / 1e3) / 1e9;

    std::cout << "shared correctness: " << (fabs(shared_total - N) < 1e-6f ? "PASS" : "FAIL") << '\n';
    std::cout << "shuffle correctness: " << (fabs(shuffle_total - N) < 1e-6f ? "PASS" : "FAIL") << '\n';

    std::cout << "shared time: " << shared_time << "ms" << '\n';
    std::cout << "shuffle time: " << shuffle_time << "ms" << '\n';

    std::cout << "shared effective BW: " << shared_BW << "GB/s" << '\n';
    std::cout << "shuffle effective BW: " << shuffle_BW << "GB/s" << '\n';

    std::cout << "shuffle speedup: " << shared_time / shuffle_time << "x\n";

    std::cout << "block | shared time | shuffle time | speedup\n"; 

    int blocks[] = {64, 128, 256, 512, 1024};
    for(int k = 0; k < 5; k++) {
        int block = blocks[k];
        grid = (N + 2 * block - 1) / (2 * block);

        size_t shared_mem = block * sizeof(float);
        float shared_time = benchmark(reduce_shared, input, output, N, grid, block, shared_mem);
        float shuffle_time = benchmark(reduce_shuffle, input, output, N, grid, block, shared_mem);

        float speedup = shared_time / shuffle_time;

        std::cout << block << " " << shared_time << "ms " << shuffle_time << "ms " << speedup << "x\n";
    }

    return 0;
}