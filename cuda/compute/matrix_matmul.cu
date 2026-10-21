#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <vector>

__global__ void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= N || col >= N) return;

    float acc = 0.0f;

    for(int k = 0; k < N; k++) {
        acc += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = acc;
}


template<int TILE>
__global__ void matmul_shared(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int num_tiles = (N + TILE - 1) / TILE;

    float acc = 0.0f;

    for(int t = 0; t < num_tiles; t++){
        int a_col = t * TILE + tx;

        int b_row = t * TILE + ty;

        if (row < N && a_col < N){
            As[ty][tx] = A[row * N + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (b_row < N && col < N){
            Bs[ty][tx] = B[b_row * N + col];
        } else{
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for(int k = 0; k < TILE; k++){
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if(row < N && col < N){
        C[row * N + col] = acc;
    }
}

bool is_correct(int N, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C) {
    std::vector<float> ref(N * N);

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double sum = 0.0f;

            for(int k = 0; k < N; k++) {
                sum += static_cast<double>(A[i * N + k])
                    * static_cast<double>(B[k * N + j]);
            }

            ref[i * N + j] = static_cast<float>(sum);
        }
    }

    float max_abs = 0.0f;
    for(int i = 0; i < N * N; i++){
        max_abs = max(max_abs, fabs(ref[i] - C[i]));
    }

    return max_abs < 1e-2f;
}

using MatmulKernel = void(*)(const float*, const float*, float*, int);

void launch_matmul(MatmulKernel kernel, const float* A, const float* B,
                   float* C, int N, dim3 grid, dim3 block) {
    kernel<<<grid, block>>>(A, B, C, N);
}

int main() {
    int N = 256, TILE = 16;

    dim3 block(TILE, TILE);

    dim3 grid(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    float *A, *B, *C;

    size_t bytes = N * N * sizeof(float);

    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (i % 17 - 8) / 8.0f;
        h_B[i] = (i % 13 - 6) / 6.0f;
    }

    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);

    cudaMemcpy(A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    matmul_naive<<<grid, block>>>(A, B, C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), C, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Naive correctness: " << (is_correct(N, h_A, h_B, h_C) ? "PASS" : "FAIL") << '\n';

    matmul_shared<16><<<grid, block>>>(A, B, C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), C, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Shared correctness: " << (is_correct(N, h_A, h_B, h_C) ? "PASS" : "FAIL") << '\n';

    cudaFree(A); cudaFree(B); cudaFree(C);


    N = 4096;

    bytes = N * N * sizeof(float);

    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);

    std::vector<float> h1_A(N * N), h1_B(N * N);

    for (int i = 0; i < N * N; i++) {
        h1_A[i] = (i % 17 - 8) / 8.0f;
        h1_B[i] = (i % 13 - 6) / 6.0f;
    }

    cudaMemcpy(A, h1_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h1_B.data(), bytes, cudaMemcpyHostToDevice);

    // MatmulKernel kernels[] = {matmul_naive, matmul_shared<16>};
    MatmulKernel shared_kernels[] = {matmul_shared<8>, matmul_shared<16>, matmul_shared<32>};

    int warmup = 5, repeat = 50;

    cudaEvent_t start, stop;

    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    int tiles[] = {8, 16, 32};

    for (int k = 0; k < 3; k++) {
        int tile = tiles[k];
        auto kernel = shared_kernels[k];

        block = dim3(tile, tile);
        grid = dim3(
            (N + tile - 1) / tile,
            (N + tile - 1) / tile
        );

        for(int i = 0; i < warmup; i++) {
            launch_matmul(kernel, A, B, C, N, grid, block);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);

        for(int i = 0; i < repeat; i++) {
            launch_matmul(kernel, A, B, C, N, grid, block);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float seconds = 0.0f;
        cudaEventElapsedTime(&seconds, start, stop);

        double gflops = 2.0 * N * N * N / (seconds / 1000.0 / repeat) / 1e9;

        
        std::cout << "TILE = " << tile << ": Shared GFLOPS = " << gflops << "GFLOP/s\n";
    }

    cudaFree(A); cudaFree(B); cudaFree(C);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout
        << "maxThreadsPerBlock: "
        << prop.maxThreadsPerBlock
        << '\n';

    std::cout
        << "maxThreadsPerSM: "
        << prop.maxThreadsPerMultiProcessor
        << '\n';

    std::cout
        << "sharedMemPerSM: "
        << prop.sharedMemPerMultiprocessor / 1024.0
        << " KiB\n";

    std::cout
        << "regsPerBlock: "
        << prop.regsPerBlock
        << '\n';

    int maxBlocksPerSM;

    cudaDeviceGetAttribute(
        &maxBlocksPerSM,
        cudaDevAttrMaxBlocksPerMultiprocessor,
        0
    );

    std::cout << "maxBlocksPerSM: " << maxBlocksPerSM << '\n';

    return 0;
}