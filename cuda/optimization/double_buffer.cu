#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>

__device__ __forceinline__ void cp_async_16(
    float* smem_ptr,
    const float* gmem_ptr
) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        :"r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile(
        "cp.async.commit_group;\n"
    );
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile(
        "cp.async.wait_group 0;\n"
    );
}

template<int TILE>
__global__ void matmul_double_buffer(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    extern __shared__ float As[2][TILE][TILE], Bs[2][TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * TILE + tx;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    int num_tiles = (N + TILE - 1) / TILE;
    int cur = 0;

    // 每个线程负责 4 个连续 float，共 64 个线程完成 256 个元素
    int elem = tid * 4;
    int tile_row = elem / TILE;
    int tile_col = elem % TILE;

    if (tid < 64) {
        cp_async_16(&As[cur][tile_row][tile_col],
                    &A[(blockIdx.y*TILE + tile_row) * N + 0*TILE + tile_col]);
        cp_async_16(&Bs[cur][tile_row][tile_col],
                    &B[(0*TILE + tile_row) * N + blockIdx.x*TILE + tile_col]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    float acc = 0.0f;
    for (int t = 0; t < num_tiles; t++) {
        int next = cur ^ 1;

        if (tid < 64 && t + 1 < num_tiles) {
            cp_async_16(&As[next][tile_row][tile_col],
                        &A[(blockIdx.y*TILE + tile_row) * N + (t+1)*TILE + tile_col]);
            cp_async_16(&Bs[next][tile_row][tile_col],
                        &B[((t+1)*TILE + tile_row) * N + blockIdx.x*TILE + tile_col]);
        }
        if (t + 1 < num_tiles) cp_async_commit();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[cur][ty][k] * Bs[cur][k][tx];
        }

        if (t + 1 < num_tiles) {
            cp_async_wait_all();
            __syncthreads();
        }

        cur = next;
    }

    if (row < N && col < N) C[row * N + col] = acc;
}

template<int TILE>
__global__ void matmul(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int num_tiles = (N + TILE - 1) / TILE;

    __shared__ float As[TILE][TILE], Bs[TILE][TILE];

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for(int t = 0; t < num_tiles; t++){
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        As[ty][tx] = A[row * N + a_col];
        Bs[ty][tx] = B[b_row * N + col];

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < TILE; k++){
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[row * N + col] = acc;
}

using MATMUL = void(*)(const float*, const float*, float*, int);

MATMUL kernels[] = {matmul_double_buffer<16>, matmul<16>};
std::string kernel_names[] = {"double buffer async GEMM", "single buffer tiled GEMM"};

void correctness(MATMUL kernel,
    const float* A, const float* B, float* C, int N, std::string kernel_name,
    std::vector<float>& h_C, std::vector<float>& t_C, size_t bytes) {
    
    int num_blocks = (N + 15) / 16;
    dim3 block = dim3(16, 16), grid = dim3(num_blocks, num_blocks);

    kernel<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(t_C.data(), C, bytes, cudaMemcpyDeviceToHost);
    
    bool ok = true;
    for(int i = 0; i < N * N; i++){
        if(fabs(h_C[i] - t_C[i]) > 1e-4f) ok = false;
    }

    std::cout << kernel_name << " Correctness: " << (ok ? "PASS" : "FAIL") << '\n';
}

double benchmark(MATMUL kernel, const float* A, const float* B, float* C, int N,
    std::string kernel_name) {
    int num_blocks = (N + 15) / 16;
    dim3 block(16, 16), grid(num_blocks, num_blocks);

    int warmup = 5, repeat = 20;
    for(int i = 0; i < warmup; i++) kernel<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < repeat; i++) kernel<<<grid, block>>>(A, B, C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float miliseconds = 0.0f;
    cudaEventElapsedTime(&miliseconds, start, stop);
    double gflops = 2.0 * N * N * N / (miliseconds / 1e3 / repeat) / 1e9;

    std::cout << kernel_name << " time: " << miliseconds / repeat << "ms " << gflops << "GFLOP/s\n";

    return gflops;
}

void bench_with_smems(const float* A, const float* B, float* C, int N,
    size_t smem_bytes, cudaDeviceProp& prop) {
    int num_blocks = (N + 15) / 16;
    dim3 block(16, 16), grid(num_blocks, num_blocks);

    smem_bytes *= 1024;
    double single_gflops, double_gflops;
    double single_occupancy, double_occupancy;

    for(int k = 0; k < 2; k++){
        auto kernel = kernels[k];

        int warmup = 5, repeat = 20;
        for(int i = 0; i < warmup; i++) kernel<<<grid, block, smem_bytes>>>(A, B, C, N);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for(int i = 0; i < repeat; i++) kernel<<<grid, block, smem_bytes>>>(A, B, C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float miliseconds = 0.0f;
        cudaEventElapsedTime(&miliseconds, start, stop);
        double gflops = 2.0 * N * N * N / (miliseconds / 1e3 / repeat) / 1e9;

        int active_blocks = 0;
        
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, 
            kernel, 256, smem_bytes);
        
        int activeWarps = active_blocks * 256 / prop.warpSize;
        int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        double occupancy = 100.0 * activeWarps / maxWarpsPerSM;
        
        if(k == 0) {
            double_gflops = gflops;
            double_occupancy = occupancy;
        } else { 
            single_gflops = gflops;
            single_occupancy = occupancy;
        }
    }
    double speedup = double_gflops / single_gflops;
    std::cout << smem_bytes / 1024 << "KB " << single_gflops << ' ' << single_occupancy << "% "
        << double_gflops << ' ' << double_occupancy << "% " << speedup << "x\n";
}

int main(){
    float *A, *B, *C;
    int N = 256;
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N, 0), t_C(N * N);
    size_t bytes = N * N * sizeof(float);

    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);

    for(int i = 0; i < N * N; i++){
        h_A[i] = (i % 17 - 8) / 8.0f;
        h_B[i] = (i % 13 - 6) / 6.0f;
    } 
    cudaMemcpy(A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                h_C[i * N + j] += h_A[i * N + k] * h_B[k * N + j];
            }
        }
    }

    for(int i = 0; i < 2; i++){
        correctness(kernels[i], A, B, C, N, kernel_names[i], h_C, t_C, bytes);
    }
    cudaFree(A); cudaFree(B); cudaFree(C);

    N = 4096;
    bytes = N * N * sizeof(float);
    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);

    h_A.resize(N * N), h_B.resize(N * N);
    for(int i = 0; i < N * N; i++){
        h_A[i] = (i % 17 - 8) / 8.0f;
        h_B[i] = (i % 13 - 6) / 6.0f;     
    }

    cudaMemcpy(A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    double double_buffer_gflops, single_buffer_gflops;
    for(int i = 0; i < 2; i++){
        double gflops = benchmark(kernels[i], A, B, C, N, kernel_names[i]);
        
        if(i == 0) double_buffer_gflops = gflops;
        else single_buffer_gflops = gflops;
    }

    double speedup = double_buffer_gflops / single_buffer_gflops;
    std::cout << "speedup: " << speedup << "x\n";

    std::cout << "shared memory | Single GFLOP/s | Single Occupancy \
     | Double GFLOP/s | Double Occupancy | Double/Single\n";

    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    size_t smem_bytes[] = {0, 16, 24, 32, 44};
    for(int k = 0; k < 5; k++){
        bench_with_smems(A, B, C, N, smem_bytes[k], prop);
    }
    return 0;
}