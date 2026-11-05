#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <random>
#include <string>

#include "../optimization/gemm_validation.h"

#define CUDA_CHECK(call) do { \
    const cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::cerr << #call << ": " << cudaGetErrorString(status) \
                  << " (line " << __LINE__ << ")\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

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

// Reference configuration: BM=BN=64, BK=16, TM=TN=4.
// Launch exactly (BN/TN, BM/TM) = (16,16) threads.
// Each thread owns interleaved rows/columns, not a contiguous 4x4 patch:
//   row = block_row + ty + i*THREADS_Y
//   col = block_col + tx + j*THREADS_X
// This keeps adjacent x lanes on adjacent columns for shared reads and stores.
// ALIGNED_FAST requires complete tiles in all dimensions. VECTOR_LOAD also
// requires 16-byte aligned A/B addresses (enforced by the host dispatcher).
template<int BM, int BN, int BK, int TM, int TN,
         bool ALIGNED_FAST = false, bool VECTOR_LOAD = false>
__global__ void matmul_register_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C, int N
) {
    static_assert(BM > 0 && BN > 0 && BK > 0 && TM > 0 && TN > 0,
                  "Tile dimensions must be positive");
    static_assert(BM % TM == 0 && BN % TN == 0,
                  "Thread tiles must divide the output tile");
    constexpr int THREADS_X = BN / TN;
    constexpr int THREADS_Y = BM / TM;
    constexpr int THREADS = THREADS_X * THREADS_Y;
    static_assert(THREADS <= 1024, "Too many threads per block");
    static_assert(!VECTOR_LOAD || (ALIGNED_FAST && BK % 4 == 0 && BN % 4 == 0),
                  "float4 requires complete tiles and four-float row alignment");

    __shared__ __align__(16) float As[BM][BK];
    __shared__ __align__(16) float Bs[BK][BN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < N; k0 += BK) {
        if constexpr (VECTOR_LOAD) {
            // v is a float4 index, p is a float index. With 256 threads,
            // each thread copies one contiguous float4 from each matrix.
            // Unlike the scalar path, a thread owns p=4*tid+[0,1,2,3].
            for (int v = tid; v < BM * BK / 4; v += THREADS) {
                const int p = 4 * v, r = p / BK, k = p % BK;
                const float4 values = *reinterpret_cast<const float4*>(
                    A + std::size_t(block_row + r) * N + k0 + k);
                *reinterpret_cast<float4*>(&As[r][k]) = values;
            }
            for (int v = tid; v < BK * BN / 4; v += THREADS) {
                const int p = 4 * v, k = p / BN, c = p % BN;
                const float4 values = *reinterpret_cast<const float4*>(
                    B + std::size_t(k0 + k) * N + block_col + c);
                *reinterpret_cast<float4*>(&Bs[k][c]) = values;
            }
        } else {
            for (int p = tid; p < BM * BK; p += THREADS) {
                const int r = p / BK, k = p % BK;
                if constexpr (ALIGNED_FAST)
                    As[r][k] = A[std::size_t(block_row + r) * N + k0 + k];
                else
                    As[r][k] = (block_row + r < N && k0 + k < N)
                        ? A[std::size_t(block_row + r) * N + k0 + k] : 0.0f;
            }
            for (int p = tid; p < BK * BN; p += THREADS) {
                const int k = p / BN, c = p % BN;
                if constexpr (ALIGNED_FAST)
                    Bs[k][c] = B[std::size_t(k0 + k) * N + block_col + c];
                else
                    Bs[k][c] = (k0 + k < N && block_col + c < N)
                        ? B[std::size_t(k0 + k) * N + block_col + c] : 0.0f;
            }
        }
        __syncthreads();  // All tile elements must be visible before use.

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a_reg[i] = As[ty + i * THREADS_Y][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b_reg[j] = Bs[k][tx + j * THREADS_X];

            // Outer product: eight shared values feed sixteen FMAs.
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
            }
        }
        __syncthreads();  // Finish reading before any thread overwrites tiles.
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = block_row + ty + i * THREADS_Y;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = block_col + tx + j * THREADS_X;
            // Outputs owned by one thread are strided, so stores stay scalar.
            if constexpr (ALIGNED_FAST)
                C[std::size_t(row) * N + col] = acc[i][j];
            else if (row < N && col < N)
                C[std::size_t(row) * N + col] = acc[i][j];
        }
    }
}

// Host launch wrappers keep block shape and output tile shape together.
using Launch = void (*)(const float*, const float*, float*, int);
struct Variant {
    const char* name;
    Launch launch;
};

void launch_naive(const float* A, const float* B, float* C, int N) {
    matmul_naive<<<dim3((N+15)/16, (N+15)/16), dim3(16,16)>>>(A,B,C,N);
}

template<int TILE>
void launch_shared(const float* A, const float* B, float* C, int N) {
    matmul_shared<TILE><<<dim3((N+TILE-1)/TILE, (N+TILE-1)/TILE),
                         dim3(TILE,TILE)>>>(A,B,C,N);
}

void launch_register(const float* A, const float* B, float* C, int N) {
    constexpr int BM=64, BN=64, BK=16, TM=4, TN=4;
    const dim3 block(BN/TN, BM/TM);
    const dim3 grid((N+BN-1)/BN, (N+BM-1)/BM);
    matmul_register_tiled<BM,BN,BK,TM,TN><<<grid,block>>>(A,B,C,N);
}

bool register_complete_tiles(int N) {
    // For the fixed BM=BN=64, BK=16 configuration, this also gives N%4==0.
    return N > 0 && N % 64 == 0;
}

bool register_float4_eligible(const float* A, const float* B, int N) {
    // cudaMalloc bases are aligned; sliced/offset pointers may not be.
    return register_complete_tiles(N)
        && reinterpret_cast<std::uintptr_t>(A) % 16 == 0
        && reinterpret_cast<std::uintptr_t>(B) % 16 == 0;
}

void launch_register_aligned(const float* A, const float* B, float* C, int N) {
    if (!register_complete_tiles(N)) {
        launch_register(A,B,C,N);
        return;
    }
    matmul_register_tiled<64,64,16,4,4,true,false>
        <<<dim3(N/64,N/64),dim3(16,16)>>>(A,B,C,N);
}

void launch_register_float4(const float* A, const float* B, float* C, int N) {
    if (!register_float4_eligible(A,B,N)) {
        launch_register(A,B,C,N);
        return;
    }
    matmul_register_tiled<64,64,16,4,4,true,true>
        <<<dim3(N/64,N/64),dim3(16,16)>>>(A,B,C,N);
}

const Variant variants[] = {
    {"naive", launch_naive},
    {"shared_8", launch_shared<8>},
    {"shared_16", launch_shared<16>},
    {"shared_32", launch_shared<32>},
    {"register_64x64x16_4x4", launch_register},
    {"register_aligned_scalar", launch_register_aligned},
    {"register_aligned_float4", launch_register_float4},
};

const char* register_path(const Variant& variant, const float* A, const float* B, int N) {
    if (variant.launch == launch_register_float4)
        return register_float4_eligible(A,B,N) ? "float4_fast" : "scalar_masked";
    if (variant.launch == launch_register_aligned)
        return register_complete_tiles(N) ? "scalar_fast" : "scalar_masked";
    if (variant.launch == launch_register) return "scalar_masked";
    return "baseline";
}

void fill_inputs(std::vector<float>& A, std::vector<float>& B) {
    std::mt19937 rng(20261105);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i=0; i<A.size(); ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }
}

bool validate_small_sizes() {
    bool passed = true;
    struct Case { int n, a_offset, b_offset; };
    // Include full tiles, partial tiles with aligned row strides (N=68),
    // and full tiles with independently misaligned A/B base addresses.
    const std::initializer_list<Case> cases = {
        {1,0,0}, {7,0,0}, {15,0,0}, {16,0,0},
        {17,0,0}, {31,0,0}, {32,0,0}, {33,0,0},
        {63,0,0}, {64,0,0}, {65,0,0}, {68,0,0},
        {127,0,0}, {128,0,0}, {129,0,0}, {256,0,0},
        {64,1,0}, {64,0,1}, {64,1,1}
    };
    for (const auto test : cases) {
        const int N=test.n;
        const std::size_t count = std::size_t(N)*N;
        const std::size_t bytes = count*sizeof(float);
        std::vector<float> h_A(count), h_B(count), h_C(count);
        fill_inputs(h_A,h_B);
        const auto reference = cpu_gemm_reference(h_A,h_B,N);
        float *A_base=nullptr, *B_base=nullptr, *C=nullptr;
        CUDA_CHECK(cudaMalloc(&A_base,bytes+test.a_offset*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B_base,bytes+test.b_offset*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&C,bytes));
        float* A=A_base+test.a_offset;
        float* B=B_base+test.b_offset;
        CUDA_CHECK(cudaMemcpy(A,h_A.data(),bytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B,h_B.data(),bytes,cudaMemcpyHostToDevice));
        for (const auto& variant : variants) {
            // Poison output so a missing store cannot inherit a previous result.
            std::fill(h_C.begin(),h_C.end(),std::numeric_limits<float>::quiet_NaN());
            CUDA_CHECK(cudaMemcpy(C,h_C.data(),bytes,cudaMemcpyHostToDevice));
            variant.launch(A,B,C,N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C.data(),C,bytes,cudaMemcpyDeviceToHost));
            const auto errors = compare_gemm(h_C,reference,1e-3,1e-4);
            std::cout << "N=" << N << " " << variant.name << ": "
                      << (errors.passed() ? "PASS" : "FAIL")
                      << " path=" << register_path(variant,A,B,N)
                      << " offsets=" << test.a_offset << "," << test.b_offset
                      << " max_abs=" << errors.max_abs
                      << " mismatches=" << errors.mismatches
                      << " nonfinite=" << errors.nonfinite << '\n';
            passed = passed && errors.passed();
        }
        CUDA_CHECK(cudaFree(A_base));
        CUDA_CHECK(cudaFree(B_base));
        CUDA_CHECK(cudaFree(C));
    }
    return passed;
}

// Large-size smoke check: deterministic CPU-FP64 samples, not full validation.
// Includes tile boundaries and randomly selected output positions.
std::vector<std::size_t> sample_positions(int N) {
    std::vector<std::size_t> positions;
    for (int r : {0,15,16,31,32,63,64,N-1})
        for (int c : {0,15,16,31,32,63,64,N-1})
            if (r<N && c<N) positions.push_back(std::size_t(r)*N+c);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> coord(0,N-1);
    for (int i=0; i<64; ++i) {
        const int r=coord(rng), c=coord(rng);
        positions.push_back(std::size_t(r)*N+c);
    }
    std::sort(positions.begin(),positions.end());
    positions.erase(std::unique(positions.begin(),positions.end()),positions.end());
    return positions;
}

bool benchmark() {
    constexpr int N=4096, warmup=5, repeat=50, samples=5;
    const std::size_t count=std::size_t(N)*N, bytes=count*sizeof(float);
    std::vector<float> h_A(count),h_B(count),h_C(count);
    fill_inputs(h_A,h_B);
    const auto positions=sample_positions(N);
    std::vector<double> reference(positions.size(),0.0);
    for (std::size_t s=0; s<positions.size(); ++s) {
        const auto row=positions[s]/N, col=positions[s]%N;
        for (int k=0; k<N; ++k)
            reference[s] += double(h_A[row*N+k])*double(h_B[std::size_t(k)*N+col]);
    }
    float *A=nullptr,*B=nullptr,*C=nullptr;
    CUDA_CHECK(cudaMalloc(&A,bytes));
    CUDA_CHECK(cudaMalloc(&B,bytes));
    CUDA_CHECK(cudaMalloc(&C,bytes));
    CUDA_CHECK(cudaMemcpy(A,h_A.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B,h_B.data(),bytes,cudaMemcpyHostToDevice));
    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    bool passed=true;
    // Preserve the original shared variants as baselines; omit slow naive timing.
    for (const auto& variant : variants) {
        if (variant.launch == launch_naive) continue;
        std::fill(h_C.begin(),h_C.end(),std::numeric_limits<float>::quiet_NaN());
        CUDA_CHECK(cudaMemcpy(C,h_C.data(),bytes,cudaMemcpyHostToDevice));
        variant.launch(A,B,C,N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C.data(),C,bytes,cudaMemcpyDeviceToHost));
        std::vector<float> actual(positions.size());
        for (std::size_t s=0; s<positions.size(); ++s) actual[s]=h_C[positions[s]];
        const auto errors=compare_gemm(actual,reference,5e-3,1e-4);
        const bool all_finite=std::all_of(h_C.begin(),h_C.end(),
                                        [](float x) { return std::isfinite(x); });
        const bool valid=errors.passed() && all_finite;
        std::cout << "N=" << N << " " << variant.name << " sampled_check="
                  << (valid ? "PASS" : "FAIL") << " checked=" << positions.size()
                  << " path=" << register_path(variant,A,B,N)
                  << " max_abs=" << errors.max_abs
                  << " all_outputs_finite=" << all_finite << '\n';
        if (!valid) { passed=false; continue; }
        for (int i=0; i<warmup; ++i) variant.launch(A,B,C,N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> timings;
        for (int s=0; s<samples; ++s) {
            CUDA_CHECK(cudaEventRecord(start));
            for (int i=0; i<repeat; ++i) variant.launch(A,B,C,N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms=0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms,start,stop));
            timings.push_back(elapsed_ms/repeat);
        }
        std::sort(timings.begin(),timings.end());
        const float median_ms=timings[samples/2];
        const double gflops=2.0*N*N*N/(median_ms/1000.0)/1e9;
        std::cout << variant.name << " median_ms=" << median_ms
                  << " GFLOP/s=" << gflops << '\n';
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return passed;
}

int main(int argc, char** argv) {
    const bool test_only=argc==2 && std::string(argv[1])=="--test-only";
    if (argc>1 && !test_only) {
        std::cerr << "Usage: matrix_matmul [--test-only]\n";
        return EXIT_FAILURE;
    }
    int device=0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
    if (!validate_small_sizes()) return EXIT_FAILURE;
    if (!test_only && !benchmark()) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
