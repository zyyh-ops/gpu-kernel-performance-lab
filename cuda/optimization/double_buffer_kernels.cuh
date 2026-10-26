#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Square contiguous row-major FP32 GEMM. TILE=16, compute: 16x16 threads.
// Two compile-time variants keep the aligned vectorized fast path unchanged.
constexpr int kGemmTile = 16;

__device__ __forceinline__ void cp_async_copy16(float* shared_dst, const float* global_src) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(global_src) : "memory");
}

// For tail/misaligned row-strides, copy one naturally aligned FP32 per thread.
// A zero-fill cp.async writes 0 into shared without reading from global memory.
// For invalid elements, callers pass the base of the allocation as a safe src.
__device__ __forceinline__ void cp_async_copy4_or_zero(
        float* shared_dst, const float* global_src, bool valid) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
    if (valid) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                     :: "r"(dst), "l"(global_src) : "memory");
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, 0;\n"
                     :: "r"(dst), "l"(global_src) : "memory");
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

template<int TILE, bool ALIGNED_FAST>
__global__ void matmul_single_buffer_kernel(const float* A, const float* B, float* C, int N) {
    static_assert(TILE == 16, "Specialized for TILE=16");
    extern __shared__ __align__(16) float smem[];
    float* As = smem;
    float* Bs = smem + TILE*TILE;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y*TILE + ty;
    const int col = blockIdx.x*TILE + tx;
    const int tiles = (N-1)/TILE + 1;
    float acc = 0.0f;
    for (int t=0; t<tiles; ++t) {
        const int ak = t*TILE + tx;
        const int bk = t*TILE + ty;
        if constexpr (ALIGNED_FAST) {
            As[ty*TILE+tx] = A[std::size_t(row)*N + ak];
            Bs[ty*TILE+tx] = B[std::size_t(bk)*N + col];
        } else {
            As[ty*TILE+tx] = row<N && ak<N ? A[std::size_t(row)*N + ak] : 0.0f;
            Bs[ty*TILE+tx] = bk<N && col<N ? B[std::size_t(bk)*N + col] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k=0; k<TILE; ++k) acc += As[ty*TILE+k]*Bs[k*TILE+tx];
        __syncthreads();
    }
    if constexpr (ALIGNED_FAST) C[std::size_t(row)*N+col] = acc;
    else if (row<N && col<N) C[std::size_t(row)*N+col] = acc;
}

// Issue both A and B tile copy into the selected ping-pong buffer.
// ALIGNED_FAST: N multiple of TILE, 64 lanes each copying 16 bytes to 16B-aligned addresses.
// General case: all 256 lanes copy 4 bytes, zero-fill OOB elements.
template<int TILE, bool ALIGNED_FAST>
__device__ __forceinline__ void prefetch_gemm_tile(
        float* smem, const float* A, const float* B, int N,
        int t, int buffer, int tid, int block_m, int block_n) {
    if constexpr (ALIGNED_FAST) {
        if (tid < TILE*TILE/4) {
            const int elem = tid*4;
            const int r = elem/TILE, c = elem%TILE;
            cp_async_copy16(smem + buffer*TILE*TILE + elem,
                            A + std::size_t(block_m*TILE+r)*N + t*TILE+c);
            cp_async_copy16(smem + (2+buffer)*TILE*TILE + elem,
                            B + std::size_t(t*TILE+r)*N + block_n*TILE+c);
        }
    } else {
        const int r=tid/TILE, c=tid%TILE;
        const int ar=block_m*TILE+r, ak=t*TILE+c;
        const int bk=t*TILE+r, bc=block_n*TILE+c;
        const bool av=(ar<N && ak<N), bv=(bk<N && bc<N);
        const float* ap = av ? A + std::size_t(ar)*N + ak : A;
        const float* bp = bv ? B + std::size_t(bk)*N + bc : B;
        cp_async_copy4_or_zero(smem + buffer*TILE*TILE+tid, ap, av);
        cp_async_copy4_or_zero(smem + (2+buffer)*TILE*TILE+tid, bp, bv);
    }
}

template<int TILE, bool ALIGNED_FAST>
__global__ void matmul_double_buffer_kernel(const float* A, const float* B, float* C, int N) {
    static_assert(TILE == 16, "Specialized for TILE=16");
    // Separate slices [A0][A1][B0][B1] within one 16B-aligned allocation.
    extern __shared__ __align__(16) float smem[];
    const int tx=threadIdx.x,ty=threadIdx.y,tid=ty*TILE+tx;
    const int row=blockIdx.y*TILE+ty, col=blockIdx.x*TILE+tx;
    const int tiles=(N-1)/TILE+1;
    int cur=0;

    prefetch_gemm_tile<TILE, ALIGNED_FAST>(smem,A,B,N,0,cur,tid,blockIdx.y,blockIdx.x);
    cp_async_commit();
    cp_async_wait_all();  // thread-local async group completion
    __syncthreads();      // CTA sees entire prefetched tile
    float acc=0.0f;
    for (int t=0; t<tiles; ++t) {
        const int next=cur^1;
        if (t+1<tiles) {
            prefetch_gemm_tile<TILE, ALIGNED_FAST>(smem,A,B,N,t+1,next,tid,blockIdx.y,blockIdx.x);
            cp_async_commit();
        }
        const float* As=smem+cur*TILE*TILE;
        const float* Bs=smem+(2+cur)*TILE*TILE;
        #pragma unroll
        for (int k=0; k<TILE; ++k) acc+=As[ty*TILE+k]*Bs[k*TILE+tx];
        if (t+1<tiles) {
            cp_async_wait_all();
            __syncthreads();  // also prevents overwriting a tile before readers finish
        }
        cur=next;
    }
    if constexpr (ALIGNED_FAST) C[std::size_t(row)*N+col]=acc;
    else if (row<N && col<N) C[std::size_t(row)*N+col]=acc;
}

constexpr std::size_t single_base_smem_bytes = 2*kGemmTile*kGemmTile*sizeof(float);
constexpr std::size_t double_base_smem_bytes = 4*kGemmTile*kGemmTile*sizeof(float);
