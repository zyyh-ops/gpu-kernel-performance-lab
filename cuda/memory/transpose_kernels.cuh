#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// Preserves the three learning kernels' layout and loop structure.
// Square, contiguous row-major FP32 transpose only (input and output must not alias).
namespace gpu_lab {
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_naive(const float* input, float* output, int n) {
    const int x = blockIdx.x * TILE_DIM + threadIdx.x;
    const int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && y + j < n) {
            output[static_cast<std::size_t>(x) * n + y + j] =
                input[static_cast<std::size_t>(y + j) * n + x];
        }
    }
}

__global__ void transpose_shared(const float* input, float* output, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && y + j < n) {
            tile[threadIdx.y + j][threadIdx.x] =
                input[static_cast<std::size_t>(y + j) * n + x];
        }
    }
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && y + j < n) {
            output[static_cast<std::size_t>(y + j) * n + x] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void transpose_shared_padded(const float* input, float* output, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && y + j < n) {
            tile[threadIdx.y + j][threadIdx.x] =
                input[static_cast<std::size_t>(y + j) * n + x];
        }
    }
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && y + j < n) {
            output[static_cast<std::size_t>(y + j) * n + x] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void copy_kernel(const float* input, float* output, std::size_t n_elements) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_elements) output[i] = input[i];
}
} // namespace gpu_lab
