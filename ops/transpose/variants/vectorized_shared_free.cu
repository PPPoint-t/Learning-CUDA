#include <algorithm>
#include <cstdint>

#include "common/cuda_check.h"
#include "ops/transpose/include/transpose.h"

#define TILE_DIM 32
#define SCALAR_TILE_DIM 16

namespace {

__global__ void transpose_scalar_fallback_kernel(const float* input, float* output, size_t rows,
                                                 size_t cols) {
  __shared__ float tile[SCALAR_TILE_DIM][SCALAR_TILE_DIM + 1];

  size_t block_rows = (cols + SCALAR_TILE_DIM - 1) / SCALAR_TILE_DIM;
  size_t block_cols = (rows + SCALAR_TILE_DIM - 1) / SCALAR_TILE_DIM;

  for (size_t b_j = blockIdx.y; b_j < block_cols; b_j += gridDim.y) {
    for (size_t b_i = blockIdx.x; b_i < block_rows; b_i += gridDim.x) {
      size_t in_tile_x = b_i * SCALAR_TILE_DIM + threadIdx.x;
      size_t in_tile_y = b_j * SCALAR_TILE_DIM + threadIdx.y;

      if (in_tile_x < cols && in_tile_y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[in_tile_y * cols + in_tile_x];
      }
      __syncthreads();

      size_t out_tile_x = b_j * SCALAR_TILE_DIM + threadIdx.x;
      size_t out_tile_y = b_i * SCALAR_TILE_DIM + threadIdx.y;

      if (out_tile_x < rows && out_tile_y < cols) {
        output[out_tile_y * rows + out_tile_x] = tile[threadIdx.x][threadIdx.y];
      }
      __syncthreads();
    }
  }
}

__global__ void transpose_vectorized_kernel(const float* input, float* output, size_t rows,
                                            size_t cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  size_t block_rows = (cols + TILE_DIM - 1) / TILE_DIM;
  size_t block_cols = (rows + TILE_DIM - 1) / TILE_DIM;

  for (size_t b_j = blockIdx.y; b_j < block_cols; b_j += gridDim.y) {
    for (size_t b_i = blockIdx.x; b_i < block_rows; b_i += gridDim.x) {
      size_t in_tile_x = b_i * TILE_DIM + threadIdx.x * 4;
      size_t in_tile_y = b_j * TILE_DIM + threadIdx.y;

      if (in_tile_y < rows) {
        if (in_tile_x + 3 < cols) {
          float4 in_vec = reinterpret_cast<const float4*>(&input[in_tile_y * cols + in_tile_x])[0];
          tile[threadIdx.y][threadIdx.x * 4 + 0] = in_vec.x;
          tile[threadIdx.y][threadIdx.x * 4 + 1] = in_vec.y;
          tile[threadIdx.y][threadIdx.x * 4 + 2] = in_vec.z;
          tile[threadIdx.y][threadIdx.x * 4 + 3] = in_vec.w;
        } else {
#pragma unroll
          for (int lane = 0; lane < 4; ++lane) {
            size_t x = in_tile_x + lane;
            if (x < cols) {
              tile[threadIdx.y][threadIdx.x * 4 + lane] = input[in_tile_y * cols + x];
            }
          }
        }
      }
      __syncthreads();

      size_t out_tile_x = b_j * TILE_DIM + threadIdx.x * 4;
      size_t out_tile_y = b_i * TILE_DIM + threadIdx.y;

      if (out_tile_y < cols) {
        if (out_tile_x + 3 < rows) {
          float4 out_vec;
          out_vec.x = tile[threadIdx.x * 4 + 0][threadIdx.y];
          out_vec.y = tile[threadIdx.x * 4 + 1][threadIdx.y];
          out_vec.z = tile[threadIdx.x * 4 + 2][threadIdx.y];
          out_vec.w = tile[threadIdx.x * 4 + 3][threadIdx.y];
          reinterpret_cast<float4*>(&output[out_tile_y * rows + out_tile_x])[0] = out_vec;
        } else {
#pragma unroll
          for (int lane = 0; lane < 4; ++lane) {
            size_t x = out_tile_x + lane;
            if (x < rows) {
              output[out_tile_y * rows + x] = tile[threadIdx.x * 4 + lane][threadIdx.y];
            }
          }
        }
      }

      __syncthreads();
    }
  }
}

}  // namespace

const char* transpose_variant_name() {
  return "vectorized_shared_free";
}

void launch_transpose(const float* d_input, float* d_out, size_t rows, size_t cols) {
  bool input_aligned = reinterpret_cast<std::uintptr_t>(d_input) % alignof(float4) == 0;
  bool output_aligned = reinterpret_cast<std::uintptr_t>(d_out) % alignof(float4) == 0;
  bool can_vectorize = input_aligned && output_aligned && rows % 4 == 0 && cols % 4 == 0;

  if (can_vectorize) {
    dim3 threads(TILE_DIM / 4, TILE_DIM);

    unsigned int target_blocks_x = (cols + TILE_DIM - 1) / TILE_DIM;
    unsigned int target_blocks_y = (rows + TILE_DIM - 1) / TILE_DIM;

    dim3 blocks(std::min(target_blocks_x, 65535u), std::min(target_blocks_y, 65535u));
    transpose_vectorized_kernel<<<blocks, threads>>>(d_input, d_out, rows, cols);
  } else {
    dim3 threads(SCALAR_TILE_DIM, SCALAR_TILE_DIM);

    unsigned int target_blocks_x = (cols + SCALAR_TILE_DIM - 1) / SCALAR_TILE_DIM;
    unsigned int target_blocks_y = (rows + SCALAR_TILE_DIM - 1) / SCALAR_TILE_DIM;

    dim3 blocks(std::min(target_blocks_x, 65535u), std::min(target_blocks_y, 65535u));
    transpose_scalar_fallback_kernel<<<blocks, threads>>>(d_input, d_out, rows, cols);
  }

  CUDA_CHECK(cudaGetLastError());
}
