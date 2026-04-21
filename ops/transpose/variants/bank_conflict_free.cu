#include <algorithm>
#include "common/cuda_check.h"
#include "ops/transpose/include/transpose.h"

#define TILE_DIM 16

namespace {

__global__ void transpose_bank_conflict_free_kernel(const float* input, float* output, size_t rows,
                                                    size_t cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
  size_t block_rows = (cols + TILE_DIM - 1) / TILE_DIM;
  size_t block_cols = (rows + TILE_DIM - 1) / TILE_DIM;

  for (size_t b_j = blockIdx.y; b_j < block_cols; b_j += gridDim.y) {
    for (size_t b_i = blockIdx.x; b_i < block_rows; b_i += gridDim.x) {
      size_t in_tile_x = b_i * TILE_DIM + threadIdx.x;
      size_t in_tile_y = b_j * TILE_DIM + threadIdx.y;
      if (in_tile_x < cols && in_tile_y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[in_tile_y * cols + in_tile_x];
      }
      __syncthreads();

      size_t out_tile_x = b_j * TILE_DIM + threadIdx.x;
      size_t out_tile_y = b_i * TILE_DIM + threadIdx.y;
      if (out_tile_x < rows && out_tile_y < cols) {
        output[out_tile_y * rows + out_tile_x] = tile[threadIdx.x][threadIdx.y];
      }
      __syncthreads();
    }
  }
}

}  // namespace

const char* transpose_variant_name() {
  return "bank_conflict_free";
}

void launch_transpose(const float* d_input, float* d_out, size_t rows, size_t cols) {
  // 1. 计算需要多少个 blocks 来覆盖整个矩阵
  // 注意：每个 block 负责转置一个 TILE_DIM x TILE_DIM 的子矩阵
  unsigned int target_blocks_x =
      (cols + TILE_DIM - 1) / TILE_DIM;  // 转置后矩阵的列数决定了 x 方向的块数
  unsigned int target_blocks_y =
      (rows + TILE_DIM - 1) / TILE_DIM;  // 转置后矩阵的行数决定了 y 方向的块数

  dim3 threads(TILE_DIM, TILE_DIM);

  dim3 blocks(std::min(target_blocks_x, 65535u), std::min(target_blocks_y, 65535u));

  transpose_bank_conflict_free_kernel<<<blocks, threads>>>(d_input, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}