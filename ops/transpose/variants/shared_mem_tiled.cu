#include <algorithm>
#include "common/cuda_check.h"
#include "ops/transpose/include/transpose.h"

#define TILE_DIM 16

namespace {

__global__ void transpose_shared_mem_tiled_kernel(const float* input, float* output, size_t rows,
                                                  size_t cols) {
  // 1. 申请共享内存中转站 (每个 Block 独立拥有一份)
  __shared__ float tile[TILE_DIM][TILE_DIM];

  // 2. Block 级别的 2D 网格跨步 (保护我们不超越 65535 的硬件红线)
  // 注意这里跨步的是 blockIdx，而不是 threadIdx
  size_t grid_width_blocks = (cols + TILE_DIM - 1) / TILE_DIM;
  size_t grid_height_blocks = (rows + TILE_DIM - 1) / TILE_DIM;
  // 当前 tile（线程块）在宏观大矩阵中的“块坐标”, 遍历顺序无影响, 即第几个tile
  for (size_t by = blockIdx.y; by < grid_height_blocks; by += gridDim.y) {
    for (size_t bx = blockIdx.x; bx < grid_width_blocks; bx += gridDim.x) {
      // ==========================================
      // 第一阶段：从 Global 读取，存入 Shared Mem
      // ==========================================
      // 当前线程在原矩阵中的 (x, y) 坐标,绝 对坐标 = 宏观块偏移量 + 微观块内偏移量
      size_t in_x = bx * TILE_DIM + threadIdx.x;  // 列
      size_t in_y = by * TILE_DIM + threadIdx.y;  // 行

      if (in_x < cols && in_y < rows) {
        // 合并读取 (因为 threadIdx.x 连续，in_x 连续，物理内存连续)
        tile[threadIdx.y][threadIdx.x] = input[in_y * cols + in_x];
      }

      // 等待 Block 内所有线程把这块 Tile 搬运完毕
      __syncthreads();

      // ==========================================
      // 第二阶段：从 Shared Mem 读出，转置后写回 Global
      // ==========================================
      // 魔法开始：此时这个 Block 要写去哪里？
      // 原来的块坐标是 (bx, by)，转置后它应该写到 (by, bx) 的位置！
      // 并且，我们依然让 threadIdx.x 负责输出矩阵的最内层连续内存！

      size_t out_x = by * TILE_DIM + threadIdx.x;  // 输出的列 (来自原矩阵的行 by)
      size_t out_y = bx * TILE_DIM + threadIdx.y;  // 输出的行 (来自原矩阵的列 bx)

      if (out_x < rows && out_y < cols) {  // 注意：边界条件反过来了
        // 1. 从 tile 中读的时候，把 x 和 y 反过来读，完成转置
        // 2. 写入 output 时，out_x 是由 threadIdx.x 驱动的，完美合并写入！
        output[out_y * rows + out_x] = tile[threadIdx.x][threadIdx.y];
      }

      // 必须再同步一次，防止部分线程跑得太快，进入下一个 block 循环并覆盖了 tile
      __syncthreads();
    }
  }
}

}  // namespace

const char* transpose_variant_name() {
  return "shared_mem_tiled";
}

void launch_transpose(const float* d_input, float* d_out, size_t rows, size_t cols) {
  dim3 threads(TILE_DIM, TILE_DIM);

  unsigned int target_blocks_x = (cols + threads.x - 1) / threads.x;
  unsigned int target_blocks_y = (rows + threads.y - 1) / threads.y;

  dim3 blocks(std::min(target_blocks_x, 65535u), std::min(target_blocks_y, 65535u));
  // tile 大小是 16x16，所以每个 block 负责转置一个 16x16 的子矩阵。为了覆盖整个输入矩阵，我们需要足够的 blocks。
  transpose_shared_mem_tiled_kernel<<<blocks, threads>>>(d_input, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}