#include "common/cuda_check.h"
#include "ops/transpose/include/transpose.h"

namespace {

__global__ void transpose_strided_kernel(const float* input, float* output, size_t rows,
                                         size_t cols) {
  size_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  size_t stride_x = gridDim.x * blockDim.x;
  size_t stride_y = gridDim.y * blockDim.y;

  for (size_t c = col_idx; c < cols; c += stride_x) {
    for (size_t r = row_idx; r < rows; r += stride_y) {
      output[c * rows + r] = input[r * cols + c];
    }
  }
}

}  // namespace

const char* transpose_variant_name() {
  return "strided_loop";
}

void launch_transpose(const float* d_input, float* d_out, size_t rows, size_t cols) {
  dim3 block_size(16, 16);
  dim3 grid_size((cols + block_size.x - 1) / block_size.x,
                 (rows + block_size.y - 1) / block_size.y);
  transpose_strided_kernel<<<grid_size, block_size>>>(d_input, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}