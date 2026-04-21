#include "common/cuda_check.h"
#include "ops/transpose/include/transpose.h"

namespace {

__global__ void transpose_naive_kernel(const float* input, float* output, size_t rows,
                                       size_t cols) {
  size_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (col_idx < cols && row_idx < rows) {
    size_t input_idx = row_idx * cols + col_idx;
    size_t output_idx = col_idx * rows + row_idx;
    output[output_idx] = input[input_idx];
  }
}

}  // namespace

const char* transpose_variant_name() {
  return "naive";
}

void launch_transpose(const float* d_input, float* d_out, size_t rows, size_t cols) {
  dim3 threads(16, 16);
  dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
  transpose_naive_kernel<<<blocks, threads>>>(d_input, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}