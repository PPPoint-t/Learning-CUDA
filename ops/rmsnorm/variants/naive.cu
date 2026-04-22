#include "common/cuda_check.h"
#include "ops/rmsnorm/include/rmsnorm.h"

namespace {

__global__ void rmsnorm_naive_kernel(const float* input, float* output, const float* weight, size_t rows, size_t cols, float eps) {
  // 一维索引
  size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (row_idx < rows) {
    float mean_square = 0.0f;
    size_t row_offset = row_idx * cols; // 这一行的内存起点

    // 求平方和
    for (size_t j = 0; j < cols; ++j) {
      float x = input[row_offset + j];
      mean_square += x * x;
    }

    mean_square /= cols;
    float inv_rms = rsqrtf(mean_square + eps); 
    for (size_t j = 0; j < cols; ++j) {
      float x = input[row_offset + j];
      output[row_offset + j] = x * inv_rms * weight[j];
    }
  }
}

}

const char* rmsnorm_variant_name() {
  return "naive";
}

void launch_rmsnorm(const float* d_input, float* d_output, const float* d_weight, size_t rows, size_t cols, float eps) {
  const int threads = 256;
  const int block_size = (rows + threads - 1) / threads;
  rmsnorm_naive_kernel<<<block_size, threads>>>(d_input, d_output, d_weight, rows, cols, eps);
  CUDA_CHECK(cudaGetLastError());
}