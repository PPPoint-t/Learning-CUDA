#include "common/cuda_check.h"
#include "ops/silu/include/silu.h"

namespace {

__global__ void silu_strided_kernel(const float* input, float* output, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < n; i += stride) {
    output[i] = input[i] / (1.0f + expf(-input[i]));
  }
}

}  // namespace

const char* silu_variant_name() {
  return "strided_loop";
}

void launch_silu(const float* d_input, float* d_output, size_t n) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  silu_strided_kernel<<<blocks, threads>>>(d_input, d_output, n);
  CUDA_CHECK(cudaGetLastError());
}
