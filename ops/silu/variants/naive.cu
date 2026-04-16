#include "common/cuda_check.h"
#include "ops/silu/include/silu.h"

namespace {

__global__ void silu_naive_kernel(const float* input, float* output, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] / (1.0f + expf(-input[idx]));
  }
}

}  // namespace

const char* silu_variant_name() {
  return "naive";
}

void launch_silu(const float* d_input, float* d_output, size_t n) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  silu_naive_kernel<<<blocks, threads>>>(d_input, d_output, n);
  CUDA_CHECK(cudaGetLastError());
}
