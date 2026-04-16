#include "common/cuda_check.h"
#include "ops/vector_add/include/vector_add.h"

namespace {

__global__ void vector_add_naive_kernel(const float* a, const float* b, float* out, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

}  // namespace

const char* vector_add_variant_name() {
  return "naive";
}

void launch_vector_add(const float* d_a, const float* d_b, float* d_out, size_t n) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  vector_add_naive_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
  CUDA_CHECK(cudaGetLastError());
}
