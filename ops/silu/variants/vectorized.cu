#include <algorithm>

#include "common/cuda_check.h"
#include "ops/silu/include/silu.h"

namespace {

__global__ void silu_vectorized_kernel(const float* input, float* output, size_t n) {
  const float4* input_vec = reinterpret_cast<const float4*>(input);
  float4* output_vec = reinterpret_cast<float4*>(output);

  size_t n_vec = n / 4;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < n_vec; i += stride) {
    float4 in = input_vec[i];
    output_vec[i].x = in.x / (1.0f + expf(-in.x));
    output_vec[i].y = in.y / (1.0f + expf(-in.y));
    output_vec[i].z = in.z / (1.0f + expf(-in.z));
    output_vec[i].w = in.w / (1.0f + expf(-in.w));
  }

  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start + idx; i < n; i += stride) {
    output[i] = input[i] / (1.0f + expf(-input[i]));
  }
}

}  // namespace

const char* silu_variant_name() {
  return "vectorized";
}

void launch_silu(const float* d_input, float* d_output, size_t n) {
  const int threads = 256;
  const int n_vec = n / 4;
  int blocks = (n_vec + threads - 1) / threads;
  blocks = std::max(blocks, 1);
  silu_vectorized_kernel<<<blocks, threads>>>(d_input, d_output, n);
  CUDA_CHECK(cudaGetLastError());
}
