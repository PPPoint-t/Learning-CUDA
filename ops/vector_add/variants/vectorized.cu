#include <algorithm>

#include "common/cuda_check.h"
#include "ops/vector_add/include/vector_add.h"

namespace {

__global__ void vector_add_vectorized_kernel(const float* a, const float* b, float* out, size_t n) {
  // 在循环外指针强转读取内存方式
  const float4* a_vec = reinterpret_cast<const float4*>(a);
  const float4* b_vec = reinterpret_cast<const float4*>(b);
  float4* out_vec = reinterpret_cast<float4*>(out);

  size_t n_vec = n / 4;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < n_vec; i += stride) {
    float4 a_val = a_vec[i];
    float4 b_val = b_vec[i];
    out_vec[i].x = a_val.x + b_val.x;
    out_vec[i].y = a_val.y + b_val.y;
    out_vec[i].z = a_val.z + b_val.z;
    out_vec[i].w = a_val.w + b_val.w;
  }

  size_t tail_start = n_vec * 4;
  // i依旧以线程索引为起点，步长为线程总数，处理剩余元素
  for (size_t i = tail_start + idx; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}

}  // namespace

const char* vector_add_variant_name() {
  return "vectorized";
}

void launch_vector_add(const float* d_a, const float* d_b, float* d_out, size_t n) {
  const int threads = 256;
  // 每个线程处理四个元素
  const int n_vec = n / 4;
  int blocks = (n_vec + threads - 1) / threads;
  blocks = std::max(blocks, 1);
  vector_add_vectorized_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
  CUDA_CHECK(cudaGetLastError());
}
