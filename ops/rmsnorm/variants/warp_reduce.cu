#include "common/cuda_check.h"
#include "ops/rmsnorm/include/rmsnorm.h"

#define WARP_SIZE 32

namespace {

// 强制内联，消除函数调用开销
__device__ __forceinline__ float warp_reduce_sum(float val) {
  // 编译器指令：展开循环，节省开销
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    // mask = 0xffffffff 表示 32 个 1 线程全部参与, 偏移量是offset
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  // 只有第一个线程是总和
  return val;
}

__global__ void rmsnorm_warp_reduce_kernel(const float* input, float* output, const float* weight, size_t rows, size_t cols, float eps) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  // warp 块编号，相当于第几行
  size_t row_idx = tid / WARP_SIZE; 
  // 线程在 warp 内编号
  size_t lane_id = tid % WARP_SIZE; 

  if (row_idx < rows) {
    // 数据在 input 的绝对位置
    size_t in_idx = row_idx * cols;

    float sq_sum = 0.0f;
    for (size_t j = lane_id; j < cols; j += WARP_SIZE) {
      float x = input[in_idx + j];
      sq_sum += x * x;
    }
    // 寄存器级封装：Warp 内树状规约求和
    sq_sum = warp_reduce_sum(sq_sum);

    // 把 0 号手里的数据，利用硬件总线直接复制（广播）给同 warp 的所有线程
    sq_sum = __shfl_sync(0xffffffff, sq_sum, 0);
    // 若选择后广播，则只有 0 号在计算，其他赋闲，不如先复制再各自计算
    float inv_rms = rsqrtf(sq_sum / cols + eps);

    for (size_t j = lane_id; j < cols; j += WARP_SIZE) {
      float x = input[in_idx + j];
      output[in_idx + j] = x * inv_rms * weight[j];
    }
  }
}

} // namespace

const char* rmsnorm_variant_name() {
  return "warp_reduce";
}

void launch_rmsnorm(const float* d_input, float* d_output, const float* d_weight, size_t rows, size_t cols, float eps) {
  // 我们给每个 Block 安排 256 个线程。
  // 这意味着，1 个 Block 内部物理打包了 8 个 Warp！(256 / 32 = 8)
  const int threads = 256; 
  const int warps_per_block = threads / WARP_SIZE; 
  
  // 1 个 warp 负责 1 行，一个 block 中有多个 warp，需计算要几个 block
  const int blocks = (rows + warps_per_block - 1) / warps_per_block;

  rmsnorm_warp_reduce_kernel<<<blocks, threads>>>(d_input, d_output, d_weight, rows, cols, eps);
  CUDA_CHECK(cudaGetLastError());
}