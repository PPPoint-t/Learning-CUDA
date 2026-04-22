#include "common/cuda_check.h"
#include "ops/rmsnorm/include/rmsnorm.h"

// 定义每个 Block 的线程数，在编译期确定方便共享内存的声明
#define THREADS_PER_BLOCK 256

namespace {

__global__ void rmsnorm_shared_reduce_kernel(const float* input, float* output, const float* weight, size_t rows, size_t cols, float eps) {
  // 【核心改变 1】：既然 1 个 Block 负责 1 行，那 blockIdx.x 就是当前行的绝对编号！
  size_t row_idx = blockIdx.x;

  if (row_idx < rows) {
    // 长度等于 Block 的线程数，便于求和
    // 这是一个 Block 编译期就申请好公用的，和线程号无关
    __shared__ float s_sum[THREADS_PER_BLOCK];
    
    size_t row_offset = row_idx * cols; 
    float local_sq_sum = 0.0f;
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
      float x = input[row_offset + j];
      local_sq_sum += x * x;
    }
    // 此时该行的平方和已经被每个线程算好了，放到共享内存里
    s_sum[threadIdx.x] = local_sq_sum;
    __syncthreads();

    // 树状规约：256进128，128进64，64进32...
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        // 前一半的线程，把后一半线程位置上的数字拿过来，加到自己身上
        s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      }
      __syncthreads(); 
    }

    // RMS 值全部聚集在 s_sum[0] 
    if (threadIdx.x == 0) {
      float mean_square = s_sum[0] / cols;
      // inv_rms 也放入 s_sum[0]
      s_sum[0] = rsqrtf(mean_square + eps); 
    }
    __syncthreads();

    float inv_rms = s_sum[0];

    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
      float x = input[row_offset + j];
      output[row_offset + j] = x * inv_rms * weight[j];
    }
  }
}

} // namespace

const char* rmsnorm_variant_name() {
  return "shared_reduce";
}

void launch_rmsnorm(const float* d_input, float* d_output, const float* d_weight, size_t rows, size_t cols, float eps) {
  const int threads = THREADS_PER_BLOCK;

  const int blocks = rows; 

  rmsnorm_shared_reduce_kernel<<<blocks, threads>>>(d_input, d_output, d_weight, rows, cols, eps);
  CUDA_CHECK(cudaGetLastError());
}