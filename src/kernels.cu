#include <algorithm>
#include <iostream>
#include <vector>
#include "../tester/utils.h"

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */

// count < pivot and == pivot on segment data[left .. left+len-1]
template <typename T>
__global__ void count_less_equal_kernel(const T *data, int left, int len,
                                        T pivot, int *d_less, int *d_equal) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len)
    return;
  T v = data[left + idx];
  if (v < pivot) {
    atomicAdd(d_less, 1);
  } else if (v == pivot) {
    atomicAdd(d_equal, 1);
  }
}

// partition segment data[left .. left+len-1] into newData[0..len-1]
// sections: [0 .. lessCount-1] = < pivot
//           [lessCount .. lessCount+equalCount-1] = == pivot
//           [lessCount+equalCount .. len-1] = > pivot
// d_lcur/d_ecur/d_gcur are device counters (should be zeroed before launch)
template <typename T>
__global__ void partition_kernel(const T *data, int left, int len, T pivot,
                                 int lessCount, int equalCount, T *newData,
                                 int *d_lcur, int *d_ecur, int *d_gcur) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len)
    return;
  T v = data[left + idx];
  if (v < pivot) {
    int pos = atomicAdd(d_lcur, 1); // 0..lessCount-1
    newData[pos] = v;
  } else if (v == pivot) {
    int pos = atomicAdd(d_ecur, 1); // 0..equalCount-1
    newData[lessCount + pos] = v;
  } else {
    int pos = atomicAdd(d_gcur, 1);
    newData[lessCount + equalCount + pos] = v;
  }
}

template <typename T> T kthLargest(const std::vector<T> &h_input, size_t k) {
  if (h_input.empty())
    return T(-100);
  size_t n = h_input.size();
  if (k < 1 || k > n)
    return T(-100);

  size_t target_desc_0based = k - 1;
  size_t rank_asc = n - 1 - target_desc_0based;

  T *d_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_data, h_input.data(), n * sizeof(T),
                        cudaMemcpyHostToDevice));

  int *d_less = nullptr;
  int *d_equal = nullptr;
  int *d_lcur = nullptr;
  int *d_ecur = nullptr;
  int *d_gcur = nullptr;
  CUDA_CHECK(cudaMalloc(&d_less, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_equal, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_lcur, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ecur, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_gcur, sizeof(int)));

  int left = 0;
  int right = (int)n; 

  const int THREADS = 256;

  T result = T(-100);

  int iter = 0;
  while (left < right) {
    ++iter;
    int len = right - left;
    if (len <= 0)
      break;

    if (len <= 32) {
      std::vector<T> tmp(len);
      CUDA_CHECK(cudaMemcpy(tmp.data(), d_data + left, len * sizeof(T),
                            cudaMemcpyDeviceToHost));
      std::sort(tmp.begin(), tmp.end());
      result = tmp[rank_asc - left];
      break;
    }

    int pivot_idx = left + (len / 2);
    T pivot_val;
    CUDA_CHECK(cudaMemcpy(&pivot_val, d_data + pivot_idx, sizeof(T),
                          cudaMemcpyDeviceToHost));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_less, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_equal, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int blocks = (len + THREADS - 1) / THREADS;
    count_less_equal_kernel<T>
        <<<blocks, THREADS>>>(d_data, left, len, pivot_val, d_less, d_equal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int lessCount = 0, equalCount = 0;
    CUDA_CHECK(
        cudaMemcpy(&lessCount, d_less, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(&equalCount, d_equal, sizeof(int), cudaMemcpyDeviceToHost));

    int pivotPos =
        left + lessCount;

    if ((int)rank_asc >= pivotPos && (int)rank_asc < pivotPos + equalCount) {
      result = pivot_val;
      break;
    }

    T *d_new = nullptr;
    CUDA_CHECK(cudaMalloc(&d_new, sizeof(T) * len));

    CUDA_CHECK(cudaMemcpy(d_lcur, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ecur, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gcur, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // launch partition kernel (d_new[0..len-1])
    int pblocks = (len + THREADS - 1) / THREADS;
    partition_kernel<T><<<pblocks, THREADS>>>(d_data, left, len, pivot_val,
                                              lessCount, equalCount, d_new,
                                              d_lcur, d_ecur, d_gcur);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int lcur = 0, ecur = 0, gcur = 0;
    CUDA_CHECK(cudaMemcpy(&lcur, d_lcur, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&ecur, d_ecur, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&gcur, d_gcur, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(d_data + left, d_new, len * sizeof(T),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_new));

    // left..right partitioned into [<][==][>]
    if ((int)rank_asc < pivotPos) {
      right = pivotPos;
    } else {
      left = pivotPos + equalCount;
    }
  }
  if (result == T(-100) && left < (int)n) {
    CUDA_CHECK(
        cudaMemcpy(&result, d_data + left, sizeof(T), cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_less));
  CUDA_CHECK(cudaFree(d_equal));
  CUDA_CHECK(cudaFree(d_lcur));
  CUDA_CHECK(cudaFree(d_ecur));
  CUDA_CHECK(cudaFree(d_gcur));

  return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

__host__ std::vector<int> computeQKVGroupMap(int query_heads, int kv_heads) {
  std::vector<int> group_map(query_heads);
  // map qh -> floor(qh * kv_heads / query_heads)
  for (int qh = 0; qh < query_heads; ++qh) {
    group_map[qh] = (int)((long long)qh * kv_heads / query_heads);
    if (group_map[qh] < 0)
      group_map[qh] = 0;
    if (group_map[qh] >= kv_heads)
      group_map[qh] = kv_heads - 1;
  }
  return group_map;
}

__global__ void flashAttentionKernel_streamed(
    const float *__restrict__ q,       // [batch, tgt_seq, q_heads, head_dim]
    const float *__restrict__ k,       // [batch, src_seq, kv_heads, head_dim]
    const float *__restrict__ v,       // [batch, src_seq, kv_heads, head_dim]
    float *__restrict__ o,             // [batch, tgt_seq, q_heads, head_dim]
    const int *__restrict__ group_map, // [query_heads]
    int batch_size, int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int head_dim, bool is_causal) {
  // batch * target_seq_len * query_heads
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * target_seq_len * query_heads;
  if (global_tid >= total)
    return;

  int tmp = global_tid;
  const int qh = tmp % query_heads;
  tmp /= query_heads;
  const int t = tmp % target_seq_len;
  tmp /= target_seq_len;
  const int batch = tmp;

  const int kvh = group_map[qh];

  const size_t q_base = ((size_t)batch * (size_t)target_seq_len + (size_t)t) *
                            (size_t)query_heads * (size_t)head_dim +
                        (size_t)qh * (size_t)head_dim;
  const size_t k_row_base0 = ((size_t)batch * (size_t)src_seq_len) *
                             (size_t)kv_heads *
                             (size_t)head_dim;
  const size_t v_row_base0 = k_row_base0;

  const float scale = rsqrtf((float)head_dim);

  float max_val = -INFINITY;
  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t) {
      continue;
    }
    // dot = q[t,qh] · k[batch,s,kvh]
    float dot = 0.0f;
    const size_t k_base =
        k_row_base0 +
        ((size_t)s * (size_t)kv_heads + (size_t)kvh) * (size_t)head_dim;
    for (int d = 0; d < head_dim; ++d) {
      float qv = q[q_base + d];
      float kv = k[k_base + d];
      dot += qv * kv;
    }
    float scaled = dot * scale;
    if (scaled > max_val)
      max_val = scaled;
  }

  if (!isfinite(max_val)) {
    for (int d = 0; d < head_dim; ++d) {
      const size_t out_idx =
          ((size_t)batch * (size_t)target_seq_len + (size_t)t) *
              (size_t)query_heads * (size_t)head_dim +
          (size_t)qh * (size_t)head_dim + (size_t)d;
      o[out_idx] = 0.0f;
    }
    return;
  }
  double sum = 0.0;

  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t)
      continue;
    float dot = 0.0f;
    const size_t k_base =
        k_row_base0 +
        ((size_t)s * (size_t)kv_heads + (size_t)kvh) * (size_t)head_dim;
    for (int d = 0; d < head_dim; ++d) {
      dot += q[q_base + d] * k[k_base + d];
    }
    float scaled = dot * scale;
    float e = expf(scaled - max_val);
    sum += (double)e;
  }

  if (sum == 0.0) {
    for (int d = 0; d < head_dim; ++d) {
      const size_t out_idx =
          ((size_t)batch * (size_t)target_seq_len + (size_t)t) *
              (size_t)query_heads * (size_t)head_dim +
          (size_t)qh * (size_t)head_dim + (size_t)d;
      o[out_idx] = 0.0f;
    }
    return;
  }

  const float inv_sum = (float)(1.0 / sum);

  for (int d = 0; d < head_dim; ++d) {
    double acc = 0.0;
    for (int s = 0; s < src_seq_len; ++s) {
      if (is_causal && s > t)
        continue;
      float dot = 0.0f;
      const size_t k_base =
          k_row_base0 +
          ((size_t)s * (size_t)kv_heads + (size_t)kvh) * (size_t)head_dim;
      const size_t v_base =
          v_row_base0 +
          ((size_t)s * (size_t)kv_heads + (size_t)kvh) * (size_t)head_dim;
      for (int dd = 0; dd < head_dim; ++dd) {
        dot += q[q_base + dd] * k[k_base + dd];
      }
      float scaled = dot * scale;
      float e = expf(scaled - max_val);
      float soft = e * inv_sum;
      acc += (double)soft * (double)v[v_base + d];
    }
    const size_t out_idx =
        ((size_t)batch * (size_t)target_seq_len + (size_t)t) *
            (size_t)query_heads * (size_t)head_dim +
        (size_t)qh * (size_t)head_dim + (size_t)d;
    o[out_idx] = (float)acc;
  }
}

template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  static_assert(std::is_same<T, float>::value,
                "flashAttention only supports float in this implementation");

  if (h_q.empty() || h_k.empty() || h_v.empty()) {
    throw std::runtime_error("Inputs must be non-empty");
  }

  size_t q_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
  size_t k_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
  size_t v_size = k_size;
  if (h_q.size() != q_size || h_k.size() != k_size || h_v.size() != v_size) {
    throw std::runtime_error("Input sizes do not match provided shapes");
  }

  h_o.assign(q_size, 0.0f);

  std::vector<int> h_group_map = computeQKVGroupMap(query_heads, kv_heads);

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  int *d_group_map = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, k_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, v_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_o, q_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_group_map, h_group_map.size() * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), k_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), v_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_group_map, h_group_map.data(),
                        h_group_map.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  const int total = batch_size * target_seq_len * query_heads;
  const int block = 256;
  const int grid = (total + block - 1) / block;

  flashAttentionKernel_streamed<<<grid, block>>>(
      d_q, d_k, d_v, d_o, d_group_map, batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, is_causal);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, q_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));
  CUDA_CHECK(cudaFree(d_group_map));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
