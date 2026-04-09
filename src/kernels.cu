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
// ---------- kthLargest (optimized) ----------
template <typename T> T kthLargest(const std::vector<T> &h_input, size_t k) {
  if (h_input.empty()) return T(-100);
  size_t n = h_input.size();
  if (k < 1 || k > n) return T(-100);

  // target rank in ascending order (0-based)
  size_t target_desc_0based = k - 1;
  size_t rank_asc = n - 1 - target_desc_0based;

  // allocate two device buffers once and reuse; we'll swap pointers each iteration
  T *d_buf0 = nullptr, *d_buf1 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf0, n * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_buf1, n * sizeof(T)));
  // initial copy into d_buf0
  CUDA_CHECK(cudaMemcpy(d_buf0, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice));

  // device counters reused across iterations
  int *d_less = nullptr, *d_equal = nullptr;
  int *d_lcur = nullptr, *d_ecur = nullptr, *d_gcur = nullptr;
  CUDA_CHECK(cudaMalloc(&d_less, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_equal, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_lcur, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ecur, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_gcur, sizeof(int)));

  // current source pointer (points to the logical array we're partitioning)
  T *d_src = d_buf0;
  T *d_dst = d_buf1;

  int left = 0;
  int right = (int)n; // exclusive
  const int THREADS = 256;
  T result = T(-100);

  while (left < right) {
    int len = right - left;
    if (len <= 0) break;

    // small-case fallback: bring that tiny block to host and sort
    if (len <= 32) {
      std::vector<T> tmp(len);
      CUDA_CHECK(cudaMemcpy(tmp.data(), d_src + left, len * sizeof(T), cudaMemcpyDeviceToHost));
      std::sort(tmp.begin(), tmp.end());
      result = tmp[rank_asc - left]; // rank_asc is global; subtract left to index into tmp
      break;
    }

    // choose pivot (device value -> copy to host)
    int pivot_idx = left + (len / 2);
    T pivot_val;
    CUDA_CHECK(cudaMemcpy(&pivot_val, d_src + pivot_idx, sizeof(T), cudaMemcpyDeviceToHost));

    // zero counters for counting kernel
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_less, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_equal, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int blocks = (len + THREADS - 1) / THREADS;
    count_less_equal_kernel<T><<<blocks, THREADS>>>(d_src, left, len, pivot_val, d_less, d_equal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int lessCount=0, equalCount=0;
    CUDA_CHECK(cudaMemcpy(&lessCount, d_less, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&equalCount, d_equal, sizeof(int), cudaMemcpyDeviceToHost));

    int pivotPos = left + lessCount;
    // if target in equal bucket -> pivot is answer
    if ((int)rank_asc >= pivotPos && (int)rank_asc < pivotPos + equalCount) {
      result = pivot_val;
      break;
    }

    // zero write cursors in device (for partition kernel)
    CUDA_CHECK(cudaMemcpy(d_lcur, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ecur, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gcur, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // partition current segment [left, left+len) from d_src into d_dst[ left .. left+len-1 ],
    // BUT we will write into d_dst starting at offset left (so global positions match)
    int pblocks = (len + THREADS - 1) / THREADS;
    partition_kernel<T><<<pblocks, THREADS>>>(d_src, left, len, pivot_val,
                                              lessCount, equalCount,
                                              d_dst + left, d_lcur, d_ecur, d_gcur);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // swap buffers (now d_dst contains the partitioned array for this iteration)
    std::swap(d_src, d_dst);

    // update search interval. Note pivotPos is computed from counts relative to previous left.
    if ((int)rank_asc < pivotPos) {
      right = pivotPos;
    } else {
      left = pivotPos + equalCount;
    }
    // loop continues, using same preallocated buffers (no malloc/free)
  }

  if (result == T(-100) && left < (int)n) {
    CUDA_CHECK(cudaMemcpy(&result, d_src + left, sizeof(T), cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaFree(d_buf0));
  CUDA_CHECK(cudaFree(d_buf1));
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

// ---------- flashAttention (optimized with scores buffer) ----------
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
// kernel to compute scores = q·k * scale for each (row, s)
// row index = (batch * target_seq_len + t) * query_heads + qh  => total_rows = batch*target_seq_len*query_heads
// we store scores in row-major: d_scores[row * src_seq_len + s]
// supports causal by writing very negative value for masked positions
__global__ void compute_scores_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ d_scores, // [total_rows * src_seq_len]
    const int* __restrict__ group_map,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {

  int global_row = blockIdx.x * blockDim.x + threadIdx.x;
  int total_rows = batch_size * target_seq_len * query_heads;
  if (global_row >= total_rows) return;

  // decode (batch, t, qh)
  int tmp = global_row;
  int qh = tmp % query_heads;
  tmp /= query_heads;
  int t = tmp % target_seq_len;
  tmp /= target_seq_len;
  int batch = tmp;

  int kvh = group_map[qh];
  float scale = rsqrtf((float)head_dim);

  // q pointer base
  size_t q_base = ((size_t)batch * (size_t)target_seq_len + (size_t)t) * (size_t)query_heads * (size_t)head_dim
                + (size_t)qh * (size_t)head_dim;

  size_t k_row_base0 = ((size_t)batch * (size_t)src_seq_len) * (size_t)kv_heads * (size_t)head_dim;

  float *row_scores = d_scores + (size_t)global_row * (size_t)src_seq_len;
  // compute dot for each s
  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t) {
      row_scores[s] = -1e20f; // masked
      continue;
    }
    size_t k_base = k_row_base0 + ((size_t)s * (size_t)kv_heads + (size_t)kvh) * (size_t)head_dim;
    float dot = 0.0f;
    // small unrolling for head_dim typical sizes can help; keep simple:
    for (int d = 0; d < head_dim; ++d) {
      dot += q[q_base + d] * k[k_base + d];
    }
    row_scores[s] = dot * scale;
  }
}

// kernel: row-wise softmax (numerically stable) for each row of d_scores
// Each row is length src_seq_len; we use a simple per-row sequential reduction for clarity.
// If src_seq_len is large and perf critical, replace with parallel reduction per row.
__global__ void softmax_rows_kernel_simple(float* d_scores, int total_rows, int src_seq_len) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= total_rows) return;
  float* row_ptr = d_scores + (size_t)row * (size_t)src_seq_len;
  // find max
  float mx = -INFINITY;
  for (int i = 0; i < src_seq_len; ++i) {
    float v = row_ptr[i];
    if (v > mx) mx = v;
  }
  // compute exp and sum
  double sum = 0.0;
  for (int i = 0; i < src_seq_len; ++i) {
    float e = expf(row_ptr[i] - mx);
    row_ptr[i] = e;
    sum += (double)e;
  }
  if (sum == 0.0) {
    for (int i = 0; i < src_seq_len; ++i) row_ptr[i] = 0.0f;
    return;
  }
  float inv = 1.0f / (float)sum;
  for (int i = 0; i < src_seq_len; ++i) row_ptr[i] *= inv;
}

// kernel: compute output O given d_scores (softmaxed) and V
// For each row (= (batch,t,qh)) and each head-dim element d, compute sum_s scores[row,s] * V[batch,s,kvh,d]
__global__ void compute_output_kernel(
    const float* __restrict__ d_scores, // [total_rows * src_seq_len]
    const float* __restrict__ v,
    float* __restrict__ o,
    const int* __restrict__ group_map,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim) {

  int global_row = blockIdx.x * blockDim.x + threadIdx.x;
  int total_rows = batch_size * target_seq_len * query_heads;
  if (global_row >= total_rows) return;

  // decode
  int tmp = global_row;
  int qh = tmp % query_heads;
  tmp /= query_heads;
  int t = tmp % target_seq_len;
  tmp /= target_seq_len;
  int batch = tmp;

  int kvh = group_map[qh];

  size_t v_row_base0 = ((size_t)batch * (size_t)src_seq_len) * (size_t)kv_heads * (size_t)head_dim;
  size_t out_base = ((size_t)batch * (size_t)target_seq_len + (size_t)t) * (size_t)query_heads * (size_t)head_dim
                  + (size_t)qh * (size_t)head_dim;

  const float* row_scores = d_scores + (size_t)global_row * (size_t)src_seq_len;

  // compute for each head-dim element
  for (int d = 0; d < head_dim; ++d) {
    double acc = 0.0;
    for (int s = 0; s < src_seq_len; ++s) {
      float sc = row_scores[s];
      // find v element
      size_t v_base = v_row_base0 + ((size_t)s * (size_t)kv_heads + (size_t)kvh) * (size_t)head_dim;
      acc += (double)sc * (double)v[v_base + d];
    }
    o[out_base + d] = (float)acc;
  }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  static_assert(std::is_same<T, float>::value, "Only float supported");

  // shape checks
  size_t q_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
  size_t k_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
  if (h_q.size() != q_size || h_k.size() != k_size || h_v.size() != k_size)
    throw std::runtime_error("Input sizes mismatch");

  // prepare output
  h_o.assign(q_size, 0.0f);

  // group map (host)
  std::vector<int> h_group_map = computeQKVGroupMap(query_heads, kv_heads);

  // device allocations
  float *d_q=nullptr, *d_k=nullptr, *d_v=nullptr, *d_o=nullptr;
  int *d_group_map=nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, k_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, k_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_o, q_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_group_map, h_group_map.size() * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), k_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), k_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_group_map, h_group_map.data(), h_group_map.size() * sizeof(int), cudaMemcpyHostToDevice));

  // allocate scores buffer: total_rows * src_seq_len
  int total_rows = batch_size * target_seq_len * query_heads;
  // check for memory pressure - if too large, you should tile over rows or src_seq
  size_t scores_elems = (size_t)total_rows * (size_t)src_seq_len;
  // naive safety check (you may adjust threshold based on GPU memory)
  // If huge, user should use tiled version; here we just attempt allocation and error out if not enough memory.
  float *d_scores = nullptr;
  CUDA_CHECK(cudaMalloc(&d_scores, scores_elems * sizeof(float)));

  // 1) compute scores
  const int block_rows = 512;
  const int grid_rows = (total_rows + block_rows - 1) / block_rows;
  compute_scores_kernel<<<grid_rows, block_rows>>>(
      d_q, d_k, d_scores, d_group_map,
      batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 2) softmax per row
  const int sblock = 512;
  const int sgrid = (total_rows + sblock - 1) / sblock;
  softmax_rows_kernel_simple<<<sgrid, sblock>>>(d_scores, total_rows, src_seq_len);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 3) compute output using d_scores and v
  const int oblock = 512;
  const int ogrid = (total_rows + oblock - 1) / oblock;
  compute_output_kernel<<<ogrid, oblock>>>(d_scores, d_v, d_o, d_group_map,
                                           batch_size, target_seq_len, src_seq_len,
                                           query_heads, kv_heads, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // copy back
  CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, q_size * sizeof(float), cudaMemcpyDeviceToHost));

  // free
  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));
  CUDA_CHECK(cudaFree(d_group_map));
  CUDA_CHECK(cudaFree(d_scores));
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
