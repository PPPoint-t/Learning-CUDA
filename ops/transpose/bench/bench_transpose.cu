#include <vector>

#include "common/benchmark_utils.h"
#include "common/cuda_check.h"
#include "common/data_utils.h"
#include "ops/transpose/include/transpose.h"

int main() {
  const size_t rows = 1ULL << 14;
  const size_t cols = 1ULL << 14;
  const int warmup_iters = 10;
  const int measure_iters = 100;

  std::vector<float> h_input = make_deterministic_floats(rows * cols, 11u, 1.0f, 0.0f);

  float* d_input = nullptr;
  float* d_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, rows * cols * sizeof(float)));
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));

  BenchmarkStats stats = benchmark_cuda_launch(
      warmup_iters, measure_iters,
      [d_input, d_out, rows, cols]() { launch_transpose(d_input, d_out, rows, cols); });

  print_benchmark_stats(stats);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}