#include <iostream>
#include <vector>

#include "common/benchmark_utils.h"
#include "common/cuda_check.h"
#include "common/data_utils.h"
#include "ops/rmsnorm/include/rmsnorm.h"

int main() {
  const size_t rows = 1ULL << 15;
  const size_t cols = 1ULL << 10;
  const float eps = 1e-5f;
  const int warmup_iters = 10;
  const int measure_iters = 100;

  std::vector<float> h_input = make_deterministic_floats(rows * cols, 11u, 1.0f, 0.0f);
  std::vector<float> h_weight = make_deterministic_floats(cols, 17u, 0.25f, 2.0f);

  float* d_input = nullptr;
  float* d_output = nullptr;
  float* d_weight = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, rows * cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weight, cols * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

  BenchmarkStats stats = benchmark_cuda_launch(warmup_iters, measure_iters, [=]() {
    launch_rmsnorm(d_input, d_output, d_weight, rows, cols, eps);
  });

  print_benchmark_header("rmsnorm", rmsnorm_variant_name());
  print_current_device_summary();
  std::cout << "  rows: " << rows << std::endl;
  std::cout << "  cols: " << cols << std::endl;
  std::cout << "  eps: " << eps << std::endl;
  print_benchmark_stats(stats);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_weight));
  return 0;
}
