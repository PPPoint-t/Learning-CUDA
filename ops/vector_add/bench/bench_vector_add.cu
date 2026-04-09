#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark_utils.h"
#include "common/cuda_check.h"
#include "common/data_utils.h"
#include "common/device_info.h"
#include "ops/vector_add/include/vector_add.h"

int main() {
  const int n = 1 << 22;
  const int warmup_iters = 10;
  const int measure_iters = 100;

  std::vector<float> h_a = make_deterministic_floats(n, 11u, 1.0f, 0.0f);
  std::vector<float> h_b = make_deterministic_floats(n, 29u, 1.0f, 0.0f);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  BenchmarkStats stats = benchmark_cuda_launch(warmup_iters, measure_iters, [&]() {
    launch_vector_add(d_a, d_b, d_out, n);
  });

  double bytes_moved = 3.0 * static_cast<double>(n) * sizeof(float);
  double bandwidth_gb_s = effective_bandwidth_gb_s(bytes_moved, stats.avg_ms);

  print_benchmark_header("vector_add", vector_add_variant_name());
  print_current_device_summary();
  std::cout << "  n: " << n << std::endl;
  print_benchmark_stats(stats);
  std::cout << "  effective_bandwidth_gb_s: " << bandwidth_gb_s << std::endl;

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
