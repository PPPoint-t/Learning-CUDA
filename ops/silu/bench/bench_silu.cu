#include <vector>

#include "common/benchmark_utils.h"
#include "common/cuda_check.h"
#include "common/data_utils.h"
#include "ops/silu/include/silu.h"

int main() {
  const size_t n = 1ULL << 27;
  const int warmup_iters = 10;
  const int measure_iters = 100;

  std::vector<float> h_input = make_deterministic_floats(n, 11u, 1.0f, 0.0f);

  float* d_input = nullptr;
  float* d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  BenchmarkStats stats = benchmark_cuda_launch(warmup_iters, measure_iters,
                                               [&]() { launch_silu(d_input, d_output, n); });

  print_benchmark_report("silu", silu_variant_name(), stats, 2.0, n);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return 0;
}
