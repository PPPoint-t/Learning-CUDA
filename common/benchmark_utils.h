#pragma once

#include <iomanip>
#include <iostream>
#include <string>

#include "common/cuda_check.h"
#include "common/device_info.h"
#include "common/timer.h"

struct BenchmarkStats {
  int warmup_iters = 0;
  int measure_iters = 0;
  float total_ms = 0.0f;
  float avg_ms = 0.0f;
};

template <typename LaunchFn>
BenchmarkStats benchmark_cuda_launch(int warmup_iters, int measure_iters, LaunchFn launch) {
  for (int i = 0; i < warmup_iters; ++i) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  // 预热之后再正式执行
  CudaEventTimer timer;
  timer.start();
  for (int i = 0; i < measure_iters; ++i) {
    launch();
  }
  float total_ms = timer.stop();

  BenchmarkStats stats;
  stats.warmup_iters = warmup_iters;
  stats.measure_iters = measure_iters;
  stats.total_ms = total_ms;
  stats.avg_ms = total_ms / static_cast<float>(measure_iters);
  return stats;
}

inline double effective_bandwidth_gb_s(double bytes_moved, float avg_ms) {
  if (avg_ms <= 0.0f) {
    return 0.0;
  }
  return bytes_moved / (avg_ms * 1.0e6);
}

inline void print_benchmark_header(const std::string& op_name, const std::string& variant_name) {
  std::cout << "Benchmarking " << op_name << " variant: " << variant_name << std::endl;
}

inline void print_benchmark_stats(const BenchmarkStats& stats) {
  std::cout << "  warmup_iters: " << stats.warmup_iters << std::endl;
  std::cout << "  measure_iters: " << stats.measure_iters << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "  avg_ms: " << stats.avg_ms << std::endl;
}

inline void print_benchmark_report(const std::string& op_name, const std::string& variant_name,
                                   const BenchmarkStats& stats, double params, size_t n) {
  double bytes_moved = params * static_cast<double>(n) * sizeof(float);
  double bandwidth_gb_s = effective_bandwidth_gb_s(bytes_moved, stats.avg_ms);

  print_benchmark_header(op_name, variant_name);
  print_current_device_summary();
  std::cout << "  n: " << n << std::endl;
  print_benchmark_stats(stats);
  std::cout << "  effective_bandwidth_gb_s: " << bandwidth_gb_s << std::endl;
}
