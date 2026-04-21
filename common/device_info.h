#pragma once

#include <iostream>

#include "common/cuda_check.h"

inline void print_current_device_summary() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  std::cout << "  device: " << prop.name << std::endl;
  std::cout << "  sm_count: " << prop.multiProcessorCount << std::endl;
  std::cout << "  max_threads_per_block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "  global_mem_gb: "
            << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << std::endl;
}
