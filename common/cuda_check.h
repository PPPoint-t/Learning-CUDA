#pragma once

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " - " << cudaGetErrorString(err__) << std::endl;            \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

