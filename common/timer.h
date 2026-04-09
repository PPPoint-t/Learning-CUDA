#pragma once

#include "common/cuda_check.h"

struct CudaEventTimer {
  cudaEvent_t start_event {};
  cudaEvent_t stop_event {};

  CudaEventTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
  }

  ~CudaEventTimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }

  void start(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(start_event, stream));
  }

  float stop(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(stop_event, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    return elapsed_ms;
  }
};
